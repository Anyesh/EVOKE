"""Verify tools-aware Jinja chat template rendering (#38B).

Loads a model, takes a sample tools+messages payload (the kind opencode /
Claude Code sends), and renders it through (a) the new
apply_chat_template_with_tools (GGUF Jinja via Python jinja2) and (b) the
existing handwritten format_qwen_chat fallback. Compares both, prints
diff statistics, and verifies the Jinja-rendered prompt round-trips
through tokenize -> detokenize cleanly.

Requires LLAMA_CPP_LIB and EVOKE_MODEL_PATH. Run on the GPU eval host.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.llama_engine import LlamaCppEngine
from evoke.templates import format_qwen_chat

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the working directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search files matching a pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "glob": {"type": "string", "description": "File glob"},
                },
                "required": ["pattern"],
            },
        },
    },
]

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Find the reconciliation logic in claims/services.py"},
]


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("FAIL: set EVOKE_MODEL_PATH")
        return 1
    engine = LlamaCppEngine(model, n_ctx=4096, n_gpu_layers=-1, verbose=False)

    tmpl_string = engine.get_chat_template_string()
    if tmpl_string is None:
        print("FAIL: model has no embedded chat template")
        return 1
    print(f"template length: {len(tmpl_string)} chars")
    print(f"template head: {tmpl_string[:200]!r}")

    try:
        jinja_prompt = engine.apply_chat_template_with_tools(
            SAMPLE_MESSAGES, tools=SAMPLE_TOOLS, add_generation_prompt=True
        )
    except RuntimeError as exc:
        print(f"FAIL: jinja render failed: {exc}")
        return 1
    print(f"\n--- jinja-rendered prompt ({len(jinja_prompt)} chars) ---")
    print(jinja_prompt)

    handwritten = format_qwen_chat(
        SAMPLE_MESSAGES, tools=SAMPLE_TOOLS, add_generation_prompt=True
    )
    print(f"\n--- handwritten prompt ({len(handwritten)} chars) ---")
    print(handwritten)

    if jinja_prompt == handwritten:
        print("\nNOTE: jinja and handwritten produced identical output")
    else:
        print(f"\nNOTE: jinja and handwritten DIFFER (this is expected)")
        print(f"  jinja length: {len(jinja_prompt)}")
        print(f"  handwritten length: {len(handwritten)}")

    # Round-trip: tokenize the jinja prompt, then detokenize, then re-tokenize,
    # confirming idempotence at the tokenizer level (the round-trip a stateful
    # server's prefix-match depends on).
    tokens = engine.tokenize(jinja_prompt)
    detoked = engine.detokenize(tokens)
    retokens = engine.tokenize(detoked)
    if tokens == retokens:
        print(f"\nPASS: tokenize round-trip is idempotent ({len(tokens)} tokens)")
    else:
        print(
            f"\nFAIL: tokenize round-trip diverged "
            f"(first {len(tokens)} vs {len(retokens)} tokens)"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
