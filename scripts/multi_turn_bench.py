from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.chat_template import detect_template, strip_thinking
from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager

MODEL = os.environ.get(
    "EVOKE_MODEL_PATH",
    str(
        Path(__file__).resolve().parents[1]
        / "models"
        / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    ),
)

FILLER = (
    "This is a passage about various topics. The weather today is partly cloudy "
    "with a chance of rain in the afternoon. Markets showed mixed performance "
    "with technology stocks leading gains while energy shares declined. "
    "Scientists have discovered a new species of deep-sea fish in the Pacific "
    "Ocean that displays bioluminescent properties. "
)


def run_multi_turn(engine, template, config):
    stop_token_ids: set[int] = set()
    for s in template.stop_strings:
        toks = engine.tokenize(s)
        if len(toks) == 1:
            stop_token_ids.add(toks[0])

    engine.reset()
    mgr = EvokeManager(engine, config)

    doc = FILLER * 30
    doc_prefix = template.wrap_document_prefix(doc)
    mgr.load_document(doc_prefix)

    turns = [
        {
            "user": "The secret project codename is AURORA-SEVEN.",
            "check": None,
        },
        {
            "user": "What color is the sky on a clear day?",
            "check": None,
        },
        {
            "user": "The budget for AURORA-SEVEN is exactly $4.2 million.",
            "check": None,
        },
        {
            "user": "How deep can the new species of fish be found?",
            "check": None,
        },
        {
            "user": "Remember to finalize the AURORA-SEVEN report by Friday.",
            "check": None,
        },
        {
            "user": "What do you know about AURORA-SEVEN?",
            "check": "AURORA",
        },
    ]

    print(f"Budget: {config.max_active_tokens} tokens")
    print(f"Block size: {config.block_size}")
    print()

    for i, turn in enumerate(turns):
        suffix = template.wrap_question_suffix(turn["user"])
        mgr.process_user_message(suffix)

        if template.think_close:
            raw = mgr.generate(
                0,
                stop_token_ids=stop_token_ids,
                think_close=template.think_close,
                thinking_budget=16384,
                answer_budget=512,
            )
        else:
            raw = mgr.generate(128, stop_token_ids=stop_token_ids)
        answer = strip_thinking(raw)

        stats = mgr.get_stats()
        print(f"Turn {i + 1}: {turn['user'][:60]}")
        print(
            f"  active={stats.active_tokens} archive={stats.archive_blocks} "
            f"promo={stats.total_promotions} demo={stats.total_demotions}"
        )

        if turn["check"]:
            found = turn["check"] in answer
            print(f"  CHECK '{turn['check']}' in answer: {found}")
            print(f"  answer: {answer[:150]!r}")
            if not answer:
                print(f"  raw[0:100]: {raw[:100]!r}")
        else:
            print(f"  answer: {answer[:80]!r}")
        print()


def main():
    model_name = Path(MODEL).stem
    n_ctx = int(os.environ.get("EVOKE_N_CTX", "131072"))
    engine = LlamaCppEngine(MODEL, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
    template = detect_template(model_name)

    print(f"Model: {model_name}")
    print(f"Template: {type(template).__name__}")
    print(f"Thinking: {template.think_close is not None}")
    print()

    configs = {
        "tight (512)": EvokeConfig(
            max_active_tokens=512,
            block_size=32,
            retrieval_threshold=0.85,
            max_retrieve_blocks=4,
            high_watermark=0.95,
            low_watermark=0.75,
        ),
        "medium (1024)": EvokeConfig(
            max_active_tokens=1024,
            block_size=32,
            retrieval_threshold=0.85,
            max_retrieve_blocks=4,
            high_watermark=0.95,
            low_watermark=0.75,
        ),
    }

    for label, config in configs.items():
        print(f"=== {label} ===")
        run_multi_turn(engine, template, config)
        print()

    engine.close()


if __name__ == "__main__":
    main()
