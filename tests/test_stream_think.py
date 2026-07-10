"""Streaming must suppress the reasoning trace even when the chat template
pre-injects the opening <think>.

Live failure (Qwen3.6-35B-A3B via claude-code-router): the model's template
appends a bare "<think>\\n" to the assistant generation prompt, so generation
begins inside the reasoning block and never emits its own opening <think>. The
stream gate keyed on that opener, so it streamed the whole reasoning trace plus
the trailing </think> to the client as visible content.
"""

from __future__ import annotations

import asyncio
import json

from evoke.config import EvokeConfig
from evoke.mock_engine import MockEngine
from evoke.server import _stream_completion
from evoke.session import SessionPool


def _stream_raw(
    gen_text: str,
    *,
    starts_in_think: bool,
    suppress: bool = False,
    keepalive_interval: float | None = None,
) -> list[str]:
    engine = MockEngine(n_ctx=4096)
    cfg = EvokeConfig(
        max_active_tokens=1_000_000,
        block_size=16,
        sink_count=0,
        recovery_mode="discard",
        suppress_thinking_strip=suppress,
    )
    pool = SessionPool(engine, config=cfg)
    prompt_tokens = engine.tokenize("prompt")
    engine.queue_tokens([ord(c) for c in gen_text] + [engine.eos_token])

    async def run() -> list[str]:
        lock = asyncio.Lock()
        kwargs = {}
        if keepalive_interval is not None:
            kwargs["keepalive_interval"] = keepalive_interval
        gen = _stream_completion(
            pool,
            "s1",
            engine,
            lock,
            prompt_tokens,
            ["<|im_end|>"],
            256,
            "cid",
            0,
            "model",
            starts_in_think=starts_in_think,
            **kwargs,
        )
        return [raw async for raw in gen]

    return asyncio.run(run())


def _content(raw_lines: list[str]) -> str:
    parts: list[str] = []
    for raw in raw_lines:
        if not raw.startswith("data: "):
            continue
        body = raw[len("data: ") :].strip()
        if body == "[DONE]":
            continue
        delta = json.loads(body)["choices"][0]["delta"]
        if delta.get("content"):
            parts.append(delta["content"])
    return "".join(parts)


def _stream_content(
    gen_text: str, *, starts_in_think: bool, suppress: bool = False
) -> str:
    return _content(
        _stream_raw(gen_text, starts_in_think=starts_in_think, suppress=suppress)
    )


REASONING = "The user asks my name. I should answer plainly."
ANSWER = "I'm Claude, Anthropic's assistant."


class TestStreamThinkSuppression:
    def test_preinjected_think_is_suppressed(self):
        # Template injected <think>, so generation starts inside the block and
        # carries no opening tag: reasoning + </think> must not reach the client.
        content = _stream_content(
            f"{REASONING}\n</think>\n\n{ANSWER}", starts_in_think=True
        )
        assert content == ANSWER

    def test_self_emitted_think_still_suppressed(self):
        # Model emits its own <think>...</think> (older Qwen3 default); the gate
        # must keep working unchanged.
        content = _stream_content(
            f"<think>{REASONING}</think>{ANSWER}", starts_in_think=False
        )
        assert content == ANSWER

    def test_plain_answer_passes_through(self):
        content = _stream_content(ANSWER, starts_in_think=False)
        assert content == ANSWER

    def test_suppress_mode_streams_reasoning_verbatim(self):
        # Hybrid-memory models set suppress_thinking_strip so the client echoes
        # the full trace back to keep the cached state aligned.
        content = _stream_content(
            f"{REASONING}\n</think>\n\n{ANSWER}",
            starts_in_think=True,
            suppress=True,
        )
        assert REASONING in content
        assert "</think>" in content
        assert ANSWER in content


class TestStreamThinkKeepalive:
    def test_keepalive_flows_while_think_is_suppressed(self):
        # A slow decode inside <think> emits no content deltas, and the
        # queue-quiet keepalive never fires because chunks keep arriving.
        # The gate must emit its own keepalives so clients on a read
        # timeout (the HF demo app among them) do not abort the stream.
        raw = _stream_raw(
            f"{REASONING}\n</think>\n\n{ANSWER}",
            starts_in_think=True,
            keepalive_interval=0.0,
        )
        assert any(line.startswith(": keepalive") for line in raw)
        assert _content(raw) == ANSWER
