"""Stop strings must match even when assembled across token boundaries.

Live failure (llama-3.1-8b via opencode): the model has no single
<|im_end|> token, so the stop string arrives as several text tokens. The
streaming scan searched from emitted_len, which advances past each token,
so the window never contained the full stop string and generation ran to
the context wall, streaming a hallucinated multi-turn conversation.
"""

from __future__ import annotations

from evoke.config import EvokeConfig
from evoke.mock_engine import MockEngine
from evoke.session import Session


def _make_session() -> tuple[Session, MockEngine]:
    engine = MockEngine(n_ctx=4096)
    cfg = EvokeConfig(
        max_active_tokens=1_000_000,
        block_size=16,
        sink_count=0,
        recovery_mode="discard",
    )
    return Session(engine, config=cfg), engine


class TestStreamStopStrings:
    def test_multi_token_stop_string_stops_stream(self):
        session, engine = _make_session()
        session.sync_prefix(engine.tokenize("prompt"))
        # One char per token: the stop string "XY" spans two tokens, so it
        # can never appear inside a single delta.
        engine.queue_tokens([ord(c) for c in "ABXYCD"])
        chunks = list(session.stream_generate(max_tokens=6, stop_strings=["XY"]))
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].full_text == "AB"

    def test_stop_string_within_single_delta_still_stops(self):
        session, engine = _make_session()
        session.sync_prefix(engine.tokenize("prompt"))
        engine.queue_tokens([ord(c) for c in "AZCD"])
        chunks = list(session.stream_generate(max_tokens=4, stop_strings=["Z"]))
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].full_text == "A"


class TestEosTextExcluded:
    # The engine detokenizes special tokens to their literal text, so the
    # eos token's rendering (<|eot_id|> on Llama, <|im_end|> on Qwen) must
    # not leak into the result text. It stays in output_tokens because the
    # cached stream must match the client's next-turn template echo, which
    # includes the end marker.

    def test_generate_excludes_eos_text(self):
        session, engine = _make_session()
        session.sync_prefix(engine.tokenize("prompt"))
        engine.queue_tokens([ord("A"), ord("B"), engine.eos_token])
        result = session.generate(max_tokens=8)
        assert result.finish_reason == "stop"
        assert result.text == "AB"
        assert result.output_tokens[-1] == engine.eos_token

    def test_stream_generate_excludes_eos_text(self):
        session, engine = _make_session()
        session.sync_prefix(engine.tokenize("prompt"))
        engine.queue_tokens([ord("A"), ord("B"), engine.eos_token])
        chunks = list(session.stream_generate(max_tokens=8))
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].full_text == "AB"
