"""Generation must stop at the physical context boundary, never crash.

Live failure (qwen3-8b opencode run): a thinking spiral generated until
prompt + output reached n_ctx, llama_decode failed with code 1 at the wall,
and deterministic client retries hit the same wall while the crashed turn's
half-evicted state broke identity recovery. The session must clamp
generation to remaining capacity and finish with "length".
"""

from __future__ import annotations

from evoke.config import EvokeConfig
from evoke.mock_engine import MockEngine
from evoke.session import Session


def _make_session(n_ctx: int) -> tuple[Session, MockEngine]:
    engine = MockEngine(n_ctx=n_ctx)
    cfg = EvokeConfig(
        max_active_tokens=1_000_000,
        block_size=16,
        sink_count=0,
        recovery_mode="discard",
    )
    return Session(engine, config=cfg), engine


class TestGenerationCapacityClamp:
    def test_generate_stops_at_context_wall(self):
        session, engine = _make_session(n_ctx=64)
        session.sync_prefix(list(range(40)))
        engine.queue_tokens(list(range(1000, 1100)))
        result = session.generate(max_tokens=100)
        assert len(result.output_tokens) == 24
        assert result.finish_reason == "length"
        assert engine.next_write_pos == 64

    def test_stream_generate_stops_at_context_wall(self):
        session, engine = _make_session(n_ctx=64)
        session.sync_prefix(list(range(40)))
        engine.queue_tokens(list(range(1000, 1100)))
        chunks = list(session.stream_generate(max_tokens=100))
        assert chunks[-1].finish_reason == "length"
        assert len(chunks[-1].output_tokens) == 24
        assert engine.next_write_pos == 64

    def test_generate_with_room_unaffected(self):
        session, engine = _make_session(n_ctx=1024)
        session.sync_prefix(list(range(40)))
        engine.queue_tokens(list(range(1000, 1010)))
        result = session.generate(max_tokens=10)
        assert len(result.output_tokens) == 10
