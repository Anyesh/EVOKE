from __future__ import annotations

from evoke.config import EvokeConfig
from evoke.mock_engine import MockEngine
from evoke.session import Session


class RecordingEngine(MockEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoded: list[list[int]] = []

    def process_tokens(self, tokens: list[int]) -> None:
        self.decoded.append(list(tokens))
        super().process_tokens(tokens)


def _config(budget: int, budget_aware: bool = True) -> EvokeConfig:
    return EvokeConfig(
        max_active_tokens=budget,
        eviction_policy="threshold",
        block_size=4,
        sink_count=0,
        position_mode="sparse",
        recovery_mode="kv_restore",
        recovery_match="identity",
        gap_fill_budget_aware=budget_aware,
    )


def _build(budget_during_build: int = 1_000_000, budget: int = 1_000_000):
    engine = RecordingEngine()
    session = Session(engine, config=_config(budget_during_build))
    prompt = list(range(20))
    session.sync_prefix(prompt)
    return engine, session, prompt


def _evict_starts(session: Session, starts: tuple[int, ...]) -> None:
    mgr = session._manager
    for start in starts:
        blk = next(b for b in mgr._positions.active_blocks if b.logical_start == start)
        mgr.force_evict([blk.block_id])


def _resident_starts(session: Session) -> list[int]:
    return sorted(b.logical_start for b in session._manager._positions.active_blocks)


def test_restores_only_what_fits_under_budget_nearest_tail_first():
    engine, session, prompt = _build()
    _evict_starts(session, (4, 8, 12))
    session._manager.set_budget(12)  # 8 resident + room for one 4-token block
    stats = session.sync_prefix(prompt)
    assert stats.blocks_recovered == 1
    assert stats.new_tokens_decoded == 0
    assert _resident_starts(session) == [0, 12, 16]
    assert session._manager.get_stats().active_tokens == 12
    assert engine._token_at_pos[12] == 12
    assert 4 not in engine._token_at_pos and 8 not in engine._token_at_pos
    assert engine.decoded == [prompt]


def test_no_headroom_restores_nothing_but_keeps_session():
    engine, session, prompt = _build()
    mgr = session._manager
    _evict_starts(session, (4, 8, 12))
    mgr.set_budget(8)
    stats = session.sync_prefix(prompt)
    assert stats.blocks_recovered == 0
    assert stats.new_tokens_decoded == 0
    assert session._manager is mgr
    assert _resident_starts(session) == [0, 16]


def test_budget_unaware_flag_restores_everything():
    engine, session, prompt = _build()
    _evict_starts(session, (4, 8, 12))
    session._manager.set_budget(12)
    session._config.gap_fill_budget_aware = False
    stats = session.sync_prefix(prompt)
    assert stats.blocks_recovered == 3
    assert _resident_starts(session) == [0, 4, 8, 12, 16]


def test_new_tail_is_charged_against_headroom():
    engine, session, prompt = _build()
    _evict_starts(session, (4, 8, 12))
    session._manager.set_budget(16)  # 8 resident + 4 tail leaves room for one block
    resend = prompt + [100, 101, 102, 103]
    stats = session.sync_prefix(resend)
    assert stats.blocks_recovered == 1
    assert stats.new_tokens_decoded == 4
    assert engine.decoded[-1] == [100, 101, 102, 103]
    assert engine._token_at_pos[20] == 100  # tail lands at its prompt index
    assert _resident_starts(session) == [0, 12, 16, 20]


def test_hole_below_changed_content_does_not_force_reset():
    engine, session, prompt = _build()
    mgr = session._manager
    _evict_starts(session, (4, 8))
    mgr.set_budget(12)
    changed = list(range(16)) + [200, 201, 202, 203]
    stats = session.sync_prefix(changed)
    assert session._manager is mgr
    assert engine.decoded[-1] == [200, 201, 202, 203]
    assert stats.new_tokens_decoded == 4
    assert engine._token_at_pos[16] == 200
    assert engine.next_write_pos == 20
    assert 4 not in engine._token_at_pos and 8 not in engine._token_at_pos


def test_trailing_saved_block_is_always_restored():
    engine, session, prompt = _build()
    _evict_starts(session, (16,))
    session._manager.set_budget(8)
    stats = session.sync_prefix(prompt)
    assert stats.blocks_recovered == 1
    assert engine.next_write_pos == 20
    assert engine._token_at_pos[16] == 16
