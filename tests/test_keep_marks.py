from __future__ import annotations

from evoke.config import EvokeConfig
from evoke.mock_engine import MockEngine
from evoke.session import Session
from evoke.types import KeepSpan


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


def _resident_starts(session: Session) -> list[int]:
    return sorted(b.logical_start for b in session._manager._positions.active_blocks)


def _block_at(session: Session, start: int):
    return next(
        b for b in session._manager._positions.active_blocks if b.logical_start == start
    )


PIN_8_16 = [KeepSpan(8, 16, "pin")]


def test_pinned_span_survives_end_of_turn_eviction():
    prompt = list(range(40))
    control = Session(RecordingEngine(), config=_config(16))
    control.sync_prefix(prompt)
    assert 8 not in _resident_starts(control) or 12 not in _resident_starts(control)

    marked = Session(RecordingEngine(), config=_config(16))
    marked.sync_prefix(prompt, keep_spans=PIN_8_16)
    starts = _resident_starts(marked)
    assert 8 in starts and 12 in starts
    assert _block_at(marked, 8).keep_pinned
    assert marked._manager.pinned_token_count == 8


def test_dropping_the_mark_releases_the_pin():
    session = Session(RecordingEngine(), config=_config(1_000_000))
    prompt = list(range(20))
    session.sync_prefix(prompt, keep_spans=PIN_8_16)
    assert _block_at(session, 8).keep_pinned
    session.sync_prefix(prompt)
    assert not any(b.keep_pinned for b in session._manager._positions.active_blocks)
    assert session._manager.pinned_token_count == 0


def test_evicted_pinned_block_is_restored_without_headroom():
    engine = RecordingEngine()
    session = Session(engine, config=_config(1_000_000))
    prompt = list(range(20))
    session.sync_prefix(prompt)
    session._manager.force_evict([_block_at(session, 8).block_id])
    session._manager.set_budget(8)
    stats = session.sync_prefix(prompt, keep_spans=[KeepSpan(8, 12, "pin")])
    assert stats.blocks_recovered == 1
    assert stats.new_tokens_decoded == 0
    assert engine._token_at_pos[8] == 8
    assert _block_at(session, 8).keep_pinned


def test_unpinned_evicted_block_stays_a_hole_without_headroom():
    engine = RecordingEngine()
    session = Session(engine, config=_config(1_000_000))
    prompt = list(range(20))
    session.sync_prefix(prompt)
    session._manager.force_evict([_block_at(session, 8).block_id])
    session._manager.set_budget(8)
    stats = session.sync_prefix(prompt)
    assert stats.blocks_recovered == 0
    assert 8 not in engine._token_at_pos


def test_prefer_boosts_score_but_does_not_pin():
    session = Session(RecordingEngine(), config=_config(1_000_000))
    prompt = list(range(20))
    session.sync_prefix(prompt, keep_spans=[KeepSpan(4, 8, "prefer")])
    block = _block_at(session, 4)
    assert block.keep_boost == session._config.keep_prefer_boost > 1.0
    assert not block.keep_pinned
    assert _block_at(session, 0).keep_boost == 1.0


def test_new_tail_splits_at_mark_boundaries():
    engine = RecordingEngine()
    session = Session(engine, config=_config(1_000_000))
    session.sync_prefix(list(range(40)), keep_spans=[KeepSpan(21, 27, "pin")])
    for block in session._manager._positions.active_blocks:
        inside = 21 <= block.logical_start < 27
        assert block.keep_pinned == inside
        assert not (block.logical_start < 21 < block.logical_end)
        assert not (block.logical_start < 27 < block.logical_end)
    assert [len(c) for c in engine.decoded] == [21, 6, 13]


def test_pin_on_resident_block_that_overlaps_partially_pins_it_whole():
    session = Session(RecordingEngine(), config=_config(1_000_000))
    prompt = list(range(20))
    session.sync_prefix(prompt)
    session.sync_prefix(prompt, keep_spans=[KeepSpan(6, 9, "pin")])
    assert _block_at(session, 4).keep_pinned
    assert _block_at(session, 8).keep_pinned
    assert not _block_at(session, 12).keep_pinned
