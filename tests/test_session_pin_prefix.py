from evoke.config import EvokeConfig
from evoke.mock_engine import MockEngine
from evoke.session import Session


def _session(budget: int = 600) -> Session:
    cfg = EvokeConfig(
        max_active_tokens=budget,
        block_size=128,
        high_watermark=0.92,
        low_watermark=0.70,
        recovery_mode="discard",
    )
    return Session(MockEngine(), config=cfg)


def _pinned_spans(session: Session) -> list[tuple[int, int, bool]]:
    return sorted(
        (b.logical_start, b.logical_end, b.pinned)
        for b in session.manager._positions.active_blocks
    )


def test_blocks_inside_pin_prefix_are_pinned_and_split_at_the_boundary():
    session = _session(budget=10_000)
    session.sync_prefix(list(range(500)), pin_prefix=300)
    spans = _pinned_spans(session)
    assert all(pinned for start, end, pinned in spans if end <= 300)
    assert not any(pinned for start, end, pinned in spans if start >= 300)
    assert any(end == 300 for _, end, _ in spans)


def test_pinned_prefix_survives_eviction_pressure_across_turns():
    session = _session(budget=600)
    prompt = list(range(400))
    session.sync_prefix(prompt, pin_prefix=400)
    for turn in range(3):
        prompt = prompt + [1000 + turn * 300 + i for i in range(300)]
        session.sync_prefix(prompt, pin_prefix=400)
    resident = {
        pos
        for b in session.manager._positions.active_blocks
        for pos in range(b.logical_start, b.logical_end)
    }
    assert set(range(400)) <= resident


def test_pin_prefix_already_resident_pins_nothing_new():
    session = _session(budget=10_000)
    session.sync_prefix(list(range(300)), pin_prefix=0)
    session.sync_prefix(list(range(500)), pin_prefix=300)
    assert not any(pinned for _, _, pinned in _pinned_spans(session))
