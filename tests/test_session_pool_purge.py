"""Dropping a session, explicitly or by LRU, must delete its saved KV,
including blocks spilled to disk: that state encodes the tenant's context."""

from evoke.config import EvokeConfig
from evoke.mock_engine import MockEngine
from evoke.session import SessionPool


def _pool(tmp_path, max_sessions: int = 8) -> tuple[SessionPool, MockEngine]:
    engine = MockEngine()
    cfg = EvokeConfig(
        max_active_tokens=512,
        block_size=64,
        sink_count=0,
        high_watermark=0.9,
        low_watermark=0.6,
        recovery_mode="kv_restore",
        kv_restore_ram_budget_bytes=64,
        kv_restore_spill_path=str(tmp_path / "spill"),
    )
    return SessionPool(engine, config=cfg, max_sessions=max_sessions), engine


def _fill(pool: SessionPool, sid: str) -> None:
    session = pool.get(sid)
    prompt = list(range(600))
    session.sync_prefix(prompt)
    session.sync_prefix(prompt + list(range(1000, 1600)))


def test_drop_deletes_the_sessions_spill_files(tmp_path):
    pool, _ = _pool(tmp_path)
    _fill(pool, "a")
    spill = tmp_path / "spill"
    assert any(spill.iterdir())
    assert pool.drop("a")
    assert not any(spill.iterdir())


def test_lru_eviction_deletes_the_victims_spill_files(tmp_path):
    pool, _ = _pool(tmp_path, max_sessions=1)
    _fill(pool, "a")
    spill = tmp_path / "spill"
    assert any(spill.iterdir())
    pool.get("b")
    assert "a" not in pool.session_ids()
    assert not any(spill.iterdir())


def test_session_reset_deletes_the_previous_archive(tmp_path):
    pool, _ = _pool(tmp_path)
    _fill(pool, "a")
    spill = tmp_path / "spill"
    assert any(spill.iterdir())
    pool.get("a").reset()
    assert not any(spill.iterdir())
