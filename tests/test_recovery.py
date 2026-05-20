import pytest

from evoke.mock_engine import MockEngine
from evoke.recovery import (
    BreadcrumbBackend,
    DiscardBackend,
    KVRestoreBackend,
    make_recovery_backend,
)
from evoke.types import ActiveBlock


def _block(block_id: int, key: str, size: int = 64) -> ActiveBlock:
    return ActiveBlock(
        block_id=block_id,
        logical_start=0,
        logical_end=size,
        token_ids=list(range(size)),
        key=key,
    )


class TestDiscardBackend:
    def test_on_evict_keeps_nothing(self):
        backend = DiscardBackend()
        backend.on_evict([_block(0, "doc#0")], step=5)
        assert backend.list_evicted() == []


class TestBreadcrumbBackend:
    def test_records_evicted_blocks(self):
        backend = BreadcrumbBackend()
        backend.on_evict(
            [_block(0, "doc#0", size=64), _block(1, "doc#1", size=32)], step=3
        )

        crumbs = {c.key: c for c in backend.list_evicted()}
        assert set(crumbs) == {"doc#0", "doc#1"}
        assert crumbs["doc#0"].token_count == 64
        assert crumbs["doc#1"].token_count == 32
        assert crumbs["doc#0"].evicted_at_step == 3

    def test_re_evicting_same_key_overwrites(self):
        backend = BreadcrumbBackend()
        backend.on_evict([_block(0, "doc#0")], step=1)
        backend.on_evict([_block(0, "doc#0")], step=9)

        crumbs = backend.list_evicted()
        assert len(crumbs) == 1
        assert crumbs[0].evicted_at_step == 9


class TestMakeRecoveryBackend:
    def test_discard(self):
        assert isinstance(make_recovery_backend("discard"), DiscardBackend)

    def test_breadcrumb(self):
        assert isinstance(make_recovery_backend("breadcrumb"), BreadcrumbBackend)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            make_recovery_backend("nonsense")


class TestKVRestoreBackend:
    def test_requires_engine(self):
        with pytest.raises(ValueError):
            make_recovery_backend("kv_restore")

    def test_make_with_engine(self):
        backend = make_recovery_backend("kv_restore", MockEngine())
        assert isinstance(backend, KVRestoreBackend)

    def test_on_evict_saves_block(self):
        engine = MockEngine()
        engine.process_tokens(list(range(200)))
        backend = KVRestoreBackend(engine)

        block = _block(0, "doc#0", size=64)
        backend.on_evict([block], step=4)

        crumbs = backend.list_evicted()
        assert [c.key for c in crumbs] == ["doc#0"]
        assert crumbs[0].token_count == 64

        saved = backend.take("doc#0")
        assert saved is not None
        assert saved.token_ids == block.token_ids
        assert saved.saved_at_step == 4
        assert backend.take("doc#0") is None


class TestKVRestoreRamBudget:
    def _engine_with_payload_size(self, payload_bytes: int):
        engine = MockEngine()
        engine.process_tokens(list(range(1000)))
        # Patch kv_block_save to return a known-sized blob so we can predict
        # when the LRU should fire.
        engine.kv_block_save = lambda p0, p1: b"\x00" * payload_bytes
        return engine

    def test_unbounded_when_budget_none(self):
        engine = self._engine_with_payload_size(1000)
        backend = KVRestoreBackend(engine, ram_budget_bytes=None)
        for i in range(10):
            backend.on_evict([_block(i, f"k#{i}")], step=i)
        # All 10 saves survive without budget.
        assert backend.total_bytes == 10_000
        assert backend.lru_evictions == 0
        for i in range(10):
            assert backend.take(f"k#{i}") is not None

    def test_lru_evicts_oldest_under_pressure(self):
        engine = self._engine_with_payload_size(1000)
        backend = KVRestoreBackend(engine, ram_budget_bytes=3000)
        for i in range(5):
            backend.on_evict([_block(i, f"k#{i}")], step=i)
        # 5 blocks * 1000 bytes = 5000 > 3000 budget. Oldest 2 should be
        # LRU-demoted. The newest 3 keep K/V; the 2 oldest survive only as
        # breadcrumbs.
        assert backend.total_bytes <= 3000
        assert backend.lru_evictions >= 2
        assert backend.take("k#0") is None
        assert backend.take("k#1") is None
        assert backend.take("k#4") is not None

    def test_lru_demoted_block_keeps_breadcrumb_and_embedding(self):
        import numpy as np

        engine = self._engine_with_payload_size(1000)
        backend = KVRestoreBackend(engine, ram_budget_bytes=1500)
        block_old = _block(0, "k#0")
        block_old.representative_embedding = np.array([1.0, 2.0, 3.0])
        backend.on_evict([block_old], step=0)
        backend.on_evict([_block(1, "k#1")], step=1)
        # k#0 was demoted (1000 + 1000 > 1500). Its K/V is gone but the
        # breadcrumb and embedding remain — the smart-recovery scorer can
        # still see this block existed.
        assert backend.take("k#0") is None
        emb = backend.peek_embedding("k#0")
        assert emb is not None
        assert list(emb) == [1.0, 2.0, 3.0]
        crumbs = {c.key for c in backend.list_evicted()}
        assert "k#0" in crumbs
        assert "k#1" in crumbs

    def test_take_reduces_total_bytes(self):
        engine = self._engine_with_payload_size(500)
        backend = KVRestoreBackend(engine, ram_budget_bytes=10000)
        backend.on_evict([_block(0, "k#0")], step=0)
        backend.on_evict([_block(1, "k#1")], step=1)
        assert backend.total_bytes == 1000
        backend.take("k#0")
        assert backend.total_bytes == 500

    def test_re_evict_same_key_does_not_double_count(self):
        engine = self._engine_with_payload_size(500)
        backend = KVRestoreBackend(engine, ram_budget_bytes=10000)
        block = _block(0, "k#0")
        backend.on_evict([block], step=0)
        backend.on_evict([block], step=5)
        assert backend.total_bytes == 500  # not 1000

    def test_make_recovery_backend_threads_budget(self):
        backend = make_recovery_backend(
            "kv_restore",
            MockEngine(),
            kv_restore_ram_budget_bytes=999,
        )
        assert isinstance(backend, KVRestoreBackend)
        assert backend._budget == 999
