import pytest

from evoke.recovery import BreadcrumbBackend, DiscardBackend, make_recovery_backend
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
