from evoke.position import PositionManager
from evoke.types import ActiveBlock


def _make_block(block_id: int, start: int, size: int = 128) -> ActiveBlock:
    return ActiveBlock(
        block_id=block_id,
        logical_start=start,
        logical_end=start + size,
        original_start=start,
        original_end=start + size,
        token_ids=list(range(size)),
    )


class TestPositionManager:
    def test_register_blocks(self):
        pm = PositionManager()
        blocks = [_make_block(0, 0), _make_block(1, 128), _make_block(2, 256)]
        pm.register_blocks(blocks)

        assert pm.active_token_count == 384
        assert pm.next_logical_pos == 384
        assert len(pm.active_blocks) == 3

    def test_remove_blocks(self):
        pm = PositionManager()
        blocks = [_make_block(0, 0), _make_block(1, 128), _make_block(2, 256)]
        pm.register_blocks(blocks)

        pm.remove_blocks({1})

        assert pm.active_token_count == 256
        assert len(pm.active_blocks) == 2

        remaining = pm.active_blocks
        assert remaining[0].block_id == 0
        assert remaining[1].block_id == 2

    def test_append_block(self):
        pm = PositionManager()
        blocks = [_make_block(0, 0), _make_block(2, 256)]
        pm.register_blocks(blocks)

        middle = _make_block(1, 128)
        pm.append_block(middle, kv_start_pos=500)

        assert len(pm.active_blocks) == 3
        assert pm.active_blocks[0].block_id == 0
        assert pm.active_blocks[1].block_id == 1
        assert pm.active_blocks[2].block_id == 2

        assert pm.active_blocks[1].logical_start == 500
        assert pm.active_blocks[1].logical_end == 628

    def test_needs_rebuild(self):
        pm = PositionManager()
        block = _make_block(0, 0, size=128)
        block.logical_start = 29000
        block.logical_end = 29128
        pm._active_blocks = [block]

        assert not pm.needs_rebuild(32768)

        block.logical_start = 30000
        block.logical_end = 30128
        assert pm.needs_rebuild(32768)

    def test_rebuild_positions_compacts(self):
        pm = PositionManager()
        b0 = _make_block(0, 0)
        b0.logical_start = 0
        b0.logical_end = 128
        b2 = _make_block(2, 256)
        b2.logical_start = 500
        b2.logical_end = 628
        pm._active_blocks = [b0, b2]

        pm.rebuild_positions()

        assert pm.active_blocks[0].logical_start == 0
        assert pm.active_blocks[0].logical_end == 128
        assert pm.active_blocks[1].logical_start == 128
        assert pm.active_blocks[1].logical_end == 256

    def test_get_block_at_original_pos(self):
        pm = PositionManager()
        blocks = [_make_block(0, 0), _make_block(1, 128)]
        pm.register_blocks(blocks)

        assert pm.get_block_at_original_pos(64).block_id == 0
        assert pm.get_block_at_original_pos(200).block_id == 1
        assert pm.get_block_at_original_pos(300) is None

    def test_remove_multiple_blocks(self):
        pm = PositionManager()
        blocks = [_make_block(i, i * 128) for i in range(4)]
        pm.register_blocks(blocks)

        pm.remove_blocks({1, 2})

        assert pm.active_token_count == 256
        remaining = pm.active_blocks
        assert [b.block_id for b in remaining] == [0, 3]
