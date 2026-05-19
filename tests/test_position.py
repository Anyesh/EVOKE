from evoke.position import PositionManager
from evoke.types import ActiveBlock


def _make_block(block_id: int, size: int = 128) -> ActiveBlock:
    return ActiveBlock(
        block_id=block_id,
        logical_start=0,
        logical_end=size,
        token_ids=list(range(size)),
    )


class TestPositionManager:
    def test_register_blocks(self):
        pm = PositionManager()
        pm.register_blocks([_make_block(0), _make_block(1), _make_block(2)])

        assert pm.active_token_count == 384
        assert pm.next_logical_pos == 384
        assert len(pm.active_blocks) == 3

    def test_register_assigns_contiguous_positions(self):
        pm = PositionManager()
        pm.register_blocks([_make_block(0), _make_block(1), _make_block(2)])

        blocks = pm.active_blocks
        assert blocks[0].logical_start == 0
        assert blocks[0].logical_end == 128
        assert blocks[1].logical_start == 128
        assert blocks[2].logical_start == 256

    def test_remove_blocks(self):
        pm = PositionManager()
        pm.register_blocks([_make_block(0), _make_block(1), _make_block(2)])

        pm.remove_blocks({1})

        assert pm.active_token_count == 256
        assert [b.block_id for b in pm.active_blocks] == [0, 2]

    def test_remove_multiple_blocks(self):
        pm = PositionManager()
        pm.register_blocks([_make_block(i) for i in range(4)])

        pm.remove_blocks({1, 2})

        assert pm.active_token_count == 256
        assert [b.block_id for b in pm.active_blocks] == [0, 3]

    def test_append_block_orders_by_id(self):
        pm = PositionManager()
        pm.register_blocks([_make_block(0), _make_block(2)])

        pm.append_block(_make_block(1), kv_start_pos=500)

        assert [b.block_id for b in pm.active_blocks] == [0, 1, 2]

    def test_recompact_compacts_positions(self):
        pm = PositionManager()
        b0 = _make_block(0)
        b2 = _make_block(2)
        b0.logical_start, b0.logical_end = 0, 128
        b2.logical_start, b2.logical_end = 500, 628
        pm._active_blocks = [b0, b2]

        pm.recompact()

        blocks = pm.active_blocks
        assert blocks[0].logical_start == 0
        assert blocks[0].logical_end == 128
        assert blocks[1].logical_start == 128
        assert blocks[1].logical_end == 256
