from __future__ import annotations

from evoke.types import ActiveBlock


class PositionManager:
    def __init__(self):
        self._active_blocks: list[ActiveBlock] = []

    @property
    def active_blocks(self) -> list[ActiveBlock]:
        return list(self._active_blocks)

    @property
    def active_token_count(self) -> int:
        return sum(len(b.token_ids) for b in self._active_blocks)

    @property
    def next_logical_pos(self) -> int:
        if not self._active_blocks:
            return 0
        return max(b.logical_end for b in self._active_blocks)

    def register_blocks(self, blocks: list[ActiveBlock]) -> None:
        self._active_blocks = sorted(blocks, key=lambda b: b.original_start)
        self._recompute_contiguous()

    def remove_blocks(self, block_ids: set[int]) -> None:
        self._active_blocks = [
            b for b in self._active_blocks if b.block_id not in block_ids
        ]

    def append_block(self, block: ActiveBlock, kv_start_pos: int) -> None:
        size = len(block.token_ids)
        block.logical_start = kv_start_pos
        block.logical_end = kv_start_pos + size
        self._active_blocks.append(block)
        self._active_blocks.sort(key=lambda b: b.original_start)

    def needs_rebuild(self, n_ctx: int) -> bool:
        if not self._active_blocks:
            return False
        max_pos = max(b.logical_end for b in self._active_blocks)
        return max_pos > int(n_ctx * 0.9)

    def rebuild_positions(self) -> None:
        self._active_blocks.sort(key=lambda b: b.original_start)
        self._recompute_contiguous()

    def _recompute_contiguous(self) -> None:
        pos = 0
        for block in self._active_blocks:
            block_size = len(block.token_ids)
            block.logical_start = pos
            block.logical_end = pos + block_size
            pos += block_size

    def get_block_at_original_pos(self, original_pos: int) -> ActiveBlock | None:
        for block in self._active_blocks:
            if block.original_start <= original_pos < block.original_end:
                return block
        return None
