"""Faithful ArkVale recall-and-evict policy on the EVOKE/llama.cpp substrate.

ArkVale (NeurIPS'24) scores every page (resident or evicted) by an upper bound on q.k
computed from the page's key bounding-volume ("cuboid", per-dim key min/max), keeps the
top-k pages resident, recalls evicted top-k from CPU, and evicts bottom-ranked resident
pages. This module implements that policy over EVOKE blocks: a per-block cuboid digest
built from the captured keys, the cuboid importance estimate, and a bounded recall-and-evict
that maintains a fixed resident block budget at the blocks' ORIGINAL positions.

The pure functions (block_cuboid, cuboid_score) are engine-independent and unit-tested on
synthetic data; ArkValePolicy.recall_and_evict drives an EvokeManager.
"""

from __future__ import annotations

import numpy as np


def block_cuboid(k_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # k_block: (n_tokens, n_heads, head_dim) keys for one block at the scoring layer.
    # Returns the per-head per-dim bounding box (kmin, kmax), each (n_heads, head_dim).
    return k_block.min(axis=0), k_block.max(axis=0)


def cuboid_score(q: np.ndarray, kmin: np.ndarray, kmax: np.ndarray) -> float:
    # ArkVale cuboid importance: an upper bound on the attention logit q.k over the
    # page's key bounding box. For each dim pick the box corner that maximizes q_d * k_d
    # (kmax if q_d>0 else kmin), sum over dims for a per-head logit upper bound, sum over
    # heads. q, kmin, kmax: (n_heads, head_dim). Higher => the page could matter more.
    best_k = np.where(q > 0.0, kmax, kmin)
    per_head = (q * best_k).sum(axis=1)
    return float(per_head.sum())


class ArkValePolicy:
    # Maintains the per-block cuboid digests (keyed by the block's persistent identity
    # key, which survives evict+recover) and applies ArkVale's bounded recall-and-evict.
    def __init__(self, budget_blocks: int):
        self.budget_blocks = budget_blocks
        self.cuboids: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def set_cuboid(self, key: str, kmin: np.ndarray, kmax: np.ndarray) -> None:
        self.cuboids[key] = (kmin, kmax)

    def build_block_cuboids(self, mgr, k_capture: np.ndarray) -> None:
        # k_capture: (n_tokens, n_heads, head_dim) for the tokens just prefilled by the
        # most recent add_context. Slice it by the newly-added blocks (in logical_start
        # order) and store each block's cuboid. Blocks shorter than the capture tail are
        # matched by their token count from the end of the capture window.
        new_blocks = sorted(
            (b for b in mgr._positions.active_blocks if b.key not in self.cuboids),
            key=lambda b: b.logical_start,
        )
        total = sum(len(b.token_ids) for b in new_blocks)
        if total == 0 or k_capture.shape[0] < total:
            return
        off = k_capture.shape[0] - total
        for b in new_blocks:
            n = len(b.token_ids)
            kmin, kmax = block_cuboid(k_capture[off : off + n])
            self.cuboids[b.key] = (kmin, kmax)
            off += n

    def recall_and_evict(self, mgr, q: np.ndarray) -> tuple[int, int]:
        # Rank every block (resident + evicted-with-saved-KV) by cuboid importance to q,
        # keep the top budget_blocks resident: recall evicted top blocks at their original
        # position, evict resident blocks that fall below the cutoff. Pinned/sink blocks are
        # never evicted. Returns (n_recalled, n_evicted).
        scored: list[
            tuple[float, str, bool, int, bool]
        ] = []  # imp,key,resident,block_id,protected
        for b in mgr._positions.active_blocks:
            protected = b.is_sink or b.pinned
            cub = self.cuboids.get(b.key)
            imp = float("inf") if (protected or cub is None) else cuboid_score(q, *cub)
            scored.append((imp, b.key, True, b.block_id, protected))
        for crumb in mgr.get_breadcrumbs():
            cub = self.cuboids.get(crumb.key)
            if cub is None:
                continue
            scored.append((cuboid_score(q, *cub), crumb.key, False, -1, False))

        scored.sort(key=lambda s: s[0], reverse=True)
        keep_keys = {s[1] for s in scored[: self.budget_blocks]}

        recalled = 0
        for imp, key, resident, _bid, _prot in scored[: self.budget_blocks]:
            if not resident:
                if mgr.recover(key, defer_budget=True):
                    recalled += 1
        to_evict = [
            s[3]
            for s in scored
            if s[2] and not s[4] and s[1] not in keep_keys and s[3] >= 0
        ]
        if to_evict:
            mgr.force_evict(to_evict)
        return recalled, len(to_evict)
