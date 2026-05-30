"""Faithful ArkVale recall-and-evict policy on the EVOKE/llama.cpp substrate.

ArkVale (NeurIPS'24) scores every page (resident or evicted) by an importance estimate
computed from the page's key bounding-volume digest, keeps the top-k pages resident, recalls
evicted top-k from CPU, and evicts bottom-ranked resident pages. This module implements that
policy over EVOKE blocks: a per-block key digest (per-dim min/max AND mean), the cuboid-mean
importance estimate, and a bounded recall-and-evict that maintains a fixed resident block
budget at the blocks' ORIGINAL positions.

Estimator: ArkVale's paper reports the plain bounding-box ("cuboid") and the mean-only
("centroid") estimators are weak, while "cuboid-mean" reaches ~95% top-1 recall. We use a
cuboid-mean estimate in that spirit: anchor at the true mean key and add the per-dim box
deviation, importance = q.mean + |q| . dev where dev_d = max(max_d - mean_d, mean_d - min_d).
The exact ArkVale CUDA kernel may differ in detail; this is the strong-estimator form so the
comparison does not strawman ArkVale.

The pure functions (block_cuboid, cuboid_score) are engine-independent and unit-tested;
ArkValePolicy.recall_and_evict drives an EvokeManager.
"""

from __future__ import annotations

import numpy as np


def block_cuboid(k_block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # k_block: (n_tokens, n_heads, head_dim) keys for one block at the scoring layer.
    # Returns the per-head per-dim digest (kmin, kmax, kmean), each (n_heads, head_dim).
    return k_block.min(axis=0), k_block.max(axis=0), k_block.mean(axis=0)


def cuboid_score(
    q: np.ndarray, kmin: np.ndarray, kmax: np.ndarray, kmean: np.ndarray
) -> float:
    # ArkVale-style cuboid-mean importance: anchor at the mean key and add the box deviation,
    # q.mean + |q| . dev with dev = max(kmax-kmean, kmean-kmin) per dim. Summed over query
    # heads. q: (n_q_heads, head_dim); digest arrays: (n_kv_heads, head_dim). Under GQA each
    # kv head serves n_q_heads/n_kv_heads query heads, so the digest is broadcast.
    n_q, n_kv = q.shape[0], kmin.shape[0]
    if n_kv > 0 and n_q != n_kv:
        group = n_q // n_kv
        kmin = np.repeat(kmin, group, axis=0)
        kmax = np.repeat(kmax, group, axis=0)
        kmean = np.repeat(kmean, group, axis=0)
    dev = np.maximum(kmax - kmean, kmean - kmin)
    return float((q * kmean).sum() + (np.abs(q) * dev).sum())


class ArkValePolicy:
    # Maintains the per-block key digests (keyed by the block's persistent identity key, which
    # survives evict+recover) and applies ArkVale's bounded recall-and-evict.
    def __init__(self, budget_blocks: int):
        self.budget_blocks = budget_blocks
        self.digests: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def set_cuboid(
        self, key: str, kmin: np.ndarray, kmax: np.ndarray, kmean: np.ndarray
    ) -> None:
        self.digests[key] = (kmin, kmax, kmean)

    def build_block_cuboids(self, mgr, k_capture: np.ndarray) -> None:
        # k_capture: (n_tokens, n_heads, head_dim) for the tokens just prefilled by the most
        # recent add_context. Slice by the newly-added blocks (logical_start order) and store
        # each block's digest. Matched from the end of the capture window by token count.
        new_blocks = sorted(
            (b for b in mgr._positions.active_blocks if b.key not in self.digests),
            key=lambda b: b.logical_start,
        )
        total = sum(len(b.token_ids) for b in new_blocks)
        if total == 0 or k_capture.shape[0] < total:
            return
        off = k_capture.shape[0] - total
        for b in new_blocks:
            n = len(b.token_ids)
            self.digests[b.key] = block_cuboid(k_capture[off : off + n])
            off += n

    def recall_and_evict(self, mgr, q: np.ndarray) -> tuple[int, int]:
        # Rank every block (resident + evicted-with-saved-KV) by cuboid-mean importance to q,
        # keep the top budget_blocks resident: recall evicted top blocks at their original
        # position, evict resident blocks below the cutoff. Pinned/sink blocks are never
        # evicted. Returns (n_recalled, n_evicted).
        scored: list[
            tuple[float, str, bool, int, bool]
        ] = []  # imp,key,resident,block_id,protected
        for b in mgr._positions.active_blocks:
            protected = b.is_sink or b.pinned
            dg = self.digests.get(b.key)
            imp = float("inf") if (protected or dg is None) else cuboid_score(q, *dg)
            scored.append((imp, b.key, True, b.block_id, protected))
        for crumb in mgr.get_breadcrumbs():
            dg = self.digests.get(crumb.key)
            if dg is None:
                continue
            scored.append((cuboid_score(q, *dg), crumb.key, False, -1, False))

        scored.sort(key=lambda s: s[0], reverse=True)
        keep_keys = {s[1] for s in scored[: self.budget_blocks]}

        recalled = 0
        for _imp, key, resident, _bid, _prot in scored[: self.budget_blocks]:
            if not resident and mgr.recover(key, defer_budget=True):
                recalled += 1
        to_evict = [
            s[3]
            for s in scored
            if s[2] and not s[4] and s[1] not in keep_keys and s[3] >= 0
        ]
        if to_evict:
            mgr.force_evict(to_evict)
        return recalled, len(to_evict)
