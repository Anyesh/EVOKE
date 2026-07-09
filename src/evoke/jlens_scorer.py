"""Workspace-content relevance scorer distilled from the Jacobian lens.

The j-space project (phases 1-2) showed that per-position workspace
statistics computed through the J-lens predict which KV blocks a later
answer depends on far better than backward-looking attention heuristics
(fact-AUC 0.891 vs SnapKV 0.622 on Qwen2.5-7B-Instruct), and distilled
that statistic into a per-layer ridge probe: prediction = h @ w + b on
the residual stream entering a scored layer. This module applies that
probe to the residuals the EVOKE fork captures per decode batch
(llama_set/get_embeddings_layer_inp), aggregates per block, and exposes
the AttentionScorerProtocol shape so RelevanceScorer can mix it in as
the w_jlens signal.

Unlike AttentionScorer this signal is content-based and forward-looking:
a block's score is fully determined the moment its tokens pass through
prefill, before any decode history exists, so it also covers the
cold-start case where attention signals are still empty. Scores are
stored raw per block and min-max normalized over live blocks at score()
time; the plan's optional staleness decay is deliberately absent because
a uniform multiplicative decay of all stored values is exactly cancelled
by min-max normalization (only per-block differential decay would change
rankings, and nothing motivates one for a static content signal).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from evoke.types import ActiveBlock


class JLensScorer:
    def __init__(
        self,
        engine,
        *,
        probe_path: str,
        layers: list[int] | None = None,
        stat: str = "kurtosis",
        block_agg: str = "mean",
    ):
        if block_agg not in ("mean", "max"):
            raise ValueError(f"unknown block_agg: {block_agg}")
        artifact = np.load(probe_path)
        available = [int(x) for x in artifact["layers"]]
        self._layers = available if layers is None else [int(x) for x in layers]
        for layer in self._layers:
            if layer not in available:
                raise KeyError(f"layer {layer} not in probe artifact (has {available})")
        # Probe layers are HF block indices whose recorder captured each
        # block's OUTPUT; llama.cpp's capture reads the residual entering a
        # layer, so block L's output must be read at capture id L + 1. Keys
        # of _w/_b are capture ids so absorb can index rows directly.
        self._capture_ids = [layer + 1 for layer in self._layers]
        self._w = {
            layer + 1: artifact[f"L{layer}_{stat}_w"].astype(np.float32) for layer in self._layers
        }
        self._b = {layer + 1: float(artifact[f"L{layer}_{stat}_b"]) for layer in self._layers}
        self._block_agg = block_agg
        self._engine = engine
        # mean agg needs the running sum and count; max agg reuses _sum as
        # the running max with _count as a presence marker.
        self._sum: dict[int, float] = {}
        self._count: dict[int, int] = {}
        engine.layer_inp_capture_enable(self._capture_ids)

    def detach(self) -> None:
        self._engine.layer_inp_capture_enable([])

    def absorb_last_decode(self, blocks: Iterable[ActiveBlock]) -> None:
        # Read the residual rows of the batch the engine just decoded and
        # fold probe predictions into the owning blocks. Called by
        # EvokeManager after every process_tokens / generate_next, same
        # cadence as AttentionScorer.absorb_last_decode. Rows map to logical
        # positions [start, start + n): position re-anchoring from earlier
        # evictions is already reflected in both `start` and the blocks'
        # logical spans, so overlap arithmetic stays valid across evictions.
        captured = self._engine.layer_inp_capture_read()
        if captured is None:
            return
        start, rows_by_layer = captured
        preds = None
        for cid in self._capture_ids:
            rows = rows_by_layer.get(cid)
            if rows is None:
                continue
            p = rows.astype(np.float32) @ self._w[cid] + self._b[cid]
            preds = p if preds is None else preds + p
        if preds is None:
            return
        preds = preds / len(self._capture_ids)
        n = preds.shape[0]

        for block in blocks:
            s = max(block.logical_start, start)
            e = min(block.logical_end, start + n)
            if e <= s:
                continue
            vals = preds[s - start : e - start]
            bid = block.block_id
            if self._block_agg == "max":
                top = float(vals.max())
                prev = self._sum.get(bid)
                self._sum[bid] = top if prev is None else max(prev, top)
                self._count[bid] = 1
            else:
                self._sum[bid] = self._sum.get(bid, 0.0) + float(vals.sum())
                self._count[bid] = self._count.get(bid, 0) + vals.shape[0]

    def score(self, block: ActiveBlock) -> float | None:
        count = self._count.get(block.block_id)
        if not count:
            return None
        values = {
            bid: self._sum[bid] / self._count[bid] if self._block_agg == "mean" else self._sum[bid]
            for bid in self._count
        }
        value = values[block.block_id]
        lo = min(values.values())
        hi = max(values.values())
        if hi <= lo:
            # A single scored block (or exact ties) reads as the strongest
            # signal present, mirroring the max-normalized 1.0 that H2O and
            # SnapKV give their top block.
            return 1.0
        return float((value - lo) / (hi - lo))

    def forget(self, block_id: int) -> None:
        self._sum.pop(block_id, None)
        self._count.pop(block_id, None)
