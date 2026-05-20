"""Attention-weight-based relevance scorer for active blocks.

After each decode step the EVOKE fork writes per-head softmax attention
weights for one tracked layer into a host buffer (see paper §3.3 and the
C primitive llama_attn_capture_*). This module reads those weights,
aggregates the attention mass landing in each active block per decode
step, and maintains an exponentially-weighted sliding window of recent
per-block scores. Plugged into RelevanceScorer via the AttentionScorerProtocol
so the multi-signal scorer can read "what did the model actually attend
to over the last N decodes?" as a per-block float in [0, 1].
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from evoke.llama_engine import LlamaCppEngine
from evoke.types import ActiveBlock


class AttentionScorer:
    def __init__(
        self,
        engine: LlamaCppEngine,
        *,
        layer: int = 20,
        n_window: int = 64,
        decay: float = 0.95,
        capture_capacity: int = 2 * 1024 * 1024,
    ):
        # capture_capacity is in f32 elements. 2M = 8 MB host buffer, enough
        # for n_query * n_heads * n_kv at typical Qwen-2.5 7B decode shapes
        # (1 * 28 * up to ~70K).
        self._engine = engine
        self._n_window = n_window
        self._decay = float(decay)
        self._buffer = np.zeros(capture_capacity, dtype=np.float32)
        self._scores: dict[int, deque[float]] = {}
        engine.attn_capture_set_layer(layer)
        engine.attn_capture_set_buffer(self._buffer)

    def detach(self) -> None:
        self._engine.attn_capture_set_layer(-1)
        self._engine.attn_capture_set_buffer(None)

    def absorb_last_decode(self, blocks: Iterable[ActiveBlock]) -> None:
        # Read the latest capture from the engine, aggregate per-block, push
        # the result into each block's sliding window. Called by EvokeManager
        # after every process_tokens / generate_next so the window stays
        # in lockstep with the engine state.
        n_q, n_h, n_kv = self._engine.attn_capture_get_dims()
        if n_q == 0 or n_h == 0 or n_kv == 0:
            return
        written = self._engine.attn_capture_get_written()
        if written == 0:
            return
        arr = self._buffer[:written].reshape(n_h, n_q, n_kv)
        for block in blocks:
            s = block.logical_start
            e = min(block.logical_end, n_kv)
            if s >= n_kv or e <= s:
                self._push(block.block_id, 0.0)
                continue
            # Per (head, query) the softmax over kv sums to 1.0; the slice
            # [s:e].sum() is the fraction of that softmax mass that landed
            # in this block. Average over heads and over the queries in
            # this decode batch gives a per-step block score in [0, 1].
            per_step = float(arr[:, :, s:e].sum(axis=-1).mean())
            self._push(block.block_id, per_step)

    def score(self, block: ActiveBlock) -> float | None:
        window = self._scores.get(block.block_id)
        if not window:
            return None
        total = 0.0
        weight_sum = 0.0
        weight = 1.0
        for v in reversed(window):
            total += v * weight
            weight_sum += weight
            weight *= self._decay
        if weight_sum == 0.0:
            return None
        return float(total / weight_sum)

    def forget(self, block_id: int) -> None:
        # Called by the manager when a block is permanently evicted (no
        # recovery saved). Keeps the score map from growing unbounded over
        # long-running sessions.
        self._scores.pop(block_id, None)

    def _push(self, block_id: int, value: float) -> None:
        win = self._scores.get(block_id)
        if win is None:
            win = deque(maxlen=self._n_window)
            self._scores[block_id] = win
        win.append(value)
