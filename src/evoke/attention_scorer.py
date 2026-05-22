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
        layer: int | None = None,
        layers: list[int] | None = None,
        n_window: int = 64,
        decay: float = 0.95,
        score_mode: str = "ewma",
        snapkv_observation_window: int = 32,
        capture_capacity: int = 8 * 1024 * 1024,
    ):
        # capture_capacity is in f32 elements (32 MB at the default). The
        # multi-layer capture writes [n_layers, n_query, n_heads, n_kv] per
        # decode; 8M f32 holds 4 layers * 1 * 28 heads * 70K kv at the
        # typical Qwen-2.5 7B decode shape.
        self._engine = engine
        self._n_window = n_window
        self._decay = float(decay)
        self._score_mode = score_mode
        self._snapkv_window = int(snapkv_observation_window)
        self._buffer = np.zeros(capture_capacity, dtype=np.float32)
        self._scores: dict[int, deque[float]] = {}
        # Lifetime attention mass per block. Tracked unconditionally so callers
        # can switch score_mode at runtime without losing prior history; the
        # ewma path ignores this dict, and the cumulative path ignores the
        # sliding window. Cleared per block by forget().
        self._cumulative: dict[int, float] = {}
        # SnapKV (Liu et al., NeurIPS 2024) observation-window scoring. The
        # paper computes per-prior-token importance as the sum of softmax
        # attention from the last W prompt tokens to each prior token, then
        # picks top-K to keep. At our block granularity we sum that mass per
        # block and normalize against the running max so high-attention
        # blocks score near 1.0. _snapkv_pending accumulates within an
        # observation phase; snapshot() freezes it into _snapkv_frozen, which
        # is what score() returns until the next snapshot. The manager calls
        # snapshot() at the end of process_user_message so eviction during
        # the subsequent generate sees the question-window scores. Cleared
        # per block by forget(); reset on each snapshot so the next user
        # turn starts from a clean pending bucket.
        self._snapkv_pending: dict[int, float] = {}
        self._snapkv_frozen: dict[int, float] | None = None
        if layers is not None:
            engine.attn_capture_set_layers(layers)
        else:
            engine.attn_capture_set_layer(20 if layer is None else layer)
        engine.attn_capture_set_buffer(self._buffer)

    def detach(self) -> None:
        self._engine.attn_capture_set_layer(-1)
        self._engine.attn_capture_set_buffer(None)

    def absorb_last_decode(self, blocks: Iterable[ActiveBlock]) -> None:
        # Read the latest capture from the engine, aggregate per-block, push
        # the result into each block's sliding window. Called by EvokeManager
        # after every process_tokens / generate_next so the window stays
        # in lockstep with the engine state. Multi-layer capture (n_layers
        # > 1) means a single block scores from each layer; we mean across
        # layers before pushing into the window. Deeper layers carry more
        # semantic attention pattern; equal-weight mean is a deliberate
        # neutral default — callers can switch to a weighted scheme later.
        dims = self._engine.attn_capture_get_dims()
        if len(dims) == 4:
            n_layers, n_q, n_h, n_kv = dims
        else:
            n_layers = 1
            n_q, n_h, n_kv = dims
        if n_layers == 0 or n_q == 0 or n_h == 0 or n_kv == 0:
            return
        written = self._engine.attn_capture_get_written()
        if written == 0:
            return
        arr = self._buffer[:written].reshape(n_layers, n_h, n_q, n_kv)

        if self._score_mode == "snapkv":
            # SnapKV scores from the last W query tokens of the current decode
            # batch (the observation window). For a probe of <= W tokens the
            # whole batch is the window; for longer prompts we restrict to the
            # tail so the score reflects the most recent intent, mirroring the
            # paper's "last W tokens of the prompt" formulation.
            n_use = min(self._snapkv_window, n_q)
            if n_use <= 0:
                # snapkv_observation_window=0 is a degenerate config; absorbing
                # would write a zero score for every block (because
                # arr[:, :, -0:, :] selects the full slice in numpy, not an
                # empty one) and corrupt the next snapshot. Bail before that.
                return
            arr_use = arr[:, :, -n_use:, :]
            for block in blocks:
                s = block.logical_start
                e = min(block.logical_end, n_kv)
                if s >= n_kv or e <= s:
                    # Block sits outside the kv range of this decode (it was
                    # added earlier and the buffer-fits cap meant n_kv was
                    # truncated). Leave its pending score untouched — zeroing
                    # would push older relevant blocks below newer noise.
                    continue
                # Sum (not mean) over layers/heads/observation tokens of the
                # mass that landed in this block. SnapKV's heavy-hitter
                # statistic is cumulative attention from the observation
                # window. Normalization against the per-snapshot max happens
                # in _score_snapkv so the policy compares blocks within a
                # single snapshot rather than across snapshots.
                val = float(arr_use[:, :, :, s:e].sum())
                self._snapkv_pending[block.block_id] = (
                    self._snapkv_pending.get(block.block_id, 0.0) + val
                )
            return

        for block in blocks:
            s = block.logical_start
            e = min(block.logical_end, n_kv)
            if s >= n_kv or e <= s:
                self._push(block.block_id, 0.0)
                continue
            # Per (layer, head, query) the softmax over kv sums to 1.0;
            # [s:e].sum() is the fraction that landed in this block. Average
            # over layers, heads, and queries gives a per-step block score
            # in [0, 1].
            per_step = float(arr[:, :, :, s:e].sum(axis=-1).mean())
            self._push(block.block_id, per_step)

    def snapshot(self) -> None:
        # Freeze SnapKV's pending accumulation into the live score table and
        # reset pending so the next observation phase starts clean. Called by
        # EvokeManager.process_user_message after the user-message decode
        # absorbs into pending so the very next _enforce_budget pass uses
        # question-window attention to pick survivors. No-op for ewma /
        # cumulative modes; safe to call unconditionally.
        if self._score_mode != "snapkv":
            return
        self._snapkv_frozen = dict(self._snapkv_pending)
        self._snapkv_pending = {}

    def is_eviction_ready(self) -> bool:
        # SnapKV's "compress once per prompt" semantics: until the manager
        # has called snapshot() at the end of the first process_user_message,
        # the scorer has no observation-window signal and any eviction would
        # fall back to insertion-order (because score() returns None and the
        # manager's 0.0 default ties every block). The SnapKV paper assumes
        # the full prompt is loaded then compressed once — our pipeline
        # loads context incrementally, so add_context's per-chunk
        # _enforce_budget would evict the needle before SnapKV ever sees the
        # question. Returning False here makes EvokeManager._enforce_budget
        # defer eviction until snapshot has fired, after which the frozen
        # scores drive a single bulk compression to the watermark. Other
        # modes (ewma, cumulative) always report ready: they score on every
        # absorb and rely on continuous eviction.
        if self._score_mode == "snapkv":
            return self._snapkv_frozen is not None
        return True

    def score(self, block: ActiveBlock) -> float | None:
        if self._score_mode == "snapkv":
            return self._score_snapkv(block)
        if self._score_mode == "cumulative":
            return self._score_cumulative(block)
        return self._score_ewma(block)

    def _score_ewma(self, block: ActiveBlock) -> float | None:
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

    def _score_cumulative(self, block: ActiveBlock) -> float | None:
        # H2O's heavy-hitter statistic: per-block lifetime attention mass, no
        # decay. Normalized against the running max so the survivor with the
        # highest cumulative attention scores exactly 1.0 (and is auto-protected
        # via the score>=1.0 rule in EvokeManager._evictable_blocks). Other
        # blocks scale linearly; the existing "evict lowest" sort in
        # _enforce_budget then reproduces H2O's equilibrium where top-K by
        # cumulative survive and the lowest get freed first.
        value = self._cumulative.get(block.block_id)
        if value is None:
            return None
        if not self._cumulative:
            return None
        max_val = max(self._cumulative.values())
        if max_val <= 0.0:
            return None
        return float(value / max_val)

    def _score_snapkv(self, block: ActiveBlock) -> float | None:
        # SnapKV (Liu et al., NeurIPS 2024) reads from the most recent frozen
        # snapshot. Returns None until snapshot() has been called at least
        # once so eviction during haystack prefill falls back to the recency /
        # coherence priors instead of using a stale or empty SnapKV signal.
        # Post-snapshot the score is the block's observation-window mass
        # normalized against the snapshot's max so the highest-attended block
        # scores 1.0 and is auto-protected by the score>=1.0 rule in
        # EvokeManager._evictable_blocks.
        if self._snapkv_frozen is None or not self._snapkv_frozen:
            return None
        value = self._snapkv_frozen.get(block.block_id)
        if value is None:
            return None
        max_val = max(self._snapkv_frozen.values())
        if max_val <= 0.0:
            return None
        return float(value / max_val)

    def forget(self, block_id: int) -> None:
        # Called by the manager when a block is permanently evicted (no
        # recovery saved). Keeps the score maps from growing unbounded over
        # long-running sessions. Drops sliding-window, cumulative, and
        # SnapKV pending+frozen entries so mode switches after a long run
        # start from a clean slate.
        self._scores.pop(block_id, None)
        self._cumulative.pop(block_id, None)
        self._snapkv_pending.pop(block_id, None)
        if self._snapkv_frozen is not None:
            self._snapkv_frozen.pop(block_id, None)

    def _push(self, block_id: int, value: float) -> None:
        win = self._scores.get(block_id)
        if win is None:
            win = deque(maxlen=self._n_window)
            self._scores[block_id] = win
        win.append(value)
        self._cumulative[block_id] = self._cumulative.get(block_id, 0.0) + value
