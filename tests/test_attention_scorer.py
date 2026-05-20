"""Unit tests for evoke.attention_scorer.

Uses a FakeEngine that satisfies the subset of LlamaCppEngine the scorer
touches (attn_capture_* methods, supports_kv_block flag, _attn_capture_buf
attribute) without loading a real model. Per-block aggregation, sliding
window, decay, and forget semantics are exercised here; the C-primitive
end-to-end correctness is covered by scripts/verify_attn_capture.py
against a real GPU.
"""

from __future__ import annotations

import numpy as np

from evoke.attention_scorer import AttentionScorer
from evoke.types import ActiveBlock


class FakeEngine:
    supports_kv_block = True

    def __init__(self) -> None:
        self._attn_capture_buf: np.ndarray | None = None
        # (n_layers, n_query, n_heads, n_kv) — 4-tuple after multi-layer.
        self._dims = (0, 0, 0, 0)
        self._written = 0

    def attn_capture_set_layer(self, layer: int) -> None:
        self._layer = layer

    def attn_capture_set_layers(self, layers: list[int]) -> None:
        self._layers = list(layers)

    def attn_capture_set_buffer(self, buf) -> None:
        self._attn_capture_buf = buf

    def attn_capture_get_dims(self) -> tuple[int, int, int, int]:
        return self._dims

    def attn_capture_get_written(self) -> int:
        return self._written

    def stub_capture(self, weights: np.ndarray) -> None:
        # weights shape: (n_layers, n_heads, n_query, n_kv); the
        # AttentionScorer provides the buffer; we copy into it. Tests
        # written against the single-layer (3D) shape are auto-promoted
        # to n_layers=1 here.
        if weights.ndim == 3:
            weights = weights[np.newaxis]
        n_layers, n_heads, n_query, n_kv = weights.shape
        flat = weights.ravel().astype(np.float32)
        assert self._attn_capture_buf is not None
        self._attn_capture_buf[: flat.size] = flat
        self._dims = (n_layers, n_query, n_heads, n_kv)
        self._written = flat.size


def _block(bid: int, start: int, end: int) -> ActiveBlock:
    size = end - start
    return ActiveBlock(
        block_id=bid,
        logical_start=start,
        logical_end=end,
        token_ids=list(range(size)),
    )


class TestAttentionScorer:
    def test_absorb_then_score_returns_value(self):
        engine = FakeEngine()
        scorer = AttentionScorer(engine, layer=0, n_window=4, decay=0.9)
        # One block covering kv positions 0..3 inclusive.
        blocks = [_block(1, 0, 4)]
        # Weights: all attention mass on block 1 -> per_step = 1.0
        weights = np.zeros((2, 1, 4), dtype=np.float32)
        weights[:, :, :] = 0.25  # uniform; block 1 gets all 4 columns => sum=1.0
        engine.stub_capture(weights)
        scorer.absorb_last_decode(blocks)
        s = scorer.score(blocks[0])
        assert s is not None
        assert abs(s - 1.0) < 1e-5

    def test_block_outside_kv_range_scores_zero(self):
        engine = FakeEngine()
        scorer = AttentionScorer(engine, layer=0, n_window=4, decay=0.9)
        far_block = _block(1, 100, 200)
        weights = np.full((2, 1, 4), 0.25, dtype=np.float32)
        engine.stub_capture(weights)
        scorer.absorb_last_decode([far_block])
        assert scorer.score(far_block) == 0.0

    def test_sliding_window_decay(self):
        engine = FakeEngine()
        scorer = AttentionScorer(engine, layer=0, n_window=3, decay=0.5)
        block = _block(1, 0, 4)
        # Push three captures: 0.0, 0.5, 1.0 (most recent last)
        for v in (0.0, 0.5, 1.0):
            w = np.full((1, 1, 4), v / 4.0, dtype=np.float32)
            engine.stub_capture(w)
            scorer.absorb_last_decode([block])
        s = scorer.score(block)
        # EWMA: (1.0*1.0 + 0.5*0.5 + 0.0*0.25) / (1.0+0.5+0.25) = 1.25 / 1.75
        assert abs(s - (1.0 * 1.0 + 0.5 * 0.5 + 0.0 * 0.25) / 1.75) < 1e-5

    def test_window_capped_at_n_window(self):
        engine = FakeEngine()
        scorer = AttentionScorer(engine, layer=0, n_window=2, decay=1.0)
        block = _block(1, 0, 4)
        # Push four captures; only the last two remain.
        for v in (0.1, 0.2, 0.3, 0.4):
            w = np.full((1, 1, 4), v / 4.0, dtype=np.float32)
            engine.stub_capture(w)
            scorer.absorb_last_decode([block])
        s = scorer.score(block)
        # Mean of last two = (0.3 + 0.4) / 2 = 0.35
        assert abs(s - 0.35) < 1e-5

    def test_score_none_before_any_absorb(self):
        engine = FakeEngine()
        scorer = AttentionScorer(engine, layer=0, n_window=4, decay=0.9)
        block = _block(1, 0, 4)
        assert scorer.score(block) is None

    def test_forget_drops_window(self):
        engine = FakeEngine()
        scorer = AttentionScorer(engine, layer=0, n_window=4, decay=0.9)
        block = _block(1, 0, 4)
        w = np.full((1, 1, 4), 0.25, dtype=np.float32)
        engine.stub_capture(w)
        scorer.absorb_last_decode([block])
        assert scorer.score(block) is not None
        scorer.forget(block.block_id)
        assert scorer.score(block) is None

    def test_absorb_noop_when_capture_empty(self):
        engine = FakeEngine()
        scorer = AttentionScorer(engine, layer=0, n_window=4, decay=0.9)
        block = _block(1, 0, 4)
        # No stub_capture: dims still (0,0,0). absorb must be a no-op.
        scorer.absorb_last_decode([block])
        assert scorer.score(block) is None
