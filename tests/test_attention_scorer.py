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


class TestAttentionScorerCumulativeMode:
    def test_cumulative_normalizes_to_max(self):
        engine = FakeEngine()
        scorer = AttentionScorer(
            engine, layer=0, n_window=4, decay=0.9, score_mode="cumulative"
        )
        block_a = _block(1, 0, 4)
        block_b = _block(2, 4, 8)
        weights = np.zeros((1, 1, 8), dtype=np.float32)
        weights[:, :, 0:4] = 0.25
        weights[:, :, 4:8] = 0.125
        engine.stub_capture(weights)
        scorer.absorb_last_decode([block_a, block_b])
        sa = scorer.score(block_a)
        sb = scorer.score(block_b)
        assert sa == 1.0
        assert abs(sb - 0.5) < 1e-5

    def test_cumulative_accumulates_without_decay(self):
        engine = FakeEngine()
        scorer = AttentionScorer(
            engine, layer=0, n_window=4, decay=0.5, score_mode="cumulative"
        )
        block = _block(1, 0, 4)
        for _ in range(3):
            w = np.full((1, 1, 4), 0.125, dtype=np.float32)
            engine.stub_capture(w)
            scorer.absorb_last_decode([block])
        # Single tracked block -> normalized score is always max == 1.0,
        # but the underlying cumulative is sum-of-per-steps. Verify by
        # checking the second block stays proportional.
        block_b = _block(2, 0, 4)
        weights2 = np.full((1, 1, 4), 0.0625, dtype=np.float32)
        engine.stub_capture(weights2)
        scorer.absorb_last_decode([block_b])
        sa = scorer.score(block)
        sb = scorer.score(block_b)
        # block cumulative = 0.5 * 3 = 1.5; block_b cumulative = 0.25.
        # max = 1.5; normalized: block=1.0, block_b ~= 0.1667.
        assert sa == 1.0
        assert abs(sb - 0.25 / 1.5) < 1e-5

    def test_cumulative_forget_drops_state(self):
        engine = FakeEngine()
        scorer = AttentionScorer(
            engine, layer=0, n_window=4, decay=0.9, score_mode="cumulative"
        )
        block = _block(1, 0, 4)
        w = np.full((1, 1, 4), 0.25, dtype=np.float32)
        engine.stub_capture(w)
        scorer.absorb_last_decode([block])
        assert scorer.score(block) is not None
        scorer.forget(block.block_id)
        assert scorer.score(block) is None

    def test_cumulative_returns_none_before_absorb(self):
        engine = FakeEngine()
        scorer = AttentionScorer(
            engine, layer=0, n_window=4, decay=0.9, score_mode="cumulative"
        )
        block = _block(1, 0, 4)
        assert scorer.score(block) is None


class TestAttentionScorerSnapKVMode:
    """SnapKV mode tests.

    SnapKV (Liu et al., NeurIPS 2024) scores from the last W query tokens of
    the most recent decode batch (the observation window), accumulates into a
    pending bucket, and freezes that bucket into the live score table on
    snapshot(). EvokeManager.process_user_message calls snapshot() at the end
    of every user message so the next eviction pass uses question-window
    attention to pick survivors.
    """

    def test_score_none_before_snapshot(self):
        # Before snapshot() the scorer must return None so eviction falls back
        # to recency / coherence priors instead of using an empty SnapKV table.
        engine = FakeEngine()
        scorer = AttentionScorer(
            engine,
            layer=0,
            n_window=4,
            decay=0.9,
            score_mode="snapkv",
            snapkv_observation_window=2,
        )
        block = _block(1, 0, 4)
        weights = np.full((1, 1, 1, 4), 0.25, dtype=np.float32)
        engine.stub_capture(weights)
        scorer.absorb_last_decode([block])
        # absorbed into pending but not yet frozen
        assert scorer.score(block) is None

    def test_snapshot_freezes_pending(self):
        engine = FakeEngine()
        scorer = AttentionScorer(
            engine,
            layer=0,
            n_window=4,
            decay=0.9,
            score_mode="snapkv",
            snapkv_observation_window=2,
        )
        block_a = _block(1, 0, 4)
        block_b = _block(2, 4, 8)
        # Two queries; both attend uniformly across all 8 kv positions. Block
        # A gets sum=0.5 per query * 2 queries = 1.0; block B same. Equal so
        # both normalize to 1.0.
        weights = np.full((1, 1, 2, 8), 0.125, dtype=np.float32)
        engine.stub_capture(weights)
        scorer.absorb_last_decode([block_a, block_b])
        scorer.snapshot()
        assert abs(scorer.score(block_a) - 1.0) < 1e-5
        assert abs(scorer.score(block_b) - 1.0) < 1e-5

    def test_observation_window_caps_query_count(self):
        # 4 queries total, observation_window=2 -> only the last 2 contribute.
        # Queries 0 and 1 give all attention to block A; queries 2 and 3 give
        # all attention to block B. SnapKV should pick block B as the winner.
        engine = FakeEngine()
        scorer = AttentionScorer(
            engine,
            layer=0,
            n_window=4,
            decay=0.9,
            score_mode="snapkv",
            snapkv_observation_window=2,
        )
        block_a = _block(1, 0, 4)
        block_b = _block(2, 4, 8)
        weights = np.zeros((1, 1, 4, 8), dtype=np.float32)
        # First two queries all mass on block A
        weights[0, 0, 0, 0:4] = 0.25
        weights[0, 0, 1, 0:4] = 0.25
        # Last two queries all mass on block B (the observation window)
        weights[0, 0, 2, 4:8] = 0.25
        weights[0, 0, 3, 4:8] = 0.25
        engine.stub_capture(weights)
        scorer.absorb_last_decode([block_a, block_b])
        scorer.snapshot()
        # SnapKV uses observation window only -> block B is max, block A
        # contributes nothing.
        assert scorer.score(block_b) == 1.0
        assert scorer.score(block_a) == 0.0

    def test_post_snapshot_absorbs_do_not_change_score(self):
        # Once frozen, subsequent absorbs accumulate into a fresh pending
        # bucket without disturbing the live score. The score only changes on
        # the next snapshot(). This matches SnapKV's "compress once per
        # prompt" semantics: generate-time decodes between user messages must
        # not shift the eviction policy mid-turn.
        engine = FakeEngine()
        scorer = AttentionScorer(
            engine,
            layer=0,
            n_window=4,
            decay=0.9,
            score_mode="snapkv",
            snapkv_observation_window=4,
        )
        block_a = _block(1, 0, 4)
        block_b = _block(2, 4, 8)
        weights1 = np.zeros((1, 1, 4, 8), dtype=np.float32)
        weights1[:, :, :, 0:4] = 0.25  # all mass on block A
        engine.stub_capture(weights1)
        scorer.absorb_last_decode([block_a, block_b])
        scorer.snapshot()
        s_a_before = scorer.score(block_a)
        s_b_before = scorer.score(block_b)
        # Now absorb a contradictory capture (all mass on block B). Without
        # another snapshot the live score must not move.
        weights2 = np.zeros((1, 1, 4, 8), dtype=np.float32)
        weights2[:, :, :, 4:8] = 0.25
        engine.stub_capture(weights2)
        scorer.absorb_last_decode([block_a, block_b])
        assert scorer.score(block_a) == s_a_before
        assert scorer.score(block_b) == s_b_before
        # After a second snapshot the score reflects only the new pending
        # bucket (block B is now max). Pending is reset on snapshot so the
        # prior turn's accumulation does not leak forward.
        scorer.snapshot()
        assert scorer.score(block_b) == 1.0
        assert scorer.score(block_a) == 0.0

    def test_snapshot_noop_for_other_modes(self):
        # snapshot() must be safe to call regardless of mode (the manager
        # calls it unconditionally on every user message). For ewma or
        # cumulative scorers it must not change behavior.
        engine = FakeEngine()
        scorer = AttentionScorer(
            engine, layer=0, n_window=4, decay=0.9, score_mode="cumulative"
        )
        block = _block(1, 0, 4)
        weights = np.full((1, 1, 1, 4), 0.25, dtype=np.float32)
        engine.stub_capture(weights)
        scorer.absorb_last_decode([block])
        before = scorer.score(block)
        scorer.snapshot()
        assert scorer.score(block) == before

    def test_forget_drops_pending_and_frozen(self):
        engine = FakeEngine()
        scorer = AttentionScorer(
            engine,
            layer=0,
            n_window=4,
            decay=0.9,
            score_mode="snapkv",
            snapkv_observation_window=4,
        )
        block = _block(1, 0, 4)
        weights = np.full((1, 1, 1, 4), 0.25, dtype=np.float32)
        engine.stub_capture(weights)
        scorer.absorb_last_decode([block])
        scorer.snapshot()
        assert scorer.score(block) is not None
        scorer.forget(block.block_id)
        assert scorer.score(block) is None

    def test_is_eviction_ready_only_after_snapshot(self):
        # Manager._enforce_budget reads this to defer eviction during
        # add_context (before any snapshot has fired). Without it, every
        # block ties at 0.0 because score() returns None pre-snapshot, and
        # insertion-order eviction drops the needle before the SnapKV
        # observation window ever sees the question.
        engine = FakeEngine()
        scorer = AttentionScorer(
            engine,
            layer=0,
            n_window=4,
            decay=0.9,
            score_mode="snapkv",
            snapkv_observation_window=2,
        )
        assert scorer.is_eviction_ready() is False
        scorer.snapshot()
        assert scorer.is_eviction_ready() is True

    def test_is_eviction_ready_true_for_non_snapkv_modes(self):
        # ewma and cumulative score on every absorb so the manager must not
        # gate them. is_eviction_ready returns True unconditionally for
        # non-snapkv modes; the manager calls hasattr so missing the method
        # entirely is also safe.
        for mode in ("ewma", "cumulative"):
            engine = FakeEngine()
            scorer = AttentionScorer(
                engine, layer=0, n_window=4, decay=0.9, score_mode=mode
            )
            assert scorer.is_eviction_ready() is True

    def test_zero_observation_window_returns_none(self):
        # Degenerate config (observation_window=0) means no queries contribute;
        # snapshot freezes an empty table and score() returns None for any
        # block. This prevents a misconfigured strategy from silently scoring
        # every block at 0 and triggering arbitrary eviction order.
        engine = FakeEngine()
        scorer = AttentionScorer(
            engine,
            layer=0,
            n_window=4,
            decay=0.9,
            score_mode="snapkv",
            snapkv_observation_window=0,
        )
        block = _block(1, 0, 4)
        weights = np.full((1, 1, 2, 4), 0.25, dtype=np.float32)
        engine.stub_capture(weights)
        scorer.absorb_last_decode([block])
        scorer.snapshot()
        assert scorer.score(block) is None
