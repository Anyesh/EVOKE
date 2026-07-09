"""Unit tests for evoke.jlens_scorer.

Uses a FakeEngine that satisfies the subset of LlamaCppEngine the scorer
touches (layer_inp_capture_* methods, supports_kv_block flag) without
loading a real model. Probe loading, per-block aggregation over batches,
min-max normalization, and forget semantics are exercised here; C-primitive
end-to-end correctness is covered by scripts/verify_layer_inp_capture.py
against a real GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from evoke.jlens_scorer import JLensScorer
from evoke.types import ActiveBlock


D = 4


class FakeEngine:
    supports_kv_block = True

    def __init__(self) -> None:
        self.enabled_layers: list[int] | None = None
        self._pending: tuple[int, dict[int, np.ndarray]] | None = None

    def layer_inp_capture_enable(self, layers: list[int]) -> None:
        self.enabled_layers = list(layers)

    def layer_inp_capture_read(self) -> tuple[int, dict[int, np.ndarray]] | None:
        pending = self._pending
        self._pending = None
        return pending

    def stub_rows(self, start: int, rows_by_layer: dict[int, np.ndarray]) -> None:
        self._pending = (start, {k: v.astype(np.float32) for k, v in rows_by_layer.items()})


def _block(bid: int, start: int, end: int) -> ActiveBlock:
    return ActiveBlock(
        block_id=bid,
        logical_start=start,
        logical_end=end,
        token_ids=list(range(end - start)),
    )


@pytest.fixture
def probe_path(tmp_path):
    # Layer 3 probe: prediction = first feature. Layer 5: second feature + 1.
    arrays = {
        "layers": np.array([3, 5]),
        "stats": np.array(["kurtosis"]),
        "L3_kurtosis_w": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "L3_kurtosis_b": np.float32(0.0),
        "L5_kurtosis_w": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "L5_kurtosis_b": np.float32(1.0),
    }
    path = tmp_path / "probe.npz"
    np.savez(path, **arrays)
    return str(path)


def rows_with_first_feature(values: list[float]) -> np.ndarray:
    rows = np.zeros((len(values), D), dtype=np.float32)
    rows[:, 0] = values
    return rows


class TestJLensScorer:
    def test_enables_artifact_layers_on_init(self, probe_path):
        engine = FakeEngine()
        JLensScorer(engine, probe_path=probe_path)
        assert engine.enabled_layers == [3, 5]

    def test_layers_subset_honored(self, probe_path):
        engine = FakeEngine()
        JLensScorer(engine, probe_path=probe_path, layers=[3])
        assert engine.enabled_layers == [3]

    def test_unknown_layer_rejected(self, probe_path):
        engine = FakeEngine()
        with pytest.raises(KeyError):
            JLensScorer(engine, probe_path=probe_path, layers=[7])

    def test_score_none_before_absorb(self, probe_path):
        engine = FakeEngine()
        scorer = JLensScorer(engine, probe_path=probe_path, layers=[3])
        assert scorer.score(_block(1, 0, 4)) is None

    def test_min_max_normalization_over_blocks(self, probe_path):
        engine = FakeEngine()
        scorer = JLensScorer(engine, probe_path=probe_path, layers=[3])
        blocks = [_block(1, 0, 2), _block(2, 2, 4), _block(3, 4, 6)]
        # Block means of the first feature: 1.0, 3.0, 2.0.
        engine.stub_rows(0, {3: rows_with_first_feature([1, 1, 3, 3, 2, 2])})
        scorer.absorb_last_decode(blocks)
        assert scorer.score(blocks[0]) == pytest.approx(0.0)
        assert scorer.score(blocks[1]) == pytest.approx(1.0)
        assert scorer.score(blocks[2]) == pytest.approx(0.5)

    def test_single_block_scores_one(self, probe_path):
        engine = FakeEngine()
        scorer = JLensScorer(engine, probe_path=probe_path, layers=[3])
        block = _block(1, 0, 3)
        engine.stub_rows(0, {3: rows_with_first_feature([-2, -2, -2])})
        scorer.absorb_last_decode([block])
        assert scorer.score(block) == pytest.approx(1.0)

    def test_mean_accumulates_across_batches(self, probe_path):
        engine = FakeEngine()
        scorer = JLensScorer(engine, probe_path=probe_path, layers=[3])
        block_a = _block(1, 0, 4)
        block_b = _block(2, 4, 6)
        engine.stub_rows(0, {3: rows_with_first_feature([0, 0])})
        scorer.absorb_last_decode([block_a])
        # Second batch finishes block_a (mean 0+0+4+4 -> 2.0) and fills
        # block_b (mean 1.0): a should outrank b.
        engine.stub_rows(2, {3: rows_with_first_feature([4, 4, 1, 1])})
        scorer.absorb_last_decode([block_a, block_b])
        assert scorer.score(block_a) == pytest.approx(1.0)
        assert scorer.score(block_b) == pytest.approx(0.0)

    def test_max_aggregation(self, probe_path):
        engine = FakeEngine()
        scorer = JLensScorer(engine, probe_path=probe_path, layers=[3], block_agg="max")
        blocks = [_block(1, 0, 2), _block(2, 2, 4)]
        # Maxes: block 1 -> 5.0, block 2 -> 2.0; means would tie at 2.0.
        engine.stub_rows(0, {3: rows_with_first_feature([5, -1, 2, 2])})
        scorer.absorb_last_decode(blocks)
        assert scorer.score(blocks[0]) == pytest.approx(1.0)
        assert scorer.score(blocks[1]) == pytest.approx(0.0)

    def test_multi_layer_predictions_averaged(self, probe_path):
        engine = FakeEngine()
        scorer = JLensScorer(engine, probe_path=probe_path)
        blocks = [_block(1, 0, 1), _block(2, 1, 2)]
        rows_l3 = rows_with_first_feature([4, 0])
        rows_l5 = np.zeros((2, D), dtype=np.float32)
        rows_l5[:, 1] = [1, 9]  # layer-5 predictions: 2.0, 10.0
        # Layer means: block 1 -> (4+2)/2 = 3, block 2 -> (0+10)/2 = 5.
        engine.stub_rows(0, {3: rows_l3, 5: rows_l5})
        scorer.absorb_last_decode(blocks)
        assert scorer.score(blocks[1]) == pytest.approx(1.0)
        assert scorer.score(blocks[0]) == pytest.approx(0.0)

    def test_partial_overlap_assigns_by_position(self, probe_path):
        engine = FakeEngine()
        scorer = JLensScorer(engine, probe_path=probe_path, layers=[3])
        blocks = [_block(1, 0, 4), _block(2, 4, 8)]
        # Batch covers positions 2..6: rows land 2 in each block.
        engine.stub_rows(2, {3: rows_with_first_feature([1, 1, 7, 7])})
        scorer.absorb_last_decode(blocks)
        assert scorer.score(blocks[0]) == pytest.approx(0.0)
        assert scorer.score(blocks[1]) == pytest.approx(1.0)

    def test_block_outside_batch_untouched(self, probe_path):
        engine = FakeEngine()
        scorer = JLensScorer(engine, probe_path=probe_path, layers=[3])
        old = _block(1, 0, 2)
        new = _block(2, 2, 4)
        engine.stub_rows(0, {3: rows_with_first_feature([6, 6])})
        scorer.absorb_last_decode([old])
        engine.stub_rows(2, {3: rows_with_first_feature([1, 1])})
        scorer.absorb_last_decode([old, new])
        assert scorer.score(old) == pytest.approx(1.0)
        assert scorer.score(new) == pytest.approx(0.0)

    def test_empty_capture_is_noop(self, probe_path):
        engine = FakeEngine()
        scorer = JLensScorer(engine, probe_path=probe_path, layers=[3])
        scorer.absorb_last_decode([_block(1, 0, 2)])
        assert scorer.score(_block(1, 0, 2)) is None

    def test_forget_drops_block(self, probe_path):
        engine = FakeEngine()
        scorer = JLensScorer(engine, probe_path=probe_path, layers=[3])
        blocks = [_block(1, 0, 2), _block(2, 2, 4)]
        engine.stub_rows(0, {3: rows_with_first_feature([1, 1, 3, 3])})
        scorer.absorb_last_decode(blocks)
        scorer.forget(1)
        assert scorer.score(blocks[0]) is None
        assert scorer.score(blocks[1]) == pytest.approx(1.0)
