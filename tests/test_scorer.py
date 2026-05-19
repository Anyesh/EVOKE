import numpy as np

from evoke.config import EvokeConfig
from evoke.scorer import RelevanceScorer, cosine_similarity
from evoke.types import ActiveBlock


def _make_block(
    block_id: int,
    logical_start: int,
    logical_end: int,
    original_start: int | None = None,
    embedding: np.ndarray | None = None,
) -> ActiveBlock:
    if original_start is None:
        original_start = logical_start
    size = logical_end - logical_start
    return ActiveBlock(
        block_id=block_id,
        logical_start=logical_start,
        logical_end=logical_end,
        original_start=original_start,
        original_end=original_start + size,
        token_ids=list(range(size)),
        representative_embedding=embedding,
    )


class TestCosine:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(cosine_similarity(a, b) + 1.0) < 1e-6

    def test_zero_vector(self):
        a = np.zeros(3)
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, b) == 0.0


class TestRelevanceScorer:
    def test_sink_blocks_always_score_1(self):
        config = EvokeConfig(sink_count=4, block_size=4)
        scorer = RelevanceScorer(config)

        sink_block = _make_block(0, 0, 4, original_start=0)
        score = scorer.score(sink_block, current_pos=1000, context_length=2000)
        assert score == 1.0

    def test_recent_blocks_score_higher_than_old(self):
        config = EvokeConfig(sink_count=4, block_size=128, recency_decay=0.01)
        scorer = RelevanceScorer(config)

        recent = _make_block(1, 900, 1000, original_start=900)
        old = _make_block(2, 100, 200, original_start=100)

        score_recent = scorer.score(recent, current_pos=1000, context_length=2000)
        score_old = scorer.score(old, current_pos=1000, context_length=2000)
        assert score_recent > score_old

    def test_coherent_blocks_score_higher(self):
        config = EvokeConfig(
            sink_count=4, block_size=128, w_recency=0.0, w_coherence=1.0
        )
        scorer = RelevanceScorer(config)

        context_emb = np.array([1.0, 0.0, 0.0, 0.0])
        scorer.update_recent_context(context_emb)

        similar = _make_block(
            1, 100, 200, original_start=100, embedding=np.array([0.9, 0.1, 0.0, 0.0])
        )
        dissimilar = _make_block(
            2, 200, 300, original_start=200, embedding=np.array([0.0, 0.0, 0.9, 0.1])
        )

        score_sim = scorer.score(similar, current_pos=500, context_length=1000)
        score_dis = scorer.score(dissimilar, current_pos=500, context_length=1000)
        assert score_sim > score_dis

    def test_score_blocks_returns_all(self):
        config = EvokeConfig(sink_count=4, block_size=128)
        scorer = RelevanceScorer(config)

        blocks = [
            _make_block(0, 0, 128, original_start=0),
            _make_block(1, 128, 256, original_start=128),
            _make_block(2, 256, 384, original_start=256),
        ]
        scores = scorer.score_blocks(blocks, current_pos=384, context_length=1000)
        assert len(scores) == 3
        assert all(0 <= s <= 1.0 for s in scores.values())
