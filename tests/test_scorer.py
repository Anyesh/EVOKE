import numpy as np

from evoke.config import EvokeConfig
from evoke.scorer import RelevanceScorer, cosine_similarity
from evoke.types import ActiveBlock, BlockSource


def _make_block(
    block_id: int,
    logical_start: int,
    logical_end: int,
    is_sink: bool = False,
    embedding: np.ndarray | None = None,
) -> ActiveBlock:
    size = logical_end - logical_start
    return ActiveBlock(
        block_id=block_id,
        logical_start=logical_start,
        logical_end=logical_end,
        token_ids=list(range(size)),
        representative_embedding=embedding,
        is_sink=is_sink,
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

        sink_block = _make_block(0, 0, 4, is_sink=True)
        score = scorer.score(sink_block, current_pos=1000, context_length=2000)
        assert score == 1.0

    def test_recent_blocks_score_higher_than_old(self):
        config = EvokeConfig(sink_count=4, block_size=128, recency_decay=0.01)
        scorer = RelevanceScorer(config)

        recent = _make_block(1, 900, 1000)
        old = _make_block(2, 100, 200)

        score_recent = scorer.score(recent, current_pos=1000, context_length=2000)
        score_old = scorer.score(old, current_pos=1000, context_length=2000)
        assert score_recent > score_old

    def test_coherent_blocks_score_higher(self):
        config = EvokeConfig(
            sink_count=4, block_size=128, w_recency=0.0, w_coherence=1.0
        )
        scorer = RelevanceScorer(config)

        scorer.update_recent_context(np.array([1.0, 0.0, 0.0, 0.0]))

        similar = _make_block(1, 100, 200, embedding=np.array([0.9, 0.1, 0.0, 0.0]))
        dissimilar = _make_block(2, 200, 300, embedding=np.array([0.0, 0.0, 0.9, 0.1]))

        score_sim = scorer.score(similar, current_pos=500, context_length=1000)
        score_dis = scorer.score(dissimilar, current_pos=500, context_length=1000)
        assert score_sim > score_dis

    def test_score_blocks_returns_all(self):
        config = EvokeConfig(sink_count=4, block_size=128)
        scorer = RelevanceScorer(config)

        blocks = [
            _make_block(0, 0, 128),
            _make_block(1, 128, 256),
            _make_block(2, 256, 384),
        ]
        scores = scorer.score_blocks(blocks, current_pos=384, context_length=1000)
        assert len(scores) == 3
        assert all(0 <= s <= 1.0 for s in scores.values())


class TestTaskFocusEmbedding:
    def test_first_message_sets_focus(self):
        config = EvokeConfig(
            sink_count=4, block_size=128, w_recency=0.0, w_coherence=1.0
        )
        scorer = RelevanceScorer(config)
        scorer.update_recent_context(np.array([1.0, 0.0, 0.0, 0.0]))

        coherent = _make_block(1, 100, 200, embedding=np.array([0.95, 0.05, 0.0, 0.0]))
        incoherent = _make_block(2, 100, 200, embedding=np.array([0.0, 0.0, 0.0, 1.0]))
        assert scorer.score(coherent, 500, 1000) > 0.8
        assert scorer.score(incoherent, 500, 1000) < 0.6

    def test_focus_snaps_on_topic_shift(self):
        config = EvokeConfig(
            sink_count=4,
            block_size=128,
            w_recency=0.0,
            w_coherence=1.0,
            task_boundary_threshold=0.3,
        )
        scorer = RelevanceScorer(config)
        # Topic 1 fills the focus, blocks coherent with topic 1 score high.
        scorer.update_recent_context(np.array([1.0, 0.0, 0.0, 0.0]))
        scorer.update_recent_context(np.array([0.95, 0.05, 0.0, 0.0]))
        topic1_block = _make_block(
            1, 100, 200, embedding=np.array([0.95, 0.05, 0.0, 0.0])
        )
        assert scorer.score(topic1_block, 500, 1000) > 0.85

        # Topic 2 message arrives — orthogonal embedding triggers implicit
        # boundary detection. Focus snaps; topic-1 blocks lose coherence.
        scorer.update_recent_context(np.array([0.0, 0.0, 0.0, 1.0]))
        topic1_after_shift = scorer.score(topic1_block, 500, 1000)
        assert topic1_after_shift < 0.6, (
            "topic-1 blocks should lose coherence after task boundary"
        )

    def test_explicit_signal_forces_boundary(self):
        config = EvokeConfig(
            sink_count=4,
            block_size=128,
            w_recency=0.0,
            w_coherence=1.0,
            # threshold low enough that the test's similar embeddings would
            # NORMALLY blend (not snap) — only the explicit signal forces it.
            task_boundary_threshold=-1.0,
        )
        scorer = RelevanceScorer(config)
        scorer.update_recent_context(np.array([1.0, 0.0, 0.0, 0.0]))
        scorer.update_recent_context(np.array([0.95, 0.05, 0.0, 0.0]))

        # Without explicit signal, a similar-ish embedding EMA-blends and
        # topic-1 blocks remain coherent.
        scorer.update_recent_context(np.array([0.5, 0.5, 0.0, 0.0]))
        topic1_block = _make_block(
            1, 100, 200, embedding=np.array([0.95, 0.05, 0.0, 0.0])
        )
        coherent_score = scorer.score(topic1_block, 500, 1000)

        # WITH explicit signal, focus snaps to a new message even when
        # similar. topic1_block's coherence drops.
        scorer.signal_task_boundary()
        scorer.update_recent_context(np.array([0.0, 0.0, 1.0, 0.0]))
        after_signal = scorer.score(topic1_block, 500, 1000)
        assert after_signal < coherent_score


class TestSourceAwareScoring:
    def test_user_block_has_score_floor(self):
        config = EvokeConfig(
            sink_count=4,
            block_size=128,
            recency_decay=0.01,
            conversation_score_floor=0.6,
        )
        scorer = RelevanceScorer(config)

        old_user_block = _make_block(1, 100, 200)
        old_user_block.source = BlockSource.USER

        score = scorer.score(old_user_block, current_pos=5000, context_length=5000)
        assert score >= 0.6

    def test_assistant_block_has_score_floor(self):
        config = EvokeConfig(
            sink_count=4,
            block_size=128,
            recency_decay=0.01,
            assistant_score_floor=0.5,
        )
        scorer = RelevanceScorer(config)

        old_assistant = _make_block(1, 100, 200)
        old_assistant.source = BlockSource.ASSISTANT

        score = scorer.score(old_assistant, current_pos=5000, context_length=5000)
        assert score >= 0.5

    def test_document_block_has_no_floor(self):
        config = EvokeConfig(
            sink_count=4,
            block_size=128,
            recency_decay=0.01,
            conversation_score_floor=0.6,
        )
        scorer = RelevanceScorer(config)

        old_doc = _make_block(1, 100, 200)
        old_doc.source = BlockSource.DOCUMENT

        score = scorer.score(old_doc, current_pos=5000, context_length=5000)
        assert score < 0.6

    def test_eviction_prefers_document_over_conversation(self):
        config = EvokeConfig(
            sink_count=4,
            block_size=128,
            recency_decay=0.01,
            conversation_score_floor=0.6,
        )
        scorer = RelevanceScorer(config)

        doc_block = _make_block(1, 100, 200)
        doc_block.source = BlockSource.DOCUMENT

        user_block = _make_block(2, 200, 300)
        user_block.source = BlockSource.USER

        score_doc = scorer.score(doc_block, current_pos=5000, context_length=5000)
        score_user = scorer.score(user_block, current_pos=5000, context_length=5000)
        assert score_user > score_doc


class TestPriorityScaling:
    def test_priority_above_one_boosts_score(self):
        config = EvokeConfig(
            sink_count=4,
            block_size=128,
            w_recency=1.0,
            w_coherence=0.0,
            conversation_score_floor=0.0,
            assistant_score_floor=0.0,
        )
        scorer = RelevanceScorer(config)
        normal = _make_block(1, 100, 200)
        high = _make_block(2, 100, 200)
        high.priority = 2.0
        s_normal = scorer.score(normal, current_pos=5000, context_length=5000)
        s_high = scorer.score(high, current_pos=5000, context_length=5000)
        assert s_high > s_normal
        # Priority capped at 1.0 — even priority=2 cannot exceed sink ceiling.
        assert s_high <= 1.0

    def test_priority_below_one_lowers_score(self):
        config = EvokeConfig(
            sink_count=4,
            block_size=128,
            w_recency=1.0,
            w_coherence=0.0,
            conversation_score_floor=0.0,
        )
        scorer = RelevanceScorer(config)
        normal = _make_block(1, 100, 200)
        low = _make_block(2, 100, 200)
        low.priority = 0.5
        s_normal = scorer.score(normal, current_pos=5000, context_length=5000)
        s_low = scorer.score(low, current_pos=5000, context_length=5000)
        assert s_low < s_normal

    def test_priority_does_not_override_sink(self):
        config = EvokeConfig(sink_count=4, block_size=128)
        scorer = RelevanceScorer(config)
        sink = _make_block(1, 0, 128, is_sink=True)
        sink.priority = 0.1
        # Sink protection wins regardless of priority.
        assert scorer.score(sink, current_pos=5000, context_length=5000) == 1.0


class TestAttentionScorerSlot:
    def test_no_attention_scorer_falls_back_to_heuristic(self):
        config = EvokeConfig(
            sink_count=4,
            block_size=128,
            w_recency=0.4,
            w_coherence=0.6,
            w_attention=0.5,
        )
        scorer = RelevanceScorer(config)  # no attention_scorer
        block = _make_block(1, 100, 200)
        # Should compute without error and produce a score in [0,1].
        s = scorer.score(block, current_pos=1000, context_length=1000)
        assert 0.0 <= s <= 1.0

    def test_attention_scorer_shifts_score(self):
        config = EvokeConfig(
            sink_count=4,
            block_size=128,
            w_recency=0.2,
            w_coherence=0.2,
            w_attention=0.6,
            conversation_score_floor=0.0,
        )

        class HighAttention:
            def score(self, block):
                return 1.0

        class LowAttention:
            def score(self, block):
                return 0.0

        block = _make_block(1, 100, 200)
        block_old = _make_block(2, 0, 100)

        hi = RelevanceScorer(config, attention_scorer=HighAttention())
        lo = RelevanceScorer(config, attention_scorer=LowAttention())
        s_hi = hi.score(block_old, current_pos=5000, context_length=5000)
        s_lo = lo.score(block_old, current_pos=5000, context_length=5000)
        # Same old block: high-attention scorer should rank it higher than
        # low-attention scorer.
        assert s_hi > s_lo

    def test_attention_score_none_treated_as_no_signal(self):
        config = EvokeConfig(
            sink_count=4, block_size=128, w_attention=0.5, w_recency=0.5
        )

        class NoSignal:
            def score(self, block):
                return None

        scorer = RelevanceScorer(config, attention_scorer=NoSignal())
        block = _make_block(1, 100, 200)
        # Should not error; falls back to recency+coherence-only branch.
        s = scorer.score(block, current_pos=1000, context_length=1000)
        assert 0.0 <= s <= 1.0

    def test_set_attention_scorer_swaps_at_runtime(self):
        config = EvokeConfig(
            sink_count=4,
            block_size=128,
            w_attention=0.8,
            w_recency=0.1,
            w_coherence=0.1,
            conversation_score_floor=0.0,
        )

        class FixedAttention:
            def __init__(self, v):
                self.v = v

            def score(self, block):
                return self.v

        scorer = RelevanceScorer(config)
        block = _make_block(1, 0, 100)
        s_initial = scorer.score(block, current_pos=5000, context_length=5000)
        scorer.set_attention_scorer(FixedAttention(0.95))
        s_with_attn = scorer.score(block, current_pos=5000, context_length=5000)
        assert s_with_attn > s_initial


class TestJLensScorerSlot:
    def test_no_jlens_scorer_falls_back_to_heuristic(self):
        config = EvokeConfig(
            sink_count=4,
            block_size=128,
            w_recency=0.4,
            w_coherence=0.6,
            w_jlens=0.5,
        )
        scorer = RelevanceScorer(config)
        block = _make_block(1, 100, 200)
        s = scorer.score(block, current_pos=1000, context_length=1000)
        assert 0.0 <= s <= 1.0

    def test_jlens_scorer_shifts_score(self):
        config = EvokeConfig(
            sink_count=4,
            block_size=128,
            w_recency=0.2,
            w_coherence=0.2,
            w_jlens=0.6,
            conversation_score_floor=0.0,
        )

        class HighJLens:
            def score(self, block):
                return 1.0

        class LowJLens:
            def score(self, block):
                return 0.0

        block = _make_block(1, 100, 200)
        high = RelevanceScorer(config, jlens_scorer=HighJLens())
        low = RelevanceScorer(config, jlens_scorer=LowJLens())
        s_high = high.score(block, current_pos=5000, context_length=5000)
        s_low = low.score(block, current_pos=5000, context_length=5000)
        assert s_high > s_low
        assert s_high - s_low > 0.4

    def test_jlens_none_signal_drops_from_mix(self):
        config = EvokeConfig(
            sink_count=4,
            block_size=128,
            w_recency=0.5,
            w_coherence=0.5,
            w_jlens=0.9,
            conversation_score_floor=0.0,
        )

        class NoSignal:
            def score(self, block):
                return None

        block = _make_block(1, 100, 200)
        with_scorer = RelevanceScorer(config, jlens_scorer=NoSignal())
        without = RelevanceScorer(config)
        pos = dict(current_pos=5000, context_length=5000)
        assert with_scorer.score(block, **pos) == without.score(block, **pos)

    def test_jlens_and_attention_combine(self):
        config = EvokeConfig(
            sink_count=4,
            block_size=128,
            w_recency=0.0,
            w_coherence=0.0,
            w_attention=0.5,
            w_jlens=0.5,
            conversation_score_floor=0.0,
        )

        class Fixed:
            def __init__(self, v):
                self._v = v

            def score(self, block):
                return self._v

        block = _make_block(1, 100, 200)
        scorer = RelevanceScorer(
            config, attention_scorer=Fixed(1.0), jlens_scorer=Fixed(0.0)
        )
        s = scorer.score(block, current_pos=5000, context_length=5000)
        assert abs(s - 0.5) < 1e-9
