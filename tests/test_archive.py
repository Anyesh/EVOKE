import numpy as np

from evoke.archive import ArchiveStore
from evoke.config import EvokeConfig
from evoke.mock_engine import MockEngine
from evoke.types import ArchiveBlock


def _make_archive_block(
    block_id: int, embedding: np.ndarray | None = None
) -> ArchiveBlock:
    if embedding is None:
        embedding = np.random.randn(8).astype(np.float32)
    return ArchiveBlock(
        block_id=block_id,
        token_ids=list(range(128)),
        original_positions=list(range(block_id * 128, (block_id + 1) * 128)),
        text=f"block {block_id}",
        representative_embedding=embedding,
        timestamp=0,
    )


class TestArchiveStore:
    def test_store_and_get(self):
        store = ArchiveStore(EvokeConfig())
        block = _make_archive_block(0)
        store.store(block)
        assert store.get(0) is block
        assert store.size == 1

    def test_remove(self):
        store = ArchiveStore(EvokeConfig())
        block = _make_archive_block(0)
        store.store(block)
        removed = store.remove(0)
        assert removed is block
        assert store.size == 0
        assert store.get(0) is None

    def test_retrieve_by_similarity(self):
        store = ArchiveStore(EvokeConfig())

        similar = _make_archive_block(
            0, embedding=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        dissimilar = ArchiveBlock(
            block_id=99,
            token_ids=list(range(128)),
            original_positions=list(range(9900, 10028)),
            text="far away block",
            representative_embedding=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            timestamp=0,
        )
        store.store(similar)
        store.store(dissimilar)

        query = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        results = store.retrieve_by_similarity(query, threshold=0.8, max_results=5)

        assert len(results) == 1
        assert results[0].block_id == 0

    def test_retrieve_increments_access_count(self):
        store = ArchiveStore(EvokeConfig())
        block = _make_archive_block(
            0, embedding=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        store.store(block)

        query = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        store.retrieve_by_similarity(query, threshold=0.5, max_results=5)
        assert block.access_count == 1

    def test_capacity_eviction(self):
        config = EvokeConfig(max_archive_blocks=3)
        store = ArchiveStore(config)

        for i in range(5):
            store.store(_make_archive_block(i))

        assert store.size == 3
        assert store.get(0) is None
        assert store.get(1) is None
        assert store.get(2) is not None

    def test_total_tokens(self):
        store = ArchiveStore(EvokeConfig())
        store.store(_make_archive_block(0))
        store.store(_make_archive_block(1))
        assert store.total_tokens == 256

    def test_neighbor_expansion_disabled_by_default(self):
        store = ArchiveStore(EvokeConfig())
        b0 = _make_archive_block(0)
        b1 = _make_archive_block(1)
        b2 = _make_archive_block(2)
        b1.text = "needle content here"
        store.store(b0)
        store.store(b1)
        store.store(b2)

        query = np.zeros(8, dtype=np.float32)
        results = store.retrieve_by_similarity(
            query, threshold=0.5, max_results=4, query_text="needle content"
        )

        result_ids = {b.block_id for b in results}
        assert 1 in result_ids
        assert 0 not in result_ids
        assert 2 not in result_ids

    def test_neighbor_expansion_when_enabled(self):
        store = ArchiveStore(EvokeConfig(expand_neighbors=True))
        b0 = _make_archive_block(0)
        b1 = _make_archive_block(1)
        b2 = _make_archive_block(2)
        b1.text = "needle content here"
        store.store(b0)
        store.store(b1)
        store.store(b2)

        query = np.zeros(8, dtype=np.float32)
        results = store.retrieve_by_similarity(
            query, threshold=0.5, max_results=4, query_text="needle content"
        )

        result_ids = {b.block_id for b in results}
        assert 1 in result_ids
        assert 0 in result_ids
        assert 2 in result_ids

    def test_results_sorted_by_position(self):
        store = ArchiveStore(EvokeConfig())
        b2 = _make_archive_block(2)
        b0 = _make_archive_block(0)
        b1 = _make_archive_block(1)
        b1.text = "needle"
        store.store(b2)
        store.store(b0)
        store.store(b1)

        query = np.zeros(8, dtype=np.float32)
        results = store.retrieve_by_similarity(
            query, threshold=0.5, max_results=4, query_text="needle"
        )

        positions = [b.pos_start for b in results]
        assert positions == sorted(positions)

    def test_min_lexical_recall_filters_weak_matches(self):
        store = ArchiveStore(EvokeConfig())
        weak_match = ArchiveBlock(
            block_id=0,
            token_ids=list(range(128)),
            original_positions=list(range(128)),
            text="The weather today is partly cloudy with rain in the afternoon",
            representative_embedding=np.zeros(8, dtype=np.float32),
            timestamp=0,
        )
        strong_match = ArchiveBlock(
            block_id=1,
            token_ids=list(range(128, 256)),
            original_positions=list(range(128, 256)),
            text="The secret project codename is AURORA-SEVEN budget $4.2 million",
            representative_embedding=np.zeros(8, dtype=np.float32),
            timestamp=0,
        )
        store.store(weak_match)
        store.store(strong_match)

        query = np.zeros(8, dtype=np.float32)

        results_no_threshold = store.retrieve_by_similarity(
            query,
            threshold=0.5,
            max_results=4,
            query_text="What is the weather forecast for tomorrow?",
            min_lexical_recall=0.0,
        )
        assert any(b.block_id == 0 for b in results_no_threshold)

        results_with_threshold = store.retrieve_by_similarity(
            query,
            threshold=0.5,
            max_results=4,
            query_text="What is the weather forecast for tomorrow?",
            min_lexical_recall=0.4,
        )
        assert not any(b.block_id == 0 for b in results_with_threshold)

    def test_min_lexical_recall_keeps_strong_matches(self):
        store = ArchiveStore(EvokeConfig())
        block = ArchiveBlock(
            block_id=0,
            token_ids=list(range(128)),
            original_positions=list(range(128)),
            text="The secret project codename is AURORA-SEVEN budget million",
            representative_embedding=np.zeros(8, dtype=np.float32),
            timestamp=0,
        )
        store.store(block)

        query = np.zeros(8, dtype=np.float32)
        results = store.retrieve_by_similarity(
            query,
            threshold=0.5,
            max_results=4,
            query_text="What is the codename of the secret project and its budget?",
            min_lexical_recall=0.4,
        )
        assert any(b.block_id == 0 for b in results)


def _word_tokenize(text: str) -> list[int]:
    return [hash(w) % 100000 for w in text.lower().split()]


class TestBPETokenRetrieval:
    def test_bpe_matching_finds_shared_tokens(self):
        store = ArchiveStore(EvokeConfig(), tokenize_fn=_word_tokenize)

        needle_text = "The secret password is CRYSTALLINE"
        needle_tokens = _word_tokenize(needle_text)
        needle = ArchiveBlock(
            block_id=0,
            token_ids=needle_tokens,
            original_positions=list(range(len(needle_tokens))),
            text=needle_text,
            representative_embedding=np.zeros(8, dtype=np.float32),
            timestamp=0,
        )
        filler_text = "The weather today is partly cloudy"
        filler_tokens = _word_tokenize(filler_text)
        filler = ArchiveBlock(
            block_id=1,
            token_ids=filler_tokens,
            original_positions=list(range(100, 100 + len(filler_tokens))),
            text=filler_text,
            representative_embedding=np.zeros(8, dtype=np.float32),
            timestamp=0,
        )
        store.store(needle)
        store.store(filler)

        query = np.zeros(8, dtype=np.float32)
        results = store.retrieve_by_similarity(
            query,
            threshold=0.5,
            max_results=4,
            query_text="What is the secret password?",
        )
        assert any(b.block_id == 0 for b in results)

    def test_idf_weights_rare_tokens_higher(self):
        store = ArchiveStore(EvokeConfig(), tokenize_fn=_word_tokenize)

        for i in range(10):
            text = f"the block number {i} has common words"
            tokens = _word_tokenize(text)
            block = ArchiveBlock(
                block_id=i,
                token_ids=tokens,
                original_positions=list(range(i * 100, i * 100 + len(tokens))),
                text=text,
                representative_embedding=np.zeros(8, dtype=np.float32),
                timestamp=0,
            )
            store.store(block)

        rare_text = "the AURORA project details"
        rare_tokens_list = _word_tokenize(rare_text)
        rare_block = ArchiveBlock(
            block_id=99,
            token_ids=rare_tokens_list,
            original_positions=list(range(9900, 9900 + len(rare_tokens_list))),
            text=rare_text,
            representative_embedding=np.zeros(8, dtype=np.float32),
            timestamp=0,
        )
        store.store(rare_block)

        rare_only = set(_word_tokenize("aurora")) - set(_word_tokenize("the"))
        common_only = set(_word_tokenize("the"))
        for t in rare_only:
            idf_rare = store._idf(t)
            for ct in common_only:
                idf_common = store._idf(ct)
                assert idf_rare > idf_common

    def test_idf_updated_on_remove(self):
        store = ArchiveStore(EvokeConfig(), tokenize_fn=_word_tokenize)

        text = "unique content here"
        tokens = _word_tokenize(text)
        block = ArchiveBlock(
            block_id=0,
            token_ids=tokens,
            original_positions=list(range(len(tokens))),
            text=text,
            representative_embedding=np.zeros(8, dtype=np.float32),
            timestamp=0,
        )
        store.store(block)
        assert store.size == 1

        token_set = set(tokens)
        for t in token_set:
            assert store._doc_freq.get(t, 0) == 1

        store.remove(0)
        for t in token_set:
            assert store._doc_freq.get(t, 0) == 0

    def test_capacity_eviction_cleans_idf(self):
        config = EvokeConfig(max_archive_blocks=2)
        store = ArchiveStore(config, tokenize_fn=_word_tokenize)

        for i in range(3):
            text = f"block{i}"
            tokens = _word_tokenize(text)
            block = ArchiveBlock(
                block_id=i,
                token_ids=tokens,
                original_positions=list(range(i * 100, i * 100 + len(tokens))),
                text=text,
                representative_embedding=np.zeros(8, dtype=np.float32),
                timestamp=0,
            )
            store.store(block)

        assert store.size == 2
        assert 0 not in store._block_token_sets
