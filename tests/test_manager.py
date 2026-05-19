from evoke.config import EvokeConfig
from evoke.manager import EvokeManager
from evoke.mock_engine import MockEngine
from evoke.types import BlockSource


def _make_long_text(n_chars: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. "
    repeats = (n_chars // len(base)) + 1
    return (base * repeats)[:n_chars]


class TestEvokeManagerBasics:
    def test_load_document_creates_blocks(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=2048, block_size=128)
        manager = EvokeManager(engine, config)

        text = _make_long_text(1024)
        manager.load_document(text)

        stats = manager.get_stats()
        assert stats.active_blocks > 0
        assert stats.active_tokens > 0

    def test_budget_enforced_on_load(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=256,
            block_size=128,
            high_watermark=0.95,
            low_watermark=0.75,
        )
        manager = EvokeManager(engine, config)

        text = _make_long_text(1024)
        manager.load_document(text)

        stats = manager.get_stats()
        assert stats.active_tokens <= config.max_active_tokens
        assert stats.archive_blocks > 0

    def test_demotion_preserves_sink_blocks(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=256,
            block_size=128,
            sink_count=4,
            high_watermark=0.95,
            low_watermark=0.75,
        )
        manager = EvokeManager(engine, config)

        text = _make_long_text(1024)
        manager.load_document(text)

        active = manager._positions.active_blocks
        has_sink = any(b.original_start == 0 for b in active)
        assert has_sink

    def test_get_stats_reports_correctly(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128)
        manager = EvokeManager(engine, config)

        text = _make_long_text(512)
        manager.load_document(text)

        stats = manager.get_stats()
        assert stats.budget == 10000
        assert stats.total_demotions == 0
        assert stats.total_promotions == 0

    def test_force_demote(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128)
        manager = EvokeManager(engine, config)

        text = _make_long_text(512)
        manager.load_document(text)

        initial_active = manager.get_stats().active_blocks
        blocks = manager._positions.active_blocks

        non_sink = [b for b in blocks if b.original_start >= config.sink_count]
        if non_sink:
            manager.force_demote([non_sink[0].block_id])
            assert manager.get_stats().active_blocks == initial_active - 1
            assert manager.get_stats().archive_blocks == 1

    def test_force_promote(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128)
        manager = EvokeManager(engine, config)

        text = _make_long_text(512)
        manager.load_document(text)

        blocks = manager._positions.active_blocks
        non_sink = [b for b in blocks if b.original_start >= config.sink_count]
        if non_sink:
            bid = non_sink[0].block_id
            manager.force_demote([bid])
            assert manager.get_stats().archive_blocks == 1

            manager.force_promote([bid])
            assert manager.get_stats().archive_blocks == 0
            assert manager.get_stats().total_promotions == 1

    def test_event_log_tracks_operations(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=256,
            block_size=128,
            high_watermark=0.95,
            low_watermark=0.75,
        )
        manager = EvokeManager(engine, config)

        text = _make_long_text(1024)
        manager.load_document(text)

        events = manager.get_event_log()
        demotion_events = [e for e in events if e.event_type == "demotion"]
        assert len(demotion_events) > 0


class TestEvokeManagerDemotionPolicy:
    def test_watermark_policy(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=512,
            block_size=128,
            demotion_policy="watermark",
            high_watermark=0.9,
            low_watermark=0.5,
        )
        manager = EvokeManager(engine, config)

        text = _make_long_text(1024)
        manager.load_document(text)

        stats = manager.get_stats()
        target = int(config.max_active_tokens * config.low_watermark)
        assert stats.active_tokens <= target + config.block_size

    def test_hard_budget_policy(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=512,
            block_size=128,
            demotion_policy="hard",
        )
        manager = EvokeManager(engine, config)

        text = _make_long_text(1024)
        manager.load_document(text)

        stats = manager.get_stats()
        assert stats.active_tokens <= config.max_active_tokens


class TestEvokeManagerRetrieval:
    def test_lexical_retrieval_finds_needle(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=256,
            block_size=64,
            retrieval_threshold=0.5,
            max_retrieve_blocks=4,
            high_watermark=0.95,
            low_watermark=0.75,
        )
        manager = EvokeManager(engine, config)

        filler = "The weather today is partly cloudy with rain. " * 40
        needle = "The secret password for the vault is CRYSTALLINE."
        words = filler.split()
        words.insert(len(words) // 2, needle)
        doc = " ".join(words)
        manager.load_document(doc)

        stats_before = manager.get_stats()
        assert stats_before.archive_blocks > 0

        needle_in_archive = any(
            "secret" in b.text for b in manager._archive.all_blocks()
        )
        assert needle_in_archive

        archive_before = {b.block_id: b.text for b in manager._archive.all_blocks()}
        needle_block_ids = {
            bid
            for bid, text in archive_before.items()
            if "secret" in text or "vault" in text
        }
        assert needle_block_ids

        manager.process_user_message("What is the secret password for the vault?")

        stats_after = manager.get_stats()
        assert stats_after.total_promotions > 0

        promoted_ids = set()
        for event in manager.get_event_log():
            if event.event_type == "promotion":
                promoted_ids.update(event.block_ids)

        assert needle_block_ids & promoted_ids

    def test_high_threshold_disables_retrieval(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=256,
            block_size=64,
            retrieval_threshold=2.0,
            max_retrieve_blocks=4,
            high_watermark=0.95,
            low_watermark=0.75,
        )
        manager = EvokeManager(engine, config)

        filler = "The weather today is partly cloudy with rain. " * 40
        needle = "The secret password for the vault is CRYSTALLINE."
        words = filler.split()
        words.insert(len(words) // 2, needle)
        doc = " ".join(words)
        manager.load_document(doc)

        assert manager.get_stats().archive_blocks > 0

        manager.process_user_message("What is the secret password for the vault?")

        assert manager.get_stats().total_promotions == 0


class TestEvokeManagerRebuild:
    def test_rebuild_triggers_when_position_space_exhausted(self):
        engine = MockEngine(n_ctx=1024)
        config = EvokeConfig(
            max_active_tokens=10000,
            block_size=64,
        )
        manager = EvokeManager(engine, config)

        text = _make_long_text(256)
        manager.load_document(text)

        engine._next_write_pos = 950

        blocks = manager._positions.active_blocks
        if blocks:
            bid = blocks[-1].block_id
            manager.force_demote([bid])
            manager.force_promote([bid])

        rebuild_events = [
            e for e in manager.get_event_log() if e.event_type == "rebuild"
        ]
        assert len(rebuild_events) > 0

    def test_rebuild_compacts_positions(self):
        engine = MockEngine(n_ctx=512)
        config = EvokeConfig(
            max_active_tokens=10000,
            block_size=64,
        )
        manager = EvokeManager(engine, config)

        text = _make_long_text(128)
        manager.load_document(text)

        engine._next_write_pos = 470

        blocks = manager._positions.active_blocks
        if blocks:
            bid = blocks[-1].block_id
            manager.force_demote([bid])
            manager.force_promote([bid])

        max_pos = max(b.logical_end for b in manager._positions.active_blocks)
        assert max_pos < 512

    def test_rebuild_on_user_message_reclaims_generation_space(self):
        engine = MockEngine(n_ctx=1024)
        config = EvokeConfig(
            max_active_tokens=10000,
            block_size=64,
        )
        manager = EvokeManager(engine, config)

        text = _make_long_text(256)
        manager.load_document(text)

        engine._next_write_pos = 950

        manager.process_user_message("Hello, this is a new turn.")

        rebuild_events = [
            e for e in manager.get_event_log() if e.event_type == "rebuild"
        ]
        assert len(rebuild_events) > 0
        assert engine._next_write_pos < 950


class TestEvokeManagerMultiTurn:
    def test_user_message_tracked_as_block(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=64)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))
        blocks_before = len(manager._positions.active_blocks)

        manager.process_user_message("What is the meaning of life?")
        blocks_after = len(manager._positions.active_blocks)
        assert blocks_after > blocks_before

    def test_generation_not_tracked(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=64)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))
        manager.process_user_message("Hello")
        blocks_before = len(manager._positions.active_blocks)

        manager.generate(32)
        blocks_after = len(manager._positions.active_blocks)
        assert blocks_after == blocks_before

    def test_old_turns_evictable(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=512,
            block_size=64,
            high_watermark=0.95,
            low_watermark=0.75,
            pin_generated=True,
        )
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))

        manager.process_user_message("First question about weather?")
        manager.generate(32)

        manager.process_user_message("Second question about science?")
        manager.generate(32)

        stats = manager.get_stats()
        assert stats.active_tokens <= config.max_active_tokens

    def test_conversation_recall(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=400,
            block_size=64,
            retrieval_threshold=0.5,
            max_retrieve_blocks=4,
            high_watermark=0.95,
            low_watermark=0.75,
        )
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(128))

        manager.process_user_message("The secret code for project alpha is ZEBRA-NINE.")
        manager.generate(16)

        manager.process_user_message("Tell me about the weather today.")
        manager.generate(16)

        manager.process_user_message("Tell me about something else entirely.")
        manager.generate(16)

        archive_texts = [b.text for b in manager._archive.all_blocks()]
        has_secret = any("secret" in t or "ZEBRA" in t for t in archive_texts)

        if has_secret:
            promotions_before = manager.get_stats().total_promotions
            manager.process_user_message("What was the secret code for project alpha?")
            promotions_after = manager.get_stats().total_promotions
            assert promotions_after > promotions_before


class TestEvokeManagerRebuildPromotion:
    def test_promote_triggers_rebuild(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128)
        manager = EvokeManager(engine, config)

        text = _make_long_text(512)
        manager.load_document(text)

        blocks = manager._positions.active_blocks
        non_sink = [b for b in blocks if b.original_start >= config.sink_count]
        assert non_sink

        manager.force_demote([non_sink[0].block_id])
        assert manager.get_stats().archive_blocks == 1

        manager.force_promote([non_sink[0].block_id])
        assert manager.get_stats().archive_blocks == 0

        rebuild_events = [
            e for e in manager.get_event_log() if e.event_type == "rebuild"
        ]
        assert len(rebuild_events) > 0

    def test_promote_compacts_positions(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128)
        manager = EvokeManager(engine, config)

        text = _make_long_text(512)
        manager.load_document(text)

        blocks = manager._positions.active_blocks
        non_sink = [b for b in blocks if b.original_start >= config.sink_count]
        assert non_sink

        manager.force_demote([non_sink[0].block_id])
        manager.force_promote([non_sink[0].block_id])

        max_pos = max(b.logical_end for b in manager._positions.active_blocks)
        total_tokens = manager._positions.active_token_count
        assert max_pos == total_tokens

    def test_promote_preserves_block_order(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128)
        manager = EvokeManager(engine, config)

        text = _make_long_text(512)
        manager.load_document(text)

        blocks = manager._positions.active_blocks
        original_order = [b.original_start for b in blocks]

        non_sink = [b for b in blocks if b.original_start >= config.sink_count]
        assert non_sink
        manager.force_demote([non_sink[0].block_id])
        manager.force_promote([non_sink[0].block_id])

        new_order = [b.original_start for b in manager._positions.active_blocks]
        assert new_order == sorted(new_order)
        assert set(new_order) == set(original_order)


class TestBlockSourceClassification:
    def test_document_blocks_tagged_as_document(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(512))

        for block in manager._positions.active_blocks:
            assert block.source == BlockSource.DOCUMENT

    def test_conversation_block_tagged_as_user(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=64)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))
        doc_block_count = len(manager._positions.active_blocks)

        manager.process_user_message("What is the meaning of life?")

        blocks = manager._positions.active_blocks
        new_blocks = blocks[doc_block_count:]
        assert len(new_blocks) > 0
        for block in new_blocks:
            assert block.source == BlockSource.USER

    def test_source_preserved_through_demotion(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=64)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))
        manager.process_user_message("The secret code is ZEBRA-NINE.")

        user_blocks = [
            b for b in manager._positions.active_blocks if b.source == BlockSource.USER
        ]
        assert user_blocks
        bid = user_blocks[0].block_id

        manager.force_demote([bid])
        archived = manager._archive.get(bid)
        assert archived is not None
        assert archived.source == BlockSource.USER

    def test_source_preserved_through_promotion(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=64)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))
        manager.process_user_message("The secret code is ZEBRA-NINE.")

        user_blocks = [
            b for b in manager._positions.active_blocks if b.source == BlockSource.USER
        ]
        assert user_blocks
        bid = user_blocks[0].block_id

        manager.force_demote([bid])
        manager.force_promote([bid])

        promoted = [b for b in manager._positions.active_blocks if b.block_id == bid]
        assert promoted
        assert promoted[0].source == BlockSource.USER
        assert promoted[0].promotion_step >= 0


class TestEvokeManagerPinning:
    def test_generated_tokens_are_pinned(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=10000,
            block_size=128,
            pin_generated=True,
        )
        manager = EvokeManager(engine, config)

        text = _make_long_text(512)
        manager.load_document(text)

        scores = manager.get_relevance_scores()
        assert len(scores) > 0


class TestEvokeManagerThinkingGeneration:
    def test_thinking_generation_stops_after_answer_budget(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))

        think_text = "<think>reasoning here</think>the answer"
        engine.queue_tokens([ord(c) for c in think_text])

        raw = manager.generate(
            0,
            think_close="</think>",
            thinking_budget=1000,
            answer_budget=20,
        )
        assert "</think>" in raw
        assert "the answer" in raw

    def test_thinking_generation_respects_thinking_budget(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))

        raw = manager.generate(
            0,
            think_close="</think>",
            thinking_budget=50,
            answer_budget=10,
        )
        tokens = engine.tokenize(raw)
        assert len(tokens) <= 60

    def test_thinking_generation_caps_answer_after_think_close(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))

        think_part = "<think>ok</think>"
        answer_part = "A" * 100
        engine.queue_tokens([ord(c) for c in think_part + answer_part])

        raw = manager.generate(
            0,
            think_close="</think>",
            thinking_budget=1000,
            answer_budget=10,
        )
        after_think = raw.split("</think>", 1)[1]
        assert len(after_think) <= 10
