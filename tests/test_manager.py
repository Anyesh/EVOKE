from evoke.config import EvokeConfig
from evoke.manager import EvokeManager
from evoke.mock_engine import MockEngine
from evoke.types import BlockSource


def _make_long_text(n_chars: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. "
    repeats = (n_chars // len(base)) + 1
    return (base * repeats)[:n_chars]


class TestEngineEviction:
    def test_evict_single_range(self):
        engine = MockEngine()
        engine.process_tokens(list(range(10)))
        engine.evict_ranges([(2, 5)])
        assert engine.next_write_pos == 7
        assert engine.get_kv_cache_token_count() == 7

    def test_evict_scattered_ranges(self):
        engine = MockEngine()
        engine.process_tokens(list(range(10)))
        engine.evict_ranges([(2, 4), (6, 8)])
        assert engine.next_write_pos == 6
        assert engine.get_kv_cache_token_count() == 6

    def test_evict_empty_is_noop(self):
        engine = MockEngine()
        engine.process_tokens(list(range(10)))
        engine.evict_ranges([])
        assert engine.next_write_pos == 10


class TestLoad:
    def test_load_document_creates_blocks(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=2048, block_size=128)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(1024))

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

        manager.load_document(_make_long_text(1024))

        stats = manager.get_stats()
        assert stats.active_tokens <= config.max_active_tokens
        assert stats.total_evictions > 0

    def test_get_stats(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(512))

        stats = manager.get_stats()
        assert stats.budget == 10000
        assert stats.total_evictions == 0
        assert stats.active_tokens > 0


class TestAddContext:
    def test_creates_keyed_document_blocks(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=64)
        manager = EvokeManager(engine, config)
        manager.load_document(_make_long_text(128))
        before = len(manager._positions.active_blocks)

        manager.add_context(_make_long_text(192), key="file:utils.py")

        new_blocks = manager._positions.active_blocks[before:]
        assert len(new_blocks) == 3
        assert all(b.source == BlockSource.DOCUMENT for b in new_blocks)
        assert [b.key for b in new_blocks] == [
            "file:utils.py#0",
            "file:utils.py#1",
            "file:utils.py#2",
        ]

    def test_old_context_evictable_under_budget(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=512,
            block_size=64,
            high_watermark=0.95,
            low_watermark=0.75,
        )
        manager = EvokeManager(engine, config)
        manager.load_document(_make_long_text(128))
        manager.add_context(_make_long_text(256), key="file:a")
        manager.add_context(_make_long_text(256), key="file:b")

        assert manager.get_stats().active_tokens <= 512

    def test_first_added_context_block_is_sink(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=64, sink_count=4)
        manager = EvokeManager(engine, config)

        manager.add_context(_make_long_text(192), key="file:first.py")

        blocks = sorted(manager._positions.active_blocks, key=lambda b: b.block_id)
        assert blocks[0].is_sink
        assert not blocks[1].is_sink


class TestEviction:
    def test_eviction_reclaims_budget(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=512,
            block_size=64,
            high_watermark=0.9,
            low_watermark=0.5,
        )
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(2048))

        stats = manager.get_stats()
        target = int(512 * 0.5)
        assert stats.active_tokens <= target + 64

    def test_force_evict_removes_block(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(512))
        blocks = manager._positions.active_blocks
        non_sink = [b for b in blocks if not b.is_sink]
        assert non_sink

        before = len(blocks)
        manager.force_evict([non_sink[0].block_id])
        assert len(manager._positions.active_blocks) == before - 1

    def test_eviction_compacts_positions(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(512))
        non_sink = [b for b in manager._positions.active_blocks if not b.is_sink]
        manager.force_evict([non_sink[0].block_id])

        pos = 0
        for block in manager._positions.active_blocks:
            assert block.logical_start == pos
            pos = block.logical_end
        assert engine.next_write_pos == pos

    def test_evict_scattered_preserves_order(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=64)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(512))
        blocks = manager._positions.active_blocks
        ids_before = [b.block_id for b in blocks]
        non_sink = [b for b in blocks if not b.is_sink]

        evict_ids = {non_sink[1].block_id, non_sink[3].block_id}
        manager.force_evict(list(evict_ids))

        remaining = [b.block_id for b in manager._positions.active_blocks]
        assert remaining == sorted(remaining)
        assert set(remaining) == set(ids_before) - evict_ids

    def test_evict_then_generate(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=512,
            block_size=64,
            high_watermark=0.9,
            low_watermark=0.5,
        )
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(2048))
        out = manager.generate(16)
        assert isinstance(out, str)
        assert manager.get_stats().active_tokens <= config.max_active_tokens

    def test_eviction_events_logged(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=256,
            block_size=128,
            high_watermark=0.95,
            low_watermark=0.75,
        )
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(1024))
        events = manager.get_event_log()
        assert any(e.event_type == "eviction" for e in events)


class TestEvictionPolicy:
    def test_watermark_policy(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=512,
            block_size=128,
            eviction_policy="watermark",
            high_watermark=0.9,
            low_watermark=0.5,
        )
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(1024))

        stats = manager.get_stats()
        target = int(config.max_active_tokens * config.low_watermark)
        assert stats.active_tokens <= target + config.block_size

    def test_hard_policy(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=512,
            block_size=128,
            eviction_policy="hard",
        )
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(1024))
        assert manager.get_stats().active_tokens <= config.max_active_tokens


class TestMultiTurn:
    def test_user_message_tracked_as_user(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=64)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))
        doc_count = len(manager._positions.active_blocks)

        manager.process_user_message("What is the meaning of life?")

        new_blocks = manager._positions.active_blocks[doc_count:]
        assert new_blocks
        assert all(b.source == BlockSource.USER for b in new_blocks)

    def test_generation_tracked_as_assistant(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=64)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))
        manager.process_user_message("Hello")
        before = len(manager._positions.active_blocks)

        manager.generate(16)

        after = manager._positions.active_blocks
        assert len(after) == before + 1
        assert after[-1].source == BlockSource.ASSISTANT

    def test_old_turns_evictable(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=512,
            block_size=64,
            high_watermark=0.95,
            low_watermark=0.75,
        )
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))
        manager.process_user_message("First question about weather?")
        manager.generate(16)
        manager.process_user_message("Second question about science?")
        manager.generate(16)

        assert manager.get_stats().active_tokens <= config.max_active_tokens


class TestBlockSource:
    def test_document_blocks_tagged_document(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(512))
        for block in manager._positions.active_blocks:
            assert block.source == BlockSource.DOCUMENT

    def test_first_block_is_sink(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128, sink_count=4)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(512))
        blocks = sorted(manager._positions.active_blocks, key=lambda b: b.block_id)
        assert blocks[0].is_sink
        assert not blocks[1].is_sink


class TestPinning:
    def test_current_turn_user_block_survives_enforcement(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=300,
            block_size=64,
            pin_generated=True,
            high_watermark=0.95,
            low_watermark=0.75,
        )
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))
        manager.process_user_message("A distinctive user question goes here now")

        user_blocks = [
            b for b in manager._positions.active_blocks if b.source == BlockSource.USER
        ]
        assert user_blocks


class TestThinkingGeneration:
    def test_thinking_answer_tracked_without_reasoning(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=64)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))
        text = "<think>" + "z" * 200 + "</think>FINALANSWER"
        engine.queue_tokens([ord(c) for c in text])

        manager.generate(
            0, think_close="</think>", thinking_budget=1000, answer_budget=50
        )

        assistant = [
            b
            for b in manager._positions.active_blocks
            if b.source == BlockSource.ASSISTANT
        ]
        assert len(assistant) == 1
        answer_text = engine.detokenize(assistant[0].token_ids)
        assert "FINALANSWER" in answer_text
        assert "z" not in answer_text

    def test_thinking_trace_does_not_blow_budget(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=512,
            block_size=64,
            high_watermark=0.95,
            low_watermark=0.75,
        )
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))
        text = "<think>" + "z" * 5000 + "</think>answer"
        engine.queue_tokens([ord(c) for c in text])

        manager.generate(
            0, think_close="</think>", thinking_budget=10000, answer_budget=50
        )

        assert manager.get_stats().active_tokens <= config.max_active_tokens

    def test_thinking_stops_after_answer_budget(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=10000, block_size=128)
        manager = EvokeManager(engine, config)

        manager.load_document(_make_long_text(256))
        text = "<think>reasoning</think>" + "A" * 100
        engine.queue_tokens([ord(c) for c in text])

        raw = manager.generate(
            0, think_close="</think>", thinking_budget=1000, answer_budget=10
        )
        after_think = raw.split("</think>", 1)[1]
        assert len(after_think) <= 10


class TestRecovery:
    def test_discard_mode_keeps_no_breadcrumbs(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=256,
            block_size=128,
            recovery_mode="discard",
            high_watermark=0.95,
            low_watermark=0.75,
        )
        manager = EvokeManager(engine, config)
        manager.load_document(_make_long_text(1024))

        assert manager.get_stats().total_evictions > 0
        assert manager.get_breadcrumbs() == []

    def test_breadcrumb_mode_records_evicted(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=256,
            block_size=128,
            recovery_mode="breadcrumb",
            high_watermark=0.95,
            low_watermark=0.75,
        )
        manager = EvokeManager(engine, config)
        manager.load_document(_make_long_text(1024))

        crumbs = manager.get_breadcrumbs()
        assert len(crumbs) > 0
        assert all(c.token_count > 0 for c in crumbs)
        assert all(c.key for c in crumbs)


class TestKVRestore:
    def test_recover_brings_block_back(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=10000, block_size=128, recovery_mode="kv_restore"
        )
        manager = EvokeManager(engine, config)
        manager.load_document(_make_long_text(512))

        non_sink = [b for b in manager._positions.active_blocks if not b.is_sink]
        target = non_sink[0]
        key = target.key
        tokens = list(target.token_ids)

        manager.force_evict([target.block_id])
        assert key not in [b.key for b in manager._positions.active_blocks]

        assert manager.recover(key) is True
        recovered = [b for b in manager._positions.active_blocks if b.key == key]
        assert len(recovered) == 1
        assert recovered[0].token_ids == tokens
        assert recovered[0].source == BlockSource.DOCUMENT
        assert manager.get_stats().total_recoveries == 1

    def test_recover_unknown_key_returns_false(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=10000, block_size=128, recovery_mode="kv_restore"
        )
        manager = EvokeManager(engine, config)
        manager.load_document(_make_long_text(256))
        assert manager.recover("no-such-key") is False

    def test_recover_unavailable_in_discard_mode(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=256,
            block_size=128,
            recovery_mode="discard",
            high_watermark=0.95,
            low_watermark=0.75,
        )
        manager = EvokeManager(engine, config)
        manager.load_document(_make_long_text(1024))
        assert manager.recover("doc#1") is False


class TestHarnessPriority:
    def test_pinned_block_never_evicted_under_pressure(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=256,
            block_size=64,
            high_watermark=0.95,
            low_watermark=0.50,
            pin_generated=False,
        )
        manager = EvokeManager(engine, config)
        # Pinned block first, then enough content to push past budget.
        manager.add_context(_make_long_text(128), "pinned_file", pinned=True)
        manager.add_context(_make_long_text(512), "filler")
        keys = [b.key for b in manager._positions.active_blocks]
        assert any(k.startswith("pinned_file") for k in keys), (
            "pinned blocks must survive eviction pressure"
        )

    def test_priority_threads_through_add_context(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=1024, block_size=64)
        manager = EvokeManager(engine, config)
        manager.add_context("hello", "h", priority=2.5, pinned=False)
        blocks = manager._positions.active_blocks
        assert blocks
        assert blocks[0].priority == 2.5
        assert blocks[0].pinned is False

    def test_priority_threads_through_load_document(self):
        engine = MockEngine()
        config = EvokeConfig(max_active_tokens=1024, block_size=64)
        manager = EvokeManager(engine, config)
        manager.load_document(_make_long_text(128), priority=3.0, pinned=True)
        blocks = manager._positions.active_blocks
        assert blocks
        assert all(b.priority == 3.0 for b in blocks)
        assert all(b.pinned for b in blocks)

    def test_pinned_excluded_from_evictable_candidates(self):
        engine = MockEngine()
        config = EvokeConfig(
            max_active_tokens=256,
            block_size=64,
            high_watermark=0.5,
            low_watermark=0.2,
            pin_generated=False,
        )
        manager = EvokeManager(engine, config)
        manager.add_context(_make_long_text(128), "a", pinned=True)
        manager.add_context(_make_long_text(128), "b", pinned=False)
        scores = manager._scorer.score_blocks(
            manager._positions.active_blocks,
            manager._positions.next_logical_pos,
            manager._positions.next_logical_pos,
        )
        evictable = manager._evictable_blocks(
            manager._positions.active_blocks,
            scores,
            manager._positions.next_logical_pos,
        )
        assert all(not b.pinned for b in evictable), (
            "pinned blocks must never appear as eviction candidates"
        )
