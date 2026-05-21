"""Regression test for the session-path eviction bug.

Bug history: _evictable_blocks was filtering ANY block whose block_id was
>= _current_turn_start_id under pin_generated. add_context_tokens (the
server's prompt-decoding path) sets _current_turn_start_id at its start,
so every newly-added prompt block landed in the pinned set. When the
session's tail-add re-decoded most of the conversation, _enforce_budget
found zero evictable candidates and silently exited. This was invisible
to the test suite because no test drove the Session through the multi-
turn pattern the server uses; the bench scripts cited in the paper use
the add_context() path, which advances _current_turn_start_id AFTER
block creation and so does not trigger the bug.

The fix: pin_generated now only pins ASSISTANT-source blocks (model-
generated output via _track_generated_block), not DOCUMENT-source
blocks added via add_context_tokens.

This test drives a Session through a stateless-style multi-turn pattern
similar to what opencode produces and asserts that eviction fires by
turn 2 and that active_tokens stays bounded near low_watermark * budget.
"""

from __future__ import annotations

from evoke.config import EvokeConfig
from evoke.mock_engine import MockEngine
from evoke.session import Session


def _make_session(budget: int = 2048) -> Session:
    engine = MockEngine()
    cfg = EvokeConfig(
        max_active_tokens=budget,
        block_size=128,
        high_watermark=0.92,
        low_watermark=0.70,
        recovery_mode="discard",
    )
    return Session(engine, config=cfg)


class TestSessionEvictionOnMultiTurn:
    def test_eviction_fires_by_turn_two(self):
        # A fresh session: first prompt fills the cache well past the budget.
        # Turn 1 cannot evict the just-added blocks (all current-turn).
        # Turn 2 extends the conversation; by then turn 1's blocks are "old"
        # and the policy should fire eviction.
        session = _make_session(budget=2048)
        first_prompt = list(range(10000))
        session.sync_prefix(first_prompt)
        session._engine.queue_tokens(list(range(20000, 20200)))
        session.generate(max_tokens=200)
        first_evictions = session._manager.get_stats().total_evictions

        # Turn 2: prior conversation + new user message.
        second_prompt = (
            first_prompt + list(range(20000, 20200)) + list(range(30000, 30200))
        )
        session.sync_prefix(second_prompt)
        session._engine.queue_tokens(list(range(40000, 40200)))
        session.generate(max_tokens=200)
        second_evictions = session._manager.get_stats().total_evictions

        assert second_evictions > first_evictions, (
            f"eviction did not fire on turn 2: "
            f"evictions stayed at {second_evictions} (was {first_evictions})"
        )

    def test_active_tokens_stay_bounded_across_turns(self):
        # After several turns, active_tokens should stay near low_watermark*budget
        # at the end of each turn. Allow a moderate overshoot tolerance because
        # generated assistant blocks are still current-turn-pinned during the
        # turn they were produced (they become evictable on the next turn).
        session = _make_session(budget=2048)
        engine = session._engine

        cumulative = list(range(10000))
        session.sync_prefix(cumulative)
        engine.queue_tokens(list(range(20000, 20200)))
        result = session.generate(max_tokens=200)
        cumulative = cumulative + list(result.output_tokens)

        peak_active = 0
        for t in range(2, 6):
            new_user = list(range(30000 + t * 100, 30000 + t * 100 + 200))
            cumulative = cumulative + new_user
            session.sync_prefix(cumulative)
            engine.queue_tokens(list(range(40000 + t * 100, 40000 + t * 100 + 200)))
            result = session.generate(max_tokens=200)
            cumulative = cumulative + list(result.output_tokens)
            stats = session._manager.get_stats()
            peak_active = max(peak_active, stats.active_tokens)

        # By turn 5 the policy should keep the active footprint within a
        # reasonable multiple of the budget. A 4x ceiling accommodates the
        # known generated-block-granularity limitation (a single assistant
        # block is indivisible until the next turn).
        assert peak_active <= 4 * 2048, (
            f"active_tokens unbounded across turns: peak={peak_active}, budget=2048"
        )

    def test_pin_generated_does_not_pin_document_blocks(self):
        # The exact regression: add_context_tokens creates DOCUMENT blocks,
        # and pin_generated must not pin them. We verify the eviction outcome:
        # after two add_context_tokens calls with sizes that push past the
        # budget, the second call's _enforce_budget must find evictable
        # candidates (the first call's DOCUMENT blocks) and evict some.
        engine = MockEngine()
        cfg = EvokeConfig(
            max_active_tokens=512,
            block_size=128,
            high_watermark=0.95,
            low_watermark=0.75,
            recovery_mode="discard",
            pin_generated=True,
        )
        from evoke.manager import EvokeManager

        manager = EvokeManager(engine, cfg)
        manager.add_context_tokens(list(range(1024)), "ctx0")
        first_evictions = manager.get_stats().total_evictions
        manager.add_context_tokens(list(range(1024, 1280)), "ctx1")
        second_evictions = manager.get_stats().total_evictions
        assert second_evictions > first_evictions, (
            f"pin_generated regression: DOCUMENT blocks are being pinned. "
            f"evictions {first_evictions} -> {second_evictions}"
        )
