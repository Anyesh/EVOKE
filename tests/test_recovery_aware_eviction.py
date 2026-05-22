"""Unit tests for the recovery-aware eviction signal.

The session-length sweep at T=28 exposed a recover-then-immediately-evict
thrash where ~80 evictions per session were redundant cycles of the same
blocks. The fix adds a per-block recovery_strength field that fresh recoveries
set to recovery_strength_init, decays by recovery_decay each turn, and the
scorer weighs by w_recovery in the combined relevance score. These tests
pin the building blocks of that fix so the scorer behavior doesn't regress.
"""

from __future__ import annotations

from evoke.config import EvokeConfig
from evoke.manager import EvokeManager
from evoke.mock_engine import MockEngine
from evoke.scorer import RelevanceScorer
from evoke.types import ActiveBlock, BlockSource


def _block(block_id: int, recovery_strength: float = 0.0) -> ActiveBlock:
    return ActiveBlock(
        block_id=block_id,
        logical_start=0,
        logical_end=64,
        token_ids=list(range(64)),
        source=BlockSource.DOCUMENT,
        recovery_strength=recovery_strength,
    )


class TestActiveBlockRecoveryStrength:
    def test_default_is_zero(self):
        b = _block(0)
        assert b.recovery_strength == 0.0

    def test_field_persists_through_dataclass(self):
        b = _block(0, recovery_strength=0.75)
        assert b.recovery_strength == 0.75


class TestScorerRecoveryWeight:
    def test_default_w_recovery_zero_is_noop(self):
        cfg = EvokeConfig(w_recency=1.0, w_coherence=0.0, w_recovery=0.0)
        scorer = RelevanceScorer(cfg)
        b_with = _block(0, recovery_strength=1.0)
        b_without = _block(1, recovery_strength=0.0)
        assert scorer.score(
            b_with, current_pos=200, context_length=200
        ) == scorer.score(b_without, current_pos=200, context_length=200)

    def test_recovery_lifts_score_when_w_recovery_positive(self):
        cfg = EvokeConfig(w_recency=1.0, w_coherence=0.0, w_recovery=1.0)
        scorer = RelevanceScorer(cfg)
        b_fresh = _block(0, recovery_strength=1.0)
        b_stale = _block(1, recovery_strength=0.0)
        # Both blocks identical except recovery_strength; current_pos far
        # past the block so recency is low and recovery is the only lever
        # that separates them.
        s_fresh = scorer.score(b_fresh, current_pos=10000, context_length=10000)
        s_stale = scorer.score(b_stale, current_pos=10000, context_length=10000)
        assert s_fresh > s_stale, f"fresh={s_fresh} stale={s_stale}"

    def test_recovery_weighted_with_other_signals(self):
        # w_recovery = 1.0, w_recency = 0.0, w_coherence = 0.0: only recovery
        # matters. A strength=1.0 block scores 1.0; strength=0.5 block scores
        # 0.5 (assuming no other floors). Excludes the source-type floors
        # by setting BlockSource.DOCUMENT (no floor applied).
        cfg = EvokeConfig(
            w_recency=0.0,
            w_coherence=0.0,
            w_recovery=1.0,
            conversation_score_floor=0.0,
            assistant_score_floor=0.0,
        )
        scorer = RelevanceScorer(cfg)
        b = _block(0, recovery_strength=0.5)
        s = scorer.score(b, current_pos=200, context_length=200)
        assert abs(s - 0.5) < 1e-6, s


class TestTickTurnDecay:
    def _mgr(self, **overrides) -> EvokeManager:
        cfg_kwargs = dict(max_active_tokens=1024, block_size=64)
        cfg_kwargs.update(overrides)
        cfg = EvokeConfig(**cfg_kwargs)
        eng = MockEngine()
        return EvokeManager(eng, cfg)

    def test_decay_below_one_reduces_strength(self):
        mgr = self._mgr(recovery_decay=0.5)
        mgr._positions.append_block(_block(0, recovery_strength=1.0), 0)
        mgr.tick_turn()
        assert mgr._positions.active_blocks[0].recovery_strength == 0.5

    def test_decay_floors_below_threshold(self):
        mgr = self._mgr(recovery_decay=0.001)
        mgr._positions.append_block(_block(0, recovery_strength=0.005), 0)
        mgr.tick_turn()
        # 0.005 * 0.001 = 5e-6, well below 1e-3 floor
        assert mgr._positions.active_blocks[0].recovery_strength == 0.0

    def test_decay_one_is_noop(self):
        mgr = self._mgr(recovery_decay=1.0)
        mgr._positions.append_block(_block(0, recovery_strength=1.0), 0)
        mgr.tick_turn()
        assert mgr._positions.active_blocks[0].recovery_strength == 1.0

    def test_decay_skips_zero_strength_blocks(self):
        mgr = self._mgr(recovery_decay=0.5)
        mgr._positions.append_block(_block(0, recovery_strength=0.0), 0)
        mgr.tick_turn()
        # Stays at 0.0; the early-continue avoids touching ordinary blocks
        # that were never recovered.
        assert mgr._positions.active_blocks[0].recovery_strength == 0.0


class TestProcessUserMessageDecays:
    def test_process_user_message_runs_tick_turn(self):
        cfg = EvokeConfig(max_active_tokens=1024, block_size=64, recovery_decay=0.5)
        eng = MockEngine()
        mgr = EvokeManager(eng, cfg)
        # Seed a "recovered" block manually so we can verify decay fires
        # without needing a real eviction-recovery round-trip.
        mgr._positions.append_block(_block(0, recovery_strength=1.0), 0)

        mgr.process_user_message("any new turn text")

        assert mgr._positions.active_blocks[0].recovery_strength == 0.5
