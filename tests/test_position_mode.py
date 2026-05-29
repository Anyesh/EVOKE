import importlib.util
from pathlib import Path

from evoke.config import EvokeConfig
from evoke.manager import EvokeManager
from evoke.mock_engine import MockEngine
from evoke.types import ActiveBlock

_BENCH_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "multifact_position_bench.py"
)
_spec = importlib.util.spec_from_file_location("multifact_position_bench", _BENCH_PATH)
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)
run_trial = _bench.run_trial


class TestEngineEvictCompactFlag:
    def test_compact_remaps_survivors(self):
        engine = MockEngine()
        engine.process_tokens(list(range(200)))
        engine.evict_ranges([(50, 100)], compact=True)
        assert engine.next_write_pos == 150
        assert engine.get_kv_cache_token_count() == 150

    def test_sparse_keeps_absolute_positions_and_write_pos(self):
        engine = MockEngine()
        engine.process_tokens(list(range(200)))
        engine.evict_ranges([(50, 100)], compact=False)
        # next_write_pos (the true max) is unchanged: new tokens keep decoding
        # at their true absolute index, leaving a hole at [50, 100).
        assert engine.next_write_pos == 200
        # resident count drops by the 50 removed cells.
        assert engine.get_kv_cache_token_count() == 150
        # a survivor above the hole keeps its original position.
        assert engine._token_at_pos.get(120) == 120
        # the hole is genuinely empty.
        assert engine._token_at_pos.get(60) is None

    def test_sparse_load_into_gap_preserves_write_pos(self):
        engine = MockEngine()
        engine.process_tokens(list(range(200)))
        saved = engine.kv_block_save(50, 100)
        engine.evict_ranges([(50, 100)], compact=False)
        ok = engine.kv_block_load(saved, new_p0=50)
        assert ok
        # splicing into the mid-cache gap must not move the tail backwards.
        assert engine.next_write_pos == 200
        assert engine._token_at_pos.get(60) == 60


def _manager(mode: str) -> tuple[EvokeManager, MockEngine]:
    cfg = EvokeConfig(
        position_mode=mode,
        recovery_mode="kv_restore",
        block_size=64,
        max_active_tokens=10_000,  # large so only force_evict evicts
    )
    engine = MockEngine()
    return EvokeManager(engine, cfg), engine


def _block_by_key(mgr: EvokeManager, key: str) -> ActiveBlock | None:
    for b in mgr._positions.active_blocks:
        if b.key == key:
            return b
    return None


class TestManagerSparseRegime:
    def test_sparse_eviction_leaves_gap_and_recovers_at_original_position(self):
        mgr, engine = _manager("sparse")
        mgr.add_context_tokens(list(range(256)), key="ctx")  # 4 blocks at 0..255
        target = _block_by_key(mgr, "ctx#1")
        assert target is not None
        original_start = target.logical_start  # 64
        survivor_before = _block_by_key(mgr, "ctx#2").logical_start  # 128

        mgr.force_evict([target.block_id])

        # sparse: tail unchanged, survivors keep their absolute positions.
        assert engine.next_write_pos == 256
        assert _block_by_key(mgr, "ctx#1") is None
        assert _block_by_key(mgr, "ctx#2").logical_start == survivor_before == 128

        assert mgr.recover("ctx#1") is True
        recovered = _block_by_key(mgr, "ctx#1")
        assert recovered is not None
        # the recalled block lands back at its ORIGINAL position (ArkVale-like).
        assert recovered.logical_start == original_start == 64
        # recovery into the gap does not advance the tail.
        assert engine.next_write_pos == 256


class TestManagerCompactRegime:
    def test_compact_eviction_recompacts_and_recovers_at_tail(self):
        mgr, engine = _manager("compact")
        mgr.add_context_tokens(list(range(256)), key="ctx")
        target = _block_by_key(mgr, "ctx#1")
        assert target is not None

        mgr.force_evict([target.block_id])

        # compact: survivors re-indexed contiguously, tail shrinks.
        assert engine.next_write_pos == 192
        assert _block_by_key(mgr, "ctx#2").logical_start == 64

        assert mgr.recover("ctx#1") is True
        recovered = _block_by_key(mgr, "ctx#1")
        assert recovered is not None
        # the recalled block lands at the new contiguous tail (EVOKE default).
        assert recovered.logical_start == 192
        assert engine.next_write_pos == 256

    def test_compact_is_the_default(self):
        assert EvokeConfig().position_mode == "compact"


class TestRecomputeArmCost:
    def test_relocation_arms_decode_zero_tokens_recompute_pays(self):
        engine = MockEngine(n_ctx=8192)

        def trial(mode: str) -> dict:
            return run_trial(
                engine,
                mode,
                distance=512,
                n_facts=4,
                seed=0,
                n_ctx=8192,
                block_size=64,
                gen_tokens=8,
            )

        compact = trial("compact")
        sparse = trial("sparse")
        recompute = trial("recompute")

        assert compact["recovered"] == 4
        assert sparse["recovered"] == 4
        assert recompute["recovered"] == 4

        # The operating-point contract: byte-splice relocation costs no forward
        # pass, recompute pays one over every recalled token.
        assert compact["recover_tokens"] == 0
        assert sparse["recover_tokens"] == 0
        assert recompute["recover_tokens"] > 0

        for r in (compact, sparse, recompute):
            assert "recover_s" in r
