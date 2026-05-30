import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from arkvale_policy import ArkValePolicy, block_cuboid, cuboid_score
from evoke.config import EvokeConfig
from evoke.manager import EvokeManager
from evoke.mock_engine import MockEngine


def test_block_cuboid_min_max_mean_per_dim():
    k_block = np.array(
        [[[1.0, 5.0]], [[3.0, 0.0]], [[2.0, 10.0]]], dtype=np.float32
    )  # (3 tokens, 1 head, 2 dim)
    kmin, kmax, kmean = block_cuboid(k_block)
    assert np.allclose(kmin, [[1.0, 0.0]])
    assert np.allclose(kmax, [[3.0, 10.0]])
    assert np.allclose(kmean, [[2.0, 5.0]])


def test_cuboid_mean_score_is_mean_plus_deviation():
    # dev per dim = max(kmax-kmean, kmean-kmin): dim0 max(5-2,2-1)=3, dim1 max(5-3,3-1)=2.
    # score = q.kmean + |q|.dev = (1*2 + 1*3) + (1*3 + 1*2) = 10.
    q = np.array([[1.0, 1.0]], dtype=np.float32)
    kmin = np.array([[1.0, 1.0]], dtype=np.float32)
    kmax = np.array([[5.0, 5.0]], dtype=np.float32)
    kmean = np.array([[2.0, 3.0]], dtype=np.float32)
    assert cuboid_score(q, kmin, kmax, kmean) == 10.0


def test_cuboid_score_point_digest_is_dot_product():
    q = np.array([[1.0]], dtype=np.float32)
    for v in (1.0, 4.0):
        pt = np.array([[v]], dtype=np.float32)
        assert cuboid_score(q, pt, pt, pt) == v  # zero deviation -> q.mean


def _resident_keys(mgr):
    return {b.key for b in mgr._positions.active_blocks}


def test_recall_and_evict_keeps_top_budget_by_importance():
    engine = MockEngine()
    cfg = EvokeConfig(
        max_active_tokens=1_000_000,
        block_size=64,
        sink_count=0,
        recovery_mode="kv_restore",
        position_mode="compact",
    )
    mgr = EvokeManager(engine, cfg)
    for name in ("b0", "b1", "b2", "b3"):
        mgr.add_context("token " * 8, key=name)

    policy = ArkValePolicy(budget_blocks=2)
    q = np.array([[1.0]], dtype=np.float32)
    values = {"b0#0": 3.0, "b1#0": 1.0, "b2#0": 4.0, "b3#0": 2.0}
    for key, v in values.items():
        pt = np.array([[v]], dtype=np.float32)
        policy.set_cuboid(key, pt, pt, pt)  # point digest -> score == value

    ids = {b.key: b.block_id for b in mgr._positions.active_blocks}
    mgr.force_evict([ids["b2#0"], ids["b3#0"]])
    assert _resident_keys(mgr) == {"b0#0", "b1#0"}

    recalled, evicted = policy.recall_and_evict(mgr, q)

    # top-2 by score: b2(4), b0(3) -> b2 recalled, b1 evicted, b0 kept, b3 left out.
    assert _resident_keys(mgr) == {"b2#0", "b0#0"}
    assert recalled == 1
    assert evicted == 1


def test_recall_and_evict_protects_sinks():
    engine = MockEngine()
    cfg = EvokeConfig(
        max_active_tokens=1_000_000,
        block_size=64,
        sink_count=64,
        recovery_mode="kv_restore",
        position_mode="compact",
    )
    mgr = EvokeManager(engine, cfg)
    for name in ("s", "a", "b"):
        mgr.add_context("token " * 8, key=name)

    policy = ArkValePolicy(budget_blocks=1)
    q = np.array([[1.0]], dtype=np.float32)
    for key, v in {"a#0": 5.0, "b#0": 1.0}.items():
        pt = np.array([[v]], dtype=np.float32)
        policy.set_cuboid(key, pt, pt, pt)
    policy.recall_and_evict(mgr, q)
    assert "s#0" in _resident_keys(mgr)
