"""Linchpin test: is byte-exact recovered KV as USABLE for recall as a fresh re-decode?

verify_kv_fidelity proved the saved K/V round-trips token-for-token in a clean layout
(neighbors resident). The autonomous loop then showed the model sometimes answers worse
after recovery than after a fresh re-decode, but that loop is too noisy to attribute the
gap (multi-turn ReAct anchoring, a 'done' attractor). This isolates the actual question:
under a tight budget that evicts the target among holes, does the model recall a fact from
RECOVERED context as reliably as from a fresh RE-DECODE or a never-evicted RESIDENT copy?

Method: N distinct facts (unguessable 4-digit values so a prior cannot supply the answer).
For each fact, build the same document (target + distractors), force the same eviction
(target plus alternating distractors, so the target sits among holes), then restore the
target per condition and probe its value greedily. Correctness is value-in-answer. The
comparison is the RATE across N facts, not a single greedy run.

Conditions:
  resident          never evicted (ceiling)
  redecode          evicted, re-decoded from source before the probe (the no_recovery path)
  recovered_compact evicted, KV spliced back at the contiguous tail (EVOKE's agent layout)
  recovered_sparse  evicted, KV spliced back in place at original position (ArkVale-like)
  discard           evicted, never restored (floor / sensitivity control)

Verdict: recovered_* must approach redecode/resident. If recovered_* << redecode, recovery
is not as usable as paying to re-decode, and that is the honest linchpin result. If
recovered_* ~= redecode, the autonomous-loop gap was a ReAct artifact, not a KV problem.

Requires EVOKE_MODEL_PATH and LLAMA_CPP_LIB (the fork build).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager

CONDITIONS = (
    "resident",
    "redecode",
    "recovered_compact",
    "recovered_sparse",
    "discard",
)
N_DISTRACTORS = 8


def make_facts(n: int) -> list[tuple[str, int]]:
    # Deterministic, distinct, 4-digit values so the model cannot supply the answer
    # from a parametric prior (the failure mode where it guessed 30/100).
    facts = []
    for i in range(n):
        key = f"CFG_{chr(65 + i % 26)}{chr(65 + (i // 26) % 26)}_{i:02d}"
        value = 1000 + i * 7
        facts.append((key, value))
    return facts


def _snippet(key: str, value: int) -> str:
    return (
        f"Setting block for {key}.\n"
        f"# {key} is consulted by the service at startup and on every reload.\n"
        f"{key} = {value}\n"
        f"The operator must restart workers after changing {key}.\n"
    )


def _distractor(j: int) -> str:
    dk = f"AUX_PARAM_{j:02d}"
    return (
        f"Auxiliary note {j}.\n"
        f"# {dk} tunes an unrelated background pathway; not relevant to the query.\n"
        f"{dk} = {500 + j}\n"
        f"This is filler to create realistic interference and eviction pressure.\n"
    )


def _config(condition: str, n_ctx: int) -> EvokeConfig:
    # Budget huge for resident (nothing evicts); tight otherwise so the explicit
    # force_evict is what removes the target, with the watermark left loose enough
    # that it does not also drop the pinned preamble or the probe.
    budget = n_ctx if condition == "resident" else 100_000
    position_mode = "sparse" if condition == "recovered_sparse" else "compact"
    recovery_mode = "kv_restore" if condition.startswith("recovered") else "discard"
    return EvokeConfig(
        max_active_tokens=budget,
        block_size=64,
        sink_count=0,
        recovery_mode=recovery_mode,
        position_mode=position_mode,
        high_watermark=0.999,
        low_watermark=0.99,
    )


def _target_blocks(mgr: EvokeManager) -> list:
    return [b for b in mgr._positions.active_blocks if b.key.startswith("target#")]


def run_fact(
    engine: LlamaCppEngine, key: str, value: int, condition: str, n_ctx: int
) -> bool:
    engine.reset()
    mgr = EvokeManager(engine, _config(condition, n_ctx))
    mgr.add_context(
        "You are a configuration lookup assistant. Answer with only the number.\n",
        key="preamble",
        pinned=True,
    )
    mgr.add_context(_snippet(key, value), key="target")
    for j in range(N_DISTRACTORS):
        mgr.add_context(_distractor(j), key=f"dist{j}")

    if condition != "resident":
        # Evict the target plus alternating distractors so the target ends up
        # surrounded by holes, matching the agent setting where cold neighbors are gone.
        victims = [b.block_id for b in _target_blocks(mgr)]
        for j in range(0, N_DISTRACTORS, 2):
            victims += [
                b.block_id
                for b in mgr._positions.active_blocks
                if b.key.startswith(f"dist{j}#")
            ]
        mgr.force_evict(victims)

        if condition in ("recovered_compact", "recovered_sparse"):
            for crumb in mgr.get_breadcrumbs():
                if crumb.key.startswith("target#"):
                    mgr.recover(crumb.key)
        elif condition == "redecode":
            mgr.add_context(_snippet(key, value), key="target_re")
        # discard: leave the hole

    mgr.process_user_message(
        f"What exact value is assigned to {key} in the configuration? Answer with only the number."
    )
    answer = mgr.generate(24)
    return str(value) in answer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--n-facts", type=int, default=24)
    args = ap.parse_args()

    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("FAIL: set EVOKE_MODEL_PATH")
        return 1

    engine = LlamaCppEngine(model, n_ctx=args.n_ctx, n_gpu_layers=-1, verbose=False)
    if not engine.supports_kv_block:
        print(
            "FAIL: kv_block primitives not bound -- set LLAMA_CPP_LIB to the fork build"
        )
        return 1

    facts = make_facts(args.n_facts)
    results: dict[str, list[bool]] = {c: [] for c in CONDITIONS}
    for key, value in facts:
        for cond in CONDITIONS:
            results[cond].append(run_fact(engine, key, value, cond, args.n_ctx))

    print(f"\nRECOVERY USABILITY  n_facts={len(facts)} n_ctx={args.n_ctx}")
    print(f"{'condition':18s} {'correct':>8s} {'rate':>7s}")
    for cond in CONDITIONS:
        n_ok = sum(results[cond])
        print(f"{cond:18s} {n_ok:>3d}/{len(facts):<3d} {n_ok / len(facts):>6.1%}")

    res = {c: sum(results[c]) / len(facts) for c in CONDITIONS}
    print()
    gap = res["redecode"] - max(res["recovered_compact"], res["recovered_sparse"])
    if res["discard"] >= res["redecode"] - 1e-9:
        print(
            "INCONCLUSIVE: discard recall is not below redecode, so the probe is not "
            "sensitive to the target block -- strengthen the values/distractors."
        )
        return 1
    if gap <= 0.05:
        print(
            "LINCHPIN SUPPORTED: recovered recall matches re-decode (within 5 points). "
            "Recompute-free recovery is as usable as paying to re-decode; the autonomous-loop "
            "gap was a multi-turn artifact, not a KV-usability problem."
        )
        return 0
    print(
        f"LINCHPIN STRAINED: re-decode recall exceeds the best recovered mode by {gap:.0%}. "
        "Recompute-free recovery is measurably less usable than re-decoding on this model; "
        "report this honestly and investigate the position/layout interaction."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
