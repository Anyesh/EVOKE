"""Discriminating experiment: does EVOKE's compact+re-anchor recall beat
ArkVale's sparse+original-position recall as recall distance grows?

Both arms share model, budget, planted fact, and the recalled block (the fact
is force-evicted and force-recovered by key), so the ONLY variable is the
position mode:

  compact  EVOKE: the recalled fact is re-anchored to the contiguous tail, so
           it sits right before the probe (effective distance ~ 0).
  sparse   ArkVale-like: the recalled fact keeps its original early position, so
           the probe attends to it across the full filler (effective distance
           ~ the swept distance D).

If compact >> sparse and the gap widens with distance, re-anchoring is EVOKE's
real contribution. If compact ~= sparse, the headline collapses to ArkVale with
named blocks. If sparse > compact, re-anchoring hurts.

Requires (for the real arm):
  EVOKE_MODEL_PATH - path to a GGUF model (Qwen2.5-7B-Instruct)
  LLAMA_CPP_LIB    - path to the EVOKE llama.cpp build with kv_block primitives

A mock arm (--engine mock) validates the harness plumbing only; it cannot
measure recall accuracy because MockEngine does not run a real model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager
from evoke.mock_engine import MockEngine

INTRO = "General project background and onboarding notes. " * 20
FILLER_UNIT = (
    "The deployment pipeline runs nightly and archives logs to cold storage; "
    "unrelated implementation details accumulate across many modules. "
)
QUESTION = (
    "\n\nQuestion: what is the secret passkey mentioned much earlier in this "
    "document?\nAnswer: The secret passkey is"
)


def _make_engine(kind: str, n_ctx: int):
    if kind == "mock":
        return MockEngine(n_ctx=n_ctx)
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        raise SystemExit("set EVOKE_MODEL_PATH for the llama engine")
    engine = LlamaCppEngine(model, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
    if not engine.supports_kv_block:
        raise SystemExit("kv_block primitives not bound -- set LLAMA_CPP_LIB")
    return engine


def _filler_tokens(engine, n_tokens: int) -> list[int]:
    unit = engine.tokenize(FILLER_UNIT)
    if not unit:
        return []
    reps = (n_tokens // len(unit)) + 1
    return (unit * reps)[:n_tokens]


def _fact_block_ids(mgr: EvokeManager) -> list[int]:
    return [
        b.block_id for b in mgr._positions.active_blocks if b.key.startswith("fact#")
    ]


def _block_by_key_prefix(mgr: EvokeManager, prefix: str):
    for b in mgr._positions.active_blocks:
        if b.key.startswith(prefix):
            return b
    return None


def run_trial(
    engine,
    mode: str,
    distance: int,
    seed: int,
    n_ctx: int,
    block_size: int,
    gen_tokens: int,
) -> dict:
    rng = random.Random(seed)
    passkey = "".join(str(rng.randint(0, 9)) for _ in range(6))
    fact = (
        f"Important system note: the secret passkey is {passkey}. Keep it confidential."
    )

    engine.reset()
    # Budget far above the working set so nothing auto-evicts: the only eviction
    # is the deliberate force_evict of the fact, so the position mode is the
    # only thing that differs between the two arms.
    config = EvokeConfig(
        position_mode=mode,
        recovery_mode="kv_restore",
        block_size=block_size,
        max_active_tokens=n_ctx * 4,
    )
    mgr = EvokeManager(engine, config)

    mgr.load_document(INTRO)
    mgr.add_context(fact, key="fact")
    fact_pos_original = _block_by_key_prefix(mgr, "fact#").logical_start

    filler = _filler_tokens(engine, distance)
    if filler:
        mgr.add_context_tokens(filler, key="filler")

    fact_ids = _fact_block_ids(mgr)
    if not fact_ids:
        return {"error": "fact block missing before evict"}
    mgr.force_evict(fact_ids)

    recovered = mgr.recover("fact#0")
    fact_block = _block_by_key_prefix(mgr, "fact#")
    fact_pos_recalled = fact_block.logical_start if fact_block else -1

    mgr.process_user_message(QUESTION)
    probe_pos = mgr._engine.next_write_pos
    answer = mgr.generate(gen_tokens)
    effective_distance = probe_pos - fact_pos_recalled if recovered else -1
    hit = passkey in answer

    return {
        "mode": mode,
        "distance": distance,
        "seed": seed,
        "passkey": passkey,
        "recovered": recovered,
        "hit": bool(hit),
        "fact_pos_original": fact_pos_original,
        "fact_pos_recalled": fact_pos_recalled,
        "probe_pos": probe_pos,
        "effective_distance": effective_distance,
        "answer": answer[:120].encode("ascii", "replace").decode("ascii"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["mock", "llama"], default="llama")
    ap.add_argument("--distances", default="1024,4096,16384,49152")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-ctx", type=int, default=131072)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--out", default="results/recall_distance.json")
    args = ap.parse_args()

    distances = [int(d) for d in args.distances.split(",") if d]
    engine = _make_engine(args.engine, args.n_ctx)

    rows: list[dict] = []
    started = time.time()
    for mode in ("compact", "sparse"):
        for distance in distances:
            for seed in range(args.seeds):
                row = run_trial(
                    engine,
                    mode,
                    distance,
                    seed,
                    args.n_ctx,
                    args.block_size,
                    args.gen_tokens,
                )
                rows.append(row)
                print(
                    f"{mode:7s} d={distance:6d} seed={seed} "
                    f"recovered={row.get('recovered')} hit={row.get('hit')} "
                    f"eff_dist={row.get('effective_distance')}",
                    flush=True,
                )

    summary: dict[str, dict[str, dict]] = {}
    for mode in ("compact", "sparse"):
        summary[mode] = {}
        for distance in distances:
            trials = [
                r
                for r in rows
                if r.get("mode") == mode and r.get("distance") == distance
            ]
            hits = sum(1 for r in trials if r.get("hit"))
            summary[mode][str(distance)] = {
                "n": len(trials),
                "hits": hits,
                "pass_rate": hits / len(trials) if trials else 0.0,
            }

    out = {
        "engine": args.engine,
        "distances": distances,
        "seeds": args.seeds,
        "block_size": args.block_size,
        "elapsed_s": round(time.time() - started, 1),
        "summary": summary,
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== pass rate by distance ===")
    for distance in distances:
        c = summary["compact"][str(distance)]["pass_rate"]
        s = summary["sparse"][str(distance)]["pass_rate"]
        print(f"d={distance:6d}  compact={c:.2f}  sparse={s:.2f}  delta={c - s:+.2f}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
