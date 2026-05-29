"""Multi-fact discriminating experiment for the compact-vs-sparse position mode.

The single-fact recall sweep (recall_distance_bench.py) was a clean A==B null:
re-anchoring (compact) and original-position splice (sparse) both recall a lone
passkey perfectly at every distance, because single-fact recall has no headroom.

This bench moves to the regime where the modes can actually diverge: N facts
spread across the context, all force-evicted and force-recovered (selection held
constant), then the probe must retrieve all of them.

  compact    EVOKE: recalled facts are re-anchored to contiguous tail positions
             via byte-splice + RoPE shift, zero forward pass. Lossy (the K/V still
             encode the original attended context) but free.
  sparse     ArkVale-like: recalled facts keep their original spread-out positions,
             preserving true distance and order. Also zero forward pass.
  recompute  Ceiling: re-decode each evicted fact at the tail so it re-attends to
             the full current context. Recall must match no_evict_tail; the price
             is a forward pass over every recalled token, which compact/sparse skip.

Metric: fraction of the N planted codes that appear in the answer, averaged over
seeds, plus the recovery cost (tokens decoded, wall-clock). The operating point:
compact > sparse at the edge (relocation beats do-nothing) while compact <
recompute (relocation is lossy vs recompute) for zero decode cost vs N.

Requires (real arm): EVOKE_MODEL_PATH, LLAMA_CPP_LIB.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager
from evoke.mock_engine import MockEngine

import os

ENTITIES = [
    "Falcon",
    "Cedar",
    "Marble",
    "Quartz",
    "Ember",
    "Willow",
    "Cobalt",
    "Saffron",
    "Onyx",
    "Maple",
    "Indigo",
    "Basalt",
    "Cypress",
    "Garnet",
    "Aspen",
    "Slate",
]
INTRO = "Internal operations handbook. Reference document, section preamble. " * 12
FILLER_UNIT = (
    "Routine maintenance windows are scheduled quarterly and logged for audit; "
    "miscellaneous operational notes accumulate between the key records. "
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
    if n_tokens <= 0:
        return []
    unit = engine.tokenize(FILLER_UNIT)
    if not unit:
        return []
    reps = (n_tokens // len(unit)) + 1
    return (unit * reps)[:n_tokens]


def run_trial(
    engine,
    mode: str,
    distance: int,
    n_facts: int,
    seed: int,
    n_ctx: int,
    block_size: int,
    gen_tokens: int,
) -> dict:
    rng = random.Random(seed)
    entities = ENTITIES[:n_facts]
    values = []
    seen: set[str] = set()
    while len(values) < n_facts:
        v = "".join(str(rng.randint(0, 9)) for _ in range(6))
        if v not in seen:
            seen.add(v)
            values.append(v)

    engine.reset()
    # position_mode only matters for the recovery arms; the two control arms
    # (no_evict, no_evict_tail) never evict so it is inert there.
    pmode = mode if mode in ("compact", "sparse") else "compact"
    config = EvokeConfig(
        position_mode=pmode,
        recovery_mode="kv_restore",
        block_size=block_size,
        max_active_tokens=n_ctx * 4,  # never auto-evict; only the forced evict fires
    )
    mgr = EvokeManager(engine, config)

    mgr.load_document(INTRO)

    def _plant_fact(i: int, ent: str, val: str) -> None:
        mgr.add_context(
            f"Record {i}: the access code for project {ent} is {val}.",
            key=f"fact{i}",
        )

    if mode == "no_evict_tail":
        # Control for compact: facts decoded NATIVELY at the tail (after all
        # filler), never evicted. compact must match this if re-anchoring to
        # the tail is faithful to native tail decoding.
        filler = _filler_tokens(engine, distance)
        if filler:
            mgr.add_context_tokens(filler, key="filler")
        for i, (ent, val) in enumerate(zip(entities, values)):
            _plant_fact(i, ent, val)
    else:
        # compact / sparse / no_evict: facts spread across `distance` tokens of
        # filler, each at an increasing original position.
        per_gap = max(0, distance // n_facts)
        for i, (ent, val) in enumerate(zip(entities, values)):
            gap = _filler_tokens(engine, per_gap)
            if gap:
                mgr.add_context_tokens(gap, key=f"gap{i}")
            _plant_fact(i, ent, val)

    recovered = 0
    recover_tokens = 0
    recover_s = 0.0
    if mode in ("compact", "sparse", "recompute"):
        fact_ids = [
            b.block_id for b in mgr._positions.active_blocks if b.key.startswith("fact")
        ]
        t_recover = time.time()
        mgr.force_evict(fact_ids)
        for i in range(n_facts):
            if mode == "recompute":
                # The recompute ceiling: re-decode the evicted fact at the tail
                # instead of byte-splicing its stale K/V, so its tokens re-attend
                # to the full current context (recall must match no_evict_tail).
                # The forward pass over the fact tokens is the cost the
                # zero-recompute relocation arms (compact/sparse) avoid entirely.
                text = f"Record {i}: the access code for project {entities[i]} is {values[i]}."
                recover_tokens += len(engine.tokenize(text))
                mgr.add_context(text, key=f"refact{i}")
                recovered += 1
            elif mgr.recover(f"fact{i}#0"):
                recovered += 1
        recover_s = time.time() - t_recover
    # no_evict / no_evict_tail: facts are left in place, never evicted. These are
    # the ground-truth controls: sparse must equal no_evict (same original
    # positions, just evicted+restored), compact must equal no_evict_tail (facts
    # adjacent to the probe). A gap there means a recovery bug, not a real effect.

    probe = (
        "\n\nQuestion: recall the secret access code for each project below.\n"
        "Projects: " + ", ".join(entities) + "\n"
        "Answer (one 'Project: code' per line):"
    )
    mgr.process_user_message(probe)
    answer = mgr.generate(gen_tokens)
    found = sum(1 for v in values if v in answer)

    return {
        "mode": mode,
        "distance": distance,
        "n_facts": n_facts,
        "seed": seed,
        "recovered": recovered,
        "found": found,
        "accuracy": found / n_facts if n_facts else 0.0,
        "recover_tokens": recover_tokens,
        "recover_s": round(recover_s, 4),
        "answer": answer[:200].encode("ascii", "replace").decode("ascii"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["mock", "llama"], default="llama")
    ap.add_argument("--distances", default="4096,16384,49152,98304")
    ap.add_argument("--facts", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-ctx", type=int, default=131072)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--out", default="results/multifact_position_qwen25.json")
    args = ap.parse_args()

    distances = [int(d) for d in args.distances.split(",") if d]
    if args.facts > len(ENTITIES):
        raise SystemExit(f"--facts max is {len(ENTITIES)}")
    engine = _make_engine(args.engine, args.n_ctx)

    modes = ("compact", "sparse", "recompute", "no_evict", "no_evict_tail")
    rows: list[dict] = []
    started = time.time()
    for mode in modes:
        for distance in distances:
            for seed in range(args.seeds):
                row = run_trial(
                    engine,
                    mode,
                    distance,
                    args.facts,
                    seed,
                    args.n_ctx,
                    args.block_size,
                    args.gen_tokens,
                )
                rows.append(row)
                print(
                    f"{mode:13s} d={distance:6d} seed={seed} "
                    f"recovered={row['recovered']}/{args.facts} "
                    f"found={row['found']}/{args.facts} acc={row['accuracy']:.2f}",
                    flush=True,
                )

    summary: dict[str, dict[str, dict]] = {}
    for mode in modes:
        summary[mode] = {}
        for distance in distances:
            trials = [
                r for r in rows if r["mode"] == mode and r["distance"] == distance
            ]
            n = len(trials)
            mean_acc = sum(r["accuracy"] for r in trials) / n if n else 0.0
            mean_tok = sum(r["recover_tokens"] for r in trials) / n if n else 0.0
            mean_s = sum(r["recover_s"] for r in trials) / n if n else 0.0
            summary[mode][str(distance)] = {
                "n": n,
                "mean_accuracy": mean_acc,
                "mean_recover_tokens": mean_tok,
                "mean_recover_s": mean_s,
            }

    out = {
        "engine": args.engine,
        "distances": distances,
        "n_facts": args.facts,
        "seeds": args.seeds,
        "block_size": args.block_size,
        "elapsed_s": round(time.time() - started, 1),
        "summary": summary,
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== mean accuracy by distance (all arms) ===")
    print(
        f"{'distance':>8} {'compact':>8} {'sparse':>8} {'recomp':>8} "
        f"{'no_evict':>9} {'no_ev_tail':>11}"
    )
    for distance in distances:
        c = summary["compact"][str(distance)]["mean_accuracy"]
        s = summary["sparse"][str(distance)]["mean_accuracy"]
        rc = summary["recompute"][str(distance)]["mean_accuracy"]
        ne = summary["no_evict"][str(distance)]["mean_accuracy"]
        net = summary["no_evict_tail"][str(distance)]["mean_accuracy"]
        print(f"{distance:>8} {c:>8.2f} {s:>8.2f} {rc:>8.2f} {ne:>9.2f} {net:>11.2f}")

    # Validation gates: sparse must match no_evict (ArkVale-position recovery is
    # faithful) and recompute must match no_evict_tail (re-decode-at-tail is
    # faithful). compact-vs-no_evict_tail is NOT a gate: it is the finding
    # (re-anchoring is lossy vs native tail decode), so it is reported but not
    # used in the PASS verdict.
    print(
        "\n=== faithfulness check (sparse/recompute should be ~0; compact gap is the finding) ==="
    )
    max_sparse_gap = 0.0
    max_recompute_gap = 0.0
    for distance in distances:
        s = summary["sparse"][str(distance)]["mean_accuracy"]
        ne = summary["no_evict"][str(distance)]["mean_accuracy"]
        c = summary["compact"][str(distance)]["mean_accuracy"]
        net = summary["no_evict_tail"][str(distance)]["mean_accuracy"]
        rc = summary["recompute"][str(distance)]["mean_accuracy"]
        sg = s - ne
        cg = c - net
        rg = rc - net
        max_sparse_gap = max(max_sparse_gap, abs(sg))
        max_recompute_gap = max(max_recompute_gap, abs(rg))
        print(
            f"d={distance:6d}  sparse-no_evict={sg:+.2f}  "
            f"recompute-no_evict_tail={rg:+.2f}  compact-no_evict_tail={cg:+.2f}(finding)"
        )
    verdict = (
        "PASS: faithful arms (sparse, recompute) match their never-evicted twins (<=0.05)"
        if max_sparse_gap <= 0.05 and max_recompute_gap <= 0.05
        else "REVIEW: a faithful arm diverges from its twin; the position effect "
        "may be contaminated by an implementation artifact"
    )
    print(
        f"\nmax|sparse-no_evict|={max_sparse_gap:.2f}  "
        f"max|recompute-no_evict_tail|={max_recompute_gap:.2f}  -> {verdict}"
    )

    # Cost axis: the operating-point claim is that compact/sparse recall their
    # blocks for zero decoded tokens, where recompute pays a forward pass over
    # every recalled token. Report the price relocation avoids.
    print("\n=== recovery cost by distance (mean tokens decoded / wall-clock s) ===")
    print(f"{'distance':>8} {'compact':>18} {'sparse':>18} {'recompute':>18}")
    for distance in distances:
        cells = []
        for m in ("compact", "sparse", "recompute"):
            d = summary[m][str(distance)]
            cells.append(
                f"{d['mean_recover_tokens']:.0f}tok/{d['mean_recover_s']:.2f}s"
            )
        print(f"{distance:>8} {cells[0]:>18} {cells[1]:>18} {cells[2]:>18}")

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
