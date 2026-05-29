"""Diagnostic: is compact's gap below no_evict_tail a multi-recover bug or fundamental?

Reading the C path analytically rules out a double-shift (update() resets shift
per apply). This confirms it empirically and characterizes the miss pattern:

  1. Position integrity: after recovering N facts in compact mode, the recovered
     blocks must occupy contiguous, non-overlapping, correctly-ordered cells at
     the tail (> the filler region). A malformed layout means a placement bug.
  2. Per-fact miss pattern: which entities the probe misses, vs each fact's
     original position and its re-anchor delta. Misses correlated with large
     re-anchor distance => fundamental RoPE/attended-context effect. Random or
     positionally-malformed misses => implementation artifact.

Requires EVOKE_MODEL_PATH, LLAMA_CPP_LIB.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager

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
]
INTRO = "Internal operations handbook. Reference document, section preamble. " * 12
FILLER_UNIT = (
    "Routine maintenance windows are scheduled quarterly and logged for audit; "
    "miscellaneous operational notes accumulate between the key records. "
)


def _filler(engine, n: int) -> list[int]:
    unit = engine.tokenize(FILLER_UNIT)
    reps = (n // len(unit)) + 1
    return (unit * reps)[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=98304)
    ap.add_argument("--facts", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-ctx", type=int, default=131072)
    ap.add_argument("--block-size", type=int, default=64)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        raise SystemExit("set EVOKE_MODEL_PATH")
    engine = LlamaCppEngine(model, n_ctx=args.n_ctx, n_gpu_layers=-1, verbose=False)
    if not engine.supports_kv_block:
        raise SystemExit("kv_block primitives not bound -- set LLAMA_CPP_LIB")

    entities = ENTITIES[: args.facts]
    values, seen = [], set()
    while len(values) < args.facts:
        v = "".join(str(rng.randint(0, 9)) for _ in range(6))
        if v not in seen:
            seen.add(v)
            values.append(v)

    engine.reset()
    cfg = EvokeConfig(
        position_mode="compact",
        recovery_mode="kv_restore",
        block_size=args.block_size,
        max_active_tokens=args.n_ctx * 4,
    )
    mgr = EvokeManager(engine, cfg)
    mgr.load_document(INTRO)

    orig_start: dict[str, int] = {}
    per_gap = args.distance // args.facts
    for i, (ent, val) in enumerate(zip(entities, values)):
        gap = _filler(engine, per_gap)
        if gap:
            mgr.add_context_tokens(gap, key=f"gap{i}")
        mgr.add_context(
            f"Record {i}: the access code for project {ent} is {val}.", key=f"fact{i}"
        )
        block = next(b for b in mgr._positions.active_blocks if b.key == f"fact{i}#0")
        orig_start[f"fact{i}"] = block.logical_start

    filler_end = engine.next_write_pos
    fact_ids = [
        b.block_id for b in mgr._positions.active_blocks if b.key.startswith("fact")
    ]
    mgr.force_evict(fact_ids)
    # compact compacts the filler down by the evicted fact-tokens, so recovery
    # appends at this shorter tail, not at the pre-eviction filler_end.
    tail_after_evict = engine.next_write_pos
    for i in range(args.facts):
        mgr.recover(f"fact{i}#0")

    recovered = {
        b.key.split("#")[0]: (b.logical_start, b.logical_end)
        for b in mgr._positions.active_blocks
        if b.key.startswith("fact")
    }

    probe = (
        "\n\nQuestion: recall the secret access code for each project below.\n"
        "Projects: " + ", ".join(entities) + "\n"
        "Answer (one 'Project: code' per line):"
    )
    mgr.process_user_message(probe)
    answer = mgr.generate(384)

    print(
        f"filler_end (pre-evict) = {filler_end}  tail_after_evict = {tail_after_evict}"
    )
    print(f"{'fact':>7} {'orig_pos':>9} {'new_pos':>9} {'delta':>9} {'found':>6}")
    spans = []
    for i in range(args.facts):
        k = f"fact{i}"
        op = orig_start[k]
        np_, ne_ = recovered.get(k, (-1, -1))
        delta = np_ - op if np_ >= 0 else None
        found = values[i] in answer
        spans.append((np_, ne_, k))
        print(f"{k:>7} {op:>9} {np_:>9} {str(delta):>9} {str(found):>6}")

    spans.sort()
    contiguous = all(spans[j][1] == spans[j + 1][0] for j in range(len(spans) - 1))
    at_tail = all(s[0] >= tail_after_evict for s in spans)
    no_overlap = all(spans[j][1] <= spans[j + 1][0] for j in range(len(spans) - 1))
    found_n = sum(1 for i in range(args.facts) if values[i] in answer)

    print(f"\nfound {found_n}/{args.facts}")
    print(
        f"positions at_tail={at_tail} contiguous={contiguous} no_overlap={no_overlap}"
    )
    if at_tail and contiguous and no_overlap:
        print(
            "POSITION LAYOUT CLEAN -> no placement bug; gap is the fundamental "
            "stale-attended-context / multi-block effect, not malformed recovery."
        )
    else:
        print(
            "POSITION LAYOUT MALFORMED -> placement/recover bug; fixing it should "
            "raise compact toward no_evict_tail."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
