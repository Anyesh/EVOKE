"""Aggregate per-cell session-length JSONs into a (turns x policy) summary.

Reads results/session_length/T<turns>_S<seed>.json and emits mean elapsed
plus a t-distribution 95% CI per (turns, policy) cell. The CI uses df=n-1
matching the n=5 14-turn reporting in RESULTS.md so the curve is directly
comparable to the existing headline.

Usage: uv run python scripts/aggregate_session_length.py [path/to/results/session_length]
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def t_critical_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        20: 2.086,
        30: 2.042,
        50: 2.009,
    }
    if df in table:
        return table[df]
    if df > 50:
        return 1.96
    keys = sorted(table.keys())
    lo = max(k for k in keys if k <= df)
    hi = min(k for k in keys if k >= df)
    if lo == hi:
        return table[lo]
    frac = (df - lo) / (hi - lo)
    return table[lo] + frac * (table[hi] - table[lo])


def ci_t(values: list[float]) -> tuple[float, float, float]:
    if len(values) <= 1:
        m = values[0] if values else 0.0
        return m, m, m
    m = mean(values)
    s = stdev(values)
    half = t_critical_95(len(values) - 1) * s / math.sqrt(len(values))
    return m, m - half, m + half


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/session_length")
    if not root.exists():
        print(f"missing {root}")
        return 1

    by_cell: dict[tuple[int, str], list[float]] = defaultdict(list)
    probe_oks: dict[tuple[int, str], list[bool]] = defaultdict(list)
    evictions: dict[tuple[int, str], list[int]] = defaultdict(list)
    recoveries: dict[tuple[int, str], list[int]] = defaultdict(list)
    active_tokens: dict[tuple[int, str], list[int]] = defaultdict(list)

    for f in sorted(root.glob("T*_S*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        turns = data["turns"]
        for r in data["results"]:
            key = (turns, r["policy"])
            if r.get("error"):
                print(f"  skip {f.name} {r['policy']}: {r['error']}")
                continue
            by_cell[key].append(r["elapsed_s"])
            probe_oks[key].append(r["probe_ok"])
            evictions[key].append(r["evictions"])
            recoveries[key].append(r["recoveries"])
            active_tokens[key].append(r["active_tokens"])

    if not by_cell:
        print(f"no JSON cells under {root}")
        return 1

    print(f"\n# Session-length scaling (mean elapsed, t-95% CI, df=n-1)\n")
    print(
        f"| turns | policy        |  n | mean (s) | 95% CI            | probe pass | evict (mean) | recov (mean) | active (mean) |"
    )
    print(
        f"|------:|---------------|---:|---------:|-------------------|-----------:|-------------:|-------------:|--------------:|"
    )
    summary: list[dict] = []
    for turns, policy in sorted(by_cell.keys()):
        elapsed = by_cell[(turns, policy)]
        m, lo, hi = ci_t(elapsed)
        n = len(elapsed)
        pass_rate = sum(probe_oks[(turns, policy)]) / n if n else 0.0
        evict_m = mean(evictions[(turns, policy)]) if evictions[(turns, policy)] else 0
        recov_m = (
            mean(recoveries[(turns, policy)]) if recoveries[(turns, policy)] else 0
        )
        active_m = (
            mean(active_tokens[(turns, policy)])
            if active_tokens[(turns, policy)]
            else 0
        )
        print(
            f"| {turns:>5} | {policy:<13} | {n:>2} | {m:>8.2f} | "
            f"[{lo:>6.2f}, {hi:>6.2f}] | {pass_rate:>9.0%} | "
            f"{evict_m:>12.1f} | {recov_m:>12.1f} | {active_m:>13.0f} |"
        )
        summary.append(
            {
                "turns": turns,
                "policy": policy,
                "n": n,
                "mean_s": m,
                "ci95_lo_s": lo,
                "ci95_hi_s": hi,
                "probe_pass_rate": pass_rate,
                "evict_mean": evict_m,
                "recov_mean": recov_m,
                "active_tokens_mean": active_m,
            }
        )

    out_summary = root / "summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary JSON: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
