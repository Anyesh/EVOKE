"""Aggregate per-run baseline_bench outputs into a tightened Section 7.3 table.

Reads results/baseline_n15/run_*.json, computes per-policy mean + 95% t-CI
of elapsed_s, and tabulates evictions/recoveries/probe_ok consistency
across runs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


T_CI_95 = {
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
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
}


def _t_ci_half_width(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    s = statistics.stdev(values)
    df = n - 1
    t = T_CI_95.get(df, 1.96)
    return t * s / math.sqrt(n)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="results/baseline_n15",
        help="Directory of per-run JSON outputs",
    )
    parser.add_argument(
        "--policies",
        default="no_eviction,truncate,evoke",
        help="Comma list of policies to aggregate",
    )
    args = parser.parse_args()

    run_files = sorted(Path(args.dir).glob("run_*.json"))
    if not run_files:
        print(f"no run_*.json found under {args.dir}")
        return 1

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    per_policy: dict[str, dict] = {
        p: {
            "elapsed_s": [],
            "evictions": [],
            "recoveries": [],
            "active_tokens": [],
            "probe_ok_count": 0,
            "n_runs": 0,
            "errors": [],
        }
        for p in policies
    }

    for rf in run_files:
        data = json.loads(rf.read_text())
        for r in data["results"]:
            p = r["policy"]
            if p not in per_policy:
                continue
            agg = per_policy[p]
            agg["n_runs"] += 1
            if r.get("error"):
                agg["errors"].append((rf.name, r["error"]))
                continue
            agg["elapsed_s"].append(float(r["elapsed_s"]))
            agg["evictions"].append(int(r["evictions"]))
            agg["recoveries"].append(int(r["recoveries"]))
            agg["active_tokens"].append(int(r["active_tokens"]))
            if r["probe_ok"]:
                agg["probe_ok_count"] += 1

    print(f"aggregating {len(run_files)} runs from {args.dir}")
    print()
    print(
        f"{'policy':<14}{'n':<4}{'probe':<8}{'mean_s':<8}"
        f"{'95% t-CI':<22}{'mean_evict':<12}{'mean_recov':<12}"
    )
    print("-" * 80)
    for p in policies:
        agg = per_policy[p]
        n = len(agg["elapsed_s"])
        if n == 0:
            print(f"{p:<14}{agg['n_runs']:<4}all-errors")
            continue
        mean_s = statistics.mean(agg["elapsed_s"])
        half = _t_ci_half_width(agg["elapsed_s"])
        ci_lo, ci_hi = mean_s - half, mean_s + half
        mean_ev = statistics.mean(agg["evictions"])
        mean_rec = statistics.mean(agg["recoveries"])
        probe = f"{agg['probe_ok_count']}/{n}"
        print(
            f"{p:<14}{n:<4}{probe:<8}{mean_s:<8.2f}"
            f"[{ci_lo:5.2f}, {ci_hi:5.2f}]{'':<6}"
            f"{mean_ev:<12.1f}{mean_rec:<12.1f}"
        )
    print()
    for p in policies:
        agg = per_policy[p]
        if agg["errors"]:
            print(f"ERRORS for policy={p}:")
            for fn, err in agg["errors"]:
                print(f"  {fn}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
