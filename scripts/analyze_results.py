"""Reviewer-pushback analysis pass over existing NIAH and multifact JSONs.

Produces four artifacts without requiring any new GPU time:

1. results/figures/niah_heatmap_<run>.png
   Per-strategy depth x budget pass-rate heatmap. The reviewer flagged "NIAH
   at 25 cells x 9 strategies x 3 budgets would benefit from a heatmap"; this
   is that figure. Each subplot is one strategy, rows are budgets, columns
   are depth percentages, cell color is pass-rate averaged over needles.

2. results/figures/multifact_bar_<run>.png
   Strategy-by-strategy bar chart with 95% Wilson CIs. The reviewer flagged
   "the multifact bar chart with CIs would land harder than Table 2"; this
   is that figure.

3. results/analysis/tail_latency_<run>.md
   p50 / p95 / p99 per (budget, strategy) elapsed time. The reviewer flagged
   "No tail latency. Means only. p95/p99 would matter for the serving claim";
   this is that table.

4. results/analysis/multifact_failure_xtab_<run>.md
   Per-(strategy, fact_id) pass-rate cross-tab over the existing n=5 multifact
   data. The reviewer's Q4 asks whether the 60% pass rate is selection failure
   (wrong block recovered) or substitution failure (right block, wrong K/V);
   the cross-tab is the first cut at telling those apart by showing which
   facts each strategy systematically misses.

Usage:
    uv run python scripts/analyze_results.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


import os

REPO = Path(__file__).resolve().parents[1]
NIAH_JSON = Path(
    os.environ.get(
        "EVOKE_ANALYZE_NIAH",
        REPO / "results" / "niah_qwen25_7b_with_snapkv_infllm.json",
    )
)
MFB_JSON = Path(
    os.environ.get(
        "EVOKE_ANALYZE_MFB",
        REPO / "results" / "mfb_qwen25_7b_with_snapkv_infllm.json",
    )
)
RUN_TAG = os.environ.get("EVOKE_ANALYZE_TAG", "qwen25_7b_snapkv_infllm")
FIG_DIR = REPO / "results" / "figures"
ANL_DIR = REPO / "results" / "analysis"


STRATEGY_ORDER = [
    "recency",
    "streaming_llm",
    "evoke_discard",
    "evoke_breadcrumb",
    "h2o",
    "snapkv",
    "infllm",
    "evoke_kv_restore",
    "evoke_attention",
]


def wilson_ci(passes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = passes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def niah_heatmap(records: list[dict], out_path: Path, run_tag: str) -> None:
    strategies = [s for s in STRATEGY_ORDER if any(r["strategy"] == s for r in records)]
    budgets = sorted({r["budget"] for r in records})
    depths = sorted({r["depth"] for r in records})

    pass_count: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for r in records:
        pass_count[(r["strategy"], r["budget"], r["depth"])].append(int(r["probe_ok"]))

    n_strats = len(strategies)
    n_cols = min(3, n_strats)
    n_rows = math.ceil(n_strats / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 2.4 * n_rows), squeeze=False
    )
    cmap = plt.cm.RdYlGn

    for idx, strategy in enumerate(strategies):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        grid = np.zeros((len(budgets), len(depths)))
        for i, b in enumerate(budgets):
            for j, d in enumerate(depths):
                vals = pass_count.get((strategy, b, d), [])
                grid[i, j] = mean(vals) if vals else 0.0
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([f"{d}%" for d in depths], fontsize=8)
        ax.set_yticks(range(len(budgets)))
        ax.set_yticklabels(budgets, fontsize=8)
        ax.set_title(strategy, fontsize=10)
        if row == n_rows - 1:
            ax.set_xlabel("depth", fontsize=9)
        if col == 0:
            ax.set_ylabel("budget", fontsize=9)
        for i in range(len(budgets)):
            for j in range(len(depths)):
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if 0.3 < grid[i, j] < 0.7 else "white",
                )
    for idx in range(n_strats, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    cbar.set_label("pass rate (avg over needles)", fontsize=8)
    fig.suptitle(
        f"NIAH pass rate by strategy / budget / depth ({run_tag})", fontsize=11
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def multifact_bar(records: list[dict], out_path: Path, run_tag: str) -> None:
    by_strategy: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in records:
        passes = sum(1 for ok in r["fact_results"].values() if ok)
        total = len(r["fact_results"])
        p, t = by_strategy[r["strategy"]]
        by_strategy[r["strategy"]] = (p + passes, t + total)

    strategies = [s for s in STRATEGY_ORDER if s in by_strategy]
    rates = []
    lows = []
    highs = []
    labels = []
    for s in strategies:
        p, t = by_strategy[s]
        rate = p / t if t else 0.0
        lo, hi = wilson_ci(p, t)
        rates.append(rate * 100)
        lows.append((rate - lo) * 100)
        highs.append((hi - rate) * 100)
        labels.append(f"{s}\n({p}/{t})")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(strategies))
    colors = [
        "#d73027"
        if r < 20
        else "#fc8d59"
        if r < 50
        else "#fee08b"
        if r < 70
        else "#1a9850"
        for r in rates
    ]
    ax.bar(x, rates, yerr=[lows, highs], color=colors, capsize=4, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("multifact pass rate (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.axhline(50, color="grey", linewidth=0.5, linestyle=":")
    ax.set_title(
        f"Multifact pass rate with 95% Wilson CI ({run_tag})",
        fontsize=11,
    )
    for i, r in enumerate(rates):
        ax.text(i, r + highs[i] + 2, f"{r:.0f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def tail_latency_table(records: list[dict], out_path: Path, run_tag: str) -> None:
    by_cell: dict[tuple[int, str], list[float]] = defaultdict(list)
    for r in records:
        by_cell[(r["budget"], r["strategy"])].append(float(r["elapsed"]))

    lines = [
        f"# NIAH per-cell tail latency ({run_tag})",
        "",
        "Each cell aggregates over the (needle, depth) combinations: 25 cells per",
        "(budget, strategy) for the standard 5-needle x 5-depth grid. p99 over",
        "25 samples is the 25th-from-best ranked value; p95 is the second-worst",
        "rounded; report p50 / p95 / p99 alongside the mean reviewers asked for.",
        "",
        "| budget | strategy        |  n |  mean | p50 | p95 | p99 | max |",
        "|-------:|-----------------|---:|------:|----:|----:|----:|----:|",
    ]
    for budget, strategy in sorted(by_cell.keys()):
        v = by_cell[(budget, strategy)]
        lines.append(
            f"| {budget:>6} | {strategy:<15} | {len(v):>2} | "
            f"{mean(v):>5.2f} | {median(v):>3.2f} | "
            f"{percentile(v, 95):>3.2f} | "
            f"{percentile(v, 99):>3.2f} | {max(v):>3.2f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def multifact_failure_xtab(records: list[dict], out_path: Path, run_tag: str) -> None:
    by_strategy_fact: dict[tuple[str, str], tuple[int, int]] = defaultdict(
        lambda: (0, 0)
    )
    fact_ids: set[str] = set()
    for r in records:
        for fact_id, ok in r["fact_results"].items():
            fact_ids.add(fact_id)
            p, t = by_strategy_fact[(r["strategy"], fact_id)]
            by_strategy_fact[(r["strategy"], fact_id)] = (p + int(ok), t + 1)

    facts_sorted = sorted(fact_ids)
    strategies = [s for s in STRATEGY_ORDER if any(s == k[0] for k in by_strategy_fact)]
    seeds_seen = sorted({r["seed"] for r in records})
    budgets_seen = sorted({r["budget"] for r in records})
    lines = [
        f"# Multifact per-fact pass-rate cross-tab ({run_tag})",
        "",
        f"Pass rate of each strategy on each of the five planted facts, aggregated",
        f"over {len(seeds_seen)} seeds x {len(budgets_seen)} budgets ({len(seeds_seen) * len(budgets_seen)} cells per (strategy, fact)).",
        "The reviewer's Q4 asks whether multifact's pass-rate headline is selection",
        "failure (wrong block recovered) or substitution failure (right block,",
        "wrong K/V). A strategy that fails the same fact across most seeds is",
        "consistent with that fact being structurally hard for the strategy's",
        "selection rule. A strategy whose failure mass is spread evenly across",
        "facts is more consistent with substitution noise.",
        "",
        "| strategy        | "
        + " | ".join(f"{f:<8}" for f in facts_sorted)
        + " |  overall |",
        "|-----------------|"
        + "|".join("-" * 10 for _ in facts_sorted)
        + "|---------:|",
    ]
    for s in strategies:
        cells = []
        total_p = 0
        total_t = 0
        for f in facts_sorted:
            p, t = by_strategy_fact[(s, f)]
            total_p += p
            total_t += t
            cells.append(f"{p}/{t}" if t else "-/-")
        overall = (total_p / total_t) if total_t else 0.0
        lines.append(
            f"| {s:<15} | "
            + " | ".join(f"{c:<8}" for c in cells)
            + f" | {overall:>7.1%} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANL_DIR.mkdir(parents=True, exist_ok=True)

    mfb = json.loads(MFB_JSON.read_text(encoding="utf-8"))
    run_tag = RUN_TAG

    multifact_bar(mfb, FIG_DIR / f"multifact_bar_{run_tag}.png", run_tag)
    multifact_failure_xtab(
        mfb, ANL_DIR / f"multifact_failure_xtab_{run_tag}.md", run_tag
    )

    if NIAH_JSON.exists():
        niah = json.loads(NIAH_JSON.read_text(encoding="utf-8"))
        niah_heatmap(niah, FIG_DIR / f"niah_heatmap_{run_tag}.png", run_tag)
        tail_latency_table(niah, ANL_DIR / f"tail_latency_{run_tag}.md", run_tag)
    else:
        print(f"NIAH JSON {NIAH_JSON} not found; skipping NIAH plots")

    print(f"figures written to {FIG_DIR}")
    print(f"analysis written to {ANL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
