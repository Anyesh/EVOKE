"""Generate EVOKE paper figures from saved measured data.

Each figure derives from a data file under results/ so the paper's visuals are reproducible
rather than hand-drawn. Writes PDFs into paper/figures/. Run: uv run python scripts/figures/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "figures"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

C_RECOVER = "#1b7837"
C_REDECODE = "#5aae61"
C_RESIDENT = "#a6dba0"
C_DISCARD = "#b2182b"
C_EVOKE = "#1b7837"
C_NOREC = "#9970ab"
C_NOEVICT = "#b2182b"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def fig_recovery_fidelity(data: dict) -> None:
    # The linchpin: recovered KV recall matches a fresh re-decode and a never-evicted
    # resident copy, both with a single eviction and under heavy churn; discard is the
    # sensitivity floor. Two panels = two churn regimes; two bars per condition = two models.
    ru = data["recovery_usability"]
    cf = data["churn_fidelity"]
    models = ["Qwen3-14B", "Qwen3.5-9B-hybrid"]
    labels = ["Qwen3-14B\n(attention)", "Qwen3.5-9B\n(hybrid)"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    panels = [
        (
            axes[0],
            "No churn (1 eviction)",
            ["resident", "redecode", "recovered_compact", "discard"],
            ["resident", "re-decode", "recovered", "discard"],
            ru,
        ),
        (
            axes[1],
            "Heavy churn (~128 evictions)",
            ["ref", "redecode", "recovered", "discard"],
            ["resident", "re-decode", "recovered", "discard"],
            cf,
        ),
    ]
    colors = [C_RESIDENT, C_REDECODE, C_RECOVER, C_DISCARD]
    width = 0.18
    for ax, title, keys, leg, src in panels:
        for ci, (k, lab, col) in enumerate(zip(keys, leg, colors)):
            vals = [src[m][k] * 100 for m in models]
            xs = [j + (ci - 1.5) * width for j in range(len(models))]
            ax.bar(
                xs, vals, width, label=lab, color=col, edgecolor="black", linewidth=0.4
            )
        ax.set_title(title)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 108)
        ax.axhline(100, color="gray", lw=0.5, ls=":")
    axes[0].set_ylabel("recall (%)")
    axes[1].legend(loc="center right", fontsize=7.5, framealpha=0.9)
    fig.suptitle(
        "Recovered KV is as usable as a fresh re-decode (recall, greedy)", y=1.02
    )
    fig.savefig(OUT / "recovery_fidelity.pdf")
    plt.close(fig)


def fig_scale_wall(data: dict) -> None:
    # The context wall: without eviction the session cannot be held (KV hits n_ctx and the
    # run dies mid-corpus); EVOKE holds the whole corpus within budget at far lower peak KV,
    # recompute-free (decoded ~ corpus once, vs the re-decode tax of no_recovery).
    sd = data["scale_demo"]
    arms = sd["arms"]
    budget = sd["budget"]
    n_ctx = sd["n_ctx"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    names = ["EVOKE", "no\nrecovery", "no\neviction"]
    keys = ["evoke", "no_recovery", "no_eviction"]
    cols = [C_EVOKE, C_NOREC, C_NOEVICT]
    peaks = [arms[k]["peak_kv"] for k in keys]
    bars = ax1.bar(names, peaks, color=cols, edgecolor="black", linewidth=0.4)
    ax1.set_xlim(-0.7, 2.6)
    ax1.axhline(budget, color="#1b7837", lw=1.0, ls="--")
    ax1.text(
        -0.62,
        budget - 700,
        f"budget {budget}",
        color="#1b7837",
        fontsize=6.5,
        ha="left",
        va="top",
    )
    ax1.axhline(n_ctx, color="#b2182b", lw=1.0, ls="--")
    ax1.text(
        -0.62,
        n_ctx + 400,
        f"n_ctx wall {n_ctx}",
        color="#b2182b",
        fontsize=6.5,
        ha="left",
        va="bottom",
    )
    ax1.set_ylabel("peak KV (tokens)")
    ax1.set_title("Peak KV held (66K-token session)")
    ax1.set_ylim(0, n_ctx * 1.16)
    for b, k in zip(bars, keys):
        tag = "dies @ sec 74" if not arms[k]["completed"] else "completes"
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 500,
            tag,
            ha="center",
            fontsize=6.5,
        )
    ax1.annotate(
        f"{sd['peak_kv_ratio_noevict_over_evoke']}x\nlower",
        xy=(0, peaks[0]),
        xytext=(0.05, n_ctx * 0.5),
        fontsize=8,
        color="#1b7837",
        ha="center",
    )

    decoded = [arms[k]["decoded"] for k in keys]
    bars2 = ax2.bar(names, decoded, color=cols, edgecolor="black", linewidth=0.4)
    ax2.set_ylabel("tokens decoded")
    ax2.set_title("Decode cost (recompute-free recovery)")
    for b, k in zip(bars2, keys):
        note = "incomplete" if not arms[k]["completed"] else ""
        if note:
            ax2.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 800,
                note,
                ha="center",
                fontsize=6.5,
            )
    ax2.text(
        1.0,
        arms["no_recovery"]["decoded"] + 3500,
        f"{sd['decode_ratio_norecovery_over_evoke']:.2f}x re-decode tax",
        fontsize=7,
        ha="center",
        color="#9970ab",
    )
    fig.savefig(OUT / "scale_wall.pdf")
    plt.close(fig)


def fig_infllm_budget(fd: dict) -> None:
    # Honest head-to-head with the one recovery-bearing competitor on the same substrate:
    # InfLLM separates at b=512, the two tie at b=1024, EVOKE leads at b=2048 with overlapping
    # CIs. Error bars are Wilson 95% CIs; no favorable cherry-pick.
    mf = fd["multifact_n15"]
    budgets = mf["budgets"]
    x = list(range(len(budgets)))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for key, lab, col, off in [
        ("evoke_kv_restore", "EVOKE", C_EVOKE, -0.06),
        ("infllm", "InfLLM", "#762a83", 0.06),
    ]:
        s = mf["Qwen2.5-7B"][key]
        lo = [m - l for m, l in zip(s["mean"], s["lo"])]
        hi = [h - m for m, h in zip(s["mean"], s["hi"])]
        ax.errorbar(
            [xi + off for xi in x],
            s["mean"],
            yerr=[lo, hi],
            label=lab,
            color=col,
            marker="o",
            capsize=3,
            lw=1.5,
            markersize=5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"b={b}" for b in budgets])
    ax.set_ylabel("multi-fact pass rate (%)")
    ax.set_title("EVOKE vs InfLLM, same substrate (Qwen2.5-7B, n=15)")
    ax.set_ylim(30, 95)
    for xi, tag in zip(
        x, ["InfLLM wins\n(separated)", "tie", "EVOKE leads\n(CIs overlap)"]
    ):
        ax.text(xi, 33, tag, ha="center", fontsize=6.5, color="gray")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(OUT / "infllm_budget.pdf")
    plt.close(fig)


def fig_crossarch(fd: dict) -> None:
    # Recovery is the dividing line, across architectures: recovery-bearing EVOKE passes
    # multi-fact on attention, hybrid Mamba/attention, and MoE; the best recovery-less baseline
    # collapses on each. n=5 grid, b=1024.
    ca = fd["crossarch_multifact"]
    labels = ca["order"]
    x = list(range(len(labels)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(
        [xi - w / 2 for xi in x],
        ca["evoke"],
        w,
        label="EVOKE (recovery-bearing)",
        color=C_EVOKE,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        [xi + w / 2 for xi in x],
        ca["best_recovery_less"],
        w,
        label="best recovery-less baseline",
        color=C_DISCARD,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("multi-fact pass rate (%)")
    ax.set_title("Recovery is the dividing line (n=5, b=1024)")
    ax.set_ylim(0, 80)
    for xi, v in zip(x, ca["evoke"]):
        ax.text(xi - w / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5)
    for xi, v in zip(x, ca["best_recovery_less"]):
        ax.text(xi + w / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(OUT / "crossarch_divide.pdf")
    plt.close(fig)




C_WORKSPACE = "#2a78d6"
C_SNAPKV = "#eb6834"
C_RECENCY = "#4a3aa7"

JLENS_MODELS = [
    ("analysis_qwen2.5-7b-instruct.json", "eviction_eval_qwen2.5-7b-instruct.json", "Qwen 2.5 7B"),
    ("analysis_qwen3-8b.json", "eviction_eval_qwen3-8b.json", "Qwen3-8B"),
]


def fig_jlens_signal() -> None:
    # Offline fact-AUC of eviction signals on both models. Full 0-1 axis with
    # the 0.5 chance line so the bar geometry cannot exaggerate the gap.
    series = [
        ("recency", C_RECENCY, lambda a: a["fact_auc"]["recency"]),
        ("SnapKV", C_SNAPKV, lambda a: a["fact_auc"]["snapkv"]),
        ("workspace", C_WORKSPACE, lambda a: a["fact_auc"][a["best_workspace_signal"]]),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    x = [0.0, 1.0]
    w = 0.26
    for i, (label, color, get) in enumerate(series):
        vals = [get(_load(af)) for af, _, _ in JLENS_MODELS]
        pos = [xi + (i - 1) * (w + 0.02) for xi in x]
        ax.bar(pos, vals, width=w, color=color, label=label, zorder=3)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xlim(-0.5, 1.78)
    ax.text(1.76, 0.512, "chance", fontsize=7, color="#666666", va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([m for _, _, m in JLENS_MODELS], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fact block AUC")
    ax.set_title("Which signal ranks the fact block highly?")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=1)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.5, zorder=0)
    fig.savefig(OUT / "jlens_signal_auc.pdf")
    plt.close(fig)


def fig_jlens_eviction() -> None:
    # Probe accuracy after masked eviction at three retention budgets, per
    # signal, both models; Wilson 95% bands. The workspace ranking keeps the
    # fact answerable at 25% where attention-history signals collapse.
    budgets = ["0.25", "0.5", "0.75"]
    xs = [25, 50, 75]
    series = [
        ("workspace", "workspace", C_WORKSPACE, "o"),
        ("SnapKV", "snapkv", C_SNAPKV, "s"),
        ("recency", "recency", C_RECENCY, "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, (_, ef, title) in zip(axes, JLENS_MODELS):
        res = _load(ef)["results"]
        for label, key, color, marker in series:
            rates = [100 * res[key][b]["rate"] for b in budgets]
            lo = [100 * res[key][b]["wilson95"][0] for b in budgets]
            hi = [100 * res[key][b]["wilson95"][1] for b in budgets]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.13, linewidth=0, zorder=2)
            ax.plot(xs, rates, color=color, marker=marker, markersize=4.5,
                    linewidth=1.6, label=label, zorder=3)
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels(["25%", "50%", "75%"])
        ax.set_xlabel("KV blocks retained")
        ax.set_xlim(18, 82)
        ax.set_ylim(0, 104)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.5, zorder=0)
    axes[0].set_ylabel("probe accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.savefig(OUT / "jlens_eviction.pdf")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = _load("session_evoke_metrics.json")
    fd = _load("figure_data.json")
    fig_recovery_fidelity(data)
    fig_scale_wall(data)
    fig_infllm_budget(fd)
    fig_crossarch(fd)
    fig_jlens_signal()
    fig_jlens_eviction()
    print(f"wrote: {sorted(p.name for p in OUT.glob('*.pdf'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
