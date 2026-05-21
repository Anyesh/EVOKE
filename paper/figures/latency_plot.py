"""Generate the latency plot for paper Section 7.1.

Data is the verbatim profile_recover.py output reproduced in the paper.
The plot shows save / load / re-prefill on a log-log axis so the linear
scaling of re-prefill (model_FLOPs * tokens) and the much shallower
scaling of save+load (bytes * tokens) are visually obvious.

Output: latency.pdf (vector) in the same directory.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

n_tok = np.array([20, 40, 80, 160, 320, 640, 1280])
save_ms = np.array([1.10, 1.61, 2.76, 4.69, 8.40, 16.37, 31.90])
load_ms = np.array([0.48, 0.70, 0.89, 1.50, 2.41, 4.34, 7.25])
reprefill_ms = np.array([11.90, 13.78, 19.00, 32.60, 59.72, 118.36, 232.18])

fig, ax = plt.subplots(figsize=(5.8, 3.4))

ax.plot(
    n_tok,
    reprefill_ms,
    marker="o",
    linewidth=1.8,
    color="#c0504d",
    label="re-prefill (forward pass)",
)
ax.plot(
    n_tok,
    save_ms,
    marker="s",
    linewidth=1.6,
    linestyle="--",
    color="#4f81bd",
    label=r"$\mathtt{kv\_block\_save}$",
)
ax.plot(
    n_tok,
    load_ms,
    marker="^",
    linewidth=1.8,
    color="#2ca02c",
    label=r"$\mathtt{kv\_block\_load}$ (recovery)",
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Block size (tokens)")
ax.set_ylabel("Time (ms)")
ax.set_xticks(n_tok)
ax.set_xticklabels([str(n) for n in n_tok])
ax.set_xlim(18, 1500)
ax.grid(True, which="major", alpha=0.3, linestyle=":")
ax.grid(True, which="minor", alpha=0.15, linestyle=":")
ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9)

# Annotate the speedup at the widest end.
ax.annotate(
    r"$32\times$ at 1280 tokens",
    xy=(1280, 7.25),
    xytext=(380, 1.4),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    color="#2ca02c",
)

plt.tight_layout()
plt.savefig("latency.pdf", bbox_inches="tight")
plt.savefig("latency.png", bbox_inches="tight", dpi=150)
print("wrote latency.pdf and latency.png")
