"""Sharper differential bench for the attention scorer (#43).

Replicates the agent_bench planted-fact scenario but with one structural
change: every filler turn references the planted fact in its user message
("given our retry policy, ..."). Recency-based scoring still discards the
old config.py block under budget pressure. Attention-based scoring should
see each filler's user-message tokens attending back to the config block
and keep it alive. Probe at the end measures whether each strategy
preserved the fact.

This is the workload where attention-vs-heuristic gap should widen across
budgets, not just at the tightest one.

Requires LLAMA_CPP_LIB + EVOKE_MODEL_PATH. Run on the GPU eval host.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.attention_scorer import AttentionScorer
from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager

FACT_KEY = "file:config.py"
EXPECTED = "17"

_SYSTEM = (
    "You are an autonomous coding assistant. Inspect files before editing "
    "and reason concisely. The retry policy is defined in config.py."
)
_FACT = (
    "config.py: central application configuration. "
    "The maximum retry limit is set to 17 attempts. "
    "Retry policy applies to all RPC calls."
)
_FILLER = (
    "This module wires together service components, integrating with the "
    "central retry policy from config.py. It uses small composable "
    "functions and the standard retry semantics."
)


@dataclass
class ContextItem:
    key: str
    text: str


def build_session() -> list[ContextItem]:
    items = [
        ContextItem("system", _SYSTEM),
        ContextItem(FACT_KEY, _FACT),
    ]
    for name in [
        "database",
        "auth",
        "handlers",
        "models",
        "cache",
        "router",
        "metrics",
        "serializer",
        "validators",
        "tasks",
    ]:
        # Each file's content explicitly references the retry policy, so
        # the model's attention has a reason to query back to the config
        # block on every turn.
        items.append(
            ContextItem(
                f"file:{name}.py",
                f"{name}.py: using the retry policy from config.py — {_FILLER * 5}",
            )
        )
    return items


PROBE = (
    "\n\nQuestion: looking at the retry policy in config.py, what is the "
    "maximum retry limit?\nAnswer:"
)

STRATEGIES: dict[str, dict] = {
    "evoke_kv_restore": dict(recovery_mode="kv_restore"),
    "evoke_attention": dict(
        recovery_mode="kv_restore",
        w_attention=0.5,
        w_recency=0.2,
        w_coherence=0.3,
    ),
}


@dataclass
class Result:
    strategy: str
    probe_ok: bool
    evictions: int
    recoveries: int
    active_tokens: int
    answer: str


def run_strategy(
    engine: LlamaCppEngine, name: str, overrides: dict, budget: int
) -> Result:
    engine.reset()
    config = EvokeConfig(
        max_active_tokens=budget,
        block_size=64,
        high_watermark=0.95,
        low_watermark=0.75,
        **overrides,
    )
    attn_scorer: AttentionScorer | None = None
    if config.w_attention > 0 and engine.supports_kv_block:
        attn_scorer = AttentionScorer(
            engine,
            layer=config.attention_capture_layer,
            n_window=config.attention_window,
            decay=config.attention_decay,
        )
    mgr = EvokeManager(engine, config, attention_scorer=attn_scorer)
    session = build_session()
    for item in session:
        mgr.add_context(item.text, item.key)
    mgr.process_user_message(PROBE)
    answer = mgr.generate(32)
    stats = mgr.get_stats()
    return Result(
        strategy=name,
        probe_ok=EXPECTED in answer,
        evictions=stats.total_evictions,
        recoveries=stats.total_recoveries,
        active_tokens=stats.active_tokens,
        answer=answer.strip().replace("\n", " ")[:60],
    )


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("set EVOKE_MODEL_PATH")
        return 1
    budgets = [
        int(b) for b in os.environ.get("EVOKE_BUDGETS", "512,1024,2048").split(",")
    ]
    engine = LlamaCppEngine(model, n_ctx=16384, n_gpu_layers=-1, verbose=False)
    print(f"keepalive bench | model={Path(model).stem}")
    header = (
        f"{'budget':<8}{'strategy':<18}{'probe':<7}{'evict':<7}"
        f"{'recov':<7}{'active':<8}answer"
    )
    print(header)
    print("-" * len(header))
    try:
        for budget in budgets:
            for name, overrides in STRATEGIES.items():
                r = run_strategy(engine, name, overrides, budget)
                mark = "PASS" if r.probe_ok else "fail"
                print(
                    f"{budget:<8}{r.strategy:<18}{mark:<7}{r.evictions:<7}"
                    f"{r.recoveries:<7}{r.active_tokens:<8}{r.answer!r}"
                )
            print("-" * len(header))
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
