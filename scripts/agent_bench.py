"""Agentic eval for EVOKE.

Simulates an agent working a coding task: it accumulates context under a fixed
KV budget (a system prompt, an early config file, then many unrelated file
reads), then is probed on a fact from the config file that was almost certainly
evicted. Compares context-management strategies on whether the fact survives,
and at what recovery cost.

Requires EVOKE_MODEL_PATH. The evoke_kv_restore strategy additionally requires
LLAMA_CPP_LIB to point at the EVOKE llama.cpp build.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager

FACT_KEY = "file:config.py"
EXPECTED = "17"
PROBE = "\n\nQuestion: what is the maximum retry limit set in config.py?\nAnswer:"

_SYSTEM = (
    "You are an autonomous coding assistant working inside a software "
    "repository. Inspect files before editing them, make minimal correct "
    "changes, run the test suite after every change, and explain your "
    "reasoning concisely. Never invent file contents you have not read. "
)

_FACT = (
    "config.py: central application configuration. "
    "The maximum retry limit is set to 17 attempts. "
    "Connection timeouts and pool sizes are also defined in this file. "
)

_FILLER = (
    "This module wires together service components. It defines helpers for "
    "request parsing, response shaping, and error propagation, and favours "
    "small composable functions over deep inheritance hierarchies. "
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
        items.append(ContextItem(f"file:{name}.py", f"{name}.py: " + _FILLER * 6))
    return items


STRATEGIES: dict[str, dict] = {
    "recency": dict(
        w_recency=1.0, w_coherence=0.0, sink_count=0, recovery_mode="discard"
    ),
    "streaming_llm": dict(
        w_recency=1.0, w_coherence=0.0, sink_count=4, recovery_mode="discard"
    ),
    "evoke_discard": dict(recovery_mode="discard"),
    "evoke_breadcrumb": dict(recovery_mode="breadcrumb"),
    "evoke_kv_restore": dict(recovery_mode="kv_restore"),
}


@dataclass
class Result:
    strategy: str
    probe_ok: bool
    evictions: int
    recoveries: int
    active_tokens: int
    recovery_ms: float
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
    mgr = EvokeManager(engine, config)

    session = build_session()
    fact_text = next(item.text for item in session if item.key == FACT_KEY)
    for item in session:
        mgr.add_context(item.text, item.key)

    mode = overrides.get("recovery_mode")
    recovery_ms = 0.0
    if mode == "kv_restore":
        start = time.perf_counter()
        for crumb in mgr.get_breadcrumbs():
            if crumb.key.startswith(FACT_KEY):
                mgr.recover(crumb.key)
        recovery_ms = (time.perf_counter() - start) * 1000.0
    elif mode == "breadcrumb":
        if any(c.key.startswith(FACT_KEY) for c in mgr.get_breadcrumbs()):
            start = time.perf_counter()
            mgr.add_context(fact_text, FACT_KEY + ":reread")
            recovery_ms = (time.perf_counter() - start) * 1000.0

    mgr.process_user_message(PROBE)
    answer = mgr.generate(32)
    stats = mgr.get_stats()
    return Result(
        strategy=name,
        probe_ok=EXPECTED in answer,
        evictions=stats.total_evictions,
        recoveries=stats.total_recoveries,
        active_tokens=stats.active_tokens,
        recovery_ms=recovery_ms,
        answer=answer.strip().replace("\n", " ")[:58],
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
    print(f"agentic eval | model={Path(model).stem}")
    print(f"kv_block primitives available: {engine.supports_kv_block}")
    header = (
        f"{'budget':<8}{'strategy':<18}{'probe':<7}{'evict':<7}"
        f"{'recov':<7}{'active':<8}{'rec_ms':<10}answer"
    )
    print(header)
    print("-" * len(header))
    try:
        for budget in budgets:
            for name, overrides in STRATEGIES.items():
                if name == "evoke_kv_restore" and not engine.supports_kv_block:
                    print(f"{budget:<8}{name:<18}SKIP (no LLAMA_CPP_LIB)")
                    continue
                try:
                    r = run_strategy(engine, name, overrides, budget)
                    mark = "PASS" if r.probe_ok else "fail"
                    print(
                        f"{budget:<8}{r.strategy:<18}{mark:<7}{r.evictions:<7}"
                        f"{r.recoveries:<7}{r.active_tokens:<8}"
                        f"{r.recovery_ms:<10.2f}{r.answer!r}"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"{budget:<8}{name:<18}ERROR: {exc}")
            print("-" * len(header))
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
