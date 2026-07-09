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

from evoke.attention_scorer import AttentionScorer
from evoke.config import EvokeConfig
from evoke.jlens_scorer import JLensScorer
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
    # Full-context baseline: eviction never fires because the cap is bigger
    # than any session this bench produces. Pairs with kv_quant runs to ask
    # "keep everything at quarter precision" vs evoke's "keep a quarter at
    # full precision".
    "no_eviction": dict(
        w_recency=1.0,
        w_coherence=0.0,
        sink_count=0,
        recovery_mode="discard",
        max_active_tokens=131072,
        eviction_policy="watermark",
        high_watermark=1.0,
        low_watermark=1.0,
    ),
    "recency": dict(w_recency=1.0, w_coherence=0.0, sink_count=0, recovery_mode="discard"),
    "streaming_llm": dict(w_recency=1.0, w_coherence=0.0, sink_count=4, recovery_mode="discard"),
    "evoke_discard": dict(recovery_mode="discard"),
    "evoke_breadcrumb": dict(recovery_mode="breadcrumb"),
    "evoke_kv_restore": dict(recovery_mode="kv_restore"),
    # Attention-driven scorer: replaces the recency+coherence heuristic with
    # the model's actual attention weights as the dominant signal. Requires
    # the EVOKE fork (LLAMA_CPP_LIB). See paper §3.3 / §4.
    "evoke_attention": dict(
        recovery_mode="kv_restore", w_attention=0.5, w_recency=0.2, w_coherence=0.3
    ),
    # H2O baseline (arXiv:2306.14048) reimplemented atop EVOKE's AttentionScorer
    # infrastructure. Heavy-hitter selection uses lifetime cumulative attention
    # mass per block (no decay), the last 10% of cache budget is unconditionally
    # protected as the recent window R, and eviction fires at the hard budget
    # threshold with no watermark slack so the cache settles at H2O's
    # equilibrium of top-K-by-cumulative survivors. Recovery is "discard"
    # because H2O has no recovery story; an evicted token is gone for good.
    "h2o": dict(
        recovery_mode="discard",
        w_attention=1.0,
        w_recency=0.0,
        w_coherence=0.0,
        attention_score_mode="cumulative",
        recent_tail_protect_frac=0.1,
        eviction_policy="hard",
    ),
    # SnapKV baseline (Liu et al., NeurIPS 2024). Reuses EVOKE's AttentionScorer
    # with a one-shot snapshot of last-W-token attention frozen at the end of
    # process_user_message; eviction during the turn keeps the top-K blocks by
    # that snapshot. No recovery (SnapKV cannot bring evicted tokens back).
    # See paper §7.3 for the head-to-head; matches H2O's hard-eviction and
    # tail-guard pattern.
    "snapkv": dict(
        recovery_mode="discard",
        w_attention=1.0,
        w_recency=0.0,
        w_coherence=0.0,
        attention_score_mode="snapkv",
        snapkv_observation_window=32,
        recent_tail_protect_frac=0.05,
        eviction_policy="hard",
    ),
    # InfLLM (Xiao et al., NeurIPS 2024) adapted onto EVOKE's kv_restore +
    # smart-recovery infrastructure. See the niah_bench STRATEGIES entry
    # for the full implementation note. In agent_bench the oracle
    # recovery path (recover the fact's known key) substitutes for
    # similarity-based retrieval — this isolates the primitive cost (kv
    # splice latency) shared with evoke_kv_restore. The InfLLM-specific
    # value here is the aggressive eviction footprint that exposes
    # whether the local-window approximation still finds the planted
    # block via the oracle recover call.
    "infllm": dict(
        recovery_mode="kv_restore",
        w_recency=0.0,
        w_coherence=0.0,
        w_attention=0.0,
        sink_count=4,
        smart_recover_k=8,
        use_retrieval_embeddings=True,
        block_embedding_strategy="mean",
        eviction_policy="watermark",
        high_watermark=0.5,
        low_watermark=0.3,
        recent_tail_protect_frac=0.25,
    ),
    # J-lens workspace signal (j-space phase 3): a distilled ridge probe over
    # residuals captured at prefill predicts which blocks hold workspace
    # content the model will later read from (offline fact-AUC 0.891 vs
    # SnapKV 0.622 on Qwen2.5-7B). Content-based and forward-looking, so the
    # score exists before any decode history. Discard recovery and the same
    # hard-eviction + tail-guard pattern as h2o/snapkv for a clean head-to-
    # head. Requires the fork's residual export plus EVOKE_JLENS_PROBE.
    "jlens": dict(
        recovery_mode="discard",
        w_recency=0.0,
        w_coherence=0.0,
        w_jlens=1.0,
        recent_tail_protect_frac=0.1,
        eviction_policy="hard",
    ),
    # Workspace signal mixed with H2O's cumulative attention at equal
    # weight: tests whether the content signal adds lift over the
    # attention-history heavy hitters rather than replacing them.
    "jlens_h2o": dict(
        recovery_mode="discard",
        w_recency=0.0,
        w_coherence=0.0,
        w_attention=0.5,
        w_jlens=0.5,
        attention_score_mode="cumulative",
        recent_tail_protect_frac=0.1,
        eviction_policy="hard",
    ),
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


def run_strategy(engine: LlamaCppEngine, name: str, overrides: dict, budget: int) -> Result:
    engine.reset()
    config_kwargs: dict = dict(
        max_active_tokens=budget,
        block_size=64,
        high_watermark=0.95,
        low_watermark=0.75,
    )
    config_kwargs.update(overrides)
    config = EvokeConfig(**config_kwargs)
    attn_scorer: AttentionScorer | None = None
    if config.w_attention > 0 and engine.supports_kv_block:
        attn_scorer = AttentionScorer(
            engine,
            layer=config.attention_capture_layer,
            n_window=config.attention_window,
            decay=config.attention_decay,
            score_mode=config.attention_score_mode,
            snapkv_observation_window=config.snapkv_observation_window,
        )
    jlens_scorer: JLensScorer | None = None
    if config.w_jlens > 0:
        # Fail loud rather than fall back: a jlens run scored by recency
        # would silently mislabel the strategy, same rationale as the
        # needs_kv_block SKIP in main().
        probe = os.environ.get("EVOKE_JLENS_PROBE", "")
        if not probe:
            raise RuntimeError("set EVOKE_JLENS_PROBE to the probe artifact npz")
        layers_env = os.environ.get("EVOKE_JLENS_LAYERS", "")
        jlens_scorer = JLensScorer(
            engine,
            probe_path=probe,
            layers=[int(x) for x in layers_env.split(",")] if layers_env else None,
        )
    mgr = EvokeManager(engine, config, attention_scorer=attn_scorer, jlens_scorer=jlens_scorer)

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
    budgets = [int(b) for b in os.environ.get("EVOKE_BUDGETS", "512,1024,2048").split(",")]

    kv_quant = os.environ.get("EVOKE_KV_QUANT", "").lower().strip()
    engine_kwargs: dict = {}
    if kv_quant and kv_quant not in ("f16", "none"):
        engine_kwargs["type_k"] = kv_quant
        engine_kwargs["type_v"] = kv_quant
    engine = LlamaCppEngine(model, n_ctx=16384, n_gpu_layers=-1, verbose=False, **engine_kwargs)
    print(f"agentic eval | model={Path(model).stem}")
    if kv_quant and kv_quant not in ("f16", "none"):
        print(f"kv cache quantization: type_k=type_v={kv_quant}")
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
                needs_kv_block = name in (
                    "evoke_kv_restore",
                    "evoke_recovery_aware",
                    "evoke_attention",
                    "h2o",
                    "snapkv",
                    "infllm",
                    "jlens",
                    "jlens_h2o",
                )
                if needs_kv_block and not engine.supports_kv_block:
                    # These baselines all depend on the fork's attention capture
                    # or kv_block splice primitives. Without LLAMA_CPP_LIB they
                    # would silently fall back to recency, which would mislabel
                    # the run as the named baseline.
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
