"""Stateful agent-loop demo: EVOKE as working memory, driven via EvokeManager.

Unlike the HTTP proxy (which the OpenAI protocol forces into full-resend, where EVOKE
degenerates), this drives EvokeManager directly as an agent's working memory: the
session is held in the KV cache, each file is appended once as a keyed delta, the
budget evicts cold files, and when the agent re-references an evicted file the manager
splices its KV back by identity (recompute-free) instead of re-reading it.

Three arms on the same trace; the proof is the contrast, not correctness alone:
  evoke        sparse + kv_restore + tight budget: in-budget AND recompute-free re-ref
  no_eviction  budget = n_ctx: cheap decode but peak active = whole codebase (over budget)
  no_recovery  discard + tight budget: in-budget but re-reference must re-read (re-decode)

Reads the Tasklet repo in scripts/demo_webapp/ (config.py holds the probe facts
MAX_TODOS_PER_USER=17, SESSION_TIMEOUT_MINUTES=45). Requires EVOKE_MODEL_PATH and
LLAMA_CPP_LIB (the fork).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager

HERE = Path(__file__).resolve().parent
PROJECT = HERE / "demo_webapp"
FILES = ["config.py", "models.py", "storage.py", "app.py", "README.md"]
# The agent needs config.py to answer, so it re-references that one file right
# before the probe: evoke recovers it (free), no_recovery re-reads it (re-decode),
# no_eviction still has it resident. Correctness is then held equal across arms and
# the contrast is budget x decode.
REREF = ["config.py"]
PROBE = (
    "From the config.py you reviewed, what is the numeric value of MAX_TODOS_PER_USER "
    "and of SESSION_TIMEOUT_MINUTES? Answer in one sentence."
)
EXPECT = ("17", "45")


def _read(name: str) -> str:
    return (PROJECT / name).read_text(encoding="utf-8")


def _is_resident(mgr: EvokeManager, key: str) -> bool:
    return any(b.key.startswith(f"{key}#") for b in mgr._positions.active_blocks)


def _evicted_keys(mgr: EvokeManager, key: str) -> list[str]:
    return [c.key for c in mgr.get_breadcrumbs() if c.key.startswith(f"{key}#")]


def _config(
    arm: str, budget: int, n_ctx: int, protect: float = 0.0, decay: float = 0.7
) -> EvokeConfig:
    if arm == "no_eviction":
        return EvokeConfig(
            max_active_tokens=n_ctx,
            block_size=64,
            sink_count=0,
            high_watermark=0.999,
            low_watermark=0.99,
            recovery_mode="discard",
            position_mode="compact",
        )
    common = dict(
        max_active_tokens=budget,
        block_size=64,
        sink_count=0,
        high_watermark=0.92,
        low_watermark=0.70,
    )
    if arm == "no_recovery":
        return EvokeConfig(recovery_mode="discard", position_mode="sparse", **common)
    return EvokeConfig(  # evoke
        recovery_mode="kv_restore",
        position_mode="sparse",
        recovery_match="identity",
        w_recovery=1.0,
        recovery_strength_init=1.0,
        recovery_protect_threshold=protect,
        recovery_decay=decay,
        **common,
    )


def _reference(
    mgr: EvokeManager, engine: LlamaCppEngine, name: str, content: str
) -> int:
    # The agent needs `name` now. Resident -> free. Evicted + kv_restore -> splice
    # its saved KV back (recompute-free, 0 decoded). Evicted + discard -> re-read
    # (re-decode). Returns tokens decoded.
    if _is_resident(mgr, name):
        return 0
    ek = _evicted_keys(mgr, name)
    if mgr._config.recovery_mode == "kv_restore" and ek:
        for k in ek:
            mgr.recover(k)
        return 0
    mgr.add_context(content, key=name)
    return len(engine.tokenize(content))


def run_arm(
    engine: LlamaCppEngine,
    arm: str,
    budget: int,
    n_ctx: int,
    gen: int,
    work_turns: int,
    protect: float = 0.0,
    decay: float = 0.7,
) -> dict:
    engine.reset()
    mgr = EvokeManager(engine, _config(arm, budget, n_ctx, protect, decay))
    decoded = 0
    contents = {name: _read(name) for name in FILES}

    # read the repo: each file appended once as a keyed delta
    for name in FILES:
        decoded += len(engine.tokenize(contents[name]))
        mgr.add_context(contents[name], key=name)

    # work phase: the agent revisits files as it works, re-referencing one file
    # (cycling) each turn. Cold ones must come back: evoke recovers (free),
    # no_recovery re-reads (re-decode). tick_turn decays prior recoveries so the
    # working set rotates and the recompute-free savings compound over the session.
    for i in range(work_turns):
        mgr.tick_turn()
        name = FILES[i % len(FILES)]
        decoded += _reference(mgr, engine, name, contents[name])

    # the agent needs config.py to answer; ensure it is resident, then probe
    decoded += _reference(mgr, engine, "config.py", contents["config.py"])
    mgr.process_user_message(PROBE)
    decoded += len(engine.tokenize(PROBE))
    answer = mgr.generate(gen)

    stats = mgr.get_stats()
    correct = all(t in answer for t in EXPECT)
    safe = answer.encode("ascii", "replace").decode("ascii")[:200]
    # How much of the final working set is pinned by the hard-protect (vs held
    # for recency/other)? If protected_tokens ~= final_active, the floor IS the
    # protection (reducible by tightening); if small, the floor is genuine.
    prot_blocks = [
        b
        for b in mgr._positions.active_blocks
        if b.recovery_strength >= protect and protect > 0.0
    ]
    return {
        "arm": arm,
        "budget": budget,
        "protect": protect,
        "decay": decay,
        "peak_active": mgr.peak_active_tokens,
        "final_active": stats.active_tokens,
        "protected_blocks": len(prot_blocks),
        "protected_tokens": sum(len(b.token_ids) for b in prot_blocks),
        "decoded": decoded,
        "evictions": stats.total_evictions,
        "recoveries": stats.total_recoveries,
        "correct": correct,
        "answer": safe,
    }


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("FAIL: set EVOKE_MODEL_PATH")
        return 1
    n_ctx = int(os.environ.get("LOOP_N_CTX", "4096"))
    budget = int(os.environ.get("LOOP_BUDGET", "512"))
    gen = int(os.environ.get("LOOP_GEN", "512"))
    work_turns = int(os.environ.get("LOOP_WORK_TURNS", "15"))

    engine = LlamaCppEngine(model, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
    if not engine.supports_kv_block:
        print("FAIL: kv_block primitives not bound -- set LLAMA_CPP_LIB")
        return 1

    if os.environ.get("LOOP_SWEEP"):
        # Floor investigation: vary the recovery protection at a fixed tight budget.
        # If a tighter protection (lower threshold / faster decay) drops final_active
        # while staying correct, the ~762 floor was hard-protect over-protection.
        # protected_tokens shows how much of the floor the protection itself pins.
        settings = [
            (0.0, 0.7),
            (0.3, 0.7),
            (0.5, 0.7),
            (0.5, 0.5),
            (0.5, 0.3),
            (0.8, 0.7),
        ]
        rows = [
            run_arm(engine, "evoke", budget, n_ctx, gen, work_turns, p, d)
            for p, d in settings
        ]
        rows.append(run_arm(engine, "no_recovery", budget, n_ctx, gen, work_turns))
        print(
            f"\nFLOOR SWEEP  budget={budget} n_ctx={n_ctx} work_turns={work_turns} "
            f"(codebase = {sum(len(engine.tokenize(_read(f))) for f in FILES)} tokens)"
        )
        print(
            f"{'arm':12s} {'protect':>7s} {'decay':>6s} {'final':>6s} {'prot_tok':>8s} "
            f"{'decoded':>8s} {'recov':>6s} {'correct':>8s}"
        )
        for r in rows:
            print(
                f"{r['arm']:12s} {r['protect']:>7.2f} {r['decay']:>6.2f} {r['final_active']:>6d} "
                f"{r['protected_tokens']:>8d} {r['decoded']:>8d} {r['recoveries']:>6d} {str(r['correct']):>8s}"
            )
        print(
            f"\n  (lowest final_active that is still correct = the true working-set floor)"
        )
        return 0

    rows = [
        run_arm(engine, arm, budget, n_ctx, gen, work_turns)
        for arm in ("evoke", "no_eviction", "no_recovery")
    ]

    print(
        f"\nbudget={budget} n_ctx={n_ctx}  (codebase = {sum(len(engine.tokenize(_read(f))) for f in FILES)} tokens)"
    )
    print(
        f"{'arm':12s} {'final_active':>12s} {'peak_active':>11s} {'over_budget':>11s} "
        f"{'decoded':>8s} {'evict':>6s} {'recov':>6s} {'correct':>8s}"
    )
    for r in rows:
        over = r["final_active"] > budget
        print(
            f"{r['arm']:12s} {r['final_active']:>12d} {r['peak_active']:>11d} {str(over):>11s} "
            f"{r['decoded']:>8d} {r['evictions']:>6d} {r['recoveries']:>6d} {str(r['correct']):>8s}"
        )
    print()
    for r in rows:
        print(f"  [{r['arm']}] answer: {r['answer']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
