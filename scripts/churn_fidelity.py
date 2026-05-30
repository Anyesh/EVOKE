"""Decisive isolation: does recovered KV degrade under compact-eviction CHURN?

recovery_usability showed recovery = re-decode = 100% with ONE eviction (no churn). scale_demo
showed recovery (10/16) < re-decode (16/16) under heavy churn. This isolates churn cleanly: a
target fact is placed AFTER several "before" distractors so that, as the tight budget evicts the
before-distractors, the target accumulates many seq_add position shifts before it is itself
evicted (replicating a mid-corpus section). The target is then restored per condition and probed.
Compare recall RATES and raw answers across conditions over N distinct facts.

  ref        budget huge, nothing evicted (ceiling: model can read the target in place)
  redecode   tight budget + churn, target re-decoded fresh before the probe (re-presents text)
  recovered  tight budget + churn, target's saved KV spliced back (compact, re-anchored to tail)
  discard    tight budget + churn, target not restored (floor / probe-sensitivity control)

Verdict:
  recovered ~= redecode  -> recovery is FAITHFUL under churn; the scale gap is the re-presentation
                            effect (re-decode re-shows the text), not a recovery defect.
  recovered <<  redecode -> recovery DEGRADES under churn -> a real fork fidelity bug to fix.

Requires EVOKE_MODEL_PATH and LLAMA_CPP_LIB (the fork build).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager

CONDITIONS = ("ref", "redecode", "recovered", "discard")
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
SYSTEM = "You are a configuration lookup assistant. Answer each question with only the number."
FILLER = (
    "This auxiliary module handles an unrelated background pathway and is documented here only "
    "to occupy working memory; none of its details bear on the queried value. "
)


def make_facts(n: int) -> list[tuple[str, int]]:
    return [
        (f"CFG_{chr(65 + i % 26)}{chr(65 + (i // 26) % 26)}_{i:02d}", 1000 + i * 7)
        for i in range(n)
    ]


def _target_text(key: str, value: int) -> str:
    return (
        f"Primary setting block for {key}.\n"
        f"# {key} is read by the service at startup and on every reload.\n"
        f"{key} = {value}\n"
        f"Operators must restart workers after changing {key}.\n"
    )


def _distractor(j: int, tokens_each: int, engine: LlamaCppEngine) -> str:
    body = f"Auxiliary note {j}: AUX_{j:03d} = {500 + j}.\n" + FILLER
    while len(engine.tokenize(body)) < tokens_each:
        body += FILLER
    return body


def _config(condition: str, budget: int, n_ctx: int) -> EvokeConfig:
    huge = condition == "ref"
    return EvokeConfig(
        max_active_tokens=n_ctx * 100 if huge else budget,
        block_size=64,
        sink_count=0,
        recovery_mode="kv_restore" if condition == "recovered" else "discard",
        position_mode="compact",
        high_watermark=0.999 if huge else 0.92,
        low_watermark=0.99 if huge else 0.70,
        w_recovery=1.0,
        recovery_strength_init=1.0,
        recovery_decay=0.7,
    )


def _resident(mgr: EvokeManager) -> bool:
    return any(b.key.startswith("target#") for b in mgr._positions.active_blocks)


def _evicted_keys(mgr: EvokeManager) -> list[str]:
    return [c.key for c in mgr.get_breadcrumbs() if c.key.startswith("target#")]


def run_fact(
    engine: LlamaCppEngine,
    key: str,
    value: int,
    condition: str,
    n_ctx: int,
    budget: int,
    n_before: int,
    n_after: int,
    tok_each: int,
    q_stop: set[int],
) -> tuple[bool, str, bool]:
    engine.reset()
    mgr = EvokeManager(engine, _config(condition, budget, n_ctx))
    mgr.add_context(f"{IM_START}system\n{SYSTEM}{IM_END}\n", key="sys", pinned=True)

    for j in range(n_before):
        mgr.add_context(_distractor(j, tok_each, engine), key=f"before{j}")
    mgr.add_context(_target_text(key, value), key="target")
    for j in range(n_after):
        mgr.add_context(_distractor(1000 + j, tok_each, engine), key=f"after{j}")

    if condition != "ref" and not _resident(mgr):
        ek = _evicted_keys(mgr)
        if condition == "recovered" and ek:
            for k in ek:
                mgr.recover(k, defer_budget=True)
        elif condition == "redecode":
            mgr.add_context(_target_text(key, value), key="target_re")
        # discard: leave the hole

    target_present = _resident(mgr)
    mgr.add_context(
        f"\n{IM_START}user\nWhat exact value is assigned to {key}? "
        f"Answer with only the number.{IM_END}\n{IM_START}assistant\n",
        key="q",
    )
    ans = mgr.generate(
        64,
        stop_token_ids=q_stop,
        think_close="</think>",
        thinking_budget=512,
        answer_budget=32,
    )
    correct = str(value) in ans.replace(",", "").replace(" ", "")
    safe = ans.encode("ascii", "replace").decode("ascii")[:80]
    return correct, safe, target_present


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--n-facts", type=int, default=20)
    ap.add_argument("--n-before", type=int, default=24)
    ap.add_argument("--n-after", type=int, default=40)
    ap.add_argument("--tok-each", type=int, default=80)
    args = ap.parse_args()

    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("FAIL: set EVOKE_MODEL_PATH")
        return 1

    engine = LlamaCppEngine(model, n_ctx=args.n_ctx, n_gpu_layers=-1, verbose=False)
    if not engine.supports_kv_block:
        print(
            "FAIL: kv_block primitives not bound -- set LLAMA_CPP_LIB to the fork build"
        )
        return 1

    q_stop = set(engine.tokenize(IM_END))
    facts = make_facts(args.n_facts)
    results: dict[str, list[bool]] = {c: [] for c in CONDITIONS}
    present: dict[str, int] = {c: 0 for c in CONDITIONS}
    samples: dict[str, list[str]] = {c: [] for c in CONDITIONS}

    for key, value in facts:
        for cond in CONDITIONS:
            ok, ans, tgt = run_fact(
                engine,
                key,
                value,
                cond,
                args.n_ctx,
                args.budget,
                args.n_before,
                args.n_after,
                args.tok_each,
                q_stop,
            )
            results[cond].append(ok)
            present[cond] += int(tgt)
            if len(samples[cond]) < 4:
                samples[cond].append(f"{key}={value}: {ans!r}")

    churn_blocks = (args.n_before + args.n_after) * (args.tok_each // 64 + 1)
    print(
        f"\nCHURN FIDELITY  n_facts={len(facts)} budget={args.budget} n_ctx={args.n_ctx} "
        f"(before={args.n_before} after={args.n_after}, ~{churn_blocks} blocks churned)"
    )
    print(f"{'condition':12s} {'correct':>8s} {'rate':>7s} {'tgt_present':>12s}")
    for cond in CONDITIONS:
        n_ok = sum(results[cond])
        print(
            f"{cond:12s} {n_ok:>3d}/{len(facts):<3d} {n_ok / len(facts):>6.1%} "
            f"{present[cond]:>9d}/{len(facts)}"
        )

    rec = sum(results["recovered"]) / len(facts)
    red = sum(results["redecode"]) / len(facts)
    dis = sum(results["discard"]) / len(facts)
    print()
    for cond in CONDITIONS:
        print(f"  sample [{cond}]:")
        for s in samples[cond]:
            print(f"    {s}")
    print()
    if dis >= red - 1e-9:
        print(
            "INCONCLUSIVE: discard not below redecode -- probe insensitive / target not evicted."
        )
        return 1
    gap = red - rec
    if gap <= 0.05:
        print(
            f"FAITHFUL UNDER CHURN: recovered ({rec:.0%}) ~= redecode ({red:.0%}). Recovery is not "
            "degraded by churn; the scale recall gap is the re-presentation effect, not a fork bug."
        )
    else:
        print(
            f"DEGRADES UNDER CHURN: redecode ({red:.0%}) exceeds recovered ({rec:.0%}) by {gap:.0%}. "
            "Recovered KV is a worse recall substrate under churn -- a real fork fidelity bug to fix."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
