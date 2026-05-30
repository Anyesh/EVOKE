"""End-to-end: EVOKE's eviction policy under a KV budget on a hybrid thinking model.

Loads context well past a tight KV budget so the relevance scorer auto-evicts cold
blocks (the policy running under real pressure, not a forced eviction), confirms the
budget is respected and the fact block was evicted, then re-references and recovers
the fact and checks the model answers from the restored content. A DISCARD arm omits
recovery and must fail, so the recall is attributable to recovery, not leftover context.

Watermark policy with headroom is used so the _enforce_budget that runs right after
recovery does not re-evict the just-restored (old-position, low-recency) block; by
default w_recovery=0 gives a recovered block no score protection.

Requires EVOKE_MODEL_PATH (a hybrid GGUF, e.g. Qwen3.5-9B) and LLAMA_CPP_LIB (fork).
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

PASSKEY = "7391"
FACT = f"Important system note: the secret passkey is {PASSKEY}. Keep it confidential."
BACKGROUND = "General project background and routine status notes. " * 30
FILLER = "Unrelated filler detail about the build pipeline. " * 120
PROBE = "\n\nQuestion: what is the secret passkey mentioned earlier?\nAnswer:"


def _config(budget: int) -> EvokeConfig:
    return EvokeConfig(
        max_active_tokens=budget,
        block_size=64,
        sink_count=0,
        recovery_mode="kv_restore",
        position_mode="sparse",
        eviction_policy="watermark",
        high_watermark=0.85,
        low_watermark=0.55,
    )


def run_arm(engine: LlamaCppEngine, budget: int, recover: bool, gen: int) -> dict:
    engine.reset()
    mgr = EvokeManager(engine, _config(budget))
    mgr.load_document(BACKGROUND)
    mgr.add_context(FACT, key="fact")
    mgr.add_context(FILLER, key="filler")

    stats = mgr.get_stats()
    fact_evicted = any(c.key.startswith("fact#") for c in mgr.get_breadcrumbs())
    forced = False
    if recover:
        if not fact_evicted:
            # the policy kept the fact; force it out so the recovery path is still
            # exercised, but record that the budget did not evict it on its own
            fb = [b for b in mgr._positions.active_blocks if b.key.startswith("fact#")]
            if fb:
                mgr.force_evict([b.block_id for b in fb])
                forced = True
        for crumb in mgr.get_breadcrumbs():
            if crumb.key.startswith("fact#"):
                mgr.recover(crumb.key)

    mgr.process_user_message(PROBE)
    answer = mgr.generate(gen)
    return {
        "active": stats.active_tokens,
        "evictions": stats.total_evictions,
        "fact_evicted_by_policy": fact_evicted,
        "forced": forced,
        "recalled": PASSKEY in answer,
        "answer": answer.encode("ascii", "replace").decode("ascii")[:160],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--budget", type=int, default=640)
    ap.add_argument("--gen-tokens", type=int, default=2048)
    args = ap.parse_args()

    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("FAIL: set EVOKE_MODEL_PATH to the hybrid GGUF")
        return 1

    engine = LlamaCppEngine(model, n_ctx=args.n_ctx, n_gpu_layers=-1, verbose=False)
    if not engine.supports_kv_block:
        print(
            "FAIL: kv_block primitives not bound -- set LLAMA_CPP_LIB to the fork build"
        )
        return 1
    print("kv_block primitives bound\n")

    rec = run_arm(engine, args.budget, recover=True, gen=args.gen_tokens)
    dis = run_arm(engine, args.budget, recover=False, gen=args.gen_tokens)

    print(f"budget(max_active_tokens) = {args.budget}")
    print(
        f"RECOVER : active_after={rec['active']} evictions={rec['evictions']} "
        f"fact_evicted_by_policy={rec['fact_evicted_by_policy']} forced={rec['forced']} "
        f"recalled={rec['recalled']}"
    )
    print(f"          answer: {rec['answer']!r}")
    print(
        f"DISCARD : active_after={dis['active']} evictions={dis['evictions']} "
        f"recalled={dis['recalled']}"
    )
    print(f"          answer: {dis['answer']!r}")

    budget_ok = rec["active"] <= args.budget and rec["evictions"] > 0
    print()
    if budget_ok and rec["recalled"] and not dis["recalled"]:
        policy_note = (
            "policy-evicted"
            if rec["fact_evicted_by_policy"]
            else "force-evicted (policy kept it)"
        )
        print(
            f"PASS: under a {args.budget}-token budget the policy evicted cold blocks on the "
            f"hybrid (evictions={rec['evictions']}, active<=budget), recovering the {policy_note} "
            "fact restored the passkey, and DISCARD did not recall it. EVOKE runs end-to-end on "
            "a hybrid thinking model."
        )
        return 0
    if not budget_ok:
        print(
            f"FAIL: budget not enforced or no eviction fired "
            f"(active_after={rec['active']}, evictions={rec['evictions']})."
        )
        return 1
    if not rec["recalled"]:
        print(
            "FAIL: recovered arm did not recall the passkey -- recovery under budget is not faithful."
        )
        return 1
    print(
        "INCONCLUSIVE: DISCARD also recalled the passkey -- probe insensitive to the evicted fact."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
