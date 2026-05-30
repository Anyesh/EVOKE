"""Experiment 1: hybrid attention-only in-place recovery fidelity (Qwen3.5-9B).

On a hybrid (SSM/GDN + attention) model the recurrent state is a fixed-size fold
that does not grow with context, so the attention KV is the only memory worth
reclaiming. This tests whether removing a mid-context block's attention KV and
splicing it back in place, with the recurrent state left untouched, reproduces
the model's behavior, measured by greedy continuation against an undisturbed run.

Arms (greedy/argmax, identical content):

  REF      block never evicted
  SPARSE   block's attention KV evicted via seq_rm_attention_only, then restored
           in place at its original position via kv_block_load; SSM state untouched
  DISCARD  block's attention KV evicted, never restored (sensitivity control)

PASS: SPARSE == REF token-for-token (attention-only in-place recovery is faithful
on a hybrid, disproving the standing assumption that it would break decode) AND
DISCARD != REF (the probe depends on the block).

Requires EVOKE_MODEL_PATH (a hybrid GGUF, e.g. Qwen3.5-9B) and LLAMA_CPP_LIB
pointing at the fork build with the kv_block primitives.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.llama_engine import LlamaCppEngine

BACKGROUND = "General project background and routine status notes. " * 40
TARGET = (
    "Internal migration memo. The migration is led by Priya Nadkarni. "
    "The cutover date is the fourteenth of November. The rollback owner is "
    "Tomas Eklund. The budget ceiling is 4.2 million dollars. Treat these as authoritative."
)
FILLER = "Unrelated implementation detail about the build pipeline. " * 60
PROBE = (
    "\n\nQuestion: From the internal migration memo, who leads the migration, "
    "what is the cutover date, who owns the rollback, and what is the budget ceiling?\nAnswer:"
)

MODES = ("ref", "sparse", "discard")


def _greedy_ids(engine: LlamaCppEngine, n: int) -> list[int]:
    eos = engine.eos_token
    out: list[int] = []
    for _ in range(n):
        if engine.next_write_pos + 1 >= engine.n_ctx:
            break
        tok = engine.generate_next()
        out.append(tok)
        if tok == eos:
            break
    return out


def run_arm(engine: LlamaCppEngine, mode: str, n_gen: int) -> list[int]:
    engine.reset()
    engine.process_tokens(engine.tokenize(BACKGROUND))
    target_p0 = engine.next_write_pos
    engine.process_tokens(engine.tokenize(TARGET))
    target_p1 = engine.next_write_pos
    engine.process_tokens(engine.tokenize(FILLER))

    if mode != "ref":
        saved = engine.kv_block_save(target_p0, target_p1)
        if not engine.seq_rm_attention_only(target_p0, target_p1):
            raise SystemExit(f"FAIL[{mode}]: seq_rm_attention_only returned False")
        if mode == "sparse" and not engine.kv_block_load(saved, target_p0):
            raise SystemExit("FAIL[sparse]: kv_block_load returned False")

    engine.process_tokens(engine.tokenize(PROBE))
    return _greedy_ids(engine, n_gen)


def _match_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--gen-tokens", type=int, default=48)
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

    results: dict[str, list[int]] = {}
    for mode in MODES:
        results[mode] = run_arm(engine, mode, args.gen_tokens)
        text = (
            engine.detokenize(results[mode]).encode("ascii", "replace").decode("ascii")
        )
        print(f"{mode:8s}: {text[:140]!r}")

    ref = results["ref"]
    print(f"\ngen_tokens = {args.gen_tokens}")
    for mode in ("sparse", "discard"):
        m = _match_prefix(results[mode], ref)
        print(
            f"{mode:8s} vs ref: prefix_match={m}/{len(ref)} exact={results[mode] == ref}"
        )

    sparse_faithful = results["sparse"] == ref
    sensitive = results["discard"] != ref

    print()
    if sparse_faithful and sensitive:
        print(
            "FIDELITY PASS: attention-only in-place recovery reproduces REF token-for-token "
            "on the hybrid (SSM state untouched), and DISCARD diverges. The recompute-free "
            "in-place eviction path is correct on a hybrid model."
        )
        return 0
    if sparse_faithful and not sensitive:
        print(
            "INCONCLUSIVE: SPARSE==REF but DISCARD==REF too -- the probe does not depend on "
            "the target block, so it cannot confirm fidelity. Strengthen TARGET/PROBE."
        )
        return 1
    print(
        "FIDELITY FAIL: SPARSE diverges from REF -- attention-only seq_rm + in-place restore "
        "desyncs the hybrid recurrent/attention position state, as the old evict_ranges comment "
        "feared. Investigate the desync and fix it; do not accept it as a limitation."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
