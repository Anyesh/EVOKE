"""PoC: principled thinking-trace removal on a HYBRID (Mamba/Attention) model.

The hybrid wall: `seq_rm` can't partially roll back the recurrent state, which
forced the `suppress_thinking_strip` hack (keep the think trace in context). The
fix: snapshot the full per-seq state before the think trace, then after the
answer restore the snapshot and replay only the answer tokens, so the recurrent
state is rolled back and re-advanced through the answer alone.

This proves the mechanism is state-correct on a hybrid model by comparing exact
GREEDY continuations across three paths:

  REF        process(C); process(A)                          -- think never existed
  MECH       process(C); snap; process(T); restore; process(A) -- think removed via snapshot+replay
  WITHTHINK  process(C); process(T); process(A)               -- think left in context (the hack)

If MECH's continuation == REF's exactly, the snapshot/restore cleanly removed T
from the hybrid state. WITHTHINK should DIFFER from REF (otherwise the test is
not sensitive to T). Greedy decode (the engine's default) makes this exact.

Requires EVOKE_MODEL_PATH=<Qwen3.5-9B hybrid GGUF>, LLAMA_CPP_LIB=<fork>.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.llama_engine import LlamaCppEngine

CONTEXT = (
    "Session notes. The project codename is Helios, the lead engineer is "
    "Dr. Vance, and the launch window is March. Keep these facts straight. "
)
THINK = (
    " Let me reason through an unrelated tangent at length: consider the tidal "
    "patterns of the north Pacific, the history of medieval cartography, the "
    "distribution of prime numbers, and a meandering aside about typography "
    "that has nothing to do with the project facts above, going on and on. "
)
ANSWER = " The project codename is Helios and the lead engineer is Dr. Vance. "
PROBE = " Question: who is the lead engineer and what is the codename? Answer:"


def _greedy_continuation(engine: LlamaCppEngine, n: int) -> list[int]:
    return [engine.generate_next() for _ in range(n)]


def run_path(engine: LlamaCppEngine, mode: str, n_gen: int) -> list[int]:
    engine.reset()
    engine.process_tokens(engine.tokenize(CONTEXT))
    if mode == "mech":
        snap = engine.state_save()
        engine.process_tokens(engine.tokenize(THINK))
        engine.state_restore(snap)
        engine.process_tokens(engine.tokenize(ANSWER))
    elif mode == "withthink":
        engine.process_tokens(engine.tokenize(THINK))
        engine.process_tokens(engine.tokenize(ANSWER))
    else:  # ref
        engine.process_tokens(engine.tokenize(ANSWER))
    engine.process_tokens(engine.tokenize(PROBE))
    return _greedy_continuation(engine, n_gen)


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
    ap.add_argument("--gen-tokens", type=int, default=40)
    args = ap.parse_args()

    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        raise SystemExit("set EVOKE_MODEL_PATH to the hybrid GGUF")
    engine = LlamaCppEngine(model, n_ctx=args.n_ctx, n_gpu_layers=-1, verbose=False)

    ref = run_path(engine, "ref", args.gen_tokens)
    mech = run_path(engine, "mech", args.gen_tokens)
    withthink = run_path(engine, "withthink", args.gen_tokens)

    mech_match = _match_prefix(mech, ref)
    wt_match = _match_prefix(withthink, ref)

    print(f"gen_tokens = {args.gen_tokens}")
    print(f"REF       : {engine.detokenize(ref)[:120]!r}")
    print(f"MECH      : {engine.detokenize(mech)[:120]!r}")
    print(f"WITHTHINK : {engine.detokenize(withthink)[:120]!r}")
    print(f"\nMECH==REF prefix match: {mech_match}/{args.gen_tokens}")
    print(f"WITHTHINK==REF prefix match: {wt_match}/{args.gen_tokens}")

    mech_clean = mech == ref
    sensitive = withthink != ref
    if mech_clean and sensitive:
        print(
            "\nPASS: snapshot+restore+replay cleanly removed the think trace from "
            "the hybrid state (MECH==REF), and the test is sensitive (WITHTHINK!=REF)."
        )
    elif mech_clean and not sensitive:
        print(
            "\nINCONCLUSIVE: MECH==REF but WITHTHINK==REF too -> the think trace did "
            "not perturb the state, so the test cannot confirm removal. Strengthen T."
        )
    else:
        print(
            "\nFAIL: MECH diverges from REF -> snapshot/restore is not state-correct "
            "on the hybrid model (recurrent state not cleanly rolled back)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
