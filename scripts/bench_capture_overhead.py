"""Decode-throughput overhead of J-lens residual capture + probe scoring.

Measures tokens/second over a fixed generation with the capture disabled
versus enabled (including per-step JLensScorer probe absorption over a
realistic block count), after identical prefills. This isolates the
per-decode-token cost that the phase-3 gate bounds at 10%; whole-run wall
times from agent_bench fold in one-time prefill scoring and are too coarse.

Requires LLAMA_CPP_LIB (current fork build), EVOKE_MODEL_PATH, and
EVOKE_JLENS_PROBE. Run on the GPU eval host.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.jlens_scorer import JLensScorer
from evoke.llama_engine import LlamaCppEngine
from evoke.types import ActiveBlock

N_GEN = 128
N_WARMUP = 16
PREFILL_REPS = 60
BLOCK_SIZE = 64


def prefill(engine: LlamaCppEngine) -> int:
    text = "The quick brown fox jumps over the lazy dog. " * PREFILL_REPS
    tokens = engine.tokenize(text)
    engine.process_tokens(tokens)
    return len(tokens)


def timed_generation(engine: LlamaCppEngine, scorer: JLensScorer | None, blocks) -> float:
    for _ in range(N_WARMUP):
        engine.generate_next()
        if scorer is not None:
            scorer.absorb_last_decode(blocks)
    start = time.perf_counter()
    for _ in range(N_GEN):
        engine.generate_next()
        if scorer is not None:
            scorer.absorb_last_decode(blocks)
    return N_GEN / (time.perf_counter() - start)


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    probe = os.environ.get("EVOKE_JLENS_PROBE")
    if not model or not probe:
        print("set EVOKE_MODEL_PATH and EVOKE_JLENS_PROBE")
        return 1

    engine = LlamaCppEngine(model, n_ctx=8192, n_gpu_layers=-1, verbose=False)
    n = prefill(engine)
    baseline_tps = timed_generation(engine, None, [])
    print(f"baseline: {baseline_tps:.2f} tok/s ({n} prefill tokens, {N_GEN} generated)")

    engine.reset()
    scorer = JLensScorer(engine, probe_path=probe, layers=[19, 23])
    n = prefill(engine)
    blocks = [
        ActiveBlock(block_id=i, logical_start=s, logical_end=min(s + BLOCK_SIZE, n), token_ids=[0])
        for i, s in enumerate(range(0, n, BLOCK_SIZE))
    ]
    scorer.absorb_last_decode(blocks)
    capture_tps = timed_generation(engine, scorer, blocks)
    print(f"jlens capture: {capture_tps:.2f} tok/s ({len(blocks)} blocks scored per step)")

    overhead = (baseline_tps - capture_tps) / baseline_tps * 100.0
    print(f"decode overhead: {overhead:.2f}% (gate: < 10%)")
    print("PASS" if overhead < 10.0 else "FAIL")
    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
