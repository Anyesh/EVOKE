"""Smoke test for the EVOKE per-layer residual capture (J-lens scoring).

Enables layer-input capture at two layers, prefills a prompt long enough
to force multi-chunk decoding, and verifies the merged capture covers
every prompt token, distinguishes layers, and survives a follow-up
single-token decode. With EVOKE_JLENS_PROBE set it also runs the probe
dot products end to end and prints per-block scores.

Requires LLAMA_CPP_LIB pointing at a current EVOKE fork build (the one
that exports llama_set/get_embeddings_layer_inp with C linkage) and
EVOKE_MODEL_PATH pointing at a GGUF. Run on the GPU eval host.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.jlens_scorer import JLensScorer
from evoke.llama_engine import LlamaCppEngine
from evoke.types import ActiveBlock


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("FAIL: set EVOKE_MODEL_PATH")
        return 1
    layers_env = os.environ.get("EVOKE_JLENS_LAYERS", "19,23")
    layers = [int(x) for x in layers_env.split(",")]

    # Small n_batch forces chunked prefill so segment merging is exercised.
    engine = LlamaCppEngine(model, n_ctx=8192, n_gpu_layers=-1, n_batch=64, verbose=False)
    if not engine.supports_kv_block:
        print("FAIL: LLAMA_CPP_LIB not set or doesn't expose the fork primitives")
        return 1
    try:
        engine.layer_inp_capture_enable(layers)
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"model loaded: n_embd={engine.n_embd}, capture layers {layers}")

    prompt = "The quick brown fox jumps over the lazy dog. " * 40
    tokens = engine.tokenize(prompt)
    engine.process_tokens(tokens)
    captured = engine.layer_inp_capture_read()
    if captured is None:
        print("FAIL: no capture after prefill")
        return 1
    start, rows = captured
    if start != 0:
        print(f"FAIL: expected capture to start at position 0, got {start}")
        return 1
    for lid in layers:
        if lid not in rows:
            print(f"FAIL: layer {lid} missing from capture")
            return 1
        got = rows[lid].shape
        if got != (len(tokens), engine.n_embd):
            print(f"FAIL: layer {lid} shape {got} != {(len(tokens), engine.n_embd)}")
            return 1
        if float(np.abs(rows[lid]).sum()) == 0.0:
            print(f"FAIL: layer {lid} capture is all zeros")
            return 1
    if len(layers) > 1:
        a, b = rows[layers[0]], rows[layers[1]]
        cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        if abs(cos - 1.0) < 1e-6:
            print("FAIL: different layers returned identical residuals")
            return 1
        print(f"layer {layers[0]} vs {layers[1]} global cosine: {cos:.4f}")
    print(f"prefill capture OK: {len(tokens)} tokens x {engine.n_embd} across {len(layers)} layers")

    tok = engine.generate_next()
    captured = engine.layer_inp_capture_read()
    if captured is None or captured[1][layers[0]].shape[0] != 1:
        print("FAIL: decode-step capture missing or wrong length")
        return 1
    if captured[0] != len(tokens):
        print(f"FAIL: decode capture start {captured[0]} != {len(tokens)}")
        return 1
    print(f"decode capture OK (token {tok})")

    probe_path = os.environ.get("EVOKE_JLENS_PROBE")
    if probe_path:
        engine.reset()
        scorer = JLensScorer(engine, probe_path=probe_path, layers=layers)
        engine.process_tokens(tokens)
        n = len(tokens)
        blocks = [
            ActiveBlock(block_id=i, logical_start=s, logical_end=min(s + 64, n), token_ids=[0])
            for i, s in enumerate(range(0, n, 64))
        ]
        scorer.absorb_last_decode(blocks)
        scores = {b.block_id: scorer.score(b) for b in blocks}
        if any(v is None for v in scores.values()):
            print("FAIL: probe scoring returned None for a decoded block")
            return 1
        print("probe block scores:", {k: round(v, 3) for k, v in scores.items()})

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
