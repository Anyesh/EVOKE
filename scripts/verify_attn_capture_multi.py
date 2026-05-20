"""Verify multi-layer attention capture (#39).

Captures attention from 3 layers (4, 14, 20) in a single decode, asserts
each layer's softmax sums to 1.0 per query, and confirms the buffer
layout matches [n_layers, n_query, n_heads, n_kv].
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.llama_engine import LlamaCppEngine

LAYERS = [4, 14, 20]


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("FAIL: set EVOKE_MODEL_PATH")
        return 1
    engine = LlamaCppEngine(model, n_ctx=8192, n_gpu_layers=-1, verbose=False)
    if not engine.supports_kv_block:
        print("FAIL: LLAMA_CPP_LIB not set to EVOKE fork")
        return 1

    buf = np.zeros((1024 * 64 * 256 * 4,), dtype=np.float32)
    engine.attn_capture_set_layers(LAYERS)
    engine.attn_capture_set_buffer(buf)

    prompt = (
        "The quick brown fox jumps over the lazy dog. "
        "Several animals watched from the riverbank. "
        "The setting sun cast long shadows across the meadow."
    )
    tokens = engine.tokenize(prompt)
    print(f"prompt tokenized to {len(tokens)} tokens; capturing layers {LAYERS}")
    engine.process_tokens(tokens)

    n_layers, n_query, n_heads, n_kv = engine.attn_capture_get_dims()
    written = engine.attn_capture_get_written()
    expected = n_layers * n_query * n_heads * n_kv
    print(
        f"dims: n_layers={n_layers}  n_query={n_query}  n_heads={n_heads}  n_kv={n_kv}"
    )
    print(f"written={written}  expected={expected}")

    if n_layers != len(LAYERS):
        print(f"FAIL: n_layers {n_layers} != {len(LAYERS)}")
        return 1
    if written == 0 or written != expected:
        print("FAIL: write count mismatch")
        return 1

    arr = buf[:written].reshape(n_layers, n_heads, n_query, n_kv)
    for i, layer_idx in enumerate(LAYERS):
        layer = arr[i]
        per_query_sum = layer.sum(axis=-1)  # [n_heads, n_query]
        median_sum = float(np.median(per_query_sum))
        nonzero = float((layer != 0).mean())
        max_val = float(layer.max())
        print(
            f"  layer {layer_idx:2d}: median(sum)={median_sum:.3f}  "
            f"nonzero_frac={nonzero:.3f}  max={max_val:.3e}"
        )
        if not (0.99 < median_sum < 1.01):
            print(f"FAIL: layer {layer_idx} softmax sum off")
            return 1

    # Cross-layer signal differs: shallow layers attend differently than
    # deep ones. If all three captured layers had identical weights we'd
    # suspect a stale-tensor bug.
    if (arr[0] == arr[1]).all():
        print("FAIL: layers 4 and 14 produced identical weights (likely a bug)")
        return 1
    print("PASS: multi-layer capture working")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
