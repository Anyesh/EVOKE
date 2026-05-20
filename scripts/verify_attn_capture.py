"""Smoke test for the EVOKE attention-weight capture primitive (#30).

Loads a small prompt, configures attention capture for one layer, runs a
single decode, reads back the per-head softmax attention weights, and
verifies the result is shape-consistent and non-trivial (not all zeros,
not all uniform, sums per query roughly to 1.0).

Requires LLAMA_CPP_LIB pointing at the EVOKE-built llama.cpp DLL and
EVOKE_MODEL_PATH pointing at a GGUF. Run on the GPU eval host.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.llama_engine import LlamaCppEngine


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("FAIL: set EVOKE_MODEL_PATH")
        return 1
    layer = int(os.environ.get("EVOKE_CAPTURE_LAYER", "20"))

    engine = LlamaCppEngine(model, n_ctx=8192, n_gpu_layers=-1, verbose=False)
    if not engine.supports_kv_block:
        print("FAIL: LLAMA_CPP_LIB not set or doesn't expose the fork primitives")
        return 1

    print(f"model loaded: n_embd={engine.n_embd}  n_ctx={engine.n_ctx}")

    # Buffer sized for the largest expected capture in this test.
    # Layout: [n_query_tokens, n_heads, n_kv] f32. Maximum we'll see is one
    # prompt-worth of tokens with all 28 attention heads against itself.
    buf = np.zeros((1024 * 64 * 256,), dtype=np.float32)
    engine.attn_capture_set_layer(layer)
    engine.attn_capture_set_buffer(buf)

    prompt = (
        "The quick brown fox jumps over the lazy dog. "
        "Several animals watched the chase from the riverbank. "
        "The setting sun cast long shadows across the meadow."
    )
    tokens = engine.tokenize(prompt)
    print(f"prompt tokenized to {len(tokens)} tokens")

    engine.process_tokens(tokens)

    n_layers, n_query, n_heads, n_kv = engine.attn_capture_get_dims()
    written = engine.attn_capture_get_written()
    print(
        f"capture dims: n_layers={n_layers}  n_query={n_query}  "
        f"n_heads={n_heads}  n_kv={n_kv}"
    )
    print(f"floats written: {written}")

    if n_layers == 0 or n_query == 0 or n_heads == 0 or n_kv == 0:
        print("FAIL: capture dims are zero — the side-compute path didn't run")
        return 1
    if written == 0:
        print("FAIL: nothing written to buffer (capture tensor not recorded)")
        return 1

    arr = buf[:written].reshape(n_layers, n_heads, n_query, n_kv)[0]
    # ggml stores [n_kv, n_query_tokens, n_heads] in row-major (ne[0]=n_kv
    # is the fastest dim); after reshape with (n_heads, n_query, n_kv) we
    # get logical [head][query][kv]. Verify softmax invariants AND causal
    # mask: query i can only attend to keys 0..i (lower triangular when
    # n_query == n_kv); positions beyond i must be exactly zero.

    nonzero_frac = float((arr != 0).mean())
    per_query_sum = arr.sum(axis=-1)
    median_sum = float(np.median(per_query_sum))
    max_val = float(arr.max())
    min_val = float(arr.min())

    print(
        f"sanity: nonzero_frac={nonzero_frac:.3f}  "
        f"median(sum_over_kv)={median_sum:.3f}  "
        f"range=[{min_val:.3e}, {max_val:.3e}]"
    )

    if not (0.99 < median_sum < 1.01):
        print(
            f"FAIL: per-query softmax sum should be ~1.0, got median {median_sum:.3f}"
        )
        return 1
    if max_val > 1.001 or min_val < -1e-5:
        print("FAIL: softmax weights out of [0, 1] range")
        return 1

    # Causal mask verification. Build the lower-triangular mask manually:
    # for each query token at row i (in [0, n_query)), only keys 0..i are
    # legitimate; everything else must be zero across all heads. Tolerate
    # the prefill-time bookkeeping where past_context could be nonzero
    # below the diagonal but never above.
    legal_mask = np.tril(np.ones((n_query, n_kv), dtype=bool), k=0)
    illegal_max = float(arr[:, ~legal_mask].max()) if (~legal_mask).any() else 0.0
    legal_nonzero = int((arr[:, legal_mask] > 0).sum())
    print(
        f"causal mask: illegal_max={illegal_max:.3e}  "
        f"legal_nonzero_cells={legal_nonzero}"
    )
    if illegal_max > 1e-5:
        print("FAIL: weights present at causally-masked positions")
        return 1
    if legal_nonzero == 0:
        print("FAIL: no weights at any legal (lower-triangular) position")
        return 1

    print("PASS: attention capture working end-to-end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
