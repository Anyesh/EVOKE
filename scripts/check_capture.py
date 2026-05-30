"""Diagnostic: verify the q/k capture returns real, discriminating values on the engine."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arkvale_policy import block_cuboid, cuboid_score
from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager


def main() -> int:
    engine = LlamaCppEngine(os.environ["EVOKE_MODEL_PATH"], n_ctx=8192, n_gpu_layers=-1)
    print("supports_kv_block:", engine.supports_kv_block)
    engine.attn_capture_set_layer(20)
    qbuf = np.zeros(4_000_000, dtype=np.float32)
    kbuf = np.zeros(4_000_000, dtype=np.float32)
    engine.query_capture_set_buffer(qbuf)
    engine.key_capture_set_buffer(kbuf)

    mgr = EvokeManager(engine, EvokeConfig(max_active_tokens=1_000_000, block_size=64))
    texts = [
        "The secret password for the vault is icarus-pinwheel-43.",
        "Bananas are a yellow tropical fruit rich in potassium.",
        "The peace treaty was signed on the fourteenth of November.",
    ]
    cubs = []
    for i, t in enumerate(texts):
        mgr.add_context_tokens(engine.tokenize(t), key=f"b{i}")
        k = engine.read_key_capture()
        if k is None:
            print(f"block {i}: key-capture is None")
            continue
        print(
            f"block {i}: kshape={k.shape} min={k.min():.3f} max={k.max():.3f} "
            f"mean={k.mean():.4f} std={k.std():.4f}"
        )
        cubs.append(block_cuboid(k))

    mgr.process_user_message("\n\nQuestion: What is the secret password?\nAnswer:")
    q = engine.read_query_capture()
    if q is None:
        print("query-capture is None")
        return 1
    ql = q[-1]
    print(
        f"q: shape={q.shape} qlast min={ql.min():.3f} max={ql.max():.3f} "
        f"mean={ql.mean():.4f} std={ql.std():.4f}"
    )
    for i, c in enumerate(cubs):
        print(f"score block{i} [{texts[i][:24]!r}]: {cuboid_score(ql, *c):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
