"""Run the EVOKE OpenAI-compatible chat completions server.

Environment:
  EVOKE_MODEL_PATH  - path to a GGUF model (required)
  LLAMA_CPP_LIB     - path to the EVOKE llama.cpp build (optional, but required
                      for kv_block save/load primitives)
  EVOKE_HOST        - bind address, default 127.0.0.1
  EVOKE_PORT        - port, default 8000
  EVOKE_N_CTX       - context size in tokens, default 32768
  EVOKE_MODEL_NAME  - logical model name returned by /v1/models, defaults to
                      the GGUF filename stem
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn

from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.server import create_app


def main() -> int:
    model_path = os.environ.get("EVOKE_MODEL_PATH")
    if not model_path:
        print("FAIL: set EVOKE_MODEL_PATH to a GGUF model file")
        return 1

    host = os.environ.get("EVOKE_HOST", "127.0.0.1")
    port = int(os.environ.get("EVOKE_PORT", "8000"))
    n_ctx = int(os.environ.get("EVOKE_N_CTX", "32768"))
    model_name = os.environ.get("EVOKE_MODEL_NAME") or Path(model_path).stem

    budget_env = os.environ.get("EVOKE_BUDGET")
    recovery_mode = os.environ.get("EVOKE_RECOVERY_MODE", "kv_restore")
    policy = os.environ.get("EVOKE_POLICY", "evoke").lower()
    config: EvokeConfig | None = None

    if policy == "truncate":
        # StreamingLLM-style: drop the oldest non-sink block under budget
        # pressure, no recovery. Pure recency, no coherence weighting, with
        # a small sink-token window.
        budget = int(budget_env) if budget_env else 1024
        config = EvokeConfig(
            max_active_tokens=budget,
            block_size=128,
            high_watermark=0.92,
            low_watermark=0.70,
            w_recency=1.0,
            w_coherence=0.0,
            sink_count=4,
            recovery_mode="discard",
        )
        print(f"  policy=truncate budget={budget} recovery=discard")
    elif policy == "no_eviction":
        # Lift the budget to n_ctx so the watermark never trips. Models how
        # a vanilla OpenAI-compatible server behaves under cache pressure.
        budget = int(budget_env) if budget_env else n_ctx
        config = EvokeConfig(
            max_active_tokens=budget,
            block_size=128,
            high_watermark=0.999,
            low_watermark=0.99,
            recovery_mode="discard",
        )
        print(f"  policy=no_eviction budget={budget}")
    elif policy == "evoke":
        if budget_env:
            budget = int(budget_env)
            config = EvokeConfig(
                max_active_tokens=budget,
                block_size=128,
                high_watermark=0.92,
                low_watermark=0.70,
                recovery_mode=recovery_mode,
            )
            print(f"  policy=evoke budget={budget} recovery={recovery_mode}")
    else:
        raise ValueError(f"unknown EVOKE_POLICY: {policy!r}")

    print(f"loading model: {model_path}")
    print(f"  n_ctx={n_ctx}  model_name={model_name}")
    engine = LlamaCppEngine(model_path, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
    print(f"  ready (n_embd={engine.n_embd}, kv_block={engine.supports_kv_block})")

    app = create_app(engine, model_name, config=config)
    print(f"serving on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
