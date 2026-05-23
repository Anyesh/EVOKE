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
    # Multi-signal scorer knobs (read once here so policy blocks below can
    # apply them). Defaults preserve pre-multi-signal behavior.
    w_attention = float(os.environ.get("EVOKE_W_ATTENTION", "0.0"))
    attention_capture_layer = int(os.environ.get("EVOKE_ATTN_LAYER", "20"))
    ram_budget_env = os.environ.get("EVOKE_KV_RESTORE_RAM_BUDGET_BYTES")
    kv_restore_ram_budget_bytes = int(ram_budget_env) if ram_budget_env else None
    kv_restore_spill_path = os.environ.get("EVOKE_KV_RESTORE_SPILL_PATH") or None
    suppress_thinking_strip = bool(os.environ.get("EVOKE_SUPPRESS_THINKING_STRIP"))
    # Smart-recovery knobs. min_similarity sets an absolute cosine floor on
    # which evicted blocks are eligible for top-K recovery, on top of the
    # resident-gate that compares against the strongest already-resident
    # block. Without a floor the gate alone lets weak matches through once
    # the resident set thins out, causing recover-then-re-evict thrash at
    # long sessions (T=28 sweep diagnosed 80 extra evictions = 2.4s of pure
    # thrash). use_retrieval is off by default because earlier deployments
    # ran without fastembed; turning it on widens the cosine band so the
    # floor is meaningful (LM hidden states crowd similarities into the
    # 0.85-0.93 band where no useful threshold lives).
    smart_recover_min_similarity = float(
        os.environ.get("EVOKE_SMART_RECOVER_MIN_SIMILARITY", "0.0")
    )
    smart_recover_k = int(os.environ.get("EVOKE_SMART_RECOVER_K", "4"))
    use_retrieval_embeddings = bool(os.environ.get("EVOKE_USE_RETRIEVAL_EMBEDDINGS"))
    # Recovery-aware eviction knobs. w_recovery > 0 enables the scorer to
    # weigh per-block recovery_strength, which fresh recovery sets to
    # recovery_strength_init and tick_turn decays by recovery_decay each turn.
    # See decision-recovery-aware-eviction in the wiki for the design.
    w_recovery = float(os.environ.get("EVOKE_W_RECOVERY", "0.0"))
    recovery_strength_init = float(
        os.environ.get("EVOKE_RECOVERY_STRENGTH_INIT", "1.0")
    )
    recovery_decay = float(os.environ.get("EVOKE_RECOVERY_DECAY", "0.7"))
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
            suppress_thinking_strip=suppress_thinking_strip,
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
            suppress_thinking_strip=suppress_thinking_strip,
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
                w_attention=w_attention,
                attention_capture_layer=attention_capture_layer,
                kv_restore_ram_budget_bytes=kv_restore_ram_budget_bytes,
                kv_restore_spill_path=kv_restore_spill_path,
                suppress_thinking_strip=suppress_thinking_strip,
                smart_recover_k=smart_recover_k,
                smart_recover_min_similarity=smart_recover_min_similarity,
                use_retrieval_embeddings=use_retrieval_embeddings,
                w_recovery=w_recovery,
                recovery_strength_init=recovery_strength_init,
                recovery_decay=recovery_decay,
            )
            print(
                f"  policy=evoke budget={budget} recovery={recovery_mode}"
                f" w_attention={w_attention} attn_layer={attention_capture_layer}"
                f" kv_ram_budget={kv_restore_ram_budget_bytes}"
                f" kv_spill={kv_restore_spill_path}"
                f" sr_k={smart_recover_k}"
                f" sr_min_sim={smart_recover_min_similarity}"
                f" use_retrieval={use_retrieval_embeddings}"
                f" w_recovery={w_recovery}"
                f" rec_init={recovery_strength_init}"
                f" rec_decay={recovery_decay}"
            )
    else:
        raise ValueError(f"unknown EVOKE_POLICY: {policy!r}")

    n_rs_seq = int(os.environ.get("EVOKE_N_RS_SEQ", "0"))
    print(f"loading model: {model_path}")
    print(f"  n_ctx={n_ctx}  model_name={model_name}  n_rs_seq={n_rs_seq}")
    engine = LlamaCppEngine(
        model_path, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False, n_rs_seq=n_rs_seq
    )
    print(f"  ready (n_embd={engine.n_embd}, kv_block={engine.supports_kv_block})")

    max_sessions = int(os.environ.get("EVOKE_MAX_SESSIONS", "8"))
    app = create_app(engine, model_name, config=config, max_sessions=max_sessions)
    print(f"serving on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
