"""End-to-end verification of the kv_restore recovery mode on a real model.

Requires:
  EVOKE_MODEL_PATH - path to a GGUF model
  LLAMA_CPP_LIB    - path to the EVOKE llama.cpp build (libllama with the
                     kv_block_save / kv_block_load primitives)

Proves the C++ primitives: a block is evicted from the KV cache, then its
saved K/V is spliced back with RoPE re-anchored, and the model can still
attend to it (recalls a planted passkey).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager

PASSKEY = "7391"


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("FAIL: set EVOKE_MODEL_PATH")
        return 1

    engine = LlamaCppEngine(model, n_ctx=8192, n_gpu_layers=-1, verbose=False)
    if not engine.supports_kv_block:
        print("FAIL: kv_block primitives not bound -- set LLAMA_CPP_LIB to the build")
        return 1
    print("kv_block primitives bound")

    config = EvokeConfig(
        max_active_tokens=4096, block_size=64, recovery_mode="kv_restore"
    )
    mgr = EvokeManager(engine, config)

    doc = (
        "The weather is mild today. " * 30
        + f"Important fact to remember: the secret passkey is {PASSKEY}. "
        + "Markets were quiet this week. " * 30
    )
    mgr.load_document(doc)

    target = None
    for block in mgr._positions.active_blocks:
        if PASSKEY in engine.detokenize(block.token_ids):
            target = block
            break
    if target is None:
        print("FAIL: passkey block not found after load")
        return 1
    key = target.key
    print(f"passkey is in block {key}")

    mgr.force_evict([target.block_id])
    if key in [b.key for b in mgr._positions.active_blocks]:
        print("FAIL: block still active after force_evict")
        return 1
    print("passkey block evicted")

    if not mgr.recover(key):
        print("FAIL: recover() returned False")
        return 1
    print(f"recovered; total_recoveries={mgr.get_stats().total_recoveries}")

    mgr.process_user_message("\n\nQuestion: what is the secret passkey?\nAnswer:")
    answer = mgr.generate(24)
    print(f"answer: {answer!r}")

    if PASSKEY in answer:
        print("PASS: model recalled the passkey from the restored KV block")
        return 0
    print("FAIL: passkey not recalled -- restored K/V is not attendable")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
