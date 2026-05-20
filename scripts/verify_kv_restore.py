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
FACT = f"Important system note: the secret passkey is {PASSKEY}. Keep it confidential."


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

    mgr.load_document("General project background. " * 50)
    mgr.add_context(FACT, key="fact")
    mgr.add_context("Unrelated implementation detail. " * 50, key="filler")

    fact_blocks = [b for b in mgr._positions.active_blocks if b.key.startswith("fact#")]
    if not fact_blocks:
        print("FAIL: fact block not found after add_context")
        return 1
    print(f"fact occupies {len(fact_blocks)} block(s)")

    mgr.force_evict([b.block_id for b in fact_blocks])
    if any(b.key.startswith("fact#") for b in mgr._positions.active_blocks):
        print("FAIL: fact block still active after force_evict")
        return 1
    print("fact block evicted")

    recovered = 0
    for crumb in mgr.get_breadcrumbs():
        if crumb.key.startswith("fact#") and mgr.recover(crumb.key):
            recovered += 1
    print(
        f"recovered {recovered} block(s); total_recoveries={mgr.get_stats().total_recoveries}"
    )
    if recovered == 0:
        print("FAIL: recover() returned False for the fact block")
        return 1

    mgr.process_user_message(
        "\n\nQuestion: what is the secret passkey mentioned earlier?\nAnswer:"
    )
    answer = mgr.generate(8192)
    safe = answer.encode("ascii", "replace").decode("ascii")
    print(f"answer: {safe!r}")

    if PASSKEY in answer:
        print("PASS: model recalled the passkey from the restored KV block")
        return 0
    print("FAIL: passkey not recalled -- restored K/V is not attendable")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
