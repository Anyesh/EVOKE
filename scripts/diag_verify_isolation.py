"""Isolate which step of verify_kv_restore is regressing on Qwen 2.5.

Runs the same flow incrementally and reports next_write_pos at each step.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager


def main() -> int:
    model = os.environ["EVOKE_MODEL_PATH"]
    engine = LlamaCppEngine(model, n_ctx=8192, n_gpu_layers=-1, verbose=False)
    cfg = EvokeConfig(max_active_tokens=4096, block_size=64, recovery_mode="kv_restore")
    mgr = EvokeManager(engine, cfg)

    print(f"  start: next_write_pos={engine.next_write_pos}")
    mgr.load_document("General project background. " * 50)
    print(f"  after load_document: next_write_pos={engine.next_write_pos}")
    mgr.add_context("Secret passkey: 7391.", key="fact")
    print(f"  after add fact: next_write_pos={engine.next_write_pos}")
    mgr.add_context("Unrelated detail. " * 50, key="filler")
    print(f"  after add filler: next_write_pos={engine.next_write_pos}")

    fact_blocks = [b for b in mgr._positions.active_blocks if b.key.startswith("fact#")]
    print(f"  fact blocks: {[b.block_id for b in fact_blocks]}")

    print(f"  force_evict fact blocks...")
    mgr.force_evict([b.block_id for b in fact_blocks])
    print(f"  after force_evict: next_write_pos={engine.next_write_pos}")

    print(f"  recovering fact via breadcrumbs...")
    crumbs = mgr.get_breadcrumbs()
    print(f"  breadcrumbs: {[c.key for c in crumbs]}")
    for c in crumbs:
        if c.key.startswith("fact#"):
            ok = mgr.recover(c.key)
            print(
                f"  recover({c.key!r}): ok={ok}; next_write_pos={engine.next_write_pos}"
            )

    print(f"  process_user_message...")
    mgr.process_user_message("\n\nQuestion: what is the secret passkey?\nAnswer:")
    print(f"  after process_user_message: next_write_pos={engine.next_write_pos}")

    print(f"  calling generate(32)...")
    try:
        answer = mgr.generate(32)
        print(f"  PASS: generated {len(answer)} chars: {answer!r}")
    except RuntimeError as e:
        print(f"  FAIL: {e}; final next_write_pos={engine.next_write_pos}")
        engine.close()
        return 1
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
