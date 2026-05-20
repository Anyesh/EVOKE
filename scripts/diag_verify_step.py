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
    model = os.environ["EVOKE_MODEL_PATH"]
    engine = LlamaCppEngine(model, n_ctx=8192, n_gpu_layers=-1, verbose=False)
    config = EvokeConfig(
        max_active_tokens=4096, block_size=64, recovery_mode="kv_restore"
    )
    mgr = EvokeManager(engine, config)

    print(f"  step 1 (load_document): start={engine.next_write_pos}")
    mgr.load_document("General project background. " * 50)
    print(f"    after: pos={engine.next_write_pos}")

    print(f"  step 2 (add_context fact)")
    mgr.add_context(FACT, key="fact")
    print(f"    after: pos={engine.next_write_pos}")

    print(f"  step 3 (add_context filler)")
    mgr.add_context("Unrelated implementation detail. " * 50, key="filler")
    print(f"    after: pos={engine.next_write_pos}")

    fact_blocks = [b for b in mgr._positions.active_blocks if b.key.startswith("fact#")]
    print(f"  step 4 (force_evict): fact blocks {[b.block_id for b in fact_blocks]}")
    mgr.force_evict([b.block_id for b in fact_blocks])
    print(f"    after: pos={engine.next_write_pos}")

    print(f"  step 5 (recover)")
    for crumb in mgr.get_breadcrumbs():
        if crumb.key.startswith("fact#"):
            ok = mgr.recover(crumb.key)
            print(f"    recover({crumb.key}): {ok}; pos={engine.next_write_pos}")

    print(f"  step 6 (process_user_message)")
    mgr.process_user_message(
        "\n\nQuestion: what is the secret passkey mentioned earlier?\nAnswer:"
    )
    print(f"    after: pos={engine.next_write_pos}")

    print(f"  step 7 (manual generate_next loop, max 10 tokens):")
    print(f"  step 7b (mgr.generate(8192) — exactly what verify does):")
    try:
        answer = mgr.generate(8192)
        print(f"    PASS: {answer[:80]!r}")
    except RuntimeError as e:
        print(f"    FAIL: {e}; final pos={engine.next_write_pos}")
        engine.close()
        return 1

    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
