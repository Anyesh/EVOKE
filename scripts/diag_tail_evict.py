"""Minimal repro for the decode -1 bug after a tail eviction.

After llama_memory_seq_rm removes the trailing cells of a sequence, the next
llama_decode at the freed positions fails. This script isolates the failure
without the full session machinery so we can see exactly what's happening.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.llama_engine import LlamaCppEngine


def main() -> int:
    model = os.environ["EVOKE_MODEL_PATH"]
    engine = LlamaCppEngine(model, n_ctx=4096, n_gpu_layers=-1, verbose=False)

    toks1 = engine.tokenize("Hello world, this is a small prefix.")
    print(f"  prefix tokens: {len(toks1)}")
    engine.process_tokens(toks1)
    print(f"  next_write_pos: {engine.next_write_pos}")

    gen_start = engine.next_write_pos
    gen_tokens = []
    for _ in range(8):
        gen_tokens.append(engine.generate_next())
    print(f"  generated 8 tokens; next_write_pos: {engine.next_write_pos}")
    print(f"  generated tokens: {gen_tokens}")

    gen_end = engine.next_write_pos
    print(f"  evicting [{gen_start}, {gen_end})")
    engine.evict_ranges([(gen_start, gen_end)])
    print(f"  after evict: next_write_pos={engine.next_write_pos}")

    toks2 = engine.tokenize(" Another small chunk.")
    print(f"  decoding {len(toks2)} more tokens at pos {engine.next_write_pos}")
    try:
        engine.process_tokens(toks2)
        print("  SUCCESS: decode after tail eviction worked")
        print(f"  final next_write_pos: {engine.next_write_pos}")
    except RuntimeError as e:
        print(f"  FAIL: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
