"""Engine-level latency profile of kv_block_save / kv_block_load.

Isolates the C++ primitive cost from the manager bookkeeping. Runs multiple
trials to expose CUDA warmup / first-call effects, and sweeps block size to
find the crossover where kv_restore beats a re-prefill.
"""

from __future__ import annotations

import os
import time

from evoke.llama_engine import LlamaCppEngine


def _make_block(engine: LlamaCppEngine, n_target: int) -> list[int]:
    text = "The maximum retry limit is set to 17 attempts. " * 200
    toks = engine.tokenize(text)
    return toks[:n_target]


def _re_prefill_ms(engine: LlamaCppEngine, tokens: list[int]) -> float:
    engine.reset()
    t0 = time.perf_counter()
    engine.process_tokens(tokens)
    return (time.perf_counter() - t0) * 1000.0


def _save_load_ms(
    engine: LlamaCppEngine, tokens: list[int]
) -> tuple[float, float, int]:
    engine.reset()
    engine.process_tokens(tokens)
    n = len(tokens)
    t0 = time.perf_counter()
    data = engine.kv_block_save(0, n)
    t1 = time.perf_counter()
    engine.evict_ranges([(0, n)])
    t2 = time.perf_counter()
    engine.kv_block_load(data, 0)
    t3 = time.perf_counter()
    return (t1 - t0) * 1000.0, (t3 - t2) * 1000.0, len(data)


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("set EVOKE_MODEL_PATH")
        return 1

    engine = LlamaCppEngine(model, n_ctx=8192, n_gpu_layers=-1, verbose=False)
    print(f"kv_block primitives: {engine.supports_kv_block}")

    warm_tokens = _make_block(engine, 40)
    print("\n--- warmup (5 cycles at 40 tokens) ---")
    for i in range(5):
        save_ms, load_ms, nbytes = _save_load_ms(engine, warm_tokens)
        print(
            f"  cycle {i}: save={save_ms:7.2f}ms  load={load_ms:7.2f}ms  bytes={nbytes}"
        )

    print("\n--- block-size sweep (3 trials each, post-warmup) ---")
    print(
        f"  {'n_tok':<6}{'save(ms)':<11}{'load(ms)':<11}{'reprefill(ms)':<15}{'bytes':<10}"
    )
    for n in [20, 40, 80, 160, 320, 640, 1280]:
        toks = _make_block(engine, n)
        if len(toks) < n:
            print(f"  {n:<6} skipped (tokenizer produced {len(toks)})")
            continue
        save_ts, load_ts, refill_ts = [], [], []
        for _ in range(3):
            s, l, nbytes = _save_load_ms(engine, toks)
            save_ts.append(s)
            load_ts.append(l)
            refill_ts.append(_re_prefill_ms(engine, toks))

        def med(xs: list[float]) -> float:
            return sorted(xs)[len(xs) // 2]

        print(
            f"  {n:<6}{med(save_ts):<11.2f}{med(load_ts):<11.2f}"
            f"{med(refill_ts):<15.2f}{nbytes:<10}"
        )

    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
