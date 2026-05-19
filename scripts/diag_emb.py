import ctypes
import os

import numpy as np

from evoke._engine_lib import llama_cpp
from evoke.llama_engine import LlamaCppEngine

model = os.environ["EVOKE_MODEL_PATH"]
engine = LlamaCppEngine(model, n_ctx=4096, n_gpu_layers=-1, verbose=False)
n_embd = engine.n_embd
tokens = engine.tokenize("Hello world test")
print("n tokens:", len(tokens), "n_embd:", n_embd)
engine.process_tokens(tokens)

emb_ptr = llama_cpp.llama_get_embeddings(engine._ctx)
print("llama_get_embeddings:", "NULL" if not emb_ptr else "non-null")

for i in range(len(tokens)):
    p = llama_cpp.llama_get_embeddings_ith(engine._ctx, i)
    if not p:
        print(f"  ith[{i}]: NULL")
    else:
        arr = np.ctypeslib.as_array(
            ctypes.cast(p, ctypes.POINTER(ctypes.c_float)), shape=(n_embd,)
        )
        print(f"  ith[{i}]: norm={float(np.linalg.norm(arr)):.4f}")

if emb_ptr:
    raw = np.ctypeslib.as_array(
        ctypes.cast(emb_ptr, ctypes.POINTER(ctypes.c_float)),
        shape=(len(tokens), n_embd),
    )
    for i in range(len(tokens)):
        print(f"  bare[{i}]: norm={float(np.linalg.norm(raw[i])):.4f}")

print("emb_cache keys:", sorted(engine._emb_cache.keys()))
for pos in sorted(engine._emb_cache.keys()):
    print(f"  cache[{pos}]: norm={float(np.linalg.norm(engine._emb_cache[pos])):.4f}")
engine.close()
