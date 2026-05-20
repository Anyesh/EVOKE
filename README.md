# EVOKE

**OS-like memory management for the LLM KV cache.**

Long-running LLM agent sessions outgrow the physical KV cache budget within a few turns. EVOKE evicts low-relevance blocks under budget pressure and **recovers them recompute-free** via a custom save/restore primitive in a forked llama.cpp — 20–32× faster than re-prefilling the same tokens.

![Eviction demo](assets/eviction-demo.gif)

*A 14-turn session with a 1024-token budget. A fact is planted at turn 1 ("favorite number = 4242"), 12 unrelated knowledge questions fill the session, and at turn 14 the fact is probed. The session survives 89 evictions and 71 recoveries, and the model still recalls "4242".*

## What it actually is

- Two new C++ primitives in a forked llama.cpp: `llama_kv_block_save` and `llama_kv_block_load`. They serialise a position range's K/V tensors to a host buffer and splice them back with per-cell RoPE re-anchoring — no `llama_decode` call.
- A Python policy layer (`evoke/manager.py`, `evoke/scorer.py`) that drives eviction under a watermark policy and routes recovery through three pluggable backends: discard, breadcrumb, or kv_restore (the recompute-free splice).
- An OpenAI-compatible chat-completions server (`evoke/server.py`) that exposes EVOKE as a stateful endpoint. The persistent KV cache survives across requests; only the new tail of each prompt is decoded.
- Cross-architecture coverage: pure attention with standard RoPE (Qwen 2.5, Llama 3), hybrid Mamba/Attention (Qwen 3.5), MoE attention with mrope and thinking mode (Qwen 3.6 35B-A3B).

## Latency table

Measured on Qwen 2.5 7B, RTX 4070 Ti SUPER, Flash Attention enabled. `kv_block_load` is the EVOKE recovery path; `re-prefill` is the cost of re-encoding the same tokens via `llama_decode`.

| Block (tokens) | save (ms) | load (ms) | re-prefill (ms) | kv_restore speedup |
|---:|---:|---:|---:|---:|
|   20 |  1.10 | 0.48 |  11.90 | 25× |
|   40 |  1.61 | 0.70 |  13.78 | 20× |
|  160 |  4.69 | 1.50 |  32.60 | 22× |
|  640 | 16.37 | 4.34 | 118.36 | 27× |
| 1280 | 31.90 | 7.25 | 232.18 | 32× |

The gap widens linearly with block size: re-prefill is `O(tokens × model_FLOPs)`, load is `O(tokens × bytes)`.

## Repository layout

```
src/evoke/
  manager.py        Eviction/recovery orchestration, block tracking
  session.py        Persistent server session with prefix matching
  server.py         FastAPI /v1/chat/completions endpoint
  templates.py      Qwen chat template + tool-call parsing
  llama_engine.py   ctypes binding for the fork's primitives
  scorer.py         Relevance scoring (recency + sink + coherence)
  recovery.py       Pluggable backends (discard / breadcrumb / kv_restore)
  position.py       Active-block position tracking
  config.py         EvokeConfig

scripts/
  evoke_serve.py        Start the OpenAI-compatible server
  eviction_demo.py      Replicate the demo GIF (14 turns, 89 evictions)
  verify_kv_restore.py  Planted-passkey end-to-end primitive test
  profile_recover.py    Latency table generator
  agent_bench.py        Probe-correctness x budget x strategy

paper/draft.md     Paper draft
examples/          Sample opencode.json provider config
assets/            Demo GIF
```

## Quick start

You need a CUDA box with the EVOKE-forked llama.cpp built (see `paper/draft.md` §B). Then:

```bash
# Install the Python package + server extras
uv sync --extra server

# Start the OpenAI-compatible server (pick a model)
LLAMA_CPP_LIB=/path/to/EVOKE_llama.cpp/build/bin/llama.dll \
EVOKE_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
EVOKE_HOST=0.0.0.0 \
EVOKE_BUDGET=1024 \
EVOKE_MODEL_NAME=qwen25 \
uv run python scripts/evoke_serve.py

# Reproduce the demo GIF (eviction + recovery + fact recall)
EVOKE_SERVER='http://YOUR_HOST:8000' EVOKE_MODEL_NAME='qwen25' \
  uv run python scripts/eviction_demo.py

# Or point opencode at the server
cp examples/opencode.json ~/your-project/
# edit baseURL and model name, then:
cd ~/your-project && opencode
```

## Status

Research prototype targeting both a working system and a paper draft (`paper/draft.md`). The mechanism is verified end-to-end across three model architectures. Known gaps tracked in the paper's "Discussion and Limitations" section: smart-recovery policy (v3), chat-template fidelity for opencode tool-use turns, iSWA dual-cache support for Gemma, and SSM-state checkpointing for purely-Mamba layers.

## License

Forked llama.cpp work follows upstream's MIT license. EVOKE policy layer is the same.
