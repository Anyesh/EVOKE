# EVOKE

**OS-like memory management for the LLM KV cache.**

Long-running LLM agent sessions outgrow the physical KV cache budget within a few turns. EVOKE evicts low-relevance blocks under budget pressure and **recovers them recompute-free** via a custom save/restore primitive in a forked llama.cpp: 20–32× faster than re-prefilling the same tokens.

### Qwen 2.5 7B (pure attention)
![Eviction demo on Qwen 2.5](assets/eviction-demo.gif)

*A 14-turn session with a 1024-token budget. A fact is planted at turn 1 ("favorite number = 4242"), 12 unrelated knowledge questions fill the session, and at turn 14 the fact is probed. The session survives **40 evictions and 13 recoveries**, and the model recalls "4242".*

### Qwen 3.5 9B (hybrid Mamba/Attention + mrope, thinking-mode)
![Eviction demo on Qwen 3.5](assets/eviction-demo-qwen35.gif)

*Same demo, hybrid architecture. The model emits a `<think>...</think>` trace each turn (visible in the truncated `asst='<think>\n...'`). With `EVOKE_SUPPRESS_THINKING_STRIP=1` the server keeps the thinking trace in the returned content so the cached state stays aligned with what the client echoes back, and no session resets fire. **26 evictions, 4 recoveries**, fact recalled.*

## What it actually is

- Two new C++ primitives in a forked llama.cpp: `llama_kv_block_save` and `llama_kv_block_load`. They serialise a position range's K/V tensors to a host buffer and splice them back with per-cell RoPE re-anchoring, with no `llama_decode` call.
- A third C primitive `llama_attn_capture_*` that taps per-head softmax attention weights from one or more chosen layers (up to 16) into a host buffer once per decode. Used by the relevance scorer to learn what the model is actually attending to.
- A Python policy layer (`evoke/manager.py`, `evoke/scorer.py`, `evoke/attention_scorer.py`) that drives eviction under a watermark policy via a multi-signal scorer (model attention + harness priority + task-focus coherence + recency) and routes recovery through three pluggable backends: discard, breadcrumb, or kv_restore (the recompute-free splice).
- An OpenAI-compatible chat-completions server (`evoke/server.py`) that exposes EVOKE as a stateful endpoint. The persistent KV cache survives across requests; only the new tail of each prompt is decoded.
- Cross-architecture coverage: pure attention with standard RoPE (Qwen 2.5, Llama 3), hybrid Mamba/Attention (Qwen 3.5), MoE attention with mrope and thinking mode (Qwen 3.6 35B-A3B).

## How does the system know what's relevant?

Four signals, combined into a per-block score in [0, 1]. Lowest scores get evicted first when the cache exceeds budget.

- **The model's own attention.** A second softmax for one or more chosen transformer layers runs alongside the main attention path, writing per-head softmax weights to a host buffer once per decode. The scorer maintains a sliding window of recent attention mass per block (last 64 decode steps, EWMA decay 0.95). Blocks the model is actually attending to score high. This is the strongest single signal, the truest answer to "what's relevant right now."
- **Harness-supplied priority tags.** A coding harness like opencode or Claude Code can set `evoke_priority` (a float multiplier on the block's final score) and `evoke_pinned` (boolean, excludes from eviction entirely) on each chat request. Useful when the harness knows things the model can't see: a file read is the central artifact of the current task; a tool scratch output is one-shot. Defaults to `1.0 / false` so harnesses that ignore these fields get the model-and-heuristic behavior.
- **Task-focus coherence.** The scorer tracks a single "task focus" embedding (not a rolling average) that updates via EMA on new user messages but **snaps** to the new message when a topic shift is detected (cosine drop below 0.3) or signaled by the harness via `evoke_task_boundary=true`. Blocks from a prior task lose their coherence score in one pass instead of decaying over five turns.
- **Recency** + **sink protection** + **source-type floors**. Stability priors: prevent thrashing on a single attention spike; protect StreamingLLM-style sink tokens; give USER and ASSISTANT turns a floor so conversation backbone isn't evicted before document content.

Final score: `min(priority * (w_attn·attn + w_rec·recency + w_coh·coherence) / Σw, 1.0)` lifted by a source-type floor (USER blocks 0.6, ASSISTANT blocks 0.5 by default) and with pinned-block protection. See paper §4 for weights and the §7.5 scorer-ablation table for measured impact: attention-driven scoring is the difference between "evict the fact block and recover later" and "recognize the fact block as still-relevant and never evict it."

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
  eviction_demo.py      Replicate the demo GIF (14 turns, 40 evictions / 13 recoveries on Qwen 2.5 7B)
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

Research prototype targeting both a working system and a paper draft (`paper/draft.md`). The mechanism is verified end-to-end across three model families (Qwen 2.5 7B, Qwen 3.5 9B, Qwen 3.6 35B-A3B) and the head-to-head, agentic, and scorer-ablation eval numbers are in §7.3 through §7.6 of the draft. Recently closed in the codebase: tools-aware Jinja chat template via Python jinja2 (so tool-using turns no longer trigger session resets), multi-session pool with state-swap on a custom `X-EVOKE-Session` header, iSWA dual-cache support in the fork primitives, multi-layer attention capture (up to 16 layers per decode). Remaining gaps tracked in the paper's §8 Discussion and Limitations: Python-side end-to-end verification of iSWA on a Gemma GGUF, a no-shift eviction mode that lets hybrid-memory models drop the thinking range from attention without compacting the recurrent half, single-model-family latency benchmarks (numbers are Qwen 2.5 7B only outside §6 and §7.2), and a non-binary answer-quality metric for the agentic probes.

## License

The **EVOKE policy layer** in this repository (`src/evoke/`, `scripts/`, `paper/`, `examples/`, `assets/`) is licensed under the **Apache License 2.0** (see `LICENSE`). This includes the patent grant: contributors are barred from initiating patent litigation over the contributed code.

The **forked llama.cpp work** (the C primitives added in `<llama-cpp-fork>`) is a derivative work of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and remains under upstream's **MIT license**.
