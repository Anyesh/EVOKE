# EVOKE

**A KV-cache memory hierarchy with recompute-free block recovery for LLM serving.**

Long-running LLM agent sessions outgrow the physical KV cache budget within a few turns. **EV**ict and rec**O**ver **K**V cache **E**ntries (EVOKE) treats the KV cache as a small fast tier of a memory hierarchy: low-relevance blocks are evicted under budget pressure and **recovered with no forward pass** by splicing the original K/V tensors back into the unified attention cache with one RoPE phase shift, via custom save/restore primitives in a forked llama.cpp. The recovery primitive is 20–32× faster than re-prefilling the same tokens (5.9–7.5× over the full save+load lifecycle). On standard needle-in-a-haystack at 4× compression, EVOKE recovers **96–100% of needles** across three model families (Qwen 2.5 7B, Qwen 3.5 9B hybrid, Qwen 3.6 35B-A3B MoE); recovery-less baselines (recency, StreamingLLM, H2O, SnapKV) flatten at 20–40% and the same-substrate InfLLM external-memory adaptation matches EVOKE at tight budgets but degrades at the loosest.

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
- **Task-focus coherence.** The scorer tracks a single "task focus" embedding (not a rolling average) that updates via EMA on new user messages but **snaps** to the new message when a topic shift is detected (cosine drop below 0.3) or signaled by the harness via `evoke_task_boundary=true`. Blocks from a prior task lose their coherence score in one pass instead of decaying over five turns. The block and query embeddings come from the model's own mean-pooled hidden states by default; the `use_retrieval_embeddings=True` config switches both to `BAAI/bge-small-en-v1.5` via `fastembed` (~30MB, ~10ms per text), which is what makes NIAH work at depths where the LM-hidden-state cosine has no discriminative headroom.
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

## Intuition: why eviction is non-destructive

Take a 20-token sentence (one token per word, periods folded into the preceding word):

```
pos:  0   1   2  3   4   5   6   7   8   9  10     11   12   13    14    15     16  17   18   19
tok: Cat sat in  a  mat and mat is  red in  color. house is  green in    color. cat is   very pretty
```

Suppose the scorer marks `house is green in color.` (positions 11–15) as low-relevance. EVOKE evicts in two engine calls. `seq_rm(seq=0, p0=11, p1=16)` frees those five cells in the unified KV buffer (no dangling reference: `Q` is never cached, only `K` and `V` are). `seq_add(seq=0, p0=16, p1=20, delta=-5)` then re-labels the survivors `cat is very pretty` from positions 16–19 to 11–14 and queues a deferred RoPE shift of `Δ = -5` on their `K` rows. K and V bytes never move in memory; only positions change.

llama.cpp applies the queued shift lazily at the next attention compute, multiplying each survivor's `K` by `R(Δ · θ_i)` per dimension pair. `V` is positional-free and untouched. After the shift, `Q_new · K_survivor` returns the same relative-position dot product the model would compute if `house is green in color.` had never been decoded. The model behaves identically to one that read the truncated sentence directly: information loss, never corruption.

## Live opencode integration

A live opencode session against Qwen 3.5 9B (hybrid Mamba/Attention + thinking, budget=2048) ran 250 cumulative evictions and 4 smart-recoveries with `active_tokens` held near 1414 (within budget) while `cached_tokens` grew to 32902, so the agent's conversation was 23x larger than what was held in GPU at any moment. Every paper §7 number is reproducible from `scripts/` per Appendix A, with raw output for the agentic eval (`results/agent_bench_qwen25_7b.txt`), the attention scorer ablation, and the keepalive workload checked into the repo.

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

paper/paper.pdf    Paper (with §B build instructions for the fork)
examples/          Sample opencode.json provider config
assets/            Demo GIFs
```

## Quick start

You need a CUDA box with the EVOKE-forked llama.cpp ([Anyesh/llama.cpp](https://github.com/Anyesh/llama.cpp)) built (see `paper/paper.pdf` §B). Then:

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

Research prototype targeting both a working system and a paper (`paper/paper.pdf`). Coverage spans three model families (Qwen 2.5 7B, Qwen 3.5 9B, Qwen 3.6 35B-A3B); head-to-head, agentic, and scorer-ablation results are in §7.3 through §7.6 of the paper. Recently landed in the codebase: tools-aware Jinja chat template via Python jinja2 (tool-using turns no longer trigger session resets), multi-session pool with state-swap on a custom `X-EVOKE-Session` header, iSWA dual-cache support in the fork primitives, multi-layer attention capture (up to 16 layers per decode). Open gaps tracked in §8 of the paper: Python-side iSWA load of a Gemma GGUF, a no-shift eviction mode that lets hybrid-memory models drop the thinking range from attention without compacting the recurrent half, latency numbers on architectures beyond Qwen 2.5 7B (outside §6 and §7.2), and a non-binary answer-quality metric for the agentic probes.

## License

The **EVOKE policy layer** in this repository (`src/evoke/`, `scripts/`, `paper/`, `examples/`, `assets/`) is licensed under the **Apache License 2.0** (see `LICENSE`). This includes the patent grant: contributors are barred from initiating patent litigation over the contributed code.

The **forked llama.cpp work** (the C primitives, hosted at [Anyesh/llama.cpp](https://github.com/Anyesh/llama.cpp)) is a derivative work of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and remains under upstream's **MIT license**.
