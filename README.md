# EVOKE

**Selective KV Cache Eviction and Recovery for Long-Context LLM Inference**

LLMs have a memory problem. Load a 200K-token document into context and every token stays in the KV cache for the entire session, even when only 5% of it matters to the current conversation. The cache is append-only: there's no way to free irrelevant context and get it back later when you need it.

EVOKE treats the KV cache as a managed memory system. It scores context blocks by relevance, evicts the least useful ones to an archive, and retrieves them when the conversation circles back. Retrieved blocks are re-encoded into the KV cache at correct RoPE positions via a full rebuild, so the model gets genuine attention over recovered context, not a text-level RAG approximation.

## How it works

```
Document/Conversation
        |
   [ Tokenize + Block ]
        |
   [ KV Cache (Active) ]  <--- budget enforced here
        |         |
   [ Score ]   [ Evict ] ---> [ Archive (embeddings + tokens) ]
        |                              |
   [ Generate ]              [ Retrieve on query match ]
        |                              |
   [ Answer ]  <--- [ Rebuild KV with recovered blocks ]
```

1. **Block**: Text is split into fixed-size token blocks (default 32 tokens).
2. **Score**: Each block gets a relevance score based on recency, position (sink tokens), and embedding similarity to recent context.
3. **Evict**: When active tokens exceed the budget, lowest-scored blocks are archived. The KV cache is rebuilt from remaining blocks.
4. **Archive**: Evicted blocks retain their token IDs, original positions, and representative embeddings.
5. **Retrieve**: When a user message matches archived blocks (via lexical overlap + embedding similarity), those blocks are promoted back.
6. **Rebuild**: Promoted blocks are inserted into the active set in original order, and the entire KV cache is re-encoded from scratch. This gives correct RoPE positions and fresh K/V tensors.

## Early results

Multi-turn conversation with Qwen 2.5 7B (Q4_K_M), 6 turns, information planted in early turns, recalled in the final turn:

| Budget | Active tokens | Archived blocks | Promotions | Recall |
|--------|--------------|-----------------|------------|--------|
| 512    | 399          | 50              | 21         | AURORA found |
| 1024   | 770          | 38              | 20         | AURORA found |

The model operates within a fixed token budget while still recalling "AURORA-SEVEN" details that were evicted turns ago. With a 512-token budget managing a conversation that would normally consume ~2000+ tokens, EVOKE keeps memory usage at 25% while preserving recall.

## Key design decisions

**Rebuild on every eviction and promotion.** Newer llama.cpp requires consecutive KV positions. Removing blocks from the middle creates gaps that crash inference. We rebuild the full KV cache after every structural change. This is more expensive per operation but always correct.

**No KV tensor save/restore.** We investigated `llama_state_seq_set_data` for per-block KV state caching. It's destructive: internally calls `seq_keep` + `seq_rm`, wiping the entire cache. Per-block state save is not viable through the current llama.cpp API. Re-encoding is the correct path.

**Thinking-model aware generation.** Thinking models (Qwen 3.x, DeepSeek R1) generate `<think>` blocks that can consume thousands of tokens before answering. EVOKE's generation pipeline supports separate budgets for thinking and answer phases, driven by the chat template layer.

**Pure Python.** The bottleneck is `llama_decode`, not Python overhead. The management logic (scoring, eviction decisions, position tracking) is negligible compared to transformer forward passes.

## Project structure

```
src/evoke/
  manager.py       Core orchestration: load, generate, evict, promote
  config.py        EvokeConfig dataclass
  engine.py        InferenceEngine protocol
  llama_engine.py  llama.cpp implementation via llama-cpp-python
  position.py      Tracks logical positions of active blocks
  scorer.py        Relevance scoring (recency + sink + coherence)
  archive.py       Archived block storage and retrieval
  chat_template.py Chat template detection and thinking tag handling
  types.py         ActiveBlock, ArchiveBlock, CacheStats, EvokeEvent
  benchmark.py     Competitive benchmark harness
```

## Quick start

```bash
uv sync
uv run pytest tests/ -x -q

# Run multi-turn benchmark
EVOKE_MODEL_PATH=/path/to/model.gguf uv run python scripts/multi_turn_bench.py
```

## Status

Research prototype. Targets both a working system and a paper: *"EVOKE: Selective KV Cache Eviction and Recovery for Long-Context LLM Inference."*

Competitive benchmarks against baselines (truncation, StreamingLLM, full context) are next.
