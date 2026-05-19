# EVOKE Architecture

## The Problem

During LLM inference, the KV cache is append-only. Every token loaded into context persists for the lifetime of the session, consuming O(n) memory and O(n) attention compute per generation step. There is no mechanism to free irrelevant tokens or recover them later when the conversation circles back.

A 200K-token document loaded into context keeps its full resource footprint even when only a fraction remains relevant to the ongoing conversation.

## Overview

EVOKE treats the KV cache as a managed memory system. It enforces a token budget on the active cache, scores context blocks by relevance, evicts the lowest-scoring blocks to an archive, and retrieves them when query similarity indicates they matter again. Retrieved blocks are re-encoded into the KV cache via a full rebuild, giving the model genuine attention over recovered context.

```
+-----------------------------------------------------+
|                    Application                       |
|              (user messages, documents)              |
+----------------------+------------------------------+
                       |
+----------------------v------------------------------+
|                  EvokeManager                        |
|                                                      |
|  +-------------+  +--------------+  +------------+  |
|  |  Relevance  |  |   Position   |  |  Archive   |  |
|  |   Scorer    |  |   Manager    |  |   Store    |  |
|  +------+------+  +------+-------+  +-----+------+  |
|         |                |                 |         |
|  +------v----------------v-----------------v------+  |
|  |         Demotion / Promotion via Rebuild        | |
|  +--------------------+---------------------------+  |
+----------------------+------------------------------+
                       |
+----------------------v------------------------------+
|              Inference Engine                        |
|         (llama.cpp KV cache + model)                |
|                                                      |
|  +------------------------------------------------+  |
|  |    Active KV Cache (GPU/system memory)         |  |
|  |    Budget-limited, contiguous positions         |  |
|  +------------------------------------------------+  |
+-----------------------------------------------------+
```

## Components

### Active Cache

The standard KV cache maintained by llama.cpp, with EVOKE enforcing a token budget. When active tokens exceed the budget, the lowest-scoring blocks are demoted to the archive and the entire KV cache is rebuilt from the remaining blocks.

Properties:
- Stored in GPU or system memory (wherever the inference engine places it)
- Participates in every attention computation
- Positions are always contiguous integers (enforced by full rebuild after every structural change)
- Budget is configurable (e.g., 512, 1024, 4096, 8192 tokens)

### Archive Store (`archive.py`)

A CPU-side ordered dictionary holding demoted blocks. Each block retains its token IDs, original positions, text, and a representative embedding for retrieval matching.

```python
@dataclass
class ArchiveBlock:
    block_id: int
    token_ids: list[int]
    original_positions: list[int]
    text: str
    representative_embedding: np.ndarray
    timestamp: int
    access_count: int = 0
```

Block size is configurable (default 128 tokens, 32 for tighter budgets). The archive has a capacity limit (`max_archive_blocks`); when exceeded, the oldest blocks are dropped via LRU.

Retrieval uses a hybrid strategy: lexical overlap (word recall against query) combined with embedding cosine similarity. Lexical hits are prioritized because they catch exact keyword matches that embedding similarity can miss. Neighbor expansion promotes blocks adjacent to any hit, preserving local coherence.

### Relevance Scorer (`scorer.py`)

Assigns a relevance score in [0, 1] to each active block based on three signals:

**Recency**: `exp(-decay * distance / context_length)`. Tokens closer to the current generation position score higher.

**Sink**: Blocks containing positions [0..sink_count-1] always score 1.0. Attention sinks are structurally necessary per StreamingLLM's finding.

**Coherence**: Cosine similarity between the block's representative embedding and the most recent context embedding. Measures semantic relatedness to what is currently being discussed. Returns 0.5 when no embeddings are available.

Composite: `(w_recency * recency + w_coherence * coherence) / (w_recency + w_coherence)`. Sink blocks bypass the composite entirely and always return 1.0.

### Position Manager (`position.py`)

Maintains the ordered list of active blocks and computes contiguous logical positions after every rebuild. The position manager does not interact with the engine directly; it only tracks bookkeeping.

After any structural change (demotion, promotion), the manager sorts blocks by `original_start` and recomputes contiguous positions starting from 0. This ensures the KV cache rebuild always produces positions 0..N-1 with no gaps.

### Demotion

Triggered when active tokens exceed the budget. Two policies:

**Watermark** (default): demote when active tokens exceed `high_watermark * budget`, free blocks until active tokens are at or below `low_watermark * budget`. This amortizes rebuild cost by batching demotions.

**Hard**: demote as soon as active tokens exceed the budget, free the minimum needed. More frequent but smaller rebuilds.

In both cases, blocks are sorted by relevance score (lowest first) and demoted until enough tokens are freed. Sink blocks and pinned generation blocks are never demoted.

After demoting, the manager removes blocks from the position list, and the engine rebuilds the entire KV cache from the remaining blocks' token IDs. This is the only correct approach with newer llama.cpp, which requires consecutive KV positions. Removing blocks from the middle via `kv_cache_seq_rm` creates gaps that crash inference.

### Promotion

When a user message matches archived blocks, those blocks are promoted back to the active set. The promotion path:

1. Retrieve matching blocks from the archive (lexical + semantic similarity)
2. Create `ActiveBlock` entries from the archived data, placed at their original positions
3. Insert into the position manager's list (sorted by `original_start`)
4. Rebuild the entire KV cache from all active blocks' token IDs
5. Recompute contiguous positions
6. If the budget is now exceeded, run demotion immediately

Promoted blocks get fresh K/V tensors via full re-encoding. This is more expensive than restoring saved tensors, but llama.cpp's state save API (`llama_state_seq_set_data`) is destructive internally (calls `seq_keep` + `seq_rm`, wiping the entire cache), making per-block KV tensor save/restore unviable. Re-encoding is the correct path.

### Thinking-Aware Generation

Thinking models (Qwen 3.x, DeepSeek R1) emit `<think>...</think>` blocks that can consume thousands of tokens before producing an answer. The `generate()` method supports two-phase generation:

```python
def generate(
    self,
    max_tokens: int,
    stop_token_ids: set[int] | None = None,
    *,
    think_close: str | None = None,
    thinking_budget: int = 16384,
    answer_budget: int = 512,
) -> str:
```

When `think_close` is provided, the generator runs in thinking mode: it generates up to `thinking_budget` tokens looking for the close tag in the token stream, then switches to answer mode and caps output at `answer_budget` tokens. The close tag is model-agnostic; it comes from the chat template layer, not hardcoded in the manager.

### Chat Templates (`chat_template.py`)

Handles model-specific prompt formatting and thinking tag detection:

- `ChatMLTemplate` for Qwen 2.x (ChatML format, `<|im_start|>` / `<|im_end|>`)
- `ChatMLThinkingTemplate` for Qwen 3.x (ChatML + `think_close = "</think>"`)
- `Llama3Template` for Llama 3.x (`<|begin_of_text|>`, `<|start_header_id|>`)
- `PassthroughTemplate` for unknown models (plain text, no special tokens)

`detect_template(model_name)` selects the template from the GGUF filename. `strip_thinking(text)` removes `<think>` blocks and chat stop tokens from generated output.

## Data Flow

### Loading a document

1. Tokenize text into token IDs
2. Split into fixed-size blocks (default 128 tokens)
3. Process all tokens through the engine (populates KV cache)
4. Compute representative embeddings for each block
5. Register blocks with the position manager
6. Enforce budget (may demote low-scoring blocks immediately)

### User message (multi-turn)

1. Check if a KV rebuild is needed (position space > 90% of n_ctx)
2. If rebuild needed: rebuild KV from active blocks, recompute positions
3. Search archive for blocks matching the user message (lexical + semantic)
4. If matches found: promote them via rebuild
5. Tokenize and process the user message tokens
6. Track as a conversation block in the position manager
7. Update the recent context embedding for future scoring
8. Enforce budget (demote if over threshold)

### Generation

1. Generate tokens one at a time via `engine.generate_next()`
2. Every `score_interval` steps, run a scoring round (may trigger demotion)
3. Stop on EOS, stop tokens, or budget exhaustion
4. For thinking models: detect close tag in token stream, then cap answer phase

### Demotion event

1. Score all active blocks
2. Sort by score, select lowest-scoring non-protected blocks
3. Archive each selected block (preserve token IDs, original positions, embedding, text)
4. Remove from position manager
5. Rebuild entire KV cache from remaining blocks
6. Recompute contiguous positions

### Promotion event

1. Query matches archived blocks above threshold
2. Move matched blocks from archive into position manager
3. Rebuild entire KV cache from all active blocks (including promoted ones)
4. Recompute contiguous positions
5. Enforce budget (promoted blocks may push over threshold)

## API

### EvokeManager

```python
class EvokeManager:
    def __init__(self, engine: InferenceEngine, config: EvokeConfig): ...

    def load_document(self, text: str) -> None: ...
    def process_user_message(self, text: str) -> None: ...
    def generate(self, max_tokens: int, stop_token_ids=None, *,
                 think_close=None, thinking_budget=16384, answer_budget=512) -> str: ...
    def get_stats(self) -> CacheStats: ...
    def get_event_log(self) -> list[EvokeEvent]: ...
    def get_relevance_scores(self) -> dict[int, float]: ...
    def force_demote(self, block_ids: list[int]) -> None: ...
    def force_promote(self, block_ids: list[int]) -> None: ...
```

### EvokeConfig

```python
@dataclass
class EvokeConfig:
    max_active_tokens: int = 8192
    block_size: int = 128
    sink_count: int = 4
    score_interval: int = 32
    recency_decay: float = 0.01
    w_recency: float = 0.4
    w_sink: float = 1.0
    w_coherence: float = 0.6
    demotion_policy: str = "watermark"    # "watermark" or "hard"
    high_watermark: float = 0.95
    low_watermark: float = 0.75
    retrieval_threshold: float = 0.85
    max_retrieve_blocks: int = 2
    max_archive_blocks: int = 1024
    pin_generated: bool = True
```

### InferenceEngine Protocol

```python
class InferenceEngine(Protocol):
    def tokenize(self, text: str) -> list[int]: ...
    def detokenize(self, tokens: list[int]) -> str: ...
    def process_tokens(self, tokens: list[int]) -> None: ...
    def generate_next(self) -> int: ...
    def get_kv_cache_token_count(self) -> int: ...
    def kv_cache_seq_rm(self, pos_start: int, pos_end: int) -> None: ...
    def get_embeddings(self, token_positions: list[int]) -> np.ndarray: ...
    def rebuild_kv(self, token_blocks: list[list[int]]) -> None: ...
    def reset(self) -> None: ...

    next_write_pos: int   # property
    n_ctx: int            # property
    n_embd: int           # property
    eos_token: int        # property
```

The critical method is `rebuild_kv`: it clears the entire KV cache, then re-processes each token block sequentially, producing a fresh cache with positions 0..N-1. This is the only safe way to modify the KV cache structure with current llama.cpp.

## Key Design Decisions

### Why rebuild instead of seq_rm + seq_add?

Newer llama.cpp requires consecutive KV positions. Removing blocks from the middle via `kv_cache_seq_rm` creates position gaps that crash inference with `llama_decode failed with code -1`. The only correct path is to rebuild: clear the cache, re-process all remaining tokens from scratch. This is more expensive per operation but always correct.

### Why not save and restore KV tensors?

We investigated `llama_state_seq_set_data` for per-block KV state caching. It is destructive internally: it calls `seq_keep` + `seq_rm`, wiping the entire cache before writing the restored state. Per-block state save is not viable through the current llama.cpp API. Re-encoding via full rebuild is the correct path.

### Why blocks, not individual tokens?

1. Semantic coherence: a 128-token block preserves enough meaning for retrieval matching. An individual token does not.
2. Scoring overhead: scoring 100 blocks is cheaper than scoring 12,800 tokens.
3. Archive index size: one representative embedding per block, not per token.
4. Retrieval quality: blocks carry enough text for lexical matching to work.

### Why hybrid lexical + semantic retrieval?

Embedding similarity alone misses exact keyword matches (proper nouns, codes, numbers). Lexical recall catches these reliably. Combining both gives better retrieval precision for the needle-in-haystack pattern that matters most in multi-turn conversations.

### Why model-agnostic thinking support?

Different models use different thinking tags (Qwen 3.x uses `<think>`, Gemma uses different markers). The thinking close tag comes from the chat template layer, not hardcoded in the generation logic. Adding a new thinking model means adding one template subclass with a `think_close` property.

### Why pure Python?

The bottleneck is `llama_decode`, not Python overhead. The management logic (scoring, eviction decisions, position tracking, archival) is negligible compared to transformer forward passes. Keeping the orchestration in Python makes it easy to experiment with different strategies.
