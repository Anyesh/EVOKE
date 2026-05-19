# UNLEARN: A Memory Management Unit for LLM Inference

## Abstract

UNLEARN is a runtime memory management layer that sits between a transformer's attention mechanism and its physical KV cache, managing context as a hierarchy of storage levels with continuous, relevance-driven promotion and demotion. Unlike existing approaches that permanently evict tokens (H2O, SnapKV) or naively page text in and out (MemGPT), UNLEARN operates at the KV cache level with a retrieval path for demoted content and predictive relevance scoring that anticipates future information needs.

## 1. The Problem

During LLM inference, the KV cache is append-only. Every token loaded into context persists for the lifetime of the session, consuming both memory (O(n) storage) and compute (O(n^2) attention per step, or O(n) with FlashAttention but still linear in cache size). There is no mechanism to:

- Identify which cached tokens are irrelevant to the current conversation direction
- Free the resources consumed by irrelevant tokens
- Retrieve previously freed tokens if they become relevant again
- Manage positions coherently after mid-sequence modifications

The result is that a 200K-token document loaded into context consumes its full resource footprint even when only 10K tokens remain relevant to the ongoing conversation.

## 2. Architecture Overview

```
+-----------------------------------------------------+
|                    Application                       |
|              (user messages, documents)              |
+----------------------+------------------------------+
                       |
+----------------------v------------------------------+
|                 UNLEARN Manager                      |
|                                                      |
|  +-------------+  +--------------+  +------------+  |
|  |  Relevance  |  |   Position   |  |  Retrieval |  |
|  |   Scorer    |  |   Manager    |  |    Gate    |  |
|  +------+------+  +------+-------+  +-----+------+  |
|         |                |                 |         |
|  +------v----------------v-----------------v------+  |
|  |            Demotion / Promotion Engine          |  |
|  +--------------------+-------------------------+    |
|                       |                              |
|  +--------------------v--------------------------+   |
|  |              Archive Store                     |  |
|  |  (CPU DRAM: token IDs, positions, embeddings, |  |
|  |   optionally saved KV tensors, block index)   |  |
|  +------------------------------------------------+  |
+----------------------+------------------------------+
                       |
+----------------------v------------------------------+
|              Inference Engine                        |
|         (llama.cpp KV cache + model)                |
|                                                      |
|  +------------------------------------------------+  |
|  |    Active KV Cache (GPU/system memory)         |  |
|  |    Budget-limited, contiguous positions         |  |
|  |    Standard attention computation               |  |
|  +------------------------------------------------+  |
+-----------------------------------------------------+
```

## 3. Components

### 3.1 Active Cache (managed by inference engine)

The standard KV cache maintained by llama.cpp, but with UNLEARN enforcing a budget: a maximum number of tokens allowed in the cache at any time. When the budget is exceeded, the Demotion Engine is triggered.

Properties:
- Stored in GPU/system memory (wherever the inference engine keeps it)
- Participates in every attention computation
- Positions are always contiguous integers (maintained by Position Manager)
- Budget is configurable (e.g., 4096, 8192, 16384 tokens)

### 3.2 Archive Store

A CPU-side data structure holding information about demoted tokens. Organized in blocks for efficient retrieval.

```python
@dataclass
class ArchiveBlock:
    block_id: int
    token_ids: list[int]
    original_positions: list[int]
    text: str
    representative_embedding: np.ndarray
    timestamp: int
    access_count: int
    kv_tensors: bytes              # saved KV cache tensors (required, not optional)
    kv_quantized: bool = False     # whether tensors are int8 quantized for storage
```

Block size is aligned with the scoring granularity (e.g., 128 tokens per block). Each block maintains a representative embedding computed from the mean of its token embeddings, used for retrieval matching.

### 3.3 Relevance Scorer

Runs every `score_interval` generation steps (default: 32). Assigns a relevance score in [0, 1] to each block in the active cache.

#### Scoring Signals

**Signal 1: Recency (always available)**
```
recency_score(block) = exp(-decay * (current_pos - block_max_pos) / context_length)
```
Tokens closer to the current generation position score higher. Decay rate controls how aggressively old content is penalized.

**Signal 2: Attention Sink (always available)**
```
sink_score(block) = 1.0 if block contains positions [0..sink_count-1] else 0.0
```
First N tokens (typically 4) are always retained per StreamingLLM's finding that attention sinks are structurally necessary.

**Signal 3: Semantic Coherence (requires embeddings)**
```
coherence_score(block) = cosine_similarity(block.embedding, recent_context_embedding)
```
How semantically related is this block to what's currently being discussed? `recent_context_embedding` is the mean embedding of the last K generated tokens.

**Signal 4: Query Relevance (requires attention weights, Phase 2+)**
```
attention_score(block) = mean(attention_weights[:, block_positions]) over recent queries
```
How much attention did recent queries actually pay to tokens in this block?

**Signal 5: Predictive Relevance (Phase 3+)**
```
predictive_score(block) = predict_future_topic(recent_trajectory) . block.embedding
```
Given the conversation's trajectory, predict what topics will come up next and score blocks by alignment.

#### Composite Score
```
relevance(block) = w_recency * recency + w_sink * sink + w_coherence * coherence + ...
```
Weights are configurable and can be learned from retrieval-miss feedback (Phase 4).

### 3.4 Demotion Engine

Triggered when active cache size exceeds the budget.

```
procedure demote():
    scores = scorer.score_all_blocks()
    blocks_to_demote = select_lowest(scores, count=overflow_blocks)

    for block in blocks_to_demote:
        archive.store(block)
        engine.kv_cache_seq_rm(block.start_pos, block.end_pos)

    position_manager.reindex()
```

Demotion policy options:
- Hard budget: demote as soon as budget is exceeded
- Watermark: demote when cache hits high watermark, demote down to low watermark (amortizes re-indexing)
- Gradual: demote the single lowest-scoring block every K steps

### 3.5 Retrieval Gate

Monitors for signals that archived content should be promoted back to the active cache.

**Semantic trigger (default):** compute similarity between current output embeddings and all archive block representatives. If any similarity exceeds a threshold, trigger retrieval.

**Perplexity trigger (Phase 2+):** a spike in generation perplexity suggests missing context. Search archive for relevant blocks.

**Explicit trigger:** user or application requests retrieval of earlier context.

### 3.6 Promotion Engine

Re-injects archived content into the active cache.

```
procedure promote(archive_block):
    # 1. Write saved KV tensors back to cache slots at their ORIGINAL positions
    engine.kv_cache_restore(archive_block.kv_tensors, archive_block.original_positions)

    # 2. Compute position delta to place them at the correct LOGICAL position
    target_logical = position_manager.compute_insertion_point(archive_block)
    delta = target_logical - archive_block.original_positions[0]

    # 3. Use seq_add to shift positions; deferred correction handles RoPE re-rotation
    engine.kv_cache_seq_add(archive_block.original_positions, delta)
    engine.memory_update()  # triggers build_rope_shift -> ggml_rope_ext_inplace

    # 4. Update position mappings
    position_manager.register_promoted(archive_block, target_logical)

    # 5. If budget exceeded, demote other low-relevance blocks
    if active_cache_size > budget:
        demote()
```

Promoted blocks are inserted at a logical position that preserves the original document order relative to other active blocks, maintaining semantic coherence. The saved KV tensors are exact originals with corrected positions, not re-computed approximations.

### 3.7 Position Manager

Maintains coherent, contiguous logical positions regardless of demotion/promotion activity.

```python
class PositionManager:
    position_map: dict[int, tuple[str, int]]  # logical_pos -> (source, original_pos)
    next_logical_pos: int

    def reindex(self):
        active_entries = sorted(engine.get_active_entries(), key=lambda e: e.original_pos)
        for i, entry in enumerate(active_entries):
            if entry.logical_pos != i:
                engine.kv_cache_seq_add(entry.logical_pos, entry.logical_pos + 1, i - entry.logical_pos)
                entry.logical_pos = i
        self.next_logical_pos = len(active_entries)
```

llama.cpp's `llama_kv_cache_seq_add` shifts position indices and handles RoPE re-rotation internally.

## 4. Data Flow

### 4.1 Normal Generation (no demotion/promotion needed)

1. User sends message
2. UNLEARN Manager tokenizes and forwards to engine
3. Engine processes tokens, appends to KV cache
4. Engine generates response tokens
5. Every score_interval steps: scorer runs, checks budget
6. If within budget: continue. If exceeded: trigger demotion.
7. Response returned to user

### 4.2 Demotion Event

1. Cache size exceeds budget
2. Scorer provides relevance scores for all blocks
3. Lowest-scoring blocks selected for demotion
4. Each block: save to archive, remove from KV cache via seq_rm
5. Reindex positions to contiguous via seq_add
6. Resume generation

### 4.3 Retrieval Event

1. Retrieval Gate detects similarity between current context and archived block
2. Select best matching archive block(s)
3. Determine insertion position (preserving document order)
4. Re-process archived tokens through the model at new positions
5. If budget exceeded after promotion, demote other low-relevance blocks
6. Resume generation with enriched context

## 5. Implementation Phases

### Phase 0: Infrastructure and Baselines
- Project structure, configuration, CLI
- Connect to llama.cpp on gpu-host (HOST)
- Verify KV cache manipulation APIs work (seq_rm, seq_add)
- Implement baseline strategies: full context, naive truncation, StreamingLLM-style (sinks + window)
- Set up benchmark suite (LongBench subset, needle-in-haystack, custom doc QA)
- Measure baseline quality and resource usage

### Phase 1: Demotion (scored eviction with archival)
- Implement block-based KV cache abstraction
- Implement Relevance Scorer (recency + sink + semantic coherence)
- Implement Demotion Engine (score, select, archive, evict, reindex)
- Implement Archive Store (in-memory, block-indexed)
- Test: does scored demotion beat naive truncation and StreamingLLM?
- Measure: quality at various budget levels (25%, 50%, 75% of full cache)

### Phase 2: Retrieval (promotion from archive)
- Implement Retrieval Gate (semantic trigger)
- Implement Promotion Engine (re-process archived tokens)
- Implement position-aware insertion (preserve document order)
- Test: can the system retrieve and use archived content?
- Measure: retrieval precision/recall, quality improvement over demotion-only

### Phase 3: Predictive Scoring
- Implement conversation trajectory tracking
- Train/implement predictive relevance signal
- Replace retrospective-only scoring with hybrid retrospective+predictive
- Test: does predictive scoring reduce retrieval-miss rate?

### Phase 4: Per-Head Management (if attention weights accessible)
- Analyze per-head attention patterns, classify heads
- Implement per-head eviction budgets
- Test: quality improvement at same total budget

### Paper: Continuous alongside implementation
- Introduction + related work (ready from RESEARCH.md)
- Method section tracks architecture evolution
- Experiments populated as benchmarks run
- Analysis from ablation studies

## 6. API Design

### UNLEARN Manager API

```python
class UnlearnManager:
    def __init__(self, engine: LlamaCppEngine, config: UnlearnConfig): ...

    def process_input(self, tokens: list[int]) -> None:
        """Process input tokens, managing cache budget."""

    def generate(self, max_tokens: int) -> str:
        """Generate with continuous cache management."""

    def get_cache_stats(self) -> CacheStats:
        """Current cache utilization, archive size, scores."""

    def force_demote(self, block_ids: list[int]) -> None:
        """Manually demote specific blocks."""

    def force_promote(self, block_ids: list[int]) -> None:
        """Manually promote specific archived blocks."""

    def get_relevance_scores(self) -> dict[int, float]:
        """Current relevance scores for all active blocks."""

    def get_archive_index(self) -> list[ArchiveBlockInfo]:
        """List of all archived blocks with metadata."""

    def get_event_log(self) -> list[UnlearnEvent]:
        """Log of all demotion/promotion events."""


class UnlearnConfig:
    max_active_tokens: int = 8192
    score_interval: int = 32
    block_size: int = 128
    sink_count: int = 4
    recency_decay: float = 0.01
    demotion_policy: str = "watermark"
    high_watermark: float = 0.95
    low_watermark: float = 0.75
    retrieval_threshold: float = 0.7
    max_retrieve_blocks: int = 4
    w_recency: float = 0.4
    w_sink: float = 1.0
    w_coherence: float = 0.6
    quantize_archive: bool = False
    max_archive_blocks: int = 1024
```

### Engine Interface

```python
class LlamaCppEngine(Protocol):
    """Interface UNLEARN expects from the inference engine."""

    def process_tokens(self, tokens: list[int]) -> None:
        """Process tokens, appending to KV cache."""

    def generate_next(self) -> int:
        """Generate one token."""

    def get_kv_cache_size(self) -> int:
        """Number of tokens in KV cache."""

    def kv_cache_seq_rm(self, pos_start: int, pos_end: int) -> None:
        """Remove KV entries in position range [start, end)."""

    def kv_cache_seq_add(self, pos_start: int, pos_end: int, delta: int) -> None:
        """Shift positions in range by delta (handles RoPE re-rotation)."""

    def get_embeddings(self, tokens: list[int]) -> np.ndarray:
        """Get token embeddings (for semantic scoring)."""

    def kv_cache_export(self, pos_start: int, pos_end: int) -> bytes:
        """Export KV tensors for archival. Optional."""

    def kv_cache_import(self, data: bytes, target_pos: int) -> None:
        """Restore archived KV tensors at target position. Optional."""
```

## 7. Key Design Decisions

### Why blocks, not individual tokens?
1. FlashAttention operates on blocks (16-128 tokens). Individual token eviction is incompatible.
2. Semantic coherence: a sentence fragment is meaningless; a 128-token block preserves meaning.
3. Reduced scoring overhead: scoring 100 blocks is cheaper than scoring 12,800 tokens.
4. Reduced archive index size: one representative embedding per block, not per token.

### Why save KV tensors instead of re-processing tokens?
Re-processing tokens in a different context produces DIFFERENT KV vectors (transformer attention is context-dependent). Saved KV tensors preserve the exact original representations, which actually encode information from tokens that may have since been evicted (a feature, not a bug, because causal attention bakes preceding context into each token's KV vectors).

Restoration uses llama.cpp's existing deferred RoPE correction: write saved tensors to cache slots, set positions to originals, call `seq_add` with the position delta, and `memory_update()` applies the corrective rotation via `ggml_rope_ext_inplace`.

Storage cost for Qwen 2.5 7B (28 layers, 4 GQA KV heads, head_dim 128): ~56KB per token, ~7MB per 128-token block, ~5.5GB for a full 100K document archive in CPU DRAM. Reducible to ~2.7GB with int8 quantization.

### Why contiguous position re-indexing instead of sparse positions?
Non-contiguous positions are catastrophic for RoPE models (arXiv 2511.04686). The corrective rotation needed for re-indexing is already implemented in llama.cpp's `llama_kv_cache_seq_add`. The overhead is O(remaining_tokens x layers x heads x head_dim) per reindex event, amortized by batching demotions.

### Why not modify the attention mechanism itself (like FoX)?
FoX's learned forget gates require training from scratch. UNLEARN works with any existing pretrained model, no fine-tuning required. Immediately deployable.

## 8. Expected Outcomes

**Quality target:** at 25% cache budget (keeping only 1/4 of the original context), UNLEARN should match or exceed StreamingLLM and H2O on LongBench benchmarks.

**Advantage over StreamingLLM:** StreamingLLM keeps recent context and discards everything else. UNLEARN keeps RELEVANT context regardless of recency, and can retrieve archived content. For document QA where the answer is in the middle of a long document (not at the end), UNLEARN should dramatically outperform StreamingLLM.

**Advantage over H2O:** H2O scores by cumulative attention and has no retrieval path. UNLEARN uses semantic coherence (not just attention), scores predictively, and can retrieve evicted content. For multi-turn conversations where topic shifts cause previously evicted content to become relevant, UNLEARN recovers while H2O cannot.

**Advantage over InfLLM:** InfLLM uses a fixed sliding window for eviction and simple representative-token matching for retrieval. UNLEARN uses learned relevance scoring (not a fixed window) and semantic embedding matching (richer than attention-max representative tokens). UNLEARN should make fewer retrieval-miss errors.
