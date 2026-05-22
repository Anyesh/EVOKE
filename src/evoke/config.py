from __future__ import annotations

from dataclasses import dataclass


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
    # Weight on the attention-from-model signal in the multi-signal scorer.
    # 0.0 disables the attention term entirely; setting > 0 makes Session
    # construct an AttentionScorer that streams per-layer attention weights
    # from the EVOKE-built llama.cpp into a per-block sliding window. The
    # model's actual attention then dominates while recency and coherence
    # remain stability priors.
    w_attention: float = 0.0
    # Which transformer layer to tap for attention capture. Deep layers
    # (close to but not at the output) encode the most semantically loaded
    # attention patterns. Qwen 2.5 7B has 28 layers; ~20 (3/4 depth) is a
    # reasonable default. Configurable so other models can be tuned.
    attention_capture_layer: int = 20
    # Sliding window: number of recent decode steps the attention scorer
    # remembers per block. Decay applies exponentially across the window.
    attention_window: int = 64
    attention_decay: float = 0.95
    # H2O uses cumulative attention without decay, where EVOKE's default uses
    # a sliding-window EWMA. Setting "cumulative" makes AttentionScorer.score
    # return per-block cumulative attention mass normalized against the running
    # max across blocks (so the value stays in [0, 1]) instead of the EWMA
    # over the last n_window steps. Other policies must leave this at "ewma"
    # to preserve the multi-signal recipe.
    attention_score_mode: str = "ewma"
    # H2O protects a recent window R against eviction unconditionally (their
    # default is 10% of cache budget). Setting > 0 excludes any block whose
    # logical_end falls inside the last int(max_active_tokens *
    # recent_tail_protect_frac) positions of the cache from the eviction
    # candidate set, so heavy-hitter selection isn't confounded by recency
    # pruning of mid-cache blocks. Default 0.0 preserves existing EVOKE
    # behavior (no unconditional tail guard; recency is a soft signal only).
    recent_tail_protect_frac: float = 0.0
    # How to derive the block's representative embedding for smart-recovery
    # similarity scoring. "mean" averages all non-zero token embeddings in
    # the block (the default because the block-defining topic terms dominate
    # the average, giving real discriminative power on retrieval workloads).
    # "last_token" reads only the last token's hidden state — cheap but the
    # value reflects whatever happens to be at the block boundary, which is
    # often a partial sentence from neighboring content; kept available as
    # the ablation cell.
    block_embedding_strategy: str = "mean"
    # Minimum cosine similarity required to fire a smart-recovery for an
    # evicted block. Default 0.0 means the policy always recovers top-k
    # (existing behavior); setting > 0 gates recovery so weak matches don't
    # pollute the cache with off-topic blocks. Tuned on NIAH at depth=90,
    # where unconditional top-4 recovery brought back unrelated haystack
    # blocks that drowned out the already-resident needle.
    smart_recover_min_similarity: float = 0.0
    # Top-K bound for smart-recovery. Default 4 matches the production
    # Session policy; setting 0 disables recovery entirely.
    smart_recover_k: int = 4

    eviction_policy: str = "watermark"
    high_watermark: float = 0.95
    low_watermark: float = 0.75

    pin_generated: bool = True

    conversation_score_floor: float = 0.6
    assistant_score_floor: float = 0.5
    context_history_size: int = 5

    # Task-boundary-aware coherence. The scorer maintains a single "task focus"
    # embedding instead of a rolling average. When a new user message arrives,
    # cosine(new, focus) below `task_boundary_threshold` is interpreted as a
    # topic shift and the focus snaps to the new message — old blocks lose
    # coherence and become eviction candidates immediately. Otherwise the focus
    # updates via exponential moving average with `task_focus_ema_alpha`
    # weight on the prior focus. A harness can force a reset via the explicit
    # signal `RelevanceScorer.signal_task_boundary()`.
    task_boundary_threshold: float = 0.3
    task_focus_ema_alpha: float = 0.7

    recovery_mode: str = "discard"
    # On hybrid (Mamba + Attention) memory models, mid-cache eviction of the
    # <think>...</think> range is impossible (the recurrent half rejects
    # partial-rollback ranges strictly before the head). With strip enabled
    # the cached state diverges from what the client echoes back (which has
    # the thinking stripped) and the session resets on every assistant turn.
    # When this flag is True the server returns the full assistant content
    # including the thinking trace; the client echoes it back verbatim;
    # cached stays aligned naturally; no session reset. Trade-off: clients
    # see the thinking text in the response and must strip it on their side
    # if they want to hide it from the user. Default False preserves the
    # pure-attention behavior (strip + evict + risk-of-reset).
    suppress_thinking_strip: bool = False
    # Host-RAM budget for the kv_restore backend. When set, the saved-block
    # pool is bounded; oldest saved blocks lose their K/V bytes (kept as
    # breadcrumbs only) when adding a new save would exceed the budget. None
    # means unbounded — fine for short-lived sessions, a leak for long-running
    # multi-session servers. For Qwen 2.5 7B at ~56 KiB/token, 4 GiB holds
    # roughly 70K tokens of evicted history; tune to your VRAM + workload.
    kv_restore_ram_budget_bytes: int | None = None
    # When set, blocks that would otherwise be LRU-dropped from RAM are
    # instead spilled to this directory on disk. Recovery reads them back
    # (slower than RAM by the NVMe latency penalty — typically ~0.5-2 ms
    # extra) but the block stays recoverable. None disables the spill tier;
    # the LRU fallback drops bytes entirely (current behavior). A typical
    # value is "/dev/shm/evoke_spill" or "C:\\tmp\\evoke_spill".
    kv_restore_spill_path: str | None = None
