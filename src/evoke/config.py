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
    # to preserve the multi-signal recipe. Setting "snapkv" runs SnapKV
    # (Liu et al., NeurIPS 2024): the scorer ignores per-step decode
    # attention and instead freezes a one-shot snapshot of attention from
    # the last `snapkv_observation_window` query tokens of the most recent
    # process_user_message call. Eviction during the subsequent generate
    # uses that frozen snapshot, matching SnapKV's "compress once per
    # prompt" policy.
    attention_score_mode: str = "ewma"
    # SnapKV's observation window: number of trailing prompt tokens whose
    # attention to prior keys defines block importance. The paper's default
    # is 32. Only meaningful when attention_score_mode == "snapkv". The
    # capture buffer must hold n_layers * obs_window * n_heads * n_kv f32
    # entries — the default 8M elements covers obs_window=32 across all
    # benchmark budgets up to ~8K tokens of n_kv for Qwen 2.5 7B.
    snapkv_observation_window: int = 32
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
    # Use a dedicated retrieval embedding model (bge-small-en-v1.5 via
    # fastembed) for block representative embeddings and probe query
    # embeddings, instead of the LM's intermediate hidden states. LM
    # hidden states have a common-mode dominance that collapses cosine
    # discrimination to ~0.85-0.93 on retrieval-style workloads (NIAH);
    # retrieval-tuned embeddings widen the band to ~0.3-0.9 and let the
    # resident-gate actually separate needle from noise. Cost: ~30 MB
    # model loaded once, ~10 ms per text embedded. Off by default so
    # builds without fastembed installed keep working.
    use_retrieval_embeddings: bool = False

    smart_recover_before_decode: bool = True
    smart_recover_resident_gate: bool = True

    eviction_policy: str = "watermark"
    high_watermark: float = 0.95
    low_watermark: float = 0.75

    pin_generated: bool = True

    # Recovery-aware eviction. Closes the recover-then-immediately-evict
    # thrash exposed by the session-length sweep at T=28 (5-seed CIs disjoint:
    # evoke 68.06s [67.23, 68.89] vs truncate 65.14s [64.35, 65.94], driven
    # by 80 redundant evictions at ~30 ms each). On each recover() the new
    # block's recovery_strength is set to recovery_strength_init; on each
    # per-turn tick_turn() the strength is multiplied by recovery_decay; the
    # scorer adds w_recovery * recovery_strength to the weighted score, so
    # a fresh recovery survives the eviction pass that fires later in the
    # same turn (when new content arrives and the watermark trips). The
    # block decays back to a normal eviction candidate over ~5 turns at
    # decay 0.7 (strength 0.7 -> 0.49 -> 0.34 -> 0.24 -> 0.17 -> ...).
    # Default w_recovery=0.0 preserves the pre-fix scorer behavior so
    # existing benches see no change until the bench explicitly opts in.
    w_recovery: float = 0.0
    recovery_strength_init: float = 1.0
    recovery_decay: float = 0.7
    # Hard eviction-exclusion for freshly recovered blocks. When > 0, a block
    # whose recovery_strength is at or above this threshold is never an eviction
    # candidate, so a just-recovered block (the active working set the agent
    # re-referenced) survives the eviction pass(es) of the turn it is used in
    # rather than being re-evicted at its old, low-recency position before the
    # model can attend to it. recovery_strength decays via tick_turn, so the
    # protection lifts after the block goes cold again. Default 0.0 keeps the
    # prior behavior (recovery contributes only through w_recovery's weighted sum).
    recovery_protect_threshold: float = 0.0

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

    # Position handling across eviction and recovery. "compact" (default, the
    # EVOKE design): eviction re-indexes survivors to stay contiguous
    # (seq_rm + seq_add) and recovery re-anchors a recalled block to a new
    # contiguous tail position via a per-cell RoPE shift. "sparse" (ArkVale-
    # like): eviction drops cells with seq_rm only, survivors keep their true
    # absolute positions (the axis grows holes), and recovery splices a block
    # back at its original index with zero RoPE re-anchoring. This is the one
    # axis that distinguishes EVOKE from ArkVale, so it must be switchable to
    # measure whether re-anchoring earns its place. Sparse is an experimental
    # measurement mode: it is not wired to the prefix-matching server path
    # (get_token_view assumes contiguous block order).
    position_mode: str = "compact"

    # Recovery trigger in the live server path (Session.sync_prefix). "identity"
    # (the north-star design) splices a saved block back in place when the client
    # re-sends its exact tokens at its original position: recompute-free, keyed on
    # content identity, never similarity. Requires position_mode="sparse" (holes
    # at original positions); it falls back to the "similarity" path otherwise.
    # "similarity" is the legacy _smart_recover cosine path, kept for ablation.
    recovery_match: str = "identity"

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
