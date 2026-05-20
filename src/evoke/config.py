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
    # Host-RAM budget for the kv_restore backend. When set, the saved-block
    # pool is bounded; oldest saved blocks lose their K/V bytes (kept as
    # breadcrumbs only) when adding a new save would exceed the budget. None
    # means unbounded — fine for short-lived sessions, a leak for long-running
    # multi-session servers. For Qwen 2.5 7B at ~56 KiB/token, 4 GiB holds
    # roughly 70K tokens of evicted history; tune to your VRAM + workload.
    kv_restore_ram_budget_bytes: int | None = None
