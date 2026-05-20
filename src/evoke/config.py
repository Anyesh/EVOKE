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
    # 0.0 disables the attention term entirely; when an AttentionScorer is
    # wired in (post-#30/#31), bump this to ~0.5 so the model's actual
    # attention dominates while recency and coherence remain stability priors.
    w_attention: float = 0.0

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
