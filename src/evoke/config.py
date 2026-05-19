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

    demotion_policy: str = "watermark"
    high_watermark: float = 0.95
    low_watermark: float = 0.75

    retrieval_threshold: float = 0.85
    max_retrieve_blocks: int = 2

    max_archive_blocks: int = 1024

    pin_generated: bool = True
