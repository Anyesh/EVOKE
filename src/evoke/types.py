from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class BlockSource(str, Enum):
    SYSTEM = "system"
    DOCUMENT = "document"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ActiveBlock:
    block_id: int
    logical_start: int
    logical_end: int
    token_ids: list[int]
    representative_embedding: np.ndarray | None = None
    source: BlockSource = BlockSource.DOCUMENT
    is_sink: bool = False
    key: str = ""
    # Harness-supplied hints. A harness (Claude Code, opencode, etc.) often
    # knows things the scorer can't infer from text alone: which file reads
    # are central to the current task, which tool output is ephemeral, which
    # blocks must never be evicted. priority multiplies the relevance score
    # at evict time; pinned removes the block from eviction candidates
    # entirely (alongside is_sink and current-turn-pin).
    priority: float = 1.0
    pinned: bool = False
    # Recovery-aware eviction: tracks "the model recently signaled this
    # block matters" via the smart_recover path. Set to recovery_strength_init
    # on each recover() call and decayed by recovery_decay on each per-turn
    # tick_turn(). The scorer adds w_recovery * recovery_strength to the
    # weighted relevance score so a freshly-recovered block survives the
    # next eviction pass that would otherwise toss it on recency alone.
    # Without this signal the recover-then-immediately-evict thrash that
    # the session-length sweep diagnosed at T=28 (80 redundant evictions
    # = ~2.4s of pure churn) reappears on every long session.
    recovery_strength: float = 0.0

    @property
    def size(self) -> int:
        return len(self.token_ids)


@dataclass
class CacheStats:
    active_tokens: int
    active_blocks: int
    budget: int
    budget_utilization: float
    total_evictions: int
    total_recoveries: int = 0


@dataclass
class EvokeEvent:
    step: int
    event_type: str
    block_ids: list[int]
    details: dict = field(default_factory=dict)
