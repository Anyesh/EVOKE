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
class ArchiveBlock:
    block_id: int
    token_ids: list[int]
    original_positions: list[int]
    text: str
    representative_embedding: np.ndarray
    timestamp: int
    access_count: int = 0
    source: BlockSource = BlockSource.DOCUMENT

    @property
    def size(self) -> int:
        return len(self.token_ids)

    @property
    def pos_start(self) -> int:
        return self.original_positions[0]

    @property
    def pos_end(self) -> int:
        return self.original_positions[-1] + 1


@dataclass
class ActiveBlock:
    block_id: int
    logical_start: int
    logical_end: int
    original_start: int
    original_end: int
    token_ids: list[int]
    representative_embedding: np.ndarray | None = None
    relevance_score: float = 1.0
    source: BlockSource = BlockSource.DOCUMENT
    promotion_step: int = -1


@dataclass
class CacheStats:
    active_tokens: int
    active_blocks: int
    archive_blocks: int
    archive_tokens: int
    budget: int
    budget_utilization: float
    total_demotions: int
    total_promotions: int
    total_retrieval_misses: int


@dataclass
class EvokeEvent:
    step: int
    event_type: str
    block_ids: list[int]
    details: dict = field(default_factory=dict)
