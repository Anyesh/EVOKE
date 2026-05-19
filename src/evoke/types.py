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
