from __future__ import annotations

import math
from collections import deque

import numpy as np

from evoke.config import EvokeConfig
from evoke.types import ActiveBlock, BlockSource


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class RelevanceScorer:
    def __init__(self, config: EvokeConfig):
        self._config = config
        self._context_history: deque[np.ndarray] = deque(
            maxlen=config.context_history_size
        )
        self._recent_embedding: np.ndarray | None = None

    def update_recent_context(self, embedding: np.ndarray) -> None:
        self._recent_embedding = embedding
        self._context_history.append(embedding)

    def score(self, block: ActiveBlock, current_pos: int, context_length: int) -> float:
        recency = self._score_recency(block, current_pos, context_length)
        sink = self._score_sink(block)
        coherence = self._score_coherence(block)

        cfg = self._config
        if sink >= 1.0:
            return 1.0

        total_weight = cfg.w_recency + cfg.w_coherence
        if total_weight == 0:
            raw = recency
        else:
            raw = (cfg.w_recency * recency + cfg.w_coherence * coherence) / total_weight

        if block.source == BlockSource.USER:
            raw = max(raw, cfg.conversation_score_floor)
        elif block.source == BlockSource.ASSISTANT:
            raw = max(raw, cfg.assistant_score_floor)

        return raw

    def score_blocks(
        self, blocks: list[ActiveBlock], current_pos: int, context_length: int
    ) -> dict[int, float]:
        return {b.block_id: self.score(b, current_pos, context_length) for b in blocks}

    def _score_recency(
        self, block: ActiveBlock, current_pos: int, context_length: int
    ) -> float:
        if context_length == 0:
            return 1.0
        distance = (current_pos - block.logical_end) / context_length
        return math.exp(-self._config.recency_decay * distance * context_length)

    def _score_sink(self, block: ActiveBlock) -> float:
        return 1.0 if block.is_sink else 0.0

    def _score_coherence(self, block: ActiveBlock) -> float:
        if not self._context_history or block.representative_embedding is None:
            return 0.5
        best_sim = max(
            cosine_similarity(block.representative_embedding, ctx_emb)
            for ctx_emb in self._context_history
        )
        return (best_sim + 1.0) / 2.0
