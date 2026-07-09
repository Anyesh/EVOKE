from __future__ import annotations

import math
from collections import deque
from typing import Protocol

import numpy as np

from evoke.config import EvokeConfig
from evoke.types import ActiveBlock, BlockSource


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class AttentionScorerProtocol(Protocol):
    # Pluggable attention-signal source. Returns a per-block score in [0, 1]
    # derived from the model's actual attention weights aggregated over a
    # sliding window of recent decode steps. None means "no signal yet" (the
    # block has not been attended to since capture started) and is treated as
    # neutral; the scorer falls back to recency + coherence for those blocks.
    # Implemented in evoke/attention_scorer.py once #30 (C primitive) lands.
    def score(self, block: ActiveBlock) -> float | None: ...


class RelevanceScorer:
    def __init__(
        self,
        config: EvokeConfig,
        attention_scorer: AttentionScorerProtocol | None = None,
        jlens_scorer: AttentionScorerProtocol | None = None,
    ):
        self._config = config
        self._attention_scorer = attention_scorer
        self._jlens_scorer = jlens_scorer
        # Task-boundary-aware coherence: instead of averaging the last N
        # message embeddings (which lags on topic shifts and conflates two
        # tasks running in the same session), maintain a single "task focus"
        # embedding. New messages either update the focus via EMA or snap
        # it to themselves on detected/signaled task boundaries.
        self._task_focus: np.ndarray | None = None
        self._recent_embedding: np.ndarray | None = None
        self._force_boundary = False
        # Kept for ABI compatibility with code that introspects scorer state
        # for debugging; not used in the score path anymore.
        self._context_history: deque[np.ndarray] = deque(maxlen=config.context_history_size)

    def signal_task_boundary(self) -> None:
        # Harness-driven explicit reset. The next update_recent_context call
        # will snap the task focus to the incoming embedding rather than
        # EMA-blending it. Used by Session.sync_prefix when the request has
        # evoke_task_boundary=true (or when an [evoke:task_boundary] system
        # message is detected).
        self._force_boundary = True

    def update_recent_context(self, embedding: np.ndarray) -> None:
        self._recent_embedding = embedding
        self._context_history.append(embedding)
        if self._task_focus is None or self._force_boundary:
            self._task_focus = embedding.copy()
            self._force_boundary = False
            return
        sim = cosine_similarity(embedding, self._task_focus)
        if sim < self._config.task_boundary_threshold:
            # Implicit boundary detected: the new message is semantically
            # unrelated to the running focus. Snap to the new focus so prior
            # blocks (coherent with the old task) lose their coherence score
            # and become eviction candidates within one or two scoring passes.
            self._task_focus = embedding.copy()
            return
        alpha = self._config.task_focus_ema_alpha
        self._task_focus = alpha * self._task_focus + (1.0 - alpha) * embedding

    def set_attention_scorer(self, attention_scorer: AttentionScorerProtocol | None) -> None:
        self._attention_scorer = attention_scorer

    def set_jlens_scorer(self, jlens_scorer: AttentionScorerProtocol | None) -> None:
        self._jlens_scorer = jlens_scorer

    def score(self, block: ActiveBlock, current_pos: int, context_length: int) -> float:
        cfg = self._config
        if self._score_sink(block) >= 1.0:
            return 1.0

        recency = self._score_recency(block, current_pos, context_length)
        coherence = self._score_coherence(block)
        attn = self._score_attention(block)
        jlens = self._score_jlens(block)
        # Recovery-aware term: the model already signaled this block matters
        # by recovering it; protect it from eviction until the decay schedule
        # in tick_turn() has thinned the signal back to noise.
        recovery = block.recovery_strength

        # Multi-signal combination. Model-derived signals (attention: what
        # the model actually attended to recently; jlens: whether the block
        # holds workspace content later computation reads from) join the
        # weighted sum only when their scorer returned a value for this
        # block, so blocks without a signal yet fall back to the recency +
        # coherence stability priors. Each absent signal also leaves the
        # denominator, which reproduces the historical two-branch behavior
        # exactly when only attention exists and keeps default weights
        # (w_attention=0, w_jlens=0, w_recovery=0) a strict no-op.
        parts = [
            (cfg.w_recency, recency),
            (cfg.w_coherence, coherence),
            (cfg.w_recovery, recovery),
        ]
        if attn is not None and cfg.w_attention > 0:
            parts.append((cfg.w_attention, attn))
        if jlens is not None and cfg.w_jlens > 0:
            parts.append((cfg.w_jlens, jlens))
        total = sum(w for w, _ in parts)
        if total == 0:
            raw = recency
        else:
            raw = sum(w * v for w, v in parts) / total

        # Source-type floors: USER and ASSISTANT turns are conversation
        # backbone; even when their coherence drops they shouldn't be evicted
        # before lower-floor DOCUMENT blocks.
        if block.source == BlockSource.USER:
            raw = max(raw, cfg.conversation_score_floor)
        elif block.source == BlockSource.ASSISTANT:
            raw = max(raw, cfg.assistant_score_floor)

        # Harness priority is a final multiplier — a hint from the caller that
        # this block matters more (priority > 1.0) or less (priority < 1.0)
        # than the model+heuristic signals alone would suggest. Capped at 1.0
        # so even very high priority cannot out-rank sinks. Pinned blocks are
        # excluded from eviction candidates upstream (in
        # EvokeManager._evictable_blocks), so priority is independent of pin.
        scaled = raw * block.priority
        return min(scaled, 1.0)

    def _score_attention(self, block: ActiveBlock) -> float | None:
        if self._attention_scorer is None:
            return None
        return self._attention_scorer.score(block)

    def _score_jlens(self, block: ActiveBlock) -> float | None:
        if self._jlens_scorer is None:
            return None
        return self._jlens_scorer.score(block)

    def score_blocks(
        self, blocks: list[ActiveBlock], current_pos: int, context_length: int
    ) -> dict[int, float]:
        return {b.block_id: self.score(b, current_pos, context_length) for b in blocks}

    def _score_recency(self, block: ActiveBlock, current_pos: int, context_length: int) -> float:
        if context_length == 0:
            return 1.0
        distance = (current_pos - block.logical_end) / context_length
        return math.exp(-self._config.recency_decay * distance * context_length)

    def _score_sink(self, block: ActiveBlock) -> float:
        return 1.0 if block.is_sink else 0.0

    def _score_coherence(self, block: ActiveBlock) -> float:
        if self._task_focus is None or block.representative_embedding is None:
            return 0.5
        sim = cosine_similarity(block.representative_embedding, self._task_focus)
        return (sim + 1.0) / 2.0
