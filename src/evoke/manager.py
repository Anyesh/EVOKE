from __future__ import annotations

import numpy as np

from evoke.config import EvokeConfig
from evoke.engine import InferenceEngine
from evoke.position import PositionManager
from evoke.recovery import Breadcrumb, make_recovery_backend
from evoke.scorer import RelevanceScorer
from evoke.types import ActiveBlock, BlockSource, CacheStats, EvokeEvent


class EvokeManager:
    def __init__(
        self,
        engine: InferenceEngine,
        config: EvokeConfig | None = None,
        *,
        attention_scorer=None,
    ):
        self._engine = engine
        self._config = config or EvokeConfig()
        self._attention_scorer = attention_scorer
        self._scorer = RelevanceScorer(self._config, attention_scorer=attention_scorer)
        self._recovery = make_recovery_backend(
            self._config.recovery_mode,
            engine,
            kv_restore_ram_budget_bytes=self._config.kv_restore_ram_budget_bytes,
            kv_restore_spill_path=self._config.kv_restore_spill_path,
        )
        self._positions = PositionManager()
        self._events: list[EvokeEvent] = []
        self._step = 0
        self._total_evictions = 0
        self._total_recoveries = 0
        self._next_block_id = 0
        self._current_turn_start_id = 0

    def _absorb_attention(self) -> None:
        # Hook called after every engine decode (process_tokens or
        # generate_next) to pull the last-decode attention weights into the
        # AttentionScorer's per-block sliding window. No-op when attention
        # scorer is unconfigured.
        if self._attention_scorer is not None:
            self._attention_scorer.absorb_last_decode(self._positions.active_blocks)

    def _new_block_id(self) -> int:
        bid = self._next_block_id
        self._next_block_id += 1
        return bid

    def load_document(
        self,
        text: str,
        *,
        priority: float = 1.0,
        pinned: bool = False,
    ) -> None:
        tokens = self._engine.tokenize(text)
        blocks: list[ActiveBlock] = []
        block_size = self._config.block_size

        for i in range(0, len(tokens), block_size):
            chunk = tokens[i : i + block_size]
            bid = self._new_block_id()
            block = ActiveBlock(
                block_id=bid,
                logical_start=i,
                logical_end=i + len(chunk),
                token_ids=chunk,
                source=BlockSource.DOCUMENT,
                is_sink=i < self._config.sink_count,
                key=f"doc#{bid}",
                priority=priority,
                pinned=pinned,
            )
            blocks.append(block)

        self._engine.process_tokens(tokens)
        self._compute_block_embeddings(blocks)
        self._positions.register_blocks(blocks)
        self._absorb_attention()

        self._current_turn_start_id = self._next_block_id
        self._enforce_budget()

    def add_context(
        self,
        text: str,
        key: str,
        *,
        priority: float = 1.0,
        pinned: bool = False,
    ) -> None:
        tokens = self._engine.tokenize(text)
        self.add_context_tokens(tokens, key, priority=priority, pinned=pinned)

    def add_context_tokens(
        self,
        tokens: list[int],
        key: str,
        *,
        priority: float = 1.0,
        pinned: bool = False,
    ) -> None:
        if not tokens:
            return
        self._current_turn_start_id = self._next_block_id
        start = self._engine.next_write_pos
        self._engine.process_tokens(tokens)

        block_size = self._config.block_size
        for i in range(0, len(tokens), block_size):
            chunk = tokens[i : i + block_size]
            bstart = start + i
            block = ActiveBlock(
                block_id=self._new_block_id(),
                logical_start=bstart,
                logical_end=bstart + len(chunk),
                token_ids=chunk,
                source=BlockSource.DOCUMENT,
                is_sink=bstart < self._config.sink_count,
                key=f"{key}#{i // block_size}",
                representative_embedding=self._last_token_embedding(bstart, len(chunk)),
                priority=priority,
                pinned=pinned,
            )
            self._positions.append_block(block, bstart)
        self._absorb_attention()

        self._enforce_budget()

    def generate(
        self,
        max_tokens: int,
        stop_token_ids: set[int] | None = None,
        *,
        think_close: str | None = None,
        thinking_budget: int = 16384,
        answer_budget: int = 512,
    ) -> str:
        eos = self._engine.eos_token
        stop = stop_token_ids or set()

        if think_close is not None:
            return self._generate_thinking(
                think_close, thinking_budget, answer_budget, stop, eos
            )

        gen_start = self._engine.next_write_pos
        n_ctx = self._engine.n_ctx
        output_tokens: list[int] = []
        for _ in range(max_tokens):
            # Stop before llama_decode runs out of cache slots. The user may
            # pass max_tokens larger than the remaining n_ctx (and some models
            # don't emit eos naturally on the prompts we use); without this
            # guard the next generate_next would crash with "no KV slot".
            if self._engine.next_write_pos + 1 >= n_ctx:
                break
            token = self._engine.generate_next()
            output_tokens.append(token)
            self._step += 1
            if token == eos or token in stop:
                break

        if output_tokens:
            self._track_generated_block(output_tokens, gen_start)
            self._absorb_attention()
        self._enforce_budget()
        return self._engine.detokenize(output_tokens)

    def _generate_thinking(
        self,
        think_close: str,
        thinking_budget: int,
        answer_budget: int,
        stop: set[int],
        eos: int,
    ) -> str:
        gen_start = self._engine.next_write_pos
        output_tokens: list[int] = []
        in_thinking = True
        answer_tokens_generated = 0
        answer_start_idx: int | None = None
        close_tokens = self._engine.tokenize(think_close)

        for _ in range(thinking_budget + answer_budget):
            token = self._engine.generate_next()
            output_tokens.append(token)
            self._step += 1

            if token == eos:
                break

            if in_thinking:
                if (
                    len(output_tokens) >= len(close_tokens)
                    and output_tokens[-len(close_tokens) :] == close_tokens
                ):
                    in_thinking = False
                    answer_start_idx = len(output_tokens)
            else:
                answer_tokens_generated += 1
                if token in stop or answer_tokens_generated >= answer_budget:
                    break

        # The thinking trace must not persist: a 16k-token reasoning span would
        # blow the budget, and a model's chat template strips reasoning from
        # history. Keep only the answer; evict the thinking span from the cache.
        end = self._engine.next_write_pos
        if answer_start_idx is not None and answer_start_idx < len(output_tokens):
            answer_tokens = output_tokens[answer_start_idx:]
            answer_pos = gen_start + answer_start_idx
            self._track_generated_block(answer_tokens, answer_pos)
            self._engine.evict_ranges([(gen_start, answer_pos)])
            self._positions.recompact()
        elif output_tokens:
            self._engine.evict_ranges([(gen_start, end)])
            self._positions.recompact()

        self._enforce_budget()
        return self._engine.detokenize(output_tokens)

    def process_user_message(self, text: str) -> None:
        self._current_turn_start_id = self._next_block_id

        msg_start = self._engine.next_write_pos
        tokens = self._engine.tokenize(text)
        self._engine.process_tokens(tokens)
        end_pos = self._engine.next_write_pos

        self._track_conversation_block(tokens, msg_start)
        self._update_recent_context_embedding(tokens, end_pos)
        self._enforce_budget()

    def get_stats(self) -> CacheStats:
        active_tokens = self._positions.active_token_count
        budget = self._config.max_active_tokens
        return CacheStats(
            active_tokens=active_tokens,
            active_blocks=len(self._positions.active_blocks),
            budget=budget,
            budget_utilization=active_tokens / budget if budget > 0 else 0,
            total_evictions=self._total_evictions,
            total_recoveries=self._total_recoveries,
        )

    def get_event_log(self) -> list[EvokeEvent]:
        return list(self._events)

    def get_breadcrumbs(self) -> list[Breadcrumb]:
        return self._recovery.list_evicted()

    def signal_task_boundary(self) -> None:
        # Forwarded by the server when a request sets evoke_task_boundary=true
        # or includes an [evoke:task_boundary] system message. The next
        # update_recent_context call will snap the task focus to the incoming
        # user message instead of blending; prior-topic blocks lose their
        # coherence score within one scoring pass and become evictable.
        self._scorer.signal_task_boundary()

    def get_token_view(self) -> list[int]:
        # Tokens currently in the engine cache, in physical position order.
        # The manager keeps its blocks aligned with engine state through every
        # eviction (engine.evict_ranges) and recovery (engine.kv_block_load),
        # so concatenating the active blocks' token_ids in block_id order is
        # the authoritative view of what the engine actually has cached. Used
        # by Session.sync_prefix so prefix-match compares the new prompt
        # against the real engine state rather than a stale extends-only list.
        tokens: list[int] = []
        for block in self._positions.active_blocks:
            tokens.extend(block.token_ids)
        return tokens

    def trim_blocks_at(self, position: int) -> None:
        # Truncate manager blocks so no content remains at or after
        # `position`. Used when Session.sync_prefix detects mid-cache
        # divergence (typical for truncate-policy sessions where old history
        # has been evicted and the next request resupplies it): the engine is
        # tail-evicted, and we mirror that on the manager side so block
        # boundaries continue to match physical positions.
        new_blocks: list[ActiveBlock] = []
        for block in self._positions.active_blocks:
            if block.logical_end <= position:
                new_blocks.append(block)
            elif block.logical_start >= position:
                continue
            else:
                keep = position - block.logical_start
                block.token_ids = block.token_ids[:keep]
                block.logical_end = position
                # Representative embedding was the last-token embedding of
                # the block as decoded; after truncation it is stale.
                block.representative_embedding = None
                new_blocks.append(block)
        self._positions._active_blocks = new_blocks

    def get_relevance_scores(self) -> dict[int, float]:
        blocks = self._positions.active_blocks
        pos = self._positions.next_logical_pos
        return self._scorer.score_blocks(blocks, pos, pos)

    def force_evict(self, block_ids: list[int]) -> None:
        self._evict_blocks(set(block_ids))

    def recover(self, key: str) -> bool:
        saved = self._recovery.take(key)
        if saved is None:
            return False

        new_p0 = self._engine.next_write_pos
        if not self._engine.kv_block_load(saved.kv_bytes, new_p0):
            return False

        bid = self._new_block_id()
        block = ActiveBlock(
            block_id=bid,
            logical_start=new_p0,
            logical_end=new_p0 + len(saved.token_ids),
            token_ids=saved.token_ids,
            representative_embedding=saved.representative_embedding,
            source=saved.source,
            key=saved.key,
        )
        self._positions.append_block(block, new_p0)
        self._total_recoveries += 1
        self._events.append(
            EvokeEvent(step=self._step, event_type="recovery", block_ids=[bid])
        )
        self._enforce_budget()
        return True

    def _enforce_budget(self) -> None:
        cfg = self._config
        active_tokens = self._positions.active_token_count

        if cfg.eviction_policy == "watermark":
            threshold = int(cfg.max_active_tokens * cfg.high_watermark)
            target = int(cfg.max_active_tokens * cfg.low_watermark)
        else:
            threshold = cfg.max_active_tokens
            target = cfg.max_active_tokens

        if active_tokens <= threshold:
            return

        tokens_to_free = active_tokens - target
        blocks = self._positions.active_blocks
        pos = self._positions.next_logical_pos
        scores = self._scorer.score_blocks(blocks, pos, pos)

        candidates = self._evictable_blocks(blocks, scores)
        candidates.sort(key=lambda b: scores.get(b.block_id, 0.0))

        to_evict: list[ActiveBlock] = []
        freed = 0
        for block in candidates:
            if freed >= tokens_to_free:
                break
            to_evict.append(block)
            freed += block.size

        if to_evict:
            self._evict_blocks({b.block_id for b in to_evict})

    def _evictable_blocks(
        self, blocks: list[ActiveBlock], scores: dict[int, float]
    ) -> list[ActiveBlock]:
        evictable = []
        for block in blocks:
            if block.is_sink or block.pinned:
                continue
            if scores.get(block.block_id, 0.0) >= 1.0:
                continue
            # pin_generated protects the model's just-decoded output (an
            # ASSISTANT block) from being immediately evicted by the same
            # _enforce_budget call that fires right after _track_generated_block.
            # It must NOT pin prompt-decoded blocks (source=DOCUMENT, added
            # via add_context_tokens) — doing so was the bug that made eviction
            # silently fail on every opencode-style turn whose tail happened
            # to be larger than the budget: all newly-added prompt blocks were
            # marked current-turn-pinned and the policy found zero candidates.
            if (
                self._config.pin_generated
                and block.source == BlockSource.ASSISTANT
                and block.block_id >= self._current_turn_start_id
            ):
                continue
            evictable.append(block)
        return evictable

    def _evict_blocks(self, block_ids: set[int]) -> None:
        blocks = [b for b in self._positions.active_blocks if b.block_id in block_ids]
        if not blocks:
            return

        self._recovery.on_evict(blocks, self._step)
        ranges = sorted((b.logical_start, b.logical_end) for b in blocks)
        self._engine.evict_ranges(ranges)
        self._positions.remove_blocks(block_ids)
        self._positions.recompact()

        # Drop attention windows for evicted blocks so the scorer's map
        # doesn't grow unbounded over long-running sessions. Block IDs are
        # never reused (monotonic counter) so this is safe.
        if self._attention_scorer is not None:
            for bid in block_ids:
                self._attention_scorer.forget(bid)

        self._total_evictions += len(blocks)
        self._events.append(
            EvokeEvent(
                step=self._step,
                event_type="eviction",
                block_ids=[b.block_id for b in blocks],
            )
        )

    def _track_conversation_block(self, tokens: list[int], start_pos: int) -> None:
        bid = self._new_block_id()
        block = ActiveBlock(
            block_id=bid,
            logical_start=start_pos,
            logical_end=start_pos + len(tokens),
            token_ids=tokens,
            source=BlockSource.USER,
            key=f"user#{bid}",
        )
        self._positions.append_block(block, start_pos)
        block.representative_embedding = self._last_token_embedding(
            start_pos, len(tokens)
        )

    def _track_generated_block(self, tokens: list[int], start_pos: int) -> None:
        bid = self._new_block_id()
        block = ActiveBlock(
            block_id=bid,
            logical_start=start_pos,
            logical_end=start_pos + len(tokens),
            token_ids=tokens,
            source=BlockSource.ASSISTANT,
            key=f"assistant#{bid}",
        )
        self._positions.append_block(block, start_pos)

    def _compute_block_embeddings(self, blocks: list[ActiveBlock]) -> None:
        try:
            total_tokens = sum(len(b.token_ids) for b in blocks)
            if total_tokens == 0:
                return
            embeddings = self._engine.get_embeddings(list(range(total_tokens)))
            for block in blocks:
                end = min(block.logical_end, len(embeddings))
                if end > 0:
                    block.representative_embedding = _normalize(embeddings[end - 1])
        except (NotImplementedError, RuntimeError):
            pass

    def _update_recent_context_embedding(self, tokens: list[int], end_pos: int) -> None:
        emb = self._last_token_embedding(end_pos - len(tokens), len(tokens))
        if emb is not None:
            self._scorer.update_recent_context(emb)

    def _last_token_embedding(self, start_pos: int, length: int) -> np.ndarray | None:
        if length <= 0:
            return None
        try:
            positions = list(range(start_pos, start_pos + length))
            embeddings = self._engine.get_embeddings(positions)
            return _normalize(embeddings[-1])
        except (NotImplementedError, RuntimeError):
            return None


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
