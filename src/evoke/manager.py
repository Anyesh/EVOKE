from __future__ import annotations

import numpy as np

from evoke.archive import ArchiveStore
from evoke.config import EvokeConfig
from evoke.engine import InferenceEngine
from evoke.position import PositionManager
from evoke.scorer import RelevanceScorer, cosine_similarity
from evoke.types import ActiveBlock, ArchiveBlock, BlockSource, CacheStats, EvokeEvent


class EvokeManager:
    def __init__(self, engine: InferenceEngine, config: EvokeConfig | None = None):
        self._engine = engine
        self._config = config or EvokeConfig()
        self._scorer = RelevanceScorer(self._config)
        self._archive = ArchiveStore(self._config, tokenize_fn=engine.tokenize)
        self._positions = PositionManager()
        self._events: list[EvokeEvent] = []
        self._step = 0
        self._total_demotions = 0
        self._total_promotions = 0
        self._total_recall_misses = 0
        self._document_token_count = 0
        self._current_turn_start = 0
        self._recent_query_text = ""

    def load_document(self, text: str) -> None:
        tokens = self._engine.tokenize(text)
        self._document_token_count = len(tokens)

        blocks: list[ActiveBlock] = []
        block_size = self._config.block_size

        for i in range(0, len(tokens), block_size):
            chunk = tokens[i : i + block_size]
            block_id = self._archive.allocate_id()

            block = ActiveBlock(
                block_id=block_id,
                logical_start=i,
                logical_end=i + len(chunk),
                original_start=i,
                original_end=i + len(chunk),
                token_ids=chunk,
            )
            blocks.append(block)

        self._engine.process_tokens(tokens)
        self._compute_block_embeddings(blocks)
        self._positions.register_blocks(blocks)

        self._current_turn_start = self._positions.next_logical_pos
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
            text = self._generate_thinking(
                think_close, thinking_budget, answer_budget, stop, eos
            )
            return text

        gen_start = self._engine.next_write_pos
        output_tokens: list[int] = []
        for _ in range(max_tokens):
            token = self._engine.generate_next()
            output_tokens.append(token)
            self._step += 1

            if self._step % self._config.score_interval == 0:
                self._scoring_round()

            if token == eos or token in stop:
                break

        if output_tokens:
            self._track_generated_block(output_tokens, gen_start)

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
        close_tokens = self._engine.tokenize(think_close)

        for _ in range(thinking_budget + answer_budget):
            token = self._engine.generate_next()
            output_tokens.append(token)
            self._step += 1

            if self._step % self._config.score_interval == 0:
                self._scoring_round()

            if token == eos:
                break

            if in_thinking:
                if (
                    len(output_tokens) >= len(close_tokens)
                    and output_tokens[-len(close_tokens) :] == close_tokens
                ):
                    in_thinking = False
            else:
                answer_tokens_generated += 1
                if token in stop or answer_tokens_generated >= answer_budget:
                    break

        if output_tokens:
            self._track_generated_block(output_tokens, gen_start)

        return self._engine.detokenize(output_tokens)

    def process_user_message(self, text: str, raw_query: str = "") -> None:
        self._check_rebuild()
        self._recent_query_text = raw_query or text
        self._current_turn_start = self._engine.next_write_pos

        recalled_blocks = self._retrieve_from_archive(self._recent_query_text)

        if recalled_blocks:
            self._promote_via_rebuild(recalled_blocks)

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
            archive_blocks=self._archive.size,
            archive_tokens=self._archive.total_tokens,
            budget=budget,
            budget_utilization=active_tokens / budget if budget > 0 else 0,
            total_demotions=self._total_demotions,
            total_promotions=self._total_promotions,
            total_retrieval_misses=self._total_recall_misses,
        )

    def get_event_log(self) -> list[EvokeEvent]:
        return list(self._events)

    def get_relevance_scores(self) -> dict[int, float]:
        blocks = self._positions.active_blocks
        current_pos = self._positions.next_logical_pos
        return self._scorer.score_blocks(
            blocks, current_pos, self._document_token_count
        )

    def force_demote(self, block_ids: list[int]) -> None:
        self._demote_blocks(set(block_ids))

    def force_promote(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            block = self._archive.get(bid)
            if block:
                self._promote_block(block)

    def _scoring_round(self) -> None:
        self._enforce_budget()

    def _enforce_budget(self) -> None:
        cfg = self._config
        active_tokens = self._positions.active_token_count

        if cfg.demotion_policy == "watermark":
            threshold = int(cfg.max_active_tokens * cfg.high_watermark)
            target = int(cfg.max_active_tokens * cfg.low_watermark)
        else:
            threshold = cfg.max_active_tokens
            target = cfg.max_active_tokens

        if active_tokens <= threshold:
            return

        tokens_to_free = active_tokens - target
        blocks = self._positions.active_blocks
        current_pos = self._positions.next_logical_pos
        scores = self._scorer.score_blocks(
            blocks, current_pos, self._document_token_count
        )

        candidates = self._demotable_blocks(blocks, scores)
        candidates.sort(key=lambda b: scores.get(b.block_id, 0))

        to_demote: list[ActiveBlock] = []
        freed = 0
        for block in candidates:
            if freed >= tokens_to_free:
                break
            to_demote.append(block)
            freed += block.logical_end - block.logical_start

        if to_demote:
            self._demote_blocks({b.block_id for b in to_demote})

    def _demotable_blocks(
        self, blocks: list[ActiveBlock], scores: dict[int, float]
    ) -> list[ActiveBlock]:
        demotable = []
        for block in blocks:
            if scores.get(block.block_id, 0) >= 1.0:
                continue

            if (
                self._config.pin_generated
                and block.original_start >= self._current_turn_start
            ):
                continue

            if (
                block.promotion_step >= 0
                and (self._step - block.promotion_step)
                < self._config.promotion_grace_steps
            ):
                continue

            demotable.append(block)
        return demotable

    def _demote_blocks(self, block_ids: set[int]) -> None:
        blocks_to_archive = [
            b for b in self._positions.active_blocks if b.block_id in block_ids
        ]

        for block in blocks_to_archive:
            archive_block = ArchiveBlock(
                block_id=block.block_id,
                token_ids=block.token_ids,
                original_positions=list(
                    range(block.original_start, block.original_end)
                ),
                text=self._engine.detokenize(block.token_ids),
                representative_embedding=block.representative_embedding
                if block.representative_embedding is not None
                else np.zeros(self._engine.n_embd),
                timestamp=self._step,
                source=block.source,
            )
            self._archive.store(archive_block)

        self._positions.remove_blocks(block_ids)

        all_blocks = self._positions.active_blocks
        token_blocks = [b.token_ids for b in all_blocks]
        self._engine.rebuild_kv(token_blocks)
        self._positions.rebuild_positions()

        self._total_demotions += len(blocks_to_archive)

        self._events.append(
            EvokeEvent(
                step=self._step,
                event_type="demotion",
                block_ids=list(block_ids),
            )
        )

    def _check_rebuild(self) -> None:
        n_ctx = self._engine.n_ctx
        engine_pressure = self._engine.next_write_pos > int(n_ctx * 0.9)
        block_pressure = self._positions.needs_rebuild(n_ctx)
        if not engine_pressure and not block_pressure:
            return

        blocks = self._positions.active_blocks
        token_blocks = [b.token_ids for b in blocks]
        self._engine.rebuild_kv(token_blocks)
        self._positions.rebuild_positions()

        self._events.append(
            EvokeEvent(
                step=self._step,
                event_type="rebuild",
                block_ids=[b.block_id for b in blocks],
            )
        )

    def _retrieve_from_archive(self, query_text: str) -> list[ArchiveBlock]:
        if self._archive.size == 0:
            return []

        query_embedding = self._scorer._recent_embedding
        if query_embedding is None:
            query_embedding = np.zeros(self._engine.n_embd)

        return self._archive.retrieve_by_similarity(
            query_embedding,
            self._config.retrieval_threshold,
            self._config.max_retrieve_blocks,
            query_text=query_text,
            min_lexical_recall=self._config.min_lexical_recall,
        )

    def _promote_via_rebuild(self, blocks: list[ArchiveBlock]) -> None:
        budget = self._config.max_active_tokens
        active = self._positions.active_token_count
        if active > int(budget * self._config.high_watermark):
            self._total_recall_misses += len(blocks)
            return
        cap = int(budget * self._config.max_promote_fraction)

        promoted_ids = []
        promoted_tokens = 0
        for block in blocks:
            block_size = len(block.token_ids)
            if promoted_tokens + block_size > cap:
                continue
            active_block = ActiveBlock(
                block_id=block.block_id,
                logical_start=block.pos_start,
                logical_end=block.pos_end,
                original_start=block.pos_start,
                original_end=block.pos_end,
                token_ids=block.token_ids,
                representative_embedding=block.representative_embedding,
                source=block.source,
                promotion_step=self._step,
            )
            self._positions.append_block(active_block, block.pos_start)
            self._archive.remove(block.block_id)
            promoted_ids.append(block.block_id)
            promoted_tokens += block_size

        if not promoted_ids:
            self._total_recall_misses += len(blocks)
            return

        all_blocks = self._positions.active_blocks
        token_blocks = [b.token_ids for b in all_blocks]
        self._engine.rebuild_kv(token_blocks)
        self._positions.rebuild_positions()
        self._current_turn_start = self._engine.next_write_pos

        self._total_promotions += len(promoted_ids)
        self._events.append(
            EvokeEvent(
                step=self._step,
                event_type="rebuild",
                block_ids=[b.block_id for b in all_blocks],
            )
        )
        self._events.append(
            EvokeEvent(
                step=self._step,
                event_type="promotion",
                block_ids=promoted_ids,
            )
        )

        self._enforce_budget()

    def _promote_block(self, archive_block: ArchiveBlock) -> None:
        self._promote_via_rebuild([archive_block])

    def _track_generated_block(self, tokens: list[int], start_pos: int) -> None:
        block_id = self._archive.allocate_id()
        block = ActiveBlock(
            block_id=block_id,
            logical_start=start_pos,
            logical_end=start_pos + len(tokens),
            original_start=start_pos,
            original_end=start_pos + len(tokens),
            token_ids=tokens,
            source=BlockSource.ASSISTANT,
        )
        self._positions.append_block(block, start_pos)

    def _track_conversation_block(self, tokens: list[int], start_pos: int) -> None:
        block_id = self._archive.allocate_id()
        block = ActiveBlock(
            block_id=block_id,
            logical_start=start_pos,
            logical_end=start_pos + len(tokens),
            original_start=start_pos,
            original_end=start_pos + len(tokens),
            token_ids=tokens,
            source=BlockSource.USER,
        )
        self._positions.append_block(block, start_pos)

        try:
            positions = list(range(start_pos, start_pos + len(tokens)))
            if positions:
                embeddings = self._engine.get_embeddings(positions)
                emb = embeddings[-1]
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                block.representative_embedding = emb
        except (NotImplementedError, RuntimeError):
            pass

    def _compute_block_embeddings(self, blocks: list[ActiveBlock]) -> None:
        try:
            total_tokens = sum(len(b.token_ids) for b in blocks)
            all_positions = list(range(total_tokens))
            if not all_positions:
                return
            embeddings = self._engine.get_embeddings(all_positions)

            for block in blocks:
                end = min(block.logical_end, len(embeddings))
                if end > 0:
                    emb = embeddings[end - 1]
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    block.representative_embedding = emb
        except (NotImplementedError, RuntimeError):
            pass

    def _update_recent_context_embedding(self, tokens: list[int], end_pos: int) -> None:
        try:
            positions = list(range(end_pos - len(tokens), end_pos))
            if positions:
                embeddings = self._engine.get_embeddings(positions)
                emb = embeddings[-1]
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                self._scorer.update_recent_context(emb)
        except (NotImplementedError, RuntimeError):
            pass
