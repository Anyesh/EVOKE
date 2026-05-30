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
        retrieval_embedder=None,
    ):
        self._engine = engine
        self._config = config or EvokeConfig()
        self._attention_scorer = attention_scorer
        self._retrieval_embedder = retrieval_embedder
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
        self._peak_active_tokens = 0
        self._next_block_id = 0
        self._current_turn_start_id = 0
        # Last user-message text, captured so the retrieval embedder can
        # produce the query embedding directly from text rather than averaging
        # LM hidden states across the probe's KV positions (which carries the
        # common-mode noise that defeats smart-recovery discrimination).
        self._last_user_text: str | None = None

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
        new_blocks: list[ActiveBlock] = []
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
            new_blocks.append(block)
        self._apply_retrieval_embeddings(new_blocks)
        self._absorb_attention()

        self._enforce_budget()

    def _apply_retrieval_embeddings(self, blocks: list[ActiveBlock]) -> None:
        # When a retrieval embedder is configured the block representative
        # embedding is overridden from the LM-derived value to a retrieval-
        # tuned embedding computed on the block's decoded text. Batched so
        # the per-add_context cost is one model call, not one per block.
        if self._retrieval_embedder is None or not blocks:
            return
        texts = [self._engine.detokenize(b.token_ids) for b in blocks]
        embeddings = self._retrieval_embedder.embed_batch(texts)
        for block, emb in zip(blocks, embeddings):
            block.representative_embedding = emb

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
        n_ctx = self._engine.n_ctx

        for _ in range(thinking_budget + answer_budget):
            # The thinking path lacked the slot guard the plain generate() has;
            # a long <think> trace would run next_write_pos into n_ctx and crash
            # generate_next with "no KV slot". Stop before that boundary.
            if self._engine.next_write_pos + 1 >= n_ctx:
                break
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

    def tick_turn(self) -> None:
        # Per-turn maintenance hook for recovery-aware eviction. Decays the
        # recovery_strength signal on every active block by recovery_decay so
        # the per-block protection from being a recent recovery target fades
        # over time, eventually returning the block to ordinary eviction
        # eligibility. Called once at the start of each user turn from both
        # the standalone-manager flow (process_user_message) and the server
        # flow (Session.sync_prefix). decay >= 1.0 disables decay entirely.
        decay = self._config.recovery_decay
        if decay >= 1.0:
            return
        for block in self._positions.active_blocks:
            if block.recovery_strength <= 0.0:
                continue
            block.recovery_strength *= decay
            if block.recovery_strength < 1e-3:
                block.recovery_strength = 0.0

    def process_user_message(self, text: str) -> None:
        # tick_turn must run BEFORE _current_turn_start_id moves so the
        # previous turn's recovered blocks lose their protection before this
        # turn's eviction pass evaluates them. A block recovered LAST turn at
        # strength 1.0 enters this turn at strength `recovery_decay`; a block
        # recovered THIS turn (via the upstream session.sync_prefix smart
        # recover) was set to recovery_strength_init AFTER the tick, so it
        # still carries full protection when eviction fires.
        self.tick_turn()
        self._current_turn_start_id = self._next_block_id
        self._last_user_text = text

        msg_start = self._engine.next_write_pos
        tokens = self._engine.tokenize(text)
        self._engine.process_tokens(tokens)
        end_pos = self._engine.next_write_pos

        self._track_conversation_block(tokens, msg_start)
        self._update_recent_context_embedding(tokens, end_pos)
        # Pull the user-message decode's attention into the scorer before the
        # eviction pass. Without this hook H2O's cumulative scorer never saw
        # attention from the question (only from add_context prefill and from
        # the gen-time tail), and SnapKV had no observation-window signal to
        # snapshot from — both relied on the user-message attention to pick
        # which prior blocks to keep. snapshot() then freezes SnapKV's
        # pending bucket so the immediately following _enforce_budget call
        # uses the question-window scores; for other score modes it is a
        # no-op.
        self._absorb_attention()
        if self._attention_scorer is not None and hasattr(
            self._attention_scorer, "snapshot"
        ):
            self._attention_scorer.snapshot()
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

    @property
    def peak_active_tokens(self) -> int:
        return self._peak_active_tokens

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
        # Position order, not block_id order: a block recovered in place (sparse
        # mode) gets a fresh high block_id but lives at an old logical_start, so
        # iterating block_id order would misorder the view and break the server's
        # prefix-match. In compact mode the two orders coincide.
        tokens: list[int] = []
        for block in sorted(
            self._positions.active_blocks, key=lambda b: (b.logical_start, b.block_id)
        ):
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

    def recover(self, key: str, *, defer_budget: bool = False) -> bool:
        saved = self._recovery.take(key)
        if saved is None:
            return False

        # Sparse mode restores the block at its original absolute position
        # (the hole left by seq_rm), so the C splice computes a zero RoPE shift.
        # Compact mode appends at the contiguous tail and re-anchors.
        if self._config.position_mode == "sparse":
            new_p0 = saved.original_start
        else:
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
            recovery_strength=self._config.recovery_strength_init,
        )
        self._positions.append_block(block, new_p0)
        self._total_recoveries += 1
        self._events.append(
            EvokeEvent(step=self._step, event_type="recovery", block_ids=[bid])
        )
        if not defer_budget:
            self._enforce_budget()
        return True

    def _enforce_budget(self) -> None:
        cfg = self._config
        # SnapKV defers eviction until the first process_user_message snapshot
        # has fired so the observation-window scores exist before any block is
        # dropped. Without this gate, add_context's per-chunk _enforce_budget
        # would tie every block at 0.0 (score() returns None pre-snapshot) and
        # evict in insertion order — meaning the needle is gone before SnapKV
        # ever sees the question. Other scorers (ewma, cumulative) always
        # report ready and proceed normally.
        if self._attention_scorer is not None and hasattr(
            self._attention_scorer, "is_eviction_ready"
        ):
            if not self._attention_scorer.is_eviction_ready():
                return
        active_tokens = self._positions.active_token_count
        # Record the high-water mark before any trim, so callers can see whether
        # the cache transiently held the full working set (the peak right after a
        # gap-fill rebuild) even when end-of-turn enforcement brings it back down.
        if active_tokens > self._peak_active_tokens:
            self._peak_active_tokens = active_tokens

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

        candidates = self._evictable_blocks(blocks, scores, pos)
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
        self, blocks: list[ActiveBlock], scores: dict[int, float], current_pos: int
    ) -> list[ActiveBlock]:
        # H2O-style recent-tail guard. Blocks whose logical_end falls within
        # the last R positions of the cache are excluded from eviction
        # candidates regardless of score, so heavy-hitter selection isn't
        # confounded by recency pruning of mid-cache blocks. Default is 0
        # (no guard) so existing EVOKE policies are unaffected.
        recent_protect_n = int(
            self._config.max_active_tokens * self._config.recent_tail_protect_frac
        )
        evictable = []
        for block in blocks:
            if block.is_sink or block.pinned:
                continue
            if scores.get(block.block_id, 0.0) >= 1.0:
                continue
            # Hard-protect a freshly recovered block: the agent just re-referenced
            # it, so it is the active working set and must not be re-evicted before
            # the model uses it this turn (its old low-recency position would
            # otherwise make the scorer drop it first). Decays via tick_turn.
            if (
                self._config.recovery_protect_threshold > 0.0
                and block.recovery_strength >= self._config.recovery_protect_threshold
            ):
                continue
            # Sparse mode leaves holes and does NOT lower next_write_pos, so
            # evicting the topmost (highest-position) block would leave the decode
            # head past the max cached position; llama_decode then rejects the
            # next, non-consecutive batch ("inconsistent sequence positions").
            # Protect the contiguous decode head so max_cached stays
            # next_write_pos-1; internal holes below it decode fine. Compact mode
            # recompacts positions, so it does not need this.
            if (
                self._config.position_mode == "sparse"
                and block.logical_end >= current_pos
            ):
                continue
            if (
                recent_protect_n > 0
                and block.logical_end >= current_pos - recent_protect_n
            ):
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
        compact = self._config.position_mode == "compact"
        self._engine.evict_ranges(ranges, compact=compact)
        self._positions.remove_blocks(block_ids)
        if compact:
            # Sparse mode must keep survivors at their true absolute positions,
            # so it skips the contiguous re-index that recompact() performs.
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
        # Conversation blocks must use the same embedding space as document
        # blocks; otherwise smart-recovery's cosine cross-dimensional explodes
        # (LM hidden state is e.g. 3584-dim, bge-small is 384-dim) when
        # the resident scan reaches the conversation block.
        self._apply_retrieval_embeddings([block])

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
        # When retrieval embeddings are configured, the task_focus must live
        # in the same embedding space as the block embeddings (RelevanceScorer
        # compares them via cosine; cross-space cosine raises a dimension
        # mismatch). Use the raw user-message text when available so the
        # focus reflects topic intent rather than an LM-mean over the
        # message's KV positions.
        if self._retrieval_embedder is not None and self._last_user_text:
            emb = self._retrieval_embedder.embed(self._last_user_text)
        else:
            emb = self._last_token_embedding(end_pos - len(tokens), len(tokens))
        if emb is not None:
            self._scorer.update_recent_context(emb)

    def _last_token_embedding(self, start_pos: int, length: int) -> np.ndarray | None:
        # Despite the legacy name this returns a block representative embedding
        # whose strategy is configurable. "mean" pools over non-zero token
        # embeddings, which is strictly better than last-token for retrieval
        # similarity (the last token of a 64-token block typically lands in
        # neighboring content rather than the block's defining terms). Kept
        # as the original method name because it is referenced in 3+ call
        # sites and renaming everywhere is noisy for a behavior-only change.
        if length <= 0:
            return None
        try:
            positions = list(range(start_pos, start_pos + length))
            embeddings = self._engine.get_embeddings(positions)
            if self._config.block_embedding_strategy == "mean":
                mask = (embeddings != 0).any(axis=1)
                if not mask.any():
                    return _normalize(embeddings[-1])
                avg = embeddings[mask].mean(axis=0)
                return _normalize(avg)
            return _normalize(embeddings[-1])
        except (NotImplementedError, RuntimeError):
            return None


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
