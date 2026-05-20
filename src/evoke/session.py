"""Persistent EVOKE session that survives across OpenAI chat-completion calls.

The OpenAI chat-completions API is stateless: every request resends the full
message history. A naive backend re-prefills the entire history each turn,
which is exactly the cost the agent stack pays today. This session keeps the
KV cache alive between requests, finds the longest token-prefix the new
request shares with what is already decoded, and processes only the new tail.

The session also wires EvokeManager into the request loop. When the cache
grows past the budget, EvokeManager evicts low-relevance blocks (saving their
K/V off-GPU under kv_restore mode); at the start of each new request the
session recovers evicted blocks so the model sees the full history again. This
lets agent sessions grow logically past the physical KV budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager


@dataclass
class GenerationResult:
    text: str
    output_tokens: list[int]
    finish_reason: str


@dataclass
class GenerationChunk:
    delta_text: str
    finish_reason: str | None
    full_text: str
    output_tokens: list[int]


@dataclass
class SyncStats:
    new_tokens_decoded: int
    blocks_recovered: int
    active_tokens_after: int
    active_blocks_after: int


class Session:
    def __init__(
        self,
        engine: LlamaCppEngine,
        *,
        config: EvokeConfig | None = None,
        detokenize_every: int = 4,
        recovery_k: int = 4,
    ) -> None:
        self._engine = engine
        self._config = config or self._default_config(engine.n_ctx)
        self._manager = EvokeManager(engine, self._config)
        self._cached_tokens: list[int] = []
        self._turn_id = 0
        self._detok_every = detokenize_every
        self._recovery_k = recovery_k

    @staticmethod
    def _default_config(n_ctx: int) -> EvokeConfig:
        # Budget is a fraction of n_ctx so eviction fires before the engine
        # hits its hard limit. Block size of 128 keeps recovery granularity
        # small enough that we are not pulling huge chunks back unnecessarily.
        budget = max(2048, int(n_ctx * 0.75))
        return EvokeConfig(
            max_active_tokens=budget,
            block_size=128,
            high_watermark=0.92,
            low_watermark=0.70,
            recovery_mode="kv_restore"
            if True  # set False to demo discard
            else "discard",
        )

    @property
    def cached_token_count(self) -> int:
        return len(self._cached_tokens)

    @property
    def manager(self) -> EvokeManager:
        return self._manager

    def reset(self) -> None:
        self._engine.reset()
        self._manager = EvokeManager(self._engine, self._config)
        self._cached_tokens.clear()
        self._turn_id = 0

    def _common_prefix_len(self, new_tokens: list[int]) -> int:
        limit = min(len(self._cached_tokens), len(new_tokens))
        for i in range(limit):
            if self._cached_tokens[i] != new_tokens[i]:
                return i
        return limit

    def _smart_recover(self, k: int = 4) -> int:
        # Score evicted blocks by embedding similarity to the last ~n_query
        # tokens just decoded (the freshest signal we have about what the
        # next answer needs) and recover only the top-K. This is the v3
        # policy: bounded per-turn recovery cost, eviction actually frees
        # memory, and the model only pays to re-attend to what is relevant
        # to the current query.
        crumbs = list(self._manager.get_breadcrumbs())
        if not crumbs:
            return 0

        query_emb = self._compute_query_embedding()
        if query_emb is None:
            # Fall back to v2 behavior if we cannot score (no embeddings yet
            # because the tail was empty, or the engine returned zeros).
            recovered = 0
            for crumb in crumbs:
                if self._manager.recover(crumb.key):
                    recovered += 1
            return recovered

        scored: list[tuple[float, str]] = []
        for crumb in crumbs:
            block_emb = self._manager._recovery.peek_embedding(crumb.key)
            if block_emb is None:
                scored.append((0.0, crumb.key))
            else:
                scored.append((float(np.dot(query_emb, block_emb)), crumb.key))
        scored.sort(key=lambda x: x[0], reverse=True)

        recovered = 0
        for _, key in scored[:k]:
            if self._manager.recover(key):
                recovered += 1
        return recovered

    def _compute_query_embedding(self, n_last: int = 32) -> np.ndarray | None:
        pos = self._engine.next_write_pos
        if pos == 0:
            return None
        start = max(0, pos - n_last)
        positions = list(range(start, pos))
        embeddings = self._engine.get_embeddings(positions)
        if embeddings is None or len(embeddings) == 0:
            return None
        nonzero_mask = (embeddings != 0).any(axis=1)
        if not nonzero_mask.any():
            return None
        avg = embeddings[nonzero_mask].mean(axis=0)
        norm = float(np.linalg.norm(avg))
        if norm == 0.0:
            return None
        return avg / norm

    def sync_prefix(self, prompt_tokens: list[int]) -> SyncStats:
        divergence = self._common_prefix_len(prompt_tokens)
        if divergence < len(self._cached_tokens):
            self.reset()
            divergence = 0

        tail = prompt_tokens[divergence:]
        if tail:
            self._manager.add_context_tokens(tail, key=f"turn{self._turn_id}")
            self._turn_id += 1
            self._cached_tokens.extend(tail)

        # Smart recovery runs AFTER the new tail is decoded so we have fresh
        # per-token embeddings for the latest user message; we score evicted
        # blocks against that and bring back the top-K most coherent ones.
        recovered = self._smart_recover(k=self._recovery_k)

        stats = self._manager.get_stats()
        return SyncStats(
            new_tokens_decoded=len(tail),
            blocks_recovered=recovered,
            active_tokens_after=stats.active_tokens,
            active_blocks_after=stats.active_blocks,
        )

    def _strip_thinking(self, output_tokens: list[int], gen_start: int) -> list[int]:
        # If the model emitted <think>...</think>, the next chat request will
        # not include the thinking trace (we strip it from parse_qwen_response
        # before returning to the client). To keep the cached state aligned
        # with what the client will send back, evict the thinking range from
        # the physical cache and from cached_tokens. Returns the tokens that
        # remain (the answer; or [] if generation never closed </think>).
        if not output_tokens:
            return output_tokens
        close_tokens = self._engine.tokenize("</think>")
        if not close_tokens:
            return output_tokens
        close_id = close_tokens[-1]
        try:
            close_idx = output_tokens.index(close_id)
        except ValueError:
            open_tokens = self._engine.tokenize("<think>")
            if open_tokens and open_tokens[-1] in output_tokens:
                # opened but never closed: try to evict everything. On hybrid
                # memory the eviction may be rejected (recurrent can't slice);
                # in that case the physical cache keeps the trace and we leave
                # cached_tokens intact so they stay aligned with the engine.
                # The next request's prefix-match will diverge against the
                # post-stripped assistant message and reset cleanly.
                if self._engine.evict_ranges(
                    [(gen_start, gen_start + len(output_tokens))]
                ):
                    self._cached_tokens = self._cached_tokens[:gen_start]
                    return []
                return output_tokens
            return output_tokens

        answer_tokens = output_tokens[close_idx + 1 :]
        think_end_abs = gen_start + close_idx + 1
        if think_end_abs > gen_start:
            if self._engine.evict_ranges([(gen_start, think_end_abs)]):
                self._cached_tokens = self._cached_tokens[:gen_start] + list(
                    answer_tokens
                )
            else:
                return output_tokens
        return answer_tokens

    def _track_and_enforce(self, output_tokens: list[int], gen_start: int) -> None:
        kept = self._strip_thinking(output_tokens, gen_start)
        if kept:
            # gen_start may have shifted if thinking was evicted; recompute.
            answer_start = self._engine.next_write_pos - len(kept)
            self._manager._track_generated_block(kept, answer_start)
        self._manager._enforce_budget()

    def generate(
        self,
        max_tokens: int,
        stop_strings: list[str] | None = None,
    ) -> GenerationResult:
        stops = stop_strings or []
        eos = self._engine.eos_token
        gen_start = self._engine.next_write_pos
        output_tokens: list[int] = []
        finish = "length"
        truncated_text: str | None = None

        for step in range(max_tokens):
            token = self._engine.generate_next()
            output_tokens.append(token)
            if token == eos:
                finish = "stop"
                break

            if stops and (step % self._detok_every == 0 or step == max_tokens - 1):
                text_so_far = self._engine.detokenize(output_tokens)
                hit_idx = -1
                for stop in stops:
                    idx = text_so_far.find(stop)
                    if idx != -1 and (hit_idx == -1 or idx < hit_idx):
                        hit_idx = idx
                if hit_idx != -1:
                    truncated_text = text_so_far[:hit_idx]
                    finish = "stop"
                    break

        if truncated_text is None:
            text = self._engine.detokenize(output_tokens)
        else:
            text = truncated_text

        self._cached_tokens.extend(output_tokens)
        self._track_and_enforce(output_tokens, gen_start)
        return GenerationResult(
            text=text, output_tokens=output_tokens, finish_reason=finish
        )

    def stream_generate(
        self,
        max_tokens: int,
        stop_strings: list[str] | None = None,
    ) -> Iterator[GenerationChunk]:
        # Token-by-token streaming. Yields a chunk per generated token with the
        # incremental delta text. Stop-string truncation is honored: the chunk
        # that crosses the stop boundary yields only the pre-stop slice. After
        # the loop the full output_tokens stay in the cache so the next request
        # can prefix-match against them, and the manager is asked to enforce
        # the budget so eviction can fire on the new generated content too.
        stops = stop_strings or []
        eos = self._engine.eos_token
        gen_start = self._engine.next_write_pos
        output_tokens: list[int] = []
        emitted_len = 0

        for step in range(max_tokens):
            token = self._engine.generate_next()
            output_tokens.append(token)

            if token == eos:
                full_text = self._engine.detokenize(output_tokens)
                delta = full_text[emitted_len:]
                self._cached_tokens.extend(output_tokens)
                self._track_and_enforce(output_tokens, gen_start)
                yield GenerationChunk(
                    delta_text=delta,
                    finish_reason="stop",
                    full_text=full_text,
                    output_tokens=output_tokens,
                )
                return

            full_text = self._engine.detokenize(output_tokens)

            hit_idx = -1
            for stop in stops:
                idx = full_text.find(stop, emitted_len)
                if idx != -1 and (hit_idx == -1 or idx < hit_idx):
                    hit_idx = idx
            if hit_idx != -1:
                truncated = full_text[:hit_idx]
                delta = truncated[emitted_len:]
                self._cached_tokens.extend(output_tokens)
                self._track_and_enforce(output_tokens, gen_start)
                yield GenerationChunk(
                    delta_text=delta,
                    finish_reason="stop",
                    full_text=truncated,
                    output_tokens=output_tokens,
                )
                return

            delta = full_text[emitted_len:]
            emitted_len = len(full_text)
            yield GenerationChunk(
                delta_text=delta,
                finish_reason=None,
                full_text=full_text,
                output_tokens=output_tokens,
            )

        full_text = self._engine.detokenize(output_tokens)
        self._cached_tokens.extend(output_tokens)
        self._track_and_enforce(output_tokens, gen_start)
        yield GenerationChunk(
            delta_text=full_text[emitted_len:],
            finish_reason="length",
            full_text=full_text,
            output_tokens=output_tokens,
        )
