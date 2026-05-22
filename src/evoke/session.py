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

import os
import sys
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from evoke.attention_scorer import AttentionScorer
from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager
from evoke.templates import parse_qwen_response


@dataclass
class GenerationResult:
    text: str
    output_tokens: list[int]
    finish_reason: str


class SessionPool:
    # Multi-session server pool. One llama_context is shared across all
    # sessions; per-session KV state is swapped in and out via the
    # engine's state_save / state_restore primitives. Only one session is
    # "live" in the engine at a time, but each session preserves its
    # Python-side state (cached tokens, manager blocks, scorer windows)
    # and its serialized engine snapshot, so context switches are fast
    # (one memcpy proportional to current cache size, no recompute).
    #
    # Sessions are keyed by an opaque client-supplied id (typically the
    # X-EVOKE-Session header on a request). LRU eviction kicks in past
    # max_sessions: the least recently used session is evicted entirely
    # (Python state + serialized engine snapshot dropped). For long-lived
    # multi-client servers, set max_sessions higher and accept the host
    # RAM cost; for stateless gateway scenarios, leave it low.

    def __init__(
        self,
        engine: LlamaCppEngine,
        *,
        config: EvokeConfig | None = None,
        max_sessions: int = 8,
    ) -> None:
        self._engine = engine
        self._config = config
        self._max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}
        self._snapshots: dict[
            str, tuple[bytes, int, int, dict[int, np.ndarray]] | None
        ] = {}
        self._lru: list[str] = []  # most-recent-last
        self._active: str | None = None
        self._evicted_count = 0

    @property
    def active_session_id(self) -> str | None:
        return self._active

    @property
    def n_sessions(self) -> int:
        return len(self._sessions)

    @property
    def evicted_count(self) -> int:
        return self._evicted_count

    def session_ids(self) -> list[str]:
        return list(self._sessions.keys())

    def get(self, session_id: str) -> Session:
        # Returns the Session for session_id, swapping it into the engine
        # if it isn't already active. Creates a new Session if first-seen.
        if session_id in self._sessions:
            # Touch LRU
            try:
                self._lru.remove(session_id)
            except ValueError:
                pass
            self._lru.append(session_id)
            if self._active != session_id:
                self._swap_in(session_id)
            return self._sessions[session_id]
        # First-seen session: snapshot the currently-active one (so it can
        # be restored later), reset the engine, create a fresh Session.
        if self._active is not None:
            self._snapshots[self._active] = self._engine.state_save()
        self._engine.reset()
        new = Session(self._engine, config=self._config)
        self._sessions[session_id] = new
        self._snapshots[session_id] = None
        self._active = session_id
        self._lru.append(session_id)
        self._maybe_evict_lru()
        return new

    def drop(self, session_id: str) -> bool:
        # Forcefully evict a session by id. Returns True if it existed.
        if session_id not in self._sessions:
            return False
        if self._active == session_id:
            self._engine.reset()
            self._active = None
        self._sessions.pop(session_id, None)
        self._snapshots.pop(session_id, None)
        try:
            self._lru.remove(session_id)
        except ValueError:
            pass
        self._evicted_count += 1
        return True

    def _swap_in(self, session_id: str) -> None:
        # Save current active to snapshot, restore target from snapshot.
        if self._active is not None:
            self._snapshots[self._active] = self._engine.state_save()
        snap = self._snapshots.get(session_id)
        if snap is None:
            self._engine.reset()
        else:
            self._engine.state_restore(snap)
        self._active = session_id

    def _maybe_evict_lru(self) -> None:
        while len(self._sessions) > self._max_sessions and self._lru:
            victim = self._lru.pop(0)
            if victim == self._active:
                # Don't evict the live session; rotate it to the back and
                # pick the next-oldest. Should be rare (max_sessions=1).
                self._lru.append(victim)
                if len(self._lru) == 1:
                    return
                victim = self._lru.pop(0)
            self._sessions.pop(victim, None)
            self._snapshots.pop(victim, None)
            self._evicted_count += 1


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
        self._attention_scorer = self._maybe_build_attention_scorer()
        self._manager = EvokeManager(
            engine, self._config, attention_scorer=self._attention_scorer
        )
        self._cached_tokens: list[int] = []
        self._turn_id = 0
        self._detok_every = detokenize_every
        self._recovery_k = recovery_k

    def _maybe_build_attention_scorer(self) -> AttentionScorer | None:
        # Construct an AttentionScorer iff the multi-signal scorer wants
        # attention (cfg.w_attention > 0) AND the engine exposes the capture
        # primitives (EVOKE-built llama.cpp). When either condition is false,
        # the scorer falls back to recency + coherence + harness priority —
        # same behavior as before this rework.
        if self._config.w_attention <= 0:
            return None
        if not getattr(self._engine, "supports_kv_block", False):
            return None
        try:
            return AttentionScorer(
                self._engine,
                layer=self._config.attention_capture_layer,
                n_window=self._config.attention_window,
                decay=self._config.attention_decay,
                score_mode=self._config.attention_score_mode,
            )
        except (OSError, RuntimeError, ValueError):
            # If the binding fails (stale fork, ctypes mismatch, buffer
            # alloc error), fall back. The session can still serve with
            # recency+coherence scoring.
            return None

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

    def _log_drift(self, prompt_tokens: list[int], divergence: int) -> None:
        # Diagnostic for round-trip mismatches between cached generation tokens
        # and the Jinja-rendered re-tokenization of the same content. Writes to
        # EVOKE_DEBUG_DRIFT_FILE if set (so a remote driver can scp the file
        # back), else stderr. Writing from Python avoids PowerShell pipeline
        # redirection issues across the SSH boundary.
        cached = self._cached_tokens
        ctx = 30

        def tok_repr(t: int) -> str:
            try:
                text = self._engine.detokenize([t])
            except Exception:
                text = "<detok-err>"
            return f"{t}:{text!r}"

        pre_lo = max(0, divergence - ctx)
        pre_cached = cached[pre_lo:divergence]
        post_cached = cached[divergence : divergence + ctx]
        post_prompt = prompt_tokens[divergence : divergence + ctx]
        try:
            pre_text = self._engine.detokenize(list(pre_cached))
            cached_after = self._engine.detokenize(list(post_cached))
            prompt_after = self._engine.detokenize(list(post_prompt))
        except Exception:
            pre_text = cached_after = prompt_after = "<detok-err>"

        block = (
            "\n=== EVOKE drift diagnostic ===\n"
            f"  cached_len={len(cached)}  prompt_len={len(prompt_tokens)}  "
            f"divergence_idx={divergence}\n"
            f"  context_before  (len={len(pre_cached)}): {pre_text!r}\n"
            f"  cached_at_div   : {tok_repr(cached[divergence]) if divergence < len(cached) else '<end-of-cache>'}\n"
            f"  prompt_at_div   : {tok_repr(prompt_tokens[divergence]) if divergence < len(prompt_tokens) else '<end-of-prompt>'}\n"
            f"  cached_after    : {cached_after!r}\n"
            f"  prompt_after    : {prompt_after!r}\n"
            f"  cached_after_ids: {list(post_cached)}\n"
            f"  prompt_after_ids: {list(post_prompt)}\n"
            "=== /drift diagnostic ===\n"
        )
        path = os.environ.get("EVOKE_DEBUG_DRIFT_FILE")
        if path:
            try:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(block)
                return
            except OSError:
                pass
        sys.stderr.write(block)
        sys.stderr.flush()

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

        # Resident-gate: only recover an evicted block if its similarity to
        # the query beats the best resident block. The active cache already
        # holds the strongest in-cache match; bringing back a weaker breadcrumb
        # appends content AFTER the probe (the model's freshest context) and
        # drowns out the resident match. NIAH at depth=90 exposed this: the
        # needle was resident, recovery brought 4 weakly-matching haystack
        # blocks to the cache tail, and the model regurgitated those instead
        # of answering from the needle.
        resident_max = self._max_resident_similarity(query_emb)
        scored = [(s, key) for s, key in scored if s > resident_max]

        threshold = self._config.smart_recover_min_similarity
        if threshold > 0.0:
            scored = [(s, key) for s, key in scored if s >= threshold]

        # Recovery appends blocks at the cache tail, so the LAST block loaded
        # becomes the model's freshest pre-generation context. Reverse the
        # top-K so the most-similar block lands last (freshest); without this
        # the lowest-similarity recovery dominates attention during decode
        # and the model regurgitates the weakest match. NIAH at depth=10
        # exposed this: with k=4, the rank-1 needle block was recovered
        # first (positionally earliest among the new tail) and ranks 2-4
        # of unrelated blocks landed after it, capturing attention.
        ordered = list(scored[:k])
        ordered.reverse()
        recovered = 0
        for _, key in ordered:
            if self._manager.recover(key):
                recovered += 1
        return recovered

    def _max_resident_similarity(self, query_emb: np.ndarray) -> float:
        # Max cosine between the query and any resident (non-sink, non-current-
        # turn) block's representative embedding. Current-turn blocks include
        # the probe itself (whose embedding is essentially the query, scoring
        # ~1.0 against itself) and any tokens just generated this turn; they
        # would always saturate the gate and prevent any recovery from firing.
        # Sinks score 1.0 unconditionally and would similarly saturate.
        # Missing embeddings (None) skip rather than score zero so they cannot
        # accidentally widen the gate.
        best = -1.0
        current_turn_start = self._manager._current_turn_start_id
        for block in self._manager._positions.active_blocks:
            if block.is_sink:
                continue
            if block.block_id >= current_turn_start:
                continue
            emb = block.representative_embedding
            if emb is None:
                continue
            sim = float(np.dot(query_emb, emb))
            if sim > best:
                best = sim
        return best

    def _compute_query_embedding(self, n_last: int = 32) -> np.ndarray | None:
        # Prefer a retrieval-tuned embedding of the raw user-message text
        # when one is available. The LM-hidden-state path that follows
        # carries the common-mode noise (cosine floor ~0.85 against any
        # paragraph in the corpus) that prevented smart-recovery from
        # distinguishing needle blocks from haystack noise on NIAH.
        if (
            self._manager._retrieval_embedder is not None
            and self._manager._last_user_text
        ):
            return self._manager._retrieval_embedder.embed(
                self._manager._last_user_text
            )
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

    def sync_prefix(
        self,
        prompt_tokens: list[int],
        *,
        priority: float = 1.0,
        pinned: bool = False,
        task_boundary: bool = False,
    ) -> SyncStats:
        if task_boundary:
            self._manager.signal_task_boundary()
        # Rebuild from the manager so prior-turn evictions and recoveries are
        # reflected. _cached_tokens is extended during generate() and trimmed
        # by canonicalize, but engine-internal evictions fired by
        # _enforce_budget aren't visible to Session; without this resync, the
        # prefix-match runs against a stale token list and "succeeds" at
        # positions where the engine actually has different content.
        self._cached_tokens = self._manager.get_token_view()
        divergence = self._common_prefix_len(prompt_tokens)
        if divergence < len(self._cached_tokens):
            if os.environ.get("EVOKE_DEBUG_DRIFT"):
                self._log_drift(prompt_tokens, divergence)
            # Tail-evict the diverged portion of the cache instead of resetting
            # the whole session. This keeps the matching prefix (often the
            # system prompt + early turns), preserves eviction stats across
            # turns, and lets truncate-policy sessions continue smoothly even
            # when manager eviction has dropped middle history that the next
            # request resupplies. Falls back to full reset only when the
            # engine refuses tail-eviction (hybrid Mamba+Attention memory).
            cached_len = len(self._cached_tokens)
            if self._engine.evict_ranges([(divergence, cached_len)]):
                self._manager.trim_blocks_at(divergence)
                self._cached_tokens = self._cached_tokens[:divergence]
            else:
                self.reset()
                divergence = 0

        tail = prompt_tokens[divergence:]
        if tail:
            self._manager.add_context_tokens(
                tail,
                key=f"turn{self._turn_id}",
                priority=priority,
                pinned=pinned,
            )
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
        #
        # When config.suppress_thinking_strip is True (typical for hybrid
        # Mamba+Attention models that cannot do mid-cache eviction), we leave
        # the thinking trace in both the cache AND the returned content. The
        # client echoes it back verbatim on the next request, the cached
        # prefix stays aligned, and no session reset is needed. See
        # parse_qwen_response's strip_thinking parameter for the response side.
        if self._config.suppress_thinking_strip:
            return output_tokens
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
            kept = self._canonicalize_assistant(kept, answer_start)
            if kept:
                answer_start = self._engine.next_write_pos - len(kept)
                self._manager._track_generated_block(kept, answer_start)
        self._manager._enforce_budget()

    def _canonicalize_assistant(
        self, answer_tokens: list[int], answer_start: int
    ) -> list[int]:
        # Make the cached assistant emit match what the next request's
        # Jinja-then-tokenize will produce. The model can emit non-canonical
        # BPE (e.g. token('**') + token(':\n') for text the canonical tokenizer
        # would have encoded as token('**:') + token('\n')), and
        # parse_qwen_response strips trailing whitespace from the content
        # returned to the client. Both make the cached prefix diverge from the
        # next request's prompt mid-history and force a session reset that
        # zeroes the eviction counters we are trying to measure. We re-decode
        # the canonical tokenization of the visible content at the same logical
        # position so the cached state matches the round-trip exactly.
        if not answer_tokens:
            return answer_tokens
        raw = self._engine.detokenize(answer_tokens)
        parsed = parse_qwen_response(raw)
        if parsed.tool_calls:
            # Tool-using responses go through format_qwen_chat (the C API
            # llama_chat_apply_template doesn't accept tools); their
            # round-trip is governed by that template, not by retokenization.
            return answer_tokens
        eos = self._engine.eos_token
        emitted_eos = answer_tokens[-1] == eos
        canonical = self._engine.tokenize(parsed.content)
        if emitted_eos:
            canonical = canonical + [eos]
        if canonical == list(answer_tokens):
            return answer_tokens
        answer_end = answer_start + len(answer_tokens)
        if not self._engine.evict_ranges([(answer_start, answer_end)]):
            # Hybrid (Mamba+Attention) memory rejects mid-cache splice on the
            # recurrent half. Leave the model's emit in place; drift may still
            # occur for these models but is documented as a known gap.
            return answer_tokens
        self._cached_tokens = self._cached_tokens[:answer_start]
        if canonical:
            self._engine.process_tokens(canonical)
            self._cached_tokens.extend(canonical)
        return canonical

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
