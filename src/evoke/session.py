"""Persistent EVOKE session that survives across OpenAI chat-completion calls.

The OpenAI chat-completions API is stateless: every request resends the full
message history. A naive backend re-prefills the entire history each turn,
which is exactly the cost the agent stack pays today. This session keeps the
KV cache alive between requests, finds the longest token-prefix the new
request shares with what is already decoded, and processes only the new tail.

This is the gating piece for plugging a long-running EVOKE session into
opencode, Aider, or any OpenAI-compatible agent harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from evoke.llama_engine import LlamaCppEngine


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


class Session:
    def __init__(self, engine: LlamaCppEngine, *, detokenize_every: int = 4) -> None:
        self._engine = engine
        self._cached_tokens: list[int] = []
        self._detok_every = detokenize_every

    @property
    def cached_token_count(self) -> int:
        return len(self._cached_tokens)

    def reset(self) -> None:
        self._engine.reset()
        self._cached_tokens.clear()

    def _common_prefix_len(self, new_tokens: list[int]) -> int:
        limit = min(len(self._cached_tokens), len(new_tokens))
        for i in range(limit):
            if self._cached_tokens[i] != new_tokens[i]:
                return i
        return limit

    def sync_prefix(self, prompt_tokens: list[int]) -> int:
        divergence = self._common_prefix_len(prompt_tokens)
        if divergence < len(self._cached_tokens):
            # client diverged from our cached state. simplest correct response
            # is to drop everything and re-decode from scratch. that is rare in
            # append-only agent loops; an incremental truncate is a v2 concern.
            self.reset()
            divergence = 0
        tail = prompt_tokens[divergence:]
        if tail:
            self._engine.process_tokens(tail)
            self._cached_tokens.extend(tail)
        return len(tail)

    def generate(
        self,
        max_tokens: int,
        stop_strings: list[str] | None = None,
    ) -> GenerationResult:
        stops = stop_strings or []
        eos = self._engine.eos_token
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
        # can prefix-match against them.
        stops = stop_strings or []
        eos = self._engine.eos_token
        output_tokens: list[int] = []
        emitted_len = 0

        for step in range(max_tokens):
            token = self._engine.generate_next()
            output_tokens.append(token)

            if token == eos:
                full_text = self._engine.detokenize(output_tokens)
                delta = full_text[emitted_len:]
                self._cached_tokens.extend(output_tokens)
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
        yield GenerationChunk(
            delta_text=full_text[emitted_len:],
            finish_reason="length",
            full_text=full_text,
            output_tokens=output_tokens,
        )
