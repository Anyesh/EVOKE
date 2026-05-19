from __future__ import annotations

import numpy as np


class MockEngine:
    def __init__(self, n_ctx: int = 32768, n_embd: int = 128, vocab_size: int = 1000):
        self._n_ctx = n_ctx
        self._n_embd = n_embd
        self._vocab_size = vocab_size
        self._kv_positions: set[int] = set()
        self._token_at_pos: dict[int, int] = {}
        self._next_gen_token = 0
        self._next_write_pos = 0
        self._seq_rm_calls: list[tuple[int, int]] = []
        self._rng = np.random.RandomState(42)
        self._embeddings: dict[int, np.ndarray] = {}

    def tokenize(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def detokenize(self, tokens: list[int]) -> str:
        return "".join(chr(t) if 0 <= t < 0x110000 else "?" for t in tokens)

    def process_tokens(self, tokens: list[int]) -> None:
        start = self._next_write_pos
        for i, token in enumerate(tokens):
            pos = start + i
            self._kv_positions.add(pos)
            self._token_at_pos[pos] = token
            self._embeddings[pos] = self._rng.randn(self._n_embd).astype(np.float32)
        self._next_write_pos += len(tokens)

    def generate_next(self) -> int:
        token = self._next_gen_token % self._vocab_size
        self._next_gen_token += 1
        pos = self._next_write_pos
        self._kv_positions.add(pos)
        self._token_at_pos[pos] = token
        self._embeddings[pos] = self._rng.randn(self._n_embd).astype(np.float32)
        self._next_write_pos += 1
        return token

    def get_kv_cache_token_count(self) -> int:
        return len(self._kv_positions)

    def kv_cache_seq_rm(self, pos_start: int, pos_end: int) -> None:
        self._seq_rm_calls.append((pos_start, pos_end))
        for pos in range(pos_start, pos_end):
            self._kv_positions.discard(pos)
            self._token_at_pos.pop(pos, None)
            self._embeddings.pop(pos, None)

    def rebuild_kv(self, token_blocks: list[list[int]]) -> None:
        self._kv_positions.clear()
        self._token_at_pos.clear()
        self._embeddings.clear()
        self._next_write_pos = 0
        for tokens in token_blocks:
            self.process_tokens(tokens)

    def reset(self) -> None:
        self._kv_positions.clear()
        self._token_at_pos.clear()
        self._embeddings.clear()
        self._next_write_pos = 0
        self._next_gen_token = 0

    def get_embeddings(self, token_positions: list[int]) -> np.ndarray:
        result = np.zeros((len(token_positions), self._n_embd), dtype=np.float32)
        for i, pos in enumerate(token_positions):
            if pos in self._embeddings:
                result[i] = self._embeddings[pos]
            else:
                result[i] = self._rng.randn(self._n_embd).astype(np.float32)
        return result

    @property
    def next_write_pos(self) -> int:
        return self._next_write_pos

    @property
    def n_ctx(self) -> int:
        return self._n_ctx

    @property
    def n_embd(self) -> int:
        return self._n_embd

    @property
    def eos_token(self) -> int:
        return 2
