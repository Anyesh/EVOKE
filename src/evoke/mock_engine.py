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
        self._rng = np.random.RandomState(42)
        self._embeddings: dict[int, np.ndarray] = {}
        self._gen_queue: list[int] = []
        self.closed = False

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

    def queue_tokens(self, tokens: list[int]) -> None:
        self._gen_queue.extend(tokens)

    def generate_next(self) -> int:
        if self._gen_queue:
            token = self._gen_queue.pop(0)
        else:
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

    def evict_ranges(self, ranges: list[tuple[int, int]], compact: bool = True) -> bool:
        # Return True/False matching the LlamaCppEngine contract: True means
        # the eviction was applied. The original MockEngine returned None
        # implicitly, which Session.sync_prefix interpreted as falsy and so
        # fell through to session.reset() on every divergent turn. That
        # masked all session-path eviction tests against the mock.
        #
        # compact=True re-indexes survivors contiguously and shrinks the tail.
        # compact=False (sparse mode) drops the evicted positions but leaves
        # survivors at their absolute index and the tail unchanged, mirroring
        # seq_rm-without-seq_add on the real engine.
        if not ranges:
            return True
        removed: set[int] = set()
        for pos_start, pos_end in ranges:
            removed.update(range(pos_start, pos_end))
        if not compact:
            self._kv_positions = {p for p in self._kv_positions if p not in removed}
            self._token_at_pos = {
                p: t for p, t in self._token_at_pos.items() if p not in removed
            }
            self._embeddings = {
                p: e for p, e in self._embeddings.items() if p not in removed
            }
            return True
        survivors = [p for p in range(self._next_write_pos) if p not in removed]
        remap = {p: i for i, p in enumerate(survivors)}
        self._kv_positions = set(remap.values())
        self._token_at_pos = {
            remap[p]: t for p, t in self._token_at_pos.items() if p in remap
        }
        self._embeddings = {
            remap[p]: e for p, e in self._embeddings.items() if p in remap
        }
        self._next_write_pos = len(survivors)
        return True

    def kv_block_save(self, p0: int, p1: int, seq_id: int = 0) -> bytes:
        tokens = [self._token_at_pos.get(p, 0) for p in range(p0, p1)]
        out = bytearray(len(tokens).to_bytes(4, "little"))
        for tok in tokens:
            out += int(tok).to_bytes(4, "little", signed=True)
        return bytes(out)

    def kv_block_load(self, data: bytes, new_p0: int, seq_id: int = 0) -> bool:
        if len(data) < 4:
            return False
        n = int.from_bytes(data[:4], "little")
        for i in range(n):
            tok = int.from_bytes(data[4 + i * 4 : 8 + i * 4], "little", signed=True)
            pos = new_p0 + i
            self._kv_positions.add(pos)
            self._token_at_pos[pos] = tok
            self._embeddings[pos] = self._rng.randn(self._n_embd).astype(np.float32)
        self._next_write_pos = max(self._next_write_pos, new_p0 + n)
        return True

    def reset(self) -> None:
        self._kv_positions.clear()
        self._token_at_pos.clear()
        self._embeddings.clear()
        self._next_write_pos = 0
        self._next_gen_token = 0

    def state_save(self):
        return (
            dict(self._token_at_pos),
            {k: v.copy() for k, v in self._embeddings.items()},
            set(self._kv_positions),
            self._next_write_pos,
            self._next_gen_token,
        )

    def state_restore(self, snapshot) -> bool:
        token_at_pos, embeddings, kv_positions, next_write, next_gen = snapshot
        self._token_at_pos = dict(token_at_pos)
        self._embeddings = {k: v.copy() for k, v in embeddings.items()}
        self._kv_positions = set(kv_positions)
        self._next_write_pos = next_write
        self._next_gen_token = next_gen
        return True

    def get_embeddings(self, token_positions: list[int]) -> np.ndarray:
        result = np.zeros((len(token_positions), self._n_embd), dtype=np.float32)
        for i, pos in enumerate(token_positions):
            if pos in self._embeddings:
                result[i] = self._embeddings[pos]
            else:
                result[i] = self._rng.randn(self._n_embd).astype(np.float32)
        return result

    @property
    def supports_kv_block(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True

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
