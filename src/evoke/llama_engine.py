from __future__ import annotations

import ctypes
from pathlib import Path

import llama_cpp
import numpy as np


class LlamaCppEngine:
    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 131072,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        verbose: bool = False,
    ):
        self._n_batch = n_batch

        llama_cpp.llama_backend_init()

        model_params = llama_cpp.llama_model_default_params()
        model_params.n_gpu_layers = n_gpu_layers
        self._model_ptr = llama_cpp.llama_model_load_from_file(
            str(model_path).encode(), model_params
        )
        if not self._model_ptr:
            raise RuntimeError(f"Failed to load model: {model_path}")

        ctx_params = llama_cpp.llama_context_default_params()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = n_batch
        ctx_params.n_ubatch = min(n_batch, 512)
        ctx_params.n_seq_max = 1
        ctx_params.embeddings = True
        ctx_params.flash_attn_type = 0
        ctx_params.no_perf = True

        self._ctx = llama_cpp.llama_init_from_model(self._model_ptr, ctx_params)
        if not self._ctx:
            llama_cpp.llama_model_free(self._model_ptr)
            raise RuntimeError("Failed to create context")

        self._vocab = llama_cpp.llama_model_get_vocab(self._model_ptr)
        self._memory = llama_cpp.llama_get_memory(self._ctx)
        self._n_embd = llama_cpp.llama_model_n_embd(self._model_ptr)
        self._batch = llama_cpp.llama_batch_init(n_batch, 0, 1)

        self._token_count = 0
        self._next_write_pos = 0
        self._emb_cache: dict[int, np.ndarray] = {}
        chain_params = llama_cpp.llama_sampler_chain_default_params()
        self._sampler = llama_cpp.llama_sampler_chain_init(chain_params)
        llama_cpp.llama_sampler_chain_add(
            self._sampler, llama_cpp.llama_sampler_init_greedy()
        )

    def tokenize(self, text: str) -> list[int]:
        text_bytes = text.encode("utf-8")
        max_tokens = len(text_bytes) + 16
        tokens = (llama_cpp.llama_token * max_tokens)()
        n = llama_cpp.llama_tokenize(
            self._vocab, text_bytes, len(text_bytes), tokens, max_tokens, False, True
        )
        if n < 0:
            max_tokens = -n + 16
            tokens = (llama_cpp.llama_token * max_tokens)()
            n = llama_cpp.llama_tokenize(
                self._vocab,
                text_bytes,
                len(text_bytes),
                tokens,
                max_tokens,
                False,
                True,
            )
        return list(tokens[:n])

    def detokenize(self, tokens: list[int]) -> str:
        pieces: list[bytes] = []
        buf = (ctypes.c_char * 256)()
        for tok in tokens:
            n = llama_cpp.llama_token_to_piece(self._vocab, tok, buf, 256, 0, True)
            if n > 0:
                pieces.append(bytes(buf[:n]))
        return b"".join(pieces).decode("utf-8", errors="replace")

    def process_tokens(self, tokens: list[int]) -> None:
        start_pos = self._next_write_pos

        for chunk_start in range(0, len(tokens), self._n_batch):
            chunk = tokens[chunk_start : chunk_start + self._n_batch]
            self._batch.n_tokens = len(chunk)

            for i, token_id in enumerate(chunk):
                self._batch.token[i] = token_id
                self._batch.pos[i] = start_pos + chunk_start + i
                self._batch.seq_id[i][0] = 0
                self._batch.n_seq_id[i] = 1
                self._batch.logits[i] = True

            ret = llama_cpp.llama_decode(self._ctx, self._batch)
            if ret != 0:
                raise RuntimeError(f"llama_decode failed with code {ret}")

            self._extract_embeddings(start_pos + chunk_start, len(chunk))

        self._next_write_pos += len(tokens)
        self._token_count += len(tokens)

    def generate_next(self) -> int:
        token = llama_cpp.llama_sampler_sample(self._sampler, self._ctx, -1)

        pos = self._next_write_pos
        self._batch.n_tokens = 1
        self._batch.token[0] = token
        self._batch.pos[0] = pos
        self._batch.seq_id[0][0] = 0
        self._batch.n_seq_id[0] = 1
        self._batch.logits[0] = True

        ret = llama_cpp.llama_decode(self._ctx, self._batch)
        if ret != 0:
            raise RuntimeError(f"llama_decode failed with code {ret}")

        self._next_write_pos += 1
        self._token_count += 1
        return token

    def get_kv_cache_token_count(self) -> int:
        return self._token_count

    def kv_cache_seq_rm(self, pos_start: int, pos_end: int) -> None:
        llama_cpp.llama_memory_seq_rm(self._memory, 0, pos_start, pos_end)
        self._token_count -= pos_end - pos_start
        for pos in range(pos_start, pos_end):
            self._emb_cache.pop(pos, None)

    def rebuild_kv(self, token_blocks: list[list[int]]) -> None:
        llama_cpp.llama_memory_clear(self._memory, True)
        self._token_count = 0
        self._next_write_pos = 0
        self._emb_cache.clear()
        for tokens in token_blocks:
            self.process_tokens(tokens)

    def reset(self) -> None:
        llama_cpp.llama_memory_clear(self._memory, True)
        self._token_count = 0
        self._next_write_pos = 0
        self._emb_cache.clear()

    @property
    def next_write_pos(self) -> int:
        return self._next_write_pos

    def get_embeddings(self, token_positions: list[int]) -> np.ndarray:
        result = np.zeros((len(token_positions), self._n_embd), dtype=np.float32)
        for i, pos in enumerate(token_positions):
            if pos in self._emb_cache:
                result[i] = self._emb_cache[pos]
        return result

    @property
    def n_ctx(self) -> int:
        return llama_cpp.llama_n_ctx(self._ctx)

    @property
    def n_embd(self) -> int:
        return self._n_embd

    @property
    def eos_token(self) -> int:
        return llama_cpp.llama_vocab_eos(self._vocab)

    @property
    def bos_token(self) -> int:
        return llama_cpp.llama_vocab_bos(self._vocab)

    def _extract_embeddings(self, batch_start_pos: int, n_tokens: int) -> None:
        emb_ptr = llama_cpp.llama_get_embeddings(self._ctx)
        if not emb_ptr:
            return

        raw = np.ctypeslib.as_array(
            ctypes.cast(emb_ptr, ctypes.POINTER(ctypes.c_float)),
            shape=(n_tokens, self._n_embd),
        )

        for i in range(n_tokens):
            self._emb_cache[batch_start_pos + i] = raw[i].copy()

    def close(self) -> None:
        if self._batch is not None:
            llama_cpp.llama_batch_free(self._batch)
            self._batch = None
        if self._sampler is not None:
            llama_cpp.llama_sampler_free(self._sampler)
            self._sampler = None
        if self._ctx is not None:
            llama_cpp.llama_free(self._ctx)
            self._ctx = None
        if self._model_ptr is not None:
            llama_cpp.llama_model_free(self._model_ptr)
            self._model_ptr = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> LlamaCppEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
