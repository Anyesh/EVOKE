from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from evoke._engine_lib import llama_cpp


@llama_cpp.llama_log_callback
def _null_log_callback(level, text, user_data):
    pass


def _suppress_llama_log():
    llama_cpp.llama_log_set(_null_log_callback, None)


def _bind_kv_block_primitives() -> ctypes.CDLL | None:
    # the EVOKE KV block primitives exist only in the custom llama.cpp build;
    # LLAMA_CPP_LIB must point at it for both llama-cpp-python and this binding
    lib_path = os.environ.get("LLAMA_CPP_LIB")
    if not lib_path:
        return None
    try:
        lib = ctypes.CDLL(lib_path)
        lib.llama_kv_block_save.restype = ctypes.c_size_t
        lib.llama_kv_block_save.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.llama_kv_block_load.restype = ctypes.c_bool
        lib.llama_kv_block_load.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int32,
        ]
        lib.llama_kv_block_seq_rm.restype = ctypes.c_bool
        lib.llama_kv_block_seq_rm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.llama_kv_block_seq_add.restype = None
        lib.llama_kv_block_seq_add.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        return lib
    except (OSError, AttributeError):
        return None


_kv_block_lib = _bind_kv_block_primitives()


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

        if not verbose:
            _suppress_llama_log()

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
        ctx_params.pooling_type = llama_cpp.LLAMA_POOLING_TYPE_NONE
        # FA must be on: it makes V row-aligned (v_trans=false), so kv_block_save
        # and kv_block_load take the contiguous single-memcpy-per-layer path. With
        # v_trans=true the V loop runs n_embd_v_gqa times per layer, dominating
        # recovery latency.
        ctx_params.flash_attn_type = 1
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

    def evict_ranges(self, ranges: list[tuple[int, int]]) -> bool:
        # Returns True if the eviction was applied. Hybrid (Mamba+Attention)
        # memories reject partial tail rollback in llama_memory_seq_rm (the
        # recurrent half cannot slice its state) and return false WITHOUT
        # mutating the attention cache; in that case we leave _next_write_pos
        # untouched so the engine and our token bookkeeping stay in sync.
        # An attention-only seq_rm is available as llama_kv_block_seq_rm but
        # using it would desync the recurrent position state from the attention
        # one and break decode on subsequent turns, so we accept that tail
        # eviction is a no-op on hybrid models.
        if not ranges:
            return True
        ranges = sorted(ranges)
        n = self._next_write_pos
        for pos_start, pos_end in ranges:
            ok = llama_cpp.llama_memory_seq_rm(self._memory, 0, pos_start, pos_end)
            if not ok:
                return False
        removed = 0
        cursor = 0
        for pos_start, pos_end in ranges:
            if removed > 0 and pos_start > cursor:
                llama_cpp.llama_memory_seq_add(
                    self._memory, 0, cursor, pos_start, -removed
                )
            removed += pos_end - pos_start
            cursor = pos_end
        if removed > 0 and cursor < n:
            llama_cpp.llama_memory_seq_add(self._memory, 0, cursor, n, -removed)
        self._next_write_pos = n - removed
        self._token_count = n - removed

        removed_set: set[int] = set()
        for pos_start, pos_end in ranges:
            removed_set.update(range(pos_start, pos_end))
        new_emb: dict[int, np.ndarray] = {}
        for pos, emb in self._emb_cache.items():
            if pos in removed_set:
                continue
            shift = sum(e - s for s, e in ranges if e <= pos)
            new_emb[pos - shift] = emb
        self._emb_cache = new_emb
        return True

    @property
    def supports_kv_block(self) -> bool:
        return _kv_block_lib is not None

    def kv_block_save(self, p0: int, p1: int, seq_id: int = 0) -> bytes:
        if _kv_block_lib is None:
            raise RuntimeError(
                "KV block primitives unavailable; set LLAMA_CPP_LIB to the "
                "EVOKE llama.cpp build"
            )
        needed = _kv_block_lib.llama_kv_block_save(self._ctx, seq_id, p0, p1, None, 0)
        if needed == 0:
            return b""
        buf = ctypes.create_string_buffer(needed)
        written = _kv_block_lib.llama_kv_block_save(
            self._ctx, seq_id, p0, p1, buf, needed
        )
        return buf.raw[:written]

    def kv_block_load(self, data: bytes, new_p0: int, seq_id: int = 0) -> bool:
        if _kv_block_lib is None:
            raise RuntimeError(
                "KV block primitives unavailable; set LLAMA_CPP_LIB to the "
                "EVOKE llama.cpp build"
            )
        ok = bool(
            _kv_block_lib.llama_kv_block_load(
                self._ctx, seq_id, data, len(data), new_p0
            )
        )
        if ok and len(data) >= 4:
            # the first 4 bytes of the buffer are the cell count (uint32) written
            # by block_write; advance our position tracking past the spliced span
            n_cells = int.from_bytes(data[:4], "little")
            self._next_write_pos = new_p0 + n_cells
            self._token_count += n_cells
        return ok

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
