from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
from jinja2 import Environment, StrictUndefined
from jinja2.exceptions import TemplateError

from evoke._engine_lib import llama_cpp


def _jinja_raise(message: str) -> None:
    # raise_exception is a Hermes/Qwen template convention used inside the
    # Jinja template to abort on malformed inputs. Minja (llama.cpp's C++
    # Jinja parser) exposes it; we mirror the semantics here so Python-side
    # rendering of the same templates respects the same contract.
    raise RuntimeError(f"chat template raise_exception: {message}")


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
        # EVOKE attention-weight capture primitives (#30, multi-layer #39).
        lib.llama_attn_capture_set_layer.restype = None
        lib.llama_attn_capture_set_layer.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.llama_attn_capture_set_layers.restype = None
        lib.llama_attn_capture_set_layers.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
        ]
        lib.llama_attn_capture_set_buffer.restype = None
        lib.llama_attn_capture_set_buffer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.llama_attn_capture_get_dims.restype = None
        lib.llama_attn_capture_get_dims.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.llama_attn_capture_get_written.restype = ctypes.c_size_t
        lib.llama_attn_capture_get_written.argtypes = [ctypes.c_void_p]
        return lib
    except (OSError, AttributeError):
        return None


_kv_block_lib = _bind_kv_block_primitives()


_GGML_KV_TYPES = {
    "f32": 0,
    "f16": 1,
    "q4_0": 2,
    "q4_1": 3,
    "q5_0": 6,
    "q5_1": 7,
    "q8_0": 8,
}


def _resolve_kv_type(value: str | int | None, default: int = 1) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    key = value.lower().strip()
    if key not in _GGML_KV_TYPES:
        raise ValueError(
            f"unknown kv cache type {value!r}; expected one of {sorted(_GGML_KV_TYPES)}"
        )
    return _GGML_KV_TYPES[key]


class LlamaCppEngine:
    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 131072,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        verbose: bool = False,
        n_rs_seq: int = 0,
        type_k: str | int | None = None,
        type_v: str | int | None = None,
    ):
        # n_rs_seq: number of per-token snapshots the recurrent half keeps for
        # partial rollback. 0 disables (upstream default — recurrent seq_rm
        # only succeeds for whole-sequence wipe). Set > 0 on hybrid
        # (Mamba+Attention) models to enable mid-sequence eviction; cost is
        # roughly (1 + n_rs_seq) x the recurrent state memory per layer. For
        # Qwen 3.5 9B's hybrid layers, n_rs_seq=4096 supports up to 4k-token
        # thinking-trace eviction at ~couple of GB extra host RAM.
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
        ctx_params.n_rs_seq = n_rs_seq
        ctx_params.embeddings = True
        ctx_params.pooling_type = llama_cpp.LLAMA_POOLING_TYPE_NONE
        # FA must be on: it makes V row-aligned (v_trans=false), so kv_block_save
        # and kv_block_load take the contiguous single-memcpy-per-layer path. With
        # v_trans=true the V loop runs n_embd_v_gqa times per layer, dominating
        # recovery latency.
        ctx_params.flash_attn_type = 1
        ctx_params.no_perf = True
        ctx_params.type_k = _resolve_kv_type(type_k, default=ctx_params.type_k)
        ctx_params.type_v = _resolve_kv_type(type_v, default=ctx_params.type_v)
        self._kv_type_k = ctx_params.type_k
        self._kv_type_v = ctx_params.type_v

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
        self._attn_capture_buf: np.ndarray | None = None
        chain_params = llama_cpp.llama_sampler_chain_default_params()
        self._sampler = llama_cpp.llama_sampler_chain_init(chain_params)
        llama_cpp.llama_sampler_chain_add(
            self._sampler, llama_cpp.llama_sampler_init_greedy()
        )

    def get_chat_template_string(self) -> str | None:
        # Return the raw Jinja chat template string embedded in the GGUF, or
        # None if the model has no template metadata. Used by
        # apply_chat_template_with_tools so we can route tool-using requests
        # through Python jinja2 (the C llama_chat_apply_template doesn't
        # accept a tools array).
        tmpl_ptr = llama_cpp.llama_model_chat_template(self._model_ptr, None)
        if not tmpl_ptr:
            return None
        if isinstance(tmpl_ptr, bytes):
            return tmpl_ptr.decode("utf-8", errors="replace")
        return ctypes.string_at(tmpl_ptr).decode("utf-8", errors="replace")

    def apply_chat_template_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        add_generation_prompt: bool = True,
    ) -> str:
        # Render the GGUF chat template via Python jinja2 so we can pass
        # the tools array (which the C API can't). Falls back to the
        # tools-less C path when tools is None/empty so we keep the
        # byte-identical path that's already verified against opencode's
        # templating. Raises RuntimeError on render failure; the caller
        # should fall back to format_qwen_chat in that case.
        if not tools:
            return self.apply_chat_template(messages, add_generation_prompt)
        template_str = self.get_chat_template_string()
        if template_str is None:
            raise RuntimeError("model has no chat template embedded in GGUF")
        env = Environment(
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
            extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
        )
        env.globals["raise_exception"] = _jinja_raise
        try:
            template = env.from_string(template_str)
            return template.render(
                messages=messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
            )
        except TemplateError as exc:
            raise RuntimeError(f"jinja template render failed: {exc}") from exc

    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
    ) -> str:
        # Use the model's own chat template (read from GGUF metadata) so the
        # prompt we feed in matches byte-for-byte what an OpenAI client would
        # template back when echoing the conversation history. Without this,
        # any whitespace drift between our handwritten format_qwen_chat and
        # what the client sends triggers a session reset every turn.
        tmpl_ptr = llama_cpp.llama_model_chat_template(self._model_ptr, None)
        if not tmpl_ptr:
            raise RuntimeError("model has no chat template embedded in GGUF")
        n = len(messages)
        if n == 0:
            return ""
        msg_arr = (llama_cpp.llama_chat_message * n)()
        # Hold bytes alive while ctypes points into them.
        bufs: list[bytes] = []
        for i, m in enumerate(messages):
            role_b = (m.get("role") or "").encode("utf-8")
            content_b = (m.get("content") or "").encode("utf-8")
            bufs.extend([role_b, content_b])
            msg_arr[i].role = role_b
            msg_arr[i].content = content_b
        needed = llama_cpp.llama_chat_apply_template(
            tmpl_ptr, msg_arr, n, add_generation_prompt, None, 0
        )
        if needed < 0:
            raise RuntimeError(f"llama_chat_apply_template failed: {needed}")
        cap = needed + 1
        buf = ctypes.create_string_buffer(cap)
        written = llama_cpp.llama_chat_apply_template(
            tmpl_ptr, msg_arr, n, add_generation_prompt, buf, cap
        )
        if written < 0:
            raise RuntimeError(f"llama_chat_apply_template write failed: {written}")
        return buf.raw[:written].decode("utf-8", errors="replace")

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

    def evict_ranges(self, ranges: list[tuple[int, int]], compact: bool = True) -> bool:
        # Returns True if the eviction was applied. Hybrid (Mamba+Attention)
        # memories reject partial tail rollback in llama_memory_seq_rm (the
        # recurrent half cannot slice its state) and return false WITHOUT
        # mutating the attention cache; in that case we leave _next_write_pos
        # untouched so the engine and our token bookkeeping stay in sync.
        # An attention-only seq_rm is available as llama_kv_block_seq_rm but
        # using it would desync the recurrent position state from the attention
        # one and break decode on subsequent turns, so we accept that tail
        # eviction is a no-op on hybrid models.
        #
        # compact=True re-indexes survivors so positions stay contiguous
        # (seq_rm + seq_add). compact=False (sparse mode) drops the cells
        # with seq_rm only: survivors keep their true absolute positions, the
        # axis grows a hole, and the tail (_next_write_pos) is unchanged so new
        # tokens keep decoding at their genuine index. The hole is later filled
        # in place by kv_block_load at the original position.
        if not ranges:
            return True
        ranges = sorted(ranges)
        n = self._next_write_pos
        removed = 0
        for pos_start, pos_end in ranges:
            ok = llama_cpp.llama_memory_seq_rm(self._memory, 0, pos_start, pos_end)
            if not ok:
                return False
            removed += pos_end - pos_start
        if not compact:
            self._token_count -= removed
            return True
        cursor = 0
        shifted = 0
        for pos_start, pos_end in ranges:
            if shifted > 0 and pos_start > cursor:
                llama_cpp.llama_memory_seq_add(
                    self._memory, 0, cursor, pos_start, -shifted
                )
            shifted += pos_end - pos_start
            cursor = pos_end
        if shifted > 0 and cursor < n:
            llama_cpp.llama_memory_seq_add(self._memory, 0, cursor, n, -shifted)
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
            # max(): in sparse mode a block is spliced into a mid-cache
            # hole at new_p0 < tail, so the tail must not move backwards.
            self._next_write_pos = max(self._next_write_pos, new_p0 + n_cells)
            self._token_count += n_cells
        return ok

    def attn_capture_set_layer(self, layer: int) -> None:
        # Configure which layer's attention weights to capture on subsequent
        # process_tokens/generate_next calls. Pass -1 to disable. Requires the
        # EVOKE fork (LLAMA_CPP_LIB must point at it).
        if _kv_block_lib is None:
            raise RuntimeError(
                "Attention capture requires the EVOKE llama.cpp build; "
                "set LLAMA_CPP_LIB"
            )
        _kv_block_lib.llama_attn_capture_set_layer(self._ctx, ctypes.c_int32(layer))

    def attn_capture_set_layers(self, layers: list[int]) -> None:
        # Configure capture across multiple layers. Pass an empty list to
        # disable. The host buffer receives [n_layers, n_query, n_heads,
        # n_kv] f32 in the order given. Up to 16 layers; deeper layers
        # carry stronger semantic signal for the relevance scorer.
        if _kv_block_lib is None:
            raise RuntimeError(
                "Attention capture requires the EVOKE llama.cpp build; "
                "set LLAMA_CPP_LIB"
            )
        n = len(layers)
        arr = (ctypes.c_int32 * n)(*layers) if n > 0 else (ctypes.c_int32 * 0)()
        _kv_block_lib.llama_attn_capture_set_layers(self._ctx, arr, ctypes.c_int32(n))

    def attn_capture_set_buffer(self, buf: np.ndarray | None) -> None:
        # Provide a float32 numpy buffer for the C side to write attention
        # weights into after each decode. The buffer must remain alive while
        # capture is active (we hand C a raw pointer). Pass None to detach.
        if _kv_block_lib is None:
            raise RuntimeError(
                "Attention capture requires the EVOKE llama.cpp build; "
                "set LLAMA_CPP_LIB"
            )
        if buf is None:
            _kv_block_lib.llama_attn_capture_set_buffer(self._ctx, None, 0)
            self._attn_capture_buf = None
            return
        if buf.dtype != np.float32 or not buf.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "attn_capture_set_buffer expects a contiguous float32 numpy array"
            )
        self._attn_capture_buf = buf  # keep alive for the C side
        _kv_block_lib.llama_attn_capture_set_buffer(
            self._ctx, buf.ctypes.data_as(ctypes.c_void_p), buf.size
        )

    def attn_capture_get_dims(self) -> tuple[int, int, int, int]:
        # Returns (n_layers, n_query_tokens, n_heads, n_kv) — the f32 buffer
        # written by the last decode has shape [n_layers, n_query, n_heads,
        # n_kv]. Single-layer capture (legacy set_layer) reports n_layers=1.
        if _kv_block_lib is None:
            return (0, 0, 0, 0)
        n_l = ctypes.c_int32(0)
        n_q = ctypes.c_int32(0)
        n_h = ctypes.c_int32(0)
        n_kv = ctypes.c_int32(0)
        _kv_block_lib.llama_attn_capture_get_dims(
            self._ctx,
            ctypes.byref(n_l),
            ctypes.byref(n_q),
            ctypes.byref(n_h),
            ctypes.byref(n_kv),
        )
        return (int(n_l.value), int(n_q.value), int(n_h.value), int(n_kv.value))

    def attn_capture_get_written(self) -> int:
        if _kv_block_lib is None:
            return 0
        return int(_kv_block_lib.llama_attn_capture_get_written(self._ctx))

    def state_save(self) -> tuple[bytes, int, int, dict[int, np.ndarray]]:
        # Snapshot the engine state for session swap: the llama internal
        # state bytes (logits + KV cache contents via llama_state_get_data)
        # plus our Python-side bookkeeping (write head, token count,
        # embedding cache). The triple is opaque to callers — pass it back
        # to state_restore to resume.
        n = llama_cpp.llama_state_get_size(self._ctx)
        buf = (ctypes.c_uint8 * n)()
        written = llama_cpp.llama_state_get_data(self._ctx, buf, n)
        return (
            bytes(buf[:written]),
            self._next_write_pos,
            self._token_count,
            dict(self._emb_cache),
        )

    def state_restore(
        self,
        state: tuple[bytes, int, int, dict[int, np.ndarray]],
    ) -> None:
        # Restore a snapshot produced by state_save. Clears any existing
        # state first so we don't accidentally accumulate.
        data, next_write_pos, token_count, emb_cache = state
        llama_cpp.llama_memory_clear(self._memory, True)
        if data:
            buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
            llama_cpp.llama_state_set_data(self._ctx, buf, len(data))
        self._next_write_pos = next_write_pos
        self._token_count = token_count
        self._emb_cache = dict(emb_cache)

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
