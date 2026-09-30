"""Loads the llama.cpp shared library, pre-loading the matching ggml DLLs.

When LLAMA_CPP_LIB points at a custom build, its ggml dependencies must be
loaded from the same directory first; otherwise Windows resolves ggml.dll to
llama-cpp-python's bundled copy and the ggml backends fail to register
(GGML_ASSERT(backend) failed). Pre-loading by full path pins the right copy:
once a DLL of a given base name is loaded, the loader reuses it.
"""

from __future__ import annotations

import ctypes
import importlib
import os


def _preload_ggml() -> None:
    lib = os.environ.get("LLAMA_CPP_LIB")
    if not lib:
        return
    lib_dir = os.path.dirname(os.path.abspath(lib))
    if not os.path.isdir(lib_dir):
        return

    dll_dirs = [lib_dir]
    # ggml-cuda.dll needs the CUDA runtime libraries; CUDA 13 places them
    # under bin\x64, older toolkit layouts put them directly under bin
    cuda = os.environ.get("CUDA_PATH")
    if cuda:
        for sub in (os.path.join("bin", "x64"), "bin"):
            cuda_dir = os.path.join(cuda, sub)
            if os.path.isdir(cuda_dir):
                dll_dirs.append(cuda_dir)

    if hasattr(os, "add_dll_directory"):
        for directory in dll_dirs:
            os.add_dll_directory(directory)

    for name in ("ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll"):
        path = os.path.join(lib_dir, name)
        if os.path.exists(path):
            try:
                ctypes.CDLL(path)
            except OSError:
                pass


_preload_ggml()

# llama-cpp-python 0.3.x selects its llama.dll from the LLAMA_CPP_LIB_PATH
# directory and ignores LLAMA_CPP_LIB entirely. This binding and llama-cpp-python
# must load the same llama.dll file: a llama_context built by one is an opaque
# pointer whose struct layout the other does not match, so calling a method on a
# cross-loaded context crashes. Route llama-cpp-python at the custom build dir so
# a single loaded module backs both.
_custom_lib = os.environ.get("LLAMA_CPP_LIB")
if _custom_lib:
    os.environ["LLAMA_CPP_LIB_PATH"] = os.path.dirname(os.path.abspath(_custom_lib))

llama_cpp = importlib.import_module("llama_cpp")


# The fork's `llama_context_params` differs from llama-cpp-python 0.3.23's
# bundled struct, and differs between fork builds: `n_rs_seq` follows
# `n_seq_max`, `ctx_type` follows `n_threads_batch`, and later fork commits add
# `n_outputs_max` (and `n_outputs_max_per_seq`) after `n_rs_seq` plus a trailing
# `ctx_other`. A struct that is short of the DLL's by even one field puts every
# later setting at the wrong offset: `type_k`/`type_v`, `embeddings` and the rope
# and YaRN fields land on other fields and are silently ignored. Passing the
# struct by value also makes the DLL read past a shorter one. The layout is
# therefore measured from the loaded DLL instead of assumed.
_U32, _I32, _F32, _PTR = (
    ctypes.c_uint32,
    ctypes.c_int32,
    ctypes.c_float,
    ctypes.c_void_p,
)
_HEAD_FIELDS = [
    ("n_ctx", _U32),
    ("n_batch", _U32),
    ("n_ubatch", _U32),
    ("n_seq_max", _U32),
    ("n_rs_seq", _U32),
]
_TAIL_FIELDS = [
    ("n_threads", _I32),
    ("n_threads_batch", _I32),
    ("ctx_type", ctypes.c_int),
    ("rope_scaling_type", ctypes.c_int),
    ("pooling_type", ctypes.c_int),
    ("attention_type", ctypes.c_int),
    ("flash_attn_type", ctypes.c_int),
    ("rope_freq_base", _F32),
    ("rope_freq_scale", _F32),
    ("yarn_ext_factor", _F32),
    ("yarn_attn_factor", _F32),
    ("yarn_beta_fast", _F32),
    ("yarn_beta_slow", _F32),
    ("yarn_orig_ctx", _U32),
    ("defrag_thold", _F32),
    ("cb_eval", _PTR),
    ("cb_eval_user_data", _PTR),
    ("type_k", ctypes.c_int),
    ("type_v", ctypes.c_int),
    ("abort_callback", _PTR),
    ("abort_callback_data", _PTR),
    ("embeddings", ctypes.c_bool),
    ("offload_kqv", ctypes.c_bool),
    ("no_perf", ctypes.c_bool),
    ("op_offload", ctypes.c_bool),
    ("swa_full", ctypes.c_bool),
    ("kv_unified", ctypes.c_bool),
    ("samplers", _PTR),
    ("n_samplers", ctypes.c_size_t),
]
_MAX_EXTRA_U32 = 4
_RAW_SIZE = 1024
_SENTINEL = 0xA5


def context_params_class(extra_u32: int, size: int = 0) -> type[ctypes.Structure]:
    fields = list(_HEAD_FIELDS)
    fields += [(f"n_extra{i}", _U32) for i in range(extra_u32)]
    fields += _TAIL_FIELDS
    cls = type("LlamaContextParamsFork", (ctypes.Structure,), {"_fields_": fields})
    if size > ctypes.sizeof(cls):
        fields.append(("_tail", ctypes.c_ubyte * (size - ctypes.sizeof(cls))))
        cls = type("LlamaContextParamsFork", (ctypes.Structure,), {"_fields_": fields})
    return cls


def detect_extra_u32(raw: bytes) -> int | None:
    # llama_context_default_params() sets the thread counts, leaves ctx_type at
    # 0, marks rope scaling, pooling, attention and flash attention as
    # unspecified (-1) and leaves the rope base and scale at 0 (from the
    # model). A layout that is one field short or long reads a different
    # field into each of those slots and fails at least one of the checks.
    for extra in range(_MAX_EXTRA_U32 + 1):
        cls = context_params_class(extra)
        if len(raw) < ctypes.sizeof(cls):
            continue
        p = cls.from_buffer_copy(raw[: ctypes.sizeof(cls)])
        if (
            p.n_threads > 0
            and p.n_threads_batch > 0
            and p.ctx_type == 0
            and p.rope_scaling_type == -1
            and p.pooling_type == -1
            and p.attention_type == -1
            and p.flash_attn_type == -1
            and p.rope_freq_base == 0.0
            and p.rope_freq_scale == 0.0
        ):
            return extra
    return None


def _measured_default_params() -> int | None:
    _bindings = llama_cpp.llama_cpp
    lib = _bindings._lib
    address = ctypes.cast(lib.llama_context_default_params, ctypes.c_void_p).value
    fill = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(address)
    buf = (ctypes.c_ubyte * _RAW_SIZE)(*([_SENTINEL] * _RAW_SIZE))
    fill(ctypes.addressof(buf))
    raw = bytes(buf)
    extra = detect_extra_u32(raw)
    if extra is None:
        return None
    used = max(i for i, b in enumerate(raw) if b != _SENTINEL) + 1
    cls = context_params_class(extra, (used + 7) // 8 * 8)
    lib.llama_context_default_params.restype = cls
    lib.llama_context_default_params.argtypes = []
    lib.llama_init_from_model.restype = ctypes.c_void_p
    lib.llama_init_from_model.argtypes = [ctypes.c_void_p, cls]
    llama_cpp.llama_context_default_params = lib.llama_context_default_params
    llama_cpp.llama_init_from_model = lib.llama_init_from_model
    llama_cpp.llama_context_params = cls
    _bindings.llama_context_params = cls
    return extra


# None means the loaded llama.dll is not a known fork layout (for example the
# stock wheel build). LlamaCppEngine refuses that combination when a custom
# fork build was requested, because its settings would be written to wrong
# offsets without any error.
CONTEXT_PARAMS_EXTRA_U32 = _measured_default_params()
