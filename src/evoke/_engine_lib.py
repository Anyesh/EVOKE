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
# `n_seq_max`, `ctx_type` follows `n_threads_batch`, later fork commits add
# `n_outputs_max` (and `n_outputs_max_per_seq`) after `n_rs_seq` plus a trailing
# `ctx_other`, and upstream (2026-10) inserts a `size_t moe_cache_size` after
# `type_v`. A struct that is short of the DLL's by even one field puts every
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
_MID_FIELDS = [
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
]
_TAIL_FIELDS = [
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


def _sized(name: str, fields: list, size: int) -> type[ctypes.Structure]:
    cls = type(name, (ctypes.Structure,), {"_fields_": fields})
    if size > ctypes.sizeof(cls):
        fields = fields + [("_tail", ctypes.c_ubyte * (size - ctypes.sizeof(cls)))]
        cls = type(name, (ctypes.Structure,), {"_fields_": fields})
    return cls


def context_params_class(
    extra_u32: int, moe_cache: bool = False, size: int = 0
) -> type[ctypes.Structure]:
    fields = list(_HEAD_FIELDS)
    fields += [(f"n_extra{i}", _U32) for i in range(extra_u32)]
    fields += _MID_FIELDS
    if moe_cache:
        fields.append(("moe_cache_size", ctypes.c_size_t))
    fields += _TAIL_FIELDS
    return _sized("LlamaContextParamsFork", fields, size)


def detect_context_layout(raw: bytes) -> tuple[int, bool] | None:
    # llama_context_default_params() sets the thread counts, leaves ctx_type at
    # 0, marks rope scaling, pooling, attention and flash attention as
    # unspecified (-1) and leaves the rope base and scale at 0 (from the
    # model). A layout that is one field short or long reads a different
    # field into each of those slots and fails at least one of the checks.
    # The flags after `type_v` (offload_kqv, no_perf, op_offload and swa_full
    # all default to true) tell whether `moe_cache_size` is present, because
    # guessing wrong lands them on zeroed callback or sampler pointers.
    for extra in range(_MAX_EXTRA_U32 + 1):
        for moe_cache in (False, True):
            cls = context_params_class(extra, moe_cache)
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
                and raw[cls.offload_kqv.offset] == 1
                and raw[cls.no_perf.offset] == 1
                and raw[cls.op_offload.offset] == 1
                and raw[cls.swa_full.offset] == 1
            ):
                return extra, moe_cache
    return None


# llama_model_params: llama-cpp-python's bundled struct matches fork builds up
# to 2026-07, but upstream (2026-10) inserts `load_mode` and `lazy_mode` after
# `split_mode`, drops `use_mmap`/`use_direct_io`/`use_mlock` and appends
# `load_mtp`. The struct crosses the boundary by value in both directions
# (default_params returns it, load_from_file takes it), so a short Python
# struct is overrun and the DLL reads the tail flags, `vocab_only` among them,
# from whatever memory follows.
_MODEL_HEAD_FIELDS = [
    ("devices", _PTR),
    ("tensor_buft_overrides", _PTR),
    ("n_gpu_layers", _I32),
    ("split_mode", ctypes.c_int),
]
_MODEL_MID_FIELDS = [
    ("main_gpu", _I32),
    ("tensor_split", _PTR),
    ("progress_callback", _PTR),
    ("progress_callback_user_data", _PTR),
    ("kv_overrides", _PTR),
    ("vocab_only", ctypes.c_bool),
]
_MODEL_TAIL_FIELDS = [
    ("check_tensors", ctypes.c_bool),
    ("use_extra_bufts", ctypes.c_bool),
    ("no_host", ctypes.c_bool),
    ("no_alloc", ctypes.c_bool),
]
_LOAD_MODE_AUTO = -1


def model_params_class(load_mode: bool, size: int = 0) -> type[ctypes.Structure]:
    fields = list(_MODEL_HEAD_FIELDS)
    if load_mode:
        fields += [("load_mode", ctypes.c_int), ("lazy_mode", ctypes.c_int)]
    fields += _MODEL_MID_FIELDS
    if not load_mode:
        fields += [
            ("use_mmap", ctypes.c_bool),
            ("use_direct_io", ctypes.c_bool),
            ("use_mlock", ctypes.c_bool),
        ]
    fields += _MODEL_TAIL_FIELDS
    if load_mode:
        fields.append(("load_mtp", ctypes.c_bool))
    return _sized("LlamaModelParamsFork", fields, size)


def detect_model_layout(raw: bytes) -> bool | None:
    # The int slot after split_mode holds main_gpu (default 0) in the bundled
    # layout and load_mode (default LLAMA_LOAD_MODE_AUTO = -1) in the newer one.
    old = model_params_class(False)
    if len(raw) < ctypes.sizeof(old):
        return None
    if ctypes.c_int32.from_buffer_copy(raw, old.n_gpu_layers.offset).value != -1:
        return None
    slot = ctypes.c_int32.from_buffer_copy(raw, old.main_gpu.offset).value
    if slot == _LOAD_MODE_AUTO:
        return True
    if slot == 0:
        return False
    return None


def _measure(fn) -> bytes:
    address = ctypes.cast(fn, ctypes.c_void_p).value
    fill = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(address)
    buf = (ctypes.c_ubyte * _RAW_SIZE)(*([_SENTINEL] * _RAW_SIZE))
    fill(ctypes.addressof(buf))
    return bytes(buf)


def _used_size(raw: bytes) -> int:
    used = max(i for i, b in enumerate(raw) if b != _SENTINEL) + 1
    return (used + 7) // 8 * 8


def _measured_context_params() -> tuple[int, bool] | None:
    _bindings = llama_cpp.llama_cpp
    lib = _bindings._lib
    raw = _measure(lib.llama_context_default_params)
    layout = detect_context_layout(raw)
    if layout is None:
        return None
    cls = context_params_class(*layout, size=_used_size(raw))
    lib.llama_context_default_params.restype = cls
    lib.llama_context_default_params.argtypes = []
    lib.llama_init_from_model.restype = ctypes.c_void_p
    lib.llama_init_from_model.argtypes = [ctypes.c_void_p, cls]
    llama_cpp.llama_context_default_params = lib.llama_context_default_params
    llama_cpp.llama_init_from_model = lib.llama_init_from_model
    llama_cpp.llama_context_params = cls
    _bindings.llama_context_params = cls
    return layout


def _measured_model_params() -> bool | None:
    _bindings = llama_cpp.llama_cpp
    lib = _bindings._lib
    raw = _measure(lib.llama_model_default_params)
    load_mode = detect_model_layout(raw)
    if not load_mode:
        return load_mode
    cls = model_params_class(True, size=_used_size(raw))
    lib.llama_model_default_params.restype = cls
    lib.llama_model_default_params.argtypes = []
    lib.llama_model_load_from_file.restype = ctypes.c_void_p
    lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, cls]
    llama_cpp.llama_model_default_params = lib.llama_model_default_params
    llama_cpp.llama_model_load_from_file = lib.llama_model_load_from_file
    llama_cpp.llama_model_params = cls
    _bindings.llama_model_params = cls
    return True


# None means the loaded llama.dll is not a known fork layout (for example the
# stock wheel build). LlamaCppEngine refuses that combination when a custom
# fork build was requested, because its settings would be written to wrong
# offsets without any error.
CONTEXT_PARAMS_LAYOUT = _measured_context_params()
CONTEXT_PARAMS_EXTRA_U32 = (
    None if CONTEXT_PARAMS_LAYOUT is None else CONTEXT_PARAMS_LAYOUT[0]
)
MODEL_PARAMS_LOAD_MODE = _measured_model_params()
