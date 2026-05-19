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


# The EVOKE fork's `llama_context_params` adds two fields that llama-cpp-python
# 0.3.23's bindings don't know about: `n_rs_seq` (after `n_seq_max`) and
# `ctx_type` (after `n_threads_batch`). With the bundled struct, every field
# after `n_seq_max` is at the wrong offset, so `embeddings = True` lands on a
# different byte and the fork sees it as false: no embeddings are computed.
# Returning the fork's larger struct into the bundled struct's smaller buffer
# also overflows. We must use the fork-shaped struct for both the default-params
# call and the init call so the field offsets line up.
class _LlamaContextParamsFork(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int),
        ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int),
        ("attention_type", ctypes.c_int),
        ("flash_attn_type", ctypes.c_int),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.c_void_p),
        ("n_samplers", ctypes.c_size_t),
    ]


_bindings = llama_cpp.llama_cpp
_lib = _bindings._lib
_lib.llama_context_default_params.restype = _LlamaContextParamsFork
_lib.llama_context_default_params.argtypes = []
_lib.llama_init_from_model.restype = ctypes.c_void_p
_lib.llama_init_from_model.argtypes = [ctypes.c_void_p, _LlamaContextParamsFork]
llama_cpp.llama_context_default_params = _lib.llama_context_default_params
llama_cpp.llama_init_from_model = _lib.llama_init_from_model
llama_cpp.llama_context_params = _LlamaContextParamsFork
_bindings.llama_context_params = _LlamaContextParamsFork
