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
