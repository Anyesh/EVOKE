import ctypes
import os

from evoke._engine_lib import llama_cpp


def describe(label, cdll):
    if cdll is None:
        print(f"{label}: None")
        return None
    name = getattr(cdll, "_name", "?")
    handle = getattr(cdll, "_handle", 0)
    print(f"{label}: name={name} handle={hex(handle)}")
    return handle


print("LLAMA_CPP_LIB      =", os.environ.get("LLAMA_CPP_LIB"))
print("LLAMA_CPP_LIB_PATH =", os.environ.get("LLAMA_CPP_LIB_PATH"))
print("llama_cpp.__version__ =", getattr(llama_cpp, "__version__", "?"))
print("llama_cpp.__file__    =", getattr(llama_cpp, "__file__", "?"))

bindings = getattr(llama_cpp, "llama_cpp", None)
lcpy_lib = getattr(bindings, "_lib", None) if bindings is not None else None
if lcpy_lib is None:
    lcpy_lib = getattr(llama_cpp, "_lib", None)
if lcpy_lib is None and bindings is not None:
    for attr in dir(bindings):
        val = getattr(bindings, attr, None)
        if isinstance(val, ctypes.CDLL):
            lcpy_lib = val
            break

h1 = describe("llama-cpp-python CDLL", lcpy_lib)

from evoke.llama_engine import _kv_block_lib

h2 = describe("evoke _kv_block_lib  ", _kv_block_lib)

if h1 is not None and h2 is not None:
    print("SAME LOADED MODULE :", h1 == h2)
