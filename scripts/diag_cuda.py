import ctypes
import os

BUILD_BIN = r"C:\Users\User\llama.cpp\build\bin"

print("build bin exists:", os.path.isdir(BUILD_BIN))
print("CUDA_PATH:", os.environ.get("CUDA_PATH"))

if os.path.isdir(BUILD_BIN):
    os.add_dll_directory(BUILD_BIN)

cuda = os.environ.get("CUDA_PATH")
if cuda:
    cuda_bin = os.path.join(cuda, "bin")
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
        print("added CUDA bin:", cuda_bin)
    else:
        print("CUDA bin not found at", cuda_bin)

for name in ["ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll", "llama.dll"]:
    path = os.path.join(BUILD_BIN, name)
    if not os.path.exists(path):
        print("MISSING:", name)
        continue
    try:
        ctypes.CDLL(path)
        print("LOAD OK:", name)
    except Exception as exc:  # noqa: BLE001
        print("LOAD FAIL:", name, "->", repr(exc))
