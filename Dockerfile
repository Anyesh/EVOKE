FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# CUDA toolkit from NVIDIA apt repo — gives nvcc for compilation without requiring
# the nvidia container runtime that the nvidia/cuda base image needs.
RUN apt-get update && apt-get install -y \
    wget gnupg cmake git python3 python3-pip curl \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-1 \
    && rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

ENV PATH="/usr/local/cuda-12.1/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

RUN git clone --depth=1 -b master https://github.com/Anyesh/llama.cpp.git /llama.cpp
RUN cmake -B /llama.cpp/build -S /llama.cpp \
      -DGGML_CUDA=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
    && cmake --build /llama.cpp/build --config Release -j$(nproc)

ENV LLAMA_CPP_LIB=/llama.cpp/build/bin/libllama.so

WORKDIR /evoke
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY demo/ demo/

RUN uv venv && uv sync --extra server --extra demo

ENV EVOKE_MODEL_PATH=/models/qwen2.5-7b-instruct-q4_k_m.gguf
ENV EVOKE_HOST=127.0.0.1
ENV EVOKE_N_CTX=32768
ENV EVOKE_BUDGET=2048
ENV EVOKE_RECOVERY_MODE=kv_restore
ENV EVOKE_POLICY=evoke
ENV EVOKE_SERVER_URL=http://127.0.0.1:8000
ENV DISCARD_SERVER_URL=http://127.0.0.1:8001

EXPOSE 7860

CMD ["bash", "-c", "\
  echo 'Downloading model...' && \
  uv run python -c \"\
import os; from huggingface_hub import hf_hub_download; \
os.makedirs('/models', exist_ok=True); \
hf_hub_download(repo_id='Qwen/Qwen2.5-7B-Instruct-GGUF', \
                filename='qwen2.5-7b-instruct-q4_k_m.gguf', \
                local_dir='/models')\" && \
  echo 'Starting EVOKE server (port 8000)...' && \
  EVOKE_PORT=8000 EVOKE_POLICY=evoke EVOKE_RECOVERY_MODE=kv_restore \
    uv run python scripts/evoke_serve.py & \
  echo 'Starting discard server (port 8001)...' && \
  EVOKE_PORT=8001 EVOKE_POLICY=truncate EVOKE_RECOVERY_MODE=discard EVOKE_BUDGET=2048 \
    uv run python scripts/evoke_serve.py & \
  echo 'Waiting for backends...' && \
  until curl -sf http://127.0.0.1:8000/health && curl -sf http://127.0.0.1:8001/health; do \
    sleep 5; \
  done && \
  echo 'Backends ready. Starting Gradio...' && \
  uv run python demo/app.py \
"]
