FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Pre-built CPU-only shared libs from the EVOKE fork (master branch, linux/amd64).
# Built locally to avoid cmake timing out on HF's free build hardware (~30 min cap).
COPY lib/ /usr/local/lib/evoke/
RUN cd /usr/local/lib/evoke && \
    ln -sf libllama.so.0.0.9627     libllama.so.0   && \
    ln -sf libllama.so.0            libllama.so     && \
    ln -sf libggml.so.0.15.1        libggml.so.0    && \
    ln -sf libggml.so.0             libggml.so      && \
    ln -sf libggml-base.so.0.15.1   libggml-base.so.0 && \
    ln -sf libggml-base.so.0        libggml-base.so && \
    ln -sf libggml-cpu.so.0.15.1    libggml-cpu.so.0 && \
    ln -sf libggml-cpu.so.0         libggml-cpu.so  && \
    ldconfig /usr/local/lib/evoke

ENV LLAMA_CPP_LIB=/usr/local/lib/evoke/libllama.so
ENV LD_LIBRARY_PATH="/usr/local/lib/evoke:$LD_LIBRARY_PATH"

WORKDIR /evoke
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY demo/ demo/

RUN uv python install 3.12 && uv venv --python 3.12 && \
    uv pip install "llama-cpp-python>=0.3.0" \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu && \
    uv sync --extra server --extra demo

ENV EVOKE_MODEL_PATH=/models/qwen2.5-3b-instruct-q4_k_m.gguf
ENV EVOKE_HOST=127.0.0.1
ENV EVOKE_N_CTX=8192
ENV EVOKE_BUDGET=384
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
hf_hub_download(repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF', \
                filename='qwen2.5-3b-instruct-q4_k_m.gguf', \
                local_dir='/models')\" && \
  echo 'Starting EVOKE server (port 8000)...' && \
  EVOKE_PORT=8000 EVOKE_POLICY=evoke EVOKE_RECOVERY_MODE=kv_restore \
    EVOKE_RECOVERY_MATCH=identity EVOKE_RECOVERY_PROTECT_THRESHOLD=0.5 \
    EVOKE_BUDGET=384 \
    uv run python scripts/evoke_serve.py & \
  echo 'Waiting for EVOKE server...' && \
  until curl -sf http://127.0.0.1:8000/health; do sleep 3; done && \
  echo 'Starting discard server (port 8001)...' && \
  EVOKE_PORT=8001 EVOKE_POLICY=truncate EVOKE_RECOVERY_MODE=discard EVOKE_BUDGET=384 \
    uv run python scripts/evoke_serve.py & \
  echo 'Waiting for discard server...' && \
  until curl -sf http://127.0.0.1:8001/health; do sleep 3; done && \
  echo 'Backends ready. Starting Gradio...' && \
  uv run python demo/app.py \
"]
