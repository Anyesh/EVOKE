#!/usr/bin/env bash
# Run baseline_bench.py 15 times against the EVOKE server on a GPU host to
# tighten the Section 7.3 head-to-head wall-clock CIs from n=5 to n=15.
# Each invocation restarts the server per policy (no_eviction, truncate, evoke)
# and drives a fresh 14-turn planted-fact session. Per-run JSONs land in
# results/baseline_n15/, and the human-readable transcript concatenates to
# results/baseline_bench_n15_qwen25_7b.txt.
#
# Required env (export before running):
#   EVOKE_SSH_HOST       user@gpu-host
#   EVOKE_LIB_PATH       remote path to the EVOKE fork's shared library
#   EVOKE_MODEL_PATH     remote path to GGUF
#   EVOKE_SERVER         http://gpu-host:port
#   EVOKE_REMOTE_DIR     remote project directory
# Optional:
#   EVOKE_BUDGET=1024
#   EVOKE_N_CTX=16384
#   EVOKE_MODEL_NAME=qwen25

set -euo pipefail

: "${EVOKE_SSH_HOST:?need EVOKE_SSH_HOST}"
: "${EVOKE_LIB_PATH:?need EVOKE_LIB_PATH}"
: "${EVOKE_MODEL_PATH:?need EVOKE_MODEL_PATH}"
: "${EVOKE_SERVER:?need EVOKE_SERVER}"
: "${EVOKE_REMOTE_DIR:?need EVOKE_REMOTE_DIR}"

export EVOKE_BUDGET="${EVOKE_BUDGET:-1024}"
export EVOKE_N_CTX="${EVOKE_N_CTX:-16384}"
export EVOKE_MODEL_NAME="${EVOKE_MODEL_NAME:-qwen25}"

cd "$(dirname "$0")/.."

OUT="results/baseline_bench_n15_qwen25_7b.txt"
JSON_DIR="results/baseline_n15"
mkdir -p "$JSON_DIR"
: > "$OUT"

for i in $(seq 1 15); do
    echo "=== Run $i ===" | tee -a "$OUT"
    EVOKE_BENCH_OUT="$JSON_DIR/run_$i.json" PYTHONUNBUFFERED=1 \
        uv run -- python -u scripts/baseline_bench.py 2>&1 | tee -a "$OUT"
done

echo "wrote $OUT"
