#!/usr/bin/env bash
# Session-length scaling sweep for reviewer concern #2 (Section 7.3).
# Runs baseline_bench.py at TURNS in {14, 28, 56, 112} for SEEDS=1..5
# per (turns, seed) pair against the EVOKE server on gpuhost.
# Each invocation restarts the server per policy (no_eviction, truncate, evoke).
# Outputs per-cell JSON to results/session_length/T<turns>_S<seed>.json.
#
# Required env (export before running):
#   EVOKE_SSH_HOST       user@gpu-host
#   EVOKE_LIB_PATH       remote path to llama.dll
#   EVOKE_MODEL_PATH     remote path to GGUF
#   EVOKE_SERVER         http://gpu-host:port (EVOKE OpenAI-compat server)
#   EVOKE_REMOTE_DIR     remote project directory
# Optional:
#   EVOKE_BUDGET=1024
#   EVOKE_N_CTX=16384
#   EVOKE_TURNS_LIST="14 28 56 112"
#   EVOKE_SEED_LIST="1 2 3 4 5"
#   EVOKE_OUT_DIR=results/session_length

set -e

: "${EVOKE_SSH_HOST:?need EVOKE_SSH_HOST}"
: "${EVOKE_LIB_PATH:?need EVOKE_LIB_PATH}"
: "${EVOKE_MODEL_PATH:?need EVOKE_MODEL_PATH}"
: "${EVOKE_SERVER:?need EVOKE_SERVER}"

OUT_DIR="${EVOKE_OUT_DIR:-results/session_length}"
mkdir -p "$OUT_DIR"

TURNS_LIST=(${EVOKE_TURNS_LIST:-14 28 56 112})
SEED_LIST=(${EVOKE_SEED_LIST:-1 2 3 4 5})

for turns in "${TURNS_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        out_json="$OUT_DIR/T${turns}_S${seed}.json"
        out_txt="$OUT_DIR/T${turns}_S${seed}.txt"
        if [ -f "$out_json" ]; then
            echo "SKIP T=${turns} S=${seed} (already done: $out_json)"
            continue
        fi
        echo "================================================================"
        echo "=== turns=${turns} seed=${seed} ($(date -Iseconds))"
        echo "================================================================"
        EVOKE_BENCH_TURNS="$turns" \
        EVOKE_BENCH_OUT="$out_json" \
            uv run python scripts/baseline_bench.py 2>&1 \
            | tee "$out_txt"
        echo "=== done turns=${turns} seed=${seed} ($(date -Iseconds))"
    done
done

echo "ALL DONE"
