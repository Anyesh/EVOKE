#!/usr/bin/env bash
# Tuning sweep for the recovery-aware eviction params (w_recovery,
# recovery_decay) at T=14, evoke policy only. The four combinations
# below sit at the corners of (low/high w_recovery) x (faster/slower
# decay). Output goes to results/tune_recovery/<tag>/T14_S*.json so the
# existing aggregator works without modification.
#
# Required env (same as run_session_length.sh):
#   EVOKE_SSH_HOST       user@gpu-host
#   EVOKE_LIB_PATH       remote path to llama.dll
#   EVOKE_MODEL_PATH     remote path to GGUF
#   EVOKE_SERVER         http://gpu-host:port
#   EVOKE_REMOTE_DIR     remote project directory
# Optional:
#   EVOKE_BUDGET=1024
#   EVOKE_N_CTX=16384

set -e

: "${EVOKE_SSH_HOST:?need EVOKE_SSH_HOST}"
: "${EVOKE_LIB_PATH:?need EVOKE_LIB_PATH}"
: "${EVOKE_MODEL_PATH:?need EVOKE_MODEL_PATH}"
: "${EVOKE_SERVER:?need EVOKE_SERVER}"

mkdir -p results/tune_recovery

combos=(
    "0.5 0.7"
    "1.0 0.7"
    "1.0 0.5"
    "0.5 0.5"
)

for combo in "${combos[@]}"; do
    read -r w d <<< "$combo"
    tag="w${w//./_}_d${d//./_}"
    echo "================================================================"
    echo "=== tuning combo: w_recovery=${w} recovery_decay=${d} (tag=${tag})"
    echo "================================================================"
    EVOKE_OUT_DIR="results/tune_recovery/${tag}" \
    EVOKE_TURNS_LIST="14" \
    EVOKE_SEED_LIST="1 2 3 4 5" \
    EVOKE_USE_RETRIEVAL_EMBEDDINGS="1" \
    EVOKE_W_RECOVERY="${w}" \
    EVOKE_RECOVERY_DECAY="${d}" \
    EVOKE_BENCH_POLICIES="evoke" \
        bash scripts/run_session_length.sh
done

echo "Tuning sweep done."
