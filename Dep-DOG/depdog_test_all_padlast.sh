#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Representative batch-eval script for the old PAD-last evaluation family.

DATASET="cd/cd4/k1"
META_FILE="meta.pkl"
TEST_FILE="cd4_test.jsonl"
OUT_ROOT="out/depdog_train"
GPU_DEVICE="cuda:0"

N_LAYER=3
N_HEAD=7
N_EMBD=224

ITERS_R1=200000
ITERS_R2=200000
ITERS_R3=200000
MAX_TIME_STEP=20

SAFE_DATASET_NAME="${DATASET//\//_}"

run_eval() {
    local suffix="$1"
    local iters="$2"
    local ckpt_dir="${OUT_ROOT}/${SAFE_DATASET_NAME}_${N_LAYER}_${N_HEAD}_${N_EMBD}_${suffix}_${iters}"

    python depdog_eval.py \
        --dataset="$DATASET" \
        --meta_file="$META_FILE" \
        --test_file="$TEST_FILE" \
        --out_dir="$ckpt_dir" \
        --ckpt_iter="$iters" \
        --device="$GPU_DEVICE" \
        --max_time_step="$MAX_TIME_STEP" \
        --save_suffix="_padlast" \
        --delay_pad
}

run_eval "round1" "$ITERS_R1"
run_eval "round2" "$ITERS_R2"
run_eval "round3" "$ITERS_R3"

echo "=== PAD-last batch evaluation finished ==="
