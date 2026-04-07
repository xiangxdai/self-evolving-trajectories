#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Representative end-to-end Dep-DOG pipeline script.
# This keeps one concrete example of the old run_pipeline_* family while the
# Python entrypoints are being consolidated.

NUM_GPUS=1
LAUNCH_CMD=(torchrun --standalone --nproc_per_node="$NUM_GPUS")
GPU_DEVICE="cuda:0"

DATASET="cd/cd4/k1"
META_FILE="meta.pkl"
TEST_FILE="cd4_test.jsonl"
OUT_ROOT="out/depdog_train"
CANONICAL_FILE="train.bin"

N_LAYER=3
N_HEAD=7
N_EMBD=224
BATCH_SIZE=$((512 / NUM_GPUS))
GEN_BATCH_SIZE=1024
COMPILE="True"

ITERS_R1=200000
ITERS_R2=200000
ITERS_R3=200000
MAX_TIME_STEP=20
MIX_RATIOS_R3="0.7,0.2,0.1"

SAFE_DATASET_NAME="${DATASET//\//_}"

model_dir() {
    local suffix="$1"
    local iters="$2"
    printf '%s/%s_%s_%s_%s_%s_%s' \
        "$OUT_ROOT" "$SAFE_DATASET_NAME" "$N_LAYER" "$N_HEAD" "$N_EMBD" "$suffix" "$iters"
}

run_eval() {
    local suffix="$1"
    local iters="$2"
    local ckpt_dir
    ckpt_dir="$(model_dir "$suffix" "$iters")"

    python depdog_eval.py \
        --dataset="$DATASET" \
        --meta_file="$META_FILE" \
        --test_file="$TEST_FILE" \
        --out_dir="$ckpt_dir" \
        --ckpt_iter="$iters" \
        --device="$GPU_DEVICE" \
        --max_time_step="$MAX_TIME_STEP" \
        --delay_pad
}

echo "=== Round 1 ==="
"${LAUNCH_CMD[@]}" depdog_train.py \
    --round=1 \
    --dataset="$DATASET" \
    --meta_file="$META_FILE" \
    --out_dir="$OUT_ROOT" \
    --canonical_file="$CANONICAL_FILE" \
    --n_layer="$N_LAYER" \
    --n_head="$N_HEAD" \
    --n_embd="$N_EMBD" \
    --batch_size="$BATCH_SIZE" \
    --gen_batch_size="$GEN_BATCH_SIZE" \
    --suffix="round1" \
    --max_iters="$ITERS_R1" \
    --compile="$COMPILE"
run_eval "round1" "$ITERS_R1"

ROUND1_CKPT="$(model_dir "round1" "$ITERS_R1")/${ITERS_R1}_ckpt.pt"

echo "=== Round 2 ==="
"${LAUNCH_CMD[@]}" depdog_train.py \
    --round=2 \
    --dataset="$DATASET" \
    --meta_file="$META_FILE" \
    --out_dir="$OUT_ROOT" \
    --canonical_file="$CANONICAL_FILE" \
    --n_layer="$N_LAYER" \
    --n_head="$N_HEAD" \
    --n_embd="$N_EMBD" \
    --batch_size="$BATCH_SIZE" \
    --gen_batch_size="$GEN_BATCH_SIZE" \
    --suffix="round2" \
    --max_iters="$ITERS_R2" \
    --prev_round_file="train_gen_round2.bin" \
    --prev_round_ckpt="$ROUND1_CKPT" \
    --mix_ratios="1.0,0.0,0.0" \
    --compile="$COMPILE"
run_eval "round2" "$ITERS_R2"

ROUND2_CKPT="$(model_dir "round2" "$ITERS_R2")/${ITERS_R2}_ckpt.pt"

echo "=== Round 3 ==="
"${LAUNCH_CMD[@]}" depdog_train.py \
    --round=3 \
    --dataset="$DATASET" \
    --meta_file="$META_FILE" \
    --out_dir="$OUT_ROOT" \
    --canonical_file="$CANONICAL_FILE" \
    --n_layer="$N_LAYER" \
    --n_head="$N_HEAD" \
    --n_embd="$N_EMBD" \
    --batch_size="$BATCH_SIZE" \
    --gen_batch_size="$GEN_BATCH_SIZE" \
    --suffix="round3" \
    --max_iters="$ITERS_R3" \
    --prev_round_file="train_gen_round3.bin" \
    --prev_files="train_gen_round2.bin" \
    --prev_round_ckpt="$ROUND2_CKPT" \
    --mix_ratios="$MIX_RATIOS_R3" \
    --compile="$COMPILE"
run_eval "round3" "$ITERS_R3"

echo "=== Dep-DOG pipeline example finished ==="
