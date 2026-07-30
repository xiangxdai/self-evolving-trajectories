#!/usr/bin/env bash
# Unified Ser-FOX launcher backed by configs/serfox_task_configs.py.
#
# Examples:
#   bash run_serfox.sh one_fg sat9 warmmix 0
#   BACKBONE=general bash run_serfox.sh one_fg cd4 warmmix 0
#   ROUNDS=2 bash run_serfox.sh one sudoku warmmix 0
#   bash run_serfox.sh all warmmix 0,1,2,3,4,5,6
#
# Useful env:
#   PY=python3                 Python executable
#   NPROC=2                    torchrun processes for one job; defaults to GPU count
#   ROUNDS=10                  training rounds from the task config
#   OUT_ROOT=out/table13_fox   explicit output root; unset uses trainer default
#   SKIP_LOSS_EVAL=1           pass --skip_loss_eval true
#   REGEN_MAX_BLOCKS=200000    pass --regen_max_blocks
#   BACKBONE=general           model size: small=per-task original, general=3-12-384
#   EFFECTIVE_GLOBAL_BATCH=512 fixed global batch target
#   TRAIN_BATCH_SIZE=256       optional per-rank micro-batch override
#   GRAD_ACCUM=2               optional global accumulation override
#   EXTRA_FLAGS="..."          appended verbatim to the trainer args
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

PY="${PY:-python3}"
ROUNDS="${ROUNDS:-10}"
OUT_ROOT="${OUT_ROOT:-}"
SKIP_LOSS_EVAL="${SKIP_LOSS_EVAL:-0}"
REGEN_MAX_BLOCKS="${REGEN_MAX_BLOCKS:-0}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
BACKBONE="${BACKBONE:-small}"
EFFECTIVE_GLOBAL_BATCH="${EFFECTIVE_GLOBAL_BATCH:-512}"
TRAINING_VARIANT="${TRAINING_VARIANT:-siwei_soft}"
SHUFFLE_SPECIAL_POLICY="${SHUFFLE_SPECIAL_POLICY:-exclude_special}"
MASTER_PORT="${MASTER_PORT:-29601}"
ALL_TASKS=(sat7 sat9 cd3 cd4 path10 path14 sudoku)

gpu_count() {
  local gpus="$1"
  if [[ -z "$gpus" ]]; then
    echo 1
  else
    awk -F',' '{print NF}' <<< "$gpus"
  fi
}

make_args() {
  local task="$1" regime="$2" out_dir="$3" nproc="$4"
  local micro_batch="${TRAIN_BATCH_SIZE:-}"
  local global_accum="${GRAD_ACCUM:-}"
  if [[ -z "$micro_batch" && -z "$global_accum" ]]; then
    (( EFFECTIVE_GLOBAL_BATCH % nproc == 0 )) || {
      echo "effective batch $EFFECTIVE_GLOBAL_BATCH is not divisible by nproc=$nproc" >&2
      return 2
    }
    micro_batch=$((EFFECTIVE_GLOBAL_BATCH / nproc))
    global_accum=$nproc
  elif [[ -z "$micro_batch" ]]; then
    (( EFFECTIVE_GLOBAL_BATCH % global_accum == 0 )) || {
      echo "effective batch $EFFECTIVE_GLOBAL_BATCH is not divisible by GRAD_ACCUM=$global_accum" >&2
      return 2
    }
    micro_batch=$((EFFECTIVE_GLOBAL_BATCH / global_accum))
  elif [[ -z "$global_accum" ]]; then
    (( EFFECTIVE_GLOBAL_BATCH % micro_batch == 0 )) || {
      echo "effective batch $EFFECTIVE_GLOBAL_BATCH is not divisible by TRAIN_BATCH_SIZE=$micro_batch" >&2
      return 2
    }
    global_accum=$((EFFECTIVE_GLOBAL_BATCH / micro_batch))
  fi
  (( micro_batch * global_accum == EFFECTIVE_GLOBAL_BATCH )) || {
    echo "TRAIN_BATCH_SIZE*GRAD_ACCUM must equal $EFFECTIVE_GLOBAL_BATCH" >&2
    return 2
  }
  (( global_accum % nproc == 0 )) || {
    echo "GRAD_ACCUM=$global_accum must be divisible by nproc=$nproc" >&2
    return 2
  }

  RUN_ARGS=(
    Ser-FOX/serfox_train.py
    --config "$task"
    --backbone "$BACKBONE"
    --regime "$regime"
    --rounds "$ROUNDS"
    --run_name "serfox_${task}_${BACKBONE}_${regime}_${ROUNDS}r"
    --training_variant "$TRAINING_VARIANT"
    --shuffle_order true
    --shuffle_special_policy "$SHUFFLE_SPECIAL_POLICY"
    --compile true
    --compile_parallel_tail true
    --compile_parallel_tail_dynamic false
    --compile_ddp_optimizer false
    --train_batch_size "$micro_batch"
    --gradient_accumulation_steps "$global_accum"
  )
  if [[ -n "$out_dir" ]]; then
    RUN_ARGS+=(--out_dir "$out_dir")
  fi
  if [[ "$SKIP_LOSS_EVAL" == "1" ]]; then
    RUN_ARGS+=(--skip_loss_eval true)
  fi
  if [[ "$REGEN_MAX_BLOCKS" != "0" ]]; then
    RUN_ARGS+=(--regen_max_blocks "$REGEN_MAX_BLOCKS")
  fi
  if [[ -n "$EXTRA_FLAGS" ]]; then
    # shellcheck disable=SC2206
    local extra=( $EXTRA_FLAGS )
    RUN_ARGS+=("${extra[@]}")
  fi
}

run_one_fg() {
  local task="$1" regime="$2" gpus="$3" port="${4:-$MASTER_PORT}"
  local nproc="${NPROC:-$(gpu_count "$gpus")}"
  local out_dir=""
  if [[ -n "$OUT_ROOT" ]]; then
    out_dir="$OUT_ROOT/${task}_${BACKBONE}_${regime}_${ROUNDS}r"
  fi
  local log="logs/serfox_${task}_${BACKBONE}_${regime}_${ROUNDS}r.log"
  make_args "$task" "$regime" "$out_dir" "$nproc"

  echo "[serfox] task=$task backbone=$BACKBONE regime=$regime rounds=$ROUNDS variant=$TRAINING_VARIANT special=$SHUFFLE_SPECIAL_POLICY eGB=$EFFECTIVE_GLOBAL_BATCH gpus=$gpus nproc=$nproc out=${out_dir:-trainer-default}"
  if [[ "$nproc" -gt 1 ]]; then
    printf -v RUN_COMMAND_ORIGINAL '%q ' "$PY" -m torch.distributed.run \
      --standalone --nproc_per_node="$nproc" --master_port="$port" \
      "${RUN_ARGS[@]}"
    export RUN_COMMAND_ORIGINAL
    CUDA_VISIBLE_DEVICES="$gpus" "$PY" -m torch.distributed.run \
      --standalone --nproc_per_node="$nproc" --master_port="$port" \
      "${RUN_ARGS[@]}" 2>&1 | tee "$log"
  else
    printf -v RUN_COMMAND_ORIGINAL '%q ' "$PY" "${RUN_ARGS[@]}"
    export RUN_COMMAND_ORIGINAL
    CUDA_VISIBLE_DEVICES="$gpus" "$PY" "${RUN_ARGS[@]}" 2>&1 | tee "$log"
  fi
}

run_one_bg() {
  local task="$1" regime="$2" gpus="$3" port="${4:-$MASTER_PORT}"
  nohup bash "$0" one_fg "$task" "$regime" "$gpus" "$port" \
    > "logs/serfox_${task}_${BACKBONE}_${regime}_${ROUNDS}r.launch.log" 2>&1 &
  echo "PID=$! log=logs/serfox_${task}_${BACKBONE}_${regime}_${ROUNDS}r.launch.log"
}

case "${1:-help}" in
  one_fg) run_one_fg "${2:?task}" "${3:?regime}" "${4:?gpus}" "${5:-$MASTER_PORT}" ;;
  one)    run_one_bg "${2:?task}" "${3:?regime}" "${4:?gpus}" "${5:-$MASTER_PORT}" ;;
  smoke)  ROUNDS=1 EXTRA_FLAGS="${EXTRA_FLAGS:-} --max_iters 200 --round_interval 100 --eval_interval 50 --checkpoint_interval 50" \
            run_one_fg "${2:?task}" "${3:?regime}" "${4:-0}" "${5:-$MASTER_PORT}" ;;
  all)
    regime="${2:?regime}"
    gpus_csv="${3:?gpu list, one per task: ${ALL_TASKS[*]}}"
    IFS=',' read -ra GPUS <<< "$gpus_csv"
    if [[ "${#GPUS[@]}" -ne "${#ALL_TASKS[@]}" ]]; then
      echo "need ${#ALL_TASKS[@]} GPUs, got ${#GPUS[@]}" >&2
      exit 1
    fi
    for i in "${!ALL_TASKS[@]}"; do
      run_one_bg "${ALL_TASKS[$i]}" "$regime" "${GPUS[$i]}" "$((MASTER_PORT + i))"
    done
    ;;
  help|*)
    cat <<'USAGE'
Usage:
  bash run_serfox.sh one_fg <task> <regime> <gpus> [port]
  bash run_serfox.sh one    <task> <regime> <gpus> [port]
  bash run_serfox.sh smoke  <task> <regime> [gpus] [port]
  bash run_serfox.sh all    <regime> <gpu_csv>

tasks:   sat7 sat9 cd3 cd4 path10 path14 sudoku
regimes: warm warmmix warmbest warmmixbest

Backbones:
  BACKBONE=small    per-task small/original backbone from configs/serfox_task_configs.py
  BACKBONE=general  shared 3-12-384 backbone
USAGE
    ;;
esac
