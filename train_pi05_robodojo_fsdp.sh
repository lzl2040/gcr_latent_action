#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DATASET_ROOT="${DATASET_ROOT:-/data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/stack_bowls_delta}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/home/jianf/lzl_project/gcr_latent_action/outputs/evaluation/robodojo/runtime_assets/pretrained/pi05_base}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${PRETRAINED_PATH}/paligemma_tokenizer.model}"
JOB_NAME="${JOB_NAME:-pi05_robodojo_delta_fsdp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs/pi05_robodojo}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${JOB_NAME}}"
LOG_DIR="${LOG_DIR:-$PWD/outputs/pi05_robodojo_logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LR="${LR:-2.5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
SEED="${SEED:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
WEIGHT_RESUME="${WEIGHT_RESUME:-false}"
EXPORT_FINAL="${EXPORT_FINAL:-true}"
MAX_UPDATES_THIS_RUN="${MAX_UPDATES_THIS_RUN:-}"
CHECK_PATHS="${CHECK_PATHS:-true}"
DRY_RUN="${DRY_RUN:-false}"

die() { echo "Error: $*" >&2; exit 1; }
positive_int() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer, got $2"; }
nonnegative_int() { [[ "$2" =~ ^(0|[1-9][0-9]*)$ ]] || die "$1 must be a non-negative integer, got $2"; }
require_bool() { [[ "$2" == true || "$2" == false ]] || die "$1 must be true or false, got $2"; }
positive_int NNODES "$NNODES"
positive_int NPROC_PER_NODE "$NPROC_PER_NODE"
positive_int STEPS "$STEPS"
positive_int BATCH_SIZE "$BATCH_SIZE"
positive_int GRAD_ACCUMULATION_STEPS "$GRAD_ACCUMULATION_STEPS"
positive_int CHUNK_SIZE "$CHUNK_SIZE"
positive_int SAVE_FREQ "$SAVE_FREQ"
nonnegative_int NODE_RANK "$NODE_RANK"
nonnegative_int NUM_WORKERS "$NUM_WORKERS"
nonnegative_int WARMUP_STEPS "$WARMUP_STEPS"
nonnegative_int SEED "$SEED"
require_bool WEIGHT_RESUME "$WEIGHT_RESUME"
require_bool EXPORT_FINAL "$EXPORT_FINAL"
require_bool CHECK_PATHS "$CHECK_PATHS"
require_bool DRY_RUN "$DRY_RUN"
[[ -z "$MAX_UPDATES_THIS_RUN" ]] || positive_int MAX_UPDATES_THIS_RUN "$MAX_UPDATES_THIS_RUN"
(( NODE_RANK < NNODES )) || die "NODE_RANK must be smaller than NNODES"
[[ "$VIDEO_BACKEND" == pyav || "$VIDEO_BACKEND" == torchcodec ]] || die "Unsupported VIDEO_BACKEND"
command -v "$PYTHON_BIN" >/dev/null || die "Python executable not found: $PYTHON_BIN"
"$PYTHON_BIN" -c 'import math, sys; value = float(sys.argv[1]); sys.exit(0 if math.isfinite(value) and value > 0 else "LR must be finite and positive")' "$LR"
if (( NNODES == 1 )); then
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    MASTER_PORT="${MASTER_PORT:-$("$PYTHON_BIN" -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')}"
else
    [[ -n "$MASTER_ADDR" && -n "$MASTER_PORT" ]] || die "Multi-node runs require MASTER_ADDR and MASTER_PORT"
    [[ "$MASTER_ADDR" != localhost && "$MASTER_ADDR" != 127.0.0.1 ]] || die "MASTER_ADDR must be reachable from other nodes"
fi
positive_int MASTER_PORT "$MASTER_PORT"
(( MASTER_PORT <= 65535 )) || die "MASTER_PORT is outside 1..65535"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-60}"
positive_int DISTRIBUTED_TIMEOUT_MINUTES "$DISTRIBUTED_TIMEOUT_MINUTES"

if [[ "$CHECK_PATHS" == true ]]; then
    [[ -f "$DATASET_ROOT/meta/info.json" && -f "$DATASET_ROOT/meta/stats.json" ]] || die "Missing dataset metadata: $DATASET_ROOT"
    [[ -f "$PRETRAINED_PATH" || -f "$PRETRAINED_PATH/model.safetensors" ]] || die "Missing base weights: $PRETRAINED_PATH"
    [[ -f "$TOKENIZER_PATH" || -d "$TOKENIZER_PATH" ]] || die "TOKENIZER_PATH must be a local tokenizer file/directory"
    if [[ "$WEIGHT_RESUME" == true ]]; then
        [[ -f "$OUTPUT_DIR/latest_checkpoint" ]] || die "No latest_checkpoint in $OUTPUT_DIR"
    else
        [[ ! -e "$OUTPUT_DIR" ]] || die "OUTPUT_DIR already exists; choose a fresh directory or WEIGHT_RESUME=true"
    fi
fi

COMMAND=(
    "$PYTHON_BIN" -m torch.distributed.run
    --nnodes "$NNODES" --nproc-per-node "$NPROC_PER_NODE" --node-rank "$NODE_RANK"
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
    --module lerobot.scripts.train_pi05_robodojo_fsdp
    --dataset-root "$DATASET_ROOT" --pretrained-path "$PRETRAINED_PATH" --tokenizer-path "$TOKENIZER_PATH"
    --output-dir "$OUTPUT_DIR" --steps "$STEPS" --batch-size "$BATCH_SIZE"
    --grad-accumulation-steps "$GRAD_ACCUMULATION_STEPS" --chunk-size "$CHUNK_SIZE"
    --num-workers "$NUM_WORKERS" --save-freq "$SAVE_FREQ" --lr "$LR"
    --warmup-steps "$WARMUP_STEPS" --seed "$SEED" --video-backend "$VIDEO_BACKEND"
)
[[ "$WEIGHT_RESUME" == false ]] || COMMAND+=(--resume)
[[ "$EXPORT_FINAL" == false ]] || COMMAND+=(--export-final)
[[ -z "$MAX_UPDATES_THIS_RUN" ]] || COMMAND+=(--max-updates-this-run "$MAX_UPDATES_THIS_RUN")
COMMAND+=("$@")
printf 'Effective batch size: %s\n' "$((NNODES * NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUMULATION_STEPS))"
printf 'Command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
[[ "$DRY_RUN" == false ]] || exit 0
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
"${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
