#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PARENT_DIR_V30="${PARENT_DIR_V30:-${DATA_ROOT_V30:-/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/}}"
DATASET_ROOT="${DATASET_ROOT:-${PARENT_DIR_V30%/}/RoboDojo/data/stack_bowls_delta}"
MODEL_AUTO_DOWNLOAD="${MODEL_AUTO_DOWNLOAD:-true}"
MODEL_LOCAL_CACHE="${MODEL_LOCAL_CACHE:-/scratch/pi05_models}"
PRETRAINED_PATH="${PRETRAINED_PATH:-${MODEL_LOCAL_CACHE}/pi05_base}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${PRETRAINED_PATH}/paligemma_tokenizer.model}"
JOB_NAME="${JOB_NAME:-pi05_robodojo_delta_fsdp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/wangxiaofa/qwen3vl_mot_exp}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${JOB_NAME}}"
LOG_DIR="${LOG_DIR:-/mnt/wangxiaofa/ace_logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
FSDP_WRAP_STRATEGY="${FSDP_WRAP_STRATEGY:-layer}"
MASTER_DTYPE="${MASTER_DTYPE:-float32}"
OPTIMIZER="${OPTIMIZER:-adamw}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-eager}"
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
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot-pi05}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$JOB_NAME}"
WANDB_MODE="${WANDB_MODE:-online}"

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
require_bool MODEL_AUTO_DOWNLOAD "$MODEL_AUTO_DOWNLOAD"
require_bool WANDB_ENABLE "$WANDB_ENABLE"
if [[ "$WANDB_ENABLE" == true ]]; then
    [[ "$WANDB_MODE" == online || "$WANDB_MODE" == offline ]] || die "WANDB_MODE must be online or offline"
fi
[[ -z "$MAX_UPDATES_THIS_RUN" ]] || positive_int MAX_UPDATES_THIS_RUN "$MAX_UPDATES_THIS_RUN"
(( NODE_RANK < NNODES )) || die "NODE_RANK must be smaller than NNODES"
[[ "$VIDEO_BACKEND" == pyav || "$VIDEO_BACKEND" == torchcodec ]] || die "Unsupported VIDEO_BACKEND"
[[ "$FSDP_WRAP_STRATEGY" == layer || "$FSDP_WRAP_STRATEGY" == root ]] || die "FSDP_WRAP_STRATEGY must be layer or root"
[[ "$MASTER_DTYPE" == float32 || "$MASTER_DTYPE" == bfloat16 ]] || die "MASTER_DTYPE must be float32 or bfloat16"
[[ "$OPTIMIZER" == adamw || "$OPTIMIZER" == adamw8bit ]] || die "OPTIMIZER must be adamw or adamw8bit"
[[ "$ATTENTION_BACKEND" == eager || "$ATTENTION_BACKEND" == sdpa ]] || die "ATTENTION_BACKEND must be eager or sdpa"
if [[ "$FSDP_WRAP_STRATEGY" == root ]]; then
    [[ "$MASTER_DTYPE" == float32 && "$OPTIMIZER" == adamw && "$ATTENTION_BACKEND" == eager ]] || die "Memory-optimized modes require FSDP_WRAP_STRATEGY=layer"
fi
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
    if [[ "$WEIGHT_RESUME" == true ]]; then
        if [[ ! -e "$OUTPUT_DIR/latest_checkpoint" && ! -L "$OUTPUT_DIR/latest_checkpoint" ]]; then
            echo "Warning: No latest_checkpoint in $OUTPUT_DIR; starting fresh (existing output will be backed up)." >&2
        fi
    else
        [[ ! -e "$OUTPUT_DIR" ]] || die "OUTPUT_DIR already exists; choose a fresh directory or WEIGHT_RESUME=true"
    fi
    if [[ "$DRY_RUN" == false && ( "$MODEL_AUTO_DOWNLOAD" == true || "$CHECK_PATHS" == true ) ]]; then
        RESOLVED_ASSETS="$("$PYTHON_BIN" -m lerobot.common.utils.pi05_model_cache \
            --pretrained-path "$PRETRAINED_PATH" --tokenizer-path "$TOKENIZER_PATH" \
            --cache-dir "$MODEL_LOCAL_CACHE" --auto-download "$MODEL_AUTO_DOWNLOAD")"
        mapfile -t ASSET_PATHS <<< "$RESOLVED_ASSETS"
        (( ${#ASSET_PATHS[@]} == 2 )) || die "Expected resolved weight and tokenizer paths"
        PRETRAINED_PATH="${ASSET_PATHS[0]}"
        TOKENIZER_PATH="${ASSET_PATHS[1]}"
    elif [[ "$DRY_RUN" == true && "$CHECK_PATHS" == true && "$MODEL_AUTO_DOWNLOAD" == false ]]; then
        [[ -f "$PRETRAINED_PATH" || -f "$PRETRAINED_PATH/model.safetensors" ]] || die "Missing base weights: $PRETRAINED_PATH"
        [[ -f "$TOKENIZER_PATH" || -d "$TOKENIZER_PATH" ]] || die "TOKENIZER_PATH must be a local tokenizer file/directory"
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
    --fsdp-wrap-strategy "$FSDP_WRAP_STRATEGY"
    --master-dtype "$MASTER_DTYPE" --optimizer "$OPTIMIZER" --attention-backend "$ATTENTION_BACKEND"
    --num-workers "$NUM_WORKERS" --save-freq "$SAVE_FREQ" --lr "$LR"
    --warmup-steps "$WARMUP_STEPS" --seed "$SEED" --video-backend "$VIDEO_BACKEND"
)
[[ "$WEIGHT_RESUME" == false ]] || COMMAND+=(--resume)
[[ "$EXPORT_FINAL" == false ]] || COMMAND+=(--export-final)
[[ -z "$MAX_UPDATES_THIS_RUN" ]] || COMMAND+=(--max-updates-this-run "$MAX_UPDATES_THIS_RUN")
if [[ "$WANDB_ENABLE" == true ]]; then
    COMMAND+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME" --wandb-mode "$WANDB_MODE")
    [[ -z "$WANDB_ENTITY" ]] || COMMAND+=(--wandb-entity "$WANDB_ENTITY")
fi
COMMAND+=("$@")
printf 'Effective batch size: %s\n' "$((NNODES * NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUMULATION_STEPS))"
printf 'Pretrained assets: auto_download=%s node_cache=%s\n' "$MODEL_AUTO_DOWNLOAD" "$MODEL_LOCAL_CACHE"
printf 'Command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
[[ "$DRY_RUN" == false ]] || exit 0
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
"${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
