#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/mnt/wangxiaofa/pt_weights}"
MODEL_DIR="${MODEL_DIR:-${WEIGHTS_ROOT}/Cosmos3-Nano}"
MODEL_AUTO_DOWNLOAD="${MODEL_AUTO_DOWNLOAD:-true}"
MODEL_LOCAL_CACHE="${MODEL_LOCAL_CACHE:-${TMPDIR:-/tmp}/cosmos3_policy_models}"
PARENT_DIR_V21="${PARENT_DIR_V21:-${DATA_ROOT_V21:-/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/}}"
PARENT_DIR_V30="${PARENT_DIR_V30:-${DATA_ROOT_V30:-/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/}}"
PARENT_DIR_EXTRA="${PARENT_DIR_EXTRA-${DATA_ROOT_EXTRA-/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/}}"
JOB_NAME="${JOB_NAME:-cosmos3_nano_fsdp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/wangxiaofa/qwen3vl_mot_exp}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${JOB_NAME}}"
LOG_DIR="${LOG_DIR:-/mnt/wangxiaofa/ace_logs}"
CHECK_PATHS="${CHECK_PATHS:-true}"
DRY_RUN="${DRY_RUN:-false}"
WEIGHT_RESUME="${WEIGHT_RESUME:-false}"
LOAD_ACTION_MODULES="${LOAD_ACTION_MODULES:-${WEIGHT_RESUME}}"

die() {
    echo "Error: $*" >&2
    exit 1
}

require_bool() {
    [[ "$2" == true || "$2" == false ]] || die "$1 must be true or false."
}

require_positive_int() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer."
}

require_bool WEIGHT_RESUME "$WEIGHT_RESUME"
require_bool LOAD_ACTION_MODULES "$LOAD_ACTION_MODULES"
require_bool MODEL_AUTO_DOWNLOAD "$MODEL_AUTO_DOWNLOAD"
require_bool CHECK_PATHS "$CHECK_PATHS"
require_bool DRY_RUN "$DRY_RUN"
if [[ "$WEIGHT_RESUME" == true && "$LOAD_ACTION_MODULES" != true ]]; then
    die "Resume must reload trained action modules; use LOAD_ACTION_MODULES=true."
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE="${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-64}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-60}"
if [[ -n "${FFMPEG_LIB_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="$FFMPEG_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#devices[@]}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-}"
require_positive_int NNODES "$NNODES"
require_positive_int NPROC_PER_NODE "$NPROC_PER_NODE"
require_positive_int DISTRIBUTED_TIMEOUT_MINUTES "$DISTRIBUTED_TIMEOUT_MINUTES"
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] || die "NODE_RANK must be a non-negative integer."
(( NODE_RANK < NNODES )) || die "NODE_RANK must be smaller than NNODES."
if (( NNODES == 1 )); then
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    MASTER_PORT="${MASTER_PORT:-$("$PYTHON_BIN" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')}"
else
    [[ -n "$MASTER_ADDR" && -n "$MASTER_PORT" ]] || die "MASTER_ADDR and MASTER_PORT are required for multi-node training."
    [[ "$MASTER_ADDR" != localhost && "$MASTER_ADDR" != 127.0.0.1 ]] || die "MASTER_ADDR must be reachable from other nodes."
fi
[[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] && (( MASTER_PORT <= 65535 )) || die "MASTER_PORT must be between 1 and 65535."
if [[ "$CHECK_PATHS" == true ]]; then
    for path in "$PARENT_DIR_V21" "$PARENT_DIR_V30"; do
        [[ -d "$path" ]] || die "Dataset root does not exist: $path"
    done
    [[ -z "$PARENT_DIR_EXTRA" || -d "$PARENT_DIR_EXTRA" ]] || die "Dataset extra root does not exist: $PARENT_DIR_EXTRA"
fi
if [[ "$DRY_RUN" != true ]]; then
    MODEL_DIR="$("$PYTHON_BIN" -m lerobot.common.utils.stage2_model_cache \
        --cosmos3-policy-dir="$MODEL_DIR" --cache-dir="$MODEL_LOCAL_CACHE" \
        --auto-download="$MODEL_AUTO_DOWNLOAD")"
fi

echo "Cosmos3 paths: model=$MODEL_DIR (missing weights: lerobot/Cosmos3-Nano)"
echo "  data_v21=$PARENT_DIR_V21 data_v30=$PARENT_DIR_V30 data_extra=${PARENT_DIR_EXTRA:-<disabled>}"
echo "  output=$OUTPUT_DIR logs=$LOG_DIR job=$JOB_NAME"
echo "  model_auto_download=$MODEL_AUTO_DOWNLOAD node_cache=$MODEL_LOCAL_CACHE resume=$WEIGHT_RESUME load_action_modules=$LOAD_ACTION_MODULES"
echo "distributed: nnodes=$NNODES node_rank=$NODE_RANK nproc_per_node=$NPROC_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"

CMD=(
    "$PYTHON_BIN" -m torch.distributed.run
    --nproc_per_node="$NPROC_PER_NODE"
    --nnodes="$NNODES"
    --node_rank="$NODE_RANK"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
    --rdzv_conf="timeout=$((DISTRIBUTED_TIMEOUT_MINUTES * 60))"
    --module lerobot.scripts.fsdp_train_contrast
    --policy.type="${POLICY_TYPE:-cosmos3_nano}"
    --policy.pretrained_path="$MODEL_DIR"
    --policy.dtype=bfloat16
    --policy.load_action_modules="$LOAD_ACTION_MODULES"
    --policy.gradient_checkpointing=true
    --policy.fps=20
    --policy.view_size='[256,256]'
    --policy.training_action_layout="${TRAINING_ACTION_LAYOUT:-libero_euler_interleaved}"
    --policy.optimizer_lr="${LEARNING_RATE:-5e-5}"
    --policy.scheduler_warmup_steps="${WARMUP_STEPS:-500}"
    --policy.scheduler_decay_steps="${DECAY_STEPS:-16000}"
    --stage=finetune
    --task_type=train_stage2
    --data_mix="${DATA_MIX:-libero}"
    --dataset.repo_id=local_libero
    --dataset.parent_dir_v21="$PARENT_DIR_V21"
    --dataset.parent_dir_v30="$PARENT_DIR_V30"
    --dataset.parent_dir_extra="$PARENT_DIR_EXTRA"
    --dataset.image_transforms.enable=false
    --dataset.image_transforms.img_size=256
    --dataset.video_backend="${VIDEO_BACKEND:-torchcodec}"
    --batch_size="${BATCH_SIZE:-2}"
    --num_workers="${NUM_WORKERS:-4}"
    --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-4}"
    --steps="${STEPS:-5000}"
    --save_freq="${SAVE_FREQ:-500}"
    --log_freq="${LOG_FREQ:-1}"
    --eval_freq=0
    --generation_eval.enabled=false
    --fsdp.fp8=false
    --weight_resume="$WEIGHT_RESUME"
    --output_dir="$OUTPUT_DIR"
    --log_dir="$LOG_DIR"
    --job_name="$JOB_NAME"
    --device=cuda
    --seed="${SEED:-1000}"
    --wandb.enable="${WANDB_ENABLE:-false}"
    --wandb.project="${WANDB_PROJECT:-lerobot-stage2}"
)
CMD+=("$@")
if [[ "$DRY_RUN" == true ]]; then
    printf 'command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi
"${CMD[@]}"
