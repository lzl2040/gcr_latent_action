#!/usr/bin/env bash
set -euo pipefail

# Single- or multi-node H100 launcher for stage-two Qwen3-VL + Generation Expert training.
# FP8 accelerates eligible Linear GEMMs; trainable master parameters stay BF16 and
# bitsandbytes stores AdamW moments in 8-bit blocks.
#
# Cluster defaults follow train_ace.sh:
#   weights: /mnt/wangxiaofa/pt_weights
#   data:    /mnt/wangxiaofa/robot_dataset
#   output:  /mnt/wangxiaofa/qwen3vl_mot_exp
#
# Override WEIGHTS_ROOT/PARENT_DIR_*/OUTPUT_DIR for a local /Data installation.
cd "$(dirname "$0")" || exit 1

WEIGHTS_ROOT="${WEIGHTS_ROOT:-/mnt/wangxiaofa/pt_weights}"
QWEN3VL_DIR="${QWEN3VL_DIR:-${WEIGHTS_ROOT}/Qwen3-VL-4B-Instruct}"
COSMOS3_DIR="${COSMOS3_DIR:-${WEIGHTS_ROOT}/Cosmos3-Edge}"
PARENT_DIR_V21="${PARENT_DIR_V21:-${DATA_ROOT_V21:-/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/}}"
PARENT_DIR_V30="${PARENT_DIR_V30:-${DATA_ROOT_V30:-/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/}}"
PARENT_DIR_EXTRA="${PARENT_DIR_EXTRA-${DATA_ROOT_EXTRA-/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/}}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/mnt/wangxiaofa/ace_stage1/step_16k}"
STAGE1_CONFIG="${STAGE1_CONFIG:-}"
JOB_NAME="${JOB_NAME:-qwen3vl_mot_fsdp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/wangxiaofa/qwen3vl_mot_exp}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${JOB_NAME}}"
LOG_DIR="${LOG_DIR:-/mnt/wangxiaofa/ace_logs}"
CHECK_PATHS="${CHECK_PATHS:-true}"
DRY_RUN="${DRY_RUN:-false}"

WEIGHT_RESUME="${WEIGHT_RESUME:-true}"

die() {
    echo "Error: $*" >&2
    exit 1
}

require_bool() {
    [[ "$2" == "true" || "$2" == "false" ]] || die "$1 must be true or false, got $2"
}

require_positive_int() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer, got $2"
}

require_nonnegative_int() {
    [[ "$2" =~ ^[0-9]+$ ]] || die "$1 must be a non-negative integer, got $2"
}

require_dir() {
    [[ -d "$2" ]] || die "$1 directory does not exist: $2"
}

require_file() {
    [[ -f "$2" ]] || die "$1 file does not exist: $2"
}

require_path() {
    [[ -e "$2" ]] || die "$1 does not exist: $2"
}

require_bool "CHECK_PATHS" "$CHECK_PATHS"
require_bool "DRY_RUN" "$DRY_RUN"
require_bool "WEIGHT_RESUME" "$WEIGHT_RESUME"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE="${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-256}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

prepare_torchao_environment() {
    local torch_series
    torch_series="$("${PYTHON_BIN}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    release = version("torch").split("+", 1)[0].split(".")
except PackageNotFoundError:
    print("")
else:
    print(".".join(release[:2]))
PY
)"
    if [[ "$torch_series" == "2.7" ]]; then
        # torchao 0.15's C++ wheel targets torch 2.9.1. Stage 2 only needs its
        # Python FP8 wrappers, which dispatch to torch 2.7's native _scaled_mm.
        export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
    fi
}

check_python_dependencies() {
    "${PYTHON_BIN}" -m lerobot.common.utils.stage2_dependencies
}

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-}"
require_positive_int "NNODES" "$NNODES"
require_positive_int "NPROC_PER_NODE" "$NPROC_PER_NODE"
require_nonnegative_int "NODE_RANK" "$NODE_RANK"
(( NODE_RANK < NNODES )) || die "NODE_RANK=$NODE_RANK must be smaller than NNODES=$NNODES"
if (( NNODES == 1 )); then
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    MASTER_PORT="${MASTER_PORT:-$("${PYTHON_BIN}" -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")}"
else
    [[ -n "$MASTER_ADDR" ]] || die "MASTER_ADDR is required when NNODES > 1"
    [[ -n "$MASTER_PORT" ]] || die "MASTER_PORT is required when NNODES > 1"
    if [[ "$MASTER_ADDR" == "127.0.0.1" || "$MASTER_ADDR" == "localhost" ]]; then
        die "MASTER_ADDR=$MASTER_ADDR is not reachable from other nodes"
    fi
fi
require_positive_int "MASTER_PORT" "$MASTER_PORT"
(( MASTER_PORT <= 65535 )) || die "MASTER_PORT=$MASTER_PORT is outside the valid port range"
command -v "$TORCHRUN_BIN" >/dev/null 2>&1 \
    || die "torchrun executable was not found: $TORCHRUN_BIN"
if [[ "$DRY_RUN" != "true" ]]; then
    prepare_torchao_environment
    check_python_dependencies
fi

LATEST_CHECKPOINT_POINTER="${OUTPUT_DIR}/latest_checkpoint"
RESUME_CHECKPOINT=""
if [[ "$WEIGHT_RESUME" == "true" ]]; then
    if [[ ! -e "$LATEST_CHECKPOINT_POINTER" ]]; then
        shopt -s nullglob
        _orphan_checkpoints=("$OUTPUT_DIR"/checkpoint_*)
        shopt -u nullglob
        if (( ${#_orphan_checkpoints[@]} > 0 )); then
            die "found checkpoint artifacts but no latest_checkpoint pointer in $OUTPUT_DIR"
        fi
        echo "resume: no ${LATEST_CHECKPOINT_POINTER}; starting a fresh Stage 2 run"
        WEIGHT_RESUME=false
    elif [[ ! -f "$LATEST_CHECKPOINT_POINTER" ]]; then
        die "resume pointer is not a regular file: $LATEST_CHECKPOINT_POINTER"
    else
        CHECKPOINT_NAME="$(<"$LATEST_CHECKPOINT_POINTER")"
        [[ "$CHECKPOINT_NAME" =~ ^checkpoint_[0-9]{8}$ ]] \
            || die "invalid checkpoint name in $LATEST_CHECKPOINT_POINTER: $CHECKPOINT_NAME"
        RESUME_CHECKPOINT="${OUTPUT_DIR}/${CHECKPOINT_NAME}"
        require_dir "latest FSDP checkpoint" "$RESUME_CHECKPOINT"
        echo "resume: found ${RESUME_CHECKPOINT}"
    fi
else
    echo "resume: disabled explicitly; starting a fresh Stage 2 run"
fi
if [[ "$WEIGHT_RESUME" != "true" && -z "$STAGE1_CHECKPOINT" ]]; then
    die "STAGE1_CHECKPOINT is required for a fresh stage-two run"
fi

# PyTorch wheels may ship a newer CUDA runtime than /usr/local/cuda. NVRTC loads its
# builtins dynamically, so put the matching wheel library first when it is available.
PACKAGED_CUDA_LIB="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import site
import torch

cuda_major = (torch.version.cuda or "").split(".", 1)[0]
for root in site.getsitepackages():
    candidate = Path(root) / "nvidia" / f"cu{cuda_major}" / "lib"
    if candidate.is_dir():
        print(candidate)
        break
PY
)"
if [[ -n "${PACKAGED_CUDA_LIB}" ]]; then
    export LD_LIBRARY_PATH="${PACKAGED_CUDA_LIB}:${LD_LIBRARY_PATH:-}"
fi

if [[ "$CHECK_PATHS" == "true" ]]; then
    require_dir "Qwen3-VL model" "$QWEN3VL_DIR"
    require_file "Qwen3-VL config" "$QWEN3VL_DIR/config.json"
    shopt -s nullglob
    _qwen_shards=("$QWEN3VL_DIR"/*.safetensors)
    shopt -u nullglob
    (( ${#_qwen_shards[@]} > 0 )) \
        || die "Qwen3-VL directory has no *.safetensors: $QWEN3VL_DIR"

    require_dir "Cosmos3-Edge model" "$COSMOS3_DIR"
    require_file "Cosmos3 VAE config" "$COSMOS3_DIR/vae/config.json"
    require_dir "dataset parent_dir_v21" "$PARENT_DIR_V21"
    require_dir "dataset parent_dir_v30" "$PARENT_DIR_V30"
    if [[ -n "$PARENT_DIR_EXTRA" ]]; then
        require_dir "dataset parent_dir_extra" "$PARENT_DIR_EXTRA"
    fi
    if [[ "$WEIGHT_RESUME" != "true" ]]; then
        require_path "Stage 1 checkpoint" "$STAGE1_CHECKPOINT"
        if [[ -n "$STAGE1_CONFIG" ]]; then
            require_file "Stage 1 config" "$STAGE1_CONFIG"
        fi
    fi
fi

STAGE1_CONFIG_ARGS=()
STAGE1_CHECKPOINT_ARGS=()
if [[ "$WEIGHT_RESUME" != "true" && -n "$STAGE1_CHECKPOINT" ]]; then
    STAGE1_CHECKPOINT_ARGS+=(--policy.stage1_checkpoint="${STAGE1_CHECKPOINT}")
fi
if [[ "$WEIGHT_RESUME" != "true" && -n "$STAGE1_CONFIG" ]]; then
    STAGE1_CONFIG_ARGS+=(--policy.stage1_config="${STAGE1_CONFIG}")
fi

echo "stage2 paths:"
echo "  qwen3vl=${QWEN3VL_DIR}"
echo "  cosmos3=${COSMOS3_DIR}"
if [[ "$WEIGHT_RESUME" != "true" ]]; then
    echo "  stage1=${STAGE1_CHECKPOINT}"
fi
echo "  data_v21=${PARENT_DIR_V21}"
echo "  data_v30=${PARENT_DIR_V30}"
echo "  data_extra=${PARENT_DIR_EXTRA:-<disabled>}"
echo "  output=${OUTPUT_DIR}"
echo "  logs=${LOG_DIR}"
echo "  resume=${WEIGHT_RESUME}${RESUME_CHECKPOINT:+ (${RESUME_CHECKPOINT})}"
echo "distributed: nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"

TORCHRUN_ARGS=()
if (( NNODES == 1 )); then
    TORCHRUN_ARGS+=(
        --standalone
        --nproc_per_node="${NPROC_PER_NODE}"
        --master_port="${MASTER_PORT}"
    )
else
    TORCHRUN_ARGS+=(
        --nnodes="${NNODES}"
        --nproc_per_node="${NPROC_PER_NODE}"
        --node_rank="${NODE_RANK}"
        --master_addr="${MASTER_ADDR}"
        --master_port="${MASTER_PORT}"
    )
fi

CMD=(
    "${TORCHRUN_BIN}"
    "${TORCHRUN_ARGS[@]}"
    lerobot/scripts/fsdp_train_contrast.py
    --policy.type="qwen3vl_mot"
    --policy.qwen3vl_dir="${QWEN3VL_DIR}"
    --policy.cosmos3_dir="${COSMOS3_DIR}"
    --policy.understanding_tuning_mode="${UNDERSTANDING_TUNING_MODE:-full}"
    --policy.understanding_lora_rank="${UNDERSTANDING_LORA_RANK:-16}"
    --policy.understanding_lora_alpha="${UNDERSTANDING_LORA_ALPHA:-16}"
    --policy.understanding_text_lora_layers="${UNDERSTANDING_TEXT_LORA_LAYERS:-0}"
    --policy.understanding_vision_lora_layers="${UNDERSTANDING_VISION_LORA_LAYERS:-4}"
    --policy.understanding_lr_scale="${UNDERSTANDING_LR_SCALE:-0.05}"
    --policy.physical_tuning_mode="${PHYSICAL_TUNING_MODE:-frozen}"
    --policy.window_mode="frames"
    --policy.chunk_size=32
    --policy.n_action_steps=32
    --policy.world_video_frames="${WORLD_VIDEO_FRAMES:-9}"
    --policy.generation_gradient_checkpointing=true
    --policy.understanding_gradient_checkpointing=true
    --policy.optimizer_lr="${LEARNING_RATE:-1e-4}"
    --policy.scheduler_warmup_steps="${WARMUP_STEPS:-500}"
    --policy.scheduler_plateau_steps="${PLATEAU_STEPS:-2000}"
    --policy.scheduler_decay_steps="${DECAY_STEPS:-100000}"
    --policy.scheduler_decay_lr="${DECAY_LR:-1.5e-6}"
    --dataset.repo_id="whatever"
    --dataset.image_transforms.enable=false
    --dataset.image_transforms.img_size=256
    --dataset.wrist_image_transforms.enable=false
    --dataset.wrist_image_transforms.is_primary=false
    --dataset.parent_dir_v21="${PARENT_DIR_V21}"
    --dataset.parent_dir_v30="${PARENT_DIR_V30}"
    --dataset.parent_dir_extra="${PARENT_DIR_EXTRA}"
    --dataset.video_backend="${VIDEO_BACKEND:-torchcodec}"
    --dataset.dataset_size_one_epoch="${SAMPLES_PER_EPOCH:-100000}"
    --data_mix="${DATA_MIX:-debug_research_data}"
    --num_workers="${NUM_WORKERS:-8}"
    --batch_size="${BATCH_SIZE:-16}"
    --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
    --fsdp.fp8="${FP8_ENABLED:-true}"
    --fsdp.fp8_recipe="${FP8_RECIPE:-rowwise_with_gw_hp}"
    --fsdp.fp8_scope="${FP8_SCOPE:-generation_vlm}"
    --fsdp.fp8_emulate="${FP8_EMULATE:-false}"
    --fsdp.fp8_min_features="${FP8_MIN_FEATURES:-128}"
    --fsdp.min_wrap_params="${FSDP_MIN_WRAP_PARAMS:-1000000}"
    --fsdp.forward_prefetch="${FSDP_FORWARD_PREFETCH:-false}"
    --fsdp.limit_all_gathers="${FSDP_LIMIT_ALL_GATHERS:-true}"
    --fsdp.replicate_frozen_params="${FSDP_REPLICATE_FROZEN:-true}"
    --output_dir="${OUTPUT_DIR}"
    --log_dir="${LOG_DIR}"
    --job_name="${JOB_NAME}"
    --weight_resume="${WEIGHT_RESUME}"
    --save_freq="${SAVE_FREQ:-2000}"
    --log_freq="${LOG_FREQ:-20}"
    --eval_freq=0
    --steps="${STEPS:-600000}"
    --task_type="train_stage2"
    --wandb.enable="${WANDB_ENABLE:-false}"
    --wandb.project="${WANDB_PROJECT:-lerobot-stage2}"
)
CMD+=("${STAGE1_CHECKPOINT_ARGS[@]}")
CMD+=("${STAGE1_CONFIG_ARGS[@]}")
CMD+=("$@")

if [[ "$DRY_RUN" == "true" ]]; then
    printf 'command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}"
