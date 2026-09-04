#!/usr/bin/env bash
# Phi-4-Multimodal MoT 世界模型集群训练入口。
#
# 单机 8 卡，默认 gen_only / stage3 / 每卡 batch 32：
#   conda activate lerobot_v2
#   bash train_mot.sh --job_name mot_stage3 --nproc_per_node 8
#
# 2 节点 16 卡（每个节点各执行一次，node_rank 分别为 0/1）：
#   bash train_mot.sh --job_name mot_stage3 --nnodes 2 --nproc_per_node 8 \
#     --node_rank 0 --master_addr 10.0.0.1 --master_port 29500
#
# 只冻结理解侧 vision encoder、训练其余部分：
#   bash train_mot.sh --job_name mot_freeze_vision --scope freeze_vision
#
# `--` 后的参数原样传给 dps_train_mot.py：
#   bash train_mot.sh --job_name mot_stage3 -- --seed=42
set -euo pipefail

cd "$(dirname "$0")" || exit 1

# ---------------------------------------------------------------- distributed
NNODES=1
NPROC_PER_NODE=8
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500

# ---------------------------------------------------------------- run
JOB_NAME=""
DATA_MIX="debug_research_data"
TASK_MIX="stage3"
SCOPE="gen_only"
EXECUTION="interleaved"
STEPS=600000
SAVE_FREQ=2000
LOG_FREQ=20
NUM_WORKERS=12
DATASET_SIZE_ONE_EPOCH=1000000
BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1
RESUME=true
RESUME_DIR=""

# ---------------------------------------------------------------- model / optimizer
LATENT_FRAMES=3
TEXT_LEN=32
CKPT_SEGMENT=4
MOT_MICROBATCH=""
GRAD_CHECKPOINTING=1
FREEZE_VISION_PROJECTOR=0
ACTION_LOSS_WEIGHT=1.0
LEARNING_RATE=1e-4
MIN_LEARNING_RATE=1e-5
WARMUP_STEPS=500
WEIGHT_DECAY=0.01

# ---------------------------------------------------------------- temporal loader
WINDOW_MODE="duration"
CHUNK_SIZE=32
GROUP_SIZE=4
CHUNK_SECONDS=1.6
SAMPLE_RATIO=5

# ---------------------------------------------------------------- mounted paths
PARENT_DIR_V21="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/"
PARENT_DIR_V30="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/"
PARENT_DIR_EXTRA="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/"
PROCESSOR="/mnt/wangxiaofa/pt_weights/InternVL3_5-2B-HF/"
PHI_DIR="/mnt/wangxiaofa/pt_weights/Phi-4-multimodal-instruct"
VAE_DIR="/mnt/wangxiaofa/pt_weights/Cosmos3-Edge/vae"
LOG_DIR="/mnt/wangxiaofa/ace_logs"
OUTPUT_DIR="/mnt/wangxiaofa/phi4_mot_exp"
DEEPSPEED_CONFIG="./ds_zero2_contrast.json"

# ---------------------------------------------------------------- logging
WANDB_ENABLE=true
WANDB_PROJECT="phi4_mot"
DRY_RUN=false
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash train_mot.sh --job_name NAME [options] [-- extra draccus args]

Core:
  --scope gen_only|freeze_vision|all
  --task_mix stage2|stage3|stage3_joint_only|action_only|TASK=WEIGHT,...
  --batch_size N                 Per-GPU batch (default: 32)
  --gradient_acc N               DeepSpeed gradient accumulation (default: 1)
  --steps N                      Number of micro-steps (default: 600000)
  --resume true|false
  --resume_dir PATH              DeepSpeed checkpoint directory; defaults to this run
                                  Batch size, accumulation and world size must match checkpoint

Distributed:
  --nnodes N --nproc_per_node N --node_rank N
  --master_addr HOST --master_port PORT

Paths:
  --phi_dir PATH --vae_dir PATH
  --parent_dir_v21 PATH --parent_dir_v30 PATH --parent_dir_extra PATH
  --output_dir PATH --log_dir PATH

Use --dry_run to print the resolved torchrun command without loading data or model weights.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes) NNODES="$2"; shift 2 ;;
        --nproc_per_node) NPROC_PER_NODE="$2"; shift 2 ;;
        --node_rank) NODE_RANK="$2"; shift 2 ;;
        --master_addr) MASTER_ADDR="$2"; shift 2 ;;
        --master_port) MASTER_PORT="$2"; shift 2 ;;
        --job_name) JOB_NAME="$2"; shift 2 ;;
        --data_mix) DATA_MIX="$2"; shift 2 ;;
        --task_mix) TASK_MIX="$2"; shift 2 ;;
        --scope) SCOPE="$2"; shift 2 ;;
        --execution) EXECUTION="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --save_freq) SAVE_FREQ="$2"; shift 2 ;;
        --log_freq) LOG_FREQ="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --dataset_len) DATASET_SIZE_ONE_EPOCH="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --gradient_acc) GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        --resume_dir) RESUME_DIR="$2"; shift 2 ;;
        --latent_frames) LATENT_FRAMES="$2"; shift 2 ;;
        --text_len) TEXT_LEN="$2"; shift 2 ;;
        --checkpoint_segment) CKPT_SEGMENT="$2"; shift 2 ;;
        --mot_microbatch) MOT_MICROBATCH="$2"; shift 2 ;;
        --grad_checkpointing) GRAD_CHECKPOINTING="$2"; shift 2 ;;
        --freeze_vision_projector) FREEZE_VISION_PROJECTOR="$2"; shift 2 ;;
        --action_loss_weight) ACTION_LOSS_WEIGHT="$2"; shift 2 ;;
        --optimizer_lr) LEARNING_RATE="$2"; shift 2 ;;
        --min_lr) MIN_LEARNING_RATE="$2"; shift 2 ;;
        --warmup_steps) WARMUP_STEPS="$2"; shift 2 ;;
        --weight_decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --window_mode) WINDOW_MODE="$2"; shift 2 ;;
        --chunk_size) CHUNK_SIZE="$2"; shift 2 ;;
        --group_size) GROUP_SIZE="$2"; shift 2 ;;
        --chunk_seconds) CHUNK_SECONDS="$2"; shift 2 ;;
        --sample_ratio) SAMPLE_RATIO="$2"; shift 2 ;;
        --parent_dir_v21) PARENT_DIR_V21="$2"; shift 2 ;;
        --parent_dir_v30) PARENT_DIR_V30="$2"; shift 2 ;;
        --parent_dir_extra) PARENT_DIR_EXTRA="$2"; shift 2 ;;
        --processor) PROCESSOR="$2"; shift 2 ;;
        --phi_dir) PHI_DIR="$2"; shift 2 ;;
        --vae_dir) VAE_DIR="$2"; shift 2 ;;
        --log_dir) LOG_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --deepspeed_config) DEEPSPEED_CONFIG="$2"; shift 2 ;;
        --wandb_enable) WANDB_ENABLE="$2"; shift 2 ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --dry_run) DRY_RUN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "未知参数: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$JOB_NAME" ]]; then
    echo "错误：必须指定 --job_name"
    exit 1
fi
if [[ "$SCOPE" != "gen_only" && "$SCOPE" != "freeze_vision" && "$SCOPE" != "all" ]]; then
    echo "错误：--scope 必须是 gen_only、freeze_vision 或 all"
    exit 1
fi
if [[ "$EXECUTION" != "interleaved" && "$EXECUTION" != "cached" ]]; then
    echo "错误：--execution 必须是 interleaved 或 cached"
    exit 1
fi
if (( STEPS % GRADIENT_ACCUMULATION_STEPS != 0 )); then
    echo "错误：--steps 必须能被 --gradient_acc 整除"
    exit 1
fi
if [[ -z "$MOT_MICROBATCH" ]]; then
    if [[ "$SCOPE" == "gen_only" ]]; then
        MOT_MICROBATCH=32
    else
        MOT_MICROBATCH=16
    fi
fi

if [[ "$DRY_RUN" != "true" ]]; then
    for required in \
        "$DEEPSPEED_CONFIG" \
        "$PHI_DIR/config.json" \
        "$VAE_DIR/config.json" \
        "./vla2root.json"; do
        if [[ ! -f "$required" ]]; then
            echo "错误：缺少训练所需文件: $required"
            exit 1
        fi
    done
fi

# W&B key is injected at runtime; never place it in this tracked script.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    for candidate in "${WANDB_KEY_FILE:-}" "${HOME}/.wandb_key" "./wandb.key"; do
        if [[ -n "$candidate" && -f "$candidate" ]]; then
            WANDB_API_KEY="$(tr -d '[:space:]' < "$candidate")"
            echo "wandb key <- ${candidate}"
            break
        fi
    done
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"
if [[ "$WANDB_ENABLE" == "true" && -z "$WANDB_API_KEY" && "$DRY_RUN" != "true" ]]; then
    echo "错误：W&B 已开启但未找到 API key；设置 WANDB_API_KEY 或使用 --wandb_enable false"
    exit 1
fi

SAFE_JOB_NAME="${JOB_NAME//[^a-zA-Z0-9_.-]/_}"
RUNTIME_DS_CONFIG="$(mktemp "/tmp/mot_zero2_${SAFE_JOB_NAME}_${NODE_RANK}_XXXXXX.json")"
cleanup() {
    rm -f "$RUNTIME_DS_CONFIG"
}
trap cleanup EXIT

python - "$DEEPSPEED_CONFIG" "$RUNTIME_DS_CONFIG" "$BATCH_SIZE" \
    "$GRADIENT_ACCUMULATION_STEPS" <<'PY'
import json
import sys

source, target, batch, accumulation = sys.argv[1:]
with open(source) as stream:
    config = json.load(stream)
config["train_micro_batch_size_per_gpu"] = int(batch)
config["gradient_accumulation_steps"] = int(accumulation)
config["gradient_clipping"] = 1.0
with open(target, "w") as stream:
    json.dump(config, stream, indent=2)
PY

export PHI_DIR VAE_DIR
export SCOPE
export MIX="$TASK_MIX"
export EXECUTION CKPT_SEGMENT MOT_MICROBATCH GRAD_CHECKPOINTING
export FREEZE_VISION_PROJECTOR ACTION_LOSS_WEIGHT
export LATENT_FRAMES TEXT_LEN
export LEARNING_RATE MIN_LEARNING_RATE WARMUP_STEPS WEIGHT_DECAY
export RESUME RESUME_DIR
export TOKENIZERS_PARALLELISM=false
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=256
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-${NCCL_ASYNC_ERROR_HANDLING:-1}}"
unset PYTORCH_CUDA_ALLOC_CONF NCCL_ASYNC_ERROR_HANDLING

COMMAND=(
    torchrun
    "--nnodes=${NNODES}"
    "--nproc_per_node=${NPROC_PER_NODE}"
    "--node_rank=${NODE_RANK}"
    "--master_addr=${MASTER_ADDR}"
    "--master_port=${MASTER_PORT}"
    lerobot/scripts/dps_train_mot.py
    "--deepspeed=${RUNTIME_DS_CONFIG}"
    "--policy.type=robo_contrast"
    "--policy.window_mode=${WINDOW_MODE}"
    "--policy.chunk_size=${CHUNK_SIZE}"
    "--policy.group_size=${GROUP_SIZE}"
    "--policy.chunk_seconds=${CHUNK_SECONDS}"
    "--dataset.repo_id=whatever"
    "--dataset.image_transforms.enable=false"
    "--dataset.wrist_image_transforms.enable=false"
    "--dataset.wrist_image_transforms.is_primary=false"
    "--dataset.processor=${PROCESSOR}"
    "--dataset.parent_dir_v21=${PARENT_DIR_V21}"
    "--dataset.parent_dir_v30=${PARENT_DIR_V30}"
    "--dataset.parent_dir_extra=${PARENT_DIR_EXTRA}"
    "--dataset.video_backend=torchcodec"
    "--dataset.sample_ratio=${SAMPLE_RATIO}"
    "--dataset.dataset_size_one_epoch=${DATASET_SIZE_ONE_EPOCH}"
    "--data_mix=${DATA_MIX}"
    "--num_workers=${NUM_WORKERS}"
    "--batch_size=${BATCH_SIZE}"
    "--steps=${STEPS}"
    "--save_freq=${SAVE_FREQ}"
    "--log_freq=${LOG_FREQ}"
    "--output_dir=${OUTPUT_DIR}"
    "--log_dir=${LOG_DIR}"
    "--task_type=train_mot"
    "--wandb.enable=${WANDB_ENABLE}"
    "--wandb.project=${WANDB_PROJECT}"
    "--job_name=${JOB_NAME}"
    "--weight_resume=${RESUME}"
    "--resume=false"
    "${EXTRA_ARGS[@]}"
)

echo "nodes=${NNODES} rank=${NODE_RANK} gpus/node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "scope=${SCOPE} task_mix=${TASK_MIX} batch/gpu=${BATCH_SIZE} grad_acc=${GRADIENT_ACCUMULATION_STEPS}"
echo "execution=${EXECUTION} checkpoint_segment=${CKPT_SEGMENT} mot_microbatch=${MOT_MICROBATCH}"
echo "output=${OUTPUT_DIR}/${JOB_NAME}"

if [[ "$DRY_RUN" == "true" ]]; then
    printf 'command:'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

"${COMMAND[@]}"
