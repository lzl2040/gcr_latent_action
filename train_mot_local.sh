#!/usr/bin/env bash
# Local launcher for Phi-4-Multimodal MoT training.
#
#   conda activate lerobot_v2
#   bash train_mot_local.sh
#
# This is a shared machine. Pin devices explicitly for reproducible runs:
#
#   CUDA_VISIBLE_DEVICES=0,3 JOB_NAME=mot_stage3 bash train_mot_local.sh
#
# Environment variables provide convenient local defaults; regular train_mot.sh arguments
# can still be appended and take precedence:
#
#   SCOPE=freeze_vision BATCH_SIZE=16 bash train_mot_local.sh --steps 1000
set -euo pipefail

cd "$(dirname "$0")" || exit 1

RESOLVE_ONLY=false
for argument in "$@"; do
    if [[ "$argument" == "--dry_run" || "$argument" == "--help" || "$argument" == "-h" ]]; then
        RESOLVE_ONLY=true
        break
    fi
done

# Follow train_ace_local.sh: select only genuinely idle cards unless the caller pins them.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES=$(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | awk -F', ' -v limit="${MAX_GPU_MEMORY_USED_MB:-5000}" \
                '$2 < limit {printf "%s%s", separator, $1; separator=","}'
    )
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if [[ "$RESOLVE_ONLY" == "true" ]]; then
        CUDA_VISIBLE_DEVICES=0
        echo "No idle GPU found; using device 0 only to resolve the command."
    else
        echo "No GPU has less than ${MAX_GPU_MEMORY_USED_MB:-5000} MB in use; refusing to start." >&2
        echo "Check nvidia-smi or set CUDA_VISIBLE_DEVICES explicitly." >&2
        exit 1
    fi
fi
export CUDA_VISIBLE_DEVICES

IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES// /}"
if [[ ${#visible_devices[@]} -eq 0 ]]; then
    echo "CUDA_VISIBLE_DEVICES did not contain any devices." >&2
    exit 1
fi
NPROC_PER_NODE=${NPROC_PER_NODE:-${#visible_devices[@]}}

# Killed local jobs can leave a fixed rendezvous port occupied, so choose a free one.
MASTER_PORT=${MASTER_PORT:-$(
    python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
)}

JOB_NAME=${JOB_NAME:-phi4_mot_local}
DATA_MIX=${DATA_MIX:-debug_research_data}
TASK_MIX=${TASK_MIX:-stage3}
SCOPE=${SCOPE:-gen_only}
BATCH_SIZE=${BATCH_SIZE:-32}
STEPS=${STEPS:-600000}
SAVE_FREQ=${SAVE_FREQ:-2000}
LOG_FREQ=${LOG_FREQ:-20}
NUM_WORKERS=${NUM_WORKERS:-12}
DATASET_SIZE_ONE_EPOCH=${DATASET_SIZE_ONE_EPOCH:-1000000}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
RESUME=${RESUME:-false}
WANDB_ENABLE=${WANDB_ENABLE:-false}

PHI_DIR=${PHI_DIR:-/Data/lzl/huggingface/Phi-4-multimodal-instruct}
VAE_DIR=${VAE_DIR:-/Data/lzl/huggingface/Cosmos3-Edge/vae}
PROCESSOR=${PROCESSOR:-/Data/lzl/huggingface/InternVL3_5-2B-HF}
PARENT_DIR_V21=${PARENT_DIR_V21:-/Data/lerobot_data_ort6d}
PARENT_DIR_V30=${PARENT_DIR_V30:-/Data/lerobot_data_ort6d/v30}
PARENT_DIR_EXTRA=${PARENT_DIR_EXTRA:-/media/v-wangxiaofa/新加卷/lerobot_data}
OUTPUT_DIR=${OUTPUT_DIR:-phi4_mot_local}
LOG_DIR=${LOG_DIR:-logs}

echo "devices=${CUDA_VISIBLE_DEVICES} processes=${NPROC_PER_NODE} master_port=${MASTER_PORT}"

LOCAL_ARGS=(
    --job_name "$JOB_NAME"
    --nnodes 1
    --nproc_per_node "$NPROC_PER_NODE"
    --master_addr 127.0.0.1
    --master_port "$MASTER_PORT"
    --data_mix "$DATA_MIX"
    --task_mix "$TASK_MIX"
    --scope "$SCOPE"
    --batch_size "$BATCH_SIZE"
    --gradient_acc "$GRADIENT_ACCUMULATION_STEPS"
    --steps "$STEPS"
    --save_freq "$SAVE_FREQ"
    --log_freq "$LOG_FREQ"
    --num_workers "$NUM_WORKERS"
    --dataset_len "$DATASET_SIZE_ONE_EPOCH"
    --resume "$RESUME"
    --phi_dir "$PHI_DIR"
    --vae_dir "$VAE_DIR"
    --processor "$PROCESSOR"
    --parent_dir_v21 "$PARENT_DIR_V21"
    --parent_dir_v30 "$PARENT_DIR_V30"
    --parent_dir_extra "$PARENT_DIR_EXTRA"
    --output_dir "$OUTPUT_DIR"
    --log_dir "$LOG_DIR"
    --wandb_enable "$WANDB_ENABLE"
)

if [[ -n "${MOT_MICROBATCH:-}" ]]; then
    LOCAL_ARGS+=(--mot_microbatch "$MOT_MICROBATCH")
fi
if [[ -n "${RESUME_DIR:-}" ]]; then
    LOCAL_ARGS+=(--resume_dir "$RESUME_DIR")
fi

bash train_mot.sh "${LOCAL_ARGS[@]}" "$@"
