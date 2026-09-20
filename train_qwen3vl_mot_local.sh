#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   conda activate lerobot_v2
#   STAGE1_CHECKPOINT=/path/to/exported_robo_contrast \
#     CUDA_VISIBLE_DEVICES=0,1 bash train_qwen3vl_mot_local.sh
#
# For a fresh run, the stage-one path must contain config.json and model.safetensors.
# A stage-two resume reads the embedded stage-one architecture from its saved config.
WEIGHT_RESUME="${WEIGHT_RESUME:-false}"
if [[ "${WEIGHT_RESUME}" != "true" ]]; then
    : "${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT for a fresh stage-two run}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE="${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-256}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MASTER_PORT="${MASTER_PORT:-$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")}"
STAGE1_CONFIG_ARGS=()
STAGE1_CHECKPOINT_ARGS=()
if [[ -n "${STAGE1_CHECKPOINT:-}" ]]; then
    STAGE1_CHECKPOINT_ARGS+=(--policy.stage1_checkpoint="${STAGE1_CHECKPOINT}")
fi
if [[ -n "${STAGE1_CONFIG:-}" ]]; then
    STAGE1_CONFIG_ARGS+=(--policy.stage1_config="${STAGE1_CONFIG}")
fi

deepspeed --master_port="${MASTER_PORT}" lerobot/scripts/dps_train_contrast.py \
    --deepspeed="${DEEPSPEED_CONFIG:-./ds_zero2_qwen3vl_mot.json}" \
    --policy.type="qwen3vl_mot" \
    --policy.qwen3vl_dir="${QWEN3VL_DIR:-/Data/lzl/huggingface/Qwen3-VL-4B-Instruct}" \
    --policy.cosmos3_dir="${COSMOS3_DIR:-/Data/lzl/huggingface/Cosmos3-Edge}" \
    --policy.understanding_tuning_mode="${UNDERSTANDING_TUNING_MODE:-lora}" \
    --policy.understanding_lora_rank="${UNDERSTANDING_LORA_RANK:-16}" \
    --policy.understanding_lora_alpha="${UNDERSTANDING_LORA_ALPHA:-16}" \
    --policy.physical_tuning_mode="${PHYSICAL_TUNING_MODE:-frozen}" \
    --policy.world_video_frames="${WORLD_VIDEO_FRAMES:-9}" \
    --policy.generation_gradient_checkpointing=true \
    --policy.understanding_gradient_checkpointing=true \
    --policy.optimizer_lr="${LEARNING_RATE:-1e-4}" \
    --policy.scheduler_warmup_steps=500 \
    --policy.scheduler_plateau_steps=2000 \
    --policy.scheduler_decay_steps=100000 \
    --policy.scheduler_decay_lr=1.5e-6 \
    --dataset.repo_id="whatever" \
    --dataset.image_transforms.enable=false \
    --dataset.image_transforms.img_size=256 \
    --dataset.wrist_image_transforms.enable=false \
    --dataset.wrist_image_transforms.is_primary=false \
    --dataset.parent_dir_v21="${DATA_ROOT_V21:-/Data/lerobot_data_ort6d}" \
    --dataset.parent_dir_v30="${DATA_ROOT_V30:-/Data/lerobot_data_ort6d/v30}" \
    --dataset.parent_dir_extra="${DATA_ROOT_EXTRA:-}" \
    --dataset.video_backend="torchcodec" \
    --dataset.dataset_size_one_epoch="${SAMPLES_PER_EPOCH:-100000}" \
    --data_mix="${DATA_MIX:-debug_research_data}" \
    --num_workers="${NUM_WORKERS:-8}" \
    --output_dir="${OUTPUT_DIR:-qwen3vl_mot}" \
    --job_name="${JOB_NAME:-qwen3vl_mot}" \
    --weight_resume="${WEIGHT_RESUME}" \
    --save_freq="${SAVE_FREQ:-2000}" \
    --log_freq="${LOG_FREQ:-20}" \
    --eval_freq=0 \
    --steps="${STEPS:-600000}" \
    --task_type="train_stage2" \
    --wandb.enable="${WANDB_ENABLE:-false}" \
    "${STAGE1_CHECKPOINT_ARGS[@]}" \
    "${STAGE1_CONFIG_ARGS[@]}" \
    "$@"
