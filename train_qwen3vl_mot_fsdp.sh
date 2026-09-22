#!/usr/bin/env bash
set -euo pipefail

# Single-node 8xH100 launcher for stage-two Qwen3-VL + Generation Expert training.
# FP8 accelerates eligible Linear GEMMs; trainable master parameters stay BF16 and
# bitsandbytes stores AdamW moments in 8-bit blocks.
WEIGHT_RESUME="${WEIGHT_RESUME:-false}"
if [[ "${WEIGHT_RESUME}" != "true" ]]; then
    : "${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT for a fresh stage-two run}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE="${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-256}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-$("${PYTHON_BIN}" -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")}"

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

STAGE1_CONFIG_ARGS=()
STAGE1_CHECKPOINT_ARGS=()
if [[ -n "${STAGE1_CHECKPOINT:-}" ]]; then
    STAGE1_CHECKPOINT_ARGS+=(--policy.stage1_checkpoint="${STAGE1_CHECKPOINT}")
fi
if [[ -n "${STAGE1_CONFIG:-}" ]]; then
    STAGE1_CONFIG_ARGS+=(--policy.stage1_config="${STAGE1_CONFIG}")
fi

"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    lerobot/scripts/fsdp_train_contrast.py \
    --policy.type="qwen3vl_mot" \
    --policy.qwen3vl_dir="${QWEN3VL_DIR:-/Data/lzl/huggingface/Qwen3-VL-4B-Instruct}" \
    --policy.cosmos3_dir="${COSMOS3_DIR:-/Data/lzl/huggingface/Cosmos3-Edge}" \
    --policy.understanding_tuning_mode="${UNDERSTANDING_TUNING_MODE:-full}" \
    --policy.understanding_lora_rank="${UNDERSTANDING_LORA_RANK:-16}" \
    --policy.understanding_lora_alpha="${UNDERSTANDING_LORA_ALPHA:-16}" \
    --policy.understanding_text_lora_layers="${UNDERSTANDING_TEXT_LORA_LAYERS:-0}" \
    --policy.understanding_vision_lora_layers="${UNDERSTANDING_VISION_LORA_LAYERS:-4}" \
    --policy.understanding_lr_scale="${UNDERSTANDING_LR_SCALE:-0.05}" \
    --policy.physical_tuning_mode="${PHYSICAL_TUNING_MODE:-frozen}" \
    --policy.window_mode="frames" \
    --policy.chunk_size=32 \
    --policy.n_action_steps=32 \
    --policy.world_video_frames="${WORLD_VIDEO_FRAMES:-9}" \
    --policy.generation_gradient_checkpointing=true \
    --policy.understanding_gradient_checkpointing=true \
    --policy.optimizer_lr="${LEARNING_RATE:-1e-4}" \
    --policy.scheduler_warmup_steps="${WARMUP_STEPS:-500}" \
    --policy.scheduler_plateau_steps="${PLATEAU_STEPS:-2000}" \
    --policy.scheduler_decay_steps="${DECAY_STEPS:-100000}" \
    --policy.scheduler_decay_lr="${DECAY_LR:-1.5e-6}" \
    --dataset.repo_id="whatever" \
    --dataset.image_transforms.enable=false \
    --dataset.image_transforms.img_size=256 \
    --dataset.wrist_image_transforms.enable=false \
    --dataset.wrist_image_transforms.is_primary=false \
    --dataset.parent_dir_v21="${DATA_ROOT_V21:-/Data/lerobot_data_ort6d}" \
    --dataset.parent_dir_v30="${DATA_ROOT_V30:-/Data/lerobot_data_ort6d/v30}" \
    --dataset.parent_dir_extra="${DATA_ROOT_EXTRA:-}" \
    --dataset.video_backend="${VIDEO_BACKEND:-torchcodec}" \
    --dataset.dataset_size_one_epoch="${SAMPLES_PER_EPOCH:-100000}" \
    --data_mix="${DATA_MIX:-debug_research_data}" \
    --num_workers="${NUM_WORKERS:-8}" \
    --batch_size="${BATCH_SIZE:-16}" \
    --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}" \
    --fsdp.fp8="${FP8_ENABLED:-true}" \
    --fsdp.fp8_recipe="${FP8_RECIPE:-rowwise_with_gw_hp}" \
    --fsdp.fp8_scope="${FP8_SCOPE:-generation_vlm}" \
    --fsdp.fp8_emulate="${FP8_EMULATE:-false}" \
    --fsdp.fp8_min_features="${FP8_MIN_FEATURES:-128}" \
    --fsdp.min_wrap_params="${FSDP_MIN_WRAP_PARAMS:-1000000}" \
    --fsdp.forward_prefetch="${FSDP_FORWARD_PREFETCH:-false}" \
    --fsdp.limit_all_gathers="${FSDP_LIMIT_ALL_GATHERS:-true}" \
    --fsdp.replicate_frozen_params="${FSDP_REPLICATE_FROZEN:-true}" \
    --output_dir="${OUTPUT_DIR:-qwen3vl_mot_fsdp}" \
    --job_name="${JOB_NAME:-qwen3vl_mot_fsdp}" \
    --weight_resume="${WEIGHT_RESUME}" \
    --save_freq="${SAVE_FREQ:-2000}" \
    --log_freq="${LOG_FREQ:-20}" \
    --eval_freq=0 \
    --steps="${STEPS:-600000}" \
    --task_type="train_stage2" \
    --wandb.enable="${WANDB_ENABLE:-false}" \
    --wandb.project="${WANDB_PROJECT:-lerobot-stage2}" \
    "${STAGE1_CHECKPOINT_ARGS[@]}" \
    "${STAGE1_CONFIG_ARGS[@]}" \
    "$@"
