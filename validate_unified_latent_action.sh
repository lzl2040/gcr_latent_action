#!/usr/bin/env bash

set -euo pipefail

source /home/v-wangxiaofa/anaconda3/etc/profile.d/conda.sh
conda activate latbot

GPU_ID="${GPU_ID:-0}"
VALIDATION_START_BATCH="${VALIDATION_START_BATCH:-0}"
NUM_VALIDATION_BATCHES="${NUM_VALIDATION_BATCHES:-200}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR:-unified_validation_step80000}"
SAVE_IMAGES="${SAVE_IMAGES:-true}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/validate_unified_latent_action.py \
    --policy.type="latent_act" \
    --policy.max_frame=16 \
    --policy.vlm_path="/Data/lzl/huggingface/InternVL3_5-2B-HF" \
    --policy.img_pred_model="/Data/lzl/huggingface/Sana_1600M_512px_diffusers" \
    --policy.img_encoder_model="/Data/lzl/huggingface/CLIP-ViT-H-14-laion2B-s32B-b79K" \
    --policy.action_expert_path="/Data/lzl/weights/pi_zero_pt/pi0_gemma_expert_only.pt" \
    --policy.train_main_layers=0 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=false \
    --policy.train_from_scratch=true \
    --is_ft=true \
    --dataset.repo_id="whatever" \
    --dataset.image_transforms.enable=false \
    --dataset.wrist_image_transforms.enable=true \
    --dataset.wrist_image_transforms.is_primary=false \
    --dataset.processor="/Data/lzl/huggingface/InternVL3_5-2B-HF" \
    --dataset.parent_dir="/Data/lerobot_data" \
    --data_mix="simpler_bridge" \
    --dataset.sample_ratio=5 \
    --batch_size=1 \
    --num_workers=2 \
    --validation_checkpoint="/Data/lzl/latent_action/0124_pretrain_latent_unfied_decoder/step80000.pt" \
    --validation_start_batch="${VALIDATION_START_BATCH}" \
    --num_validation_batches="${NUM_VALIDATION_BATCHES}" \
    --num_inference_steps="${NUM_INFERENCE_STEPS}" \
    --action_dims=6 \
    --validation_output_dir="${VALIDATION_OUTPUT_DIR}" \
    --save_images="${SAVE_IMAGES}" \
    --max_saved_images="${NUM_VALIDATION_BATCHES}"
