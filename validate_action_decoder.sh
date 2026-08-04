#!/bin/bash
# Validation script for action encoder and decoder
# Single GPU validation

# conda activate lerobot_v2

CUDA_VISIBLE_DEVICES=0 python scripts/validate_action_decoder.py \
    --policy.type="robo_clip" \
    --dataset.repo_id="whatever" \
    --dataset.image_transforms.enable=false \
    --dataset.wrist_image_transforms.enable=false \
    --dataset.wrist_image_transforms.is_primary=false \
    --dataset.processor="/Data/lzl/huggingface/InternVL3_5-2B-HF" \
    --dataset.parent_dir="/Data/lerobot_data_ort6d" \
    --data_mix="libero" \
    --dataset.sample_ratio=5 \
    --policy.pretrained_path="/Data/lzl/ace_weights/action_decoder_weights" \
    --policy.frozen_ace=true \
    --policy.chunk_size=16 \
    --policy.action_dim=7 \
    --policy.hidden_dim=768 \
    --policy.max_action_dim=32 \
    --policy.vision_model_name="/Data/lzl/huggingface/siglip2-base-patch16-224" \
    --job_name="validate_action_decoder"