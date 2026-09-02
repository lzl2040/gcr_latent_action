#!/usr/bin/env bash
# Benchmark the MoT world model under a given trainability scope.
#
# The dataset flags mirror train_ace_local.sh so the measurement runs on the same real data;
# only the trainer differs (no deepspeed, no wandb). Knobs come from the environment because
# draccus rejects unknown CLI flags.
#
#   SCOPE=freeze_vision BATCH=8 bash scripts/bench_mot_scope.sh
set -euo pipefail

export SCOPE=${SCOPE:-gen_only}
export BATCH=${BATCH:-8}
export STEPS=${STEPS:-10}
export WARMUP=${WARMUP:-3}
export PER_TASK=${PER_TASK:-1}
export STAGE=${STAGE:-3}

python -u scripts/train_mot_world.py \
    --policy.type="robo_contrast" \
    --is_ft=false \
    --dataset.repo_id="whatever" \
    --dataset.image_transforms.enable=false \
    --dataset.wrist_image_transforms.enable=false \
    --dataset.wrist_image_transforms.is_primary=false \
    --dataset.processor="/Data/lzl/huggingface/InternVL3_5-2B-HF" \
    --dataset.parent_dir_v21="/Data/lerobot_data_ort6d" \
    --dataset.parent_dir_v30="/Data/lerobot_data_ort6d/v30" \
    --dataset.parent_dir_extra="/media/v-wangxiaofa/新加卷/lerobot_data" \
    --dataset.video_backend="torchcodec" \
    --data_mix="debug_research_data" \
    --dataset.sample_ratio=5 \
    --dataset.dataset_size_one_epoch=1000000 \
    --num_workers=12 \
    --output_dir="/tmp/mot_bench_${SCOPE}" \
    --steps=100 \
    --policy.chunk_size=32 \
    --policy.group_size=4 \
    --policy.chunk_seconds=1.6 \
    --task_type="train_contrastive" \
    "$@"
