#!/bin/bash

# 默认参数值
NNODES=1
NPROC_PER_NODE=2
JOB_NAME=""
DATA_MIX="oxe_magic_soup_plus"
OPTIMIZER_LR=2.5e-5
OPTIMIZER_DECAY_LR=2.5e-6
SCHEDULER_WARMUP_STEPS=2000
SCHEDULER_DECAY_STEPS=60000
SCHEDULER_PLATFORM_STEPS=1
WEIGHT_DECAY=1e-5
PRETRAINED_PATH=""
GRADIENT_ACCUMULATION_STEPS=4
BATCH_SIZE=10
SAVE_FREQ=5000
USE_LORA=false
MAX_FRAME=3
MAX_HISTORY=10
CALVIN_SUB_TASK=0
USE_STATE=true
LOSS_TYPE="raw"
IS_FT=true
IMG_DECODER_PART_TRAIN=true

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --data_mix)
            DATA_MIX="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --job_name)
            JOB_NAME="$2"
            shift 2
            ;;
        --optimizer_lr)
            OPTIMIZER_LR="$2"
            shift 2
            ;;
        --scheduler_decay_lr)
            OPTIMIZER_DECAY_LR="$2"
            shift 2
            ;;
        --scheduler_warmup_steps)
            SCHEDULER_WARMUP_STEPS="$2"
            shift 2
            ;;
        --scheduler_decay_steps)
            SCHEDULER_DECAY_STEPS="$2"
            shift 2
            ;;
        --scheduler_platform_steps)
            SCHEDULER_PLATFORM_STEPS="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --bs)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_frame)
            MAX_FRAME="$2"
            shift 2
            ;;
        --max_history_frame)
            MAX_HISTORY="$2"
            shift 2
            ;;
        --img_decoder_part_train)
            IMG_DECODER_PART_TRAIN="$2"
            shift 2
            ;;
        --calvin_sub_task)
            CALVIN_SUB_TASK="$2"
            shift 2
            ;;
        --loss_type)
            LOSS_TYPE="$2"
            shift 2
            ;;
        --save_freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --gradient_acc)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --is_ft)
            IS_FT="$2"
            shift 2
            ;;
        --pre_path)
            PRETRAINED_PATH="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查必要参数
if [[ -z "$JOB_NAME" ]]; then
    echo "错误：必须指定 --job_name"
    exit 1
fi

# 固定输出目录（根据需求修改）
FIXED_OUTPUT_DIR="/mnt/wangxiaofa/world_model_exp"

# 执行训练命令
CUDA_LAUNCH_BLOCKING=1 torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m lerobot.scripts.fsdp_train_world \
    --policy.type="latent_wm" \
    --policy.use_state=$USE_STATE \
    --policy.max_frame=$MAX_FRAME \
    --policy.max_history_frame=$MAX_HISTORY \
    --policy.loss_type=$LOSS_TYPE \
    --policy.img_decoder_part_train=$IMG_DECODER_PART_TRAIN \
    --policy.chunk_size=$((MAX_FRAME-1)) \
    --policy.n_action_steps=$((MAX_FRAME-1)) \
    --output_dir="$FIXED_OUTPUT_DIR" \
    --dataset.repo_id="whatever" \
    --dataset.image_transforms.enable=false \
    --dataset.wrist_image_transforms.enable=false \
    --dataset.wrist_image_transforms.is_primary=false \
    --batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    --data_mix=$DATA_MIX \
    --save_freq=$SAVE_FREQ \
    --is_ft=$IS_FT \
    --dataset.processor="/mnt/wangxiaofa/pt_weights/InternVL3_5-2B-HF/" \
    --dataset.parent_dir="/mnt/wangxiaofa/robot_dataset/lerobot-format/" \
    --policy.scheduler_warmup_steps=$SCHEDULER_WARMUP_STEPS \
    --policy.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS \
    --policy.scheduler_platform_steps=$SCHEDULER_PLATFORM_STEPS \
    --policy.optimizer_weight_decay=$WEIGHT_DECAY \
    --policy.optimizer_lr=$OPTIMIZER_LR \
    --policy.scheduler_decay_lr=$OPTIMIZER_DECAY_LR \
    --policy.pretrained_path=$PRETRAINED_PATH \
    --wandb.enable=true \
    --wandb.project="world_model" \
    --job_name="$JOB_NAME" \
    --log_dir="/mnt/wangxiaofa/world_model_logs" \
    --weight_resume=true \
    --resume=false