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
FROZEN_ACE=false
GRADIENT_ACCUMULATION_STEPS=4
BATCH_SIZE=10
SAVE_FREQ=5000
IS_FT=true
CHUNK_SIZE=16
TASK_TYPE="train_ace"
DATASET_SIZE_ONE_EPOCH=1000_0000
PRRENT_DIR="/mnt/wangxiaofa/robot_dataset/lerobot-format-v21-ort6d/"
PARENT_DIR_V21="/mnt/wangxiaofa/robot_dataset/lerobot-format-v21-ort6d/"
PARENT_DIR_V30="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/"
LOG_DIR="/mnt/wangxiaofa/ace_logs"
OUTPUT_DIR="/mnt/wangxiaofa/action_chunk_encoder_exp"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=32
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
        --frozen_ace)
            FROZEN_ACE="$2"
            shift 2
            ;;
        --task_type)
            TASK_TYPE="$2"
            shift 2
            ;;
        --parent_dir)
            PARENT_DIR="$2"
            shift 2
            ;;
        --parent_dir_v21)
            PARENT_DIR_V21="$2"
            shift 2
            ;;
        --parent_dir_v30)
            PARENT_DIR_V30="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --chunk_size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --dataset_len)
            DATASET_SIZE_ONE_EPOCH="$2"
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
FIXED_OUTPUT_DIR=$OUTPUT_DIR
export PATH="/opt/conda/envs/lerobot/bin:$PATH"
ffmpeg
which ffmpeg
echo $PATH

# 执行训练命令
export CUDA_LAUNCH_BLOCKING=1
torchrun --nproc_per_node=${NPROC_PER_NODE} lerobot/scripts/dps_train_ace.py \
    --deepspeed="./ds_zero2.json" \
    --policy.type="robo_clip" \
    --policy.chunk_size=$CHUNK_SIZE \
    --policy.n_action_steps=$CHUNK_SIZE \
    --output_dir="$FIXED_OUTPUT_DIR" \
    --dataset.repo_id="whatever" \
    --dataset.image_transforms.enable=false \
    --dataset.wrist_image_transforms.enable=false \
    --dataset.wrist_image_transforms.is_primary=false \
    --dataset.dataset_size_one_epoch=$DATASET_SIZE_ONE_EPOCH \
    --batch_size=16 \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    --data_mix=$DATA_MIX \
    --save_freq=$SAVE_FREQ \
    --is_ft=$IS_FT \
    --dataset.processor="/mnt/wangxiaofa/pt_weights/InternVL3_5-2B-HF/" \
    --dataset.parent_dir=$PARENT_DIR \
    --dataset.parent_dir_v21=$PARENT_DIR_V21 \
    --dataset.parent_dir_v30=$PARENT_DIR_V30 \
    --dataset.video_backend="torchcodec" \
    --policy.scheduler_warmup_steps=$SCHEDULER_WARMUP_STEPS \
    --policy.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS \
    --policy.scheduler_platform_steps=$SCHEDULER_PLATFORM_STEPS \
    --policy.optimizer_weight_decay=$WEIGHT_DECAY \
    --policy.optimizer_lr=$OPTIMIZER_LR \
    --policy.scheduler_decay_lr=$OPTIMIZER_DECAY_LR \
    --policy.pretrained_path=$PRETRAINED_PATH \
    --policy.frozen_ace=$FROZEN_ACE \
    --task_type=$TASK_TYPE \
    --wandb.enable=true \
    --wandb.project="ace" \
    --job_name="$JOB_NAME" \
    --log_dir=$LOG_DIR \
    --weight_resume=true \
    --resume=false