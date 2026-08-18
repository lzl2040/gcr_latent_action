#!/bin/bash
# 感知 <-> 物理 对比预训练（robo_contrast），集群版。
#
# 单机：
#   bash train_ace.sh --job_name my_run --nproc_per_node 8
# 多机（每个节点各跑一次，NODE_RANK 不同）：
#   bash train_ace.sh --job_name my_run --nnodes 2 --nproc_per_node 8 \
#        --node_rank 0 --master_addr 10.0.0.1 --master_port 29500
#
# `--` 之后的参数原样透传给训练脚本，用来临时覆盖任何这里没显式暴露的配置：
#   bash train_ace.sh --job_name my_run -- --policy.tactile_dropout=0.5
#
# 注意：`vla2root.json` 和 ds_zero2_contrast.json 都是按**相对路径**读的，所以这里先 cd 到
# 仓库根目录，否则集群上从别的工作目录起任务会找不到数据集映射表。
cd "$(dirname "$0")" || exit 1

# ---------------------------------------------------------------- 默认参数值
NNODES=1
NPROC_PER_NODE=8
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500
JOB_NAME=""
DATA_MIX="debug_research_data"
TASK_TYPE="train_contrastive"

# 优化器 / 调度器
OPTIMIZER_LR=1e-4
SCHEDULER_WARMUP_STEPS=500
SCHEDULER_DECAY_STEPS=25000
SCHEDULER_PLATFORM_STEPS=20000
WEIGHT_DECAY=1e-5

# 训练规模
STEPS=60_0000
SAVE_FREQ=2000
LOG_FREQ=20
EVAL_FREQ=250
NUM_WORKERS=12
DATASET_SIZE_ONE_EPOCH=1000000
SAMPLE_RATIO=5
# 每卡 micro batch。ds json 里写死了一个值，这里给了就临时改写一份配置，
# 不改动仓库里的共享 json（对比学习吃 batch，集群卡型不同就得调）。
BATCH_SIZE=""

# 时间窗口 / 模型结构
WINDOW_MODE="duration"
CHUNK_SIZE=32
GROUP_SIZE=4
CHUNK_SECONDS=1.6

# 触觉塔："resnet18"（从头学）或 "ftp1"（加载 FTP-1 预训练的 per-sensor tokenizer，冻结）。
# 选 ftp1 时 FTP1_TACTILE_DIR 必须指向集群上 hpt_tokenizer/*.safetensors 所在目录，
# 且 __post_init__ 会把触觉图像尺寸强制成 224。
TACTILE_BACKBONE="resnet18"
FTP1_TACTILE_DIR="/mnt/wangxiaofa/pt_weights/ftp1_v0426_50kstep/"
# `--` 之后收集到这里，原样透传给训练脚本
EXTRA_ARGS=()

# 路径（集群挂载）
PARENT_DIR_V21="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/"
PARENT_DIR_V30="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/"
# 逗号分隔的额外根目录，给不在上面两个挂载点里的数据集用（例如 OpenNeoData）
PARENT_DIR_EXTRA="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/"
PROCESSOR="/mnt/wangxiaofa/pt_weights/InternVL3_5-2B-HF/"
VISION_MODEL="/mnt/wangxiaofa/pt_weights/dinov3-vitb16-pretrain-lvd1689m"
TEXT_MODEL="/mnt/wangxiaofa/pt_weights/siglip2-base-patch16-224"
LOG_DIR="/mnt/wangxiaofa/ace_logs"
OUTPUT_DIR="/mnt/wangxiaofa/robo_contrast_exp"
PRETRAINED_PATH=""

export LEROBOT_VIDEO_DECODER_CACHE_SIZE=256
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ---------------------------------------------------------------- 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes) NNODES="$2"; shift 2 ;;
        --nproc_per_node) NPROC_PER_NODE="$2"; shift 2 ;;
        --node_rank) NODE_RANK="$2"; shift 2 ;;
        --master_addr) MASTER_ADDR="$2"; shift 2 ;;
        --master_port) MASTER_PORT="$2"; shift 2 ;;
        --job_name) JOB_NAME="$2"; shift 2 ;;
        --data_mix) DATA_MIX="$2"; shift 2 ;;
        --task_type) TASK_TYPE="$2"; shift 2 ;;
        --optimizer_lr) OPTIMIZER_LR="$2"; shift 2 ;;
        --scheduler_warmup_steps) SCHEDULER_WARMUP_STEPS="$2"; shift 2 ;;
        --scheduler_decay_steps) SCHEDULER_DECAY_STEPS="$2"; shift 2 ;;
        --scheduler_platform_steps) SCHEDULER_PLATFORM_STEPS="$2"; shift 2 ;;
        --weight_decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --save_freq) SAVE_FREQ="$2"; shift 2 ;;
        --log_freq) LOG_FREQ="$2"; shift 2 ;;
        --eval_freq) EVAL_FREQ="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --dataset_len) DATASET_SIZE_ONE_EPOCH="$2"; shift 2 ;;
        --sample_ratio) SAMPLE_RATIO="$2"; shift 2 ;;
        --window_mode) WINDOW_MODE="$2"; shift 2 ;;
        --chunk_size) CHUNK_SIZE="$2"; shift 2 ;;
        --group_size) GROUP_SIZE="$2"; shift 2 ;;
        --chunk_seconds) CHUNK_SECONDS="$2"; shift 2 ;;
        --tactile_backbone) TACTILE_BACKBONE="$2"; shift 2 ;;
        --ftp1_tactile_dir) FTP1_TACTILE_DIR="$2"; shift 2 ;;
        --parent_dir_v21) PARENT_DIR_V21="$2"; shift 2 ;;
        --parent_dir_v30) PARENT_DIR_V30="$2"; shift 2 ;;
        --parent_dir_extra) PARENT_DIR_EXTRA="$2"; shift 2 ;;
        --processor) PROCESSOR="$2"; shift 2 ;;
        --vision_model) VISION_MODEL="$2"; shift 2 ;;
        --text_model) TEXT_MODEL="$2"; shift 2 ;;
        --log_dir) LOG_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --pre_path) PRETRAINED_PATH="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$JOB_NAME" ]]; then
    echo "错误：必须指定 --job_name"
    exit 1
fi

# ftp1 触觉塔要读本地权重目录，而配置里的默认值是开发机路径，集群上不存在。
# 与其等模型初始化到一半才炸，不如在这里就说清楚。
TACTILE_ARGS=(--policy.tactile_backbone="$TACTILE_BACKBONE")
if [[ "$TACTILE_BACKBONE" == "ftp1" ]]; then
    if [[ -z "$FTP1_TACTILE_DIR" ]]; then
        echo "错误：--tactile_backbone ftp1 必须同时指定 --ftp1_tactile_dir（存放 hpt_tokenizer/*.safetensors 的目录）"
        exit 1
    fi
    if [[ ! -d "$FTP1_TACTILE_DIR" ]]; then
        echo "错误：--ftp1_tactile_dir 不存在: ${FTP1_TACTILE_DIR}"
        exit 1
    fi
    TACTILE_ARGS+=(--policy.ftp1_tactile_dir="$FTP1_TACTILE_DIR")
elif [[ -n "$FTP1_TACTILE_DIR" ]]; then
    echo "警告：--ftp1_tactile_dir 已忽略，因为 --tactile_backbone 是 ${TACTILE_BACKBONE}（只有 ftp1 会读它）"
fi

# ---------------------------------------------------------------- deepspeed 配置
# 每卡 batch 只在 ds json 里生效（训练脚本从 json 读 train_micro_batch_size_per_gpu），
# 所以要改就复制一份改副本，避免多个并发任务互相污染仓库里的共享文件。
DS_CONFIG="./ds_zero2_contrast.json"
if [[ -n "$BATCH_SIZE" ]]; then
    DS_CONFIG="/tmp/ds_zero2_contrast_${JOB_NAME}_${NODE_RANK}.json"
    python - "$BATCH_SIZE" "$DS_CONFIG" <<'PY'
import json, sys
cfg = json.load(open("./ds_zero2_contrast.json"))
cfg["train_micro_batch_size_per_gpu"] = int(sys.argv[1])
json.dump(cfg, open(sys.argv[2], "w"), indent=4)
PY
    echo "ds config -> ${DS_CONFIG} (micro batch ${BATCH_SIZE}/gpu)"
fi

echo "nodes=${NNODES} rank=${NODE_RANK} gpus/node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"

# ---------------------------------------------------------------- 执行训练命令
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    lerobot/scripts/dps_train_contrast.py \
    --deepspeed="$DS_CONFIG" \
    --policy.type="robo_contrast" \
    --policy.vision_model_name="$VISION_MODEL" \
    --policy.text_model_name="$TEXT_MODEL" \
    --policy.window_mode=$WINDOW_MODE \
    --policy.chunk_size=$CHUNK_SIZE \
    --policy.group_size=$GROUP_SIZE \
    --policy.chunk_seconds=$CHUNK_SECONDS \
    "${TACTILE_ARGS[@]}" \
    --policy.scheduler_warmup_steps=$SCHEDULER_WARMUP_STEPS \
    --policy.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS \
    --policy.scheduler_platform_steps=$SCHEDULER_PLATFORM_STEPS \
    --policy.optimizer_weight_decay=$WEIGHT_DECAY \
    --policy.optimizer_lr=$OPTIMIZER_LR \
    --policy.pretrained_path="$PRETRAINED_PATH" \
    --is_ft=false \
    --dataset.repo_id="whatever" \
    --dataset.image_transforms.enable=false \
    --dataset.wrist_image_transforms.enable=false \
    --dataset.wrist_image_transforms.is_primary=false \
    --dataset.processor="$PROCESSOR" \
    --dataset.parent_dir_v21="$PARENT_DIR_V21" \
    --dataset.parent_dir_v30="$PARENT_DIR_V30" \
    --dataset.parent_dir_extra="$PARENT_DIR_EXTRA" \
    --dataset.video_backend="torchcodec" \
    --dataset.sample_ratio=$SAMPLE_RATIO \
    --dataset.dataset_size_one_epoch=$DATASET_SIZE_ONE_EPOCH \
    --data_mix=$DATA_MIX \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --save_freq=$SAVE_FREQ \
    --log_freq=$LOG_FREQ \
    --eval_freq=$EVAL_FREQ \
    --output_dir="$OUTPUT_DIR" \
    --log_dir="$LOG_DIR" \
    --task_type=$TASK_TYPE \
    --wandb.enable=true \
    --wandb.project="robo_contrast" \
    --job_name="$JOB_NAME" \
    --weight_resume=true \
    --resume=false \
    "${EXTRA_ARGS[@]}"
