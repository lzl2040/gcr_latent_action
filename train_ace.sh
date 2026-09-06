#!/bin/bash
set -euo pipefail

# 感知 <-> 物理 对比预训练（robo_contrast），集群版。
#
# 单机：
#   bash train_ace.sh --job_name my_run --nproc_per_node 8
# 多机（每个节点各跑一次，NODE_RANK 不同）：
#   bash train_ace.sh --job_name my_run --nnodes 2 --nproc_per_node 8 \
#        --node_rank 0 --master_addr 10.0.0.1 --master_port 29500
#
# `--` 之后的参数原样透传给训练脚本，用来临时覆盖任何这里没显式暴露的配置：
#   bash train_ace.sh --job_name my_run -- --policy.modality_dropout_tactile=0.5
#
# Cosmos3 vision + Wan VAE target：
#   bash train_ace.sh --job_name cosmos_vae \
#     --vision_backbone cosmos3 --perception_recon_target vae
#
# 使用另一套统一权重目录：
#   bash train_ace.sh --job_name local_paths --weights_root /Data/lzl/huggingface
#
# 注意：`vla2root.json` 和 ds_zero2_contrast.json 都是按**相对路径**读的，所以这里先 cd 到
# 仓库根目录，否则集群上从别的工作目录起任务会找不到数据集映射表。
cd "$(dirname "$0")" || exit 1

usage() {
    cat <<'EOF'
用法：
  bash train_ace.sh --job_name NAME [选项] [-- 额外 Draccus 参数]

分布式：
  --nnodes N --nproc_per_node N --node_rank N
  --master_addr HOST --master_port PORT

权重：
  --weights_root DIR             所有默认权重的根目录
  --vision_model DIR             DINOv3 权重
  --text_model DIR               SigLIP2 text 权重
  --cosmos3_dir DIR              Cosmos3-Edge（vision_encoder/ 和 vae/）
  --qwen3vl_dir DIR              Qwen3-VL 本地 snapshot
  --processor DIR                dataset processor
  --ftp1_tactile_dir DIR         FTP-1 hpt_tokenizer 权重目录
  --anytouch_checkpoint FILE     AnyTouch stage-2 checkpoint
  --pre_path PATH                预训练 RoboContrast checkpoint
  --check_paths true|false       启动前检查当前激活路径，默认 true

感知分支：
  --vision_backbone dinov3|cosmos3|qwen3vl
  --perception_recon_target vision|vae
  --num_cls_tokens N --num_change_queries N
  --num_evidence_layers N --num_fusion_layers N --num_predictor_layers N
  --perception_recon_weight FLOAT --vae_repeat_frames N
  --patch_token_stride N --freeze_vision true|false --freeze_text true|false
  --perception_only true|false --perception_camera_mode primary|all
  --query_probe_freq N

物理/触觉分支：
  --tactile_backbone resnet18|ftp1|anytouch
  --tactile_frames N --tactile_tokens_per_pad 1|2 --max_tactile_views N
  --tactile_dead_std FLOAT --tactile_pretrained true|false
  --tactile_recon_weight FLOAT --tactile_lr_scale FLOAT
  --anytouch_forward_batch_size N
  --ftp1_tactile_sensors NAME1,NAME2
  --num_physical_layers N
  --gradient_checkpointing true|false
  --modality_dropout_tactile FLOAT
  --modality_dropout_state FLOAT --modality_dropout_action FLOAT

时间窗口与采样：
  --window_mode duration|frames --chunk_size N --group_size N
  --n_action_steps N --chunk_seconds FLOAT
  --chunk_frames_min N --chunk_frames_max N [--frame_horizon N]
  --same_dataset_frac FLOAT --episode_group_frac FLOAT
  --episode_group_size N --min_frame_gap N --false_negative_frame_gap N

训练：
  --batch_size N                 每卡 micro batch
  --gradient_acc N              梯度累积步数
  --steps N --save_freq N --log_freq N --eval_freq N --num_workers N
  --dataset_len N --sample_ratio FLOAT
  --optimizer_lr FLOAT --scheduler_decay_lr FLOAT --weight_decay FLOAT
  --scheduler_warmup_steps N --scheduler_decay_steps N
  --scheduler_platform_steps N
  --weight_resume true|false --resume true|false --save_checkpoint true|false
  --seed N --deepspeed_config FILE --dry_run

其他：
  --data_mix NAME --task_type train_contrastive|train_perception
  [--job_type NAME]
  --parent_dir_v21 DIR --parent_dir_v30 DIR --parent_dir_extra DIRS
  --video_backend NAME --output_dir DIR --log_dir DIR
  --wandb_enable true|false --wandb_project NAME [--wandb_entity NAME]
EOF
}

# ---------------------------------------------------------------- 默认参数值
NNODES=1
NPROC_PER_NODE=8
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500
JOB_NAME=""
JOB_TYPE=""
DATA_MIX="debug_research_data"
TASK_TYPE="train_contrastive"
SEED=1000
DRY_RUN=false
CHECK_PATHS=true

# 优化器 / 调度器
OPTIMIZER_LR=1e-4
SCHEDULER_DECAY_LR=2.5e-6
SCHEDULER_WARMUP_STEPS=500
SCHEDULER_DECAY_STEPS=30000
SCHEDULER_PLATFORM_STEPS=2000
WEIGHT_DECAY=1e-5

# 训练规模
STEPS=600000
SAVE_FREQ=2000
LOG_FREQ=20
EVAL_FREQ=250
NUM_WORKERS=12
DATASET_SIZE_ONE_EPOCH=1000000
SAMPLE_RATIO=5
BATCH_SIZE=256
GRADIENT_ACCUMULATION_STEPS=1
SAVE_CHECKPOINT=true
WEIGHT_RESUME=true
RESUME=false
DEEPSPEED_CONFIG="./ds_zero2_contrast.json"
VIDEO_BACKEND="torchcodec"

# 时间窗口 / 模型结构
WINDOW_MODE="duration"
CHUNK_SIZE=32
N_ACTION_STEPS=16
GROUP_SIZE=4
CHUNK_SECONDS=1.6
CHUNK_FRAMES_MIN=8
CHUNK_FRAMES_MAX=48
FRAME_HORIZON=""

# 感知分支
VISION_BACKBONE="dinov3"
PERCEPTION_RECON_TARGET="vision"
NUM_CLS_TOKENS=1
NUM_CHANGE_QUERIES=16
NUM_EVIDENCE_LAYERS=5
NUM_FUSION_LAYERS=5
NUM_PREDICTOR_LAYERS=3
PERCEPTION_RECON_WEIGHT=1.0
VAE_REPEAT_FRAMES=1
PATCH_TOKEN_STRIDE=1
FREEZE_VISION_ENCODER=true
FREEZE_TEXT_ENCODER=true
PERCEPTION_ONLY=false
PERCEPTION_CAMERA_MODE="primary"
QUERY_PROBE_FREQ=50

# 物理分支和困难负样本
NUM_PHYSICAL_LAYERS=14
SAME_DATASET_FRAC=0.75
EPISODE_GROUP_FRAC=0.75
EPISODE_GROUP_SIZE=8
MIN_FRAME_GAP=32
FALSE_NEGATIVE_FRAME_GAP=32

# 触觉塔："resnet18"、"ftp1" 或 "anytouch"。后两者加载预训练权重并冻结。
TACTILE_BACKBONE="resnet18"
TACTILE_FRAMES=4
TACTILE_TOKENS_PER_PAD=2
MAX_TACTILE_VIEWS=6
TACTILE_DEAD_STD=0.002
TACTILE_PRETRAINED=true
TACTILE_RECON_WEIGHT=0.1
TACTILE_LR_SCALE=0.1
FTP1_TACTILE_SENSORS=""
ANYTOUCH_FORWARD_BATCH_SIZE=128
GRADIENT_CHECKPOINTING=true
MODALITY_DROPOUT_TACTILE=0.3
MODALITY_DROPOUT_STATE=0.15
MODALITY_DROPOUT_ACTION=0.1
# `--` 之后收集到这里，原样透传给训练脚本
EXTRA_ARGS=()

# 路径（集群挂载）
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/mnt/wangxiaofa/pt_weights}"
PROCESSOR="${PROCESSOR:-}"
VISION_MODEL="${VISION_MODEL:-}"
TEXT_MODEL="${TEXT_MODEL:-}"
COSMOS3_DIR="${COSMOS3_DIR:-}"
QWEN3VL_DIR="${QWEN3VL_DIR:-}"
FTP1_TACTILE_DIR="${FTP1_TACTILE_DIR:-}"
ANYTOUCH_CHECKPOINT="${ANYTOUCH_CHECKPOINT:-}"
PARENT_DIR_V21="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/"
PARENT_DIR_V30="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/"
# 逗号分隔的额外根目录，给不在上面两个挂载点里的数据集用（例如 OpenNeoData）
PARENT_DIR_EXTRA="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/"
LOG_DIR="/mnt/wangxiaofa/ace_logs"
OUTPUT_DIR="/mnt/wangxiaofa/robo_contrast_exp"
PRETRAINED_PATH="${PRETRAINED_PATH:-}"

# W&B。**不要把 API key 写进这个文件**——它是入库的，写进来等于把凭据提交进 git 历史，
# 谁能读仓库谁就能用你的账号。key 按下面的顺序找：
#   1. 环境变量 WANDB_API_KEY（集群上建议用任务提交系统的 secret 注入）
#   2. $WANDB_KEY_FILE 指向的文件
#   3. ~/.wandb_key
#   4. 仓库根目录的 wandb.key（.gitignore 里的 `*.key` 已覆盖，不会被提交）
WANDB_ENABLE=true
WANDB_PROJECT="robo_contrast"
WANDB_ENTITY=""

export LEROBOT_VIDEO_DECODER_CACHE_SIZE=256
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

# ---------------------------------------------------------------- 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --nproc_per_node) NPROC_PER_NODE="$2"; shift 2 ;;
        --node_rank) NODE_RANK="$2"; shift 2 ;;
        --master_addr) MASTER_ADDR="$2"; shift 2 ;;
        --master_port) MASTER_PORT="$2"; shift 2 ;;
        --job_name) JOB_NAME="$2"; shift 2 ;;
        --job_type) JOB_TYPE="$2"; shift 2 ;;
        --data_mix) DATA_MIX="$2"; shift 2 ;;
        --task_type) TASK_TYPE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --optimizer_lr) OPTIMIZER_LR="$2"; shift 2 ;;
        --scheduler_decay_lr) SCHEDULER_DECAY_LR="$2"; shift 2 ;;
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
        --gradient_acc|--gradient_accumulation_steps)
            GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
        --save_checkpoint) SAVE_CHECKPOINT="$2"; shift 2 ;;
        --weight_resume) WEIGHT_RESUME="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        --deepspeed_config) DEEPSPEED_CONFIG="$2"; shift 2 ;;
        --dataset_len) DATASET_SIZE_ONE_EPOCH="$2"; shift 2 ;;
        --sample_ratio) SAMPLE_RATIO="$2"; shift 2 ;;
        --video_backend) VIDEO_BACKEND="$2"; shift 2 ;;
        --window_mode) WINDOW_MODE="$2"; shift 2 ;;
        --chunk_size) CHUNK_SIZE="$2"; shift 2 ;;
        --n_action_steps) N_ACTION_STEPS="$2"; shift 2 ;;
        --group_size) GROUP_SIZE="$2"; shift 2 ;;
        --chunk_seconds) CHUNK_SECONDS="$2"; shift 2 ;;
        --chunk_frames_min) CHUNK_FRAMES_MIN="$2"; shift 2 ;;
        --chunk_frames_max) CHUNK_FRAMES_MAX="$2"; shift 2 ;;
        --frame_horizon) FRAME_HORIZON="$2"; shift 2 ;;
        --vision_backbone) VISION_BACKBONE="$2"; shift 2 ;;
        --perception_recon_target|--recon_target)
            PERCEPTION_RECON_TARGET="$2"; shift 2 ;;
        --num_cls_tokens) NUM_CLS_TOKENS="$2"; shift 2 ;;
        --num_change_queries) NUM_CHANGE_QUERIES="$2"; shift 2 ;;
        --num_evidence_layers) NUM_EVIDENCE_LAYERS="$2"; shift 2 ;;
        --num_fusion_layers) NUM_FUSION_LAYERS="$2"; shift 2 ;;
        --num_predictor_layers) NUM_PREDICTOR_LAYERS="$2"; shift 2 ;;
        --num_physical_layers) NUM_PHYSICAL_LAYERS="$2"; shift 2 ;;
        --perception_recon_weight) PERCEPTION_RECON_WEIGHT="$2"; shift 2 ;;
        --vae_repeat_frames) VAE_REPEAT_FRAMES="$2"; shift 2 ;;
        --patch_token_stride) PATCH_TOKEN_STRIDE="$2"; shift 2 ;;
        --freeze_vision) FREEZE_VISION_ENCODER="$2"; shift 2 ;;
        --freeze_text) FREEZE_TEXT_ENCODER="$2"; shift 2 ;;
        --perception_only) PERCEPTION_ONLY="$2"; shift 2 ;;
        --perception_camera_mode) PERCEPTION_CAMERA_MODE="$2"; shift 2 ;;
        --query_probe_freq) QUERY_PROBE_FREQ="$2"; shift 2 ;;
        --tactile_backbone) TACTILE_BACKBONE="$2"; shift 2 ;;
        --tactile_frames) TACTILE_FRAMES="$2"; shift 2 ;;
        --tactile_tokens_per_pad) TACTILE_TOKENS_PER_PAD="$2"; shift 2 ;;
        --max_tactile_views) MAX_TACTILE_VIEWS="$2"; shift 2 ;;
        --tactile_dead_std) TACTILE_DEAD_STD="$2"; shift 2 ;;
        --tactile_pretrained) TACTILE_PRETRAINED="$2"; shift 2 ;;
        --tactile_recon_weight) TACTILE_RECON_WEIGHT="$2"; shift 2 ;;
        --tactile_lr_scale) TACTILE_LR_SCALE="$2"; shift 2 ;;
        --ftp1_tactile_dir) FTP1_TACTILE_DIR="$2"; shift 2 ;;
        --ftp1_tactile_sensors) FTP1_TACTILE_SENSORS="$2"; shift 2 ;;
        --anytouch_checkpoint) ANYTOUCH_CHECKPOINT="$2"; shift 2 ;;
        --anytouch_forward_batch_size) ANYTOUCH_FORWARD_BATCH_SIZE="$2"; shift 2 ;;
        --gradient_checkpointing) GRADIENT_CHECKPOINTING="$2"; shift 2 ;;
        --modality_dropout_tactile) MODALITY_DROPOUT_TACTILE="$2"; shift 2 ;;
        --modality_dropout_state) MODALITY_DROPOUT_STATE="$2"; shift 2 ;;
        --modality_dropout_action) MODALITY_DROPOUT_ACTION="$2"; shift 2 ;;
        --same_dataset_frac) SAME_DATASET_FRAC="$2"; shift 2 ;;
        --episode_group_frac) EPISODE_GROUP_FRAC="$2"; shift 2 ;;
        --episode_group_size) EPISODE_GROUP_SIZE="$2"; shift 2 ;;
        --min_frame_gap) MIN_FRAME_GAP="$2"; shift 2 ;;
        --false_negative_frame_gap) FALSE_NEGATIVE_FRAME_GAP="$2"; shift 2 ;;
        --weights_root) WEIGHTS_ROOT="$2"; shift 2 ;;
        --parent_dir_v21) PARENT_DIR_V21="$2"; shift 2 ;;
        --parent_dir_v30) PARENT_DIR_V30="$2"; shift 2 ;;
        --parent_dir_extra) PARENT_DIR_EXTRA="$2"; shift 2 ;;
        --processor) PROCESSOR="$2"; shift 2 ;;
        --vision_model|--dinov3_dir) VISION_MODEL="$2"; shift 2 ;;
        --text_model) TEXT_MODEL="$2"; shift 2 ;;
        --cosmos3_dir) COSMOS3_DIR="$2"; shift 2 ;;
        --qwen3vl_dir) QWEN3VL_DIR="$2"; shift 2 ;;
        --log_dir) LOG_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --pre_path) PRETRAINED_PATH="$2"; shift 2 ;;
        --check_paths) CHECK_PATHS="$2"; shift 2 ;;
        --wandb_enable) WANDB_ENABLE="$2"; shift 2 ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --wandb_entity) WANDB_ENTITY="$2"; shift 2 ;;
        --dry_run) DRY_RUN=true; shift ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "未知参数: $1（用 --help 查看支持项）" >&2; exit 1 ;;
    esac
done

# 未单独覆盖的权重路径都从同一个根目录派生。这样集群用默认 /mnt，开发机只需
# `--weights_root /Data/lzl/huggingface`，仍可再用单模型参数覆盖某一个目录。
PROCESSOR="${PROCESSOR:-${WEIGHTS_ROOT}/InternVL3_5-2B-HF}"
VISION_MODEL="${VISION_MODEL:-${WEIGHTS_ROOT}/dinov3-vitb16-pretrain-lvd1689m}"
TEXT_MODEL="${TEXT_MODEL:-${WEIGHTS_ROOT}/siglip2-base-patch16-224}"
COSMOS3_DIR="${COSMOS3_DIR:-${WEIGHTS_ROOT}/Cosmos3-Edge}"
QWEN3VL_DIR="${QWEN3VL_DIR:-${WEIGHTS_ROOT}/Qwen3-VL-4B-Instruct}"
FTP1_TACTILE_DIR="${FTP1_TACTILE_DIR:-${WEIGHTS_ROOT}/ftp1_v0426_50kstep}"
ANYTOUCH_CHECKPOINT="${ANYTOUCH_CHECKPOINT:-${WEIGHTS_ROOT}/anytouch_encoder.pth}"

die() {
    echo "错误：$*" >&2
    exit 1
}

require_choice() {
    local name="$1"
    local value="$2"
    shift 2
    local candidate
    for candidate in "$@"; do
        [[ "$value" == "$candidate" ]] && return 0
    done
    die "${name}=${value}，可选值：$*"
}

require_bool() {
    [[ "$2" == "true" || "$2" == "false" ]] || die "$1 必须是 true 或 false，当前为 $2"
}

require_positive_int() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 必须是正整数，当前为 $2"
}

require_nonnegative_int() {
    [[ "$2" =~ ^[0-9]+$ ]] || die "$1 必须是非负整数，当前为 $2"
}

require_minus_one_or_positive_int() {
    [[ "$2" == "-1" || "$2" =~ ^[1-9][0-9]*$ ]] \
        || die "$1 必须是 -1 或正整数，当前为 $2"
}

require_dir() {
    [[ -d "$2" ]] || die "$1 目录不存在：$2"
}

require_file() {
    [[ -f "$2" ]] || die "$1 文件不存在：$2"
}

if [[ -z "$JOB_NAME" ]]; then
    die "必须指定 --job_name"
fi

require_choice "vision_backbone" "$VISION_BACKBONE" dinov3 cosmos3 qwen3vl
require_choice "perception_recon_target" "$PERCEPTION_RECON_TARGET" vision vae
require_choice "tactile_backbone" "$TACTILE_BACKBONE" resnet18 ftp1 anytouch
require_choice "window_mode" "$WINDOW_MODE" duration frames
require_choice "task_type" "$TASK_TYPE" train_contrastive train_perception
require_choice "perception_camera_mode" "$PERCEPTION_CAMERA_MODE" primary all

require_bool "freeze_vision" "$FREEZE_VISION_ENCODER"
require_bool "freeze_text" "$FREEZE_TEXT_ENCODER"
require_bool "perception_only" "$PERCEPTION_ONLY"
require_bool "tactile_pretrained" "$TACTILE_PRETRAINED"
require_bool "gradient_checkpointing" "$GRADIENT_CHECKPOINTING"
require_bool "save_checkpoint" "$SAVE_CHECKPOINT"
require_bool "weight_resume" "$WEIGHT_RESUME"
require_bool "resume" "$RESUME"
require_bool "wandb_enable" "$WANDB_ENABLE"
require_bool "check_paths" "$CHECK_PATHS"

require_positive_int "nnodes" "$NNODES"
require_positive_int "nproc_per_node" "$NPROC_PER_NODE"
require_nonnegative_int "node_rank" "$NODE_RANK"
require_positive_int "master_port" "$MASTER_PORT"
require_positive_int "batch_size" "$BATCH_SIZE"
require_positive_int "gradient_acc" "$GRADIENT_ACCUMULATION_STEPS"
require_positive_int "chunk_size" "$CHUNK_SIZE"
require_positive_int "group_size" "$GROUP_SIZE"
require_positive_int "n_action_steps" "$N_ACTION_STEPS"
require_positive_int "chunk_frames_min" "$CHUNK_FRAMES_MIN"
require_positive_int "chunk_frames_max" "$CHUNK_FRAMES_MAX"
require_positive_int "num_cls_tokens" "$NUM_CLS_TOKENS"
require_positive_int "num_change_queries" "$NUM_CHANGE_QUERIES"
require_nonnegative_int "num_evidence_layers" "$NUM_EVIDENCE_LAYERS"
require_nonnegative_int "num_fusion_layers" "$NUM_FUSION_LAYERS"
require_nonnegative_int "num_predictor_layers" "$NUM_PREDICTOR_LAYERS"
require_positive_int "vae_repeat_frames" "$VAE_REPEAT_FRAMES"
require_positive_int "patch_token_stride" "$PATCH_TOKEN_STRIDE"
require_positive_int "num_physical_layers" "$NUM_PHYSICAL_LAYERS"
require_positive_int "tactile_frames" "$TACTILE_FRAMES"
require_positive_int "tactile_tokens_per_pad" "$TACTILE_TOKENS_PER_PAD"
require_positive_int "max_tactile_views" "$MAX_TACTILE_VIEWS"
require_positive_int "anytouch_forward_batch_size" "$ANYTOUCH_FORWARD_BATCH_SIZE"
require_positive_int "episode_group_size" "$EPISODE_GROUP_SIZE"
require_nonnegative_int "min_frame_gap" "$MIN_FRAME_GAP"
require_nonnegative_int "false_negative_frame_gap" "$FALSE_NEGATIVE_FRAME_GAP"
require_positive_int "steps" "$STEPS"
require_positive_int "save_freq" "$SAVE_FREQ"
require_nonnegative_int "log_freq" "$LOG_FREQ"
require_nonnegative_int "eval_freq" "$EVAL_FREQ"
require_nonnegative_int "num_workers" "$NUM_WORKERS"
require_nonnegative_int "query_probe_freq" "$QUERY_PROBE_FREQ"
require_positive_int "dataset_len" "$DATASET_SIZE_ONE_EPOCH"
require_nonnegative_int "scheduler_warmup_steps" "$SCHEDULER_WARMUP_STEPS"
require_minus_one_or_positive_int "scheduler_decay_steps" "$SCHEDULER_DECAY_STEPS"
require_nonnegative_int "scheduler_platform_steps" "$SCHEDULER_PLATFORM_STEPS"
require_nonnegative_int "seed" "$SEED"
if [[ -n "$FRAME_HORIZON" ]]; then
    require_positive_int "frame_horizon" "$FRAME_HORIZON"
fi

(( NODE_RANK < NNODES )) || die "node_rank=${NODE_RANK} 必须小于 nnodes=${NNODES}"
(( MASTER_PORT <= 65535 )) || die "master_port=${MASTER_PORT} 超出有效端口范围"
if (( NNODES > 1 )) && [[ "$MASTER_ADDR" == "127.0.0.1" || "$MASTER_ADDR" == "localhost" ]]; then
    die "多机训练不能使用 master_addr=${MASTER_ADDR}，请指定 rank 0 节点可访问的地址"
fi
(( CHUNK_SIZE % GROUP_SIZE == 0 )) \
    || die "chunk_size=${CHUNK_SIZE} 必须能被 group_size=${GROUP_SIZE} 整除"
(( N_ACTION_STEPS <= CHUNK_SIZE )) \
    || die "n_action_steps=${N_ACTION_STEPS} 不能大于 chunk_size=${CHUNK_SIZE}"
(( CHUNK_FRAMES_MIN <= CHUNK_FRAMES_MAX )) \
    || die "chunk_frames_min=${CHUNK_FRAMES_MIN} 不能大于 chunk_frames_max=${CHUNK_FRAMES_MAX}"
(( NUM_CLS_TOKENS <= NUM_CHANGE_QUERIES )) \
    || die "num_cls_tokens=${NUM_CLS_TOKENS} 不能大于 num_change_queries=${NUM_CHANGE_QUERIES}"
(( MAX_TACTILE_VIEWS <= 6 )) \
    || die "max_tactile_views=${MAX_TACTILE_VIEWS} 超过数据层上限 6"
[[ "$TACTILE_TOKENS_PER_PAD" == "1" || "$TACTILE_TOKENS_PER_PAD" == "2" ]] \
    || die "tactile_tokens_per_pad 只能是 1 或 2"
(( TACTILE_FRAMES >= 2 )) || die "tactile_frames 至少为 2"
if [[ "$TACTILE_BACKBONE" == "anytouch" ]]; then
    (( TACTILE_FRAMES >= 3 )) || die "AnyTouch 至少需要 3 帧触觉图像"
    if [[ "$TACTILE_TOKENS_PER_PAD" == "2" ]]; then
        (( TACTILE_FRAMES >= 4 )) || die "AnyTouch 输出两个 token 时至少需要 4 帧"
    fi
fi

if [[ "$TASK_TYPE" == "train_perception" ]]; then
    PERCEPTION_ONLY=true
    TRAIN_ENTRYPOINT="lerobot/scripts/dps_train_perception.py"
elif [[ "$PERCEPTION_ONLY" == "true" ]]; then
    die "perception_only=true 时 task_type 必须是 train_perception"
else
    TRAIN_ENTRYPOINT="lerobot/scripts/dps_train_contrast.py"
fi

PREDICTOR_ENABLED="$(
    python - \
        "$OPTIMIZER_LR" "$SCHEDULER_DECAY_LR" "$WEIGHT_DECAY" \
        "$SAMPLE_RATIO" "$CHUNK_SECONDS" "$PERCEPTION_RECON_WEIGHT" \
        "$TACTILE_DEAD_STD" "$TACTILE_RECON_WEIGHT" "$TACTILE_LR_SCALE" \
        "$SAME_DATASET_FRAC" "$EPISODE_GROUP_FRAC" \
        "$MODALITY_DROPOUT_TACTILE" "$MODALITY_DROPOUT_STATE" \
        "$MODALITY_DROPOUT_ACTION" "$NUM_PREDICTOR_LAYERS" <<'PY'
import math
import sys

names = (
    "optimizer_lr",
    "scheduler_decay_lr",
    "weight_decay",
    "sample_ratio",
    "chunk_seconds",
    "perception_recon_weight",
    "tactile_dead_std",
    "tactile_recon_weight",
    "tactile_lr_scale",
    "same_dataset_frac",
    "episode_group_frac",
    "modality_dropout_tactile",
    "modality_dropout_state",
    "modality_dropout_action",
)
raw_values = sys.argv[1:-1]
try:
    values = {name: float(raw) for name, raw in zip(names, raw_values, strict=True)}
except ValueError as exc:
    print(f"错误：浮点参数无法解析：{exc}", file=sys.stderr)
    raise SystemExit(1)
if any(not math.isfinite(value) for value in values.values()):
    print("错误：浮点参数不能是 NaN 或 Inf", file=sys.stderr)
    raise SystemExit(1)
for name in ("optimizer_lr", "scheduler_decay_lr", "sample_ratio", "chunk_seconds"):
    if values[name] <= 0:
        print(f"错误：{name} 必须大于 0，当前为 {values[name]}", file=sys.stderr)
        raise SystemExit(1)
for name in (
    "weight_decay",
    "perception_recon_weight",
    "tactile_dead_std",
    "tactile_recon_weight",
    "tactile_lr_scale",
):
    if values[name] < 0:
        print(f"错误：{name} 必须大于等于 0，当前为 {values[name]}", file=sys.stderr)
        raise SystemExit(1)
for name in (
    "same_dataset_frac",
    "episode_group_frac",
    "modality_dropout_tactile",
    "modality_dropout_state",
    "modality_dropout_action",
):
    if not 0 <= values[name] <= 1:
        print(f"错误：{name} 必须位于 [0, 1]，当前为 {values[name]}", file=sys.stderr)
        raise SystemExit(1)
print(
    "true"
    if int(sys.argv[-1]) > 0 and values["perception_recon_weight"] > 0
    else "false"
)
PY
)"

if [[ "$TASK_TYPE" == "train_perception" && "$PREDICTOR_ENABLED" != "true" ]]; then
    die "train_perception 需要 num_predictor_layers > 0 且 perception_recon_weight > 0"
fi

FTP1_ALL_SENSORS=(
    SharpaWave
    OpenLoongVTouch
    GelSightMini
    MCTac
    ViTaMIn
    FreeTacMan
    exUMI
)
FTP1_ACTIVE_SENSORS=()
if [[ "$PERCEPTION_ONLY" != "true" && "$TACTILE_BACKBONE" == "ftp1" ]]; then
    if [[ -z "$FTP1_TACTILE_SENSORS" ]]; then
        FTP1_ACTIVE_SENSORS=("${FTP1_ALL_SENSORS[@]}")
    else
        IFS=',' read -r -a FTP1_ACTIVE_SENSORS <<< "$FTP1_TACTILE_SENSORS"
        for _index in "${!FTP1_ACTIVE_SENSORS[@]}"; do
            _sensor="${FTP1_ACTIVE_SENSORS[$_index]}"
            _sensor="${_sensor#"${_sensor%%[![:space:]]*}"}"
            _sensor="${_sensor%"${_sensor##*[![:space:]]}"}"
            [[ -n "$_sensor" ]] || die "ftp1_tactile_sensors 包含空名称"
            case "$_sensor" in
                SharpaWave|OpenLoongVTouch|GelSightMini|MCTac|ViTaMIn|FreeTacMan|exUMI) ;;
                *) die "未知 FTP-1 触觉传感器：${_sensor}" ;;
            esac
            FTP1_ACTIVE_SENSORS[$_index]="$_sensor"
        done
        FTP1_TACTILE_SENSORS="$(IFS=,; printf '%s' "${FTP1_ACTIVE_SENSORS[*]}")"
    fi
fi

if [[ "$CHECK_PATHS" == "true" ]]; then
    require_file "DeepSpeed config" "$DEEPSPEED_CONFIG"
    require_dir "dataset processor" "$PROCESSOR"
    require_dir "SigLIP2 text model" "$TEXT_MODEL"

    case "$VISION_BACKBONE" in
        dinov3)
            require_dir "DINOv3 model" "$VISION_MODEL"
            ;;
        cosmos3)
            require_dir "Cosmos3-Edge" "$COSMOS3_DIR"
            require_file "Cosmos3 config" "$COSMOS3_DIR/config.json"
            require_file "Cosmos3 vision weights" \
                "$COSMOS3_DIR/vision_encoder/model.safetensors"
            ;;
        qwen3vl)
            require_dir "Qwen3-VL model" "$QWEN3VL_DIR"
            require_file "Qwen3-VL config" "$QWEN3VL_DIR/config.json"
            shopt -s nullglob
            _qwen_shards=("$QWEN3VL_DIR"/*.safetensors)
            shopt -u nullglob
            (( ${#_qwen_shards[@]} > 0 )) \
                || die "Qwen3-VL 目录中没有 *.safetensors：$QWEN3VL_DIR"
            ;;
    esac

    if [[ "$PERCEPTION_RECON_TARGET" == "vae" && "$PREDICTOR_ENABLED" == "true" ]]; then
        require_dir "Cosmos3 VAE" "$COSMOS3_DIR/vae"
        require_file "Cosmos3 VAE config" "$COSMOS3_DIR/vae/config.json"
    fi

    if [[ "$PERCEPTION_ONLY" != "true" ]]; then
        case "$TACTILE_BACKBONE" in
            ftp1)
                require_dir "FTP-1 tactile model" "$FTP1_TACTILE_DIR"
                require_file "FTP-1 shared tactile weights" \
                    "$FTP1_TACTILE_DIR/hpt_tokenizer/shared_image_chunk_encoder.safetensors"
                for _sensor in "${FTP1_ACTIVE_SENSORS[@]}"; do
                    require_file "FTP-1 ${_sensor} weights" \
                        "$FTP1_TACTILE_DIR/hpt_tokenizer/${_sensor}_image_224_224_3.safetensors"
                done
                ;;
            anytouch)
                require_file "AnyTouch checkpoint" "$ANYTOUCH_CHECKPOINT"
                ;;
        esac
    fi

    if [[ -n "$PRETRAINED_PATH" ]]; then
        require_file "pre_path" "$PRETRAINED_PATH"
    fi
fi

# ---------------------------------------------------------------- W&B 凭据
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    for _candidate in "${WANDB_KEY_FILE:-}" "${HOME}/.wandb_key" "./wandb.key"; do
        if [[ -n "$_candidate" && -f "$_candidate" ]]; then
            WANDB_API_KEY="$(tr -d '[:space:]' < "$_candidate")"
            echo "wandb key <- ${_candidate}"
            break
        fi
    done
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"
if [[ "$WANDB_ENABLE" == "true" && -z "$WANDB_API_KEY" && "$DRY_RUN" == "false" ]]; then
    # 集群节点通常没跑过 `wandb login`，没有 key 会卡在交互式提示上直到任务超时，
    # 与其那样不如现在就说清楚。
    die "--wandb_enable true 但没找到 API key。请 export WANDB_API_KEY、写入 ~/.wandb_key，或关闭 W&B"
fi

# 分组构建参数，既便于 dry-run 检查，也避免某个可选权重漏传到 Draccus 配置。
PERCEPTION_ARGS=(
    --policy.vision_backbone="$VISION_BACKBONE"
    --policy.vision_model_name="$VISION_MODEL"
    --policy.text_model_name="$TEXT_MODEL"
    --policy.cosmos3_dir="$COSMOS3_DIR"
    --policy.qwen3vl_dir="$QWEN3VL_DIR"
    --policy.perception_recon_target="$PERCEPTION_RECON_TARGET"
    --policy.perception_recon_weight="$PERCEPTION_RECON_WEIGHT"
    --policy.vae_repeat_frames="$VAE_REPEAT_FRAMES"
    --policy.num_cls_tokens="$NUM_CLS_TOKENS"
    --policy.num_change_queries="$NUM_CHANGE_QUERIES"
    --policy.num_evidence_layers="$NUM_EVIDENCE_LAYERS"
    --policy.num_fusion_layers="$NUM_FUSION_LAYERS"
    --policy.num_predictor_layers="$NUM_PREDICTOR_LAYERS"
    --policy.patch_token_stride="$PATCH_TOKEN_STRIDE"
    --policy.freeze_vision_encoder="$FREEZE_VISION_ENCODER"
    --policy.freeze_text_encoder="$FREEZE_TEXT_ENCODER"
    --policy.perception_only="$PERCEPTION_ONLY"
    --policy.perception_camera_mode="$PERCEPTION_CAMERA_MODE"
    --policy.query_probe_freq="$QUERY_PROBE_FREQ"
)

PHYSICAL_ARGS=(
    --policy.window_mode="$WINDOW_MODE"
    --policy.chunk_size="$CHUNK_SIZE"
    --policy.n_action_steps="$N_ACTION_STEPS"
    --policy.group_size="$GROUP_SIZE"
    --policy.chunk_seconds="$CHUNK_SECONDS"
    --policy.chunk_frames_min="$CHUNK_FRAMES_MIN"
    --policy.chunk_frames_max="$CHUNK_FRAMES_MAX"
    --policy.num_physical_layers="$NUM_PHYSICAL_LAYERS"
    --policy.same_dataset_frac="$SAME_DATASET_FRAC"
    --policy.episode_group_frac="$EPISODE_GROUP_FRAC"
    --policy.episode_group_size="$EPISODE_GROUP_SIZE"
    --policy.min_frame_gap="$MIN_FRAME_GAP"
    --policy.false_negative_frame_gap="$FALSE_NEGATIVE_FRAME_GAP"
)
if [[ -n "$FRAME_HORIZON" ]]; then
    PHYSICAL_ARGS+=(--policy.frame_horizon="$FRAME_HORIZON")
fi

TACTILE_ARGS=(
    --policy.tactile_backbone="$TACTILE_BACKBONE"
    --policy.tactile_frames="$TACTILE_FRAMES"
    --policy.tactile_tokens_per_pad="$TACTILE_TOKENS_PER_PAD"
    --policy.max_tactile_views="$MAX_TACTILE_VIEWS"
    --policy.tactile_dead_std="$TACTILE_DEAD_STD"
    --policy.tactile_pretrained="$TACTILE_PRETRAINED"
    --policy.tactile_recon_weight="$TACTILE_RECON_WEIGHT"
    --policy.tactile_lr_scale="$TACTILE_LR_SCALE"
    --policy.gradient_checkpointing="$GRADIENT_CHECKPOINTING"
    --policy.modality_dropout_tactile="$MODALITY_DROPOUT_TACTILE"
    --policy.modality_dropout_state="$MODALITY_DROPOUT_STATE"
    --policy.modality_dropout_action="$MODALITY_DROPOUT_ACTION"
)
if [[ "$PERCEPTION_ONLY" != "true" ]]; then
    case "$TACTILE_BACKBONE" in
        ftp1)
            TACTILE_ARGS+=(--policy.ftp1_tactile_dir="$FTP1_TACTILE_DIR")
            if [[ -n "$FTP1_TACTILE_SENSORS" ]]; then
                TACTILE_ARGS+=(--policy.ftp1_tactile_sensors="$FTP1_TACTILE_SENSORS")
            fi
            ;;
        anytouch)
            TACTILE_ARGS+=(
                --policy.anytouch_checkpoint="$ANYTOUCH_CHECKPOINT"
                --policy.anytouch_forward_batch_size="$ANYTOUCH_FORWARD_BATCH_SIZE"
            )
            ;;
    esac
fi

# ---------------------------------------------------------------- deepspeed 配置
# micro batch 和 gradient accumulation 必须同时写进 DeepSpeed JSON。顶层同名参数只用于
# TrainPipelineConfig；只改其中一边会让日志与实际 global batch 不一致。
_safe_job_name="$(printf '%s' "$JOB_NAME" | tr -c 'A-Za-z0-9_.-' '_')"
DS_CONFIG="$(mktemp "/tmp/ds_zero2_contrast_${_safe_job_name}_${NODE_RANK}_XXXXXX.json")"
cleanup() {
    rm -f "$DS_CONFIG"
}
trap cleanup EXIT

python - "$DEEPSPEED_CONFIG" "$BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" "$DS_CONFIG" <<'PY'
import json, sys
source, micro_batch, grad_acc, output = sys.argv[1:]
with open(source) as f:
    cfg = json.load(f)
cfg["train_micro_batch_size_per_gpu"] = int(micro_batch)
cfg["gradient_accumulation_steps"] = int(grad_acc)
with open(output, "w") as f:
    json.dump(cfg, f, indent=4)
PY

GLOBAL_BATCH=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE * NNODES))
echo "distributed: nodes=${NNODES} rank=${NODE_RANK} gpus/node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "batch: micro=${BATCH_SIZE}/gpu grad_acc=${GRADIENT_ACCUMULATION_STEPS} global=${GLOBAL_BATCH}"
echo "perception: backbone=${VISION_BACKBONE} recon=${PERCEPTION_RECON_TARGET} predictor=${PREDICTOR_ENABLED} cls=${NUM_CLS_TOKENS} only=${PERCEPTION_ONLY}"
if [[ "$PERCEPTION_ONLY" == "true" ]]; then
    echo "physical: disabled"
else
    echo "physical: window=${WINDOW_MODE} chunk=${CHUNK_SIZE}/${GROUP_SIZE} tactile=${TACTILE_BACKBONE}"
fi
echo "weights: root=${WEIGHTS_ROOT}"
case "$VISION_BACKBONE" in
    dinov3) echo "  vision=${VISION_MODEL}" ;;
    cosmos3) echo "  vision=${COSMOS3_DIR}/vision_encoder" ;;
    qwen3vl) echo "  vision=${QWEN3VL_DIR}" ;;
esac
echo "  text=${TEXT_MODEL}"
if [[ "$PERCEPTION_RECON_TARGET" == "vae" && "$PREDICTOR_ENABLED" == "true" ]]; then
    echo "  vae=${COSMOS3_DIR}/vae"
fi
if [[ "$PERCEPTION_ONLY" != "true" ]]; then
    case "$TACTILE_BACKBONE" in
        ftp1) echo "  tactile=${FTP1_TACTILE_DIR}" ;;
        anytouch) echo "  tactile=${ANYTOUCH_CHECKPOINT}" ;;
    esac
fi

# ---------------------------------------------------------------- 执行训练命令
CMD=(
    # torchrun
    # --nnodes="$NNODES"
    # --nproc_per_node="$NPROC_PER_NODE"
    # --node_rank="$NODE_RANK"
    # --master_addr="$MASTER_ADDR"
    # --master_port="$MASTER_PORT"
    python "$TRAIN_ENTRYPOINT"
    --deepspeed="$DS_CONFIG"
    --policy.type="robo_contrast"
    "${PERCEPTION_ARGS[@]}"
    "${PHYSICAL_ARGS[@]}"
    "${TACTILE_ARGS[@]}"
    --policy.scheduler_warmup_steps="$SCHEDULER_WARMUP_STEPS"
    --policy.scheduler_decay_steps="$SCHEDULER_DECAY_STEPS"
    --policy.scheduler_platform_steps="$SCHEDULER_PLATFORM_STEPS"
    --policy.scheduler_decay_lr="$SCHEDULER_DECAY_LR"
    --policy.optimizer_weight_decay="$WEIGHT_DECAY"
    --policy.optimizer_lr="$OPTIMIZER_LR"
    --policy.pretrained_path="$PRETRAINED_PATH"
    --is_ft=false
    --dataset.repo_id="whatever"
    --dataset.image_transforms.enable=false
    --dataset.wrist_image_transforms.enable=false
    --dataset.wrist_image_transforms.is_primary=false
    --dataset.processor="$PROCESSOR"
    --dataset.parent_dir_v21="$PARENT_DIR_V21"
    --dataset.parent_dir_v30="$PARENT_DIR_V30"
    --dataset.parent_dir_extra="$PARENT_DIR_EXTRA"
    --dataset.video_backend="$VIDEO_BACKEND"
    --dataset.sample_ratio="$SAMPLE_RATIO"
    --dataset.dataset_size_one_epoch="$DATASET_SIZE_ONE_EPOCH"
    --data_mix="$DATA_MIX"
    --batch_size="$BATCH_SIZE"
    --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS"
    --seed="$SEED"
    --num_workers="$NUM_WORKERS"
    --steps="$STEPS"
    --save_checkpoint="$SAVE_CHECKPOINT"
    --save_freq="$SAVE_FREQ"
    --log_freq="$LOG_FREQ"
    --eval_freq="$EVAL_FREQ"
    --output_dir="$OUTPUT_DIR"
    --log_dir="$LOG_DIR"
    --task_type="$TASK_TYPE"
    --wandb.enable="$WANDB_ENABLE"
    --wandb.project="$WANDB_PROJECT"
    --job_name="$JOB_NAME"
    --weight_resume="$WEIGHT_RESUME"
    --resume="$RESUME"
)
if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb.entity="$WANDB_ENTITY")
fi
if [[ -n "$JOB_TYPE" ]]; then
    CMD+=(--job_type="$JOB_TYPE")
fi
CMD+=("${EXTRA_ARGS[@]}")

if [[ "$DRY_RUN" == "true" ]]; then
    printf 'command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}"
