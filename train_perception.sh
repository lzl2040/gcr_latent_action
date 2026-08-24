#!/bin/bash
# Stage 1：感知分支的纯视觉预训练（robo_contrast），集群版。
# 本地版见 train_perception_local.sh；两者的训练超参默认值保持一致，差异只在集群相关的部分。
#
# 单机：
#   bash train_perception.sh --job_name my_run --nproc_per_node 8
# 多机（每个节点各跑一次，NODE_RANK 不同）：
#   bash train_perception.sh --job_name my_run --nnodes 2 --nproc_per_node 8 \
#        --node_rank 0 --master_addr 10.0.0.1 --master_port 29500
#
# `--` 之后的参数原样透传给训练脚本，用来临时覆盖任何这里没显式暴露的配置：
#   bash train_perception.sh --job_name my_run -- --policy.recon_loss_weight=0.5
#
# 训练完把权重交给 train_ace.sh：
#   --pre_path <OUTPUT_DIR>/<job_name>/<job_name>/<global_step_N>/mp_rank_00_model_states.pt
# 注意 job_name 出现**两次**：TrainPipelineConfig.validate() 已经把它拼进 output_dir，
# 训练脚本里又拼了一次（dps_train_contrast.py 同样如此）。路径难看但是确定的，
# 照抄启动时打印的那一行就不会错。
# 它只加载感知分支，物理分支留在初始化状态。
#
# 看日志里的 `qgain`，不要看 `loss`。frame t+H 的大部分内容从 frame t 就能预测出来，
# 所以无论 change query 是否携带了信息，loss 都会下降。`qgain` 衡量的是把一个样本的
# query 换成另一个样本的之后重建变差了多少；它要是贴着零，这次训练就没产出任何
# 第二阶段能用的东西——loss 曲线再好看也一样。
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
TASK_TYPE="train_perception"

# 优化器 / 调度器
OPTIMIZER_LR=1e-4
SCHEDULER_WARMUP_STEPS=500
SCHEDULER_DECAY_STEPS=25000
SCHEDULER_PLATFORM_STEPS=20000
WEIGHT_DECAY=1e-5

# 训练规模
STEPS=30_0000
SAVE_FREQ=2000
LOG_FREQ=20
# 阶段一没有下游评测可跑，本地版就是关掉的，这里保持一致。
# dps_train_perception.py 根本不读 eval_freq（阶段一没有下游评测可跑）。
# 保留这个参数只是为了和 train_ace.sh 的接口对齐，改它不会有任何效果。
EVAL_FREQ=0
NUM_WORKERS=12
DATASET_SIZE_ONE_EPOCH=1000000
SAMPLE_RATIO=5
# 每卡 micro batch。ds json 里写死了一个值，这里给了就临时改写一份配置，
# 不改动仓库里的共享 json。
BATCH_SIZE=""

# 时间窗口 / 模型结构
WINDOW_MODE="duration"
CHUNK_SIZE=32
GROUP_SIZE=4
CHUNK_SECONDS=1.6
# 每多少步跑一次 query 置换探针，也就是 qgain 的来源。调大省算力，但这是本阶段
# 唯一能判断训练是否有效的指标，不建议关掉。
QUERY_PROBE_FREQ=50
# primary = 每个数据集只读一路相机；all = 每一路非触觉相机都当成独立数据流。
# 阶段一要的就是数据量和视角多样性，all 能成倍放大可用视频，而且单样本解码成本不变
# （每个视角是独立的 dataset 对象，只解码自己那一路）。默认保守用 primary。
PERCEPTION_CAMERA_MODE="primary"

# 触觉塔的参数在这里一概没有：`--policy.perception_only=true` 会整个跳过物理分支，
# 触觉塔属于物理分支。要连触觉一起训是 train_ace.sh 的事。

# `--` 之后收集到这里，原样透传给训练脚本
EXTRA_ARGS=()

# 路径（集群挂载）
PARENT_DIR_V21="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/"
PARENT_DIR_V30="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/"
# 逗号分隔的额外根目录，给不在上面两个挂载点里的数据集用（例如 OpenNeoData）
PARENT_DIR_EXTRA="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/"
PROCESSOR="/mnt/wangxiaofa/pt_weights/InternVL3_5-2B-HF/"
# 这两个**必须**显式传。configuration_robo_contrast.py 里它们的默认值是开发机路径
# （/Data/lzl/huggingface/...），集群上不存在；本地版没传是因为在开发机上默认值刚好是对的。
# 漏了的话要等模型初始化到一半才炸。
VISION_MODEL="/mnt/wangxiaofa/pt_weights/dinov3-vitb16-pretrain-lvd1689m"
TEXT_MODEL="/mnt/wangxiaofa/pt_weights/siglip2-base-patch16-224"
LOG_DIR="/mnt/wangxiaofa/ace_logs"
OUTPUT_DIR="/mnt/wangxiaofa/perception_pretrain"

# 被抢占后从 output_dir/job_name 里的 checkpoint 续跑，同时恢复 optimizer、lr scheduler 和 step。
# 本地版默认 false（开发机上每次都想从头跑），集群上必须是 true，否则每被抢占一次就从 step 0 重来。
# 目录里没有 checkpoint 时它是安全的空操作。
WEIGHT_RESUME=true

# W&B。**不要把 API key 写进这个文件**——它是入库的，写进来等于把凭据提交进 git 历史，
# 谁能读仓库谁就能用你的账号。key 按下面的顺序找：
#   1. 环境变量 WANDB_API_KEY（集群上建议用任务提交系统的 secret 注入）
#   2. $WANDB_KEY_FILE 指向的文件
#   3. ~/.wandb_key
#   4. 仓库根目录的 wandb.key（.gitignore 里的 `*.key` 已覆盖，不会被提交）
WANDB_ENABLE=true
WANDB_PROJECT="robo_contrast"

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
        --query_probe_freq) QUERY_PROBE_FREQ="$2"; shift 2 ;;
        --camera_mode) PERCEPTION_CAMERA_MODE="$2"; shift 2 ;;
        --parent_dir_v21) PARENT_DIR_V21="$2"; shift 2 ;;
        --parent_dir_v30) PARENT_DIR_V30="$2"; shift 2 ;;
        --parent_dir_extra) PARENT_DIR_EXTRA="$2"; shift 2 ;;
        --processor) PROCESSOR="$2"; shift 2 ;;
        --vision_model) VISION_MODEL="$2"; shift 2 ;;
        --text_model) TEXT_MODEL="$2"; shift 2 ;;
        --log_dir) LOG_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --weight_resume) WEIGHT_RESUME="$2"; shift 2 ;;
        --wandb_enable) WANDB_ENABLE="$2"; shift 2 ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$JOB_NAME" ]]; then
    echo "错误：必须指定 --job_name"
    exit 1
fi

# checkpoint 目录是 output_dir/job_name/job_name —— job_name 出现两次不是笔误：
# TrainPipelineConfig.validate() 已经拼过一次，dps_train_perception.py 里又拼了一次
# （沿用 dps_train_contrast.py 的写法）。这里打印的是**真实**路径，交接权重给
# train_ace.sh 时照抄这一行再加 /<global_step_N>/mp_rank_00_model_states.pt。
#
# 另外：**job_name 相同的两次实验共用一个 checkpoint 目录**，而 WEIGHT_RESUME=true 会
# 让后起的那个直接加载前一个的权重和 optimizer 状态接着跑。
# 症状很隐蔽：任务不报错，日志里 "Resumed training from step N" 一闪而过，
# 最后两个实验报出一模一样的指标。做对比实验时每个配置必须给不同的 --job_name。
echo "checkpoints -> ${OUTPUT_DIR}/${JOB_NAME}/${JOB_NAME}  (weight_resume=${WEIGHT_RESUME})"

# ---------------------------------------------------------------- W&B 凭据
if [[ -z "${WANDB_API_KEY}" ]]; then
    for _candidate in "${WANDB_KEY_FILE}" "${HOME}/.wandb_key" "./wandb.key"; do
        if [[ -n "$_candidate" && -f "$_candidate" ]]; then
            WANDB_API_KEY="$(tr -d '[:space:]' < "$_candidate")"
            echo "wandb key <- ${_candidate}"
            break
        fi
    done
fi
export WANDB_API_KEY
if [[ "$WANDB_ENABLE" == "true" && -z "${WANDB_API_KEY}" ]]; then
    # 集群节点通常没跑过 `wandb login`，没有 key 会卡在交互式提示上直到任务超时，
    # 与其那样不如现在就说清楚。
    echo "错误：--wandb_enable true 但没找到 API key。"
    echo "      请任选一种：export WANDB_API_KEY=xxx / 写进 ~/.wandb_key / 写进 ./wandb.key"
    echo "      （或者加 --wandb_enable false 关掉）"
    exit 1
fi

# 视觉塔和文本塔的权重目录在集群上不存在的话，要等模型初始化到一半才炸，
# 那时候数据集已经扫完一遍了。不如现在就说清楚。
for _pair in "VISION_MODEL:$VISION_MODEL" "TEXT_MODEL:$TEXT_MODEL" "PROCESSOR:$PROCESSOR"; do
    _name="${_pair%%:*}"; _path="${_pair#*:}"
    if [[ ! -d "$_path" ]]; then
        echo "错误：${_name} 目录不存在: ${_path}"
        echo "      集群挂载点和开发机不同，用 --vision_model / --text_model / --processor 指定。"
        exit 1
    fi
done

# ---------------------------------------------------------------- deepspeed 配置
# 每卡 batch 只在 ds json 里生效（训练脚本从 json 读 train_micro_batch_size_per_gpu），
# 所以要改就复制一份改副本，避免多个并发任务互相污染仓库里的共享文件。
#
# 写入必须是原子的。集群按 rank 逐进程拉起这个脚本，同一节点上 8 个进程会写**同一个**
# 目标文件；json.dump 直接写会先清空再写，另一个进程正好读到的就是半截 JSON，
# 报一个和 batch size 毫无关系的 JSONDecodeError。先写临时文件再 os.replace，
# 重命名是原子的，读到的要么是旧内容要么是新内容，不会是半个。
DS_CONFIG="./ds_zero2_contrast.json"
if [[ -n "$BATCH_SIZE" ]]; then
    DS_CONFIG="/tmp/ds_zero2_contrast_${JOB_NAME}_${NODE_RANK}.json"
    python - "$BATCH_SIZE" "$DS_CONFIG" <<'PY'
import json, os, sys, tempfile
cfg = json.load(open("./ds_zero2_contrast.json"))
cfg["train_micro_batch_size_per_gpu"] = int(sys.argv[1])
target = sys.argv[2]
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target) or ".", suffix=".json")
with os.fdopen(fd, "w") as f:
    json.dump(cfg, f, indent=4)
os.replace(tmp, target)
PY
    echo "ds config -> ${DS_CONFIG} (micro batch ${BATCH_SIZE}/gpu)"
fi

# 注意：走 python 分支时下面这些值**只是打印出来给人看的**，一个都没传给训练进程。
# 真正决定分布式拓扑的是集群注入的 RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT
# 环境变量（deepspeed.init_distributed 读的是它们）。所以 --nproc_per_node 8 写成 4
# 也不会少起进程，别拿这行的输出去判断实际拓扑；要看日志里各 rank 自己报的 world_size。
# 只有取消 torchrun 注释之后这些参数才真正生效。
echo "[仅供参考，未传入] nodes=${NNODES} rank=${NODE_RANK} gpus/node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[实际生效] RANK=${RANK:-unset} WORLD_SIZE=${WORLD_SIZE:-unset} MASTER=${MASTER_ADDR:-unset}:${MASTER_PORT:-unset}"

# ---------------------------------------------------------------- 执行训练命令
# 启动方式与 train_ace.sh 保持一致：由集群自己按 rank 拉起进程，这里直接 python。
# 要退回 torchrun 自己拉起就取消下面的注释（两个脚本要一起改，不然两阶段的
# 进程模型不一致，多机时排查起来很麻烦）。
# torchrun \
#     --nnodes=$NNODES \
#     --nproc_per_node=$NPROC_PER_NODE \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
python lerobot/scripts/dps_train_perception.py \
    --deepspeed="$DS_CONFIG" \
    --policy.type="robo_contrast" \
    --policy.perception_only=true \
    --policy.vision_model_name="$VISION_MODEL" \
    --policy.text_model_name="$TEXT_MODEL" \
    --policy.window_mode=$WINDOW_MODE \
    --policy.chunk_size=$CHUNK_SIZE \
    --policy.group_size=$GROUP_SIZE \
    --policy.chunk_seconds=$CHUNK_SECONDS \
    --policy.query_probe_freq=$QUERY_PROBE_FREQ \
    --policy.perception_camera_mode=$PERCEPTION_CAMERA_MODE \
    --policy.scheduler_warmup_steps=$SCHEDULER_WARMUP_STEPS \
    --policy.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS \
    --policy.scheduler_platform_steps=$SCHEDULER_PLATFORM_STEPS \
    --policy.optimizer_weight_decay=$WEIGHT_DECAY \
    --policy.optimizer_lr=$OPTIMIZER_LR \
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
    --wandb.enable=$WANDB_ENABLE \
    --wandb.project="$WANDB_PROJECT" \
    --job_name="$JOB_NAME" \
    --weight_resume=$WEIGHT_RESUME \
    --resume=false \
    "${EXTRA_ARGS[@]}"
