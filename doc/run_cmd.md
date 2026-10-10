# Qwen3-VL MoT Stage 2 运行命令

本文记录 `qwen3vl_mot_stage2` 分支当前可用的集群和本地启动命令。

正式 H100 训练入口：

```text
train_qwen3vl_mot_fsdp.sh
```

该 launcher 默认使用当前环境中的 `torchrun`。

本地轻量 LoRA/DeepSpeed 入口：

```text
train_qwen3vl_mot_local.sh
```

两个脚本都需要先激活：

```bash
conda activate lerobot_v2
```

首次运行或基础镜像更新后，安装 Stage 2 已验证的 FP8/8-bit optimizer 版本：

```bash
python -m pip install --upgrade \
  "peft>=0.18,<0.19" \
  "torchao>=0.15,<0.16" \
  "bitsandbytes>=0.48,<0.51"
```

该命令**不会替换基础镜像中的 PyTorch/CUDA wheel**。launcher 当前支持两条明确路径：

| PyTorch | TorchAO | TorchAO 扩展模式 |
|---|---|---|
| `2.7.x` | `0.15.x` | 只使用 Python FP8 wrapper，自动跳过为 2.9.1 编译的 C++ extensions |
| `2.9.1` | `0.15.x` | 使用与 wheel 匹配的扩展 |

集群原有的 `torch 2.7.0 + torchao 0.16.0` 不属于支持组合；只需用上面的命令把
PEFT/TorchAO/bitsandbytes 调整到约束范围，不需要升级 PyTorch。不要仅为满足 PEFT 0.19
而保留 TorchAO 0.16；这会重新引入版本冲突。

`train_qwen3vl_mot_fsdp.sh` 在启动 `torchrun` 前会检查版本，并实际执行一次 CPU emulated FP8
前后向；检测到 H100/B200 时还会执行一次 native CUDA FP8 前后向。同时检查 `_scaled_mm`
schema、classic FSDP 和 distributed checkpoint API，并校验上传源码同时包含 AdamW8bit
签名兼容和 mounted-storage checkpoint I/O marker。
`DRY_RUN=true` 只检查命令拼接，因此跳过依赖检查。

Stage 2 的 distributed process group 默认超时为 60 分钟，而不是 PyTorch 默认的 10 分钟。
这是为了允许数十亿参数的 model shards 和每个 rank 的 AdamW8bit state 写入集群挂载盘。
如果挂载盘更慢，可以显式提高：

```bash
DISTRIBUTED_TIMEOUT_MINUTES=120 bash train_qwen3vl_mot_fsdp.sh
```

训练 tensor collective 继续使用 NCCL；DCP save plan、metadata object collective 和
checkpoint phase 状态汇总使用独立的 Gloo process group，避免把大量 CPU metadata 经 NCCL
暂存到 GPU。

默认输出位于 BlobFuse/共享挂载，因此 launcher 默认使用：

```bash
FSDP_CHECKPOINT_SYNC_FILES=false
FSDP_CHECKPOINT_THREADS=1
FSDP_CHECKPOINT_HEARTBEAT_SECONDS=60
```

`sync_files=false` 只关闭 PyTorch `FileSystemWriter` 在每个大型 `.distcp` 文件结尾执行的
`fsync()`；文件仍会正常写入并关闭。所有 rank 写完后，rank 0 会读取 DCP metadata，检查其中
引用的每个 shard 已可见且文件大小完整，然后才原子发布 checkpoint 和更新
`latest_checkpoint`。这避免 BlobFuse 上长时间卡在 `fsync()`，代价是节点或存储服务在文件
关闭后立刻故障时，没有 POSIX `fsync` 级别的持久性保证。若输出是可靠的本地 POSIX 文件系统，
可显式设置：

```bash
FSDP_CHECKPOINT_SYNC_FILES=true bash train_qwen3vl_mot_fsdp.sh
```

FSDP 使用默认 NCCL group 构造一维 DeviceMesh，使 sharded model state 使用 DTensor，而不是
Torch 2.7 的 legacy ShardedTensor。这样即使参数首维小于 world size、部分 rank 的 local
shard 为空，也能正常保存；不会再出现 `Only single local shard is supported`。此前已经完整
发布的 legacy ShardedTensor DCP checkpoint 仍可加载到新的 DTensor state dict。

保存日志会分别显示 `prepare save directory`、`materialize model state`、
`write model shards`、`verify model checkpoint`、`save rank-local optimizer state`、
`save checkpoint metadata` 和 `publish checkpoint` 的开始与完成耗时。耗时阶段每 60 秒输出
一次 heartbeat；写 model shard 时还会报告当前可见的 `.distcp` 文件数和总大小。因此：

- 卡在 `materialize model state`：问题位于 FSDP state-dict/CPU offload；
- 卡在 `write model shards` 且文件持续增长：问题是挂载盘吞吐；
- 卡在 `write model shards` 且没有文件：问题位于 DCP planning/control collective；
- 进入 `verify model checkpoint` 后失败：某些 shard 没有在共享挂载上完整可见。

被 watchdog 终止的保存不是有效 checkpoint，通常会留下：

```text
${OUTPUT_DIR}/.checkpoint_00002000.tmp
```

launcher 现在会在训练前检测这类残留并立即退出，避免从头训练 2000 步后才发现目录冲突。
确认旧任务已经结束后，删除日志中列出的**具体临时目录**，或者改用新的 `JOB_NAME/OUTPUT_DIR`；
不要把 `.tmp` 目录改名成正式 checkpoint，也不能用它 resume。

不要把 `WANDB_API_KEY` 或其他凭据写入本文、launcher 或 AMLT YAML。集群运行时应通过
任务系统 secret 或环境变量注入。

### 固定的稀疏样本起点池

Stage 1/Stage 2 的 contrastive dataset 默认会为每个 mixture 建立一个持久化的样本起点池。
它只稀疏训练窗口的起点 `t`，不会稀疏窗口内部的 action/state：

```text
30 fps, anchor stride = 6

sample 0: start=0, action/state=[0, 1, ..., 31]
sample 1: start=6, action/state=[6, 7, ..., 37]
sample 2: start=12, action/state=[12, 13, ..., 43]
```

默认规则为：

```text
true_fps <= 10: anchor_stride = 1
true_fps > 10 : anchor_stride = max(2, round(true_fps / 5))
```

因此 15/20/24/30 fps 对应的起点间隔分别为 3/4/5/6 帧。`window_mode=frames` 下每个
样本内部仍连续读取 32 帧 action/state；9 帧 world video 仍从同一个 `[t,t+31]` 窗口均匀
读取。

池不会保存数亿个 frame index，而是每个可用 episode 只保存一行：

```text
episode_start, anchor_count, cumulative_anchor_count
```

默认共享路径和节点本地 mmap cache 为：

```text
SAMPLE_POOL_ROOT=${OUTPUT_ROOT}/_sample_pools
SAMPLE_POOL_CACHE_DIR=${TMPDIR:-/tmp}/robo_contrast_sample_pools
```

rank 0 首次构建共享池，后续任务根据 fingerprint 直接复用；每个节点只把紧凑的
`episodes.npy` 复制到本地 cache，然后由本节点所有 rank mmap。fingerprint 包含 dataset
顺序、episode 边界、真实 fps、窗口 horizon 和采样规则，任一项变化都会建立新目录而不是
覆盖旧池。

可用配置：

```bash
SAMPLE_POOL_ENABLED=true
SAMPLE_POOL_KEEP_ALL_BELOW_FPS=10
SAMPLE_POOL_TARGET_HZ=5
SAMPLE_POOL_ROOT=/mnt/wangxiaofa/qwen3vl_mot_exp/_sample_pools
SAMPLE_POOL_CACHE_DIR=/tmp/robo_contrast_sample_pools
```

dataset mixture 权重和 source-equivalent epoch 现在按有效 anchor 数量计算，而不是继续按原始
帧数计算。FSDP checkpoint 的 training geometry 和 DeepSpeed client state 都记录 pool
fingerprint；Resume 时池发生变化会直接报错。若要恢复本功能加入之前保存的旧 checkpoint，
必须保持旧的数据几何：

```bash
SAMPLE_POOL_ENABLED=false bash train_qwen3vl_mot_fsdp.sh
```

新实验应保留默认的 `SAMPLE_POOL_ENABLED=true`。

### 固定 held-out episode 生成验证

Stage 2 FSDP launcher 可以从指定数据集各留出固定数量的**完整 episode**，训练 sample pool
不会再包含这些 episode 的任何窗口。训练 mixture 之外的数据集会以 validation-only source
加载，采样权重为 0，不会进入训练 batch。

新实验启用方式：

```bash
GENERATION_EVAL_ENABLED=true \
GENERATION_EVAL_FREQ=10000 \
GENERATION_EVAL_DATASETS=open_neo_arx5,ms_data_xdof_1,interna1_dual_arm_1,ftp_1_sharpa \
GENERATION_EVAL_EPISODES_PER_DATASET=2 \
GENERATION_EVAL_BATCH_SIZE=8 \
TACTILE_GENERATION_TARGET=spatial_patches \
bash train_qwen3vl_mot_fsdp.sh
```

split 由 seed、数据集名和 episode 边界确定，并写到：

```text
${OUTPUT_DIR}/generation_eval/split_manifest.json
```

每次评估写入独立的原子目录：

```text
${OUTPUT_DIR}/generation_eval/step_00010000/
├── index.html
├── metrics.json
└── <dataset>/episode_<id>_frame_<id>/
    ├── index.html
    ├── video_comparison.mp4
    ├── video_contact_sheet.png
    ├── action_comparison.png
    ├── action_summary.html
    └── tactile_comparison.png
```

- RGB 视频逐帧并排显示 `Ground truth | Prediction | Absolute difference`。
- Action 使用当前数据集自己的 mean/std 去归一化，只展示 `action_mask` 中有效的 canonical
  维度；若 metadata 提供 min/max，预测会裁剪到该数据集的有效物理范围，并在 HTML 中报告
  clip 比例。
- 触觉图像预测目标是 Stage 1 ResNet codec 的末帧空间 patch latent，位置编码为
  `(pad, row, column)`；输出再经冻结的 Stage 1 tactile decoder 解码，并按该 pad 的像素统计
  反归一化。
- 没有触觉图像的 episode 仍会输出 RGB 和 action；不会伪造 tactile artifact。

`GENERATION_EVAL_ENABLED` 默认关闭，避免无意中改变已有实验的数据几何。严格 held-out 会改变
训练 episode ranges、sample-pool fingerprint 和 checkpoint training geometry，因此**不能**
在已经训练过这些 episode 的旧 Stage 2 checkpoint 上直接开启后继续 resume。要得到严格未见
验证，必须使用新的 `JOB_NAME/OUTPUT_DIR` 从 Stage 1 启动新训练。旧 Stage 2 checkpoint 的
`tactile_generation_target=context_tokens` 也无法可靠解码触觉图像；新训练 launcher 默认使用
`spatial_patches`。

---

## 1. 集群默认路径

`train_qwen3vl_mot_fsdp.sh` 使用与 `train_ace.sh` 一致的集群挂载：

| 内容 | 默认路径 |
|---|---|
| 权重根目录 | `/mnt/wangxiaofa/pt_weights` |
| Qwen3-VL-4B | `/mnt/wangxiaofa/pt_weights/Qwen3-VL-4B-Instruct` |
| Cosmos3-Edge | `/mnt/wangxiaofa/pt_weights/Cosmos3-Edge` |
| Stage 1 | `/mnt/wangxiaofa/ace_stage1/step_16k` |
| v2.1/v3 主数据目录 | `/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/` |
| 额外数据目录 | `/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/` |
| Stage 2 输出根目录 | `/mnt/wangxiaofa/qwen3vl_mot_exp` |
| 日志目录 | `/mnt/wangxiaofa/ace_logs` |

若实际 Stage 1 权重放在其他位置，只需要覆盖 `STAGE1_CHECKPOINT`。launcher 默认会检查
模型、数据和 Stage 1 路径，缺失时在创建分布式进程前直接报错。

---

## 2. 集群 8×H100 正式训练

先在单个 8×H100 节点进入代码目录，然后运行：

```bash
conda activate lerobot_v2
cd /path/to/gcr_latent_action

JOB_NAME=qwen3vl_mot_stage2_full \
STAGE1_CHECKPOINT=/mnt/wangxiaofa/ace_stage1/step_16k \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
BATCH_SIZE=16 \
GRADIENT_ACCUMULATION_STEPS=1 \
UNDERSTANDING_TUNING_MODE=full \
UNDERSTANDING_LR_SCALE=0.05 \
FP8_ENABLED=true \
FP8_EMULATE=false \
WANDB_ENABLE=true \
WANDB_PROJECT=lerobot-stage2 \
bash train_qwen3vl_mot_fsdp.sh
```

默认结果目录：

```text
/mnt/wangxiaofa/qwen3vl_mot_exp/qwen3vl_mot_stage2_full
```

该配置对应：

```text
8×H100
micro-batch/GPU = 16
global batch = 128
Qwen language transformer = full tuning
vision = last 4 blocks rank-16 LoRA
Generation Expert = full tuning
Physical Encoder = frozen
TorchAO FP8 = native H100
optimizer = bitsandbytes AdamW8bit
```

单节点时：

```text
NNODES=1
NODE_RANK=0
```

launcher 会自动使用 `torchrun --standalone`。

### 2.1 多节点

多节点时，每个节点各运行一次相同命令，并设置不同的 `NODE_RANK`：

```bash
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=<rank-0-node-address> \
MASTER_PORT=29500 \
NPROC_PER_NODE=8 \
bash train_qwen3vl_mot_fsdp.sh
```

第二个节点：

```bash
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=<rank-0-node-address> \
MASTER_PORT=29500 \
NPROC_PER_NODE=8 \
bash train_qwen3vl_mot_fsdp.sh
```

所有节点必须保持相同的：

- `NNODES`、`MASTER_ADDR` 和 `MASTER_PORT`；
- `NPROC_PER_NODE`、batch 和 FP8/FSDP 配置；
- `JOB_NAME/OUTPUT_DIR`；
- 代码版本和共享数据挂载。

多节点 global batch：

```text
BATCH_SIZE × NPROC_PER_NODE × NNODES × GRADIENT_ACCUMULATION_STEPS
```

如果 W&B secret 没有注入，先使用：

```bash
WANDB_ENABLE=false
```

避免任务等待交互式登录。

### 2.2 设置 Stage 2 任务比例

两个 launcher 都支持用逗号分隔的环境变量设置任务列表和相对权重：

```text
STAGE2_TASK_NAMES
STAGE2_TASK_WEIGHTS
```

可用任务名按默认顺序为：

```text
t2v,i2v,action_video_prediction,forward_dynamics,inverse_dynamics,action_prediction,state_prediction,tactile_prediction
```

先只训练当前 RGB 条件的视频生成和触觉图像生成时，推荐**保留完整任务列表**，把暂不训练的
任务设为 0：

```bash
STAGE2_TASK_NAMES=t2v,i2v,action_video_prediction,forward_dynamics,inverse_dynamics,action_prediction,state_prediction,tactile_prediction \
STAGE2_TASK_WEIGHTS=0,0.7,0,0,0,0,0,0.3 \
TACTILE_GENERATION_TARGET=spatial_patches \
JOB_NAME=qwen3vl_mot_rgb_tactile \
bash train_qwen3vl_mot_fsdp.sh
```

权重是相对值，`0,7,0,0,0,0,3` 与上面的 `0,0.7,...,0.3` 等价。模型会先屏蔽当前 global
batch 中没有有效监督的任务，再对剩余权重重新归一化；因此没有触觉图像的 batch 不会误采样
`tactile_prediction`，其概率会转给当前可训练的 RGB 任务。

若还要训练纯文本生成 RGB，可使用：

```bash
STAGE2_TASK_WEIGHTS=0.1,0.6,0,0,0,0,0,0.3
```

`action_video_prediction` 是独立的联合生成任务：当前 RGB 和 available text 进入
Understanding，当前 state 作为干净 Generation 条件，未来 RGB 与完整 action chunk 同时
加噪并在同一次 Generation forward 中预测。例如将 40% step 分配给该任务：

```bash
STAGE2_TASK_WEIGHTS=0,0.3,0.4,0,0,0,0,0.3
```

任务列表决定 Generation Expert 的 task embedding 形状。恢复 checkpoint 时可以在保持
`STAGE2_TASK_NAMES` 完全一致的前提下修改 `STAGE2_TASK_WEIGHTS`；不能增加、删除或重排任务。
因此多阶段 curriculum 应从第一阶段就保留后续任务并先把其权重设为 0。若确实需要改变任务
列表，必须使用新的 `OUTPUT_DIR` 从 Stage 1 checkpoint 开始训练。两个变量必须同时设置；
不设置时，新训练使用 policy 默认配比（联合任务权重为 0），恢复训练使用 checkpoint 中保存
的比例。
在 `action_video_prediction` 加入前创建的七任务 checkpoint 没有该 task embedding，不能通过
resume 直接启用新任务；它仍可按原七任务列表继续恢复训练。

---

## 3. 集群单步 smoke

正式长跑前先确认路径、真实数据、checkpoint、FP8、FSDP 和 AdamW8bit 能一起运行：

```bash
conda activate lerobot_v2
cd /path/to/gcr_latent_action

JOB_NAME=qwen3vl_mot_stage2_smoke \
STAGE1_CHECKPOINT=/mnt/wangxiaofa/ace_stage1/step_16k \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
SAMPLES_PER_EPOCH=128 \
NUM_WORKERS=4 \
STEPS=1 \
SAVE_FREQ=1 \
WANDB_ENABLE=false \
bash train_qwen3vl_mot_fsdp.sh
```

只检查最终命令和路径解析，不启动训练：

```bash
DRY_RUN=true \
JOB_NAME=qwen3vl_mot_stage2_dry_run \
STAGE1_CHECKPOINT=/mnt/wangxiaofa/ace_stage1/step_16k \
bash train_qwen3vl_mot_fsdp.sh
```

路径尚未挂载但只想检查参数拼接时，可以额外设置：

```bash
CHECK_PATHS=false DRY_RUN=true bash train_qwen3vl_mot_fsdp.sh
```

---

## 4. 从 FSDP checkpoint 恢复

launcher 默认：

```bash
WEIGHT_RESUME=true
```

这是自动恢复模式：

1. 检查 `${OUTPUT_DIR}/latest_checkpoint`；
2. 若该位置没有 pointer，但旧版双层目录
   `${OUTPUT_DIR}/${JOB_NAME}/latest_checkpoint` 存在，则自动切换到旧目录恢复；
3. pointer 不存在且目录中没有 `checkpoint_*` 时，自动改为 `WEIGHT_RESUME=false`，
   从 Stage 1 开始新训练；
4. pointer 指向有效的 `checkpoint_XXXXXXXX` 目录时恢复；
5. pointer 损坏，或已有 `checkpoint_*` 但 pointer 丢失时直接报错，不会静默重新训练。

因此新 `JOB_NAME/OUTPUT_DIR` 和已有 checkpoint 的任务可以使用同一条启动命令。

2026-10-10 之前的 Stage 2 FSDP 入口把 job name 拼了两次：launcher 先构造
`${OUTPUT_ROOT}/${JOB_NAME}`，通用 `TrainPipelineConfig.validate()` 又追加一次，因此权重实际
位于 `${OUTPUT_ROOT}/${JOB_NAME}/${JOB_NAME}`，而 launcher 曾错误地在上一层检查 resume。
当前 FSDP config 将 `OUTPUT_DIR` 视为最终目录，不再追加第二次，并保留上述旧目录自动探测，
已有 checkpoint 不需要搬迁。

恢复时必须保持以下训练几何不变：

- GPU/world size；
- per-rank batch size；
- gradient accumulation；
- FSDP 配置；
- FP8 scope 和 recipe；
- sampler/data mixture 配置。

使用与原任务相同的 `JOB_NAME` 或显式传入同一个 `OUTPUT_DIR`：

```bash
conda activate lerobot_v2
cd /path/to/gcr_latent_action

JOB_NAME=qwen3vl_mot_stage2_full \
OUTPUT_DIR=/mnt/wangxiaofa/qwen3vl_mot_exp/qwen3vl_mot_stage2_full \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
BATCH_SIZE=16 \
GRADIENT_ACCUMULATION_STEPS=1 \
FP8_ENABLED=true \
FP8_RECIPE=rowwise_with_gw_hp \
FP8_SCOPE=generation_vlm \
WANDB_ENABLE=true \
bash train_qwen3vl_mot_fsdp.sh
```

检测到 checkpoint 并恢复时，不重新加载 Stage 1 checkpoint；Stage 1 结构已经嵌入
Stage 2 checkpoint 配置。

如果输出目录已经有 checkpoint，但明确要求从头开始：

```bash
WEIGHT_RESUME=false bash train_qwen3vl_mot_fsdp.sh
```

此时应使用新的 `OUTPUT_DIR`，避免后续保存与旧 checkpoint 名称冲突。

---

## 5. 本地 4×A6000 FSDP 单步调试

该命令走与 H100 正式训练相同的 classic FSDP + AdamW8bit 路径。A6000 没有原生 FP8，
因此必须设置 `FP8_EMULATE=true`；它只能验证功能，不能估计 H100 FP8 性能。

```bash
conda activate lerobot_v2
cd /home/v-wangxiaofa/lzl/gcr_latent_action

JOB_NAME=qwen3vl_mot_local_fsdp_smoke \
WEIGHTS_ROOT=/Data/lzl/huggingface \
STAGE1_CHECKPOINT=/Data/lzl/ace_stage1/step_16k \
PARENT_DIR_V21=/Data/lerobot_data_ort6d \
PARENT_DIR_V30=/Data/lerobot_data_ort6d/v30 \
PARENT_DIR_EXTRA='' \
OUTPUT_DIR=/Data/lzl/qwen3vl_mot_local_fsdp_smoke \
LOG_DIR=/Data/lzl/qwen3vl_mot_logs \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
FP8_ENABLED=true \
FP8_EMULATE=true \
SAMPLES_PER_EPOCH=128 \
NUM_WORKERS=4 \
STEPS=1 \
SAVE_FREQ=1 \
WANDB_ENABLE=false \
bash train_qwen3vl_mot_fsdp.sh
```

本地完整 40-step 稳定性检查只需将：

```bash
STEPS=40
SAVE_FREQ=40
```

其余参数保持不变。

---

## 6. 本地 LoRA/DeepSpeed 调试

如果只需要较轻的本地功能调试，而不是复现 H100 FSDP/FP8 路径：

```bash
conda activate lerobot_v2
cd /home/v-wangxiaofa/lzl/gcr_latent_action

STAGE1_CHECKPOINT=/Data/lzl/ace_stage1/step_16k \
CUDA_VISIBLE_DEVICES=0,1 \
UNDERSTANDING_TUNING_MODE=lora \
UNDERSTANDING_TEXT_LORA_LAYERS=0 \
UNDERSTANDING_VISION_LORA_LAYERS=4 \
UNDERSTANDING_LORA_RANK=16 \
UNDERSTANDING_LR_SCALE=0.1 \
SAMPLES_PER_EPOCH=128 \
NUM_WORKERS=4 \
STEPS=1 \
SAVE_FREQ=1 \
OUTPUT_DIR=/Data/lzl/qwen3vl_mot_local_lora_smoke \
WANDB_ENABLE=false \
bash train_qwen3vl_mot_local.sh
```

这条命令使用 DeepSpeed ZeRO-2 BF16，不用于判断 H100 原生 FP8 吞吐。

---

## 7. AMLT 提交和查看任务

当前工作区使用 `doc/amlt_example.yaml` 作为 Stage 2 AMLT 提交配置。该文件包含
环境专属配置并保持未跟踪；若使用其他 AMLT YAML，job 的 command 同样应调用：

```bash
bash train_qwen3vl_mot_fsdp.sh
```

并通过 AMLT 环境变量注入本节前面的 `JOB_NAME`、`STAGE1_CHECKPOINT`、batch 和 W&B
配置。提交已有 YAML 的标准命令是：

```bash
amlt run <config.yaml> :<job-name> <experiment-name> \
  --description "**Qwen3-VL MoT Stage 2**: 8xH100 FSDP + FP8 + AdamW8bit"
```

修改代码后不要直接使用 `amlt rerun`：它默认不重新上传代码，会继续执行旧的
`/scratch/amlt_code/...` 快照。应使用新的 job 名重新 `amlt run`；若要强制覆盖代码缓存：

```bash
amlt run <config.yaml> :<job-name>=<new-job-name> <experiment-name> \
  --no-md5 \
  --description "**Qwen3-VL MoT Stage 2**: upload latest dependency compatibility fix"
```

新快照启动时日志必须包含：

```text
Stage 2 source check: optimizer=.../lerobot/common/optim/optimizers.py adamw8bit_compat=1
```

没有这行就不是包含 AdamW8bit 兼容修复的 launcher。

提交后等待后端接收，再检查状态：

```bash
sleep 180
amlt status <experiment-name>
```

查看指定 job 最近 50 行日志：

```bash
amlt logs view -n 50 <experiment-name> :<job-name>
```

查看更详细的后端信息：

```bash
amlt show <experiment-name> :<job-name>
```

不要使用 `amlt logs tail -f` 写进自动化脚本，它会持续阻塞。

AMLT/Singularity 使用 launcher 自己生成每节点 GPU 进程时，应保持：

```yaml
process_count_per_node: 1
```

AMLT 会为这一个外层进程提供：

```text
NODE_RANK
MASTER_ADDR
MASTER_PORT
```

job command 还需要把 YAML 的节点数传给 launcher：

```yaml
- conda run --name lerobot env NNODES=$NODES NPROC_PER_NODE=$GPUS \
    JOB_NAME=$JOB_NAME bash train_qwen3vl_mot_fsdp.sh
```

launcher 在 `NNODES=1` 时使用 `--standalone`；在 `NNODES>1` 时改用 static rendezvous，
由所有节点共同组成 `NNODES × NPROC_PER_NODE` 的全局进程组。

---

## 8. 常用覆盖参数

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `WEIGHTS_ROOT` | `/mnt/wangxiaofa/pt_weights` | Qwen/Cosmos 权重根目录 |
| `STAGE1_CHECKPOINT` | `/mnt/wangxiaofa/ace_stage1/step_16k` | Stage 1 权重目录 |
| `PARENT_DIR_V21` | 集群 v30-0710 挂载 | v2.1 数据目录 |
| `PARENT_DIR_V30` | 集群 v30-0710 挂载 | v3 数据目录 |
| `PARENT_DIR_EXTRA` | 集群额外数据挂载 | 额外数据目录；允许设为空 |
| `OUTPUT_ROOT` | `/mnt/wangxiaofa/qwen3vl_mot_exp` | 默认输出根目录 |
| `OUTPUT_DIR` | `${OUTPUT_ROOT}/${JOB_NAME}` | 当前任务的最终 checkpoint 目录；旧双层目录会自动探测 |
| `LOG_DIR` | `/mnt/wangxiaofa/ace_logs` | 文本日志目录 |
| `NNODES` | `1` | 训练节点数 |
| `NODE_RANK` | `0` | 当前节点编号，范围 `[0, NNODES)` |
| `NPROC_PER_NODE` | `8` | 本节点 GPU 进程数 |
| `MASTER_ADDR` | 单节点为 `127.0.0.1` | 多节点 rank 0 可访问地址 |
| `MASTER_PORT` | 单节点自动选择 | 多节点共享 rendezvous 端口 |
| `DISTRIBUTED_TIMEOUT_MINUTES` | `60` | NCCL 和 checkpoint Gloo collective 超时 |
| `BATCH_SIZE` | `16` | 每卡 micro-batch |
| `GRADIENT_ACCUMULATION_STEPS` | `1` | 梯度累积 |
| `STEPS` | `600000` | optimizer step 上限 |
| `SAVE_FREQ` | `2000` | 每隔多少 optimizer step 保存一次 |
| `GENERATION_EVAL_ENABLED` | `false` | 是否启用固定 held-out episode 生成验证 |
| `GENERATION_EVAL_FREQ` | `10000` | 每隔多少 optimizer step 生成一次 artifact |
| `GENERATION_EVAL_DATASETS` | 四个固定数据集 | 逗号分隔的验证数据集名 |
| `GENERATION_EVAL_EPISODES_PER_DATASET` | `2` | 每个数据集完整留出的 episode 数 |
| `GENERATION_EVAL_BATCH_SIZE` | `8` | 每个 rank 同步执行的生成验证 batch |
| `GENERATION_EVAL_VIDEO_FPS` | `8` | 导出 MP4 的播放帧率 |
| `STAGE2_TASK_NAMES` | 未设置 | 逗号分隔的任务列表；恢复时必须与 checkpoint 完全一致 |
| `STAGE2_TASK_WEIGHTS` | 未设置 | 与任务列表等长的非负相对权重；global batch 内按可用监督重归一化 |
| `TACTILE_GENERATION_TARGET` | `spatial_patches` | 新训练使用可解码触觉空间 latent |
| `FSDP_CHECKPOINT_SYNC_FILES` | `false` | DCP 是否对每个 shard 强制 `fsync` |
| `FSDP_CHECKPOINT_THREADS` | `1` | 每个 rank 的 DCP writer 线程数 |
| `FSDP_CHECKPOINT_HEARTBEAT_SECONDS` | `60` | checkpoint 进度日志间隔；`0` 关闭 |
| `FP8_ENABLED` | `true` | 是否转换 eligible Linear |
| `FP8_EMULATE` | `false` | 非 H100 上的功能模拟 |
| `TORCHRUN_BIN` | `torchrun` | 分布式启动程序 |
| `WEIGHT_RESUME` | `true` | 自动检测并恢复 `OUTPUT_DIR` 最新 checkpoint；不存在则新训练 |
| `CHECK_PATHS` | `true` | 启动前检查权重和数据路径 |
| `DRY_RUN` | `false` | 只打印最终命令 |