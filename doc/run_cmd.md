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
schema、classic FSDP 和 distributed checkpoint API。
`DRY_RUN=true` 只检查命令拼接，因此跳过依赖检查。

Stage 2 的 distributed process group 默认超时为 60 分钟，而不是 PyTorch 默认的 10 分钟。
这是为了允许数十亿参数的 model shards 和每个 rank 的 AdamW8bit state 写入集群挂载盘。
如果挂载盘更慢，可以显式提高：

```bash
DISTRIBUTED_TIMEOUT_MINUTES=120 bash train_qwen3vl_mot_fsdp.sh
```

保存时日志会分别显示 `prepare save directory`、`save model shards`、
`save rank-local optimizer state`、`save checkpoint metadata` 和 `publish checkpoint` 的开始与
完成耗时。若再次卡住，最后一条 `phase started` 就是实际慢的阶段。

被 watchdog 终止的保存不是有效 checkpoint，通常会留下：

```text
${OUTPUT_DIR}/.checkpoint_00002000.tmp
```

launcher 现在会在训练前检测这类残留并立即退出，避免从头训练 2000 步后才发现目录冲突。
确认旧任务已经结束后，删除日志中列出的**具体临时目录**，或者改用新的 `JOB_NAME/OUTPUT_DIR`；
不要把 `.tmp` 目录改名成正式 checkpoint，也不能用它 resume。

不要把 `WANDB_API_KEY` 或其他凭据写入本文、launcher 或 AMLT YAML。集群运行时应通过
任务系统 secret 或环境变量注入。

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
2. pointer 不存在且目录中没有 `checkpoint_*` 时，自动改为 `WEIGHT_RESUME=false`，
   从 Stage 1 开始新训练；
3. pointer 指向有效的 `checkpoint_XXXXXXXX` 目录时恢复；
4. pointer 损坏，或已有 `checkpoint_*` 但 pointer 丢失时直接报错，不会静默重新训练。

因此新 `JOB_NAME/OUTPUT_DIR` 和已有 checkpoint 的任务可以使用同一条启动命令。

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
| `OUTPUT_DIR` | `${OUTPUT_ROOT}/${JOB_NAME}` | 当前任务 checkpoint 目录 |
| `LOG_DIR` | `/mnt/wangxiaofa/ace_logs` | 文本日志目录 |
| `NNODES` | `1` | 训练节点数 |
| `NODE_RANK` | `0` | 当前节点编号，范围 `[0, NNODES)` |
| `NPROC_PER_NODE` | `8` | 本节点 GPU 进程数 |
| `MASTER_ADDR` | 单节点为 `127.0.0.1` | 多节点 rank 0 可访问地址 |
| `MASTER_PORT` | 单节点自动选择 | 多节点共享 rendezvous 端口 |
| `BATCH_SIZE` | `16` | 每卡 micro-batch |
| `GRADIENT_ACCUMULATION_STEPS` | `1` | 梯度累积 |
| `STEPS` | `600000` | optimizer step 上限 |
| `FP8_ENABLED` | `true` | 是否转换 eligible Linear |
| `FP8_EMULATE` | `false` | 非 H100 上的功能模拟 |
| `TORCHRUN_BIN` | `torchrun` | 分布式启动程序 |
| `WEIGHT_RESUME` | `true` | 自动检测并恢复 `OUTPUT_DIR` 最新 checkpoint；不存在则新训练 |
| `CHECK_PATHS` | `true` | 启动前检查权重和数据路径 |
| `DRY_RUN` | `false` | 只打印最终命令 |