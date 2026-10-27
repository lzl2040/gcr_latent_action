# Qwen3-VL MoT Stage 2 运行命令

本文记录 `qwen3vl_mot_pretrained_generation` 分支当前可用的集群和本地启动命令。

正式 A100/H100 训练入口：

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

## 最大训练 epoch

两个 launcher 都支持 `MAX_EPOCHS`，对应 Python CLI 的 `--max_epochs`：

```bash
MAX_EPOCHS=100 bash train_qwen3vl_mot_fsdp.sh
# 本地入口使用相同参数；保留原命令所需的权重、数据和输出目录配置
MAX_EPOCHS=100 bash train_qwen3vl_mot_local.sh
```

该值必须是正整数，表示从训练开始累计的 **sampler epoch 总上限**，
不是恢复 checkpoint 后再额外训练的轮数。显式设置会替代原有
`source_equivalent_epochs + 100` 计划，也可以小于一个完整源数据覆盖周期。
不传时保留原逻辑；LIBERO finetune 全覆盖模式通常为 `1 + 100 = 101` 轮。
预训练的 sampler epoch 按每轮采样量定义，不一定等于全源数据一遍；
`SAMPLES_PER_EPOCH` 控制采样量，不是最大 epoch，finetune 则自动使用有效 anchor 总数。

FSDP 仍同时受 `STEPS`（累计 optimizer update 上限，默认 `600000`）限制，
先达到 epoch 或 step 上限就停止。DeepSpeed 本地入口保持原行为，`STEPS` 不参与停止条件。
续训使用当前传入的 epoch 上限和已保存进度；若已达到上限，不再更新参数。
仅用 `WEIGHT_RESUME=true` 自动恢复权重时，仍需传入希望使用的 `MAX_EPOCHS`；
完整训练配置会保存 `max_epochs`，通过 `--config_path` 读取时保留保存值，
也可以用 `--max_epochs` 显式覆盖。修改参数需重新启动训练，不影响已运行进程。

## 预训练生成：Edge / Nano

两个 launcher 的新实验默认 `GENERATION_BACKBONE=edge`。理解侧仍是 Qwen3-VL-4B，
不加载或替换为 Cosmos 理解专家。生成侧严格加载原生生成 block、video input/output、
时间 embedding 和 normalization，再训练 Qwen K/V 条件 adapter 与新的物理/触觉接口。

本地权重：

```text
/Data/lzl/huggingface/Cosmos3-Edge
/Data/lzl/huggingface/Cosmos3-Nano
```

Nano 已下载完整 snapshot；生成 checkpoint 的 7 个 shards 合计约 30.35 GB，
其中也包含不会加载的原始理解侧参数。

本地新训练：

```bash
GENERATION_BACKBONE=edge \
  STAGE1_CHECKPOINT=/path/to/exported_robo_contrast \
  OUTPUT_DIR=qwen3vl_mot_pretrained_edge \
  JOB_NAME=qwen3vl_mot_pretrained_edge \
  bash train_qwen3vl_mot_local.sh
```

选择 Nano 只需将 `GENERATION_BACKBONE=nano` 并使用新的输出目录。
集群生成权重默认取 `${WEIGHTS_ROOT}/Cosmos3-Edge` 或 `${WEIGHTS_ROOT}/Cosmos3-Nano`；
可以用 `GENERATION_PRETRAINED_DIR` 指定其他根目录（必须包含 `transformer/` 与 `vae/`，
不是直接传 transformer 子目录）。

```bash
GENERATION_BACKBONE=nano \
  WEIGHT_RESUME=false \
  JOB_NAME=qwen3vl_mot_pretrained_nano \
  bash train_qwen3vl_mot_fsdp.sh
```

`COSMOS3_DIR` 仍控制 VAE 路径。官方 Edge/Nano VAE 权重相同，因此可以沿用 Edge VAE；
若自定义 VAE，初始化会检查结构与 latent statistics，不能静默换成不兼容 codec。
新生成模块无需安装新版 Diffusers，也不会改动当前 PyTorch/CUDA wheel。

`GENERATION_TUNING_MODE=full` 为默认；时间或 optimizer 显存紧张时可设置 `frozen`，
冻结复用的生成参数，只训练其条件 adapters、task/camera embeddings 与新的物理/触觉头。
理解侧的 full/LoRA/frozen 仍由原 `UNDERSTANDING_TUNING_MODE` 控制，默认不变。
冻结能减少 optimizer/梯度成本，不会省去生成骨干 forward，也不能保证收敛时间。
Edge 生成侧约 1.49B，Nano 约 7.07B（含新 adapters、1024D tactile、单视角）。

resume 时不设置 `GENERATION_BACKBONE`/`GENERATION_TUNING_MODE`，读取保存的模式。
原始 Cosmos transformer 权重只在 fresh run 加载，resume 不再读取、不覆盖已训练权重。
不能从旧 random checkpoint 直接 resume 成 Edge/Nano，也不能 Edge↔Nano 或 full↔frozen
直接续训；需新建 `JOB_NAME/OUTPUT_DIR`。复现旧结构的新实验可显式设
`GENERATION_BACKBONE=random`。旧 config 未保存新字段时仍恢复 random 架构。

预训练只提供生成先验，Qwen 条件和机器人新模态仍需适配训练，不保证零样本保持原生
Cosmos pipeline 的视频质量。现有 Euler sampler 与 task/multi-view/teacher 契约保留。
官方只测试 BF16；初次迁移建议 `FP8_ENABLED=false`，确认表现后再开启仓库的 FP8 路径。

FSDP 入口默认 `FP8_ENABLED=auto`：只有全部训练 rank 的 GPU compute capability
均不低于 9.0 才启用原生 FP8；A100（8.0）、A6000 等自动使用 BF16，并记录明确日志。
多卡统一选择，避免部分 rank 用 FP8、部分 rank 用 BF16。显式 `FP8_ENABLED=true`
仍要求原生 FP8 支持，在 A100 上会明确报错，不会静默切换或自动启用模拟。

A100 正式训练可以直接使用默认 `auto`，也可以显式设置：

```bash
FP8_ENABLED=false bash train_qwen3vl_mot_fsdp.sh
```

`FP8_EMULATE=true` 只用于功能正确性 smoke test，不是 A100 的高效 FP8 训练方案。
关闭 FP8 不改变 BF16 master weights、模型结构、task 或数据采样模式。

### Blob 缺少权重时自动下载到节点本地

两个 launcher 默认 `MODEL_AUTO_DOWNLOAD=true`。优先读取指定的模型目录，包括 Blob
挂载中的现有权重；配置、必要的 processor/tokenizer、index 所需 shards 不完整或
safetensors 文件被截断时，改为下载官方模型到**节点本地磁盘**，不修改原来的挂载目录。

默认缓存为 `${TMPDIR:-/tmp}/qwen3vl_mot_models`，可以指定本地 SSD/scratch：

```bash
MODEL_LOCAL_CACHE=/path/to/node-local-ssd/qwen3vl_mot_models \
GENERATION_BACKBONE=nano \
JOB_NAME=qwen3vl_mot_pretrained_nano \
bash train_qwen3vl_mot_fsdp.sh
```

`MODEL_LOCAL_CACHE` 不要设为 Blob、NFS 等共享挂载或 tmpfs；启动器会检查实际
filesystem（包括目录 symlink 指向的挂载），不允许下载到这些位置。每节点在启动
`torchrun`/DeepSpeed 前完成下载，用文件锁避免并发重复下载，该节点的全部 GPU 共用
同一份缓存。多节点分别缓存，不要求各节点通过共享目录等待下载完成。
多节点 rendezvous 的等待时间使用 `DISTRIBUTED_TIMEOUT_MINUTES`（默认 60 分钟），
避免已有缓存的节点过早超时；下载特别慢时可增大该值。

只下载训练实际需要的文件：Qwen 权重与 processor/tokenizer；Cosmos VAE；fresh
Edge/Nano 的 transformer shards 与配套 VAE 配置，不下载 Cosmos 原生视觉塔、音频等
无关组件。官方仓库 revision 固定以便复现。Hub 和 Xet 下载缓存也在本地目录，不受
指向 Blob 的 `HF_HOME` 影响。HF 登录信息照常读取；需要认证时将 `HF_TOKEN` 通过
集群 secret/环境变量注入，不要写到脚本或 YAML。

已有完整本地缓存可直接复用，无需联网；下载失败、磁盘空间不足或下载后检查失败
会终止启动，不会退回随机权重。Nano transformer 约 30.35 GB，另需预留 Qwen、VAE
和下载缓存的磁盘空间。自动处理不包括 Stage 1 私有 checkpoint，仍需提供它。

resume 只补齐当前构造流程需要的 Qwen 和 VAE，不下载原始生成 transformer、
不覆盖训练 checkpoint。FSDP `DRY_RUN=true` 不联网、不创建模型缓存。
设置 `MODEL_AUTO_DOWNLOAD=false` 可禁用自动下载，让缺失模型明确报错。

## LIBERO 下游微调：全部数据与训练内验证

两个入口默认 `TRAINING_STAGE=auto`：LIBERO mixture（包括四个子集或纯 LIBERO
自定义 mixture）自动使用 `finetune`，其他 mixture 保留原 `pretrain` 行为。
其他下游数据可显式指定 `TRAINING_STAGE=finetune`。

`finetune` 强制禁用持久化 sample pools，即使设置 `SAMPLE_POOL_ENABLED=true` 也不会
读取、创建或写入这些缓存。每个 epoch 打乱并遍历所有参与训练数据源的**每个有效完整
窗口起点**，不降 Hz、不按 mixture 相对权重重复抽样，也不受 `SAMPLES_PER_EPOCH`
限制。权重为零的数据源不参与训练。保留所有训练 episode；不足完整窗口的 episode
或末尾不足 32 帧的位置不作为起点。多卡按同一个全局顺序分片，仅最后一个 global
batch 为保持固定 batch size 可能重复开头的样本补齐。

FSDP 开启生成验证时，`GENERATION_EVAL_DATASETS=auto` 默认从训练 mixture 抽固定窗口；
也可指定 `libero`、`libero_object` 或逗号分隔的具体 dataset 名。mixture 名会展开为
实际子数据集，缺失子集仍会明确报错，而不是跳过。验证不会移除训练 episode：
manifest、metrics JSON 和 HTML 标为 **training / Training-set diagnostics**，
只能用于训练拟合诊断，不能当作独立 held-out 泛化结果或 LIBERO 环境成功率。

新实验示例（按前文配置 Stage 1 和模型/数据路径）：

```bash
DATA_MIX=libero \
TRAINING_STAGE=finetune \
GENERATION_EVAL_ENABLED=true \
GENERATION_EVAL_DATASETS=auto \
GENERATION_EVAL_FREQ=1000 \
WEIGHT_RESUME=false \
JOB_NAME=libero_finetune \
bash train_qwen3vl_mot_fsdp.sh
```

本地 DeepSpeed 入口使用同样的 `DATA_MIX` / `TRAINING_STAGE` 完整遍历设置；上述生成
验证开关属于 FSDP 入口。没有触觉的数据继续使用缺失模态 mask，无需改变模型结构。
显式 `TRAINING_STAGE=pretrain` 则保留原 sample pools 和隔离 held-out episode 的行为。

从旧 sample-pool 模式切换到完整遍历会改变 epoch 和采样游标，不能直接 resume 原
checkpoint；需要新的实验输出目录。新模式保存后的 checkpoint 可以在数据、batch、
world size 和 seed 不变时正常续训。这次改动不增加通用 Stage 2 weights-only 迁移入口。

## 多视角条件与未来视频生成

默认仍为 primary 单视角。新实验可开启：

```bash
RGB_VIEWS=primary,secondary,wrist \
  JOB_NAME=qwen3vl_mot_multiview \
  bash train_qwen3vl_mot_fsdp.sh
```

本地入口 `train_qwen3vl_mot_local.sh` 使用相同环境变量；只需 primary+wrist 时设置
`RGB_VIEWS=primary,wrist`。首次训练仍需按下面命令配置 Stage 1 checkpoint 和模型/数据路径。
相机来源是 `oxe_configs.py` 的有效映射，缺失视角不参与条件注意力和视频 loss。
预测视频的任务（含 `joint_prediction`）联合预测有效视角未来视频，
**latent-action teacher 仍只监督 primary 单视角**。

多视角增加 token、解码和显存成本；先调整 `BATCH_SIZE`，不要假设与单视角吞吐相同。
resume 不设置 `RGB_VIEWS` 时读取保存的列表，显式设置必须与 checkpoint 完全一致。
旧单视角 checkpoint 不能通过 resume 直接切换为多视角，需新建实验输出目录。
held-out 验证保留 primary 原文件名，secondary/wrist 的视频与 contact sheet 位于对应角色
子目录，HTML 展示各有效视角，metrics 同时包含逐视角与平均视频 MAE。

## 可选 RGB 数据增强

两个 Stage 2 launcher 默认关闭 RGB 增强，可用同一开关显式启用：

```bash
RGB_AUG_ENABLED=true \
  bash train_qwen3vl_mot_fsdp.sh
```

本地入口 `train_qwen3vl_mot_local.sh` 也支持。直接 Python CLI 对应
`--dataset.image_transforms.enable=true`；Stage 2 使用独立的轻量窗口增强 preset，
不套用通用单帧增强里的强 saturation/hue 或随机子集策略。

开启后每个有效相机窗口执行：面积约 `90%-100%`、不改变宽高比的随机裁剪并恢复原尺寸，
`[-5,5]` 度小角度旋转，以及 brightness/contrast `0.9-1.1`、
saturation `0.95-1.05` 的颜色抖动。没有翻转、90/180 度旋转或 hue 变化。
**同一相机的当前帧、结束帧和全部视频帧共用一次采样的增强参数**；
多视角各自采样，但 primary 的 `image_t0/image_t1/video` 与多视角字段保持一致。
缺失视角维持原来的零填充和 mask，不修改 action/state、语言或触觉。

只增强训练读取；生成验证的 `get_heldout_item` 始终读取未增强图像，
包括 finetune 从训练集抽取的固定诊断窗口。开启时视频条件和未来 GT target 同步增强，
不会把逐帧随机变化当作真实运动。该操作在 DataLoader 中批量处理整段窗口，
不增加模型参数；启用后仍有额外 CPU 开销。
resume 时同样需显式设置 `RGB_AUG_ENABLED=true` 才继续增强，默认不会自动开启。

## Physical embedding-space flow

两个 Stage 2 launcher 的新实验默认 `PHYSICAL_FLOW_SPACE=embedding`：

```bash
PHYSICAL_FLOW_SPACE=embedding \
  JOB_NAME=qwen3vl_mot_embedding_flow \
  bash train_qwen3vl_mot_fsdp.sh
```

保留冻结的 Physical Transformer，但只向它输入干净 state/action；分别编码各模态，
在输出的 `(B,8,1024)` embedding 上加噪和计算 flow loss。
Generation 预测 embedding velocity，state/action decoder 经预测 embedding 的重建损失
训练，最终输出 `(B,8,160)`，reshape 为 `(B,32,40)`。
action 推理先编码当前 state 一次，然后直接在 action embedding 空间做 Euler 更新，
不会每一步把带噪 action 重新送入 Physical Transformer。

新模式要求 `PHYSICAL_TUNING_MODE=frozen`，以保持 encoder 输出和 flow target 空间固定。
旧 checkpoint/config 未保存模式时默认为 `raw`；resume 不设置 `PHYSICAL_FLOW_SPACE`
就读取保存的模式，不能显式改模式续训，因为输出头、decoder 和 optimizer state 不兼容。
需要复现旧的原始空间 flow 时，新实验可显式设置 `PHYSICAL_FLOW_SPACE=raw`。
多视角、RGB flow、触觉 codec 和 primary-only teacher 的契约均保持不变。

embedding 实验默认启用预测 action/state 重建，两项初始权重均为 `0.1`：
计算 `z0_hat = z_sigma - sigma * v_pred` 后，通过对应 decoder 与真实归一化
canonical trajectory 计算 masked MSE。预测 embedding 不 detach，损失更新生成网络和 decoder；
state 只监督未来帧，不计算已知第 0 帧。原来解码真实干净 embedding 的 codec 损失已经移除。
日志为 `action_prediction_reconstruction_loss` / `state_prediction_reconstruction_loss`
（控制台缩写 `action_pred` / `state_pred`）。

```bash
ACTION_PRED_RECON_WEIGHT=0.1 \
STATE_PRED_RECON_WEIGHT=0.1 \
  bash train_qwen3vl_mot_fsdp.sh
```

设置对应权重为 `0` 可关闭；两项只适用于 embedding 模式，raw 默认关闭。
resume 保留已保存的权重；缺字段时 embedding 默认 `0.1`，raw 默认 `0`。
旧实验若 action prediction 权重保存为 `0`，移除 codec 后需显式设置正权重才能继续监督 action decoder。
旧字段 `physical_reconstruction_loss_weight` 读取时会警告并清空，不再用于训练。
inverse/joint/action_prediction 有 action 重建，joint/state_prediction 有未来 state 重建；
forward 不引入 state，也不重建干净 action 条件。flow loss 和多步推理采样器保持。

### 训练噪声时间分布

两个 launcher 都支持 `TIME_SAMPLING_DISTRIBUTION=uniform|beta|mixed|shifted_uniform`：

```bash
TIME_SAMPLING_DISTRIBUTION=mixed \
  bash train_qwen3vl_mot_fsdp.sh
```

未设置时 fresh 默认 `uniform`，resume 使用保存的分布。
`beta` 默认 `Beta(1.5,1)`；`mixed` 默认每个样本以 50% 概率选择 Beta，其余选择 Uniform，
不是将两个采样值求平均。所有分布映射到现有 `[sigma_min,sigma_max]`。
可选 `TIME_SAMPLING_BETA_ALPHA`、`TIME_SAMPLING_BETA_BETA`、
`TIME_SAMPLING_BETA_PROBABILITY`（mixed 中 Beta 占比，默认 `0.5`）调整参数。
直接 Python CLI 等价选项是 `--policy.time_sampling_distribution=mixed`。
resume 时显式设置这些参数可以覆盖保存的设置。

需要增加高噪声训练样本占比时，显式选择 `shifted_uniform`：

```bash
TIME_SAMPLING_DISTRIBUTION=shifted_uniform \
TIME_SAMPLING_SHIFT=3 \
  bash train_qwen3vl_mot_fsdp.sh
```

`train_qwen3vl_mot_local.sh` 接受相同环境变量。
先采样 `u ~ Uniform(0,1)`，再计算
`q = shift*u / (1 + (shift-1)*u)`，
最后映射 `sigma = sigma_min + (sigma_max-sigma_min)*q`。
`TIME_SAMPLING_SHIFT` 必须为有限正数，默认 `1`；
`shift=1` 等价于均匀分布，`shift>1` 偏高噪声，`0<shift<1` 偏低噪声。
例如 `shift=3` 时归一化噪声比例 `q>0.8` 的概率约为 `42.86%`，
均匀分布为 `20%`。
直接 Python CLI 等价参数为
`--policy.time_sampling_distribution=shifted_uniform --policy.time_sampling_shift=3`。
resume 未显式设置时保留保存的分布及 shift；旧配置缺少 shift 时按 `1` 读取。
只设置 shift 不会切换分布，它不改变 `uniform`、`beta`、`mixed` 的行为。

`joint_prediction` 的 video/action/state 目标共享每个样本的 sigma，但高斯噪声各自独立；
当前帧和 current-state 条件保持干净。以上开关只改变训练分布；
推理仍用默认 20 步等间隔 Euler，不从这些分布随机抽推理网格。

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
t2v,i2v,joint_prediction,forward_dynamics,inverse_dynamics,action_prediction,state_prediction,tactile_prediction
```

先只训练当前 RGB 条件的视频生成和触觉图像生成时，推荐**保留完整任务列表**，把暂不训练的
任务设为 0：

```bash
STAGE2_TASK_NAMES=t2v,i2v,joint_prediction,forward_dynamics,inverse_dynamics,action_prediction,state_prediction,tactile_prediction \
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

`joint_prediction` 替代原来的 `action_video_prediction`：当前 RGB 和 available text 进入
Understanding，当前 RGB latent 和 current state 作为干净条件，未来 RGB、完整 action chunk
和未来 state 同时加噪并在同一次 Generation forward 中预测。例如将 40% step 分配给该任务：

```bash
STAGE2_TASK_WEIGHTS=0,0.3,0.4,0,0,0,0,0.3
```

任务列表决定 Generation Expert 的 task embedding 形状。恢复 checkpoint 时可以在保持
`STAGE2_TASK_NAMES` 完全一致的前提下修改 `STAGE2_TASK_WEIGHTS`；不能增加、删除或重排任务。
因此多阶段 curriculum 应从第一阶段就保留后续任务并先把其权重设为 0。若确实需要改变任务
列表，必须使用新的 `OUTPUT_DIR` 从 Stage 1 checkpoint 开始训练。两个变量必须同时设置；
不设置时，新训练使用 policy 默认配比（联合任务权重为 0），恢复训练使用 checkpoint 中保存
的比例。
当前任务语义保存为 `task_contract_version=2`。旧 checkpoint 的 forward/inverse 使用了 state，
旧联合任务也不预测 state，因此涉及这些任务的旧 checkpoint 会被训练恢复入口拒绝，
不能只替换任务名继续恢复。应从 Stage 1 初始化一个新实验。

只训练以下三个任务、例如在 LIBERO 上下游微调时：

```bash
STAGE2_TASK_NAMES=forward_dynamics,inverse_dynamics,joint_prediction \
STAGE2_TASK_WEIGHTS=0.3,0.3,0.4 \
DATA_MIX=libero \
TRAINING_STAGE=finetune \
WEIGHT_RESUME=false \
JOB_NAME=qwen3vl_mot_libero_joint_v2 \
OUTPUT_DIR=/path/to/new/libero_joint_v2 \
bash train_qwen3vl_mot_fsdp.sh
```

沿用自己的 Stage 1、Qwen、Cosmos 权重路径，并使用新的输出目录；上述比例仅为示例。
`forward_dynamics` 是当前 RGB + action → 未来 RGB；`inverse_dynamics` 是完整窗口 video →
该窗口的 action。两者既不以 state 为条件，也没有 state flow/decoder 重建 loss，
不要求有效 state 维度。`joint_prediction` 才要求 current state，联合监督 action、未来
video 和 state。可用文本仍作为辅助条件，不改变以上物理模态定义。

只启用这三个任务也支持生成验证：joint 在同一轮 Euler 采样中输出 action、video 和 state，
不再要求额外启用 `action_prediction`、`i2v` 或 `tactile_prediction`。
无触觉图像时跳过触觉生成；预测 state 的未来部分另存 `state_comparison.png`、
`state_summary.html` 和反归一化 `state_mae`，不把给定的 current state 计入未来误差。

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
| `MAX_EPOCHS` | 未设置 | sampler epoch 累计总上限；省略时使用源数据覆盖轮数 + 100；FSDP 与 `STEPS` 先到先停 |
| `SAVE_FREQ` | `2000` | 每隔多少 optimizer step 保存一次 |
| `GENERATION_EVAL_ENABLED` | `false` | 是否启用固定 held-out episode 生成验证 |
| `GENERATION_EVAL_FREQ` | `10000` | 每隔多少 optimizer step 生成一次 artifact |
| `GENERATION_EVAL_DATASETS` | 四个固定数据集 | 逗号分隔的验证数据集名 |
| `GENERATION_EVAL_EPISODES_PER_DATASET` | `2` | 每个数据集完整留出的 episode 数 |
| `GENERATION_EVAL_BATCH_SIZE` | `8` | 每个 rank 同步执行的生成验证 batch |
| `GENERATION_EVAL_VIDEO_FPS` | `8` | 导出 MP4 的播放帧率 |
| `STAGE2_TASK_NAMES` | 未设置 | 逗号分隔的任务列表；恢复时必须与 checkpoint 完全一致 |
| `STAGE2_TASK_WEIGHTS` | 未设置 | 与任务列表等长的非负相对权重；global batch 内按可用监督重归一化 |
| `RGB_AUG_ENABLED` | `false` | 可选训练 RGB 增强：90%-100% 裁剪和轻量颜色抖动，不含旋转；验证关闭 |
| `ACTION_PRED_RECON_WEIGHT` | embedding 缺省 `0.1`；resume 保留保存值 | 预测 embedding 解码 action 的 masked MSE 权重；`0` 关闭，仅支持 embedding |
| `STATE_PRED_RECON_WEIGHT` | embedding 缺省 `0.1`；resume 保留保存值 | 预测 embedding 解码未来 state 的 masked MSE 权重；不监督第 0 帧，`0` 关闭 |
| `TIME_SAMPLING_DISTRIBUTION` | fresh `uniform`；resume 使用保存值 | 训练时间分布：`uniform` / `beta` / `mixed` / `shifted_uniform`；joint 三个目标共享 sigma |
| `TIME_SAMPLING_SHIFT` | fresh `1.0`；resume 使用保存值 | 仅影响 `shifted_uniform`；有限正数，`>1` 偏高噪声，`=1` 均匀，`<1` 偏低噪声；不改变推理网格 |
| `TIME_SAMPLING_BETA_ALPHA` | `1.5` | Beta 的 alpha 参数，必须为有限正数 |
| `TIME_SAMPLING_BETA_BETA` | `1.0` | Beta 的 beta 参数，必须为有限正数 |
| `TIME_SAMPLING_BETA_PROBABILITY` | `0.5` | mixed 中选 Beta 的概率，范围 `[0,1]` |
| `TACTILE_GENERATION_TARGET` | `spatial_patches` | 新训练使用可解码触觉空间 latent |
| `FSDP_CHECKPOINT_SYNC_FILES` | `false` | DCP 是否对每个 shard 强制 `fsync` |
| `FSDP_CHECKPOINT_THREADS` | `1` | 每个 rank 的 DCP writer 线程数 |
| `FSDP_CHECKPOINT_HEARTBEAT_SECONDS` | `60` | checkpoint 进度日志间隔；`0` 关闭 |
| `FP8_ENABLED` | `auto` | 全部 rank 支持原生 FP8 才启用；`true` 强制原生，`false` 使用 BF16 |
| `FP8_EMULATE` | `false` | 仅正确性 smoke test 的功能模拟，不用于正式训练 |
| `TORCHRUN_BIN` | `torchrun` | 分布式启动程序 |
| `WEIGHT_RESUME` | `true` | 自动检测并恢复 `OUTPUT_DIR` 最新 checkpoint；不存在则新训练 |
| `CHECK_PATHS` | `true` | 启动前检查权重和数据路径 |
| `DRY_RUN` | `false` | 只打印最终命令 |