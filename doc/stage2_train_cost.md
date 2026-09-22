# 阶段2训练花费估计

本文同时记录组件级工程估计、H100 synthetic smoke 和 4×RTX A6000
真实模型/真实数据端到端测量；目前仍没有 8×H100 完整模型 benchmark。H100
区间根据 A6000 组件测量、H100 SXM 的 BF16/FP8 吞吐、显存带宽以及 8 卡
NVLink/FSDP 通信开销估算。视频解码和共享存储速度会直接影响端到端结果，因此
不应把“模型计算时间”当成最终 dataloader 吞吐。

## 已实现的 8×H100 FSDP + FP8 + AdamW8bit 路径

正式入口是：

```bash
STAGE1_CHECKPOINT=/path/to/exported_robo_contrast \
bash train_qwen3vl_mot_fsdp.sh
```

启动脚本默认使用：

```text
8 GPUs
micro-batch/GPU = 16
gradient accumulation = 1
global batch = 128
understanding_tuning_mode = full
FP8 scope = Generation Expert + VLM Transformer layers
FP8 recipe = rowwise_with_gw_hp
optimizer = bitsandbytes AdamW8bit
```

这里采用的是 classic FSDP `FULL_SHARD + use_orig_params=True`，而不是 FSDP2。
实测 bitsandbytes 0.48.2 的 CUDA optimizer kernel 不能直接更新 FSDP2 的
DTensor 参数，会在 `optimizer_update_8bit_blockwise` 报 mixed Tensor/DTensor
错误。classic FSDP 保留普通参数视图，能够执行真正的 8-bit AdamW step。

“FP8 模型参数”在训练语义上需要区分：

- 可训练 master parameter 保持 BF16，保证 optimizer update 和 checkpoint 精度；
- TorchAO `Float8Linear` 动态将 eligible Linear 的输入、权重和梯度 GEMM 转为
  FP8；`rowwise_with_gw_hp` 会让 weight-gradient 路径保持更高精度；
- classic FSDP 当前仍以 BF16 all-gather 权重，不是将持久化 checkpoint 或所有
  通信权重直接保存为 FP8；
- LayerNorm、embedding、LoRA adapter、不满足 16 对齐的小输出 head、teacher、
  physical encoder 和 Wan VAE 不转换为 FP8。

冻结参数默认复制在每张卡上，避免 teacher、VAE、冻结 vision/text 权重每次前向
都发生 FSDP all-gather；Generation blocks 和打开训练的 VLM Transformer blocks
按 block full-shard。

checkpoint 使用组合格式：

- 模型权重由 Distributed Checkpoint 保存，可做模型分片重组；
- bitsandbytes optimizer state 按 rank 保存，因为标准 FSDP optimizer state
  转换不支持同一 state key 同时包含 uint8 大张量和 FP32 小张量；
- resume 必须保持相同 world size、FSDP/FP8 wrap、per-rank batch、gradient
  accumulation 和 sampler geometry，避免 8-bit moment 绑定到不同参数分片或
  数据游标静默错位。

本地已在 2×RTX A6000 上用 `fp8_emulate=true` 完成
forward/backward、gradient clipping、AdamW8bit step、模型/optimizer/scheduler
保存、恢复后继续 step 的 smoke。之后又在真实 H100 80GB 上完成原生 FP8 验证：

- 1×H100：native FP8、AdamW8bit 和 checkpoint/resume 通过；
- 8×H100：classic FSDP full-shard、native FP8、8-bit moments 和每 rank
  optimizer checkpoint/resume 全部通过；
- 8 卡 synthetic smoke 每卡峰值约 0.567 GiB，首个 step 含 kernel warmup 和
  通信初始化约 3.08–4.32 秒；该小模型数字不用于推算完整 6B 模型吞吐；
- optimizer state 同时包含 FP32 scale/step 小 tensor 和 uint8 `state1/state2`。

A6000 没有 H100 FP8 Tensor Core，因此本地 emulation 只验证逻辑；H100 smoke
验证了真实 FP8 kernel，但仍不是完整 Qwen3-VL + VAE + 数据管线 benchmark。

## 基于 Qwen3-VL-4B 文本冻结、视觉 LoRA 和 1.429B Generation Expert 的估计

### 模型配置

| 模块 | 配置 |
|---|---|
| Understanding Expert | Qwen3-VL-4B，基础权重冻结 |
| 文本塔 | 不使用 LoRA，参数冻结 |
| 视觉塔 | 最后 4 层 attention 使用 rank-16 LoRA |
| Latent action queries | 16 个，宽度 2560 |
| Generation Expert | hidden 2048，28 层，16Q/8KV GQA |
| Generation MLP | intermediate 9216，ReLU² |
| 视频 latent | Wan VAE，48 channels，2×2 spatial patchify |
| 视频 token 数 | 9 帧输入得到 3×16×16 latent，再 patchify 为 3×8×8 = 192 tokens |
| Physical Expert | 继承阶段一，默认冻结 |
| Teacher 与 VAE | 冻结，只执行前向 |

参数量如下：

| 参数集合 | 参数量 |
|---|---:|
| Qwen3-VL-4B 基础模型 | 4,437,815,808 |
| 视觉 LoRA | 393,216 |
| Latent query 参数 | 40,960 |
| Generation Expert | 1,429,469,696 |
| Latent-action projection | 2,629,632 |
| **总可训练参数** | **1,432,533,504** |

移除文本 LoRA 后少训练 2,621,440 个参数。这个变化主要用于避免语言能力漂移，
对显存和速度影响很小。虽然文本权重被冻结，但 latent queries 和视觉 LoRA 的
梯度仍需穿过 Qwen language model，因此不能把文本塔的反向计算完全省掉。

### A6000 实测校准

下表使用 BF16、activation checkpoint、batch size 16，分别运行各组件。峰值显存
来自独立进程，不能直接相加；时间可以近似相加，用于估算单步模型计算成本。

| 组件 | A6000 时间 | 峰值 allocated 显存 |
|---|---:|---:|
| Qwen3-VL-4B understanding forward/backward | 1.049 s | 9.81 GiB |
| 1.429B Generation Expert forward/backward | 0.653 s | 6.01 GiB |
| Wan VAE 9 帧 encoder forward | 0.633 s | 4.55 GiB |
| 阶段一 perception teacher forward | 0.145 s | 1.76 GiB |
| **组件时间合计** | **约 2.48 s** | 不适用 |

其中 Generation Expert 使用最重的 tactile prediction token 组合：
192 个 video tokens、8 个 state tokens、8 个 action tokens 和约 20 个 tactile
tokens。Qwen understanding 序列约为 146 tokens。

### 8×80GB H100 的 batch size

| 训练方式 | 建议 micro-batch/GPU | 8 卡 global batch | 说明 |
|---|---:|---:|---|
| 当前 DeepSpeed ZeRO-2 BF16 | 8 起步，目标 16 | 64–128 | 权重复制，但优化器和梯度分片 |
| FSDP BF16 + activation checkpoint | **16 起步，目标 32** | **128–256** | classic FSDP full shard |
| FSDP + Generation/VLM FP8 Linear | 16 起步，可尝试 24–32 | 128–256 | master weights、teacher、VAE 保持 BF16 |

在 8 卡 FSDP + AdamW8bit 下，按 BF16 参数 2 bytes、BF16 梯度 2 bytes 和两份
8-bit moments 约 2 bytes，即约 6 bytes/可训练参数估算，1.433B 可训练参数的
分片训练状态约为 **1.00 GiB/GPU**。小于 `min_8bit_size` 的 tensor 仍使用
FP32 moments，另外还需要复制的冻结权重、FSDP all-gather buffer、activation、
CUDA workspace 和 dataloader 输入。FP8 Linear 主要减少 GEMM 成本，不会把这
部分持久化训练状态再减半。

从显存角度看，micro-batch 48–64 可能仍能放下，但 Wan VAE 的临时显存约随
batch 线性增长，A6000 上从 batch 16 的 4.55 GiB 增加到 batch 32 的
8.80 GiB。继续放大 micro-batch 不会明显提高矩阵乘吞吐，反而会放大视频解码、
host memory 和失败重跑成本。因此实际推荐值是 16，稳定后再尝试 24 和 32。

### 8×80GB H100 的速度

| micro-batch/GPU | global batch | 模型计算估计 | 含数据与通信的端到端估计 | 全局吞吐 |
|---:|---:|---:|---:|---:|
| 16 | 128 | 0.55–0.80 s/step | **0.9–1.3 s/step** | **约 100–140 samples/s** |
| 32 | 256 | 1.0–1.5 s/step | **1.5–2.2 s/step** | **约 115–170 samples/s** |

如果视频位于慢速共享存储，9 帧随机解码可能成为主要瓶颈，端到端吞吐会低于上表。
离线缓存 Wan latents 和阶段一 latent-action targets，通常比继续扩大 GPU
micro-batch 更有效。

### 推荐启动参数

当前默认保持文本冻结、视觉 LoRA：

```bash
UNDERSTANDING_TUNING_MODE=lora \
UNDERSTANDING_TEXT_LORA_LAYERS=0 \
UNDERSTANDING_VISION_LORA_LAYERS=4 \
UNDERSTANDING_LORA_RANK=16 \
UNDERSTANDING_LR_SCALE=0.1 \
bash train_qwen3vl_mot_local.sh
```

对应 learning rate 为：

```text
Generation Expert / latent queries: 1e-4
Vision LoRA:                         1e-5
Text model:                          frozen
```

## 基于 Qwen3-VL-4B VLM Transformer 全量、视觉 LoRA 和 1.429B Generation Expert 的估计

### 模型配置

这里的“VLM Transformer 全量”不是把整个 Qwen3-VL 全量训练。Qwen3-VL 没有
额外独立的 cross-attention fusion tower：视觉 token 和文本 token 共同进入
`language_model.layers`，因此这 36 个 decoder blocks 就是本节所指的多模态
Transformer。

具体训练范围是：

- `language_model.layers`：全量训练；
- vision encoder：基础权重冻结，最后 4 层继续使用 rank-16 LoRA；
- text token embedding 与 language final norm：冻结；
- text attention 不使用 LoRA；
- Generation Expert：全量训练；
- Wan VAE、physical expert 和阶段一 teacher：冻结。

```text
understanding_tuning_mode = "full"
understanding_text_lora_layers = 0
understanding_vision_lora_layers = 4
understanding_gradient_checkpointing = true
generation_gradient_checkpointing = true
```

| 参数集合 | 参数量 |
|---|---:|
| 全量 VLM Transformer layers | 3,633,509,376 |
| Vision LoRA | 393,216 |
| Latent query 参数 | 40,960 |
| Generation Expert | 1,429,469,696 |
| Latent-action projection | 2,629,632 |
| **总可训练参数** | **5,066,042,880** |

冻结的 Qwen 参数仍包括约 415M 的 vision base、388,956,160 个 token embedding
参数和 final norm。模型前向仍使用这些权重，但它们不产生 weight gradient 或
optimizer state。

### 显存变化

按照约 6 bytes/可训练参数估算，8 卡 FSDP + AdamW8bit 下分片训练状态约为：

```text
5.066B × 6 bytes ÷ 8 = 3.54 GiB/GPU
```

相比默认视觉 LoRA 配置增加约 **2.54 GiB/GPU** 的分片训练状态。Activation
checkpoint 后，activation 规模变化不大；主要新增项是 36 个 VLM Transformer
blocks 的 BF16 参数/梯度和 8-bit Adam moments。正式长训练使用 FSDP full shard；
当前 ZeRO-2 可以用于容量验证，但会在每张卡复制所有模型权重。

| 训练方式 | 建议 micro-batch/GPU | 8 卡 global batch | 说明 |
|---|---:|---:|---|
| DeepSpeed ZeRO-2 BF16 | 8 起步，可尝试 16 | 64–128 | 权重仍在每卡复制 |
| FSDP BF16 + activation checkpoint | **16 起步，可尝试 32** | **128–256** | 32 需实机确认通信和 VAE 峰值 |
| FSDP + Generation/VLM FP8 Linear | **16 起步，可尝试 24–32** | **128–256** | 默认同时转换打开训练的两个分支 |

micro-batch 16/GPU 已经能在 8 卡上形成 global batch 128，不需要梯度累积。
如果从 8/GPU 起步，则设置 gradient accumulation 2 得到相同 global batch。

### 速度变化

默认配置虽然不计算 VLM Transformer 的 weight gradient，但仍需把梯度传回
latent queries 和 vision LoRA。将 36 个 Transformer blocks 全量打开后，需要
额外计算这些 Linear 的 weight gradient，因此 Qwen understanding 部分预计慢约
25–35%，整个训练 step 预计慢约 12–25%；同时还增加 FSDP reduce-scatter 通信。
H100 上同时启用 VLM FP8 后会回收其中一部分计算成本，但下表仍是未做 H100
实机 benchmark 前的保守估计。

| micro-batch/GPU | gradient accumulation | global batch/optimizer step | 端到端估计 | 全局吞吐 |
|---:|---:|---:|---:|---:|
| 8 | 2 | 128 | 1.2–1.8 s | 约 71–107 samples/s |
| 16 | 1 | 128 | **1.0–1.5 s** | **约 85–128 samples/s** |
| 32 | 1 | 256 | 1.8–2.7 s | 约 95–142 samples/s |

VLM Transformer 全量时，micro-batch 16 通常比 8×gradient accumulation 2
更快，因为后者需要执行两次 VAE、teacher、Qwen 和 Generation
forward/backward。

### 推荐启动参数

VLM Transformer 全量建议先把 understanding learning rate 设为 `5e-6`，
Generation Expert 保持 `1e-4`。当前 optimizer 将 VLM Transformer 和 vision
LoRA 放在同一个 understanding parameter group，可以通过
`UNDERSTANDING_LR_SCALE` 调整：

```bash
UNDERSTANDING_TUNING_MODE=full \
UNDERSTANDING_LR_SCALE=0.05 \
bash train_qwen3vl_mot_fsdp.sh
```

对应：

```text
Generation Expert / latent queries: 1e-4
VLM Transformer layers:              5e-6
Vision LoRA:                         5e-6
Text embedding / final norm:         frozen
```

`1e-5` 可以作为 VLM Transformer 全量微调的上限实验，但不建议直接从该值开始。
这些层同时处理视觉和文本 token，较大学习率仍可能造成语言能力和跨模态表示漂移。

### 两种方案的建议

| 项目 | Transformer 冻结 + 视觉 LoRA | VLM Transformer 全量 + 视觉 LoRA |
|---|---:|---:|
| 可训练参数 | 1.433B | 5.066B |
| 推荐 micro-batch/GPU | 16，目标 32 | 16 起步，可尝试 32 |
| global batch 128 | 16×8，无累积 | 16×8，或 8×8×累积 2 |
| 预计端到端 step | 0.9–1.3 s | 1.0–1.5 s |
| 语言能力漂移风险 | 低 | 中高 |
| 推荐优先级 | **默认方案** | 第二阶段消融或后期解冻 |

建议先训练“VLM Transformer 冻结 + 视觉 LoRA + Generation 全量”版本。若
latent-action distillation 和 generation loss 已稳定收敛，再从该 checkpoint
解冻 `language_model.layers`，使用更低的 learning rate 做短周期联合微调，
而不是一开始就同时训练 5.066B 参数。

## 使用 `step_16k` 的真实 Stage 2 端到端实测

本节记录 2026-09-22 在提交 `04ebb99` 上完成的真实数据调试，不再是各组件独立
benchmark。阶段一输入为：

```text
/Data/lzl/ace_stage1/step_16k/mp_rank_00_model_states.pt
global_steps = 16000
checkpoint format = DeepSpeed ZeRO-2 model state
```

根据 checkpoint 中的模型结构恢复出的 Stage 1 配置为 Qwen3-VL-4B vision、
最后 4 层 rank-16 vision LoRA、5 层 evidence、5 层 fusion、3 层 predictor、
14 层 physical transformer、16 个 change queries、ResNet-18 tactile encoder
和 Wan VAE reconstruction target。加载检查结果为：

```text
checkpoint keys = 1397
shape mismatches = 0
missing keys = 0
unexpected keys = 0
Stage 1 -> Qwen vision transfer = 100% (293 tensors)
Stage 1 -> Qwen text transfer = 0% (expected; Qwen text keeps pretrained weights)
```

精确匹配的配置已写入：

```text
/Data/lzl/ace_stage1/step_16k/config.json
```

### 实测配置

| 项目 | 配置 |
|---|---|
| GPU | 4×RTX A6000 48GB |
| PyTorch | 2.9.1 + CUDA 13.0 |
| 数据 | `debug_research_data`，真实视频/state/action/tactile |
| micro-batch/GPU | 1 |
| global batch | 4 |
| gradient accumulation | 1 |
| Understanding | Qwen3-VL-4B Transformer 全量，vision 最后 4 层 LoRA |
| Generation Expert | 1.429B，全量训练 |
| Physical Expert | Stage 1 权重，冻结 |
| Teacher / Wan VAE | 冻结 |
| FSDP | classic `FULL_SHARD + use_orig_params=True` |
| FP8 | TorchAO `rowwise_with_gw_hp`，A6000 上使用 emulation |
| FP8 Linear 数 | 431 |
| FSDP wrapped blocks | 64 |
| Optimizer | bitsandbytes AdamW8bit |
| 可训练参数 | 5.066B |

A6000 不具备 H100 的原生 FP8 Tensor Core，因此这里的耗时**不能用于估计原生
FP8 加速比**；它主要验证完整模型、真实数据、FSDP 分片、FP8 Linear 路径和
AdamW8bit 更新能够一起运行。

这次机器上的 `agibot_alpha` 打开失败，`language_table`、`ms_data_xdof_3`
和 `ego_dex_split4` 未找到，因此统计对应成功加载的 9 个数据集。补齐这些数据后，
模型单步计算量基本不变，但 source frames、自然训练步数和数据读取长尾需要重新
计算。

### 40 个 optimizer steps 的测量

40 步覆盖了 `t2v`、`i2v`、forward/inverse dynamics、action/state prediction
和 tactile prediction。`num_workers=0`，因此数据时间也包含同步视频读取。

| 指标 | 平均 | 中位数 | P90 | 范围 |
|---|---:|---:|---:|---:|
| forward + backward | 10.61 s | 10.71 s | 11.22 s | 3.10–12.56 s |
| dataloader | 1.07 s | 0.34 s | 0.86 s | 0.16–27.01 s |
| 端到端 step | 11.68 s | 11.10 s | 11.82 s | 3.33–37.05 s |
| PyTorch peak allocated/GPU | 11.45 GiB | 11.4 GiB | 11.4 GiB | 11.4–13.2 GiB |

中位端到端吞吐约为：

```text
4 samples / 11.10 s = 0.36 samples/s
```

唯一一次 27.01 秒 dataloader 尖峰将平均数据时间从中位数 0.34 秒拉高到
1.07 秒，说明共享存储长尾仍需要在正式训练中监控。运行期间 `nvidia-smi`
观察到约 17.4 GiB/GPU device memory used；它高于 PyTorch
`max_memory_allocated`，因为还包含 allocator reserve、CUDA context 和
workspace。

随后使用修复后的 `STEPS=1` 做了受控退出验证：

```text
loss = 3.729
forward + backward = 4.825 s
dataloader = 1.043 s
peak allocated = 13.2 GiB/GPU
completed optimizer steps = 1
```

该单步受益于前一次运行生成的 CUDA/NVRTC cache，不能替代 40 步统计；容量判断
采用 13.2 GiB allocated 和约 17.4 GiB device-used，速度判断采用更保守的
40 步中位数。

### 当前默认 8×H100 完整训练预算

正式 launcher 默认：

```text
micro-batch/GPU = 16
global batch = 128
dataset_size_one_epoch = 100000
gradient accumulation = 1
steps cap = 600000
```

按本次实际加载到的 42,629,776 个 source frames 计算：

```text
steps/sampler epoch = floor(100000 / 128) = 781
actual samples/sampler epoch = 781 * 128 = 99,968
source-equivalent epochs = ceil(42,629,776 / 99,968) = 427
extra epochs = 100
planned optimizer steps = (427 + 100) * 781 = 411,587
planned sample draws = 411,587 * 128 = 52,683,136
```

自然 epoch 计划为 411,587 步，小于 `STEPS=600000`，所以当前默认完整训练会在
约 411.6k 步结束。`STEPS` 现在是有效的 optimizer-step 上限，可用于更短的
smoke 或训练预算控制。

沿用上文“VLM Transformer 全量 + vision LoRA + Generation 全量”的
**1.0–1.5 s/step H100 估计**：

| 项目 | 乐观 | 保守 |
|---|---:|---:|
| 单步时间 | 1.0 s | 1.5 s |
| 411,587 步墙钟 | 114.3 h | 171.5 h |
| 墙钟天数 | 4.76 天 | 7.15 天 |
| 8 卡 GPU-hours | 915 | 1,372 |

若为数据长尾、checkpoint、评估和重启预留 10–20%，建议实际排期按约
**5.2–8.6 天**准备。这个总成本仍是估算；只有在 8×H100 上使用完整模型、
原生 FP8、micro-batch 16 和同一数据存储完成稳定多步 benchmark 后，才能替换
为实测值。
