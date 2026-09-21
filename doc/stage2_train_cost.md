# 阶段2训练花费估计

本文记录的是工程估计，不是 8×H100 的实机 benchmark。估计先在单张
RTX A6000 48GB 上分别测量理解专家、Generation Expert、Wan VAE 和阶段一
teacher，再按照 H100 SXM 的 BF16 吞吐、显存带宽以及 8 卡 NVLink/FSDP2
通信开销给出区间。视频解码和共享存储的速度会直接影响端到端结果，因此不应把
“模型计算时间”当成最终 dataloader 吞吐。

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
| FSDP2 BF16 + activation checkpoint | **16 起步，目标 32** | **128–256** | 推荐正式训练方案 |
| FSDP2 + Generation FP8 | 32 起步，可尝试 48 | 256–384 | Qwen、teacher、VAE 仍保持 BF16 |

在 8 卡 FSDP2 下，按 BF16 参数和梯度、FP32 master weight 与 Adam moments 共
约 16 bytes/可训练参数估算，1.433B 可训练参数的训练状态约为
**2.67 GiB/GPU**。实际还需要冻结权重、FSDP all-gather buffer、activation、
CUDA workspace 和 dataloader 输入。

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

## 基于 Qwen3-VL-4B VLM 全量微调和 1.429B Generation Expert 的估计

### 模型配置

Generation Expert、Wan VAE、physical expert 和 teacher 均保持上一节配置，只把
Qwen3-VL-4B 的 language model 与 vision model 从冻结/LoRA 改为全量训练：

```text
understanding_tuning_mode = "full"
understanding_text_lora_layers 和 understanding_vision_lora_layers 不再生效
understanding_gradient_checkpointing = true
generation_gradient_checkpointing = true
```

| 参数集合 | 参数量 |
|---|---:|
| 全量 Qwen3-VL-4B | 4,437,815,808 |
| Latent query 参数 | 40,960 |
| Generation Expert | 1,429,469,696 |
| Latent-action projection | 2,629,632 |
| **总可训练参数** | **5,869,956,096** |

这不是一个 5.87B 独立模型，而是约 4.44B 的 Qwen understanding 路径与
1.43B generation 路径同时训练。阶段一 teacher、physical expert 和 Wan VAE
仍然冻结，不计入可训练参数。

### 显存变化

同样按照约 16 bytes/可训练参数估算，8 卡 FSDP2 下训练状态约为：

```text
5.870B × 16 bytes ÷ 8 = 10.93 GiB/GPU
```

相比视觉 LoRA 配置增加约 **8.27 GiB/GPU**。Activation checkpoint 后，
activation 规模变化不大；主要新增项是 Qwen 权重的梯度、FP32 master weights
和 Adam moments。全量微调必须优先使用 FSDP2 full shard 或 ZeRO-3，不建议把
当前 ZeRO-2 作为长期训练配置。

| 训练方式 | 建议 micro-batch/GPU | 8 卡 global batch | 说明 |
|---|---:|---:|---|
| DeepSpeed ZeRO-2 BF16 | 4–8 起步，可尝试 16 | 32–128 | 权重仍在每卡复制 |
| FSDP2 BF16 + activation checkpoint | **8 起步，推荐 16** | **64–128** | 预计 32 能放下，但需实机确认通信峰值 |
| FSDP2 + Generation FP8 | 16 起步，可尝试 24–32 | 128–256 | VLM 仍为 BF16 时收益有限 |

micro-batch 16/GPU 已经能在 8 卡上形成 global batch 128，不需要梯度累积。
如果从 8/GPU 起步，则设置 gradient accumulation 2 得到相同 global batch。

### 速度变化

视觉 LoRA 配置虽然不计算 Qwen 基础权重梯度，但仍需把梯度传回 latent queries
和 vision LoRA。全量微调额外计算每个 Qwen Linear 的 weight gradient，因此
Qwen understanding 部分预计慢约 25–35%，整个训练 step 预计慢约 15–30%；
同时还增加 FSDP reduce-scatter 通信。

| micro-batch/GPU | gradient accumulation | global batch/optimizer step | 端到端估计 | 全局吞吐 |
|---:|---:|---:|---:|---:|
| 8 | 2 | 128 | 1.3–1.9 s | 约 67–98 samples/s |
| 16 | 1 | 128 | **1.1–1.6 s** | **约 80–116 samples/s** |
| 32 | 1 | 256 | 2.0–2.9 s | 约 88–128 samples/s |

全量微调时，micro-batch 16 通常比 8×gradient accumulation 2 更快，因为后者
需要执行两次 VAE、teacher、Qwen 和 Generation forward/backward。

### 推荐启动参数

全量微调建议先把 Qwen learning rate 设为 `5e-6`，Generation Expert 保持
`1e-4`。当前 optimizer 已将 understanding 参数放在独立 parameter group，
可以通过 `UNDERSTANDING_LR_SCALE` 调整：

```bash
UNDERSTANDING_TUNING_MODE=full \
UNDERSTANDING_LR_SCALE=0.05 \
bash train_qwen3vl_mot_local.sh
```

对应：

```text
Generation Expert / latent queries: 1e-4
Qwen3-VL vision + language:          5e-6
```

`1e-5` 可以作为 Qwen 全量微调的上限实验，但不建议直接从该值开始。阶段一迁移
后的视觉权重和 Qwen 原始语言权重来自不同训练目标，全量使用较大学习率更容易导致
语言能力与视觉表示同时漂移。

### 两种方案的建议

| 项目 | 文本冻结 + 视觉 LoRA | VLM 全量微调 |
|---|---:|---:|
| 可训练参数 | 1.433B | 5.870B |
| 推荐 micro-batch/GPU | 16，目标 32 | 8 起步，推荐 16 |
| global batch 128 | 16×8，无累积 | 16×8，或 8×8×累积 2 |
| 预计端到端 step | 0.9–1.3 s | 1.1–1.6 s |
| 语言能力漂移风险 | 低 | 高 |
| 推荐优先级 | **默认方案** | 第二阶段消融或后期解冻 |

建议先训练“文本冻结 + 视觉 LoRA + Generation 全量”版本。若 latent-action
distillation 和 generation loss 已稳定收敛，再从该 checkpoint 解冻整个 VLM，
使用更低的 Qwen learning rate 做短周期联合微调，而不是一开始就同时训练
5.87B 参数。