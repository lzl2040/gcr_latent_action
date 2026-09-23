# Qwen3-VL MoT Stage 2 模型说明

> **维护约定**
>
> 本文是 `qwen3vl_mot_stage2` 分支的模型结构权威说明。凡是修改以下任一内容，
> 必须在同一个提交中同步更新本文：
>
> - 模型模块、维度、参数量或冻结范围；
> - 数据输入、时间窗口、视角数量或 canonical action/state 空间；
> - Stage 1 权重迁移方式；
> - task family、噪声方式、loss 或采样比例；
> - action/video 推理接口；
> - FSDP、FP8 或 optimizer 对模型训练语义的影响。
>
> 当前内容的模型实现基线对应分支 `qwen3vl_mot_stage2`、提交 `f99e22a`。

---

## 1. 设计目标

Stage 2 在 Stage 1 已学到的“视觉变化—物理变化”对齐基础上，进一步训练一个统一的
多模态动力学模型。模型需要同时支持：

- 文本到视频；
- 当前图像到未来视频；
- 给定 action 预测未来视频和 state；
- 给定视觉变化反推动作；
- 当前图像和 state 到 action；
- state 与 tactile 预测；
- 不同机器人 action/state 维度共存。

当前实现采用 **单向 Mixture-of-Transformers（MoT）**：

- Qwen3-VL-4B 作为 Understanding Expert；
- 1.429B Generation Expert 负责多模态 flow matching；
- Stage 1 Physical Encoder 负责将 state、action 和 tactile 转成物理 token；
- 冻结的 Wan VAE 将视频压缩到 latent；
- 冻结的 Stage 1 perception branch 提供 latent-action 蒸馏目标。

这里的 MoT 不是带路由门控的传统 MoE。Understanding 和 Generation 是两个职责不同的
Transformer，信息只从 Understanding 单向流向 Generation。

---

## 2. 总体结构

```text
current primary image + language
                 │
                 ▼
       Qwen3-VL Understanding Expert
                 │
                 ├── 16 latent queries
                 │        │
                 │        ▼
                 │   1024-d projection
                 │        │
                 │        └── MSE against frozen Stage 1 change queries
                 │
                 └── native K/V from 8 Qwen decoder layers ─────────────┐
                                                                        │
9-frame primary video ──> frozen Wan VAE ──> video latent tokens        │
state/action/tactile ──> Stage 1 Physical Encoder ──> physical tokens    │
                                                                        ▼
                                                    1.429B Generation Expert
                                                                        │
                                   ┌────────────────┬───────────────────┼──────────────┐
                                   ▼                ▼                   ▼              ▼
                              video velocity   state velocity      action velocity  tactile velocity
```

关键约束：

1. Qwen 使用 causal attention，末尾 latent queries 可以读取图像和文本。
2. Generation 使用非 causal attention，所有 generation modality tokens 可以相互读取。
3. Generation query 同时读取 Qwen 原生 K/V 和 Generation 自身 K/V。
4. Qwen 永远不会读取 Generation tokens，因此不存在 Generation 信息泄漏回 Understanding。

---

## 3. 输入数据契约

当前 `MultiModalContrastiveDataset` 为 Stage 2 提供以下核心字段：

| 字段 | 典型形状 | 含义 |
|---|---|---|
| `image_t0` | `(B,3,H,W)` | 当前 primary 相机图像 |
| `image_t1` | `(B,3,H,W)` | 时间窗口末端 primary 图像 |
| `video` | `(B,9,3,H,W)` | 同一窗口内均匀采样的 9 帧 primary 视频 |
| `task` | `list[str]` | 语言指令 |
| `pair_is_valid` | `(B,)` | 未来端点是否未被 episode 边界截断 |
| `observation.state` | `(B,32,40)` | canonical state trajectory |
| `state_mask` | `(B,40)` | 当前机器人有效的 state 维度 |
| `action` | `(B,32,40)` | canonical action chunk |
| `action_mask` | `(B,40)` | 当前机器人有效的 action 维度 |
| `tactile_signal` | `(B,32,32)` | 最多 32 维的触觉信号 |
| `tactile_image` | `(B,V,4,3,112,112)` | 最多 6 个 pad、每个 pad 4 帧 |
| `tactile_image_mask` | `(B,V)` | 有效触觉图像视角 |
| `sample_rate` | `(B,)` | 数据真实采样率 |

### 3.1 时间窗口

正式配置为：

```text
window_mode       = frames
chunk_size        = 32
group_size        = 4
world_video_frames = 9
```

因此：

- action/state 使用连续的 32 个 source commands；
- 32 步被划分为 8 个物理 group；
- 视频在同一个 `[t, t+31]` 窗口内均匀读取 9 帧；
- 物理轨迹与视频终点严格一致。

### 3.2 当前视角状态

虽然 `oxe_configs.py` 定义了 `primary / secondary / wrist` 三个槽位，但当前模型接口仍然是：

```text
image_t0: (B,C,H,W)
video:    (B,T,C,H,W)
```

目前实际只使用 primary 相机。`secondary` 和 `wrist` 尚未作为独立视角进入
Understanding 或 Generation，`use_wrist_image` 默认也为 `False`。

---

## 4. Stage 1 初始化与继承

实现位于：

```text
lerobot/common/policies/qwen3vl_mot/modeling_qwen3vl_mot.py
lerobot/common/policies/qwen3vl_mot/stage1_transfer.py
```

### 4.1 Fresh Stage 2 初始化

首次构造 Stage 2 时，必须提供：

```text
policy.stage1_checkpoint
policy.stage1_config
```

当前真实 checkpoint：

```text
/Data/lzl/ace_stage1/step_16k/mp_rank_00_model_states.pt
```

对应 Stage 1 结构：

- Qwen3-VL-compatible vision backbone；
- vision 最后 4 层 rank-16 LoRA；
- 16 个 change queries；
- 14 层、hidden 1024 的 Physical Encoder；
- chunk size 32、group size 4；
- ResNet-18 tactile codec；
- Wan VAE reconstruction target。

加载时会检查：

- checkpoint 是否包含 perception vision；
- 是否包含 state/action projections；
- 是否包含 physical transformer blocks；
- Stage 1 与 Stage 2 的 chunk、group、hidden 和 canonical dimensions 是否完全一致。

任何结构不匹配都会直接报错，不允许静默部分加载。

### 4.2 Vision 与 text 权重迁移

初始化顺序是：

1. 加载原始 Qwen3-VL-4B；
2. 将 Stage 1 perception vision 中同名同 shape 的权重复制到 Qwen visual；
3. 尝试复制 text 权重；
4. 完成迁移后再挂载新的 LoRA。

当前 `step_16k` 的实际迁移结果：

```text
Stage 1 -> Qwen vision = 100%（293 tensors）
Stage 1 -> Qwen text   = 0%
```

text 迁移为 0% 是预期行为：Stage 1 text branch 是 SigLIP2，而 Stage 2 使用 Qwen
language model。因此 Stage 2 文本侧保留 Qwen pretrained initialization。

### 4.3 直接复用 Physical Encoder

Physical Encoder 不是复制一份，而是直接从 Stage 1 teacher 中移动到 Stage 2：

```python
self.physical_encoder = teacher.physical_encoder
teacher.physical_encoder = None
```

这样不会同时保留两份约 200M 级别的 physical tower。

### 4.4 冻结 perception teacher

Stage 1 perception encoder 被保留为冻结 teacher，但会删除：

- Stage 1 predictor；
- Stage 1 VAE。

teacher 只负责根据：

```text
image_t0 + image_t1 + language
```

产生 Stage 1 的 16 个 latent-action targets。

### 4.5 Stage 2 自包含恢复

Stage 2 保存时会把 Stage 1 policy config 嵌入自身配置，并将：

```text
initialize_from_stage1 = false
```

之后从 Stage 2 checkpoint 恢复时，不再依赖原始 Stage 1 checkpoint 的配置文件。

---

## 5. Qwen3-VL Understanding Expert

实现位于：

```text
lerobot/common/policies/qwen3vl_mot/modeling_understanding.py
```

### 5.1 Qwen3-VL-4B 结构

当前 checkpoint 的主要尺寸：

| 项目 | 数值 |
|---|---:|
| text hidden size | 2560 |
| text decoder layers | 36 |
| text query heads | 32 |
| text KV heads | 8 |
| head dim | 128 |
| text intermediate size | 9728 |
| vision hidden size | 1024 |
| vision blocks | 24 |
| vision patch size | 16 |
| spatial merge size | 2 |

### 5.2 Understanding 输入序列

当前图像统一 resize 到 `256×256`。Qwen vision 使用 16×16 patch 和 2×2 merger：

```text
256 / 16 = 16
16×16 patches / 2×2 merge = 8×8 = 64 image tokens
```

Understanding 序列为：

```text
[vision_start]
[64 image tokens]
[vision_end]
[最多 64 text tokens]
[16 learnable latent queries]
```

最大长度约为：

```text
2 + 64 + 64 + 16 = 146 tokens
```

当任务为 `t2v` 时没有输入图像，序列只包含 text 和 latent queries。

### 5.3 16 个 latent queries

Qwen 最终 norm 后，取最后 16 个 query hidden states：

```text
(B,16,2560)
```

然后经过：

```text
LayerNorm(2560)
Linear(2560 -> 1024)
LayerNorm(1024)
```

得到：

```text
predicted_latent_action: (B,16,1024)
```

对于所有使用当前图像的 task，冻结的 Stage 1 teacher 同时生成：

```text
target_latent_action: (B,16,1024)
```

两者计算 MSE。这个损失保证 Qwen query 保留 Stage 1 已学到的视觉变化和物理变化语义，
而不是只依赖视频生成 loss。

### 5.4 原生 Qwen K/V 导出

默认从 Qwen 36 个 decoder layers 中均匀选取 8 层：

```text
0, 5, 10, 15, 20, 25, 30, 35
```

每层导出：

```text
key:   (B,8,L,128)
value: (B,8,L,128)
```

实现没有绕过 decoder block。代码正常调用完整 layer forward，并通过临时 hooks 捕获：

- `k_norm` 输出；
- `v_proj` 输出。

key 随后应用 Qwen 原生 rotary embedding。这样既保留原生 K/V 几何和梯度，也兼容
classic FSDP block wrapping。

### 5.5 Understanding 训练范围

配置类默认：

```text
understanding_tuning_mode = lora
text LoRA layers          = 0
vision LoRA layers        = 4
```

正式 launcher 默认覆盖为：

```text
understanding_tuning_mode = full
```

这里的 `full` 表示：

- 36 个 `language_model.layers` 全量训练；
- token embedding 冻结；
- language final norm 冻结；
- text attention 不挂 LoRA；
- vision base 冻结；
- vision 最后 4 层 attention 的 `qkv/proj` 使用 rank-16 LoRA。

---

## 6. Physical Encoder

Physical Encoder 位于 Stage 1 的：

```text
lerobot/common/policies/ace/modeling_robo_contrast.py
```

### 6.1 Canonical state/action

每个机器人都映射到固定的 40 维 canonical slot：

```text
state:  (B,32,40)
action: (B,32,40)
```

缺失维度由 `state_mask/action_mask` 标记。投影输入不是单独的 value，而是：

```text
[value * mask, mask]
```

因此模型能区分：

- 真实测量值恰好为 0；
- 该 embodiment 根本不存在该维度。

### 6.2 分组 token

32 个 timestep 每 4 个组成一组：

```text
num_groups = 32 / 4 = 8
```

主要 token 布局：

```text
CLS                       1 token
state                     8 tokens
action                    8 tokens
tactile signal            8 tokens
tactile image             max_views × tokens_per_pad = 6 × 2
```

Physical Encoder 输出 hidden size 1024 的 contextual tokens。

### 6.3 触觉设计

触觉图像不会展开成数百个 patch tokens，而是每个 pad 压缩为 1–2 个 token，避免高维
触觉图像主导 state/action。

触觉还有两个从 0 开始的可学习 gate：

```text
tactile_signal_gate
tactile_image_gate
```

训练开始时模型等价于 state/action-only，只有触觉真正降低 loss 时，gate 才会逐渐打开。

当前 Stage 2 默认：

```text
physical_tuning_mode = frozen
```

---

## 7. 视频 VAE 与 video tokens

Stage 2 使用 `Cosmos3-Edge` 目录中的 Wan causal VAE encoder，参数冻结。

输入：

```text
(B,9,3,256,256)
```

VAE 输出：

```text
(B,48,3,16,16)
```

应用 checkpoint 自带的 latent mean/std 标准化，再做 2×2 spatial patchify：

```text
3 × 16 × 16
    ↓ 2×2 patchify
3 × 8 × 8 = 192 tokens
```

每个 token 的宽度：

```text
48 channels × 2 × 2 = 192
```

同时生成三维位置：

```text
(temporal, row, column)
```

VAE 只在训练视频任务时使用。当前 action-only 推理不需要运行 VAE。

---

## 8. Generation Expert

实现位于：

```text
lerobot/common/policies/qwen3vl_mot/modeling_generation.py
```

### 8.1 参数配置

| 项目 | 数值 |
|---|---:|
| hidden size | 2048 |
| depth | 28 |
| query heads | 16 |
| KV heads | 8 |
| head dim | 128 |
| intermediate size | 9216 |
| activation | ReLU² |
| 参数量 | 1,429,469,696 |

Generation 的 KV 几何必须和 Qwen 原生 K/V 一致：

```text
8 KV heads × 128 head dim
```

Generation 可以使用 16 个 query heads，因为 GQA 要求的只是 query heads 能被 KV heads 整除。

### 8.2 Modality stream

| stream | 输入 token 宽度 | 输出宽度 |
|---|---:|---:|
| video | 192 | 192 |
| state | 1024 | `4×40=160` |
| action | 1024 | `4×40=160` |
| tactile | 1024 | 1024 |

每个 stream 都会加入：

- modality embedding；
- task embedding；
- sigma/timestep embedding；
- 一维或三维位置编码。

所有存在的 stream 随后沿 token 维拼接。

### 8.3 非对称注意力

每个 Generation block 计算：

```text
Q = Generation Q
K = concat(Qwen native K, Generation K)
V = concat(Qwen native V, Generation V)
```

即：

```text
Generation <- Qwen understanding
Generation <- Generation modalities
Qwen        <-/ Generation
```

Generation attention 不使用 causal mask，所以 video、state、action 和 tactile token
可以双向交互。

28 个 Generation blocks 会按深度映射到 8 组 Qwen K/V：

```text
context_index = floor(block_index × 8 / 28)
```

因此 Generation 不只是读取 Qwen 最后一层，而是逐深度读取从浅层到深层的多级语义。

### 8.4 输出头

每个 modality 有独立 output head，初始化为全 0：

```text
video head
state head
action head
tactile head
```

零初始化使模型开始训练时预测接近零速度，避免随机大输出破坏 flow objective。

---

## 9. Task family

任务角色定义位于：

```text
lerobot/common/policies/qwen3vl_mot/tasks.py
```

### 9.1 ModalityRole

| role | 含义 |
|---|---|
| `ABSENT` | 不进入 Generation sequence |
| `CLEAN` | 作为干净条件，sigma=0，不计算该模态 loss |
| `NOISY` | 整个模态加噪并预测 |
| `FUTURE_NOISY` | 当前部分保持干净，只预测未来部分 |
| `CURRENT_ONLY` | 只保留当前 state，并重复到整个 chunk |

### 9.2 当前任务

| 任务 | Understanding 条件 | 干净 Generation 条件 | Flow target |
|---|---|---|---|
| `t2v` | text | 无 | 完整 video |
| `i2v` | image + available text | video 当前 latent frame | 未来 video |
| `forward_dynamics` | image + available text | action | 未来 video + 未来 state |
| `inverse_dynamics` | image + available text | video + state | action |
| `action_prediction` | image + available text | current state | action |
| `state_prediction` | image + available text | video + action | 未来 state |
| `tactile_prediction` | image + available text | video + state + action | tactile feature |

默认任务权重：

```text
t2v                 0.10
i2v                 0.20
forward_dynamics    0.20
inverse_dynamics    0.15
action_prediction   0.15
state_prediction    0.10
tactile_prediction  0.10
```

### 9.3 分布式任务选择

每个 rank 先统计当前 batch 对每个任务的有效样本数，再做 all-reduce。rank 0 只在全局
存在有效样本的任务中按权重采样，并将 task id broadcast 给所有 rank。

因此同一个 distributed step 上，所有 GPU 执行同一个 task，但每个 rank 可以有不同数量的
有效 rows。loss 使用全局有效元素数归一化。

---

## 10. Flow Matching

视频、state、action 和 tactile 都使用 rectified flow。

对干净数据 `x` 和高斯噪声 `ε`：

\[
x_\sigma=(1-\sigma)x+\sigma\epsilon
\]

目标速度：

\[
v^\star=\epsilon-x
\]

模型预测：

\[
\hat v_\theta(x_\sigma,\sigma,\mathrm{condition})
\]

然后在有效 mask 上计算 MSE。

### 10.1 Sigma 解耦

video、state、action 和 tactile 分别调用 `_sample_sigma()`，因此不同模态的 sigma
相互独立，不只训练在同一个 sigma 对角线上。

### 10.2 FUTURE_NOISY

视频：

- 第一个 VAE temporal latent frame 保持干净；
- 该帧不计算 video loss；
- 其余 latent frames 加噪并预测。

state：

- `t=0` state 保持干净；
- 其余 31 个 state timestep 加噪并预测。

### 10.3 总损失

任务存在的 flow targets 参与：

\[
L_\mathrm{flow} =
\lambda_v L_\mathrm{video}
+ \lambda_a L_\mathrm{action}
+ \lambda_s L_\mathrm{state}
+ \lambda_t L_\mathrm{tactile}
\]

默认：

```text
video   1.0
action  1.0
state   1.0
tactile 0.25
```

所有包含当前图像的任务还加入：

\[
L_\mathrm{latent}
=
\mathrm{MSE}
\left(
\mathrm{Proj}(q_\mathrm{Qwen}),
q_\mathrm{Stage1}
\right)
\]

最终：

\[
L=L_\mathrm{flow}+\lambda_\mathrm{latent}L_\mathrm{latent}
\]

其中 `latent_action_loss_weight` 默认是 `1.0`。

---

## 11. Action 推理

当前已实现：

```python
sample_canonical_action(batch)
```

流程：

1. 从高斯噪声初始化 `(B,32,40)` canonical action；
2. 用 `action_mask` 清零不存在的维度；
3. 将当前 state 重复到 32 个 timestep；
4. 通过 frozen Physical Encoder 得到 state/action tokens；
5. 使用 `action_prediction` task；
6. 从 `sigma=1` 到 `0` 做默认 20 次 Euler 更新；
7. 每次更新后重新应用 action mask；
8. 返回前 `n_action_steps`，当前为 32。

更新公式：

\[
x_{\sigma_{next}}
=
x_\sigma
+
(\sigma_{next}-\sigma)
\hat v_\theta
\]

### 11.1 尚未实现的部署层

`select_action()` 当前主动抛出 `NotImplementedError`，因为仍缺少：

- canonical 40D 到具体机器人 action layout 的反映射；
- 数据集归一化的逆变换；
- embodiment-specific controller 接口；
- action chunk 执行和重规划策略。

当前也没有完整的多步 video sampler 和 VAE pixel decoder 推理接口。

---

## 12. 参数量和正式训练范围

### 12.1 主要参数

| 参数集合 | 参数量 |
|---|---:|
| Qwen3-VL-4B base | 4,437,815,808 |
| Generation Expert | 1,429,469,696 |
| Vision LoRA | 393,216 |
| Latent queries | 40,960 |
| Latent-action projection | 2,629,632 |

这里只统计核心 Qwen 和 Generation。运行时还会加载冻结的 Physical Encoder、
Stage 1 perception teacher 和 Wan VAE。

### 12.2 正式 launcher 的训练范围

`train_qwen3vl_mot_fsdp.sh` 默认：

| 模块 | 状态 |
|---|---|
| Qwen `language_model.layers` | 全量训练 |
| Qwen token embedding | 冻结 |
| Qwen final norm | 冻结 |
| Vision base | 冻结 |
| Vision 最后 4 层 attention | rank-16 LoRA |
| Generation Expert | 全量训练 |
| Latent queries/projection | 全量训练 |
| Physical Encoder | 冻结 |
| Stage 1 teacher | 冻结 |
| Wan VAE | 冻结 |

总可训练参数：

```text
5,066,042,880
```

optimizer parameter groups：

```text
main          lr = 1e-4
understanding lr = 1e-4 × 0.05 = 5e-6
physical      lr = 1e-4 × 0.3（仅 physical full tuning 时存在）
```

---

## 13. FSDP、FP8 与 optimizer

正式训练路径：

```text
train_qwen3vl_mot_fsdp.sh
  -> lerobot/scripts/fsdp_train_contrast.py
```

默认配置：

```text
8×H100
micro-batch/GPU = 16
global batch = 128
classic FSDP FULL_SHARD
use_orig_params = true
TorchAO FP8 scope = generation_vlm
FP8 recipe = rowwise_with_gw_hp
bitsandbytes AdamW8bit
understanding checkpointing = true
generation checkpointing = true
```

FP8 的实际语义：

- trainable master parameter 保持 BF16；
- eligible Linear GEMM 动态使用 FP8；
- optimizer-facing gradient 保持 BF16；
- FSDP all-gather 当前仍是 BF16；
- checkpoint 保存的是 BF16 master weights，不是永久 FP8 权重。

冻结参数默认在每张 GPU 复制，避免 teacher、VAE 和冻结 vision 权重每次 forward 都触发
FSDP all-gather。

当前真实模型中：

```text
FP8 converted Linear = 431
FSDP wrapped blocks = 64
```

64 个 wrapped blocks 对应主要的 36 个 Qwen decoder blocks 和 28 个 Generation blocks。

---

## 14. 当前限制与后续方向

### 14.1 三视角尚未接入

需要将输入扩展为：

```text
images_t0:   (B,3,C,H,W)
camera_mask: (B,3)
```

三个槽位应对应 `primary / secondary / wrist`，但还需要额外相机语义，因为 `secondary`
在不同数据集中可能是第二外部相机，也可能是左腕相机。

第一版计划只把三路当前图像作为 Understanding 条件，不立即为所有视角增加未来 Qwen
feature loss。未来视频仍优先生成 primary。

### 14.2 Video inference 尚未完成

训练具备 video flow objective，但没有：

- video ODE/Euler sampler；
- latent unpatchify；
- VAE decoder 输出视频；
- classifier-free guidance。

### 14.3 Tactile 是 feature prediction

当前 tactile task 预测 Physical Encoder 产生的 1024 维 tactile tokens，不是直接预测
未来触觉原始图像或低维信号。

当前 `use_tactile_conditioning=False`，且现有任务中没有把 tactile 标为 `CLEAN`，因此 tactile
只在 `tactile_prediction` 中作为预测目标，尚未作为 video、state 或 action 任务的条件。

### 14.4 Teacher 只在训练时需要

Stage 1 perception teacher 和 Wan VAE 增加训练开销，但 canonical action 推理不依赖它们。
部署 checkpoint 可以在明确不需要 video generation 和 latent distillation 后裁剪。

---

## 15. 模型变更检查清单

每次修改模型后，至少检查并更新本文以下内容：

- [ ] 输入字段和 tensor shape；
- [ ] 相机视角和 camera mask；
- [ ] Qwen token 数、query 数和 K/V 层数；
- [ ] Generation hidden/depth/heads/intermediate；
- [ ] Physical group size 和 canonical dimensions；
- [ ] VAE frames、latent shape 和 patch size；
- [ ] task roles、采样权重和 valid-row 条件；
- [ ] sigma 是否独立、哪些 token 保持 clean；
- [ ] loss 项和 loss 权重；
- [ ] Stage 1 迁移覆盖率和 contract；
- [ ] trainable/frozen 参数范围；
- [ ] 总参数和可训练参数量；
- [ ] action/video inference 接口；
- [ ] FSDP wrap、FP8 scope 和 optimizer 语义；
- [ ] `tests/test_qwen3vl_mot.py` 中对应的结构回归测试。

相关文档：

```text
doc/stage2_train_cost.md
doc/problem_and_solution.md
```

---

## 16. 变更记录

| 日期 | 模型实现基线 | 说明 |
|---|---|---|
| 2026-09-23 | `f99e22a` | 首次完整记录 Qwen Understanding、Stage 1 继承、Generation Expert、task family、flow loss、FSDP/FP8 训练范围和当前限制 |