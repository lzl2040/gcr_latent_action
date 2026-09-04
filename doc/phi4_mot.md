# Phi-4-Multimodal MoT 世界模型（分支 `phi4_mot`）

本分支复刻 Cosmos3 的 Mixture-of-Transformers（MoT）条件生成结构：

- 理解侧（UND）使用完整的 `microsoft/Phi-4-multimodal-instruct` 视觉语言路径；
- 生成侧（GEN）是从零训练的 1.42B MoT expert；
- GEN 每一层读取同层 UND 的 K/V，可同时做视频流匹配、动作流匹配和动力学任务；
- 原始视频由冻结的 Wan VAE 在线编码，也可以预先缓存 latent。

旧版本实际是 **Qwen3-VL ViT + 自定义 merger + 纯文本 Phi-4-mini**。这不是
Phi-4-Multimodal，已经全部替换；Qwen 参数和 `QWEN_DIR` 不再参与当前模型。

---

## 1. 模型结构

### 1.1 MoT 的耦合方向

每层有两套参数：

```text
UND: causal self-attention(UND)
GEN: full attention(query=GEN, key/value=[UND, GEN])
```

因此：

1. UND 不读取 GEN，但训练时默认仍按 Cosmos3 的顺序逐层交错执行；
2. GEN 每层必须接收对应 UND 层导出的 pre-RoPE K/V；
3. 两侧只必须共享 `head_dim=128` 和 `num_key_value_heads=8`；
4. GEN 的 hidden size、query heads 和 MLP size可以独立设置。

训练和推理保留两种数学等价的执行顺序：

```text
# training_execution=interleaved（训练默认）
for layer in layers:
    und, k_und, v_und = layer.und_forward(und)
    gen = layer.gen_forward(gen, k_und, v_und)

# cached（推理默认，也可用于 A/B）
for layer in layers:
    und, k_und, v_und = layer.und_forward(und)
    kv_cache.append((k_und, v_und))
for layer, (k_und, v_und) in zip(layers, kv_cache):
    gen = layer.gen_forward(gen, k_und, v_und)
```

训练默认的 interleaved 路径与 Diffusers `Cosmos3VLTextMoTDecoderLayer.forward()` 一致：
同层 UND K/V 立即供同层 GEN 使用，用完即可释放。cached 路径仍完整保留，适合固定
image/text 条件下跨多个 denoising step 复用。

当前 GEN 取：

| 配置 | 值 |
|---|---:|
| 层数 | 32 |
| hidden size | 2048 |
| query heads | 16 |
| KV heads | 8 |
| head dim | 128 |
| MLP intermediate | 7680 |
| action dim | 40 |
| embodiment domains | 32 |

### 1.2 使用了 Phi-4-Multimodal 的哪些权重

官方 checkpoint 共约 5.574B 参数，其中还包含音频编码器和 speech LoRA。当前机器人模型
只加载视觉语言路径：

| 官方权重组 | 参数 | 当前模型 |
|---|---:|:-:|
| language base（embedding、32 层、norm） | 3.836B | ✓ |
| vision LoRA | 369.099M | ✓ |
| SigLIP NaViT + image projector | 441.550M | ✓ |
| audio encoder / projection | 466.416M | ✗ |
| speech LoRA | 461.373M | ✗ |

选择性 safetensors loader 只读取目标 tensor，不会先把音频权重整体载入内存。

### 1.3 官方视觉路径

视觉塔参数：

| 配置 | 值 |
|---|---:|
| 输入 | 448×448 RGB |
| patch | 14×14 |
| 初始网格 | 32×32 = 1024 token |
| hidden size | 1152 |
| 层数 | 27 |
| heads | 16 |
| MLP | 4304 |
| 取特征层 | `-2` |
| 压缩 | 2×2 average pool |
| 压缩后网格 | 16×16 = 256 token |

官方 dynamic-HD 的方形单图包含一个 global crop 和一个 sub crop。每个 16×16 特征块在每行
后插入一个可学习分隔符：

```text
16 x (16 image token + 1 row separator) = 272
sub block + 1 global separator + global block = 272 + 1 + 272 = 545 token
```

机器人 loader 当前给出方形图像，所以 global crop 与 sub crop 的像素完全相同。实现只跑
一次 SigLIP，再复用特征构造两块 272-token 序列；FP32 下与官方跑两次相同 crop 的输出相对
误差为 `2.7e-6`。

另外裁掉了官方 forward 中不会被 `feature_layer=-2` 使用的最后一个视觉层、post norm 和
pooling head。它们的 checkpoint 权重仍然加载并计入 resident parameters，但不参与前向，
也不会被误放进 optimizer。

> 当前快速路径的前提是输入已被数据管线变成方形。若以后直接输入任意宽高原图，应恢复官方
> dynamic-HD 多 crop 和 attention mask，而不是把长方形无条件拉伸成 448×448。

### 1.4 Vision LoRA

Phi-4-Multimodal 不只是“SigLIP + Phi”。语言主干每层还有官方 vision adapter：

| 参数 | 值 |
|---|---:|
| rank | 256 |
| alpha | 512 |
| scale | 2.0 |

每层作用于四个 projection：

```text
self_attn.qkv_proj
self_attn.o_proj
mlp.gate_up_proj
mlp.down_proj
```

行为与官方 `InputMode` 一致：

- 有图任务：启用 vision LoRA；
- `t2i` / `t2v` 纯文本任务：完全关闭 vision LoRA。

只加载视觉塔而不加载这 369.1M adapter，不能称为完整 Phi-4-Multimodal 理解分支。

### 1.5 位置编码

Phi-4-Multimodal 的 image token 和 text token 在语言主干里都使用普通一维 LongRoPE。
因此 UND 侧把三条 mRoPE 轴都设成同一个递增位置：

```text
t = h = w = arange(sequence_length)
```

这样三段频率看到相同位置，严格退化为原始一维 RoPE。LongRoPE 的短序列 factor 全为 1，
同时保留官方 `attention_scaling=1.190238`。

GEN 侧继续使用视频 latent 的三维位置：

- 时间轴对应 latent frame；
- 空间轴对应 8×8 patch 网格；
- action token 接在视频位置之后作为一维序列。

### 1.6 为大 batch 做的内存优化

一张图会把 UND 长度从 32 个文本 token 提高到约 577。Phi 的 fused MLP 在 batch 128 时会
产生很大的临时张量。训练时 `WorldModelConfig.und_microbatch_size=32` 会沿 batch 维同时
切分 **UND 和 GEN**：

- 外部/global batch 不变；
- 每个 microbatch 完整经过同一组 UND/GEN 层；
- loss、负样本和 optimizer step 语义不变；
- batch 内样本没有相互 attention，因此切分在数学上等价；
- BF16 因 GEMM kernel 形状不同可能有舍入差异，回归测试 cosine 为 `0.99999`。

训练默认把连续 8 个双路径层作为一个 checkpoint segment。层内仍保持 Cosmos3 的
UND→同层 GEN 顺序；反向只保留 segment 边界并重算段内层，而不是让全部 32 层 K/V 或全部
32 个 UND hidden 存活到 backward。以 batch 32、577 个 UND token 计算，单是原始 BF16
K/V cache 就约 2.25 GiB，还不含 autograd graph。

推理或显式设置 `training_execution="cached"` 时仍走旧路径：只对 UND 做 microbatch，再按层
合并 K/V，供一次或多次 `forward_gen()` 使用。

最后一个 UND 层只需要导出 K/V；它的 attention output、MLP 和最终 norm 不会被 GEN 或 loss
读取。两种执行路径都跳过这些死计算，并冻结 94.378M 永远不可能收到梯度的参数。

---

## 2. 参数量与可训练范围

真实数据使用 `action_dim=40` 时：

| 模块 | resident parameters |
|---|---:|
| SigLIP NaViT | 428.565M |
| 官方 image projector + separators | 12.985M |
| Phi language base + vision LoRA | 4.205B |
| 完整 UND | **4.647B** |
| GEN（含视频/action/time heads） | **1.420B** |
| **总模型** | **6.067B** |
| 冻结 Wan VAE（模型外） | 704.7M |

`trainable_scope`：

| scope | SigLIP | projector | language + vision LoRA | GEN | 实际可训练 |
|---|:-:|:-:|:-:|:-:|---:|
| `gen_only` | ✗ | ✗ | ✗ | ✓ | **1.420B** |
| `freeze_vision` | ✗ | ✓ | ✓ | ✓ | **5.544B** |
| `all` | ✓ | ✓ | ✓ | ✓ | **5.942B** |

说明：

- `gen_only` 是 Cosmos3 风格。官方 projector 已经预训练，不再像旧自定义 merger 那样默认训练；
- `freeze_vision` 对应用户要求的 π0.5 风格：冻结视觉 encoder，训练其余所有有效路径；
- `all` 还训练实际被 `feature_layer=-2` 使用的 26 个 SigLIP 层；
- resident 总参数始终是 6.067B；“实际可训练”排除了官方 checkpoint 中不会进入当前 loss
  路径的视觉尾层/head和 UND 最后 post-KV 子层；
- `freeze_vision_projector=True` 可以在任何 scope 上额外冻结 projector。

---

## 3. 视频和 action 怎么训练

两条流都使用 rectified-flow velocity objective：

```text
sigma ~ U(0, 1)
x_sigma = (1 - sigma) * x + sigma * noise
target = noise - x
loss = MSE(model(x_sigma, sigma), target)
```

视频和动作分别采样自己的 `sigma_video` / `sigma_action`，不能共用一个 sigma。共用时只覆盖
二维噪声平面的对角线，小 sigma 的干净未来画面会让动作预测退化为逆动力学；部署时又没有
未来帧，训练条件和推理条件不一致。

每条流有三种角色：

- `noisy`：加噪，是预测目标，有 loss；
- `clean`：不加噪，只作为条件，没有 loss；
- `absent`：完全不进入 GEN 序列。

| 任务 | context latent | UND 图像 | video | action | GEN token |
|---|---:|:-:|---|---|---:|
| `t2i` | 0 | ✗ | noisy（1 帧） | absent | 64 |
| `t2v` | 0 | ✗ | noisy（3 帧） | absent | 192 |
| `i2v` | 1 | ✓ | noisy | absent | 192 |
| `v2v` | 2 | ✓ | noisy | absent | 192 |
| `joint_action` | 1 | ✓ | noisy | noisy | 224 |
| `fwd_dyn` | 1 | ✓ | noisy | clean | 224 |
| `inv_dyn` | 1 | ✓ | clean | noisy | 224 |
| `policy` | 1 | ✓ | absent | noisy | **96** |

action 处理：

```text
noisy_action
  -> per-domain action_proj_in(40 -> 2048)
  -> + action_modality_embed
  -> 与视频 token 拼接
  -> 32 层 GEN
  -> per-domain action_proj_out(2048 -> 40)
  -> velocity loss
```

32 个 embodiment domain 各有独立输入/输出投影，避免 xyz+ort6d+gripper 与 joint 空间的同一列
在不同机器人上表达不同含义。canonical action 的空位由数据层 mask/padding 统一处理。

`policy` 完全删除未来帧，只保留当前 latent 的 64 token 和 32 个 action token，因此训练条件
与部署一致；不是把未来帧设成“小噪声”或偷偷提供干净目标帧。

---

## 4. `MIX` 的含义和比例

### 4.1 `MIX` 采样什么

`MIX` 是 **每个 optimizer step 为整个 batch 选择任务的分类分布**，不是在一个 batch 内按
比例拆样本：

```text
p(task=t) = weight[t] / sum(weight)
task ~ Categorical(p)
loss_step = loss_task(whole_batch)
```

因此 `policy=0.20` 表示长时间平均约 20% 的 step 使用 `policy`，且这些 step 的整个 batch
都是 `policy`。

它与另外两个参数不同：

| 参数 | 控制内容 |
|---|---|
| `MIX` | 各任务出现为一个 step 的频率 |
| `action_loss_weight` | 同一个 action step 内 action loss 相对 video loss 的权重 |
| `data_mix` | batch 从哪些机器人数据集读取 |

`joint_action` 的总 loss 是：

```text
loss_video + action_loss_weight * loss_action
```

提高 `joint_action` 的 MIX 权重会增加这种 step 的数量；提高 `action_loss_weight` 才会放大
每次出现时的动作梯度。

### 4.2 预设比例

| preset | 比例 |
|---|---|
| `stage2` | t2i .25 / t2v .25 / i2v .30 / v2v .20 |
| `stage3` | policy .20 / joint_action .15 / inv_dyn .10 / fwd_dyn .05 / i2v .20 / v2v .15 / t2v .10 / t2i .05 |
| `stage3_joint_only` | joint_action .50 / i2v .20 / v2v .15 / t2v .10 / t2i .05 |
| `action_only` | policy .50 / joint_action .20 / inv_dyn .20 / fwd_dyn .10 |

`stage3` 先分成 50% action + 50% generation：

| 大类 | 任务 | 全部 step | 类内占比 | 原因 |
|---|---|---:|---:|---|
| action | `policy` | 20% | 40% | 唯一完全匹配部署条件 |
| action | `joint_action` | 15% | 30% | 联合视频/action，但不独占 action 预算 |
| action | `inv_dyn` | 10% | 20% | 单独学习观察变化到动作 |
| action | `fwd_dyn` | 5% | 10% | 用动作预测后果的辅助动力学 |
| generation | `i2v` | 20% | 40% | 最接近机器人条件生成 |
| generation | `v2v` | 15% | 30% | 保留多帧续写 |
| generation | `t2v` | 10% | 20% | 保留文本视频生成 |
| generation | `t2i` | 5% | 10% | 基础任务，但离机器人 action 最远 |

这些是工程起点和消融基线，不是 Cosmos 官方公布的最优比例。

### 4.3 配置示例

```bash
# 正式训练：必须关闭逐任务轮询
PER_TASK=0 MIX=stage3 bash scripts/bench_mot_scope.sh

# 相对权重会自动归一化为 .6/.2/.2
PER_TASK=0 MIX="policy=3,fwd_dyn=1,i2v=1" bash scripts/bench_mot_scope.sh
```

- `PER_TASK=0`：按 MIX 随机采样，是真正训练模式；
- `PER_TASK=1`：固定轮询全部 8 个任务，只用于 benchmark，完全忽略 MIX。

---

## 5. 数据与 Wan VAE

`train_mot_world.py` 在真实 `debug_research_data` 上使用：

- 9 个原始 RGB 帧；
- Wan VAE 压成 3 个 latent 帧；
- tactile 默认关闭，但 dataset 仍支持有/无触觉以及异构触觉；
- canonical action width = 40；
- task text 使用 Phi-4-Multimodal tokenizer。

Wan 的时间压缩满足：

```text
pixel frames = 1 + 4 * (latent_frames - 1)
```

所以 3 latent 对应 9 个原视频帧。长度不满足 `1+4k` 时可能被 VAE 静默截断，训练脚本会按
latent frame 数反推需要读取的 RGB 帧数。

真实 mixture 当前成功加载：

```text
fractal20220817_data
taco_play
ftp_1_sharpa
ftp_1_VisuoTactile_D-WHEEL
ftp_1_exUMI
ftp_1_RH20TCfg5Franka
ftp_1_RDP_Bimanual
```

---

## 6. 正确性验证

### 6.1 官方 Phi-4-Multimodal 数值对齐

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/check_mot_und.py --fp32 --seq 2
```

| 检查 | 相对误差 | cosine |
|---|---:|---:|
| SigLIP + projector 的 545 image embeddings | `2.685e-6` | ≈ 1.0 |
| language mode（LoRA 关闭） | `5.198e-6` | 1.0 |
| vision mode（vision LoRA 开启） | `9.520e-6` | ≈ 1.0 |

视觉模块 state dict 与官方 checkpoint 的 454 个 key 为 **454/454 完全匹配**。语言侧每层
导出 K/V，language 和 vision 模式均为 32 层。

BF16 下 fused SDPA 与官方 eager reference 会累积不同舍入，但三项仍通过：

```text
image cosine    0.99970
language cosine 0.99994
vision cosine   0.99802
```

FP32 结果证明结构等价；BF16 差异来自 kernel 数值路径，而不是漏权重或错 token 布局。

### 6.2 两种执行顺序等价

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/check_mot_execution.py
```

小型三层 MoT 对照结果：

```text
cached vs interleaved output max diff = 0
cached vs interleaved gradient max diff = 0
checkpointed interleaved output max diff = 0
batch slicing cosine = 0.999987
frozen UND 没有梯度，GEN 梯度正常
```

这证明执行顺序的改变没有改变 MoT 数学关系。batch slicing 的非零差异来自 BF16 GEMM 在
不同 batch shape 下选择不同 kernel；FP32 或不切 batch 时 cached/interleaved 逐元素一致。

### 6.3 8 个任务

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/check_mot_tasks.py \
  --batch 1 --text_len 4 --action_len 4 --random_init
```

全部任务通过：

- context latent 放大 1000 倍后没有泄漏进 video loss；
- `clean` / `absent` 流不产生对应 loss；
- t2i/t2v 完全不调用 vision；
- action/video 需要的 projection 均收到梯度。

### 6.4 trainable scope

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/check_trainable_scope.py --layers 1
```

三个 scope 的静态 `requires_grad` 和动态梯度覆盖均通过。检查要求每一个被标记为 trainable
的有效 tensor 都必须实际收到梯度，而不只是“这个组里至少有一个梯度”。

### 6.5 完整模型与 batch 128

新的 interleaved 路径已在 `gen_only`、完整 6.070B 模型上通过 batch 128：

```text
batch 128
training_execution = interleaved
checkpoint segment = 8 layers
完整 AdamW optimizer step 通过
当前进程 peak = 22.1 GiB
所有 366 个可训练 tensor 有梯度
```

测试时该卡还被另一个进程占用约 19 GiB，因此使用 microbatch 1 才能在共享卡上完成，测得的
42.1 s 不能作为吞吐基准；它只证明包含 AdamW 的 batch 128 完整路径可以运行。旧 cached
路径在独占 A6000、microbatch 32 下为 14.338 s/step、36.4 GiB，二者不是同条件速度对比。
原视频端到端还要运行 705M Wan VAE，正式基准仍使用 batch 32/64。

---

## 7. RTX A6000 cached 基线

> 本节数值是在切换到 interleaved 训练前测得的 `training_execution=cached` 基线。当前四张
> A6000 均被其他进程占用，不能可靠刷新绝对吞吐；这些数值不能直接当作新的默认训练速度。
> 同卡受干扰 A/B 显示：冻结 UND 的 `gen_only` interleaved checkpoint 会因反向重算 UND 而
> 更慢；`freeze_vision` 两种路径接近。需在独占 GPU 上重新测量后再刷新集群外推。

### 7.1 forward-only 阶段拆解

`profile_mot_world.py --batch 8`：

| 阶段 | 时间 | 估算有效吞吐 |
|---|---:|---:|
| SigLIP + projector | 98.8 ms | — |
| UND KV stack | 425.9 ms | 75.8 TFLOP/s |
| GEN stack | 107.0 ms | 47.7 TFLOP/s |

UND 有约 577 token，GEN 联合任务只有 224 token，所以即使两侧参数量接近，image-conditioned
任务仍明显由 UND 主导。

### 7.2 batch 32 前向/反向分解

`joint_action`、合成 tensor：

| scope | forward no-grad | forward grad | forward+backward | + optimizer |
|---|---:|---:|---:|---:|
| `gen_only` | 2445 ms | 2446 ms | 3595 ms | 3699 ms |
| `freeze_vision` | 2417 ms | 2416 ms | 8699 ms | 单卡放不下完整 optimizer |

两种 scope 的 forward 基本相同，因为都必须跑完整 Phi-4-MM；`freeze_vision` 的额外成本主要
在 language base、vision LoRA 和 projector 的反向。

### 7.3 真实数据、batch 32

`debug_research_data`，9 原视频帧 → 3 latent，`PER_TASK=1`：

| 任务 | `gen_only` model | `freeze_vision` model |
|---|---:|---:|
| `t2i` | 616 ms | 767 ms |
| `t2v` | 1351 ms | 1513 ms |
| `i2v` | 3804 ms | 8883 ms |
| `v2v` | 3607 ms | 8900 ms |
| `joint_action` | 3810 ms | 9173 ms |
| `fwd_dyn` | 3809 ms | 9092 ms |
| `inv_dyn` | 3816 ms | 9080 ms |
| `policy` | 3052 ms | 8351 ms |
| **按 stage3 MIX 加权** | **3222 ms** | **7710 ms** |

说明：

- `gen_only` 包含 fused AdamW step；
- `freeze_vision` 使用 `NO_OPT=1`，表中是 forward+backward；集群外推额外加每卡约 30 ms
  的 ZeRO-2 optimizer shard 估计；
- 在线 Wan VAE 在 batch 32 为 **1306 ms**；
- peak memory：`gen_only` 26.9 GiB；`freeze_vision` 25.6 GiB（无 optimizer）；
- `freeze_vision` 有 5.544B 有效可训练参数，单卡完整 Adam state 不适合 48 GiB，正式训练
  应使用 ZeRO-2。

补充锚点：

| 配置 | 时间/显存 |
|---|---|
| `gen_only`, batch 64, 真实原视频 | 35.8 GiB peak |
| `freeze_vision`, batch 64, synthetic, no optimizer | 17.48 s fwd+bwd，34.3 GiB |
| `gen_only`, batch 128, synthetic latent | 14.338 s/step，36.4 GiB |

---

## 8. 16 卡 A100 / B200 cached 基线

以下时间仍由第 7 节 cached A6000 锚点推得，不代表新的 interleaved 默认训练路径。

### 8.1 口径

这里严格使用用户指定的“一个 batch sample 过一帧”口径：

```text
frames = hours * 60 * 60 * 20
global batch = 16 * 32 = 512
steps = frames / 512
```

所以：

| 数据量 | 帧数 | optimizer steps |
|---|---:|---:|
| 1 万小时 | 7.20e8 | 1.40625e6 |
| 3 万小时 | 2.16e9 | 4.21875e6 |

不是按 1.6 秒 window 数量除 batch，也不是把 9 帧 clip 当成 1 个小时口径 sample。

### 8.2 原视频路径的假设

表中数据是**原视频，不是预提 latent**：

- 每卡 batch = 32；
- 每个节点 8 卡，共 2 节点；
- 挂载存储单 clip 延迟假设 200 ms；
- 每卡 12 loader workers → 60 clips/s；
- batch 32 的 loader floor = `32/60 = 533 ms/step`；
- 在线 Wan VAE 使用 A6000 实测 1306 ms/batch 32，并按设备效率外推；
- A100/B200 使用可达到效率区间，不直接套宣传峰值；
- `freeze_vision` 用 ZeRO-2，`gen_only` 用 DDP；
- 通信模型包含节点内 NVLink、节点间 NIC、反向 overlap 和最后不可隐藏 bucket。

### 8.3 8×200G IB

| scope | 设备 | 1 万小时 | 3 万小时 | 主要瓶颈 |
|---|---|---:|---:|---|
| `gen_only` | 16× RTX A6000 | 73.7 d | 221.2 d | compute |
| `gen_only` | 16× A100-40GB | 20.3–24.4 d | 61.0–73.2 d | compute |
| `gen_only` | 16× A100-80GB | 20.3–24.4 d | 61.0–73.2 d | compute |
| `gen_only` | 16× B200 | **8.7 d** | **26.0 d** | raw-video I/O |
| `freeze_vision` | 16× RTX A6000 | 147.3 d | 442.0 d | compute |
| `freeze_vision` | 16× A100-40GB | 40.7–48.8 d | 122.1–146.4 d | compute |
| `freeze_vision` | 16× A100-80GB | 40.7–48.8 d | 122.1–146.4 d | compute |
| `freeze_vision` | 16× B200 | **9.3–17.0 d** | **28.0–51.0 d** | compute / I/O |

A100-40GB 与 80GB 在固定 batch 32 下算力相同，因此时间相同；80GB 的价值是允许更大的
per-GPU batch 或减少 checkpoint/offload，而不是让同一个 batch 的 kernel 自动变快。

B200 `gen_only` 精确撞上原视频 loader floor：

```text
2.16e9 frames / (16 GPU * 60 clips/s) = 26.0 days
```

所以继续增加 B200 算力不会低于 26 天，除非减少每 clip 延迟、提高 worker 吞吐或预提 latent。

### 8.4 网络敏感性

3 万小时、100G Ethernet：

| scope | A100-40/80GB | B200 |
|---|---:|---:|
| `gen_only` | 61.5–73.7 d | 26.0–29.6 d |
| `freeze_vision` | 124.1–148.5 d | **54.8–61.3 d** |

`freeze_vision` 每步需要归约约 11.1 GB bf16 gradient。A100 的长计算能隐藏大部分通信；
B200 的 backward 很短，100G 网络无法隐藏，因此比 8×200G IB 明显变慢。

完整表可重算：

```bash
python scripts/extrapolate_cluster.py --hours 10000 --batch-per-gpu 32
python scripts/extrapolate_cluster.py --hours 30000 --batch-per-gpu 32
```

这些是**外推，不是 A100/B200 实测**。最不确定的两个量是：

1. B200 对 2048-wide GEN 矩阵的实际利用率；
2. 集群挂载盘真实的单 clip 延迟和可并发 worker 数。

---

## 9. 常用命令

```bash
conda activate lerobot_v2

# 官方数值等价
CUDA_VISIBLE_DEVICES=2 python -u scripts/check_mot_und.py --fp32 --seq 2

# 8 个任务
CUDA_VISIBLE_DEVICES=2 python -u scripts/check_mot_tasks.py \
  --batch 1 --text_len 4 --action_len 4 --random_init

# cached / interleaved 输出与梯度等价
CUDA_VISIBLE_DEVICES=2 python -u scripts/check_mot_execution.py

# scope 梯度
CUDA_VISIBLE_DEVICES=2 python -u scripts/check_trainable_scope.py \
  --layers 3 --batch 2 --microbatch 1

# 完整 pretrained smoke
CUDA_VISIBLE_DEVICES=2 python -u scripts/smoke_mot_world.py \
  --batch 1 --latent_frames 3 --gen_checkpointing \
  --execution interleaved --checkpoint_segment 8

# 明确保留的 cached 训练 A/B 路径
CUDA_VISIBLE_DEVICES=2 python -u scripts/smoke_mot_world.py \
  --batch 1 --latent_frames 3 --gen_checkpointing --execution cached

# 真实数据 benchmark
CUDA_VISIBLE_DEVICES=2 BATCH=32 STEPS=24 WARMUP=8 \
  SCOPE=gen_only PER_TASK=1 bash scripts/bench_mot_scope.sh

CUDA_VISIBLE_DEVICES=2 BATCH=32 STEPS=16 WARMUP=8 \
  SCOPE=freeze_vision NO_OPT=1 PER_TASK=1 bash scripts/bench_mot_scope.sh

# 正式按 stage3 MIX 采样
CUDA_VISIBLE_DEVICES=2 BATCH=32 SCOPE=gen_only \
  PER_TASK=0 MIX=stage3 EXECUTION=interleaved CKPT_SEGMENT=8 \
  bash scripts/bench_mot_scope.sh
```

---

## 10. 尚未完成或仍需实机确认

- 当前只有训练 forward，没有机器人部署用的多步 action denoising sampler；
- cached UND K/V 的底层接口已保留，但部署 sampler 还需负责在 denoising steps 间持有并复用；
- A100/B200 数字来自 A6000 实测外推，尚未在对应集群实测；
- 第 7、8 节仍是 cached 基线；interleaved 的独占 GPU benchmark 尚未刷新；
- `freeze_vision` 的单卡 optimizer 放不下，当前只实测前反向；正式 ZeRO-2 optimizer 时间是估计；
- 原视频 blob 延迟 200 ms/clip 是场景假设，必须在目标集群重新跑 I/O probe；
- 方形图像快速路径不等价于任意长宽比的官方 dynamic-HD；
- audio encoder 和 speech LoRA 被有意排除，当前模型是视觉语言机器人世界模型，不是完整音视频
  对话模型；
- tactile 在此版本的数据入口中是可选的，当前 MoT benchmark 为了隔离世界模型成本将其关闭。
