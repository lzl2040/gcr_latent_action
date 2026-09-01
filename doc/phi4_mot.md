# Phi-4-mini MoT 世界模型（分支 `phi4_mot`）

复刻 Cosmos3 的 Mixture-of-Transformers 训练结构，但把理解分支换成 `microsoft/Phi-4-mini-instruct`。

本文记录：为什么"换掉理解分支"不是换骨干、最终选了什么结构、各模块参数量、以及实测数据。

---

## 1. 先说清楚 Cosmos3 是怎么组织的

结论先行：**Cosmos3-Edge 的理解分支不是一个可插拔模块**，它是一个 MoT 的一半。

证据来自两处一手材料。

### 1.1 权重实测

用 HTTP range 请求只读 `transformer/` 两个分片的 safetensors 头部（不下载 5 GB 数据），得到 549 个张量。
其中 255 个被理解分支的 `model.safetensors.index.json` 引用，294 个是生成专属。逐层看：

| 层内张量 | 归属 |
|---|---|
| `input_layernorm` / `post_attention_layernorm` | UND |
| `self_attn.to_q` / `to_k` / `to_v` / `to_out` | UND |
| `mlp.up_proj` / `down_proj` | UND |
| `input_layernorm_moe_gen` / `post_attention_layernorm_moe_gen` | GEN |
| `self_attn.add_q_proj` / `add_k_proj` / `add_v_proj` / `to_add_out` | GEN |
| `norm_added_q` / `norm_added_k` | GEN |
| `mlp_moe_gen.up_proj` / `down_proj` | GEN |
| `self_attn.k_norm_und_for_gen` | UND（但只被 GEN 用到） |

非层张量里还有 `proj_in [2048,192]`、`proj_out [192,2048]`、`time_embedder`、
`action_modality_embed [2048]`、`action_proj_in.fc [32,131072]`、`action_proj_out.fc [32,131072]`。
后两个是 **32 个 embodiment 各自一份的动作投影**（131072 = 64 × 2048），
对应 `transformer/config.json` 里的 `action_gen: true`、`action_dim: 64`、`num_embodiment_domains: 32`。
也就是说 Cosmos3-Edge 原生就能生成机器人动作。

参数量核对：UND 1.95B（28×50.3M + embed 268M + lm_head 268M）+ GEN 1.42B = 3.37B，
bf16 恰为 6.74 GB，与 HF 上 `transformer/` 的实际体积一致。

### 1.2 耦合方向（关键）

`diffusers` 主分支 `transformer_cosmos3.py` 的 `Cosmos3AttnProcessor`：

```python
causal_out = dispatch_attention_fn(q_und, k_und, v_und, is_causal=True)      # UND 只看自己
all_k = torch.cat([k_und_for_gen, k_gen]); all_v = torch.cat([v_und, v_gen])
full_out = dispatch_attention_fn(q_gen, all_k, all_v, is_causal=False)       # GEN 每层都吃 UND 的 K/V
```

**UND 不依赖 GEN，但 GEN 每一层、每个头都寄生在 UND 的 K/V 上。**
所以换掉 UND，1.42B 的生成权重全部失去条件信号，必须重训——这才是"换 LLM"的真实成本。

### 1.3 Phi-4-mini 对不上的地方

| | Cosmos3-Edge UND | Phi-4-mini |
|---|---|---|
| hidden / 层数 | 2048 / 28 | 3072 / 32 |
| 注意力头 | 16（KV 8） | 24（KV 8 ✓） |
| head_dim | 128 | 128 ✓ |
| 激活 | relu²（无 gate，2 矩阵） | silu+gate（3 矩阵） |
| vocab | 131072 | 200064 |
| RoPE | 3D mRoPE [24,20,20]，θ=1e8 | 1D LongRoPE，partial 0.75，θ=1e4 |

只有 `head_dim` 和 KV 头数对得上——而这**恰好是仅有的两个必须对上的**（见 §2.2）。

### 1.4 官方训练代码

`NVIDIA/cosmos-framework` 公开，但只有 post-training/SFT 配方
（`launch_sft_vision_edge.sh`、DROID/LIBERO 动作策略等），**没有从零预训练**。
所以"复刻训练"的主要成本在数据和预训练，不在换 LLM。

---

## 2. 本分支的结构

### 2.1 总览

```
frame_t  --[Qwen3-VL ViT, 冻结]--> 256 tok --[2x2 merge, 可训练]--> 64 tok --.
instruction --[Phi embed, 冻结]--> text tok ------------------------------.  |
                                                                          v  v
                                              und 流（Phi-4-mini 32 层，全冻结）
                                                              | 逐层 K/V
                                                              v
frame_t+H --[Wan VAE, 冻结]--> latent --加噪--> patch --> gen 流（32 层，从零）--> velocity
                                                              |
                                                    action --> 每 embodiment 投影 --> action velocity
```

### 2.2 两个专家之间只有两处硬约束

`all_k = cat([k_und, k_gen])` 沿序列维拼接，所以两路必须共享 **`head_dim`** 和 **`num_key_value_heads`**。
除此之外 GEN 的宽度、查询头数、MLP 大小全都自由——`add_q_proj` 和 `to_add_out` 在两个宽度之间架桥。

这一条是整个方案能落地的关键：按 Phi 宽度镜像 GEN 需要 ~3.2 B 新参数，
AdamW 状态就要 ~98 GB，这台机器装不下；取 `d_gen=1536` 后只剩 614 M。

约束：GEN 查询头数必须是 KV 头数（8）的整数倍，否则 GQA 报错。
`MoTConfig.validate()` 会拦住（最初写 12 头就是被它拦下的）。

### 2.3 RoPE：把 partial RoPE 塞进 3D mRoPE

Phi 的 `head_dim=128`、`partial_rotary_factor=0.75` → 旋转 96 维 → `inv_freq` 48 项。
所以 `mrope_section` 必须凑成 48，取 `[16,16,16]`。

关键性质：**文本 token 取 `t=h=w=pos` 时，三段用的是同一个位置，mRoPE 精确退化为普通 1D RoPE。**
于是一个纯文本预训练的 LLM 可以原样放进这个栈，位置行为不变，而视频 token 仍能拿到真正的 3D 位置。

另外两个必须从参考实现读、不能想当然的量：

- `attention_scaling = 1.190238`。LongRoPE 会给 cos/sin 乘这个因子（`sqrt(1+ln32/ln4096)`），
  漏掉它整个 attention 的尺度就错了。代码里从 HF 的 rotary 模块读出来传入，不硬编码。
- `short_factor` 全为 1.0 且 `original_max_position_embeddings=4096`，
  所以我们这种短序列下 LongRoPE 的插值是恒等的，可以不实现。**这是被验证过的，不是假设。**

位置分配沿用 Qwen2-VL 方案：各段顺序排布，每段把共享计数器推进 `max(t,h,w)`，模态之间不会撞位置。
GEN 的空间坐标再乘 `vision_grid/latent_side`，否则 8×8 的 latent 只会覆盖 16×16 图像网格的一半。

### 2.4 为什么自己重写 Phi 的前向

需要两件 HF 接口给不了的东西：控制 RoPE（要换成 mRoPE），以及导出**逐层 pre-RoPE 的 K/V**。
所以 `MoTLayer.und_forward` 手写了一遍 Phi3 的层（融合 `qkv_proj`、融合 `gate_up_proj`、
`RMSNorm` 先转 fp32 再转回原 dtype 后乘 weight）。

这种重写最容易悄悄写错，所以有 `scripts/check_mot_und.py` 对着 HF 逐项比：
`inv_freq` 完全一致（0.0），32 层后隐状态 fp32 下相对误差 **8.3e-6**，余弦 1.0。

### 2.5 UND 全冻结带来的简化

因为 UND 是纯因果自注意力、**从不读 GEN token**，两路不必逐层交错：
可以先整栈跑完 UND，只留每层 K/V，再跑 GEN 栈。等价，但 Phi 的激活完全不进图。

代价是：投影层可训练时，梯度仍要**穿过**冻结的 Phi 才能到达它，此时 und 栈必须带梯度（已加检查点）。
实测这一项占 58% 的步时间（见 §4）。

---

## 3. 各模块参数量

| 模块 | 参数 | 可训练 |
|---|---|---|
| Qwen3-VL ViT（4B 版视觉塔） | 306.2 M | ✗ |
| VisionMerger（2×2 merge + 4096→3072→3072） | 22.0 M | ✓ |
| Phi-4-mini und 专家（embed 614.6 M + 32×100.7 M） | 3.836 B | ✗ |
| GEN 专家 32 层 | 604.1 M | ✓ |
| `proj_in` / `proj_out`（192↔1536） | 0.59 M | ✓ |
| `time_embedder`（256→1536→1536） | 2.76 M | ✓ |
| `action_proj_in/out`（32 domain × 64↔1536） | 6.34 M | ✓ |
| **合计** | **4.778 B** | **636 M（13.3%）** |

GEN 单层 18.877 M 的构成：注意力 6.291 M（`add_q/add_k/add_v/to_add_out` 各 1.572 M）
+ MLP 12.583 M（relu²，只有 up/down 两个矩阵）+ 5 个 norm 共 3456。

Wan VAE 704.7 M 冻结，不计入模型（作为数据侧的潜变量编码器）。

---

## 4. 实测（batch 128，bf16，单张 48 GB 卡，机器共用）

阶段拆解（`scripts/profile_mot_world.py`，前向）：

| 阶段 | 耗时 | 算力 |
|---|---|---|
| 视觉塔 | 465 ms | 43 TFLOP/s |
| und 栈 32 层 | 1162 ms | **81 TFLOP/s** |
| gen 栈 32 层 | 692 ms | 36 TFLOP/s |

und 已接近峰值，没有可榨的空间。gen 效率低是**结构固有**——它的矩阵小得多
（1536/1024 vs Phi 的 3072/16384），不是实现问题。

整步（`scripts/smoke_mot_world.py`）：

| 配置 | s/step | 峰值显存 |
|---|---|---|
| 投影层可训练 + gen 检查点 | 7.125 | 21.4 GiB |
| 投影层冻结（und 走 no_grad） + gen 检查点 | **4.514** | 18.8 GiB |
| 不加 gen 检查点 | OOM | — |

由此得到一个可用的训练日程：**先让投影层穿过 Phi 热身若干步，再冻结它**，
后续训练提速 1.58×。两个端点都是实测的。

---

## 5. 验证

`scripts/check_mot_und.py`（und 专家对齐 HF）：

| 检查 | 结果 |
|---|---|
| `inv_freq` 与 HF 一致 | 0.0 |
| 32 层隐状态相对误差（fp32） | 8.3e-6 |
| 余弦相似度 | 1.0 |

`scripts/smoke_mot_world.py`（世界模型）：

| 检查 | 结果 | 为什么查这个 |
|---|---|---|
| patchify/unpatchify 往返 | 0.0 | latent 布局写错在 loss 上看不出来，但空间结构会被毁 |
| 372 个可训练张量全部拿到梯度 | 0 缺失 | GEN 若被接断，光靠视频头仍能收出一条像样的 loss 曲线 |
| 冻结张量携带梯度数 | 0 | 确认 Phi 确实没进优化器 |
| GEN 张量收到梯度 | 352 = 32×11 | 逐层两个专家都接上了 |

---

## 6. 还没做的

- **数据接入**。现有 dataset 给的是两帧 `image_t0/image_t1`（`chunk_seconds=1.6`）而非视频片段。
  Wan VAE 在 T=1 时输出 1 个 latent 帧，所以 v1 可以直接做
  "给定当前帧 + 语言 → 预测 1.6 s 后那帧的 latent + 动作"，无需改数据集；
  多帧片段（T=1+4k）留到之后。
- 文本侧目前用随机 id 冒烟，尚未接 Phi 的 tokenizer。
- 采样/推理循环（训练目标是 rectified flow，采样器还没写）。
- 动作维度默认 64（对齐 Cosmos），接数据时要改成本仓库的规范动作维度。
