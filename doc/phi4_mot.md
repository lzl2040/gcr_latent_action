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

GEN 专家按 `d_gen=2048 / 16 头 / intermediate 7680` 配置，目的是让生成侧参数量与
Cosmos3-Edge 的生成分支（实测 1.423 B）对齐。

| 模块 | 参数 | 可训练 |
|---|---|---|
| Qwen3-VL ViT（4B 版视觉塔） | 306.2 M | ✗ |
| VisionMerger（2×2 merge + 4096→3072→3072） | 22.0 M | ✓ |
| Phi-4-mini und 专家（embed 614.6 M + 32×100.7 M） | 3.836 B | ✗ |
| GEN 专家 32 层（44.045 M/层） | 1.409 B | ✓ |
| `proj_in` / `proj_out`（192↔2048） | 0.79 M | ✓ |
| `time_embedder`（256→2048→2048） | 4.72 M | ✓ |
| `action_proj_in/out`（32 domain × 40↔2048） | 5.31 M | ✓ |
| **合计** | **5.585 B** | **1.442 B（25.8%）** |

`action_dim` 取 40 而不是 Cosmos 的 64：本仓库规范动作向量的宽度是
`canonical_space.CANON_DIM = 40`，用 64 会让 per-domain 投影期待 loader 永远不会填的列。

Wan VAE 704.7 M 冻结，不计入模型（作为数据侧的潜变量编码器）。

### 3.1 可训练范围开关

上表是默认档。哪些权重训练由 `WorldModelConfig.trainable_scope` 控制，作用在四组权重上
（`TRAINABLE_SCOPES`，`world_model.py`）：

| scope | vision | merger | und | gen | 可训练 | 说明 |
|---|:-:|:-:|:-:|:-:|---|---|
| `gen_only`（默认） | ✗ | ✓ | ✗ | ✓ | 1.445 B | Cosmos3 的做法 |
| `freeze_vision` | ✗ | ✓ | ✓ | ✓ | 5.281 B | π0.5 的做法：只冻视觉塔 |
| `all` | ✓ | ✓ | ✓ | ✓ | 5.588 B | 全训 |

两条设计取舍：

* **`k_norm_und_for_gen` 跟着 GEN 走，不跟 UND。** 它把 und 的 K 归一化后交给 gen 用，
  只因为 gen 存在才存在。若按"属于 und 层"归类，冻结 und 时这个**两个专家之间的接口**
  会被钉死在初始化的 scale 上。
* **UND 冻结 ≠ UND 免费。** 可训练的 merger 在 und 之前，梯度必须穿过全部 32 层冻结的
  Phi，所以 und 栈照样要建图（带检查点）。这也解释了下面 1.27× 这个比参数比小得多的数字。

`freeze_vision_projector=True` 是叠加在 scope 之上的覆写：它让 `gen_only` 档下 und 栈能整段
跑在 `no_grad` 里（实测 1.58×）；一旦 und 可训练，它就不再有加速作用。

两个曾经的隐藏 bug（`scripts/check_trainable_scope.py` 会同时静态和动态地查）：

* `encode_und` 里视觉塔外面套的是**无条件** `torch.no_grad()`。`scope="all"` 下每个 ViT
  权重的 `requires_grad` 都是 `True`，却一个梯度也收不到，而 loss 照常下降——这种失败是
  完全静默的。现已按 scope 开关。
* `und_needs_grad` 原先只看 merger，und 可训练时 und 栈会被误判为不需要建图。

---

## 4. 实测

### 4.1 阶段拆解（`scripts/profile_mot_world.py`，batch 8，gen = 1.423 B）

| 阶段 | 耗时 | 算力 | 占 A6000 可用峰值 |
|---|---|---|---|
| 视觉塔 | 33.7 ms | — | — |
| und 栈 32 层 | 63.1 ms | **78.5 TFLOP/s** | ~101% |
| gen 栈 32 层 | 69.9 ms | 52.1 TFLOP/s | 67% |

und/gen FLOP 比 1.36×，时间比 0.90×。

这里的"可用峰值"是 **77.4 TFLOP/s** 而不是官方标称的 155：RTX A6000 是 GA102，
bf16 张量核在 FP32 累加下是半速率，而 PyTorch 的 bf16 矩阵乘正是 FP32 累加。
und 实测 78.5 恰好压在这条线上，这既说明 und 没有可榨的空间，也反过来验证了
FLOP 计数是对的。

gen 仍比 und 低约三分之一，因为它的矩阵更小（2048/7680 vs Phi 的 3072/16384）；
但把 `d_gen` 从 1536 加宽到 2048 后，这个差距已经从此前的 46% 收窄到 67%，
说明加宽确实换来了效率而不只是参数量。

> 注：`profile_mot_world.py` 早期版本把两侧参数量硬编码成常数，gen 扩容后没跟着变，
> 一度把 gen 报成 23 TFLOP/s。现在两个数都从 `param_report()` 取，und 侧还扣掉了
> embedding（查表不是矩阵乘）。本节数字是修正后的。

### 4.2 整步（`scripts/smoke_mot_world.py`）

| 配置 | s/step | 峰值显存 |
|---|---|---|
| batch 8，投影层可训练 + gen 检查点 | 0.622 | 21.2 GiB |

早期在 gen = 614 M 的配置下测过 batch 128：投影层可训练 7.125 s/step（21.4 GiB），
冻结投影层 4.514 s/step（18.8 GiB），不加 gen 检查点则 OOM。由此得到的日程仍然成立：
**先让投影层穿过 Phi 热身若干步，再冻结它**，后续提速 1.58×。gen 扩容后这两个端点
没有重测，真实训练速度以 §7 的实测为准。

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

## 6. 数据接入与任务族

### 6.1 loader：多帧片段 + 触觉可选

两个开关都通过 `getattr(policy_cfg, ...)` 读取，默认值等于原行为，所以对比学习那条路
一行都不用改。

**`rgb_frames`（默认 2）**。原来 RGB 只读窗口两端：`pair_stamps = [0, horizon/index_fps]`。
现在改用触觉相机早就在用的等距公式 `[horizon*i/((n-1)*index_fps)]`，它在 `n=2` 时
**恰好还原成原式**——默认路径是被代数保住的，不是靠另开一个分支保住的。

`_extract_frames` 额外返回整段 clip，`image_clip` 只在 `rgb_frames > 2` 时出现。
`scripts/check_multiframe_dataset.py` 在 4 个数据集上实测：`clip[0]` 与 `image_t0`、
`clip[-1]` 与 `image_t1` 逐像素差为 **0**，而中间帧与首帧的平均绝对差是 10–61，
说明多出来的帧确实带信息，而不是端点的副本。

**`use_tactile`（默认 True）**。关掉时触觉视频从 `video_keys_to_decode` 移除，
触觉列不再申请时间窗，触觉字段**整个不出现在 batch 里**而不是填零：padded view 张量
约 3.6 MB/样本，batch 128 就是几百 MB 的零在 collate 和 PCIe 上搬；而且误用应该立刻
KeyError，而不是安静地在全零上训练。

实测每样本读取耗时：

| 路径 | ms/样本 |
|---|---|
| 原 2 帧 + 触觉 | 64.9 |
| 9 帧 clip + 无触觉 | **37.2（0.57×）** |

多读 4.5 倍 RGB 帧反而比原路径快 43%。这印证了 `doc/results.md` §21 的结论——解码时间被
seek 关键帧和跨 span 解码主导，中间帧本来就在解、只是被丢掉——同时说明触觉才是 loader 的
真正开销。

### 6.2 Wan VAE

`lerobot/common/policies/mot/vae_latents.py`，用 `Cosmos3-Edge/vae`
（即 `Wan2.2-TI2V-5B` 的 `AutoencoderKLWan`，704.7 M 冻结）。两个实测确认的性质：

- 时间 4× 压缩且首帧单独成一个 latent，所以 clip 必须是 **T = 1+4k**：
  实测 T=1/5/9/17 → 1/2/3/5 个 latent 帧。长度不对会静默截断。
- 空间 256→16，与 `latent_grid=16` 对齐；48 个通道对应 `proj_in` 的 192 = 48×2×2。

latent 用 checkpoint 自带的 `latents_mean/std` 逐通道归一化。这一步不能省：该 checkpoint 的
通道 std 跨越 0.35–1.17，不归一化的话 loss 会被少数几个通道支配。

### 6.3 一条代码路径覆盖五个任务

Cosmos 第二/三阶段的任务差别只有两点：**多少个 latent 帧是干净的**，以及
**理解侧有没有输入图**。所以把噪声水平从 per-sample 改成 **per-frame** 之后，五个任务
共用同一份 rectified-flow 代码，而不是五个会各自漂移的分支。

| 任务 | context 帧 | und 输入图 | 动作 token |
|---|---|---|---|
| `t2i` | 0 | ✗ | ✗ |
| `t2v` | 0 | ✗ | ✗ |
| `i2v` | 1 | ✓ | ✗ |
| `v2v` | 2 | ✓ | ✗ |
| `action` | 1 | ✓ | ✓ |

context 帧 σ=0，原样进 transformer 并**排除出 loss**；其余帧共享一次采样。loss 按目标
token 数归一化而不是全部 token 数，所以 context 长度不同的任务数值可以横向比较。
`forward_gen` 因此需要接受 `(B, L)` 形状的 timestep，`encode_und` 需要接受
`pixel_values=None`（t2i/t2v 没有输入帧，喂一张空白图等于花一次完整 ViT 前向去教模型
"没有图"长什么样）。

验证方式（`scripts/check_mot_tasks.py`）：把 context 帧的 latent **放大 1000 倍**，
泄漏就会变得无法忽视。五个任务的 loss 全部落在 **2.32–2.35**，即单位方差下
E‖noise−latents‖²=2 的理论地板；若 mask 失效则是 1e6 量级。另用计数 hook 确认视觉塔
恰好只在 i2v/v2v/action 运行。

### 6.4 action 是怎么训的

**和视频用同一个 rectified-flow 目标，在同一次前向里联合去噪**，不是单独接一个回归头。

```
σ ~ U(0,1)                                  # 每个样本一次，视频目标帧和动作共用
noisy_a = (1-σ)·a + σ·ε ,  target = ε - a   # 速度场，与视频侧完全同构
tok_a   = action_proj_in(noisy_a, domain_id) + action_modality_embed
gen_tokens = [视频 patch token ... , tok_a ...]     # 拼在一起进 GEN 流
loss = loss_video + w · MSE(action_proj_out(h_a, domain_id), target)
```

四个设计点：

1. **动作 token 拼进 GEN 序列，而不是另起一路。** GEN 的自注意力是双向的
   （`is_causal=False`，见 `modeling_mot.py:302`），且每层都跨 `cat([k_und, k_gen])`，
   所以 32 个动作 token 在每一层既能双向看全部视频 token、也能看到 und 侧的文本和
   当前帧图像 K/V。动作和未来画面是被**联合**建模的，不是画面预测完再回归动作。
2. **σ 与视频目标帧共享一次采样。** 推理时两者按同一条噪声调度一起去噪；如果各采各的，
   模型在训练里就见不到"画面已经很清晰但动作还很糊"这类组合。
3. **`action_proj_in/out` 是 per-domain 的**（32 个 embodiment domain 各一套 40↔2048）。
   数据集的动作空间不统一（xyz+ort6d+gripper 与 joint 混杂），共享一套投影会让不同本体
   的同一列含义打架；`domain_id` 由 loader 给出。
4. **`action_modality_embed`** 是一个可学习偏置，让 GEN 流能区分动作 token 和视频 token
   ——两者进来时都是 2048 维，没有这个偏置就只能靠位置编码去猜。

动作只在 `task="action"` 上有 loss（`TaskSpec.action`），在 stage-3 混合里占 50%。

---

## 7. 训练速度实测与集群外推

`scripts/train_mot_world.py`：dataset → Wan VAE → MoT → AdamW，真实数据跑通，loss 下降
（t2i 1.680 → 1.229）。batch 32、单张 RTX A6000、每任务 15 步、fused AdamW：

| 任务 | data | vae | model | step | clips/s |
|---|---|---|---|---|---|
| `t2i` | 1 ms | 1271 ms | 614 ms | 1886 ms | 16.97 |
| `t2v` | 1 ms | 1271 ms | 1345 ms | 2617 ms | 12.23 |
| `i2v` | 1 ms | 1272 ms | 2147 ms | 3421 ms | 9.36 |
| `v2v` | 1 ms | 1274 ms | 2324 ms | 3599 ms | 8.89 |
| `action` | 1 ms | 1275 ms | 2338 ms | 3615 ms | 8.85 |

峰值显存 26.1 GiB。三点值得注意：

- **data 只有 1 ms**：12 个 worker 的预取把 37 ms/样本完全藏在 GPU 计算后面。
- **VAE 占 37%**，且与任务无关。真实训练应把 latent 预先缓存（1 万小时 ≈ **1.66 TB**，
  放得下），这也是把三段计时分开报的原因——只有 model 那段会随显卡变快。
- 分解自洽：`t2v − t2i` = 731 ms 对应 2 个额外 latent 帧；`i2v − t2v` = 802 ms 是视觉塔加
  und 里的图像 token；`action − i2v` = 191 ms 是 32 个动作 token。i2v 与 v2v 计算量本就
  相同，实测 2147 vs 2324 ms（差 8%，同批噪声水平），说明这批数字可用。

### 7.1 两种可训练范围的代价（`SCOPE=... bash scripts/bench_mot_scope.sh`）

同一份代码路径（fused AdamW + `expandable_segments`）、batch 32、stage-3：

| 任务 | `gen_only` | `freeze_vision` | 比值 |
|---|---|---|---|
| `t2i` | 614 ms | 1173 ms | 1.91× |
| `t2v` | 1345 ms | 2069 ms | 1.54× |
| `i2v` | 2147 ms | 2613 ms | 1.22× |
| `v2v` | 2324 ms | 2613 ms | 1.12× |
| `action` | 2338 ms | 3018 ms | 1.29× |
| **混合加权** | **2112 ms** | **2689 ms** | **1.27×** |
| 峰值显存 | 26.1 GiB | 40.9 GiB | +14.8 GiB |
| 端到端（含 VAE） | 10.57 clips/s | 8.96 clips/s | 0.85× |

**可训练参数涨 3.7×（1.45 B → 5.28 B），算力只涨 1.27×。** 原因在 §2.5：GEN 每层都读
UND 的 K/V，所以两个档位里 und 的前向本来就要跑、而且本来就带检查点；训练 und 多出来的
只是权重梯度那次 matmul，不是第二次前向。相对代价在 `t2i` 上最大（1.91×），因为那个任务
GEN 侧的活最少，und 反向占比最高。

显存这一侧才是真正的分界：5.28 B 可训练 = 参数 10.4 + 梯度 9.8 + AdamW 两个动量 19.7 GiB。
第一次跑直接 OOM 在 `torch._foreach_sqrt` 上——多张量 AdamW 会额外开一份和状态等大的临时
buffer。换成 `fused=True`（无临时 buffer）+ `PYTORCH_ALLOC_CONF=expandable_segments:True`
后 batch 32 能压进 40.9 GiB，单卡 48 GiB 放得下，不强制要 ZeRO。
注意动量是 bf16（跟随参数 dtype），真实长训要考虑 fp32 master weights + fp32 动量，
那会再多约 59 GiB（master 19.7 + 两个动量各 19.7），届时必须 ZeRO 分片。

`NO_OPT=1` 可以只测前反向（gen_only 15.0 GiB / freeze_vision 21.5 GiB），用来在放不下
优化器状态的卡上比较纯计算。

### 7.2 前向一次要多久（`scripts/measure_forward.py`）

batch 32、`task=action`、合成张量、6 次计时取均值：

| scope | fwd (no_grad) | fwd (建图) | fwd+bwd | 整步 (+opt) |
|---|---|---|---|---|
| `gen_only` | 834 ms | 828 ms | 2810 ms | 2338 ms\* |
| `freeze_vision` | 899 ms | 840 ms | 3052 ms | 3018 ms\* |

\* 整步那一列来自 §7.1 的独立无争用测量；fwd/bwd 那三列跑的时候卡上有别人 19 GiB 的进程，
所以 fwd+bwd 反而比整步还高——是争用，不是模型变慢了。按无争用整步 3018 ms 折算，
**`freeze_vision` 纯前向约 830 ms，占整步 27%**。

`fwd(建图) ≈ fwd(no_grad)` 是梯度检查点的直接后果：前向不保存中间激活，所以建不建图几乎
不影响前向耗时；代价挪到了反向——反向要重跑一遍前向，于是 `bwd ≈ 2.4 × fwd`，
`fwd+bwd ≈ 3.4 × fwd`，而不是通常的 3×。

### 7.3 外推（`scripts/extrapolate_cluster.py`）

**"1 万小时"到底是多少步，取决于什么算一个 sample，三种口径差 32 倍。**
我们的一个 sample 是 1.6 s 窗口，**读 9 帧、跨 32 帧**：

| 口径 | sample 数 | 含义 |
|---|---|---|
| `frames` | **7.2e8** | 1 sample = 1 帧（你的口径：10000×3600×20） |
| `windows` | 2.25e7 | 1 sample = 一个不重叠 1.6 s 窗口 |
| `coverage` | 8e7 | 让磁盘上每一帧至少被读到一次 |

`coverage` 之所以存在：过一遍窗口只喂进 2.02e8 帧 = 全部数据的 **28%**，
要每帧都看到需要 3.6 倍的窗口。下表三列分别对应这三种口径。

**你的算法我复核过，是对的。** 16 卡 × bs 32 = 512，7.2e8 / 512 = **1.41e6 步**；
乘我们 A6000 上的 ~3 s/step = 48.8 天，正好落在下表 A6000 那一行（51.8 d，多出来的是
ZeRO-2 通信）。A100 相对 A6000 的加速只有 ÷3.63（算力）×（latent 预缓存 ÷1.57），
不足以把 16.9 天压到 12 小时——我之前给的 12 h 是 `windows` 口径下的数，
**不能和你的帧口径直接比**。

**step 时间不与 batch 成正比。** 实测 batch 32 → 2112 ms、batch 64 → 3943 ms，翻倍只涨
1.87×，因为有一个约 **281 ms 与 batch 无关的地板**（kernel launch + 优化器）。所以外推用
`281 ms + 57.2 ms/clip` 拟合，而不是按比例放大——后者会把 batch 128 高估 13%。

锚点可信的关键：A6000 是 GA102，bf16 在 FP32 累加下**半速率**，可用峰值约 77 TFLOP/s
而非官方标称的 155。实测 und 跑在 78.5 TFLOP/s，即基本打满硬件，这才让缩放有意义。
A100 / A6000 = 312 / 77.4 = 4.03×，乘效率 0.75–0.90 → **3.0–3.6×**。

**显存是硬边界，而且我之前的表在这点上是错的。** 实测（47.6 GiB 卡）：

| scope | b8 | b32 | b64 | b128 | DDP 常驻状态 | ZeRO-2 ×16 |
|---|---|---|---|---|---|---|
| `gen_only` | 20.4 G | 26.1 G | 35.0 G | **OOM** | 18.5 GiB | 10.9 GiB |
| `freeze_vision` | 40.8 G | 40.9 G | **OOM** | — | **39.9 GiB** | 12.3 GiB |

常驻状态 = 参数 + 梯度 + 两个 bf16 动量。**`freeze_vision` 的 39.9 GiB 在 40 GB A100 上
任何 batch 都放不下**（连一个样本都跑不了），必须 ZeRO-2 分片；分片后降到 12.3 GiB，
反而很宽裕。`gen_only` 的 batch 128 需要 ~52 GiB，同样超出 40 GB 和 48 GiB。

16 卡、stage-3 混合、**帧口径**（1.41e6 步）。step = `max(计算 + 暴露的通信, I/O)`，
三者取最大而不是相加，因为 loader 和 all-reduce 都是和计算并行的：

| scope | 设备 | b/GPU | 网络 | 数据 | 计算 | 暴露通信 | I/O | step | 瓶颈 | 帧口径 |
|---|---|---|---|---|---|---|---|---|---|---|
| `gen_only` | A100-40G | 64 | 8×200G IB | latent | 1087 | 2 | 320 | 1089 ms | 计算 | **8.9 d** |
| `gen_only` | A100-40G | 64 | 100G Eth | 原视频 | 1087 | 13 | 1067 | 1100 ms | 计算 | **8.9 d** |
| `gen_only` | B200 | 128 | 8×200G IB | latent | 476 | 2 | 640 | 640 ms | **I/O** | **2.6 d** |
| `gen_only` | B200 | 128 | 8×200G IB | 原视频 | 476 | 2 | 2133 | 2133 ms | **I/O** | 8.7 d |
| `freeze_vision` | A100-40G | 32 | 8×200G IB | latent | 741 | 6 | 160 | 748 ms | 计算 | **12.2 d** |
| `freeze_vision` | A100-40G | 32 | 8×200G IB | 原视频 | 741 | 6 | 533 | 748 ms | 计算 | 12.2 d |
| `freeze_vision` | A100-40G | 32 | 100G Eth | 原视频 | 741 | **385** | 533 | 1127 ms | 通信 | **18.3 d** |
| `freeze_vision` | A100-80G | 128 | 8×200G IB | latent | 2732 | 6 | 640 | 2739 ms | 计算 | **11.1 d** |
| `freeze_vision` | B200 | 128 | 8×200G IB | latent | 620 | 6 | 640 | 640 ms | **I/O** | **2.6 d** |
| `freeze_vision` | B200 | 128 | 8×200G IB | 原视频 | 620 | 6 | 2133 | 2133 ms | **I/O** | 8.7 d |
| `freeze_vision` | B200 | 128 | 100G Eth | latent | 620 | **473** | 640 | 1093 ms | 通信 | **4.4 d** |

**B200 的结论：最好 2.6 天，最差 8.7 天，而最差那一档和 A100-80GB 打平。**
它的算力（620 ms 计算）根本用不上，因为 12 个 worker 读原始视频只能供上 2133 ms 一步。
B200 想跑出 2.6 天，两个前提缺一不可：latent 离线缓存 + 不是 100G 以太网。

**通信这一块我之前建模是错的。** 原来写的是"每 GPU 19.8 GB ÷ 20 GB/s × (1−0.7)"，
两个毛病：

1. **没分节点内/节点间。** 16 卡 = 2 节点 × 8 卡。NCCL 走层级 all-reduce：节点内 NVLink
   reduce-scatter → 节点间只传 1/8 的分片 → 节点内 all-gather。真正过网卡的量是
   `2(n−1)/n × G`，n=2 时就是完整的 G，而这些字节由**同节点 8 张卡共享一条链路**。
   `freeze_vision` 的 G = 10.56 GB：

   | 网络 | NVLink 段 | 网卡段 | 合计 |
   |---|---|---|---|
   | 8×200G IB（ND A100 v4 级） | 74 ms | 53 ms | 127 ms |
   | 1×200G IB（整节点一张网卡） | 74 ms | 423 ms | 497 ms |
   | 100G 以太网 | 74 ms | **845 ms** | **919 ms** |

   一个扁平的"20 GB/s per GPU"对好网络低估 8 倍、对普通云 VM 高估 8 倍。

2. **`OVERLAP=0.7` 是拍脑袋的。** 它等于宣称能把 919 ms 的以太网流量塞进一个 1.4 s 的 step，
   物理上做不到。现在改成有界的：重叠只能用**反向那段窗口**（实测反向占 step 的 72%，
   见 §7.2），且最后一个 bucket 必然暴露，所以
   `暴露 = max(通信 − 0.72×计算, 0.05×通信)`。这才让"100G 以太网 + freeze_vision"
   如实显示出 385–473 ms 的暴露通信，把 A100 从 12.2 天推到 18.3 天。

至于**延迟**本身：ring all-reduce 是 2(N−1)=30 跳，每跳约 8 µs，合计 0.24 ms。
在这个消息尺寸下延迟完全不是问题，**带宽才是**——所以上面按带宽建模是对的。

**训 und 在集群上比在单卡上更贵**，而且贵多少完全取决于网络：单卡 1.27×；
8×200G IB 上仍是 ~1.27×（通信藏得住）；100G 以太网上变成 1.5×，因为全归约体积
涨了 3.7×（2.89 → 10.56 GB）而它藏不进反向。

给的是乐观端：A100/B200 的效率是假设，不是实测。B200 效率区间刻意取得宽且下沿低
（0.30–0.55），因为我们的 GEN 专家在 A6000 上就只跑到可用峰值的 67%（52.1 vs und 的
78.5 TFLOP/s），原因是矩阵偏小——张量核越大，这个缺口只会越明显。

**换数据量：`python scripts/extrapolate_cluster.py --hours 30000`。**
墙钟对小时数是**严格线性**的，因为 step 时间只取决于 batch，与语料大小无关。
唯一非线性的是 latent 缓存体积：

| 数据量 | 帧数 | latent 缓存 | `freeze_vision` A100-40G(IB) | `freeze_vision` B200(IB+latent) | B200(IB+原视频) |
|---|---|---|---|---|---|
| 1 万 h | 7.2e8 | 1.66 TB | 12.2 d | 2.6 d | 8.7 d |
| **3 万 h** | **2.16e9** | **4.98 TB** | **36.5 d** | **7.8 d** | **26.0 d** |

4.98 TB 这个数才是 3 万小时真正的新问题：1.66 TB 可以整份塞进一个 8 卡节点的本地 NVMe
（ND A100 v4 是 6.4 TB），5 TB 就只剩不到 1.5 TB 放 checkpoint 和临时文件，基本等于
放不下。也就是说 3 万小时时"把 latent 预取到本地盘"这条退路没了，只能直接读 blob，
而 B200 恰恰是最依赖这条退路的（见 §7.4）。

### 7.4 挂载盘 I/O（`scripts/measure_io.py`）——我之前这里说错了

原文写的是"loader 上限 323 clips/s/GPU，即使 B200 也没触到，所以视频解码不是瓶颈"。
**这个结论是错的**，它建立在 37.2 ms/clip 这个数上，而那是 12 个 worker + 页缓存已热的
本地读。用 `/proc/self/io` 的 `read_bytes` 在 `num_workers=0` 下冷读实测：

| 量 | 实测 |
|---|---|
| 每 clip 块设备字节 | **677 KiB** |
| 每 clip read 系统调用 | **6 次** |
| 单 worker 冷读延迟 | **780.8 ms/clip** |

比我引用的 37.2 ms 差 **21 倍**。查了才知道原因：`lsblk -d -o NAME,ROTA` 显示 `/Data`
所在的 `sdc`（Seagate ST16000NM000J）和 `sda` 都是 **ROTA=1，机械盘**；本机唯一的 SSD
是 `sdb`，上面没有数据。之前那个 1 ms 是多 worker 并发 + 页缓存把冷随机寻道盖住了。

集群是挂载存储（blobfuse/NFS），没有寻道惩罚但每次 read 有一次网络往返，而我们**每 clip
要发 6 次 read**。真正的约束是**每卡能开多少 worker**：一个 8 卡节点 96 vCPU 就是每卡 12 个，
于是 loader 吞吐 = `12 / 延迟`，这是个硬天花板：

| 数据形态 | 单 clip 延迟 | loader 上限 (clips/s/GPU) |
|---|---|---|
| 本地热页缓存 | 37 ms | 323 |
| **blob 存 latent** | 60 ms | **200** |
| **blob 存原视频** | 200 ms | **60** |
| blob 被限流 | 500 ms | 24 |

对照各卡的需求（batch/step）：

| 设备 | b/GPU | 需要 clips/s/GPU | blob 原视频(60) | blob latent(200) |
|---|---|---|---|---|
| A100-40G | 32 | 43 | 够（72% 占用） | 够 |
| A100-80G | 128 | 61 | **刚好不够** | 够 |
| B200 | 128 | **200** | **差 3.3×** | 刚好 |

**这就是 B200 从 2.6 天退化到 8.7 天的全部原因**，和它的算力无关。16 卡合计需求
2.0 GiB/s、18,000 次随机读/秒——带宽不算什么，**每秒上万次随机小读**在 blobfuse 上很容易
打到限流。三个应对：

1. **离线缓存 VAE latent**（1.66 TB）。loader 从"seek 进 mp4 解码"变成"顺序读 72 KiB 张量"，
   read 次数 6 → 1，且完全没有解码。**在网络挂载上这个收益比本地大得多**，
   对 B200 是从 8.7 天到 2.6 天的分水岭。
2. **worker 数按延迟配**：`workers ≈ clips/s × 延迟`。780 ms（机械盘）要 ~24 个，
   200 ms（挂载）要 ~6–9 个，37 ms（热/SSD）只要 1.2 个。默认 12 个在挂载盘上刚够 A100，
   B200 一定不够——但 B200 节点的 vCPU 也不会多到能开 40 个，所以只能靠第 1 条。
3. **本地 NVMe 暂存**：把当前 shard 预取到节点本地盘，把随机读从 blob 挪到 SSD。

---

## 8. 还没做的

- 文本侧已接 Phi 的 tokenizer（`train_mot_world.py`），但冒烟脚本仍用随机 id。
- 采样/推理循环（训练目标是 rectified flow，采样器还没写）。
- latent 离线缓存（上面 1.66 TB 那条）还没实现，目前是在线编码。
- 多卡：只在单卡验证过，ZeRO/DeepSpeed 接入未做——`freeze_vision` 档下这是必需项而非
  优化项（fp32 优化器状态放不下，且全归约体积 10.56 GB 需要和反向重叠）。
- `scope="all"`（解冻视觉塔）代码路径已通过 `check_trainable_scope.py`，但没有实测速度，
  也没有验证训练稳定性——视觉塔可训时 latent action 的表征塌缩风险还没评估。
