# 问题与解决记录

本文记录开发过程中遇到的、**不会报错但会静默损坏模型**的问题：出现的场景、根因、为什么
必须那样解决，以及用什么证据确认解决对了。

这类问题的共同特征是：代码能跑、shape 对、loss 会下降，但学到的东西是错的。所以每条记录
都必须附带一个能"失败"的检查，而不是一句"我检查过了"。

---

## 1. Qwen3-VL 的 patch 排序：为什么要自己实现，以及它埋了什么坑

**涉及文件**：`lerobot/common/policies/ace/qwen3vl_encoder.py`、
`scripts/check_qwen3vl_vision.py`

### 1.1 背景：这个模型为什么在乎 patch 的空间顺序

感知分支不是把图像编码成一个全局向量就完事。它依赖 patch token 的**空间位置含义**，有两处
硬依赖：

1. **变化证据流（evidence stream）**。模型取 `t` 和 `t+H` 两帧的 patch 特征 `v0`、`v1`，
   把逐位置的差 `v1[i] - v0[i]` 作为"第 i 个位置发生了什么变化"的证据，喂给 change query
   做交叉注意力。这个减法**只有在 `v0[i]` 和 `v1[i]` 指向同一个空间位置时才有意义**。
2. **VAE 重建目标**。`perception_recon_target=vae` 时，重建目标是 Wan VAE 编码出的
   16×16 = 256 个 latent，按行主序排列。预测头输出的第 i 个 token 要去对齐第 i 个 latent。

也就是说，"第 i 个 token 对应图像上第 i 个格子（行主序）"不是一个实现细节，而是这两处设计
成立的前提。一旦顺序错了，两处都会静默地学错：减法减的是两个不同位置，重建对齐的是错位的
目标。**loss 依然会下降**——模型会去拟合这个被打乱的映射——只是学到的东西不是我们要的。

### 1.2 问题出现：这个塔根本不吃 `pixel_values`

接入 DINOv3 和 Cosmos3 时，视觉塔的接口都是标准的 `model(pixel_values=x) -> (B, N, D)`。
换到 Qwen3-VL 时，`Qwen3VLVisionModel.forward` 的签名是：

```python
forward(self, hidden_states, grid_thw, **kwargs)
```

它要的不是图像，而是**已经展平好的 patch 序列** `(seq_len, C · temporal_patch_size · p²)`
= `(seq_len, 1536)`，外加一张 `grid_thw` 表描述每张图的格点形状；内部用 `cu_seqlens` 做
变长注意力，整个 batch 是拼接成一条序列的。

所以"把图像切成 patch 并展平"这一步**不在模型里**，它在 HuggingFace 的
`Qwen2VLImageProcessorFast` 里。我们要么调它，要么自己实现。

### 1.3 为什么必须自己实现，而不是调官方 processor

我实测了三条理由，都不是风格偏好：

**(a) 速度差 369 倍，且发生在 CPU 上、阻塞训练循环。**

| 做法 | 256 张图耗时 |
|---|---:|
| 自己实现（GPU 上的纯 view/permute） | **0.95 ms** |
| 官方 processor（CPU 计算 + 拷回 GPU） | **350.51 ms** |

当前整步是 1.70 s，多加 350 ms 就是 **+20%**，而且是同步的 CPU 阻塞，dataloader 的
worker 并行度救不了它（它发生在主进程的前向里）。

**(b) 接口对不上，会引入 GPU→CPU→GPU 往返。**

processor 吃的是 numpy/PIL 的 uint8 HWC 图像，返回 **CPU 张量**（实测
`device=cpu`、`requires_grad=False`）。而我们的 dataloader 交出来的已经是 GPU 上归一化好的
float 张量（`_to_pixel_values` 按 backbone 各自的 mean/std 处理过）。走 processor 意味着每步
把整个 batch 搬回 CPU、转 numpy、再搬回 GPU，并且它会用自己的 rescale/normalize 逻辑覆盖掉
我们按 backbone 定制的归一化。

**(c) 最关键：它的格点是自适应的，会随图像长宽比变化。**

processor 按"像素预算"决定格点（`shortest_edge = 65536` 像素 = 256²），实测：

| 输入 | `grid_thw` | token 数 |
|---|---|---:|
| 224×224 | `[1, 16, 16]` | 256（被放大到 256²） |
| 256×256 | `[1, 16, 16]` | 256 |
| 240×320 | `[1, 16, 20]` | **320** |
| 480×640 | `[1, 30, 40]` | **1200** |

而 §1.1 的两处依赖都要求**固定的 16×16 = 256 格点**：VAE 的 latent 网格恰好是 16×16，
evidence bank 的尺寸也是固定的。让 processor 决定格点，等于让 batch 里不同来源、不同分辨率
的数据产生不同长度的 token 序列，直接破坏对齐。

结论：patch 化必须是我们自己控制的、GPU 上的、格点固定的纯张量操作。

### 1.4 坑：Qwen 的 patch 顺序不是行主序

自己实现就必须**逐位复刻**官方的内存布局——因为位置编码是按官方顺序训练出来的，顺序错了
每个 patch 都会拿到别的位置的位置编码。

#### 先约定符号

官方源码和本仓库用到的变量含义如下（以我们的实际配置 256×256 输入为例）：

| 变量 | 含义 | 我们的取值 |
|---|---|---|
| `b` / `batch_size` | 一个 batch 里的图像张数 | 例如 256 |
| `c` / `channel` | 图像通道数 | 3（RGB） |
| `p` / `patch_size` | 每个 patch 的边长（像素） | 16 |
| `grid_h`（代码里 `gh`） | **格点**的行数 = 图像高 ÷ `p` | 256 ÷ 16 = 16 |
| `grid_w`（代码里 `gw`） | 格点的列数 = 图像宽 ÷ `p` | 16 |
| `grid_t` | **时间**方向的格点数 = 帧数 ÷ `temporal_patch_size` | 1（单张静止图） |
| `temporal_patch_size`（检查脚本里 `tp`） | 时间方向每个 patch 吃几帧 | 2 |
| `m` / `merge_size` | merger 做 2×2 池化时的块边长 | 2 |
| `bh` | 块的行数 = `gh // m` | 16 ÷ 2 = 8 |
| `bw` | 块的列数 = `gw // m` | 8 |
| `mi` / `mj`（源码里的 `merge_h`/`merge_w`） | 块**内部**的行、列下标，取值 0..m-1 | 0 或 1 |
| `seq_len` | 展平后的 token 数 = `grid_t · gh · gw` | 1 × 16 × 16 = 256 |

关于 `grid_t`：Qwen 的这个塔图像和视频共用一套代码，所以时间维一直存在。视频有多帧时
`grid_t > 1`；我们喂的是**单张静止图**，所以 `grid_t = 1`。但 `temporal_patch_size = 2`
意味着每个 patch 在时间上要吃 2 帧，单图不够，于是官方把这一帧复制一份凑满
（`patches[:, -1:].repeat(...)`）。这就是为什么每个 patch 的向量长度是
`c · temporal_patch_size · p · p = 3 × 2 × 16 × 16 = 1536` 而不是 768。

#### 什么是"行主序"

**行主序（row-major）就是"从左到右、从上到下逐行扫描"的排列**——和读中文/英文的顺序一样。
对一个 `gh × gw` 的格点，位于第 `r` 行第 `c` 列的格子，其行主序下标是：

```
index = r * gw + c
```

例如 16×16 的格点：格子 (0,0) → 0，(0,1) → 1，……，(0,15) → 15，然后换行，
(1,0) → **16**，(1,1) → 17。

这是绝大多数视觉模型（含 DINOv3、SigLIP、Cosmos3）patch token 的默认排列，也是本仓库其余
部分默认的约定：`tokens[i]` 就是图像上第 `i` 个格子。VAE 的 16×16 latent 网格同样按行主序
展平。

#### Qwen 用的不是行主序

读官方实现（`image_processing_qwen2_vl_fast.py:242-262`）：

```python
patches = patches.view(
    batch_size, grid_t, temporal_patch_size, channel,
    grid_h // merge_size, merge_size, patch_size,   # 行方向拆成: 块行 bh, 块内行 mi, 像素 p
    grid_w // merge_size, merge_size, patch_size,   # 列方向拆成: 块列 bw, 块内列 mj, 像素 p
)
patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
# -> (batch, grid_t, bh, bw, mi, mj, channel, temporal_patch_size, p, p)
```

关键是 permute 之后、参与展平的维度顺序是 `grid_t, bh, bw, mi, mj`。reshape 成一维时，
**最右边的维度变化最快**，所以 token 的遍历顺序是：先固定一个 2×2 的块，走完块内 4 个格子，
再换下一个块。这么设计是为了让后面的 merger 直接 reshape 就能完成 2×2 池化。

> ⚠️ **官方注释在这里有个命名陷阱。** 源码那行注释写的是
> `(batch, grid_t, grid_h, grid_w, merge_h, merge_w, ...)`，但其中的 `grid_h`、`grid_w`
> 指的是 `grid_h // merge_size`、`grid_w // merge_size`，也就是**块的行列数**（本文的
> `bh`、`bw` = 8），而不是格点的行列数（16）。这个名字复用极易让人把布局误读成"在完整格点上
> 的行主序"——正是这个坑最容易被漏掉的原因。本文一律用 `bh`/`bw` 指块数，避免歧义。

用一个 **4×4 格点（`m=2`，即 2×2 个块）**的小例子看最清楚：

```
图像格点（(行,列)）        行主序下标            Qwen 的 token 编号
  (0,0) (0,1) (0,2) (0,3)    0   1   2   3         t0  t1  t4  t5
  (1,0) (1,1) (1,2) (1,3)    4   5   6   7         t2  t3  t6  t7
  (2,0) (2,1) (2,2) (2,3)    8   9  10  11         t8  t9  t12 t13
  (3,0) (3,1) (3,2) (3,3)   12  13  14  15         t10 t11 t14 t15
```

左上角那个 2×2 块占用了 `t0..t3`，而它们在行主序里是 0、1、**4**、**5** —— 不连续。

回到真实的 16×16 格点：

```
Qwen token 下标 : 0   1   2   3   4   5   6   7  ...
实际图像格子     : 0   1  16  17   2   3  18  19  ...
```

token 2 看着像"第 2 个格子"（第 0 行第 2 列），实际是第 16 个格子（第 1 行第 0 列）。
**256 个 token 里有 224 个落在错误的位置上**；恰好重合的 32 个也不是随便哪里，而是偶数行最
左两列和奇数行最右两列（`(0,0),(0,1),(1,14),(1,15),(2,0),(2,1),…`）——即两种排列在行首/行尾
的交汇处，属于巧合而非任何有意义的规律。

这个坑的危险之处在于它**完全不报错**：shape 是 `(B, 256, 1024)`，完全正确；训练照常进行；
loss 照常下降。只是 `v1[i] - v0[i]` 变成了"位置 A 的新特征减去位置 B 的旧特征"，而 VAE 重建
在拿一个被 2×2 分块置换过的目标做监督。

### 1.5 解决

**第一步，复刻官方布局。** 在 GPU 上用纯 view/permute 实现（`qwen3vl_encoder.py`）：

```python
x = pixel_values.view(b, c, bh, m, p, bw, m, p)
#                     b  c  ↑行方向三级↑  ↑列方向三级↑
#                           bh  mi p      bw  mj p
x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)          # (b, bh, bw, mi, mj, c, p, p)
x = x.unsqueeze(6).expand(..., temporal_patch_size, ...)   # 静止帧填满时间维
flat = x.reshape(b * gh * gw, c * temporal_patch_size * p * p)
```

第一行 `view` 把高、宽各拆成三级：高 = `bh`（块行）× `m`（块内行 `mi`）× `p`（块内像素行），
宽同理。`permute` 再把它们排成 `(b, bh, bw, mi, mj, c, p, p)`——**前面的 `bh, bw, mi, mj`
决定 token 顺序，后面的 `c, p, p` 是每个 token 的内容**。这个顺序与官方 permute 的结果逐位
对应。

时间维的处理也与官方一致：官方对单图是 `patches[:, -1:].repeat(...)` 复制最后一帧凑满
`temporal_patch_size`，我们用 `expand` 重复同一帧，结果相同（由 §1.6 的数值比对确认）。

**第二步，在塔的输出侧把顺序还原成行主序**，让模型其余部分完全不必知道这个塔的内部约定：

```python
tokens = tokens.view(b, bh, bw, m, m, d).permute(0, 1, 3, 2, 4, 5).reshape(b, gh * gw, d)
#          view 后:  (b, bh, bw, mi, mj, d)      d = 特征维（1024）
#       permute 后:  (b, bh, mi, bw, mj, d)
```

拆开看这个 permute 就是把**行方向的两级下标凑到一起、列方向的两级下标凑到一起**：

- `bh` 与 `mi` 相邻 → 合起来就是真实行号 `r = bh * m + mi`（0..15）
- `bw` 与 `mj` 相邻 → 合起来就是真实列号 `c = bw * m + mj`（0..15）

于是 `(b, bh, mi, bw, mj, d)` 实际上就是 `(b, r, c, d)`，最后 `reshape(b, gh*gw, d)` 把
`(r, c)` 按 `r * gw + c` 展平——这正是 §1.4 定义的行主序。

注意这里的取舍：**位置编码必须按 Qwen 的顺序喂进去**（否则用错位置编码），**输出必须按行主序
交出来**（否则下游用错位置）。两者缺一不可，不能只做一半——只做前者，下游全部错位；只做后者，
每个 patch 拿到别人的位置编码。

### 1.6 验证：四层，每层都能独立失败

只验证"能跑通"对这个问题毫无意义。`scripts/check_qwen3vl_vision.py` 做四层检查：

| 层次 | 检查什么 | 结果 |
|---|---|---|
| 归一化 | 模型用的 mean/std vs checkpoint 的 preprocessor 配置 | **一致**（0.5/0.5） |
| 输入布局 | 我们的展平结果 vs 官方 processor 的输出 | 最大绝对误差 **1.2e-07** |
| 逆变换 | 还原后的顺序 vs 手工算的行主序 | 最大绝对误差 **0.0** |
| 端到端因果 | 扰动图像第 i 个格子，变化最大的 token 是不是第 i 个 | 探测 0 / 17 / 35 / 255，**全部命中** |

关于 1.2e-07 而非严格 0：官方走的是融合的 `rescale_and_normalize`，检查脚本里是
`(x · rescale - mean) / std`，浮点运算顺序不同，这是 fp32 的舍入量级（fp32 的 eps 约
1.2e-07）。**布局是完全一致的，只有算术舍入有差异。**

> **归一化常数本身也是一个坑（已核实）。** `(x/255 - 0.5)/0.5` 确实是 Qwen3-VL 的官方归一
> 化：checkpoint 的 `preprocessor_config.json` 写的是
> `image_mean = image_std = [0.5, 0.5, 0.5]`，`rescale_factor = 1/255`。
>
> 但**类的默认值不是 0.5**。`Qwen2VLImageProcessorFast` 在源码里硬编码
> `image_mean = OPENAI_CLIP_MEAN = [0.481, 0.458, 0.408]`、
> `image_std = OPENAI_CLIP_STD`（Qwen2-VL 当年用的确实是 CLIP 归一化），只有从 checkpoint
> 加载时才被 `preprocessor_config.json` 覆盖成 0.5。
>
> ```
> loaded from checkpoint : [0.5, 0.5, 0.5]                          [0.5, 0.5, 0.5]
> bare class default     : [0.48145466, 0.4578275, 0.40821073]      [0.26862954, ...]
> ```
>
> 也就是说，照着 transformers 源码读默认值、或者从 Qwen2-VL 的代码里抄常数，都会拿到错误的
> 归一化——而且同样不会报错，只是把一个冻结的塔喂到了分布外。本仓库在
> `modeling_robo_contrast.py` 里为 `qwen3vl` 显式使用 0.5/0.5，与 checkpoint 一致。

**"端到端因果"这一层是最有价值的**：前面几层验证的是我推导的排列公式对不对，而它绕过了我
所有的推导——不看任何中间张量，只问"我改了图像右下角，模型输出里动得最厉害的是不是右下角
那个 token"。如果我对官方布局的理解从头到尾就是错的（正变换和逆变换错得一致，两两抵消），
前面几层完全可能同时通过，而这一层会失败。

顺带还检查了 batch 独立性：这个塔用 `cu_seqlens` 做变长注意力，如果分段失效，同一 batch 里
的图像会互相看到。打乱 batch 顺序后逐样本特征差异为 **0.0**，确认没有串扰。

### 1.7 复现

```bash
conda activate lerobot_v2
CUDA_VISIBLE_DEVICES=3 python -u scripts/check_qwen3vl_vision.py
```

### 1.8 教训

- **接口不同，往往意味着内存布局约定也不同。** `forward` 签名从 `pixel_values` 变成
  `(hidden_states, grid_thw)` 是一个信号：预处理被移出了模型，那么预处理里的隐含约定
  就成了调用方的责任。
- **凡是自己复刻官方预处理，必须和官方实现对数值**，不能只对 shape。shape 对而顺序错，是
  这类 bug 最典型的形态。
- **验证要能失败。** 布局类的检查如果我的理解整体错了就可能一起通过（正变换和逆变换错得
  一致会互相抵消）；真正兜底的是那个不依赖任何推导的因果探测。
- **不要从库的源码里读默认常数。** 归一化那条注记就是例子：类默认值是 Qwen2-VL 时代的 CLIP
  统计量，只有 checkpoint 的配置才是这个模型真正用的。常数要从权重目录里读，并写成断言。

---

> 第 1 章专门记录“不报错但会静默训练错误”的感知预处理问题。以下 Stage 2 专章同时记录
> 显式运行错误、分布式死锁风险、checkpoint 恢复错误和数据解码容错。

## 2. Stage 2 真实 checkpoint 不是普通 policy checkpoint

**涉及文件**：

```text
lerobot/common/policies/qwen3vl_mot/modeling_qwen3vl_mot.py
/Data/lzl/ace_stage1/step_16k/
```

### 2.1 现象

用户提供的 Stage 1 权重目录不是 `save_pretrained()` 导出的普通 policy，而是 DeepSpeed
ZeRO-2 checkpoint：

```text
mp_rank_00_model_states.pt
global_steps = 16000
```

直接按普通 policy checkpoint 加载会找不到模型文件或无法恢复配置。

同时，标准配置 JSON 顶层包含：

```json
{"type": "robo_contrast", ...}
```

旧 loader 直接交给 Draccus 解码，会报：

```text
The fields `type` are not valid for RoboContrastConfig
```

### 2.2 根因

- DeepSpeed 把真正参数放在 `module` 字段；
- checkpoint 目录没有最初完整的 Stage 1 config；
- 配置解码的两个入口对顶层 `type` 处理不一致。

### 2.3 解决

1. 支持解析 `mp_rank_00_model_states.pt` 的 `module`；
2. 根据权重 shape 恢复 Stage 1 精确配置；
3. 将恢复后的配置写入：

   ```text
   /Data/lzl/ace_stage1/step_16k/config.json
   ```

4. 所有 Stage 1 config 入口统一复用 `_decode_stage1_config()`，先剥离 `type`；
5. 加载前检查 Stage 2 必需的 vision、state/action projections 和 physical blocks。

### 2.4 证据

```text
checkpoint keys = 1397
model keys = 1397
shape mismatches = 0
missing = 0
unexpected = 0
```

这比 `strict=False` 后没有异常更强：它证明实际恢复出的配置与 checkpoint 完全一致。

### 2.5 `config.json` 中的模型目录不能覆盖当前运行环境

Stage 1 的 `config.json` 会完整保存训练时使用的资源路径，例如：

```text
qwen3vl_dir = /Data/lzl/huggingface/Qwen3-VL-4B-Instruct
cosmos3_dir = /Data/lzl/huggingface/Cosmos3-Edge
```

这些路径描述训练机器，不属于 checkpoint 的结构。若 Stage 2 在集群直接采用它们，即使
launcher 已传入 `/mnt/wangxiaofa/pt_weights/...`，构造 Stage 1 teacher 时仍会先访问旧的
`/Data` 路径。

Stage 2 现在只从 Stage 1 config 保留结构参数，并用当前 `Qwen3VLMoTConfig` 的
`qwen3vl_dir`、`cosmos3_dir` 覆盖资源路径。该规则同时用于：

- 首次从 Stage 1 DeepSpeed 或 safetensors checkpoint 初始化；
- 从已嵌入 `stage1_policy_config` 的 Stage 2 checkpoint 恢复；
- FSDP resume 读取已保存的 Stage 2 架构时，保留本次 launcher 传入的模型目录。

因此 checkpoint 可以在本地与集群之间移动，而不需要手工修改其中的 JSON。

---

## 3. Stage 1 vision 能迁移，但 text 不能假装迁移成功

**涉及文件**：

```text
lerobot/common/policies/qwen3vl_mot/stage1_transfer.py
```

### 3.1 现象

Stage 1 perception 的 vision 是 Qwen3-VL-compatible，但 text 是 SigLIP2。若只使用
`strict=False`，容易把大量 text missing keys 当成正常情况，而没有明确区分预期不兼容与
真正漏加载。

### 3.2 解决

迁移逻辑只复制：

- 去掉已知 wrapper prefix 后同名；
- tensor shape 完全相同。

vision 与 text 分别生成覆盖率报告，并允许独立设置：

```text
require_stage1_vision_transfer
require_stage1_text_transfer
```

当前结果：

```text
vision coverage = 100%
text coverage   = 0%
```

text 0% 是预期的，因此 Qwen text 保留自身 pretrained initialization；vision 若低于
90% 则直接报错。

---

## 4. FSDP2 DTensor 与 bitsandbytes AdamW8bit 不兼容

**涉及文件**：

```text
lerobot/common/utils/fsdp_training.py
lerobot/scripts/fsdp_train_contrast.py
```

### 4.1 现象

FSDP2 配合 bitsandbytes 0.48.2 做 optimizer step 时出现：

```text
bitsandbytes.optimizer_update_8bit_blockwise.default:
got mixed torch.Tensor and DTensor
```

模型 forward/backward 可以运行，但 optimizer update 失败。

### 4.2 根因

FSDP2 暴露的是 DTensor 参数和梯度，bitsandbytes 8-bit optimizer kernel 当前要求普通
`torch.Tensor`，不理解 DTensor placement。

### 4.3 解决

Stage 2 改用 classic FSDP：

```text
ShardingStrategy.FULL_SHARD
use_orig_params = true
```

这保留普通参数视图，同时仍分片参数、梯度和 optimizer state。

### 4.4 不能采用的替代方案

- 退回普通 AdamW：失去用户要求的 8-bit optimizer；
- 在 optimizer 前手工把 DTensor 转普通 Tensor：会破坏参数身份和分片语义；
- 吞掉异常继续训练：optimizer 根本没有更新。

---

## 5. 不能绕过 FSDP wrapper 直接访问 Qwen block 内部子模块

**涉及文件**：

```text
lerobot/common/policies/qwen3vl_mot/modeling_understanding.py
```

### 5.1 现象

最初为了导出 Qwen K/V，代码直接调用 decoder layer 内的：

```text
input_layernorm
k_proj
v_proj
```

classic FSDP block wrap 后，部分 rank 看到长度为 0 的 parameter shard，真实模型 forward
失败。

### 5.2 根因

FSDP 只在调用被包装模块的 `forward()` 时执行 all-gather。直接访问 wrapper 内部子模块，
等于绕过 FSDP 的参数 materialization 生命周期。

### 5.3 解决

必须正常调用完整 decoder layer forward。为获取 native K/V：

1. 在 `k_norm` 注册临时 forward hook；
2. 在 `v_proj` 注册临时 forward hook；
3. 正常执行 decoder block；
4. 取出 hook 输出；
5. 对 key 应用 Qwen 原生 rotary embedding；
6. 立即移除 hooks。

这样同时满足：

- FSDP block all-gather；
- gradient checkpointing；
- native K/V gradient；
- Qwen 原始 forward 语义。

相关回归测试会比较 checkpointed 与 non-checkpointed 的输出和梯度。

---

## 6. FP8 不是把 checkpoint 和 optimizer 参数永久变成 FP8

### 6.1 容易产生的误解

“FSDP 模型参数 FP8”容易被理解为：

- checkpoint 里存 FP8；
- optimizer 直接更新 FP8 parameter；
- FSDP 通信也是 FP8。

当前实现并不是这样。

### 6.2 当前训练语义

TorchAO FP8 的实际行为：

- trainable master parameters 保持 BF16；
- optimizer-facing gradients 保持 BF16；
- eligible Linear 的 GEMM 动态量化到 FP8；
- FSDP all-gather 仍是 BF16；
- checkpoint 保存 BF16 master weights；
- LayerNorm、embedding、小输出头、LoRA、teacher、VAE 和 physical encoder 不转换。

bitsandbytes AdamW8bit 只把大 tensor 的两个 moment 压到 8-bit；同一 optimizer state 中
仍可能混合 `uint8` 和 FP32 tensor。

### 6.3 A6000 与 H100 的区别

A6000 没有原生 FP8 Tensor Core：

```text
fp8_emulate = true
```

只能验证数值路径和模块兼容性，不能用于估算 H100 原生 FP8 加速比。原生 FP8 要求 compute
capability 9.0 或更高。

---

## 7. FSDP checkpoint 不能直接套标准 optimizer-state 聚合

**涉及文件**：

```text
lerobot/common/utils/fsdp_training.py
```

### 7.1 问题一：AdamW8bit state 类型混合

标准 FSDP optimizer-state 汇总假设 state tensor 能按统一规则重分片。bitsandbytes 中同一个
逻辑 state 可能同时包含：

- 8-bit moments；
- FP32 scale 和 metadata。

因此不能安全使用标准 FSDP optimizer state 汇总。

### 7.2 解决

- 模型使用 Distributed Checkpoint 保存可重分片的 FSDP model state；
- optimizer state 按 rank 保存原始本地 state；
- resume 要求 world size 不变；
- metadata 锁定 batch、accumulation、dataset、FSDP 和 FP8 geometry。

### 7.3 问题二：复制的冻结参数不会自动恢复

冻结 teacher、VAE 等参数通过 `ignored_states` 在每张卡复制。FSDP
`set_model_state_dict()` 不会自动复制这些 ignored parameters。

解决方式是：

1. 去掉 `_fsdp_wrapped_module.` 前缀得到 canonical name；
2. 从 model state 中找到对应 tensor；
3. 手动 `copy_()` 回冻结参数；
4. 缺失任一冻结参数都直接报错。

### 7.4 问题三：单 rank checkpoint I/O 失败会让其他 rank 永久等待

如果一个 rank 在保存或加载时抛错，其他 rank 可能继续卡在 barrier。

解决方式：

- 把 checkpoint I/O 拆成阶段；
- 每阶段后通过 collective 汇总成功/失败；
- 任意 rank 失败时，让所有 rank 一致抛出包含原始 rank 和错误信息的异常；
- 不让健康 rank 继续进入下一次 barrier。

故障注入证明双卡下不会再出现一个 rank 报错、另一个 rank 永久挂起。

### 7.5 模型初始化 seed

Generation Expert 是随机初始化的。FSDP shard 前所有 rank 必须使用同一个模型 seed，否则
每个 rank shard 的不是同一个全局模型。

正确顺序：

1. 所有 rank 设置同一个 model seed；
2. 构造完整 policy；
3. FSDP wrap；
4. 再设置 `model_seed + rank` 作为各 rank 的训练随机 seed。

---

## 8. Python 与 CUDA 动态库版本不一致

### 8.1 错误的 Python 环境

首次真实运行时 launcher 使用了基础 Python 3.12，而不是 `lerobot_v2` 环境，导致依赖和
扩展版本不一致。

解决方式：

- launcher 使用当前 `PYTHON_BIN`；
- 通过当前 Python 执行 `python -m torch.distributed.run`；
- 正式运行前显式进入 `lerobot_v2`。

### 8.2 NVRTC builtins 版本冲突

环境中：

```text
PyTorch = 2.9.1+cu130
LD_LIBRARY_PATH 优先指向 CUDA 12.4
```

NVRTC 报：

```text
failed to open libnvrtc-builtins.so.13.0
```

根因是 PyTorch wheel 使用 CUDA 13.0 runtime，但动态链接器先找到系统 CUDA 12.4。

launcher 会读取：

```python
torch.version.cuda
```

并优先把匹配的：

```text
site-packages/nvidia/cu13/lib
```

加入 `LD_LIBRARY_PATH`。

---

## 9. `--steps` 曾经不能限制 Stage 2 optimizer step

### 9.1 现象

设置：

```text
--steps=1
```

训练仍继续执行完整 epoch schedule，短 smoke 无法自动退出。

### 9.2 根因

训练循环只按 source-equivalent epoch schedule 结束，没有把 `cfg.steps` 作为 optimizer
update 上限。

### 9.3 解决

计划步数改为：

```text
min(natural_optimizer_steps, configured_steps)
```

循环同时检查：

- natural epoch schedule；
- optimizer update step cap。

最终真实数据 smoke 能在恰好 1 个 optimizer step 后退出。

---

## 10. 视频时间戳问题不是一种问题

**涉及文件**：

```text
lerobot/common/datasets_v30/video_utils.py
lerobot/common/datasets/contrastive_dataset.py
scripts/check_timestamp.py
```

### 10.1 loaded timestamp 比 query 大不一定是数据错误

视频只能返回真实存在的 frame PTS。query 落在两帧之间时，最近帧可能在 query 之后，因此：

```text
loaded_timestamp > query_timestamp
```

本身是正常的。真正需要判断的是绝对误差是否超过 tolerance。

### 10.2 `ego10k` 的亚毫秒边界误差

出现：

```text
tensor([0.0002]) > tolerance_s=0.0002
```

打印值被四舍五入，实际值略大于 `2e-4`。这类误差远小于一帧，不应丢弃样本。

v3 dataset 默认 tolerance 调整为：

```text
2e-4 -> 5e-4 seconds
```

0.5 ms 仍远低于 30 FPS 的约 33.3 ms 帧间隔。

### 10.3 `file-199.mp4` 的 33 ms 偏移不是同步错误

问题视频：

```text
/media/v-wangxiaofa/新加卷/lerobot_data/file-199.mp4
```

报错：

```text
query  8830.5332, 8832.1338
loaded 8830.5664, 8832.1670
error  about 33.2 ms
```

MP4 中正确 PTS 实际存在：

```text
8830.533268
8832.133268
```

视频在 `6206.6666s` 处只有一次极小 cadence 调整：

```text
normal PTS delta = 512 / 15360 s
one PTS delta    = 511 / 15360 s
```

TorchCodec 0.8.0 的 `seek_mode=approximate` 会根据平均帧率估算 seek 位置。该微小 PTS
提前使 approximate seek 在后半段跳到下一帧。`seek_mode=exact` 能返回正确帧，
但每个 200 MB 文件的 decoder 初始化从约 0.02 s 增加到约 1.0–1.3 s，不适合全局启用。

### 10.4 float32 不能用于长视频的亚毫秒 timestamp check

在约 8832 秒处，float32 的 ULP 已达到：

```text
0.9765625 ms
```

原始查询：

```text
8832.133333...
```

转成 float32 后打印为：

```text
8832.1338
```

即使正确帧也可能因为 float32 量化看起来相差约 1 ms。`check_timestamp.py` 使用 float64，
所以它能找到正确 PTS，而训练旧代码会误判。

解决方式：

- TorchCodec 和 PyAV 的 timestamp distance 全部使用 float64；
- 不把 tolerance 粗暴提高到 34 ms。

### 10.5 当前 decoder fallback 链

现在的顺序是：

```text
TorchCodec
  └─ FrameTimestampError -> PyAV 读取同一组 timestamps
       └─ 仍失败 -> dataset 随机选择同数据集的另一个 frame
```

只对 `FrameTimestampError` 自动切换 PyAV；其他 TorchCodec 初始化或解码异常仍显式抛出，
避免用 fallback 掩盖真正的软件错误。

真实 `file-199.mp4` 上，PyAV 返回结果与 TorchCodec exact 模式逐像素一致。

---

## 11. 读取失败后固定回退到 `dataset[0]` 会产生偏置

### 11.1 旧行为

任何样本读取异常后：

```python
item = dataset[0]
```

训练继续，但会产生三个问题：

- 高频错误时反复训练同一个第 0 帧；
- sampler 设计的 episode grouping 被破坏；
- 第 0 帧本身坏掉时会再次抛错，没有解释。

### 11.2 新行为

PyAV 仍失败后，从同一个 dataset 内选择另一个 frame：

- 排除原失败 frame；
- seed 由 dataset seed、epoch、dataset id 和请求索引组成；
- 相同实验可复现；
- 保持 dataset mixture 比例；
- fallback frame 也失败时显式终止，而不是继续伪造成功。

返回 batch 会包含：

```text
data_read_fallback = 1
requested_frame_index = 原始索引
frame_index = 随机替换索引
```

训练日志与 W&B 汇总：

```text
data/read_fallback_count_interval
data/read_fallback_rate_interval
data/read_fallback_count_total
data/read_fallback_rate_total
data/read_fallback_count_by_dataset/<name>
data/read_fallback_rate_by_dataset/<name>
```

如果 PyAV 成功读取原样本，则不算 sample replacement，因此不会增加
`data_read_fallback`。

---

## 12. 数据集声明 FPS、真实 FPS 和 timestamp time base 不能混用

部分 FTP-1 数据声明 30 FPS，但真实采集只有约 10–15 FPS。

需要区分：

- `index_fps`：metadata 声明值，决定 parquet timestamp 和 delta query 的时间基准；
- `true_fps`：真实采集率，决定一个时间窗口覆盖多少真实运动时间；
- video PTS：解码器实际返回的 presentation timestamp。

delta timestamps 必须使用 `index_fps` 构造，否则 loader 会查询错误帧；窗口长度和
`sample_rate` 则应使用 `true_fps`，否则物理时间跨度错误。

这也是 `dataset_fps.py` 单独存在的原因。

---

## 13. 缺失数据集和 dataloader 长尾必须显式记录

真实 `debug_research_data` 运行时发现：

```text
agibot_alpha       打开失败
language_table     不存在
ms_data_xdof_3     不存在
ego_dex_split4     不存在
```

dataset constructor 会记录 warning 并跳过无法打开的数据集，而不是让整个 mixture 初始化失败。
因此训练开始日志必须记录实际成功加载的数据集和 source frame 数，不能只看配置 mixture。

4×A6000 的 40 步真实训练中：

```text
dataloader median = 0.34 s
single long tail  = 27.01 s
```

平均值会被少数冷读、seek 或损坏视频拉高，应同时看 median、P90 和最大值。

---

## 14. GitHub push 失败不是 Git 对象损坏

提交：

```text
683b5fe2e34a3ac02dfedae010fd3e41a5f77581
```

无法上传时检查结果：

- commit 很小；
- 没有大文件；
- Git 对象没有损坏；
- HTTPS 没有 credential helper；
- SSH key 对应用户 `Penation`；
- 该账号对 `lzl2040/gcr_latent_action` 没有写权限；
- 未发现可用的 `Penation/gcr_latent_action` fork。

因此解决方向是仓库权限或 fork，而不是重写 commit、压缩 Git 对象或删除历史。用户选择
“先不提交”，所以 commit 保留在本地，未 push。

---

## 15. 当前仍未解决或尚未接入的事项

### 15.1 Stage 2 三视角

OXE 数据已有 `primary / secondary / wrist` 三槽位，但 Stage 2 当前仍只使用 primary。
后续接入时必须同时处理：

- 缺失视角 camera mask；
- `secondary` 在不同数据集中的语义差异；
- 三路时间戳分别同步；
- 总 visual token budget；
- view dropout；
- 是否只生成 primary future。

### 15.2 Embodiment action adapter

模型能够生成 normalized canonical 40D action，但 `select_action()` 尚未实现：

- canonical slot 到真实机器人 action layout；
- inverse normalization；
- controller 输出格式。

### 15.3 Video sampler

目前只有 video flow training objective，没有完整的：

- 多步采样；
- unpatchify；
- VAE decode；
- 视频保存与评估。

### 15.4 8×H100 完整模型速度

当前有：

- 1×H100 与 8×H100 synthetic native-FP8 smoke；
- 4×A6000 完整模型、真实数据、FP8 emulation 实测。

仍缺少 8×H100 上完整 6B+ 模型、真实数据、micro-batch 16 的稳定 20–50 步测量。
因此 `doc/stage2_train_cost.md` 中完整训练天数仍是估计，而不是 H100 实测值。
