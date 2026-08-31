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
