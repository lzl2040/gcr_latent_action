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

读官方实现（`image_processing_qwen2_vl_fast.py:242-262`）：

```python
patches = patches.view(
    batch_size, grid_t, temporal_patch_size, channel,
    grid_h // merge_size, merge_size, patch_size,
    grid_w // merge_size, merge_size, patch_size,
)
patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
# -> (batch, grid_t, gh/m, gw/m, m, m, channel, temporal_patch_size, p, p)
```

关键在这个 permute：token 维的展开顺序是 `grid_t, bh, bw, m_h, m_w`。也就是说，
**它先按 2×2 的 merge block 走，再走 block 内部的 4 个格子**——这是为了让后面的 merger
能直接 reshape 就完成 2×2 池化。

这**不是**行主序。把 token 下标映射回图像格子（16×16 格点，m=2）：

```
Qwen token 下标 : 0   1   2   3   4   5   6   7  ...
实际图像格子     : 0   1  16  17   2   3  18  19  ...
```

token 2 看着像"第 2 个格子"，实际是第 16 个格子（第 1 行第 0 列）。**256 个 token 里有
224 个落在错误的位置上**；恰好重合的 32 个也不是随便哪里，而是偶数行最左两列和奇数行最右
两列（`(0,0),(0,1),(1,14),(1,15),(2,0),(2,1),…`）——即两种排列在行首/行尾的交汇处，属于巧合
而非任何有意义的规律。

这个坑的危险之处在于它**完全不报错**：shape 是 `(B, 256, 1024)`，完全正确；训练照常进行；
loss 照常下降。只是 `v1[i] - v0[i]` 变成了"位置 A 的新特征减去位置 B 的旧特征"，而 VAE 重建
在拿一个被 2×2 分块置换过的目标做监督。

### 1.5 解决

**第一步，复刻官方布局。** 在 GPU 上用纯 view/permute 实现（`qwen3vl_encoder.py`）：

```python
x = pixel_values.view(b, c, bh, m, p, bw, m, p)
x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)          # (b, bh, bw, m, m, c, p, p)
x = x.unsqueeze(6).expand(..., temporal_patch_size, ...)   # 静止帧填满时间维
flat = x.reshape(b * gh * gw, c * temporal_patch_size * p * p)
```

维度顺序 `(b, bh, bw, m_h, m_w, c, T, p, p)` 与官方 permute 的结果逐位对应。时间维的处理也
与官方一致：官方对单图是 `patches[:, -1:].repeat(...)` 复制最后一帧，我们用 `expand` 重复同
一帧，结果相同。

**第二步，在塔的输出侧把顺序还原成行主序**，让模型其余部分完全不必知道这个塔的内部约定：

```python
tokens = tokens.view(b, bh, bw, m, m, d).permute(0, 1, 3, 2, 4, 5).reshape(b, gh * gw, d)
#                    (b, bh, m_h, bw, m_w, d) -> (b, gh, gw, d)
```

把 `m_h` 挪到 `bh` 之后、`m_w` 挪到 `bw` 之后，就是标准的行主序展开。

注意这里的取舍：**位置编码必须按 Qwen 的顺序喂进去**（否则用错位置编码），**输出必须按行主序
交出来**（否则下游用错位置）。两者缺一不可，不能只做一半。

### 1.6 验证：三层，每层都能独立失败

只验证"能跑通"对这个问题毫无意义。`scripts/check_qwen3vl_vision.py` 做三层检查：

| 层次 | 检查什么 | 结果 |
|---|---|---|
| 输入布局 | 我们的展平结果 vs 官方 processor 的输出 | 最大绝对误差 **5.9e-08** |
| 逆变换 | 还原后的顺序 vs 手工算的行主序 | 最大绝对误差 **0.0** |
| 端到端因果 | 扰动图像第 i 个格子，变化最大的 token 是不是第 i 个 | 探测 0 / 17 / 35 / 255，**全部命中** |

关于 5.9e-08 而非严格 0：官方走的是融合的 `rescale_and_normalize`，我们是
`(x/255 - 0.5)/0.5`，浮点运算顺序不同，这是 fp32 的舍入量级。**布局是完全一致的，只有算术
舍入有差异。**

第三层是最有价值的：前两层验证的是我推导的排列公式对不对，而第三层绕过了我所有的推导——它
不看任何中间张量，只问"我改了图像左下角，模型输出里动得最厉害的是不是左下角那个 token"。
如果我对官方布局的理解从头到尾就是错的（两层推导错得一致），前两层可能同时通过，但第三层
会失败。

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
- **验证要能失败。** 前两层检查如果我的理解整体错了就可能一起通过；真正兜底的是第三层那个
  不依赖任何推导的因果探测。
