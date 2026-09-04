# 触觉模块设计与 AnyTouch 图像编码流程

本文档说明当前 `robo_contrast` 模型如何统一处理异构触觉数据，重点介绍
`tactile_backbone="anytouch"` 时触觉图像从数据读取到进入物理 Transformer 的完整路径。

相关实现：

- 数据读取：`lerobot/common/datasets/contrastive_dataset.py`
- 配置：`lerobot/common/policies/ace/configuration_robo_contrast.py`
- AnyTouch encoder：`lerobot/common/policies/ace/anytouch_tactile.py`
- 物理分支：`lerobot/common/policies/ace/modeling_robo_contrast.py`

---

## 1. 触觉数据分为两条独立路径

数据集里的触觉并不统一，因此模型不强制把所有触觉都转换成图像或都转换成向量，而是分为：

| 类型 | 例子 | 模型输入 | 编码路径 |
|---|---|---|---|
| 信号型触觉 | 力、力矩、关节扭矩、阵列信号 | `(B, S, 32)` | `signal_proj`，不经过 AnyTouch |
| 图像型触觉 | GelSight、Sharpa、D-WHEEL、MCTac | `(B, V, F, 3, H, W)` | ResNet、FTP-1 或 AnyTouch |
| 无触觉 | 普通视觉机器人数据 | 全零占位 + mask | learned missing token |

其中：

- `B`：batch size。
- `S`：state/action 的时间长度，当前默认 32。
- `V`：每个样本的触觉 pad/camera 数，当前最多 6。
- `F`：每个 pad 读取的触觉帧数，当前默认 4。

例如 `ftp_1_RH20TCfg5Franka` 的触觉是六维夹爪力/力矩信号，因此它只走
`tactile_signal` 路径，不会送进 AnyTouch。Sharpa 和 D-WHEEL 提供触觉视频，才会进入图像路径。

两条触觉路径最后都会变成物理 Transformer 的 token，因此同一个 batch 可以同时包含：

- 有图像触觉的样本；
- 只有信号触觉的样本；
- 完全没有触觉的样本。

---

## 2. 图像触觉的数据读取

### 2.1 时间采样

对于每个触觉 pad，dataset 在与 vision、state 和 action 相同的时间窗口内均匀读取 4 帧。
默认窗口约为 1.6 秒，因此可以近似理解为：

```text
frame 0: t
frame 1: t + H/3
frame 2: t + 2H/3
frame 3: t + H
```

实际时间长度还会受到数据集帧率和 `chunk_frames_min/max` 的约束。输出为：

```text
tactile_image:      uint8 (B, V, 4, 3, S, S)
tactile_image_mask: float (B, V)
```

使用 AnyTouch 或 FTP-1 时，配置会把 `S` 强制设为 224。

### 2.2 无效触觉 pad 过滤

部分数据虽然存在触觉视频文件，但某些 pad 长时间为全黑、纯色或静态占位图。dataset 会计算每一帧、
每个颜色通道内部的空间标准差：

```text
spatial_std = std(frame[channel, :, :])
```

只有当至少一个采样帧、至少一个通道具有足够空间结构时，该 pad 才被标为有效。默认阈值：

```text
tactile_dead_std = 0.002
```

这个阈值基于 `[0,1]` 像素范围；一个 8-bit 灰度级约为 `1/255 = 0.0039`。

被判定为无效的 pad：

- 不进入触觉图像 backbone；
- 不产生真实 backbone feature；
- 在物理 Transformer 中由 learned missing token 表示。

模型 forward 还会根据 `tactile_image_mask` 只挑出 batch 中真实有效的 pad。假设输入是：

```text
(B=128, V=6, F=4, 3, 224, 224)
```

最多有 768 个 pad，但如果只有 110 个有效 pad，AnyTouch 只编码这 110 个，不会处理另外 658 个
零占位 pad。

---

## 3. 可选的图像触觉 backbone

配置项：

```text
policy.tactile_backbone
```

支持三种实现：

| backbone | 单帧/时序编码 | 输出 | 是否训练 | 触觉重建 |
|---|---|---:|---|---|
| `resnet18` | 4 帧分别经过 ResNet，再用 `TactilePadTemporal` 融合 | 2 × 512/pad | 训练 | 开启 |
| `ftp1` | 4 帧分别经过 sensor-specific tokenizer，再做时序融合 | 2 × 512/pad | 主干冻结 | 关闭 |
| `anytouch` | 两组三帧直接经过动态 ViT | 2 × 768/pad，再适配到 512 | 主干冻结 | 关闭 |

AnyTouch 不再经过 `TactilePadTemporal`，因为它的 3D patch embedding 和 ViT 已经直接处理三帧时序。

---

## 4. AnyTouch 权重加载

当前使用的权重：

```text
/Data/lzl/huggingface/anytouch_encoder.pth
```

它实际上是 AnyTouch 官方完整 stage-2 checkpoint，约 2.9 GiB，包含：

- text tower；
- vision tower；
- touch encoder；
- MAE decoder。

本项目不会把整个 checkpoint 注册为模型参数，只加载触觉推理需要的部分：

```text
touch_model                  24 层 CLIP ViT-L/14
video_patch_embedding        三帧 Conv3d patch embedding
sensor_token                 10 组、每组 5 个 sensor token
touch_projection             1024 -> 768
```

不会加载：

- text tower；
- 普通 vision tower；
- MAE decoder；
- 未被官方动态 forward 使用的 `video_position_embedding`。

保留的 CLIP encoder 使用严格 key/shape 加载。任何权重缺失或形状不匹配都会直接报错，避免随机初始化
的层产生看似正常但实际错误的特征。

当前实际注册到模型里的 AnyTouch 参数约为：

```text
305.2M，全部冻结
```

代码基于 AnyTouch 仓库 commit：

```text
9c43a1a6eb38d904fd767712eb9dcb2d98b8d56b
```

---

## 5. 四帧如何变成两个 AnyTouch token

AnyTouch 官方动态 encoder 一次使用三帧。当前默认每个 pad 有四帧，因此构造两个重叠窗口：

```text
window 0 = [frame 0, frame 1, frame 2]
window 1 = [frame 1, frame 2, frame 3]
```

形状变化为：

```text
输入有效 pad:
(N, 4, 3, 224, 224)

构造两个窗口:
(N, 2, 3, 3, 224, 224)

合并 pad 和窗口维:
(2N, 3, 3, 224, 224)
```

这里的两个 `3` 分别表示：

```text
第一个 3：三帧时间
第二个 3：RGB
```

两个窗口分别输出一个 CLS feature，所以每个 pad 最终产生两个时序 token：

```text
(N, 2, 768)
```

如果设置：

```text
policy.tactile_tokens_per_pad=1
```

则只使用前三帧 `[frame 0, frame 1, frame 2]`，每个 pad 只运行一次 AnyTouch，计算量大约减半。

当 `tactile_frames > 4` 时，双 token 模式使用最前面的三帧和最后面的三帧；默认配置固定为 4 帧。

---

## 6. AnyTouch 图像预处理

每个三帧窗口执行：

```text
uint8 [0,255]
    -> float [0,1]
    -> resize 224 x 224
    -> ImageNet normalization
```

归一化参数为：

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

不进行 center crop。

需要注意 AnyTouch 官方动态路径采用了一个不常见但必须保留的 tensor 布局：

```text
(B, T=3, C=3, H, W)
```

它直接把这个 tensor 送入：

```text
Conv3d(
    in_channels=3,
    out_channels=1024,
    kernel_size=(3, 14, 14),
    stride=(3, 14, 14),
)
```

因此官方预训练权重实际上把：

- 三帧时间放在 Conv3d 的 input-channel 维；
- RGB 放在 Conv3d 的 depth 维。

虽然常规视频模型一般使用 `(B,C,T,H,W)`，这里不能主动“修正”维度顺序，否则同一组权重会被应用到
不同语义的维度上，得到的就不再是 AnyTouch 训练出来的运算。

---

## 7. AnyTouch ViT 内部 token

经过 3D patch embedding 后：

```text
224 / 14 = 16
16 x 16 = 256 patch tokens
patch hidden size = 1024
```

送入 ViT 的完整序列为：

```text
1 CLS token
+ 5 sensor tokens
+ 256 tactile patch tokens
= 262 tokens
```

Transformer 配置：

```text
CLIP ViT-L/14
hidden size       = 1024
layers            = 24
attention heads   = 16
MLP hidden size   = 4096
patch size        = 14
```

编码完成后不平均全部 patch，而是使用 AnyTouch stage-2 实际监督的 CLS 表示：

```text
last_hidden_state[:, 0]
    -> post LayerNorm
    -> pretrained touch_projection
    -> 768-dimensional feature
```

这比自行对 256 个 patch token 求平均更符合原 checkpoint 的训练目标。

---

## 8. Universal sensor token

AnyTouch 的已知 sensor id 包括：

```text
0: GelSight
1: DIGIT
2: GelSlim
3: GelSight Mini
4: DuraGel
```

当前 `debug_research_data` 中的主要图像触觉包括 Sharpa 和 OpenLoong VTouch，无法可靠映射到上述
传感器。当前实现统一使用：

```text
sensor_id = -1
```

也就是最后一组 universal sensor token。

AnyTouch 预训练时专门以一定概率把已知 sensor id 替换成 `-1`，因此这个 token 的用途就是在传感器
身份未知或未见过时提供共享表示。相比把 Sharpa 强行声明成 GelSight，使用 universal token 更稳妥。

左手、右手以及不同触觉 pad 的身份不由 AnyTouch sensor token 表示，而由物理分支自己的：

```text
tactile_view_embed
```

表示，因此不同手指的特征不会因为共用 universal sensor token 而失去位置身份。

---

## 9. 从 768 维 AnyTouch 特征接入物理分支

AnyTouch 每个 pad 输出：

```text
(N, 2, 768)
```

首先经过可训练 adapter：

```text
Linear(768, 512)
LayerNorm(512)
```

得到：

```text
(N, 2, 512)
```

然后把有效 pad 放回原 batch/view 位置：

```text
(B, V, 2, 512)
```

再把 pad 和 token 维展开，通过：

```text
tactile_img_proj: Linear(512, 1024)
```

得到物理 Transformer 的触觉图像 token：

```text
(B, V * 2, 1024)
```

每个 token 还会加入三种 embedding：

```text
tactile_view_embed   哪一个手指/pad/camera
tactile_token_embed  该 pad 的第一个还是第二个时序窗口
modality_embed       这是触觉图像模态
```

最后与其他物理 token 拼接：

```text
[physical CLS]
[state tokens]
[action tokens]
[tactile signal tokens]
[tactile image tokens]
```

默认：

```text
group count G = chunk_size / group_size = 32 / 4 = 8
CLS tokens    = 1
state         = 8
action        = 8
tactile signal= 8
tactile image = 6 pads * 2 tokens = 12
total         = 37 tokens
```

所以即使 AnyTouch 内部使用 256 个 patch token，每个 pad 离开 AnyTouch 后仍只有两个 token，不会让
高维触觉图像淹没 state 和 action。

---

## 10. 冻结、梯度和防止触觉占主导

### 10.1 冻结范围

以下 AnyTouch 参数全部冻结，并且始终保持 eval mode：

- 24 层 tactile ViT；
- Conv3d patch embedding；
- sensor token；
- AnyTouch 自带的 `1024 -> 768` projection。

训练的部分包括：

- `768 -> 512` adapter；
- `tactile_img_proj`；
- tactile gate；
- view/token/modality embedding；
- 后续 physical Transformer。

AnyTouch forward 位于 `torch.no_grad()` 中，因此不保存 24 层 ViT 的反向激活。

### 10.2 触觉 gate

触觉图像 token 在进入物理 Transformer 前会乘以：

```text
tanh(tactile_image_gate)
```

该参数初始化为 0，所以模型训练开始时触觉图像通道是关闭的。模型先利用 state/action 学习基础对齐，
只有当触觉能够降低目标损失时，gate 才逐渐打开。

这也意味着第一步中主要先更新 gate；adapter 会在 gate 打开后逐渐获得有效梯度。

### 10.3 Modality dropout

默认：

```text
modality_dropout_tactile = 0.3
```

训练时会随机隐藏整个样本的触觉模态，避免模型把数据集身份或任务结果完全建立在某一种触觉传感器上。

### 10.4 为什么关闭 tactile reconstruction

AnyTouch 模式会强制：

```text
tactile_recon_weight = 0
```

原因是：

1. AnyTouch 主干已经冻结；
2. 本项目没有加载其 MAE decoder；
3. 在冻结 feature 后额外训练一个 RGB decoder 无法反过来塑造 AnyTouch encoder。

AnyTouch adapter 和物理分支仍然通过感知侧/物理侧的对比损失训练。

---

## 11. 缺失触觉与分布式训练

当某个样本没有触觉图像时，相应 pad 使用 learned missing token。

当一个 rank 的整个 local batch 都没有有效图像触觉时：

- 不运行冻结的 AnyTouch ViT；
- 创建一个零的 768 维占位 feature；
- 仍然经过 trainable adapter 和 `tactile_img_proj`；
- 最终由 mask 把它替换成 missing token。

这样既不浪费 305M AnyTouch 主干的前向计算，也保证 adapter/projection 始终出现在 autograd 图中。
这对 ZeRO-2 很重要：不同 rank 不能因为本地 batch 是否包含触觉而改变参与梯度同步的可训练参数集合。

---

## 12. 显存控制

同数据集采样会使某些 batch 包含大量有效触觉 pad。AnyTouch 每个 pad 又有两个三帧窗口，因此实际
ViT batch 是：

```text
有效 pad 数 x tactile_tokens_per_pad
```

配置：

```text
policy.anytouch_forward_batch_size=128
```

会把窗口拆成最多 128 个一组依次执行。因为 AnyTouch：

- 完全冻结；
- 没有 BatchNorm；
- 不在不同窗口之间做 attention；

所以分块不会改变输出，只用于限制 ViT attention/MLP 的瞬时显存。

---

## 13. 参数量和实测开销

默认 DINOv3 感知分支下：

| 触觉 backbone | 总参数 | 可训练参数 | batch 128 代表性 step | CUDA allocated 峰值 |
|---|---:|---:|---:|---:|
| ResNet-18 | 774.4M | 406.5M | 1.01 s | 10.83 GiB |
| AnyTouch | 1058.6M | 385.4M | 1.62 s | 9.66 GiB |

AnyTouch 增加了约 305.2M 冻结参数，但移除了可训练 ResNet、`TactilePadTemporal` 和触觉重建头，所以
可训练参数反而从 406.5M 降到 385.4M。

上述结果来自 RTX A6000、bf16、真实 `debug_research_data`、batch 128。代表性 AnyTouch step
处理约 176 个有效 pad，即 352 个三帧窗口。共享机器上的绝对时间会受其他任务影响，应主要参考
ResNet/AnyTouch 相对差异。

正式 `train_ace.sh` 的单卡 ZeRO-2 batch 128 实测也已跑通：

```text
有效触觉 pad: 110
模型更新: 3.54 s
```

该次测试还包含正式训练入口、DeepSpeed 初始化和真实视频数据读取。

---

## 14. 当前局限

### 14.1 时间尺度与 AnyTouch 原训练分布不同

AnyTouch 原始动态数据通常使用相邻或较近的历史帧，而当前模型的四帧均匀分布在约 1.6 秒机器人
窗口内。因此它更接近编码：

```text
接触状态在一段机器人动作中的变化
```

而不是：

```text
高频振动或毫秒级滑动
```

这是明确的时序分布偏移。AnyTouch 应当和 ResNet 基线做真实训练 A/B，而不能只因为使用了更大的
预训练模型就假定一定更好。

### 14.2 Universal token 不是传感器校准

Universal sensor token 能处理未知传感器，但不能代替：

- 颜色和曝光校准；
- gel 几何校准；
- 力/形变量标定；
- sensor-specific domain adaptation。

如果未来某类触觉传感器数据量足够，可以考虑增加传感器映射、轻量 LoRA 或蒸馏，但初版不建议直接
解冻 305M ViT。

### 14.3 在线编码成本

双窗口 AnyTouch 比 ResNet 明显更慢。如果训练吞吐优先，可以依次考虑：

1. 设置 `tactile_tokens_per_pad=1`，每个 pad 只编码一个三帧窗口；
2. 离线缓存 AnyTouch feature；
3. 用 AnyTouch feature 蒸馏当前小型触觉 encoder；
4. 保持 ResNet 为主训练路径，只把 AnyTouch 用于 A/B 或 teacher。

### 14.4 权重许可证

AnyTouch 仓库代码是 MIT License，但 checkpoint 页面没有单独明确权重许可证。科研实验可以保留
当前来源和 commit 记录；严格商业使用前应向作者确认权重授权范围。

---

## 15. 使用方式

### 本地

`train_ace_local.sh` 会透传额外配置：

```bash
conda activate lerobot_v2

bash train_ace_local.sh \
  --policy.tactile_backbone=anytouch \
  --policy.anytouch_checkpoint=/Data/lzl/huggingface/anytouch_encoder.pth \
  --policy.anytouch_forward_batch_size=128
```

单 token、低计算量版本：

```bash
bash train_ace_local.sh \
  --policy.tactile_backbone=anytouch \
  --policy.anytouch_checkpoint=/Data/lzl/huggingface/anytouch_encoder.pth \
  --policy.tactile_tokens_per_pad=1
```

### 集群

```bash
bash train_ace.sh \
  --job_name anytouch_contrast \
  --tactile_backbone anytouch \
  --anytouch_checkpoint /mnt/wangxiaofa/pt_weights/anytouch_encoder.pth \
  --anytouch_forward_batch_size 128
```

### 独立检查 checkpoint 和窗口顺序

```bash
python scripts/check_anytouch_tactile.py \
  --checkpoint /Data/lzl/huggingface/anytouch_encoder.pth \
  --device cuda
```

该脚本会检查：

- checkpoint 能否严格加载；
- 输出是否为 `(N, 2, 768)`；
- 重复前向是否确定；
- 修改第 4 帧是否只影响 `[1,2,3]` 对应的第二个 token；
- 输出 AnyTouch 的总参数量和可训练参数量，供人工确认冻结状态。

### 真实数据性能分析

```bash
python scripts/profile_contrastive_step.py \
  --mix debug_research_data \
  --batch_size 128 \
  --steps 6 \
  --tactile_backbone anytouch \
  --anytouch_checkpoint /Data/lzl/huggingface/anytouch_encoder.pth
```
