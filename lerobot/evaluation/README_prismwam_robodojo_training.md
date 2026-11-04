# PrismWAM 在 RoboDojo 上的训练配置

整理日期：2026-11-04。本文汇总上游公开的 **PrismWAM 基础版**
RoboDojo 训练配方，不是本仓库 Qwen3VL-MoT 可直接加载的配置，也不表示已在本机训练或复现。

## 1. 模型版本与初始化

“没有预训练”准确地说是**没有大规模机器人动作预训练**，不是完全随机初始化。

| 项目 | 基础版公开配方 |
| --- | --- |
| 视频骨干 | `Wan-AI/Wan2.2-TI2V-5B`，使用预训练权重 |
| Stage 1 | 将视频 Transformer 适配为深度视频预测模型 |
| Stage 2 共享 RGB/depth blocks | 从 Stage 1 的 `step_015000.pt` 初始化 |
| Stage 2 RGB 输入/输出层 | 从预训练视频模型初始化 |
| 动作专家 | `init/action_initializer.pt`，配置为 `action_init_source: wan` |
| VAE | `init/depth_vae.pt`、`init/rgb_vae.safetensors` |
| 文本 | 使用预计算指令文本特征；不是本仓库的 Qwen 理解分支 |

上游报告的结果，非本机复现：

| 模型 | RoboDojo 综合得分 | 成功率 |
| --- | ---: | ---: |
| PrismWAM 基础版 | 25.10 | 18.55% |
| PrismWAM* | 36.23 | 29.30% |

PrismWAM* 是约 15B 的探索版本，从 Wan2.1-14B 初始化，且有额外视频预训练。
**不要把下面的基础版配方当作星号版的完整配方或效果保证。**

## 2. 数据、观测与动作

| 参数 | 配置 |
| --- | --- |
| 训练任务 | RoboDojo 的 35 个训练任务，ARX X5 |
| 数据格式 | 原始 HDF5，`DATA_ROOT/robodojo/<task>/arx_x5/data/` |
| episode 选择 | `episode_split: full`，验证 episode 也包含在训练中 |
| 采样策略 | `sampling_mode: task_balanced` |
| 相机 | `cam_head`、`cam_left_wrist`、`cam_right_wrist` |
| 画面组织 | 三路相机分别拼成一幅 RGB 图和一幅 depth 图，尺寸 `384 x 320` |
| 深度范围 | `depth_min: 0.0`、`depth_max: 1500.0`，毫米 |
| 深度序列 | 当前帧加 8 个未来帧，`num_video_frames: 9`、`frame_stride: 4` |
| 窗口起点间隔 | `sample_stride: 1` |
| 训练末尾窗口 | `drop_incomplete_windows: false` |
| 动作 | `action/joint_vector`，14 维双臂**绝对关节目标** |
| 当前机器人状态 | `state/joint_vector`，14 维关节状态 |
| 动作长度 | `action_horizon: 32` |
| 时间对齐 | `action_start_offset: 1`、`proprio_start_offset: 0` |
| 动作归一化 | `normalize_actions: true`，统计缓存 `robodojo/action_stats.pt` |
| 标准差下限 | `action_std_floor: 1e-6` |
| 统计缓存策略 | `action_stats_cache_policy: fixed` |
| 指令来源 | HDF5，`instruction_type: seen`、`instruction_selection: episode_hash` |
| 指令上下文长度 | `context_len: 128` |

Stage 2 的 RGB 记忆是 **episode 首帧、32 步前、当前帧**：
`rgb_history_num_frames: 3`、`sparse_recent_action_offset: 32`、
`sparse_context_mode: anchor_recent_current`。
历史帧使用独立编码，`sparse_frame_encoding: independent`。

训练时 RGB 条件 latent 加噪概率为 `0.2`，depth 条件加噪概率为 `0.0`。
RoboDojo 配方设定 `rgb_proprio_context: false`：RGB 分支的上下文不包含机器人状态。

## 3. 两阶段训练预算与目标

| 配置 | Stage 1：深度预热 | Stage 2：联合训练 |
| --- | --- | --- |
| 入口 | `scripts/robodojo/stage1_depth.sh` | `scripts/robodojo/stage2_joint.sh` |
| 配方 | `configs/robodojo/stage1_depth.yaml` | `configs/robodojo/stage2_joint.yaml` |
| 输入 | 当前深度、指令 | RGB 记忆、当前深度、当前关节状态、指令 |
| 目标 | 8 帧未来深度 | 1 帧 RGB 目标、8 帧未来深度、32 步 action |
| RGB 目标时刻 | 无 | 当前时刻之后第 32 步，`rgb_target_mode: f8` |
| 训练部分 | 视觉 Transformer | 共享 RGB/depth Transformer 与独立动作专家 |
| 训练步数预算 | `15000` | `40000` |
| global batch | `1024` | `1024` |
| 单卡 batch 默认值 | `32` | `16` |
| DataLoader workers | `16` | `16` |
| `num_epochs` 配置字段 | `20`，继承阶段配置 | `20`，继承阶段配置 |
| loss | depth flow matching，权重 `1.0` | RGB/depth/action flow matching，权重各 `1.0` |
| 最终权重 | `stage1_depth/checkpoints/weights/step_015000.pt` | `stage2_joint/checkpoints/weights/step_040000.pt` |

两阶段都是 flow matching。Stage 2 的 RGB、depth、action 使用各自的噪声水平；
三个 scheduler 的 `train_shift` 和 `infer_shift` 均为 `5.0`，
`num_train_timesteps: 1000`。

联合阶段的配置预算是 `40000 x 1024 = 40,960,000` 次样本呈现，
深度预热阶段另有 `15000 x 1024 = 15,360,000` 次。
这只是按步数和 global batch 计算的预算；窗口与任务平衡采样使它不能直接换算为
“每一帧恰好遍历多少次”。上游按 15k/40k 步描述配方，不能只看到
`num_epochs: 20` 就将其等同于本仓库训练循环中的 epoch 定义。

## 4. 优化与保存

| 参数 | 两阶段共用设置 |
| --- | --- |
| 优化器 | AdamW |
| betas | `(0.9, 0.95)` |
| 学习率 | `1e-4` |
| warmup | 前 5%，按阶段预算分别约 750 / 2000 步 |
| 调度 | cosine，衰减至 `1e-6` |
| weight decay | `0.01` |
| 梯度裁剪 | `max_grad_norm: 1.0` |
| 精度与并行 | BF16，DeepSpeed ZeRO-2 |
| 随机种子 | `seed: 42`、`eval_seed: 42` |
| 日志 | `log_every: 10` |
| 权重保存 | `save_every: 1000`，`save_at_end: true` |
| 完整训练状态保存 | `state_save_every: 5000` |
| 训练内验证 | `eval_every: 500`，`eval_samples_per_process: 1` |
| 验证采样 | `eval_sampling_mode: robodojo_episode_random` |
| 验证时间预算 | Stage 1 为 180 秒，Stage 2 为 300 秒 |

不要把训练内验证等同于完整 RoboDojo 闭环成功率评测。

启动器自动计算：

```text
gradient_accumulation_steps =
    global_batch_size / (batch_size_per_gpu x total_gpu_count)
```

例如使用 16 卡且保持默认单卡 batch，Stage 1 梯度累积为 2，Stage 2 为 4，
两者的 global batch 仍为 1024。这是配置算术示例，不是显存可行性验证。

## 5. 架构与训练/推理的区别

RGB 与 depth 共享视觉 Transformer blocks；action 使用独立专家。
视觉分支为 30 层、hidden size 3072，动作分支为 30 层、hidden size 1024。
注意力遵循单向链路：

```text
RGB 语义条件 -> Depth 几何运动 -> Action
```

稀疏未来 RGB 帧只作为训练目标，没有其他分支读取其预测 token，因此推理时不生成
未来 RGB，只联合去噪未来 depth 和 action。
**它不预测未来关节 state 向量，但仍通过未来深度监督物理演化；
不是简单的“删掉 state loss，只保留 RGB 视频和 action”。**

## 6. 上游运行方式

以下命令在**上游 PrismWAM 仓库**执行，不是在本仓库执行。
先按上游 data guide 准备 HDF5 索引、动作统计和文本缓存，并下载视频骨干与 `init/`。
`DATA_ROOT` 指向包含 `robodojo/` 的父目录；`WEIGHTS_ROOT` 指向视频模型与 `init/` 的父目录。

```bash
export DATA_ROOT=/path/to/data
export WEIGHTS_ROOT=/path/to/weights
export OUTPUT_ROOT=/path/to/outputs
export ROBODOJO_ROOT=/path/to/RoboDojo

bash scripts/robodojo/stage1_depth.sh --dry-run
bash scripts/robodojo/stage1_depth.sh
bash scripts/robodojo/stage2_joint.sh --dry-run
bash scripts/robodojo/stage2_joint.sh
```

Stage 2 默认读取：
`$OUTPUT_ROOT/robodojo/stage1_depth/checkpoints/weights/step_015000.pt`。
多节点时，每个节点配置 `NNODES`、`NODE_RANK`、`MASTER_ADDR`，共享数据、权重与输出路径。
具体依赖、数据准备和导出流程以上游指南为准；这里没有安装环境或启动训练。

## 7. 闭环评测设置与复现边界

| 配置 | 上游报告设置 |
| --- | --- |
| RoboDojo 版本 | `ee67a1468510da7624a089164402359f2afc72c8` |
| XPolicyLab 版本 | `ddf5dcfff5e089a323da1d22891909b72864647e` |
| Assets revision | `c70d27efddca73b784441942bc3b8d1b361c684e` |
| 深度补丁 | `patches/robodojo_depth.patch` |
| 深度输入 | `approximate_depth`，uint16 毫米；三相机启用 `distance_to_image_plane` |
| 动作类型 | `joint` |
| 预测 / 执行长度 | `32 / 16` |
| 去噪步数 | `10` |
| depth/action inference shift | 均为 `5.0` |
| 评测 seed | `0` |
| 推理条件加噪 | 关闭 |

上游明确提示后续 RoboDojo 版本可能因渲染同步和关节 reset 改动产生不同分数。
它使用当前深度和绝对关节控制；本仓库现有 Qwen 评测主要使用 RGB、当前末端状态和
delta 末端控制。**不能复用我们的 delta 动作统计，也不能只改 loss 权重就视为完成复现。**

## 8. 官方来源

以下链接指向上游 `main`，未来可能变化；本文记录的是本次整理时读取的公开配方。

- [仓库及作者报告结果](https://github.com/THUSI-Lab/PrismWAM)
- [项目页：PrismWAM 与 PrismWAM* 的区别](https://prism-wam.github.io/)
- [模型与训练初始化资产](https://github.com/THUSI-Lab/PrismWAM/blob/main/docs/models.md)
- [数据准备](https://github.com/THUSI-Lab/PrismWAM/blob/main/docs/data.md)
- [训练指南](https://github.com/THUSI-Lab/PrismWAM/blob/main/docs/training.md)
- [RoboDojo 指南及版本约束](https://github.com/THUSI-Lab/PrismWAM/blob/main/docs/benchmarks/robodojo.md)
- [公共配置](https://github.com/THUSI-Lab/PrismWAM/blob/main/configs/base.yaml)
- [Stage 1 公共配置](https://github.com/THUSI-Lab/PrismWAM/blob/main/configs/stages/stage1_depth.yaml)
- [Stage 1 RoboDojo 配置](https://github.com/THUSI-Lab/PrismWAM/blob/main/configs/robodojo/stage1_depth.yaml)
- [Stage 2 公共配置](https://github.com/THUSI-Lab/PrismWAM/blob/main/configs/stages/stage2_joint.yaml)
- [Stage 2 RoboDojo 配置](https://github.com/THUSI-Lab/PrismWAM/blob/main/configs/robodojo/stage2_joint.yaml)
