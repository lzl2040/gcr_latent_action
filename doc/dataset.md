# 对比学习数据管线

本文描述 `MultiModalContrastiveDataset` 及其配套的规范空间与批采样器：它们如何把一堆
模态、维度、坐标约定都不一致的机器人数据集，变成模型可以直接消费的统一样本。

涉及三个文件：

| 文件 | 职责 |
| --- | --- |
| `lerobot/common/datasets/canonical_space.py` | 定义 40 维规范物理空间，以及每个数据集到它的映射表 |
| `lerobot/common/datasets/contrastive_dataset.py` | 数据集类本体 + `contrastive_collate_fn` |
| `lerobot/common/datasets/contrastive_sampler.py` | `ContrastiveBatchSampler`，决定一个 batch 里的负样本长什么样 |

---

## 1. 规范物理空间（`canonical_space.py`）

数据集之间最大的不一致在物理侧：有的只给末端位姿，有的只给关节角，有的两者都给，
臂数也不同。这里的做法是**定义一个足够宽的槽位空间，每个数据集只填自己有的槽位，
其余留空并用 mask 标记**。

```
CANON_DIM = 40

[ 0:3 ] eef0_xyz      [ 3:9 ] eef0_rot6d    [ 9:10] eef0_gripper
[10:13] eef1_xyz      [13:19] eef1_rot6d    [19:20] eef1_gripper
[20:28] joint0 (7 关节 + 夹爪)
[28:36] joint1 (7 关节 + 夹爪)
[36:40] reserved
```

**为什么必须配 mask，而不是直接补零。** 只给关节角的数据集（如
`ftp_1_VisuoTactile_D-WHEEL_split_0`，它连 `action` 这个 key 都没有，只有
`action.arm_joint`）在 `[0:20]` 全是零。如果不带 mask，模型就无法区分"这个数据集没有末端
位姿"和"末端恰好在原点"。物理编码器里每个投影都吃 `[value * mask, mask]`，正是为了消除
这个歧义。

映射用一张 `PHYSICAL_SPECS` 表描述，每条是一个 `(源 key, 源起, 源止, 目标起)` 的拷贝指令：

```python
"ftp_1_sharpa_split_0": {
    "action": [
        _seg("action.eef_pose",  0, 20,  0),   # 双臂末端位姿 -> [0:20]
        _seg("action.arm_joint", 0,  8, 20),   # 左臂关节     -> [20:28]
        _seg("action.arm_joint", 8, 16, 28),   # 右臂关节     -> [28:36]
    ],
    "state": [...],
    "tactile_image": ["observation.images.tactile_left_0", ...],
},
```

表里没有的数据集走 `default_spec()`，按宽度启发式猜：宽度 10 → 单臂末端，20 → 双臂末端，
14/16 → 对半拆成两个关节块，其余丢进关节槽。这是在信息不足时唯一诚实的解释，**新数据集
应当显式写进 `PHYSICAL_SPECS`，不要依赖启发式**。

触觉分两类，处理方式完全不同：

- `tactile_signal`：力/力矩/taxel 等低维读数，多个 key 顺序拼接进 32 维向量。它和
  state/action 本质同模态，直接投影成一个 token。
- `tactile_image`：触觉相机，最多 4 路。维度远高于其他物理模态，需要单独的 CNN 和门控
  （见 §5）。

---

## 2. 数据集类（`MultiModalContrastiveDataset`）

### 2.1 构造

按 mixture 名（如 `debug_research_data`）从 `OXE_NAMED_MIXTURES` 取数据集列表，逐个：

1. 读 `meta/info.json` 判断 codebase 版本，**v2.1 与 v3.0 走不同的 `LeRobotDataset` 实现**；
2. `get_spec()` 拿到规范映射；
3. 解析相机角色（`primary` / `secondary` / `wrist`）到真实 key，主相机找不到时退化为任意
   非触觉相机，保证样本仍可用；
4. 构造 `delta_timestamps`，决定读哪些时刻的哪些 key；
5. 把该数据集自己的统计量投影进规范槽位。

其中两处值得留意。

**只 chunk 真正会用到的 action key。** `resolve_delta_timestamps` 默认会给所有 `action.*`
列都排上时间偏移，但 spec 可能只读其中两个；给一个 44 维手部关节列做 16 帧 chunk 是纯粹
浪费 IO。所以做了一次过滤：

```python
wanted_action_keys = {src for src, *_ in spec.get("action", [])}
delta_timestamps = {k: v for k, v in resolved.items() if k in wanted_action_keys}
```

**归一化是 per-dataset 而不是全 mixture 的**，这是刻意的。对比学习模型极其擅长找捷径，如果
各数据集共用一套均值方差，那么"这条样本的数值尺度"本身就泄露了它来自哪个数据集，模型可以
靠匹配数据集身份而不是匹配运动来降低损失。按数据集各自归一化，就把这条捷径堵死了。

### 2.2 两种索引方式

```python
dataset[i]                  # 第 i 条 per-epoch 采样计划
dataset[(ds_idx, frame_idx)] # 显式指定数据集与帧，供 ContrastiveBatchSampler 使用
```

训练走的是第二种。第一种（`_build_sampling_plan` 按权重预抽一份计划）保留给不使用自定义
采样器的场景。`set_epoch(epoch)` 会用 `seed + epoch` 重建计划。

任何一帧读取失败都会被 catch 并回退到 `dataset[0]`，只打 warning——一个损坏的视频帧不应该
让整个训练崩掉。

### 2.3 每个模态读多少帧

对比学习对齐的是**窗口 `[t, t+H]` 内发生的事**，所以物理侧原则上每个模态都该读整个窗口。
但读取代价差异极大，因此按代价分档：

| 模态 | 存储 | 读取范围 | 理由 |
| --- | --- | --- | --- |
| action | parquet | 16 帧 | 同 row group，几乎免费 |
| state | parquet | 16 帧 | 同上 |
| 触觉信号 | parquet | `t`, `t+H` | 同上，但只需要变化量 |
| 触觉图像 | **视频** | `t`, `t+H` | 16 帧 × 4 路解码正好砸在磁盘瓶颈上 |
| RGB | **视频** | `t`, `t+H` | 感知侧本来就只需要两端 |

**为什么 state 值得读整个窗口。** 本 mixture 里多数数据集的 `action` 是绝对目标位姿，动作块
本身已经描绘了窗口内的意图轨迹。但 action 是**被命令的**，state 轨迹是**实际发生的**——两者
在接触、柔顺、够不到、夹爪打滑这些情形下会分叉，而这恰恰是视觉上看得见的部分。对比目标来自
视频里实际发生的变化，所以实际轨迹严格更接近监督信号。对于 action 是增量/速度、或只有关节
的数据集（如 `ftp_1_VisuoTactile_D-WHEEL_split_0`），这就不只是精修了。

**为什么触觉必须读两帧。** 触觉的全部价值在接触事件——抓取闭合、滑移、碰撞——而接触是
**窗口内发生的事件**。只取 `t` 时刻，如果抓取是在 `t+8` 完成的，那么 `t` 帧显示的是接触之前，
字面意义上什么都没有，最有信息量的信号被系统性地看不见。两帧以 2× 而非 16× 的解码代价
捕捉到"接触前 → 接触后"。

实测三个触觉数据集在窗口内的变化幅度（随机 mid-episode 采样 12 条，触觉图像为 0–255 尺度）：

| 数据集 | state 漂移(中位) | 触觉图像 \|t1−t0\| | 触觉信号 delta |
| --- | --- | --- | --- |
| `ftp_1_RH20TCfg5Franka` | 0.580 | — | 0.834 |
| `ftp_1_sharpa_split_0` | 0.308 | 0.493（max 7.60） | — |
| `ftp_1_VisuoTactile_D-WHEEL_split_0` | 0.326 | 2.741 | — |

sharpa 的中位数低而最大值高，正说明大部分窗口无接触变化、少数窗口有——这正是我们希望模型
去分辨的信号。

> 注意：在 **episode 开头**采样时这些数值会全为 0（机器人还没动）。排查窗口读取问题时务必用
> mid-episode 帧，否则会把"还没开始动"误判成"窗口没读到"。

---

## 3. 返回值

`__getitem__` 返回一个 dict。下表中 `H = frame_horizon`（默认等于 `chunk_size` = 16），
`S = img_size`（`dataset.image_transforms.img_size`，默认 224）；触觉图为 `policy.tactile_img_size`，默认 112。

### 感知侧

| key | shape / dtype | 含义 |
| --- | --- | --- |
| `image_t0` | `(3, S, S)` uint8 | 主相机在 `t` 时刻 |
| `image_t1` | `(3, S, S)` uint8 | 主相机在 `t + H` 时刻 |
| `pair_is_valid` | `()` float32 | `t + H` 越过 episode 结尾被 clamp 时为 0 |
| `task` | `str` | 语言指令（**不是 tensor**） |

### 物理侧

| key | shape / dtype | 含义 |
| --- | --- | --- |
| `action` | `(16, 40)` float32 | 规范空间动作块，已归一化并乘过 mask |
| `action_mask` | `(40,)` float32 | 哪些槽位有效（整个 chunk 共用一份） |
| `observation.state` | `(16, 40)` float32 | 规范空间状态**轨迹**，与动作块同窗口 |
| `state_mask` | `(40,)` float32 | 同上 |
| `tactile_signal` | `(2, 32)` float32 | 低维触觉读数，`t` 与 `t+H` 两帧 |
| `tactile_signal_mask` | `()` float32 | 标量，1 表示这条样本有触觉信号 |
| `tactile_image` | `(4, 2, 3, 112, 112)` uint8 | 触觉相机，4 路 × 2 帧，槽位对齐，缺的补零 |
| `tactile_image_mask` | `(4,)` float32 | 逐路有效性 |

### 元信息

| key | shape / dtype | 含义 |
| --- | --- | --- |
| `sample_rate` | `()` int64 | 该数据集 fps，模型用它做 embedding |
| `dataset_id` | `()` int64 | mixture 内的数据集下标 |
| `episode_uid` | `()` int64 | `ds_idx * 1_000_000 + episode_index`，跨数据集唯一 |
| `frame_index` | `()` int64 | 绝对帧号 |

几点说明：

- **每个模态读多少帧，由它的读取代价决定。** state 和触觉信号是 parquet 列，与 action 同
  row group，取满 16 帧几乎免费；触觉相机是视频，4 路 × 16 帧解码正好砸在本管线已经存在的
  磁盘瓶颈上，所以只取两端。见 §2.3。
- **图像保持 uint8 到模型内部才归一化。** 这样 CPU→GPU 的传输量是 float32 的 1/4，而
  resize 之后的归一化在 GPU 上几乎免费。
- `action_mask` / `state_mask` 是 `spec` 声明的槽位与该数据集统计量里实际存在的 key
  取交集的结果——声明了但 stats 里没有的槽位会被置 0。整个 chunk 共用一份 mask，因为槽位
  的有效性由数据集 schema 决定，不随时间变化。
- `sample_rate` 是必要的：同样 16 帧动作，在 10 fps 和 30 fps 数据集上覆盖的物理时长差三倍，
  不告诉模型的话它没法把动作块和视觉变化对齐。
- `episode_uid` 与 `frame_index` 供损失函数排除假负样本（同 episode 且帧距过近的样本）。
- `pair_is_valid` 目前**由数据集产出但模型未消费**，是一个已知的待办。
- 某个 key 若因故没读到窗口（只回来一帧），`_build_canonical_vector` /
  `_build_tactile_*` 会把它广播成整个窗口而不是报错。这保证了鲁棒性，但也意味着**"窗口读取
  失败"是静默的**——排查时可以测"窗口内是否真的有变化"，见 §6。

`contrastive_collate_fn` 对 tensor 做 stack，对 `task` 保留为 python list（tokenizer 在模型里调用）。

---

## 4. 批采样器（`ContrastiveBatchSampler`）

均匀随机采样会让对比任务过于简单：把厨房场景和工厂场景区分开完全不需要理解运动。这个采样器
**从构造上**制造困难负样本：

```
batch (256)
├── same_dataset_frac = 0.75  -> 192 条来自同一个数据集（共享本体、相机、场景统计）
│   ├── episode_group_frac = 0.75 -> 144 条按 episode_group_size = 8 成组抽自同一 episode
│   └── 剩余 48 条在该数据集内随机
└── 剩余 64 条从整个 mixture 随机（保留一些简单负样本，避免坍缩到单一数据集统计）
```

同 episode 的帧是最强的负样本：场景、机器人完全相同，**只有运动不同**，模型只能靠理解运动
来区分。

但这引入了假负样本风险：同 episode 里相隔几帧的两个样本描述的几乎是同一段运动，把它们当负
样本会把本该相近的表征推开。所以组内帧被强制至少相隔 `min_frame_gap = 32` 帧，实现上用的是
**基于 stride 的抽取而非拒绝采样**——把可用区间切成 `span // min_frame_gap` 个槽，无放回地
抽槽再在槽内随机，这样间隔约束是构造性满足的，不会退化成死循环。模型侧的
`_false_negative_mask` 是这条约束的兜底。

每个 episode 的可用区间是 `[start, end - horizon)`，保证未来帧和完整动作块都存在、无需 clamp。

分布式方面：所有 rank 共享同一个 seed，由 `global_batch_id = local_batch_id * num_replicas + rank`
去相关，因此各 rank 必须对采样计划取得一致——这也是训练脚本里不给各 rank 分配不同 seed 的原因。

---

## 5. 数据侧与模型侧的衔接

有几处设计是数据侧和模型侧配合完成的，单看一边会觉得奇怪：

**触觉图像的槽位对齐。** 数据集固定输出 4 路（不足补零 + mask），模型侧缺失的路用学习到的
`missing` token 顶替。这样 0 路触觉的数据集和 4 路触觉的数据集产生**完全相同形状**的
token 序列，无需动态 shape。

**这不只是为了方便。** 在 ZeRO-2 下，"哪些参数收到梯度"决定梯度归约调度；如果某个 rank 的
batch 恰好没有触觉数据而跳过了触觉参数，它就会与其他 rank 失步，表现为 600 秒 NCCL 超时而
不是报错。模型侧因此始终保留至少一行输入过 CNN（没有真实数据时把贡献乘 0）。

**触觉不做池化。** 每路触觉一个 token，因为不同手指接触物体的不同部位，平均掉恰好毁掉我们
想要的接触模式。防止触觉主导靠的是另外三件事：从零初始化的 tanh 门控、
`modality_dropout_tactile = 0.3`、以及 0.1× 的触觉学习率。

**两帧被折进同一个 token，而不是各占一个。** 触觉图像 token 由 `[feat_t, feat_t1 - feat_t]`
投影而成，触觉信号同理把两帧拼接后投影。这样触觉从"看不到接触发生"变成"看得到接触变化"，
而 token 数一点没涨——否则触觉会从 14 个内容 token 里的 36% 涨到 50%，反过来挤压动作块。
差分项才是重点：模型要对齐的是窗口内的变化，不是 `t` 时刻的绝对读数。

**state 与 action 共享位置编码。** 第 `i` 组 state 和第 `i` 组 action 覆盖同样的帧，给它们
相同的位置编码，trunk 才能在对应时刻上比较"命令值"与"实际值"；模态 embedding 已经负责区分
两者的身份。

最终物理侧序列为 **22 个 token**：1 CLS + 8 state + 8 action + 1 触觉信号 + 4 触觉图像。

---

## 6. 已知坑与注意事项

- **不要用 `datasets.Dataset.select(非连续索引)`。** 它会在磁盘上物化一份 indices mapping；
  在 3100 万行的 YAM 数据集上实测 2646 ms，而 `hf_dataset[list][key]` 只要 0.44 ms。上游注释
  推荐 `.select()`，在这个数据规模下是错的。
- **吞吐瓶颈在磁盘。** `/Data` 是机械盘，约 95% util、100 ms await、208 IOPS。计算侧
  1.5–2 s/step 基本是免费的，这也是模型能放心扩到 700M+ 的原因。也正因如此，触觉视频取 16 帧
  的方案被否掉——它涨的恰好是已经最紧的那个资源。
- `data_s` 只在 rank 0 上统计，所以 rank 1 的读取停顿会表现为 `updt_s` 异常升高，不要误判成
  通信问题。
- **窗口读取失败是静默的**：某个 key 只回来一帧时会被广播成整个窗口，不报错。验证方法是测
  "窗口内是否真有变化"，并且**必须用 mid-episode 帧**——episode 开头机器人还没动，任何模态
  都是常数。
- 新增数据集时请显式写 `PHYSICAL_SPECS` 条目并核对槽位；启发式映射只是兜底。
- mask / 光流等感知模态在设计中已预留位置，但尚未实现。
