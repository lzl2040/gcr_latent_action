# LIBERO 评测记录

更新日期：2026-10-24。

本文汇总已完成的正式评测，共 60 份结果、12,000 次 rollout。
其中旧协议 42 份、8,400 次，单次初始化 RNG 的新协议 18 份、3,600 次，分开记录。
数据来自对应的
`results.json`，不包含 smoke test、首次成功即停止的检查或尚未完成的评测。

## 历史评测设置

下表及“旧协议结果汇总”使用旧协议：每个 episode 前按
`args.seed + task_id * 1000 + trial` 重设随机种子。
当前评测代码已改为仅在启动时用 `args.seed` 初始化一次 RNG，
后续任务、episode 和 replanning 连续推进随机状态，不再重设种子；
固定 benchmark 初态的选择不变。新结果记录
`sampling_rng="run_seeded_continuous"`，应与旧协议结果分开标注。

| 项目 | 设置 |
| --- | --- |
| 任务与初态 | 每套 10 个任务，任务编号 0–9；每个任务使用前 20 个固定初态，共 200 次 |
| 随机种子 | `seed=0`；环境 episode seed 为 `task_id * 1000 + trial` |
| 动作预测与执行 | 每次预测 32 步，执行前 10 步后重新规划 |
| Flow 积分步数 | 20 |
| 控制频率 | 20 Hz |
| 初始静置 | 10 步，不计入下方任务控制步数上限 |
| 控制步数上限 | Object：280；Spatial：220；Goal：300 |
| 图像 | 256 × 256 RGB，按 checkpoint 保存的视角配置输入 |
| 相机映射 | `primary`：`agentview_image`；`wrist`：`robot0_eye_in_hand_image` |
| 图像方向 | 模拟器图像旋转 180°，与训练数据方向一致 |
| 状态与动作 | canonical 40D，LIBERO 前 10D 有效；使用各 suite 独立的训练数据 mean/std |
| 输出 | 保存每次 rollout 的动作和失败 rollout 视频；不解码、导出预测视频 |

训练数据根目录为 `/Data/lerobot_data_ort6d/v30`，分别使用
`libero_object_no_noops_lerobot`、`libero_spatial_no_noops_lerobot` 和
`libero_goal_no_noops_lerobot` 的统计量。

## 旧协议结果汇总

每个单元格格式为 **成功率（成功次数 / 总次数）**。
`—` 表示没有已完成的正式评测记录，不代表成功率为 0。

| 训练版本 | Checkpoint | 输入视角 | Object | Spatial | Goal |
| --- | --- | --- | --- | --- | --- |
| `1014_v0` | 18k | `primary` | 71.5%（143/200） | — | 47.0%（94/200） |
| `1014_v0` | 24k | `primary` | 80.0%（160/200） | 69.5%（139/200） | 53.0%（106/200） |
| `1014_v0` | 32k | `primary` | 85.0%（170/200） | 81.0%（162/200） | — |
| `1014_v0` | 40k | `primary` | 86.0%（172/200） | 85.5%（171/200） | 64.0%（128/200） |
| `1014_v0` | 48k | `primary` | 88.0%（176/200） | 87.0%（174/200） | — |
| `1014_v1_8gpu` | 10k | `primary` + `wrist` | 82.0%（164/200） | 78.5%（157/200） | 65.0%（130/200） |
| `1014_v1_8gpu` | 16k | `primary` + `wrist` | 92.0%（184/200） | 87.0%（174/200） | 63.5%（127/200） |
| `1014_v1_8gpu` | 20k | `primary` + `wrist` | 94.5%（189/200） | 73.0%（146/200） | 73.5%（147/200） |
| `1014_v1_8gpu` | 24k | `primary` + `wrist` | 91.5%（183/200） | 83.5%（167/200） | 83.0%（166/200） |
| `1014_v1_8gpu` | 28k | `primary` + `wrist` | 97.0%（194/200） | 85.0%（170/200） | 81.5%（163/200） |
| `1014_v1_8gpu` | 32k | `primary` + `wrist` | 96.0%（192/200） | 89.5%（179/200） | 86.5%（173/200） |
| `1014_v1_8gpu` | 36k | `primary` + `wrist` | 95.5%（191/200） | 89.5%（179/200） | 89.0%（178/200） |
| `1014_v1_8gpu` | 40k | `primary` + `wrist` | 96.0%（192/200） | 92.0%（184/200） | 88.5%（177/200） |
| `1014_v1_8gpu` | 44k | `primary` + `wrist` | 97.0%（194/200） | 91.0%（182/200） | 88.5%（177/200） |
| `1014_v1_8gpu` | 48k | `primary` + `wrist` | 97.5%（195/200） | 90.0%（180/200） | 87.0%（174/200） |

`1014_v0` 的 48k 相比同版本 40k，Object 提升 2.0 个百分点，
Spatial 提升 1.5 个百分点。

`1014_v1_8gpu` 的 16k 相比同版本 10k，Object 提升 10.0 个百分点，
Spatial 提升 8.5 个百分点，Goal 下降 1.5 个百分点。

`1014_v1_8gpu` 的 20k 相比同版本 16k，Object 提升 2.5 个百分点，
Spatial 下降 14.0 个百分点，Goal 提升 10.0 个百分点。

`1014_v1_8gpu` 的 24k 相比同版本 20k，Object 下降 3.0 个百分点，
Spatial 提升 10.5 个百分点，Goal 提升 9.5 个百分点。
Goal 任务 0（打开中层抽屉）从 20k 的 0/20 提高到 6/20（30.0%），
仍是 24k Goal 中成功率最低的任务。

`1014_v1_8gpu` 的 28k 相比同版本 24k，Object 提升 5.5 个百分点，
Spatial 提升 1.5 个百分点，Goal 下降 1.5 个百分点。
Goal 任务 0（打开中层抽屉）为 5/20（25.0%），24k 为 6/20（30.0%）；
任务 3（打开上层抽屉并将碗放入）为 9/20（45.0%）。
这两个任务合计贡献 28k Goal 的 26/37 次失败。

`1014_v1_8gpu` 的 32k 相比同版本 28k，Object 下降 1.0 个百分点，
Spatial 提升 4.5 个百分点，Goal 提升 5.0 个百分点。
Goal 任务 0（打开中层抽屉）从 28k 的 5/20（25.0%）提高到 11/20（55.0%）；
任务 3（打开上层抽屉并将碗放入）和任务 9（将酒瓶放上酒架）均为 12/20（60.0%）。
这三个任务合计贡献 32k Goal 的 25/27 次失败。

`1014_v1_8gpu` 的 36k 相比同版本 32k，Object 下降 0.5 个百分点，
Spatial 持平，Goal 提升 2.5 个百分点。
Goal 任务 0（打开中层抽屉）为 11/20（55.0%），与 32k 持平；
任务 3（打开上层抽屉并将碗放入）为 15/20（75.0%），任务 9（将酒瓶放上酒架）为 12/20（60.0%）。
这三个任务贡献 36k Goal 的全部 22 次失败，其余 7 个任务合计 140/140。

`1014_v1_8gpu` 的 40k 相比同版本 36k，Object 提升 0.5 个百分点，
Spatial 提升 2.5 个百分点，Goal 下降 0.5 个百分点。
Spatial 任务 4（从上层抽屉取碗）从 36k 的 6/20（30.0%）提高到 9/20（45.0%）。
Goal 任务 0（打开中层抽屉）从 36k 的 11/20（55.0%）提高到 12/20（60.0%）；
任务 3（打开上层抽屉并将碗放入）和任务 9（将酒瓶放上酒架）均为 14/20（70.0%）。
这三个任务合计贡献 40k Goal 的 20/23 次失败。

`1014_v1_8gpu` 的 44k 相比同版本 40k，Object 提升 1.0 个百分点，
Spatial 下降 1.0 个百分点，Goal 持平；三套合计均为 553/600（92.17%）。
Spatial 任务 4（从上层抽屉取碗）从 9/20 降至 8/20（40.0%）。
Goal 任务 0（打开中层抽屉）从 12/20 降至 9/20（45.0%），
任务 3（打开上层抽屉并将碗放入）从 14/20 提高到 15/20（75.0%），
任务 9（将酒瓶放上酒架）从 14/20 降至 13/20（65.0%）。
这三个任务贡献 44k Goal 的全部 23 次失败，其余 7 个任务合计 140/140。

`1014_v1_8gpu` 的 48k 相比同版本 44k，Object 提升 0.5 个百分点，
Spatial 下降 1.0 个百分点，Goal 下降 1.5 个百分点；
三套合计从 553/600（92.17%）降至 549/600（91.50%）。
Spatial 任务 4（从上层抽屉取碗）从 8/20 提高到 9/20（45.0%）。
Goal 任务 0（打开中层抽屉）从 9/20 提高到 11/20（55.0%），
任务 3（打开上层抽屉并将碗放入）从 15/20 降至 13/20（65.0%），
任务 9（将酒瓶放上酒架）维持 13/20（65.0%）。
这三个任务贡献 48k Goal 的 23/26 次失败，
另外 3 次失败分别来自任务 1、4、5（各 19/20）。
本轮仍使用旧的逐 episode 重设种子协议，不能标作单次初始化 RNG 的新协议结果。

`1014_v0` 与 `1014_v1_8gpu` 的训练步数、输入视角和训练配置不同；
虽然使用相同评测协议，但这些结果不是仅增加 wrist 视角的受控消融实验。

## 单次初始化 RNG 协议结果

### 48k：与旧协议对照

本轮使用 `1014_v1_8gpu` 的同一份 48k 权重，BF16、`primary` + `wrist`，
每套 10 个任务、每任务前 20 个固定初态，其余模型配置与控制设置不变。
仅在评测进程启动时用 `args.seed=0` 初始化 Python、NumPy、Torch RNG；
任务、episode 和 replanning 之间不再重设种子，
结果标记为 `sampling_rng="run_seeded_continuous"`。
权重、模型配置快照和评测代码指纹在本轮完成后均保持不变。

| Suite | 旧协议 | 新协议 | 变化 |
| --- | --- | --- | --- |
| Object | 97.5%（195/200） | 98.0%（196/200） | +0.5 个百分点 |
| Spatial | 90.0%（180/200） | 88.0%（176/200） | -2.0 个百分点 |
| Goal | 87.0%（174/200） | 90.5%（181/200） | +3.5 个百分点 |
| 合计 | 91.50%（549/600） | 92.17%（553/600） | +0.67 个百分点 |

Goal 任务 0（打开中层抽屉）从旧协议的 11/20 提高到 16/20（80.0%），
任务 3（打开上层抽屉并将碗放入）从 13/20 提高到 14/20（70.0%），
任务 9（将酒瓶放上酒架）从 13/20 降至 12/20（60.0%）。
这三个任务贡献新协议 Goal 的 18/19 次失败，
另一次来自任务 7（开炉灶，19/20）。

这是单个全局种子的随机流协议对照，不是模型权重改进；
不能据此断言单次初始化 RNG 在其他种子上也一定提高成功率。
由于 RNG 连续推进，改变任务顺序、任务子集或前序 rollout 长度，
也会改变后续 episode 的采样随机状态。

### 54k：与同协议 48k 对照

54k 仅评测单次初始化 RNG 的新协议，没有运行旧协议。
同样使用 BF16、`primary` + `wrist`、每任务前 20 个固定初态，
预测 32 步、执行 10 步、Flow 20 步；每个 suite 的评测进程仅在启动时
用 `args.seed=0` 初始化一次 RNG。模型配置与 48k 相同，
权重、模型配置、运行时代码及归一化统计量的指纹在评测前后均保持不变。

| Suite | 48k（新协议） | 54k（新协议） | 变化 |
| --- | --- | --- | --- |
| Object | 98.0%（196/200） | 97.0%（194/200） | -1.0 个百分点 |
| Spatial | 88.0%（176/200） | 90.5%（181/200） | +2.5 个百分点 |
| Goal | 90.5%（181/200） | 91.5%（183/200） | +1.0 个百分点 |
| 合计 | 92.17%（553/600） | 93.00%（558/600） | +0.83 个百分点 |

54k 的全部任务成绩如下，各 suite 的任务编号独立，均为 0–9：

| 任务编号 | Object | Spatial | Goal |
| --- | --- | --- | --- |
| 0 | 90.0%（18/20） | 100.0%（20/20） | 80.0%（16/20） |
| 1 | 100.0%（20/20） | 100.0%（20/20） | 100.0%（20/20） |
| 2 | 100.0%（20/20） | 100.0%（20/20） | 100.0%（20/20） |
| 3 | 100.0%（20/20） | 100.0%（20/20） | 65.0%（13/20） |
| 4 | 100.0%（20/20） | 35.0%（7/20） | 100.0%（20/20） |
| 5 | 95.0%（19/20） | 95.0%（19/20） | 95.0%（19/20） |
| 6 | 100.0%（20/20） | 100.0%（20/20） | 100.0%（20/20） |
| 7 | 85.0%（17/20） | 100.0%（20/20） | 100.0%（20/20） |
| 8 | 100.0%（20/20） | 100.0%（20/20） | 100.0%（20/20） |
| 9 | 100.0%（20/20） | 75.0%（15/20） | 75.0%（15/20） |

Spatial 的主要瓶颈仍为任务 4（从上层抽屉取碗），
从 48k 的 8/20（40.0%）降至 7/20（35.0%），贡献 13/19 次失败；
任务 9（从木柜上取碗）为 15/20（75.0%），贡献另外 5 次失败，
剩余 1 次失败来自任务 5（从 ramekin 上取碗）。
因此整体 Spatial 成绩上升并不代表抽屉任务改善。

Goal 任务 0（打开中层抽屉）维持 16/20（80.0%）；
任务 3（打开上层抽屉并将碗放入）从 14/20 降至 13/20（65.0%），
任务 9（将酒瓶放上酒架）从 12/20 提高到 15/20（75.0%）。
这三个任务贡献 16/17 次失败，另一次来自任务 5（将盘子推到炉灶前方）。
Object 的 6 次失败来自任务 0（字母汤，18/20）、
任务 5（番茄酱，19/20）和任务 7（牛奶，17/20）。

54k 在此次固定初态、单个全局种子的评测中净增 5 次成功；
尚未进行多随机种子重复，不能据此断言提升具有统计显著性。

### 60k：单卡串行，与同协议 54k 对照

本轮在同一张物理 GPU 3 上按 Object、Spatial、Goal 串行完成，
没有同时运行多个评测进程。每个 suite 单独启动进程，
仅在启动时用 `args.seed=0` 初始化一次 RNG。
沿用 BF16、`primary` + `wrist`、每任务前 20 个固定初态、
预测 32 步、执行 10 步、Flow 20 步及各 suite 的控制步数上限；
模型配置与 54k 相同。权重、模型配置、运行时代码和归一化统计量的指纹
在评测前后均保持不变。

| Suite | 54k（新协议） | 60k（新协议） | 变化 |
| --- | --- | --- | --- |
| Object | 97.0%（194/200） | 98.5%（197/200） | +1.5 个百分点 |
| Spatial | 90.5%（181/200） | 91.0%（182/200） | +0.5 个百分点 |
| Goal | 91.5%（183/200） | 90.0%（180/200） | -1.5 个百分点 |
| 合计 | 93.00%（558/600） | 93.17%（559/600） | +0.17 个百分点 |

60k 的全部任务成绩如下，各 suite 的任务编号独立，均为 0–9：

| 任务编号 | Object | Spatial | Goal |
| --- | --- | --- | --- |
| 0 | 100.0%（20/20） | 100.0%（20/20） | 80.0%（16/20） |
| 1 | 100.0%（20/20） | 100.0%（20/20） | 100.0%（20/20） |
| 2 | 100.0%（20/20） | 100.0%（20/20） | 100.0%（20/20） |
| 3 | 100.0%（20/20） | 90.0%（18/20） | 45.0%（9/20） |
| 4 | 95.0%（19/20） | 45.0%（9/20） | 100.0%（20/20） |
| 5 | 95.0%（19/20） | 95.0%（19/20） | 95.0%（19/20） |
| 6 | 100.0%（20/20） | 90.0%（18/20） | 100.0%（20/20） |
| 7 | 95.0%（19/20） | 95.0%（19/20） | 100.0%（20/20） |
| 8 | 100.0%（20/20） | 100.0%（20/20） | 100.0%（20/20） |
| 9 | 100.0%（20/20） | 95.0%（19/20） | 80.0%（16/20） |

Spatial 任务 4（从上层抽屉取碗）从 7/20（35.0%）提高到 9/20（45.0%），
但仍贡献 11/18 次失败，是本套主要瓶颈。
任务 9（从木柜上取碗）从 15/20 提高到 19/20（95.0%）；
任务 3、6、7 分别从 20/20 回落到 18/20、18/20、19/20，
抵消了大部分改善。

Goal 任务 0（打开中层抽屉）维持 16/20（80.0%），
任务 3（打开上层抽屉并将碗放入）从 13/20（65.0%）降至 9/20（45.0%），
任务 9（将酒瓶放上酒架）从 15/20 提高到 16/20（80.0%）。
这三个任务贡献 19/20 次失败，另一次来自任务 5（将盘子推到炉灶前方）。
Object 的 3 次失败分别来自任务 4（ketchup）、5（tomato sauce）、7（牛奶），
各为 19/20，其余任务均为 20/20。

60k 在此次评测中仅比 54k 净增 1 次成功，总体基本持平；
尚未进行多随机种子重复，不能据此断言稳定优于 54k。

### 64k：双卡排队，与同协议 60k 对照

Object 在物理 GPU 2、Spatial 在物理 GPU 3 并行评测；
Spatial 完成后，Goal 复用 GPU 3，没有占用其他 GPU。
每套仍使用独立进程，仅在启动时用 `args.seed=0` 初始化一次 RNG。
沿用 BF16、`primary` + `wrist`、每任务前 20 个固定初态、
预测 32 步、执行 10 步、Flow 20 步及各 suite 的控制步数上限。
模型配置与 60k 相同，权重、模型配置、运行时代码及归一化统计量的指纹
在评测前后均保持不变。

| Suite | 60k（新协议） | 64k（新协议） | 变化 |
| --- | --- | --- | --- |
| Object | 98.5%（197/200） | 99.0%（198/200） | +0.5 个百分点 |
| Spatial | 91.0%（182/200） | 88.5%（177/200） | -2.5 个百分点 |
| Goal | 90.0%（180/200） | 90.5%（181/200） | +0.5 个百分点 |
| 合计 | 93.17%（559/600） | 92.67%（556/600） | -0.50 个百分点 |

64k 的全部任务成绩如下，各 suite 的任务编号独立，均为 0–9：

| 任务编号 | Object | Spatial | Goal |
| --- | --- | --- | --- |
| 0 | 100.0%（20/20） | 100.0%（20/20） | 80.0%（16/20） |
| 1 | 100.0%（20/20） | 100.0%（20/20） | 95.0%（19/20） |
| 2 | 100.0%（20/20） | 100.0%（20/20） | 100.0%（20/20） |
| 3 | 100.0%（20/20） | 100.0%（20/20） | 60.0%（12/20） |
| 4 | 100.0%（20/20） | 35.0%（7/20） | 95.0%（19/20） |
| 5 | 95.0%（19/20） | 80.0%（16/20） | 100.0%（20/20） |
| 6 | 100.0%（20/20） | 90.0%（18/20） | 95.0%（19/20） |
| 7 | 95.0%（19/20） | 95.0%（19/20） | 100.0%（20/20） |
| 8 | 100.0%（20/20） | 100.0%（20/20） | 100.0%（20/20） |
| 9 | 100.0%（20/20） | 85.0%（17/20） | 80.0%（16/20） |

Spatial 任务 4（从上层抽屉取碗）从 9/20（45.0%）降至 7/20（35.0%），
贡献 13/23 次失败，仍是本套主要瓶颈。
任务 5（从 ramekin 上取碗）从 19/20 降至 16/20（80.0%），
任务 9（从木柜上取碗）从 19/20 降至 17/20（85.0%）；
任务 3（从饼干盒上取碗）从 18/20 提高到 20/20。
这些变化合计使 Spatial 比 60k 少成功 5 次。

Goal 任务 0（打开中层抽屉）和任务 9（将酒瓶放上酒架）均维持 16/20（80.0%），
任务 3（打开上层抽屉并将碗放入）从 9/20 提高到 12/20（60.0%）。
这三个任务贡献 16/19 次失败，其余 3 次分别来自任务 1、4、6（各 19/20）。
Object 的 2 次失败分别来自任务 5（tomato sauce）和任务 7（牛奶），
各为 19/20，其余任务均为 20/20。

64k 在此次评测中比 60k 少成功 3 次；Object、Goal 的小幅上升未抵消 Spatial 的回落。
尚未进行多随机种子重复，不能仅据此认定模型出现稳定退化。

### 1023_v2_4nodes_8gpu：4k

本轮使用新训练版本 `1023_v2_4nodes_8gpu` 的 4k 权重，
实际 checkpoint 位于 `step4k/checkpoint_00004000`。
训练 metadata 记录 `world_size=32`、`update_step=4000`，
每卡 batch size 为 32、梯度累积为 1，对应 global batch size 1024。
32 个模型分片在单个评测进程中恢复，不需要 32 张评测 GPU。

Object 在物理 GPU 2、Spatial 在物理 GPU 3 并行评测；
Spatial 完成后，Goal 复用 GPU 3。
沿用 BF16、`primary` + `wrist`、每任务前 20 个固定初态、
预测 32 步、执行 10 步、Flow 20 步及各 suite 的控制步数上限。
每套独立进程仅在启动时设置一次 `seed=0`，
不在任务、episode 或 replanning 内重设种子。
全部 600 次 rollout 的任务/初态覆盖、动作数组、失败视频及统计一致性均已核验；
权重、配置、运行时代码、归一化统计量及训练 metadata 的指纹在评测前后保持不变。

使用本版本自己保存的配置，不替换为 v1 配置。
相较 v1 64k，`understanding_lr_scale` 从 0.2 改为 0.01，
`action_prediction_reconstruction_loss_weight` 从 5.0 改为 10.0，
`task_weights` 从 `[0.15, 0.15, 0.7]` 改为 `[0.1, 0.1, 0.8]`。
训练版本、步数及训练规模均不同，以下结果不能视为仅更换 checkpoint 的受控对照。

| Suite | v2 4k（新协议） |
| --- | --- |
| Object | 88.0%（176/200） |
| Spatial | 76.5%（153/200） |
| Goal | 67.0%（134/200） |
| 合计 | 77.17%（463/600） |

全部任务成绩如下，各 suite 的任务编号独立，均为 0–9：

| 任务编号 | Object | Spatial | Goal |
| --- | --- | --- | --- |
| 0 | 85.0%（17/20） | 100.0%（20/20） | 5.0%（1/20） |
| 1 | 100.0%（20/20） | 70.0%（14/20） | 100.0%（20/20） |
| 2 | 100.0%（20/20） | 100.0%（20/20） | 95.0%（19/20） |
| 3 | 95.0%（19/20） | 100.0%（20/20） | 5.0%（1/20） |
| 4 | 85.0%（17/20） | 5.0%（1/20） | 95.0%（19/20） |
| 5 | 90.0%（18/20） | 90.0%（18/20） | 20.0%（4/20） |
| 6 | 100.0%（20/20） | 90.0%（18/20） | 80.0%（16/20） |
| 7 | 55.0%（11/20） | 85.0%（17/20） | 100.0%（20/20） |
| 8 | 75.0%（15/20） | 95.0%（19/20） | 100.0%（20/20） |
| 9 | 95.0%（19/20） | 30.0%（6/20） | 70.0%（14/20） |

Spatial 的主要失败集中在任务 4（从上层抽屉取碗，1/20）、
任务 9（从木柜上取碗，6/20）和任务 1（取 ramekin 旁的碗，14/20），
合计贡献 39/47 次失败。
Goal 的任务 0（打开中层抽屉，1/20）、任务 3（打开上层抽屉并将碗放入，1/20）
和任务 5（将盘子推到炉灶前方，4/20）合计贡献 54/66 次失败。
Object 的任务 7（牛奶，11/20）和任务 8（chocolate pudding，15/20）
合计贡献 14/24 次失败。

作为跨训练版本的参考，v1 64k 为 556/600（92.67%），
本轮 v2 4k 少成功 93 次、低 15.50 个百分点。
这不是相同步数或相同训练配置下的受控比较，也没有多随机种子重复。

### 1023_v2_4nodes_8gpu：8k，与同协议 4k 对照

本轮在 4k 的三套评测全部完成后自动启动，使用上传目录 `step8k/model`。
上传内容只有模型：8 个 DCP 分片的 tensor 名称、形状及 dtype
与 4k 一致，均可完整恢复；使用的父目录策略配置也与 4k 完全相同。
但该目录没有 checkpoint 训练 metadata，
因此“8k”按用户提供的目录标注，实际 `update_step` 和训练 `world_size`
未独立核验，不能用 8 个存储分片推断训练使用 8 张 GPU。

Object 在物理 GPU 2、Spatial 在物理 GPU 3 并行评测；
Object 完成后，Goal 复用 GPU 2。
沿用与 4k 相同的 BF16、双视角、固定初态、动作预测/执行长度、
Flow 积分步数和控制步数上限，每套进程只在开头设置一次 `seed=0`。
全部 600 次 rollout 的覆盖、动作、失败视频、统计及实际评测设置均已核验；
权重、策略配置、运行时代码、归一化统计量及父目录训练配置的指纹保持不变。

| Suite | v2 4k（新协议） | v2 8k（新协议） | 变化 |
| --- | --- | --- | --- |
| Object | 88.0%（176/200） | 76.5%（153/200） | -11.5 个百分点 |
| Spatial | 76.5%（153/200） | 59.5%（119/200） | -17.0 个百分点 |
| Goal | 67.0%（134/200） | 51.0%（102/200） | -16.0 个百分点 |
| 合计 | 77.17%（463/600） | 62.33%（374/600） | -14.83 个百分点 |

8k 的全部任务成绩如下，各 suite 的任务编号独立，均为 0–9：

| 任务编号 | Object | Spatial | Goal |
| --- | --- | --- | --- |
| 0 | 60.0%（12/20） | 75.0%（15/20） | 0.0%（0/20） |
| 1 | 95.0%（19/20） | 45.0%（9/20） | 85.0%（17/20） |
| 2 | 85.0%（17/20） | 90.0%（18/20） | 85.0%（17/20） |
| 3 | 95.0%（19/20） | 95.0%（19/20） | 0.0%（0/20） |
| 4 | 85.0%（17/20） | 5.0%（1/20） | 30.0%（6/20） |
| 5 | 40.0%（8/20） | 55.0%（11/20） | 50.0%（10/20） |
| 6 | 90.0%（18/20） | 85.0%（17/20） | 85.0%（17/20） |
| 7 | 25.0%（5/20） | 45.0%（9/20） | 95.0%（19/20） |
| 8 | 95.0%（19/20） | 80.0%（16/20） | 60.0%（12/20） |
| 9 | 95.0%（19/20） | 20.0%（4/20） | 20.0%（4/20） |

Object 的主要失败集中在任务 7（牛奶，5/20）、任务 5（tomato sauce，8/20）
和任务 0（alphabet soup，12/20），合计贡献 35/47 次失败。
Spatial 的任务 4（从上层抽屉取碗）仍为 1/20，
任务 9（从木柜上取碗）降至 4/20；此外，
任务 5（从 ramekin 上取碗）从 18/20 降至 11/20，
任务 7（从炉灶上取碗）从 17/20 降至 9/20。
Goal 的两个抽屉任务 0、3 均为 0/20；
任务 4（将碗放到木柜上）、9（将酒瓶放上酒架）、8（将碗放到盘子上）
分别从 19/20、14/20、20/20 降至 6/20、4/20、12/20。
这三个任务合计少成功 31 次，贡献 Goal 净下降 32 次中的大部分。

本轮三套均低于 4k，总计少成功 89 次。
评测协议和实际加载的策略配置相同，但缺少 8k 的训练 metadata，
也没有多随机种子重复，不能仅凭本次结果确定训练退化的具体原因。

## 权重位置

| 训练版本 | 评测时使用的 checkpoint 路径 |
| --- | --- |
| `1014_v0` | `/Data/lzl/qwen3vl_mot_weights/ft_on_libero/1014_v0/step{18,24,32,40,48}k/model` |
| `1014_v1_8gpu` | `/Data/lzl/qwen3vl_mot_weights/ft_on_libero/1014_v1_8gpu/step{10,16,20,24,28,32,36,40,44,48,54,60,64}k/model` |
| `1023_v2_4nodes_8gpu` | `/Data/lzl/qwen3vl_mot_weights/ft_on_libero/1023_v2_4nodes_8gpu/step4k/checkpoint_00004000/model` |
| `1023_v2_4nodes_8gpu` | `/Data/lzl/qwen3vl_mot_weights/ft_on_libero/1023_v2_4nodes_8gpu/step8k/model` |

以上为评测时使用的位置；历史权重目录后续可能被归档或移除，结果以保存的
`results.json` 为准。

## 原始结果文件

以下路径均相对于仓库根目录。

16k Goal 首次尝试在模型加载阶段因显存不足退出，未开始 rollout；
下表记录的是随后重试完成的 200 次评测，未合并失败尝试。

### 旧协议

| 训练版本 | Checkpoint | Suite | `results.json` |
| --- | --- | --- | --- |
| `1014_v0` | 18k | Object | `outputs/evaluation/libero/step18k_object_20trials_20261015/results.json` |
| `1014_v0` | 18k | Goal | `outputs/evaluation/libero/step18k_goal_20trials_20261015/results.json` |
| `1014_v0` | 24k | Object | `outputs/evaluation/libero/step24k_object_20trials_20261015/results.json` |
| `1014_v0` | 24k | Spatial | `outputs/evaluation/libero/step24k_spatial_20trials_20261015/results.json` |
| `1014_v0` | 24k | Goal | `outputs/evaluation/libero/step24k_goal_20trials_20261015/results.json` |
| `1014_v0` | 32k | Object | `outputs/evaluation/libero/step32k_object_20trials_20261016/results.json` |
| `1014_v0` | 32k | Spatial | `outputs/evaluation/libero/step32k_spatial_20trials_20261016/results.json` |
| `1014_v0` | 40k | Object | `outputs/evaluation/libero/step40k_object_20trials_20261016/results.json` |
| `1014_v0` | 40k | Spatial | `outputs/evaluation/libero/step40k_spatial_20trials_20261016/results.json` |
| `1014_v0` | 40k | Goal | `outputs/evaluation/libero/step40k_goal_20trials_20261016/results.json` |
| `1014_v0` | 48k | Object | `outputs/evaluation/libero/step48k_object_20trials_20261017/results.json` |
| `1014_v0` | 48k | Spatial | `outputs/evaluation/libero/step48k_spatial_20trials_20261017/results.json` |
| `1014_v1_8gpu` | 10k | Object | `outputs/evaluation/libero/v1_8gpu_step10k_object_20trials_20261017/results.json` |
| `1014_v1_8gpu` | 10k | Spatial | `outputs/evaluation/libero/v1_8gpu_step10k_spatial_20trials_20261017/results.json` |
| `1014_v1_8gpu` | 10k | Goal | `outputs/evaluation/libero/v1_8gpu_step10k_goal_20trials_20261017/results.json` |
| `1014_v1_8gpu` | 16k | Object | `outputs/evaluation/libero/v1_8gpu_step16k_object_20trials_20261018/results.json` |
| `1014_v1_8gpu` | 16k | Spatial | `outputs/evaluation/libero/v1_8gpu_step16k_spatial_20trials_20261018/results.json` |
| `1014_v1_8gpu` | 16k | Goal | `outputs/evaluation/libero/v1_8gpu_step16k_goal_20trials_20261018_retry1/results.json` |
| `1014_v1_8gpu` | 20k | Object | `outputs/evaluation/libero/v1_8gpu_step20k_object_20trials_20261018/results.json` |
| `1014_v1_8gpu` | 20k | Spatial | `outputs/evaluation/libero/v1_8gpu_step20k_spatial_20trials_20261018/results.json` |
| `1014_v1_8gpu` | 20k | Goal | `outputs/evaluation/libero/v1_8gpu_step20k_goal_20trials_20261018/results.json` |
| `1014_v1_8gpu` | 24k | Object | `outputs/evaluation/libero/v1_8gpu_step24k_object_20trials_20261018/results.json` |
| `1014_v1_8gpu` | 24k | Spatial | `outputs/evaluation/libero/v1_8gpu_step24k_spatial_20trials_20261018/results.json` |
| `1014_v1_8gpu` | 24k | Goal | `outputs/evaluation/libero/v1_8gpu_step24k_goal_20trials_20261018/results.json` |
| `1014_v1_8gpu` | 28k | Object | `outputs/evaluation/libero/v1_8gpu_step28k_object_20trials_20261019/results.json` |
| `1014_v1_8gpu` | 28k | Spatial | `outputs/evaluation/libero/v1_8gpu_step28k_spatial_20trials_20261019/results.json` |
| `1014_v1_8gpu` | 28k | Goal | `outputs/evaluation/libero/v1_8gpu_step28k_goal_20trials_20261019/results.json` |
| `1014_v1_8gpu` | 32k | Object | `outputs/evaluation/libero/v1_8gpu_step32k_object_20trials_20261019/results.json` |
| `1014_v1_8gpu` | 32k | Spatial | `outputs/evaluation/libero/v1_8gpu_step32k_spatial_20trials_20261019/results.json` |
| `1014_v1_8gpu` | 32k | Goal | `outputs/evaluation/libero/v1_8gpu_step32k_goal_20trials_20261019/results.json` |
| `1014_v1_8gpu` | 36k | Object | `outputs/evaluation/libero/v1_8gpu_step36k_object_20trials_20261019/results.json` |
| `1014_v1_8gpu` | 36k | Spatial | `outputs/evaluation/libero/v1_8gpu_step36k_spatial_20trials_20261019/results.json` |
| `1014_v1_8gpu` | 36k | Goal | `outputs/evaluation/libero/v1_8gpu_step36k_goal_20trials_20261019/results.json` |
| `1014_v1_8gpu` | 40k | Object | `outputs/evaluation/libero/v1_8gpu_step40k_object_20trials_20261020/results.json` |
| `1014_v1_8gpu` | 40k | Spatial | `outputs/evaluation/libero/v1_8gpu_step40k_spatial_20trials_20261020/results.json` |
| `1014_v1_8gpu` | 40k | Goal | `outputs/evaluation/libero/v1_8gpu_step40k_goal_20trials_20261020/results.json` |
| `1014_v1_8gpu` | 44k | Object | `outputs/evaluation/libero/v1_8gpu_step44k_object_20trials_20261020/results.json` |
| `1014_v1_8gpu` | 44k | Spatial | `outputs/evaluation/libero/v1_8gpu_step44k_spatial_20trials_20261020/results.json` |
| `1014_v1_8gpu` | 44k | Goal | `outputs/evaluation/libero/v1_8gpu_step44k_goal_20trials_20261020/results.json` |
| `1014_v1_8gpu` | 48k | Object | `outputs/evaluation/libero/v1_8gpu_step48k_object_20trials_20261020/results.json` |
| `1014_v1_8gpu` | 48k | Spatial | `outputs/evaluation/libero/v1_8gpu_step48k_spatial_20trials_20261020/results.json` |
| `1014_v1_8gpu` | 48k | Goal | `outputs/evaluation/libero/v1_8gpu_step48k_goal_20trials_20261020/results.json` |

### 单次初始化 RNG 协议

| 训练版本 | Checkpoint | Suite | `results.json` |
| --- | --- | --- | --- |
| `1014_v1_8gpu` | 48k | Object | `outputs/evaluation/libero/v1_8gpu_step48k_run_seeded_object_20trials_20261020/results.json` |
| `1014_v1_8gpu` | 48k | Spatial | `outputs/evaluation/libero/v1_8gpu_step48k_run_seeded_spatial_20trials_20261020/results.json` |
| `1014_v1_8gpu` | 48k | Goal | `outputs/evaluation/libero/v1_8gpu_step48k_run_seeded_goal_20trials_20261020/results.json` |
| `1014_v1_8gpu` | 54k | Object | `outputs/evaluation/libero/v1_8gpu_step54k_run_seeded_object_20trials_20261022/results.json` |
| `1014_v1_8gpu` | 54k | Spatial | `outputs/evaluation/libero/v1_8gpu_step54k_run_seeded_spatial_20trials_20261022/results.json` |
| `1014_v1_8gpu` | 54k | Goal | `outputs/evaluation/libero/v1_8gpu_step54k_run_seeded_goal_20trials_20261022/results.json` |
| `1014_v1_8gpu` | 60k | Object | `outputs/evaluation/libero/v1_8gpu_step60k_run_seeded_object_20trials_20261022/results.json` |
| `1014_v1_8gpu` | 60k | Spatial | `outputs/evaluation/libero/v1_8gpu_step60k_run_seeded_spatial_20trials_20261022/results.json` |
| `1014_v1_8gpu` | 60k | Goal | `outputs/evaluation/libero/v1_8gpu_step60k_run_seeded_goal_20trials_20261022/results.json` |
| `1014_v1_8gpu` | 64k | Object | `outputs/evaluation/libero/v1_8gpu_step64k_run_seeded_object_20trials_20261023/results.json` |
| `1014_v1_8gpu` | 64k | Spatial | `outputs/evaluation/libero/v1_8gpu_step64k_run_seeded_spatial_20trials_20261023/results.json` |
| `1014_v1_8gpu` | 64k | Goal | `outputs/evaluation/libero/v1_8gpu_step64k_run_seeded_goal_20trials_20261023/results.json` |
| `1023_v2_4nodes_8gpu` | 4k | Object | `outputs/evaluation/libero/v2_4nodes_8gpu_step4k_run_seeded_object_20trials_20261023/results.json` |
| `1023_v2_4nodes_8gpu` | 4k | Spatial | `outputs/evaluation/libero/v2_4nodes_8gpu_step4k_run_seeded_spatial_20trials_20261023/results.json` |
| `1023_v2_4nodes_8gpu` | 4k | Goal | `outputs/evaluation/libero/v2_4nodes_8gpu_step4k_run_seeded_goal_20trials_20261023/results.json` |
| `1023_v2_4nodes_8gpu` | 8k | Object | `outputs/evaluation/libero/v2_4nodes_8gpu_step8k_run_seeded_object_20trials_20261023/results.json` |
| `1023_v2_4nodes_8gpu` | 8k | Spatial | `outputs/evaluation/libero/v2_4nodes_8gpu_step8k_run_seeded_spatial_20trials_20261023/results.json` |
| `1023_v2_4nodes_8gpu` | 8k | Goal | `outputs/evaluation/libero/v2_4nodes_8gpu_step8k_run_seeded_goal_20trials_20261023/results.json` |

各结果目录同时保留 `evaluation.log`、逐次 rollout 的 `*_actions.npy`，
以及失败 rollout 的 `*_failure.mp4`。任务级成功次数和成功率见
`results.json` 中的 `tasks` 字段。
