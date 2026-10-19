# LIBERO 评测记录

更新日期：2026-10-19。

本文汇总已完成的正式评测，共 30 份结果、6,000 次 rollout。数据来自对应的
`results.json`，不包含 smoke test、首次成功即停止的检查或尚未完成的评测。

## 评测设置

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

## 结果汇总

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

`1014_v0` 与 `1014_v1_8gpu` 的训练步数、输入视角和训练配置不同；
虽然使用相同评测协议，但这些结果不是仅增加 wrist 视角的受控消融实验。

## 权重位置

| 训练版本 | 评测时使用的 checkpoint 路径 |
| --- | --- |
| `1014_v0` | `/Data/lzl/qwen3vl_mot_weights/ft_on_libero/1014_v0/step{18,24,32,40,48}k/model` |
| `1014_v1_8gpu` | `/Data/lzl/qwen3vl_mot_weights/ft_on_libero/1014_v1_8gpu/step{10,16,20,24,28,32}k/model` |

以上为评测时使用的位置；历史权重目录后续可能被归档或移除，结果以保存的
`results.json` 为准。

## 原始结果文件

以下路径均相对于仓库根目录。

16k Goal 首次尝试在模型加载阶段因显存不足退出，未开始 rollout；
下表记录的是随后重试完成的 200 次评测，未合并失败尝试。

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

各结果目录同时保留 `evaluation.log`、逐次 rollout 的 `*_actions.npy`，
以及失败 rollout 的 `*_failure.mp4`。任务级成功次数和成功率见
`results.json` 中的 `tasks` 字段。
