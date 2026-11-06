# LIBERO 微调审计：对照 Cosmos3 与 LaWAM

审计日期：2026-11-06。仓库代码基线：`333814e`。

本文对照 [Cosmos3 官方文档][cosmos-doc]、[LaWAM 官方文档][lawam-doc]、
固定版本的上游实现、实际保存的 v4/v5/v6 配置和更正上传后的 checkpoint。
**只记录问题、证据与实验建议，不修改训练行为，也不启动训练。**
模型结构见 [Stage 2 说明](stage2_model.md)，实际成绩与评测身份见
[LIBERO 评测记录](evaluation/libero.md)。

## 1. 结论与证据等级

目前确实有值得优先解决的问题，但不能将成功率差距直接归因于 Stage 1 或某个 loss。
最重要的不是继续同时改多个超参数，而是先确认优化器确实在有效更新参数，
再建立可信的原生 baseline 和单因素消融。

| 优先级 | 发现 | 已确认的事实 | 尚未确认的部分 |
|---|---|---|---|
| P0 | BF16 主参数叠加 v5 极小 Understanding LR | 当前入口直接将参数转 BF16；v5 对 Qwen 模型参数的峰值 LR 为 `1e-6`；实测多个可训练矩阵在 16k→18k 几乎不变 | 有多少更新被舍入吞掉、修正 precision 能提升多少闭环成功率 |
| P0 | 当前 baseline 不是官方原生 Cosmos3 的严格复现 | 我们替换了 Understanding、连接方式、物理表示、视角布局和多项训练 recipe；无 Stage 1 MLP baseline 仍是自定义 MoT | 性能差距分别来自这些变化中的哪一项 |
| P1 | latent-action 的使用方式与 LaWAM 不同 | 我们的投影后 change query 只显式参与辅助 MSE；LaWAM 用预测 latent 解码未来视觉特征，并将其作为 action flow 条件 | 增加对应路径是否适合当前 MoT、是否提升控制 |
| P1 | episode 尾部监督与官方不同 | 我们保留尾部 anchor，约 `19.19%` 窗口含 padding；官方 Cosmos3 将含 padding 的整个样本排除出 loss | 人工静止动作/重复图像监督对终止能力和早停失败的净影响 |
| P1 | 实验与选择协议不足以支持某些归因 | v6 同时改了 task 和 shift；训练内生成验证不是闭环验证；困难子集不是完整 benchmark | 多 seed、全 suite 下优势是否稳定 |
| P2 | codec、归一化、horizon、solver、增强和 EMA 均未对齐 | 存在明确实现/recipe 差异，详见下文 | 每一项的独立贡献，不能先当作已证性能根因 |

这里“确认差异”不等于“确认 bug”，“参数几乎不变”也不等于“计算图断开”。
当前代码可以正常训练、预测和完成闭环评测；本次发现主要是优化有效性风险、
基线定义问题，以及尚未完成验证的建模取舍。

## 2. 官方结果能说明什么

两份官方文档报告的四 suite 结果均为 **40 个任务 × 每任务 50 trials**：

| 方法 | Spatial | Object | Goal | LIBERO-10 | 总计 |
|---|---:|---:|---:|---:|---:|
| Cosmos3 Nano，四 suite 微调 | 490/500 | 497/500 | 487/500 | 484/500 | 1958/2000，97.90% |
| Cosmos3 Edge，四 suite 微调 | 493/500 | 497/500 | 486/500 | 483/500 | 1959/2000，97.95% |
| LaWAM | 492/500 | 498/500 | 490/500 | 490/500 | 1970/2000，98.50% |

这说明两条路线都能在其发布权重和评测协议下取得很好的 LIBERO 成绩，
但不能用这些全任务结果减去我们的困难子集成功率，作为公平的性能差距。
文档中的 quick-start `n_episodes=10` 也不是上述表格的统计分母。

我们最近的有效更正上传结果只覆盖 Goal `[0,3,4,9]`、
Spatial `[4,6,9]`，每任务 20 trials、单次 suite seed：

| checkpoint | Goal | Spatial | 合计 |
|---|---:|---:|---:|
| v4 shift2 16k | 67/80 | 41/60 | 108/140 |
| v5 shift2 16k，更正上传 | 64/80 | 38/60 | 102/140 |
| v4 shift2 18k | 75/80 | 38/60 | 113/140 |
| v5 shift2 18k，更正上传 | 68/80 | 39/60 | 107/140 |

v5 在两个同 step 对照中都少 6 次成功，但尚不能将其全部归因于低 LR。
旧的、与 v4 权重重复的“v5 16k”结果已失效，不参与本次分析。
固定相同初态和 suite seed 也不自动保证不同策略在每个 trial 使用成对相同的 diffusion noise。

LaWAM 特别要求复现环境使用 **MuJoCo 3.3.2**，并记录了 3.8.1 的明显退化。
我们的历史环境是 **3.2.3**，不是该警告中的 3.8.1：
只能说环境未完全对齐，不能直接断言现有失败就是该已知版本问题。
应在隔离环境中用发布策略做 sanity check，不修改现有共享环境破坏历史可比性。

## 3. 模型和 recipe 对照

以下 Cosmos3 栏结合官方 LIBERO recipe 和发布策略配置；
LaWAM 栏结合其发布 LIBERO 策略配置，而非只看 Python class 默认值。

| 项目 | 官方 Cosmos3 | 官方 LaWAM | 当前 Stage 1 路线 | 无 Stage 1 MLP baseline |
|---|---|---|---|---|
| 初始化 | 完整原生 Understanding/Generation MoT 联合预训练权重 | 已联合预训练的 VLM、LAM 与 action policy | Qwen3-VL-4B + Cosmos3 Generation + Stage 1 两侧模块 | 原始 Qwen3-VL-4B + Cosmos3 Generation；无 Stage 1 |
| 跨分支连接 | 保留原生预训练的注意力和位置契约 | VLM features、当前/预测未来视觉特征条件化 flow head | 选取 Qwen 层的原生 K/V，经可训练 adapter 接 Generation | 同一个自定义 K/V→Generation 框架 |
| 动作目标 | 归一化 raw action；10D 有效值，pad 至 64D | 归一化 raw action；7D 有效值，pad 至 32D | 冻结 Physical Transformer 的 action embedding | 归一化 raw canonical action，独立可训练 MLP 编码 |
| latent-action 作用 | 不依赖我们的 Stage 1 change teacher | 蒸馏 LAM latent，并解码未来视觉特征供 action flow 使用 | 预测 change query 的辅助 MSE；与物理 embedding 是两种 latent | 不加载 change teacher，不使用该蒸馏 loss |
| state | LIBERO recipe `use_state=false` | 发布策略 `flow_use_state=false` | joint 预测未来 state，另有干净当前 state 条件；VA 不预测未来 state | 由所选对照 task 决定，不应擅自切换 |
| 多视角 | 两相机左右拼成 canvas，再编码 | primary/wrist 输入，primary 用于 LAM 视觉特征 | 每相机独立 VAE 编码、camera embedding、联合注意力 | 同当前自定义实现 |
| action normalization | q01/q99 quantile | min-max，另有 gripper snap/binarize | 每个数据源各自 mean/std | 当前也是每源 mean/std |
| 时间窗口 | 16 actions，17 observation frames | 有效 horizon=8、执行 8；`chunk_size=50` 是 padded flow 序列 | 32 actions，稀疏 9 帧覆盖 `[t,t+31]`；最近评测执行 10 步重规划 | 保持所选受控对照窗口 |
| 推理 | UniPC 30，sampling shift=5，无 CFG | flow 10 步 | Euler 20，均匀 sigma 网格，无 CFG | 同当前自定义 sampler |
| 训练精度 | 默认 FP32 主参数，BF16 compute/reduce | backend 对 VLM/LAM 使用 BF16 autocast，对 flow/loss 使用 FP32 autocast | 当前入口 BF16 主参数，BF16 autocast/reduce | 同当前训练入口 |
| RGB augmentation | 不能仅由 LaWAM 或我们的设置推断其配置 | 官方明确不使用随机图像增强 | 当前开启时为 90%–100% crop + mild jitter，无旋转 | 默认继承当前数据管线，需显式记录开关 |
| optimizer / 稳定化 | AdamW；原生 recipe 使用 Power EMA，评估 EMA 权重 | 发布 config 有自身 optimizer/loss 设置，不能直接套 Cosmos3 recipe | AdamW8bit；当前训练入口未实现 EMA | 同当前训练入口 |

因此，应将当前实现称为 **“Cosmos3 Generation 初始化的自定义 Qwen3-VL MoT”**，
而不是已经严格复现官方 Cosmos3。
MLP baseline 能回答我们内部方案的工程对照问题，但不能代替原生 Cosmos3 baseline。
官方完整 Nano 的总参数量、默认可训练参数量，与我们 DCP 中包含冻结 teacher 等模块的
tensor 元素总数也不是同一个统计口径。

官方的另一个复现陷阱是：通过 `policy.type + pretrained_path` 使用 LIBERO class recipe，
和通过 `policy.path` 直接继承 base checkpoint 的 DROID 配置，含义不同。
这不表示我们已触发该错误；只是后续运行原生 baseline 时必须检查有效配置。
新建官方 action 模块与恢复已训练 action 模块的 `load_action_modules` 设置也不同，
不能将官方 flag 生搬硬套到我们的 DCP 恢复流程。

## 4. 实际 v4/v5/v6 配置与训练预算

配置来自以下根目录的 `config.json`、`train_config.json`，不是当前 launcher 的默认值：

```text
/Data/lzl/qwen3vl_mot_weights/ft_on_libero/
  1031v4_4nodes_8gpu_wo_rotate_high_noise_shift2/
  1104v5_4nodes_8gpu_wo_rotate_und_lr_scale_1e_2_shift_2/
  1104v6_4nodes_8gpu_wo_rotate_shift_5_va_pred/
```

| 配置项 | v4 | v5 | v6 |
|---|---|---|---|
| task | `joint_prediction` | `joint_prediction` | `va_prediction` |
| task weight | 1 | 1 | 1 |
| `time_sampling_distribution` | `shifted_uniform` | `shifted_uniform` | `shifted_uniform` |
| `time_sampling_shift` | 2 | 2 | 5 |
| `understanding_lr_scale` | 1 | 0.01 | 1 |
| Qwen 模型参数峰值 LR | `1e-4` | `1e-6` | `1e-4` |

共同设置：Nano Generation、primary/wrist 双视角、`chunk_size=32`、
`group_size=4`、`world_video_frames=9`、`physical_flow_space=embedding`、
`physical_tuning_mode=frozen`；旧配置未保存 `physical_encoder_type`，
对应原 Stage 1 Transformer 路径，不是新 MLP baseline。
三组均为 Understanding `full`、vision 最后 4 块 LoRA；
这里的 `full` 不代表 token embedding、最终 norm 和整个视觉塔全部解冻。

共同训练值如下：

| 项目 | 保存值 / 有效含义 |
|---|---|
| 每 rank batch / accumulation | 16 / 1 |
| optimizer | AdamW8bit，LR `1e-4`，betas `(0.9,0.95)`，weight decay `0.01`，grad clip `10` |
| scheduler | warmup 1000 updates，plateau 100，再 cosine decay 20000 updates；主组最低 LR `1.5e-6` |
| flow losses | video/action/state 权重各 1；实际是否产生该项由 task 决定 |
| predicted-clean reconstruction | action 权重 10，state 权重 5；VA 没有 future-state prediction loss |
| change-query 蒸馏 | 权重 1 |
| 数据 | `stage=finetune`、`data_mix=libero`、sample pool 禁用、全帧 anchor 覆盖 |
| 生成验证 | 训练内样本，每数据源 8 episodes，频率 2000 updates；不是环境 rollout |
| FP8 | 关闭；仍显式使用 BF16 autocast，不能由保存的 `use_amp=false` 推断没有混合精度 |
| 最大 steps | 配置上限 600000；真实停止还受 epoch 计划限制，并非证明已经训练了 600000 updates |

`get_optim_params()` 只对名称以 `understanding.model.` 开头的参数应用 Understanding scale。
**`understanding.query_tokens`、latent projection、Generation、physical decoders 属于主组，
不会跟着 v5 一起缩小 100 倍。**
`physical_lr_scale=0.3` 只作用于 physical encoder；encoder 冻结时，它不表示
可训练 physical decoder 的 LR 也是主组的 0.3 倍。

我们的 `step16k` 指 **16000 次 optimizer updates**；
官方 Cosmos3 LeRobot 示例的 `steps` 按 **microbatches** 计数，
文档中 8 次 accumulation 的 16000 microbatches 对应 2000 updates。
官方给出的原生 LIBERO-10/四 suite 预算分别为约 2000/5000 updates，
global batch 2048，最佳 checkpoint 约在 1500/4500 updates。
不能直接比较目录中的 “16k” 和官方 “16k”。

本次未从原始 trainer 日志独立确证历史 `world_size`，不能用目录名或 DCP 文件数代替证据。
我们的 global batch 应写成 `16 × world_size × 1`。
**若**确为 32 ranks，则 18k updates 看过 `18,000×512=9,216,000` 个 anchor，
恰好等于官方 `4500×2048` 的样本数；这不是“训练步数更大，所以一定已经训练得更多”。
比较时还应报告有效非 padding targets、horizon、重复加噪次数、GPU 时间和 samples/s。

## 5. P0：BF16 主参数与微小 LR 的数值风险

当前 [FSDP 入口](../lerobot/scripts/fsdp_train_contrast.py#L313-L355)
在构建 optimizer 前调用 `policy.to(..., dtype=torch.bfloat16)`；
[FSDP precision](../lerobot/common/utils/fsdp_training.py#L348-L356)
也配置 BF16 param/reduce/buffer。
[AdamW8bit 包装](../lerobot/common/optim/optimizers.py#L87-L130)
直接创建 bitsandbytes optimizer，本仓库没有为这些参数另外创建显式 FP32 master 副本。
这是 **BF16 参数存储**，不是“FP32 主参数 + BF16 前后向”的同义词。
官方 Cosmos3 明确推荐后者。

v5 Qwen 模型参数的峰值 LR 只有 `1e-6`，之后还会衰减；
按当前 scheduler，其最终下限是 `1.5e-8`，不是主组的 `1.5e-6`。
例如 BF16 中接近 `0.01` 的可表示值为 `0.010009765625`，
相邻更大值间隔为 `6.103515625e-5`；
从它减去 `1e-6` 再转换回 BF16，结果完全不变。
这是一个舍入示例，不是对每个 Adam update 大小的假设或完整证明。

### 5.1 实际 checkpoint 探针

本次仅用 CPU 读取更正上传后的 v5 12k/16k/18k DCP 中六个 tensor，
没有构造全模型、占用 GPU 或执行训练。比较的是两个 checkpoint 间
**数值不相等的元素占比**，不是梯度非零率。

| tensor 简称 | 元素数 | 12k→16k 变化比例 | 16k→18k 变化比例 |
|---|---:|---:|---:|
| Qwen `layers.0.self_attn.q_proj.weight` | 10,485,760 | 0.099716% | 0.017538% |
| Qwen `layers.17.mlp.down_proj.weight` | 24,903,680 | 0.111618% | 0.015395% |
| Qwen `layers.35.self_attn.q_proj.weight` | 10,485,760 | 0.027351% | 0.004692% |
| `understanding.query_tokens` | 40,960 | 16.997070% | 7.229004% |
| `generation.blocks.0.attention.q_proj.weight` | 16,777,216 | 57.957387% | 21.451318% |
| `physical_decoders.action.3.weight` | 163,840 | 1.666870% | 0.548706% |

三个 Qwen 矩阵属于 `full` 模式下可训练的 language layers，并非误选冻结 teacher。
它们 16k→18k 的元素变化数分别仅为 **1839、3834、492**；
这与 query/Generation 仍在变化形成明显对照。
探针的 Qwen 完整前缀是
`model.understanding.model.base_model.model.language_model.`；
其他行加 `model.` 前缀即可定位。
六个 tensor 的存储 dtype 均为 BF16。

该现象支持“名义 full finetune 未必意味着有效更新足够”的风险判断，
**不能据此推断所有 Qwen 参数都不动，也不能排除梯度本来很小、scheduler、稀疏任务敏感性等原因**。
仍需记录真正的 grad norm、step 前后相对更新量、BF16 ULP，
并在相同初始化和样本顺序下做 precision 对照，才能归因。

优先实验是保留其余设置，先比较 BF16 与 FP32 master；
然后在 FP32 master 下单独比较 Understanding LR scale。
不能同时切 optimizer、task、shift 后把结果统称为 precision 修复。
FP32 master 会增加参数/optimizer 显存与 checkpoint 成本，
需要审计 FSDP wrapping、8bit optimizer 和恢复契约；
不应简单把冻结 teacher/VAE 也全部转成 FP32，或直接复用旧 optimizer state。

## 6. P1：我们的 latent-action 不等同于 LaWAM

必须区分我们模型里的两种 latent：

| 名称 | 来源 | 用途 |
|---|---|---|
| change query / 蒸馏 latent action | 冻结 Stage 1 perception 的 primary 起止图像与语言 | 监督 Qwen query 的投影输出 |
| physical action embedding | 冻结 Stage 1 Physical Transformer 编码真实 action chunk | 作为 action flow 的干净目标，再通过新 decoder 恢复 raw action |

LaWAM 的预测 latent 更接近第一种，不是第二种物理动作 codec 的简单替代品。

### 6.1 LaWAM 的显式未来特征路径

固定版本的 [LaWAM backend][lawam-backend] 实现了：

```text
current images + language -> VLM -> predicted latent action
current visual features + predicted latent -> LAM decoder -> predicted future features
current features + predicted future features + VLM features -> action flow
```

训练时用未来图像提取 teacher 特征和 LAM latent，分别提供 perceptual MSE 与 latent distillation；
推理时不需要真实未来图像。
发布策略启用 `future_prediction`、`enable_loss_distill`、`unfreeze_lam_decoder`，
perceptual 和 latent distillation 权重均为 0.1。
其 `detach_future_feature=true` 还意味着训练时 action flow 的这条未来特征条件被 detach：
不能误称 action loss 必然经此路径端到端更新 latent predictor。
flow 同时接收当前视觉/VLM 特征，因此它也不是没有旁路的唯一 latent bottleneck。

### 6.2 当前实现缺少的不是“一个同名 MSE”

我们的 [forward](../lerobot/common/policies/qwen3vl_mot/modeling_qwen3vl_mot.py#L2022-L2159)
对投影后的 change query 做 teacher MSE，
但不将该投影结果显式传给 Generation，也没有在 Stage 2 中用它和当前视觉特征
重建未来 DINO 特征的 loss。
Generation 接收 Qwen 原生 K/V，其中包括 query token 的对应位置；
所以不能说 query 与动作完全无关、或蒸馏没有梯度作用。
准确说法是：**我们有共享 query/KV 的间接联系，但缺少 LaWAM 那种显式
“预测 latent→未来特征→动作条件”的路径和对应的未来特征监督。**

Stage 1 曾有的重建目标，也不等于 Stage 2 当前这个 Qwen predictor 已经保留同样的重建约束。
可以先验证 query 是否真的提供控制相关信息，再设计独立实验：
保留预测 query，在 action 条件中 mask/shuffle 其 K/V 位置，观察动作误差与闭环变化；
若增加未来特征预测，应明确 decoder 初始化/冻结、teacher stop-grad、是否 detach action 条件、
loss normalization 和开销，不能直接照搬 LaWAM 的 0.1 就认为权重已平衡。

### 6.3 物理 codec 的可逆性仍缺少独立验收

当前 embedding flow 的重建监督实际为：

```text
z_sigma = (1-sigma) * E(action) + sigma * noise
z0_pred = z_sigma - sigma * predicted_velocity
action_reconstruction_loss = MSE(D(z0_pred), normalized_raw_action)
```

不是先用 `D(E(action))` 将 codec 训练好，再训练 flow。
Physical encoder 冻结，decoder 是 Stage 2 新建，Stage 1 不提供已校准的逆映射。
低 latent MSE 不自动保证 xyz、旋转、gripper 的控制精度。

需要独立测量 `D(E(a))` 的 oracle 解码误差，并与 `D(z0_pred)` 对比：
按 xyz 物理单位、旋转角度、gripper 分类和 chunk 时间位置报告。
若 oracle 都差，应先检查 encoder 信息保留与 decoder 训练；
若 oracle 好而预测差，再排查 flow、噪声标定和采样器。
还应测 latent 各通道方差与有效 SNR，不能假设对比预训练后的 embedding
天然就是各维标准正态、与统一单位高斯噪声最匹配。
这些诊断尚未执行，不能先认定 codec 不可逆。

## 7. P1：尾部 padding 是真实监督差异，不是缺失 episode

本机四份数据的 metadata 为：

| suite | episodes | frames | 最短 episode | 32-step action 含 padding 的 anchor |
|---|---:|---:|---:|---:|
| Spatial | 432 | 52,970 | 75 | 13,392 |
| Object | 454 | 66,984 | 114 | 14,074 |
| Goal | 428 | 52,042 | 75 | 13,268 |
| LIBERO-10 | 379 | 101,469 | 150 | 11,749 |
| 合计 | 1693 | 273,465 | — | 52,483，19.191853% |

官方 [LIBERO-Cosmos3 metadata][cosmos-data] 也恰好是 **1693 episodes、273465 frames、20 FPS**。
因此没有证据说我们仅因 episode 数量不足而落后。
总数相同仍不证明图像内容、任务映射、旋转转换和统计文件逐项完全相同。

当前 [FullCoverageBatchSampler](../lerobot/common/datasets/contrastive_sampler.py#L349-L422)
遍历每个 episode 的所有帧，而不是仅完整窗口起点。
[reader](../lerobot/common/datasets/contrastive_dataset.py#L1407-L1556)
允许末尾图像/state 重复；LIBERO 的 action padding 使用零位移、
identity rot6d、保持最后 gripper，再进行 mean/std normalization。
当前 action mask 是有效通道 mask，没有据 `action_is_pad` 排除这些时间位置。
`pair_is_valid` 也只要求当前图像有效，不要求未来端点未被截断。

由于所有 episode 均超过 31 帧，每个 episode 最后 31 个 anchor 的 action chunk 含 padding。
按唯一 anchor 统计、不计最终 global batch 的重复补齐，完整覆盖一次共监督
`273465×32=8,750,880` 个 action 时间位置，
其中人工填充为 `1693×(1+…+31)=839,728` 个，约 **9.595926%**。
这是进入 encoder/loss 的时间位置占比，不是独立 loss 项的精确权重。

官方 Cosmos3 的 [prepare_loss_mask][cosmos-padding] 检查所有 `*_is_pad`，
只要一个样本的 clip/action 越过 episode 边界，就将整个样本排除出训练 loss。
两者对终止附近的监督明显不同。
我们当前行为可以理解为“完整保留演示并学习末尾静止”，不是跨 episode 读取或必然错误；
但应显式记录并做受控实验，不能把它描述成与官方相同的完整窗口训练。

建议比较现策略与“只监督真实时间位置/只取完整窗口”的版本，
保持原始 episode 集合不变，分别报告有效 targets 数、任务尾段误差、gripper 行为与闭环早停。
对非因果 physical encoder，不能只在最后的 raw loss mask 掉 padding 就认为影响完全消失：
填充值也可能参与真实位置 embedding 的上下文编码，需同时明确 encoder padding/attention 契约。

## 8. 其他差异：需要实验，不应先报成 bug

**噪声与 solver。** 官方 Cosmos3 发布策略使用 `waver` 时间分布，
training shift 按 canvas resolution tier 选择，sampling shift=5、UniPC 30；
我们的三组使用 shifted-uniform，v4/v5 shift2、v6 shift5，
推理则固定均匀 sigma 网格和 Euler 20。
改变训练 shift 不会自动改变当前推理网格。
这是需要控制的预训练分布/采样差异，但训练采样分布与数值积分网格不同本身不是错误。
当前 `x_sigma=(1-sigma)x0+sigma*noise`、target=`noise-x0`、sigma 从 1 积分到 0
在符号上是一致的，没有由此发现“加噪/去噪反号”问题。

**时间对齐与 horizon。** 我们 32 个连续 commands 与稀疏 video 覆盖 `[t,t+31]`；
官方 16 actions 配 17 帧到 `[t,t+16]`。
需要按 observation/action transition 的实际定义比较，而非仅按数组长度宣判 off-by-one。
当前代码已用 `sample_rate×8/31` 计算稀疏视频有效 FPS，并传给 Generation 做 FPS modulation：
20 Hz 数据对应约 5.161 Hz 稀疏视频，不能误报成简单忘记传 FPS。
32-step 模型只执行前 10 步，与 LaWAM 8/8 或官方 Cosmos3 16/16 也不等价。

**归一化与旋转表示。** 我们训练/评测都使用对应 suite 的相同 mean/std，
不是已确认的统计不匹配。quantile/min-max 与 mean/std 仍会改变各通道尺度和噪声 SNR。
现有转换将原 native 三维旋转控制按 Euler XYZ 编成 rot6d，评测用其逆映射恢复 native control；
与官方标准 axis-angle↔rotation matrix 语义不同，但 encode/decode 约定闭环一致，
不能只凭“Euler”就声称控制一定错误。
应先做真实标注往返误差和环境一步动作一致性检查。
更换 normalization/旋转语义需要重训并同步 reader、统计、codec 和 evaluator，
不能只修改现有 checkpoint 的推理后处理。

**增强。** 保存 JSON 里通用 `ImageTransformsConfig` 的 rotation 字段
不等于 Stage 2 实际用了旋转。当前 reader 调用固定的
[temporal RGB preset](../lerobot/common/datasets/transforms.py#L358-L369)：
crop 90%–100%、brightness/contrast 0.1、saturation 0.05，**无旋转**；
每相机一次采样供整段时序共用。LaWAM 明确无随机增强，值得做关闭 crop/jitter 的单因素对照。
不能仅凭目录名推断历史运行的完整数据实现；历史训练复现仍需当时源码/日志。

**冻结与优化目标。** 官方 Cosmos3 默认只训练 generation stream、冻结 vision encoder；
我们的 Qwen language layers 全量训练，vision 部分 LoRA，并增加 state flow、
change 蒸馏、predicted-clean 重建等目标。
官方 action projections 还有 5 倍 LR multiplier、grad clip=1；
我们 decoder 在主 LR 组、grad clip=10。
这些是实质差异，但 10/5/1 等 loss 系数不能脱离归一化、元素平均和梯度范数直接比较。
应测各 loss 对共享模块的 grad norm、方向冲突和闭环相关性。
EMA 可以另做对照，不能在数值更新未确认前把它当作首要补丁。

**验证与消融。** 当前生成验证明确标记 `validation_split=training`，
用于看模型是否拟合训练窗口，不是 held-out 泛化或 LIBERO success。
v4→v6 同时从 joint 改 VA、shift2 改 shift5，
不能将差异只归因于“去掉 state prediction”。
也不能仅因可视化视频更好就选择最终 action checkpoint。

## 9. 建议实验顺序

每组先做短程诊断、记录速度和实际有效更新，再决定是否跑完整预算。
固定初始化、数据顺序、global batch、task、shift、eval 协议；
架构必然带来的多项变更必须明确列出，不包装成单变量结果。

| 顺序 | 实验 / 验收 | 保持或控制 | 判定指标 |
|---|---|---|---|
| E0 | 数据/动作 roundtrip；发布策略的隔离环境 sanity；核对真实训练 world size | 不修改历史评测环境，固定代码/权重/初态身份 | native action 一步一致性、数据读取 fallback 计数、发布策略闭环结果、真实 batch/updates |
| E1a | BF16 master vs FP32 master 短程对照 | 同一个 v5 LR scale、optimizer、数据顺序；计算仍可 BF16 | 分层 grad norm、实际参数更新率/ULP、loss、显存、samples/s |
| E1b | 在确认更新有效后单独扫 Understanding LR scale | precision、task、shift 和预算固定 | Qwen/adapter/query 分组更新、固定协议闭环 success |
| E2 | 完成现有无 Stage 1 raw MLP baseline；另建官方原生 Cosmos3 参考 | 前者任务等按选定对照保留；后者采用完整官方 recipe | 记录所有架构/loss 差异、真实训练预算、完整 suite success |
| E3 | physical codec oracle 与 predicted-clean 解码诊断 | 固定 checkpoint、归一化和动作后处理 | xyz/rotation/gripper 误差、时间位置误差、latent 方差/SNR |
| E4 | joint vs VA，分别固定同一个 shift；另测 shift2 vs5 | 不同时改 task、shift、solver | 动作误差、state/video loss 与闭环收益是否一致 |
| E5 | 末尾 padding 监督策略对照 | 原始 episode 集合不变，匹配有效 targets/预算 | 尾段行为、早停/抖动、gripper、任务成功率 |
| E6 | 显式未来特征监督/条件路径 | 先明确 teacher、decoder、detach 和 loss 契约 | latent 的使用度、未来特征误差、动作与闭环收益、额外开销 |
| E7 | 逐项测试短 horizon、solver/grid、增强、normalization、EMA | 每次只改变一个可解释因素；表示变化必须重训 | 成功率/延迟/显存 trade-off，不仅是训练 MSE |

已有 MLP baseline 的安全契约是 `physical_encoder_type=mlp`、
`physical_flow_space=raw`、`physical_tuning_mode=full`、
`latent_action_loss_weight=0`、action/state predicted reconstruction 权重均为 0，
使用新输出目录，不恢复旧 Stage 1 optimizer/checkpoint 契约。
同时显式选择非触觉 task、关闭 `use_tactile_conditioning`，不要继承包含触觉的默认任务列表。
这组 baseline 同时改变 teacher、codec、flow 空间和相关 loss，
所以它能比较总体方案，但不单独量化“Stage 1 权重”的净收益。

模型筛选可以继续使用固定困难子集以控制成本，但最终比较应覆盖四 suite、
每任务 50 trials，明确训练和采样 seed、重规划步数、环境版本与失败视频。
条件允许时使用至少 3 个独立训练 seed，并区分模型选择数据与最终报告数据；
若复用初态做选择，必须披露，不能当作完全独立验证。

**推荐先做 E0/E1/E2/E3，不建议在当前 v5 上只延长训练，
或同时加入新 decoder、改 shift、换 solver 后再猜测原因。**

## 10. 可追溯证据与局限

本次源码审计基于当前 `333814e`。最近更正上传的评测使用冻结的推理快照
`45fc4ee70ec3c64e5a70c0ad1b57a6424e3543c4`；
实际权重探针直接读更正后的 DCP。
这不等于已逐行恢复每个历史训练任务当时运行的全部源码，
也没有替代对原始 trainer 日志、optimizer state 和更新量的进一步验收。

三份实际 `config.json` 的 SHA256：

```text
v4 0790bc154326368189102a8bb723cd2faec921a1b3359395b802bdb651549418
v5 4cf599fed3237d50c1e635d5c3a9c3e5ff1112f46be886d5ac4da81fa4d3eae2
v6 4c79d51219cfc894360d8365bbb1170aa2df784df82f6d47497227508995e73f
```

收尾复查时（2026-11-06 15:14 UTC），上述 v5 原目录已不再存在；
本次没有移动或删除权重，也未追踪其新位置。v5 配置 hash 与此前归档的更正上传
16k `preflight.json` 中 `config_sha256` 一致，CPU 权重探针在目录变化前已经完成。
因此本文保留当次读取的证据，但不表示原路径目前仍可用于训练/评测；
再次运行前需要重新定位并核验权重身份。

CPU 探针原始数据保存在本次 session 的
`files/libero_ft_audit_20261106_precision_probe.json`。
episode 长度统计来自本机
`/Data/lerobot_data_ort6d/v30/libero_{spatial,object,goal,10}_no_noops_lerobot/meta/episodes/`
的 `length` 列；本报告已保存关键计数，不依赖 session 文件才能阅读结论。

上游 LeRobot 实现固定为
`d490e472a5d1a207cfe0255caaaf74b6d66e4656`：
[Cosmos3 配置][cosmos-config]、[Cosmos3 实现][cosmos-model]、
[LaWAM 配置][lawam-config]、[LaWAM backend][lawam-backend]。
发布权重配置另固定为
[Cosmos3 Nano LIBERO][cosmos-release] 和 [LaWAM LIBERO SFT][lawam-release]，
数据 metadata 固定为 [LIBERO-Cosmos3][cosmos-data]。

不要混淆官方文档的原生 optimizer-update recipe、LeRobot microbatch CLI 和发布 config：
例如 Cosmos3 发布 config 保存的 warmup/decay 是 2000/64000，
LaWAM 发布 config 的 `scheduler_decay_steps=30000`；
它们不能脱离对应训练 accumulation/入口解释，
也不能用另一个版本的 class 默认值覆盖发布策略后仍称为原配置。
两份官方文档网页是本次查阅时的 `main`，会随上游更新；代码与发布配置以上述固定版本为准。

[cosmos-doc]: https://huggingface.co/docs/lerobot/main/en/cosmos3
[lawam-doc]: https://huggingface.co/docs/lerobot/main/en/lawam
[cosmos-config]: https://github.com/huggingface/lerobot/blob/d490e472a5d1a207cfe0255caaaf74b6d66e4656/src/lerobot/policies/cosmos3/configuration_cosmos3.py
[cosmos-model]: https://github.com/huggingface/lerobot/blob/d490e472a5d1a207cfe0255caaaf74b6d66e4656/src/lerobot/policies/cosmos3/modeling_cosmos3.py
[cosmos-padding]: https://github.com/huggingface/lerobot/blob/d490e472a5d1a207cfe0255caaaf74b6d66e4656/src/lerobot/policies/cosmos3/modeling_cosmos3.py#L237-L244
[lawam-config]: https://github.com/huggingface/lerobot/blob/d490e472a5d1a207cfe0255caaaf74b6d66e4656/src/lerobot/policies/lawam/configuration_lawam.py
[lawam-backend]: https://github.com/huggingface/lerobot/blob/d490e472a5d1a207cfe0255caaaf74b6d66e4656/src/lerobot/policies/lawam/vlas/lawam.py#L494-L720
[cosmos-release]: https://huggingface.co/lerobot/Cosmos3-Nano-Policy-LIBERO/blob/29563132da0815a4e034435ca55aaaf061ecb5ba/config.json
[lawam-release]: https://huggingface.co/jialei02/lawam-libero-sft-lerobot/blob/f3d7b131bf612127a6e384ea84f563bc23564832/config.json
[cosmos-data]: https://huggingface.co/datasets/lerobot/LIBERO-Cosmos3/blob/c28343a22d01ac8cd798042b2d82932cd43a21ae/meta/info.json
