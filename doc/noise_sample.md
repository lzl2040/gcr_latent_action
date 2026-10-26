# Latent 加噪、预测目标与推理采样

## 1. 结论

在 latent / embedding 上加噪，并不决定网络必须直接预测什么。
网络可以预测 Flow Matching 速度，也可以预测干净 latent，但训练目标与推理公式必须配套。

| 实现 | 网络直接输出 | 如何得到干净 latent | 推理时使用的速度 |
| --- | --- | --- | --- |
| Cosmos3-Nano 原生生成头 | Flow velocity | `z_sigma - sigma * velocity` | 网络输出 |
| 本项目当前 Action/State 生成头 | Flow velocity | `z_sigma - sigma * velocity` | 网络输出 |
| LoLA v07 | 干净 latent `z0_pred` | 网络输出 | `(z_sigma - z0_pred) / sigma` |

因此，当前代码中的 `predicted_clean = noisy - sigma * velocity` 不能只删除 `sigma`。
LoLA 没有在重建时乘这个因子，是因为它直接输出干净 latent，
并在转换为速度时除以对应的时间因子，而不是省略了时间依赖。

## 2. 当前实现：训练时在 embedding 上加噪

记 `z0` 为干净 embedding，`epsilon` 为标准高斯噪声，
`sigma` 为噪声混合比例。训练时：

```python
z_sigma = (1 - sigma) * z0 + sigma * epsilon
velocity_target = epsilon - z0
```

`sigma=0` 对应干净端点，`sigma=1` 对应纯噪声端点。
训练中的 `sigma` 由 `_sample_sigma()` 按配置采样；
加噪与速度目标由 `_flow_corrupt()` 构造。

当前 `physical_flow_space="embedding"` 路径先分别编码干净的 Action/State，
取出 PhysicalEncoder 的 contextual tokens 并 `detach()`，然后在这些 embedding 上加噪。
不是先给原始 Action/State 数值加噪，再送进 PhysicalEncoder。
作为已知条件的模态保持干净；以下加噪及监督针对需要预测的目标模态。

网络直接预测速度：

```python
velocity_pred = generation_model(z_sigma, sigma, conditions)
flow_loss = masked_mse(velocity_pred, epsilon - z0)
```

随后，用预测速度反算干净 embedding，再解码到归一化的 canonical Action/State 空间：

```python
z0_pred = z_sigma - sigma * velocity_pred
raw_pred = physical_decoder(z0_pred)
reconstruction_loss = masked_mse(raw_pred, raw_target)
```

实际代码会按配置权重组合 Flow loss 与 Action/State 重建 loss。
日志中的 `action_pred`、`state_pred` 对应后者，
不是 embedding 的 Flow loss，也不是从纯噪声完整采样后的 rollout 误差。

### 为什么不能直接去掉 `sigma`

假设网络准确预测 `velocity_target`，则：

```text
z_sigma - sigma * velocity_target
= (1 - sigma) * z0 + sigma * epsilon - sigma * (epsilon - z0)
= z0
```

若直接删除 `sigma`：

```text
z_sigma - velocity_target
= z0 + (1 - sigma) * (z0 - epsilon)
```

一般不等于 `z0`。只有 `sigma=1` 等特殊情况下，两种写法才一致。
当 `sigma` 很小时，输入已经接近干净 embedding，减去完整速度反而会过度修正。

## 3. 当前实现：推理时从高斯 embedding 开始积分

推理时没有未来 Action/State 的干净 embedding。
目标 embedding 直接初始化为高斯噪声；
当前可观测 State 则单独编码为干净条件，不参与目标去噪更新。

下面是省略 mask、多模态拼接等细节后的采样逻辑：

```python
z = torch.randn(batch_size, num_groups, physical_hidden_dim)
schedule = torch.linspace(1.0, 0.0, inference_steps + 1)

for sigma, next_sigma in zip(schedule[:-1], schedule[1:]):
    velocity = generation_model(z, sigma, conditions)
    z = z + (next_sigma - sigma) * velocity

raw_prediction = physical_decoder(z)
```

这里 `next_sigma - sigma` 为负数，积分方向是从噪声端点走向干净端点。
每一步都会根据当前 latent 和噪声强度重新预测速度。
训练中的 `z_sigma - sigma * velocity_pred` 是单个采样点上的干净估计，
不意味着有预测误差时可以直接用这一步替代完整的多步采样。

## 4. Cosmos3-Nano 原本输出什么

本地官方模型配套配置：

`/Data/lzl/huggingface/Cosmos3-Nano/scheduler/scheduler_config.json`

```json
{
  "predict_x0": true,
  "prediction_type": "flow_prediction"
}
```

这里两项设置的含义不同：

- `prediction_type="flow_prediction"`：网络输出按 Flow Matching 速度解释。
- `predict_x0=true`：UniPC 调度器将网络输出转换为干净样本估计，并采用基于该估计的更新算法；不是说网络直接输出干净样本。

本机 Diffusers 的 `UniPCMultistepScheduler.convert_model_output()` 对应分支为：

```python
elif self.config.prediction_type == "flow_prediction":
    sigma_t = self.sigmas[self.step_index]
    x0_pred = sample - sigma_t * model_output
```

这与当前 `noisy - sigma * velocity` 是同一个转换。
这里的 Flow velocity 不应与 Diffusers 中用于另一种扩散参数化的
`prediction_type="v_prediction"` 混为一谈；Nano 配置明确使用 `flow_prediction`。

本项目通过 `cosmos3_pretrained.py` 将原生 `proj_out` 加载到
`output_heads.video`，复用的是视频输出头。
Action/State 输入和输出头是本项目新增的，目前同样训练为速度预测；
没有原样加载 Nano 的原生 domain action head。
此外，本项目当前使用前述 Euler 采样，并未直接采用 Nano 配套的 UniPC 求解器。

## 5. LoLA v07 的训练与推理

参考文件：`NsurrenderX/lola_lerobot` 的
`src/lerobot/policies/lola_v07/modeling_lola_v07.py`，
以下行号按 2026-10-26 读取的 `main` 版本记录。
其 `t` 对应本文的 `sigma`。

### 训练

LoLA 同样在 ActionEncoder 的 bottleneck latent 上做线性加噪，
分别处理 arm 和 gripper：

```python
z_t = (1 - t) * z0 + t * epsilon
velocity_target = epsilon - z0

z0_pred = dit_to_latent(dit_output)
velocity_pred = (z_t - z0_pred) / t.clamp(min=1e-5)
flow_loss = mse(velocity_pred, velocity_target)
```

然后直接解码 `z0_pred`，计算 arm 的 Huber loss 与 gripper 的 BCE loss。
因此它虽然直接预测干净 latent，Flow loss 仍然监督转换后的速度。
在分母未触发 clamp 时，该速度损失代数上等于：

```text
flow_loss = ||z0_pred - z0||^2 / t^2
```

不能把它简单理解为没有时间权重的干净 latent MSE。

### 推理

LoLA 从标准高斯 latent、`t=1` 开始，每一步先预测干净 latent，再转换为速度：

```python
z0_pred = model(z_t, t, conditions)
velocity_pred = (z_t - z0_pred) / t.clamp(min=1e-5)
z_t = z_t + dt * velocity_pred
```

其中 `dt = -1 / num_inference_steps`。
循环结束后才解码最终 latent，并不是每步都直接用 `z0_pred` 替换 `z_t`。

## 6. 参数化切换的边界

如果要参考 LoLA，直接预测干净 embedding，需要同步处理：

- 输出含义：从 `velocity_pred` 改为 `z0_pred`。
- Flow loss：先通过 `(z_sigma - z0_pred) / sigma` 转成速度，再与速度目标比较，并处理小 `sigma` 的数值稳定性。
- 重建 loss：直接解码 `z0_pred`，不再把它当速度代入 `z_sigma - sigma * output`。
- 推理更新：使用转换后的速度积分；已有速度预测 checkpoint 的输出不能未经转换就解释为干净 embedding。

两种参数化都可以用于 latent Flow Matching。
上述区别本身不能证明哪一种训练更稳定或机器人成功率更高，需要独立实验验证。

## 7. 代码位置

本项目：

| 环节 | 代码位置 |
| --- | --- |
| 噪声强度采样 | `lerobot/common/policies/qwen3vl_mot/modeling_qwen3vl_mot.py`，`_sample_sigma()`，927 行起 |
| 加噪与速度目标 | 同上，`_flow_corrupt()`，946 行起 |
| 干净物理 embedding | 同上，`_clean_physical_embedding()`，1471 行起 |
| embedding 加噪接入 | 同上，`_prepare_embedding_physical_streams()`，1512 行起 |
| Flow 与重建监督 | 同上，2046–2075 行 |
| 联合推理采样 | 同上，`_sample_joint_prediction()`，1290 行起 |
| 原生视频输出头映射 | `lerobot/common/policies/qwen3vl_mot/cosmos3_pretrained.py`，127–128 行 |

本机调度器：

`/home/v-wangxiaofa/anaconda3/envs/lerobot_v2/lib/python3.10/site-packages/diffusers/schedulers/scheduling_unipc_multistep.py:705–707`

LoLA 参考：

- [训练加噪与速度目标，L748–752](https://github.com/NsurrenderX/lola_lerobot/blob/main/src/lerobot/policies/lola_v07/modeling_lola_v07.py#L748-L752)
- [预测干净 latent、转换速度及训练损失，L922–978](https://github.com/NsurrenderX/lola_lerobot/blob/main/src/lerobot/policies/lola_v07/modeling_lola_v07.py#L922-L978)
- [高斯 latent 初始化，L1177–1181](https://github.com/NsurrenderX/lola_lerobot/blob/main/src/lerobot/policies/lola_v07/modeling_lola_v07.py#L1177-L1181)
- [转换速度、Euler 更新与最终解码，L1227–1244](https://github.com/NsurrenderX/lola_lerobot/blob/main/src/lerobot/policies/lola_v07/modeling_lola_v07.py#L1227-L1244)
