## version
- ace_0413: ace主要版本，没有引入语言
- ace_plus_0701: 引入语言
- dinov3_perception: 基于dinov3做latent action跟物理侧的对齐
- cosmos3_perception: 基于cosmos3 encoder (不继承它原来权重)来提取latent action跟物理侧对齐
- 


## AI prompt record
### 0816: 引入触觉数据
我目前有一个想法，是先做机器人领域感知侧和物理侧在某段时间内的对比学习。
感知侧包括vision, language, mask，光流等包含全局场景信息的模态
物理侧包括state、action、触觉等包含机器人信息的模态

目前有包含触觉的数据作为测试:
```text
("ftp_1_RH20TCfg5Franka", 0.1),
("ftp_1_sharpa_split_0", 0.1),
("ftp_1_VisuoTactile_D-WHEEL_split_0", 0.1)
```
但注意这些数据集的触觉模态是不统一的，有些是图片，有些是信号

之前运行的代码是：
```shell
conda activate lerobot_v2
bash train_ace_local.sh
```
目前的问题的是：
我之前跑的代码是不支持触觉的，而且数据集里面有些有触觉数据，有些是没有的

同时，模型结构也需要重新设计，但主体要遵循对比学习的思路，之前模型代码在：/home/v-wangxiaofa/lzl/gcr_latent_action/lerobot/common/policies/ace下面。

设计模型的时候需要考虑：
- 怎么利用文本提取两个时刻的视觉特征之间的变化，和物理侧提取的特征对齐
- action和state空间的设计，数据集中大部分是xyz+ort6d+gripper，但也有部分是含有joint信息的，怎么设计action空间填充这些内容
- 触觉信息的处理：触觉有些是信号，跟action和state可以看作是相同模态，但有些是image，维度很高，这可能会让模型在学习的时候触觉占主导，这个需要设计一下
- 数据的读取：目前的dataset类不支持读取触觉，而且不支持读取两个时刻的帧，而且，可能存在一些小问题
- 负样本的选择：因为是对比学习，假如一次选了256个batch，能不能设计成大部分来自相同的数据集，其保证有一部分是同一个epsisode的作为负样本，但注意同一个episode的负样本不能跟当前样本靠的太近

我现在需要你帮我解决上面的问题，最终能够在conda activate lerobot_v2下成功运行bash train_ace_local.sh，模型mixture=debug_research_data
注意：
- 遇到比较大的更新就提交git，不需要push
- 最终的模型单卡支持的batch_size最好不少于128
- 训练效率：最终的模型训练不能太慢