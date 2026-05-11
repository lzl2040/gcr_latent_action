# Action Chunk Encoder
给定action chunk，固定维度是B * Chunk_Size * Action_Dim

我想把这个action信息encode为一个compact的embedding，具体做法如下：
- action_dim pad到32维
- 在chunk_size维度上，G个action为一组（默认为4），即action维度会变成 B * (Chunk_size / G) * (G * Action_dim)，然后升到768维度(可选)
- 一个根据sample_rate而改变得到的embedding跟action embedding拼接,输入到12层双向的self-attention,类似Bert
- 最终取sample_rate对应部分的attention output作为最终的embedding
- 注意在不同意义的部分加rope

在/home/v-wangxiaofa/lzl/gcr_latent_action/lerobot/common/policies/ace文件夹下实现这个模块, 并用随机数据测试