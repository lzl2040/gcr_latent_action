# Stage 1 tactile codec visualization

这个工具只加载 Stage 1 checkpoint 中的：

```text
physical_encoder.tactile_cnn.*
physical_encoder.tactile_recon.*
```

不会构造 Qwen、Cosmos VAE 或完整 Physical Transformer。它从本地 LeRobot V3 数据集中
抽取真实触觉帧，并输出：

- 原始触觉图像；
- encoder patch 相对空间平均特征的偏差热力图；
- decoder 去 z-score 后的 RGB 重建；
- 放大 4 倍的绝对误差；
- pixel MAE/MSE、PSNR 和训练目标同定义的 z-score MSE；
- 相对“只输出该数据集均值图像”基线的 MSE 改善率。

## 运行

默认 checkpoint 和数据根目录已经设置为：

```text
/Data/lzl/ace_stage1
/Data/lerobot_data_ort6d/v30/FTP-1
/media/v-wangxiaofa/新加卷/lerobot_data/OpenNeoData
```

直接运行：

```bash
conda activate lerobot_v2
python scripts/tactile_codec_visualization/visualize.py \
  --output-dir /Data/lzl/ace_stage1/tactile_codec_eval
```

先快速检查 `sharpa` 和 OpenNeo UR：

```bash
python scripts/tactile_codec_visualization/visualize.py \
  --dataset ftp_1_sharpa \
  --dataset open_neo_ur \
  --episodes-per-dataset 2 \
  --frames-per-episode 2 \
  --max-images-per-dataset 12 \
  --output-dir /Data/lzl/ace_stage1/tactile_codec_eval_smoke
```

输出目录：

```text
tactile_codec_eval/
├── index.html
├── metrics.json
├── metrics.csv
├── COMPLETE
├── ftp_1_sharpa/
│   ├── index.html
│   ├── contact_sheet.png
│   └── episode_*.png
└── open_neo_ur/
    ├── index.html
    ├── contact_sheet.png
    └── episode_*.png
```

浏览器打开 `index.html` 即可查看。`mean-baseline improvement` 为正，表示 learned
encoder/decoder 比固定输出数据集均值图像保留了更多当前触觉接触结构。

默认会使用和训练一致的 `tactile_dead_std` 跳过空间上完全平坦的死 pad；若需要查看这些
输入，可加 `--include-dead`。输出目录必须是一个尚不存在的新目录，避免不同 checkpoint
的结果被混在一起。
