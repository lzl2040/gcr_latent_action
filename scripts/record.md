## Download model
使用transformers调用下载的模型会有snapshot, blobs，要想没有这些东西，使用：
```shell
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./Qwen2.5-VL-7B-Instruct --local-dir-use-symlinks False
```
这个不会有snapshot, blobs等杂七杂八的

## InternVL
没有后缀的是最完整的版本，摘自它的huggingface:
```shell
If you're unsure which version to use, please select the one without any suffix, as it has completed the full training pipeline.
```
