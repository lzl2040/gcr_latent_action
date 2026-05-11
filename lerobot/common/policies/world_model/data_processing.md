# 训练数据的内部处理
## 处理流程
- 根据data_id调用decode_video_frames_torchcodec函数返回历史帧(包括当前帧)+未来帧的集合，以及历史帧的数目
    - 如果里面出现问题，返回的是全为0的frames
- 图像resize成256 * 256
- 对于历史帧，如果少于max_history_frame，则使用最后一帧，即当前帧补充, 数据源格式还是PIL.Image
- 将历史帧和文本构造成一个模板，输入给VLM处理成input ids
- 对于未来帧，使用生成模型的processor处理
```python
self.video_processor = T.Compose(
    [
        ToTensorVideo(),  # TCHW
        ResizeCrop((cfg.dataset.default_image_size, cfg.dataset.default_image_size)),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ]
)
furture_images_np = np.array([np.array(img) for img in furture_images])  # T H W C
 furture_images = torch.from_numpy(furture_images_np).clone().permute(0, 3, 1, 2)  # TCHW
```
- 对于少于max_frame的，使用最后一帧填充
