"""Reproduce the cluster's mixed-precision regime: parameters cast to bf16, buffers left alone.

That split is what turns a frozen BatchNorm (all buffers, no parameters) into an fp32 island
inside a bf16 ResNet.
"""

import sys

import torch
import torch.nn as nn

from lerobot.common.policies.ace.configuration_robo_contrast import RoboContrastConfig
from lerobot.common.policies.ace.modeling_robo_contrast import RoboContrast

cfg = RoboContrastConfig()
cfg.vision_model_name = "/Data/lzl/huggingface/dinov3-vitb16-pretrain-lvd1689m"
cfg.text_model_name = "/Data/lzl/huggingface/siglip2-base-patch16-224"
m = RoboContrast(cfg).cuda()

for p in m.parameters():
    p.data = p.data.to(torch.bfloat16)

tac = m.physical_encoder.tactile_cnn
bn = tac.stem[1]
print("stem conv param :", tac.stem[0].weight.dtype)
print("frozen BN buffer:", bn.weight.dtype, f"({type(bn).__name__}, params={len(list(bn.parameters()))})")

B, C = 4, 40
S, A, V = cfg.chunk_size, cfg.chunk_size, cfg.max_tactile_views
batch = {
    "image_t0": torch.randint(0, 255, (B, 3, 224, 224), dtype=torch.uint8).cuda(),
    "image_t1": torch.randint(0, 255, (B, 3, 224, 224), dtype=torch.uint8).cuda(),
    "task": ["pick up the cube"] * B,
    "observation.state": torch.randn(B, S, C).cuda(),
    "state_mask": torch.ones(B, C).cuda(),
    "action": torch.randn(B, A, C).cuda(),
    "action_mask": torch.ones(B, C).cuda(),
    "tactile_signal": torch.randn(B, S, cfg.max_tactile_signal_dim).cuda(),
    "tactile_signal_mask": torch.ones(B).cuda(),
    "tactile_image": torch.randint(
        0, 255, (B, V, 2, 3, cfg.tactile_img_size, cfg.tactile_img_size), dtype=torch.uint8
    ).cuda(),
    "tactile_image_mask": torch.ones(B, V).cuda(),
    "tactile_sensor_id": torch.zeros(B, V, dtype=torch.long).cuda(),
    "tactile_img_mean": torch.zeros(B, V, 3).cuda(),
    "tactile_img_std": torch.ones(B, V, 3).cuda(),
    "sample_rate": torch.full((B,), 15).cuda(),
    "pair_is_valid": torch.ones(B).cuda(),
    "dataset_index": torch.zeros(B, dtype=torch.long).cuda(),
    "episode_index": torch.arange(B).cuda(),
    "episode_uid": torch.arange(B).cuda(),
    "frame_index": torch.arange(B).cuda(),
}

m.train()
try:
    out = m.forward(batch)
    loss = out[0] if isinstance(out, tuple) else out["loss"]
    print("forward OK, loss =", float(loss.detach()))
except RuntimeError as exc:
    print("forward FAILED:", exc)
    sys.exit(1)
