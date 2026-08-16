import sys
sys.path.insert(0, "/path/to/lerobot/common/policies/ace")

from lerobot.common.policies.ace.configuration_robo_clip import ACEConfig, RobotCLIPConfig
from modeling_ace import ActionChunkEncoder
from modeling_robo_clip import RobotCLIP
import torch
from PIL import Image

# ACE 模块
config = ACEConfig(action_dim=7, chunk_size=16, hidden_dim=128, num_attention_heads=8)
model = ActionChunkEncoder(config)
embedding = model(torch.randn(2, 16, 7), sample_rate=0)

# RobotCLIP 模块
config = RobotCLIPConfig(action_dim=7, chunk_size=16, hidden_dim=768)
model = RobotCLIP(config)
image_path = "/home/v-wangxiaofa/lzl/gcr_latent_action/debug.png"
pil_image = Image.open(image_path).convert("RGB")

loss = model({'images': [pil_image, pil_image], 'actions': torch.randn(2, 16, 7)})
print(loss)
