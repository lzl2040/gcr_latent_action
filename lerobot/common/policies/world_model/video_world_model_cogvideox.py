from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel


class VideoWorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        print(f"Load diffusion model from {config.video_pred_model}")
        self.config = config
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            config.video_pred_model, 
            subfolder="transformer",
            local_files_only=True
        )
        self.vae = AutoencoderKLCogVideoX.from_pretrained(config.video_pred_model, 
                                                          subfolder="vae")

        self.scheduler = CogVideoXDPMScheduler.from_pretrained(
            config.video_pred_model, 
            subfolder="scheduler"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(config.video_pred_model, 
                                                       subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(
            config.video_pred_model, 
            subfolder="text_encoder"
        )
        self.transformer.enable_gradient_checkpointing()
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        self.prompt_proj = nn.Linear(self.config.vlm_token_dim, self.transformer.config.caption_channels)
    
    
    def forward(self, img_embeds, sc_embeds, act_embeds, task_info_dict, target_imgs, actions):
        if task_info_dict:
            task_embeds = task_info_dict["embeds"]
        else:
            task_embeds = torch.zeros((img_embeds.shape[0], 0, img_embeds.shape[2]), dtype=img_embeds.dtype, device=img_embeds.device)
        prompt_embeds = torch.cat([img_embeds, task_embeds, sc_embeds, act_embeds], dim = 1)
        prompt_embeds = self.prompt_proj(prompt_embeds)
        device = prompt_embeds.device
        
        