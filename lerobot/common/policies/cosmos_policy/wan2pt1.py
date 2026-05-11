import torch.nn as nn
import torch
from diffusers import AutoencoderKLCosmos, AutoencoderKLWan

class Wan2pt1VAEInterface(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vae = AutoencoderKLWan.from_pretrained(
            config.vae_path, 
            subfolder="vae", 
            locals_files_only=True
        )
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1)
        self.latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1)
        # self.scale = [self.vae.config.latents_mean, 1 / self.vae.config.latents_std]
        
    
    @torch.no_grad()
    def encode(self, videos):
        device = videos.device
        dtype = videos.dtype
        self.latents_mean = self.latents_mean.to(device=device, dtype=dtype)
        self.latents_std = self.latents_std.to(device=device, dtype=dtype)
        video_latents = self.vae.encode(videos).latent_dist.sample()
        video_latents = (video_latents - self.latents_mean) / self.latents_std
        return video_latents
