import torch.nn as nn
import torch
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    SanaPipeline,
    SanaTransformer2DModel,
)
from transformers import Gemma2Model, AutoTokenizer, AutoConfig
import torchvision.transforms as transforms
import math

def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device = "cuda"):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma




class ImagePredictionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_pred_model = config.img_pred_model
        self.weighting_scheme = "logit_normal"
        self.logit_mean = 0
        self.logit_std = 1
        self.mode_scale = 1.29
        print(f"Load diffusion model from {self.img_pred_model}")

        self.vae = AutoencoderDC.from_pretrained(
            self.img_pred_model,
            subfolder="vae",
            local_files_only=True
        )

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.img_pred_model, 
            subfolder="scheduler",
            local_files_only=True
        )
        
        self.transformer = SanaTransformer2DModel.from_pretrained(
            self.img_pred_model, 
            subfolder="transformer", 
            locals_files_only=True
        )

        self.con_proj = nn.Linear(self.config.vlm_token_dim, 2304)
        
        # 梯度检查点
        self.transformer.enable_gradient_checkpointing()
        self.vae.requires_grad_(False)
        self.vae_config_scaling_factor = self.vae.config.scaling_factor
        self.transform = transforms.Compose(
            [
                # transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.replace_module()

        # for param in self.transformer.parameters():
        #     param.requires_grad = False


    def replace_module(self):
        print("Initializing the new channel of DIT from the pretrained DIT.")
        in_channels = 2 * self.transformer.config.in_channels # 48 for mask
        out_channels = self.transformer.patch_embed.proj.out_channels

        load_num_channel = self.transformer.config.in_channels
        print("new in_channels",in_channels)
        print("load_num_channel",load_num_channel)

        self.transformer.register_to_config(in_channels=in_channels)
        print("transformer.pos_embed.proj.weight.shape", self.transformer.patch_embed.proj.weight.shape)
        print("load_num_channel", load_num_channel)
        with torch.no_grad():
            new_proj = nn.Conv2d(
                in_channels, out_channels, kernel_size=(self.transformer.config.patch_size, self.transformer.config.patch_size),
                stride=self.transformer.config.patch_size, bias=True
            )
            print("new_proj", new_proj)

            new_proj.weight.zero_()
            # init.kaiming_normal_(new_proj.weight, mode='fan_out', nonlinearity='relu')
            # if new_proj.bias is not None and transformer.pos_embed.proj.bias is not None:
            #     new_proj.bias.copy_(transformer.pos_embed.proj.bias)
            # else:
            #     if new_proj.bias is not None:
            #         new_proj.bias.zero_()
            new_proj = new_proj.to(self.transformer.patch_embed.proj.weight.dtype)
            new_proj.weight[:, :load_num_channel, :, :].copy_(self.transformer.patch_embed.proj.weight)
            new_proj.bias.copy_(self.transformer.patch_embed.proj.bias)
            print("new_proj", new_proj.weight.shape)
            print("transformer.pos_embed.proj", self.transformer.patch_embed.proj.weight.shape)
            self.transformer.patch_embed.proj = new_proj
    
    def forward(self, prompt_embds, cond_image, target_image):
        prompt_embds = self.con_proj(prompt_embds)
        cond_image = cond_image / 255
        target_image = target_image / 255
        cond_image = self.transform(cond_image)
        target_image = self.transform(target_image)
        target_latents = self.vae.encode(target_image).latent
        target_latents = target_latents * self.vae_config_scaling_factor
        noise = torch.randn_like(target_latents)
        bsz = target_latents.shape[0]
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        if self.weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            u = torch.normal(mean=self.logit_mean, std=self.logit_std, size=(bsz,), device="cpu")
            u = torch.nn.functional.sigmoid(u)
        elif self.weighting_scheme == "mode":
            u = torch.rand(size=(bsz,), device="cpu")
            u = 1 - u - self.mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=(bsz,), device="cpu")
        
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=target_latents.device)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = get_sigmas(self.noise_scheduler, timesteps, n_dim=target_latents.ndim, dtype=target_latents.dtype)
        noisy_model_input = (1.0 - sigmas) * target_latents + sigmas * noise

        # Get the additional image embedding for conditioning.
        # Instead of getting a diagonal Gaussian here, we simply take the mode.
        original_image_embeds = self.vae.encode(cond_image.to(self.vae.dtype)).latent
        # B 32 64 64
        concatenated_noisy_latents = torch.cat([noisy_model_input, original_image_embeds], dim=1)
        # 1=keep, 0=remove
        prompt_attention_mask = torch.ones(prompt_embds.shape[0], prompt_embds.shape[1], dtype=torch.long, device=prompt_embds.device)
        # print(prompt_embds.shape)
        model_pred = self.transformer(
            hidden_states=concatenated_noisy_latents,
            encoder_attention_mask=prompt_attention_mask,
            encoder_hidden_states=prompt_embds,
            timestep=timesteps,
            return_dict=False,
            # mask_index = mask_index
        )[0]

        if self.weighting_scheme == "sigma_sqrt":
            weighting = (sigmas ** -2.0).float()
        elif self.weighting_scheme == "cosmap":
            bot = 1 - 2 * sigmas + 2 * sigmas ** 2
            weighting = 2 / (math.pi * bot)
        else:
            weighting = torch.ones_like(sigmas)

        # flow matching loss
        # print(noise.shape, target_latents.shape)
        target = noise - target_latents
        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        return loss
