import torch.nn as nn
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
)
import copy
import torch
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
    # def __init__(self, img_pred_model):
        super().__init__()
        self.config = config
        self.img_pred_model = config.img_pred_model
        self.weighting_scheme = "logit_normal"
        self.logit_mean = 0
        self.logit_std = 1
        self.mode_scale = 1.29
        print(f"Load diffusion model from {self.img_pred_model}")
        self.vae = AutoencoderKL.from_pretrained(
            self.img_pred_model,
            subfolder="vae",
        )
        self.transformer = SD3Transformer2DModel.from_pretrained(
            self.img_pred_model, 
            subfolder="transformer", 
        )
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.img_pred_model, subfolder="scheduler"
        )
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)

        self.con_proj = nn.Linear(self.config.vlm_token_dim, 4096)
        self.con_proj_pool = nn.Linear(self.config.vlm_token_dim, 2048)

        self.vae.requires_grad_(False)
        self.replace_module()

        # 4B可学习参数会报错: Error invalid configuration argument at line 220 in file /src/csrc/ops.cu
        for param in self.transformer.parameters():
            param.requires_grad = True

    def replace_module(self):
        print("Initializing the new channel of DIT from the pretrained DIT.")
        in_channels = 2 * self.transformer.config.in_channels # 48 for mask
        out_channels = self.transformer.pos_embed.proj.out_channels

        load_num_channel = self.transformer.config.in_channels
        print("new in_channels",in_channels)
        print("load_num_channel",load_num_channel)

        self.transformer.register_to_config(in_channels=in_channels)
        print("transformer.pos_embed.proj.weight.shape", self.transformer.pos_embed.proj.weight.shape)
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
            new_proj = new_proj.to(self.transformer.pos_embed.proj.weight.dtype)
            new_proj.weight[:, :load_num_channel, :, :].copy_(self.transformer.pos_embed.proj.weight)
            new_proj.bias.copy_(self.transformer.pos_embed.proj.bias)
            print("new_proj", new_proj.weight.shape)
            print("transformer.pos_embed.proj", self.transformer.pos_embed.proj.weight.shape)
            self.transformer.pos_embed.proj = new_proj

    def forward(self, prompt_embds, cond_image, target_image):
        pool_feats = torch.mean(prompt_embds, dim=1)
        pool_feats = self.con_proj_pool(pool_feats)
        # print(f"Pool feats:{pool_feats.shape}")
        prompt_embds = self.con_proj(prompt_embds)
        cond_image = 2 * (cond_image / 255) - 1
        cond_image = cond_image.to(dtype=self.vae.dtype)
        target_image = 2 * (target_image / 255) - 1
        target_image = target_image.to(dtype=self.vae.dtype)
        latents = self.vae.encode(target_image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        # print("latent", latents.shape)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
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

        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=latents.device)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        sigmas = get_sigmas(self.noise_scheduler_copy, timesteps, n_dim=latents.ndim, 
                            dtype=latents.dtype, device = latents.device)
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

        # Get the additional image embedding for conditioning.
        # Instead of getting a diagonal Gaussian here, we simply take the mode.
        original_image_embeds = self.vae.encode(cond_image.to(self.vae.dtype)).latent_dist.mode()
        # B 32 64 64
        concatenated_noisy_latents = torch.cat([noisy_model_input, original_image_embeds], dim=1)
        # print(prompt_embds.shape)
        model_pred = self.transformer(
            hidden_states=concatenated_noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embds,
            pooled_projections=pool_feats,
            return_dict=False,
            # mask_index = mask_index
        )[0]
        # print(model_pred.shape) # torch.Size([1, 64, 4096])
        model_pred = model_pred * (-sigmas) + noisy_model_input
        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        if self.weighting_scheme == "sigma_sqrt":
            weighting = (sigmas ** -2.0).float()
        elif self.weighting_scheme == "cosmap":
            bot = 1 - 2 * sigmas + 2 * sigmas ** 2
            weighting = 2 / (math.pi * bot)
        else:
            weighting = torch.ones_like(sigmas)

        target = latents
        # Conditioning dropout to support classifier-free guidance during inference. For more details
        # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.

        # Concatenate the `original_image_embeds` with the `noisy_latents`.

        # Get the target for loss depending on the prediction type
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        return loss


if __name__ == "__main__":
    model = ImagePredictionModel("stabilityai/stable-diffusion-3.5-medium")
