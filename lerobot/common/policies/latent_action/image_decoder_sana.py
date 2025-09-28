import torch.nn as nn
import torch
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    SanaPipeline,
    SanaTransformer2DModel,
)
from diffusers.utils.torch_utils import get_device, is_torch_version, randn_tensor
from diffusers.image_processor import PixArtImageProcessor
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from transformers import Gemma2Model, AutoTokenizer, AutoConfig
import torchvision.transforms as transforms
import math
import copy
from typing import Optional, Dict, Any
from PIL.Image import Image

def forward_ip_adapter(
    self,
    hidden_states,
    condition,
    attention_mask,
    encoder_hidden_states,
    encoder_attention_mask,
    timestep,
    height,
    width,
):
    batch_size = hidden_states.shape[0]

    # 1. Modulation
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
    ).chunk(6, dim=1)

    # 2. Self Attention
    norm_hidden_states = self.norm1(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
    norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

    attn_output = self.attn1(norm_hidden_states)
    hidden_states = hidden_states + gate_msa * attn_output

    # 3. Cross Attention
    if self.attn2 is not None:
        attn_output = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = attn_output + hidden_states
        attn_output = self.attn3(
            hidden_states,
            encoder_hidden_states=condition,
            # attention_mask=encoder_attention_mask2,
        )
        attn_output = self.zero_linear(attn_output)
        hidden_states = attn_output + hidden_states

    # 4. Feed-forward
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

    norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)
    ff_output = self.ff(norm_hidden_states)
    ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
    hidden_states = hidden_states + gate_mlp * ff_output

    return hidden_states


def forward_c(
    self,
    hidden_states: torch.Tensor,
    condition: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
):
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 1. Input
    batch_size, num_channels, height, width = hidden_states.shape
    p = self.config.patch_size
    post_patch_height, post_patch_width = height // p, width // p

    hidden_states = self.patch_embed(hidden_states)

    timestep, embedded_timestep = self.time_embed(
        timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
    )

    encoder_hidden_states = self.caption_projection(encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
    encoder_hidden_states = self.caption_norm(encoder_hidden_states)

    # 2. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        def create_custom_forward(module, return_dict=None):
            def custom_forward(*inputs):
                if return_dict is not None:
                    return module(*inputs, return_dict=return_dict)
                else:
                    return module(*inputs)
            return custom_forward

        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
        for block in self.transformer_blocks:
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                condition,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                post_patch_height,
                post_patch_width,
                **ckpt_kwargs,
            )
    else:
        counter = 0
        for block in self.transformer_blocks:
            if counter < 14:
                hidden_states = block(
                    hidden_states,
                    condition,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    # condition is omitted for these blocks
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
            counter += 1

    # 3. Normalization
    shift, scale = (
        self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)
    ).chunk(2, dim=1)
    hidden_states = self.norm_out(hidden_states)

    # 4. Modulation
    hidden_states = hidden_states * (1 + scale) + shift
    hidden_states = self.proj_out(hidden_states)

    # 5. Unpatchify
    hidden_states = hidden_states.reshape(
        batch_size, post_patch_height, post_patch_width, self.config.patch_size, self.config.patch_size, -1
    )
    hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
    output = hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)

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
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 32
        )
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.transform = transforms.Compose(
            [
                # transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        # follow instruct pix2pix
        self.replace_module()

        # follow ip-adapter
        # https://github.com/rotem154154/ControlNet-Sana/blob/main/models/finetuner_ip_adapter.py#L115
        # self.preapre_transformer_for_ip_adapter()

    # def preapre_transformer_for_ip_adapter(self):
    #     # Replace forward with our custom forward.
    #     self.transformer.forward = forward_c.__get__(self.transformer)
    #     for i in range(len(self.transformer.transformer_blocks)):
    #         if i < 14:
    #             self.transformer.transformer_blocks[i].forward = forward_ip_adapter.__get__(self.transformer.transformer_blocks[i])
    #             self.transformer.transformer_blocks[i].attn3 = copy.deepcopy(self.transformer.transformer_blocks[i].attn2)
    #             for param in self.transformer.transformer_blocks[i].attn3.parameters():
    #                 param.requires_grad = True
    #             self.transformer.transformer_blocks[i].zero_linear = nn.Linear(1152, 1152).to(next(self.transformer.transformer_blocks[i].parameters()).device)
    #             nn.init.zeros_(self.transformer.transformer_blocks[i].zero_linear.weight)
    #             nn.init.zeros_(self.transformer.transformer_blocks[i].zero_linear.bias)

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
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, 
                        dtype, device, generator = None, latents=None):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def forward(self, prompt_embds, cond_image, target_image):
        prompt_embds = self.con_proj(prompt_embds)
        # print(torch.max(cond_image), torch.max(target_image))
        # cond_image = cond_image / 255
        # target_image = target_image / 255
        # cond_image = self.transform(cond_image)
        # target_image = self.transform(target_image)
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
        # prompt_attention_mask = torch.ones(prompt_embds.shape[0], prompt_embds.shape[1], dtype=torch.long, device=prompt_embds.device)
        # print(prompt_embds.shape)
        model_pred = self.transformer(
            hidden_states=concatenated_noisy_latents,
            # encoder_attention_mask=prompt_attention_mask,
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

    # def forward(self, prompt_embds, cond_image, target_image):
    #     prompt_embds = self.con_proj(prompt_embds)
    #     cond_latents = self.vae.encoder(cond_image)
    #     # torch.Size([25, 32, 16, 16])
    #     print(cond_latents.shape)

