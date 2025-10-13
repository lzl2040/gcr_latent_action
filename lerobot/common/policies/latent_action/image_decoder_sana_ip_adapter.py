import torch.nn as nn
import torch
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    SanaPipeline,
    SanaTransformer2DModel,
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from diffusers.models.attention import BasicTransformerBlock,JointTransformerBlock
from diffusers.utils.torch_utils import get_device, is_torch_version, randn_tensor
from diffusers.models.transformers.sana_transformer import SanaTransformerBlock, SanaAttnProcessor2_0
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.image_processor import PixArtImageProcessor
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
from typing import Any, Dict, Optional, Tuple, Union
import copy
from transformers import CLIPImageProcessor
from PIL import Image
import torchvision.transforms as transforms
import math
from lerobot.common.policies.latent_action.image_resampler import Resampler


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device = "cuda"):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, 
                 clip_embeddings_dim=1024, 
                 clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class SanaTransformerBlock_IP(SanaTransformerBlock):
    def __init__(self, is_ip_adapter, ip_token_num, **kwargs):
        super().__init__(**kwargs)
        self.is_ip_adapter = is_ip_adapter
        self.ip_token_num = ip_token_num
        if self.is_ip_adapter:
            cross_attention_dim = kwargs.get("cross_attention_dim", 1152)
            # self.attn3 = copy.deepcopy(self.attn2)
            qk_norm = kwargs.get("qk_norm", None)
            num_cross_attention_heads = kwargs.get("num_cross_attention_heads", 20)
            cross_attention_head_dim = kwargs.get("cross_attention_head_dim", 20)
            dropout = kwargs.get("dropout", 0)
            attention_out_bias = kwargs.get("attention_out_bias", True)
            # self.attn3 = Attention(
            #     query_dim = kwargs.get("dim", 2240),
            #     qk_norm=qk_norm,
            #     kv_heads=num_cross_attention_heads if qk_norm is not None else None,
            #     cross_attention_dim=kwargs.get("cross_attention_dim", 2240),
            #     heads=num_cross_attention_heads,
            #     dim_head=cross_attention_head_dim,
            #     dropout=dropout,
            #     bias=True,
            #     out_bias=attention_out_bias,
            #     processor=SanaAttnProcessor2_0(),
            # )
            self.attn3 = copy.deepcopy(self.attn2)
            self.zero_linear = nn.Linear(cross_attention_dim, cross_attention_dim)
            nn.init.zeros_(self.zero_linear.weight)
            nn.init.zeros_(self.zero_linear.bias)

    def forward(self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        height: int = None,
        width: int = None,):

        num_tokens = self.ip_token_num
        end_pos = encoder_hidden_states.shape[1] - num_tokens
        encoder_hidden_states, ip_encoder_hidden_states = (
            encoder_hidden_states[:, :end_pos, :],
            encoder_hidden_states[:, end_pos:, :],
        )
        encoder_attention_mask, ip_encoder_attention_mask = (
            encoder_attention_mask[:, :, :end_pos],
            encoder_attention_mask[:, :, end_pos:],
        )
        # print(end_pos, num_tokens)

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
            # print(hidden_states.shape, encoder_hidden_states.shape, encoder_attention_mask.shape)
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask
            )
            # hidden_states = attn_output + hidden_states

            if self.is_ip_adapter:
                # follow https://github.com/rotem154154/ControlNet-Sana/blob/main/models/finetuner_ip_adapter.py#L19
                attn_output_ip = self.attn3(
                    hidden_states,
                    encoder_hidden_states = ip_encoder_hidden_states,
                    attention_mask=ip_encoder_attention_mask
                )
                attn_output_ip = self.zero_linear(attn_output_ip)
                # print(torch.max(attn_output), torch.max(attn_output_ip))
                attn_output = attn_output + attn_output_ip
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
    ip_tokens: torch.Tensor,
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
    
    encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim = 1)
    
    encoder_attention_mask = torch.ones(encoder_hidden_states.shape[0], encoder_hidden_states.shape[1], dtype=torch.long, device=encoder_hidden_states.device)

    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 2. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for index_block, block in enumerate(self.transformer_blocks):
            from torch.utils.checkpoint import checkpoint
            # self._gradient_checkpointing_func
            hidden_states = checkpoint(
                block.forward,
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                post_patch_height,
                post_patch_width,
                use_reentrant=True
            )
    else:
        for index_block, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                post_patch_height,
                post_patch_width,
            )

    # 3. Normalization
    hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)

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

class ImagePredictionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_pred_model = config.img_pred_model
        self.img_encoder_model = config.img_encoder_model
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

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.img_encoder_model)
        if config.ip_token_gen_type == "cls_proj":
            self.img_proj_model = ImageProjModel(
                cross_attention_dim=self.transformer.config.cross_attention_dim,
                clip_embeddings_dim=self.image_encoder.config.projection_dim,
                clip_extra_context_tokens=self.config.ip_token_num)
        else:
            self.img_proj_model = Resampler(
                                        dim=self.transformer.config.cross_attention_dim,
                                        depth=4,
                                        dim_head=64,
                                        heads=6,
                                        num_queries=config.ip_token_num,
                                        embedding_dim=self.image_encoder.config.hidden_size,
                                        output_dim=self.transformer.config.cross_attention_dim,
                                        ff_mult=2)
        self.clip_processor = CLIPImageProcessor()

        self.vae_config_scaling_factor = self.vae.config.scaling_factor
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 32
        )
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.con_proj = nn.Linear(self.config.vlm_token_dim, self.transformer.config.caption_channels)

        self.transform = transforms.Compose(
            [
                # transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.vae_config_scaling_factor = self.vae.config.scaling_factor
        
        # 梯度检查点
        self.transformer.enable_gradient_checkpointing()
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        
        self.prepare_ip_adapter_module(n = self.config.ip_skip_num, ip_token_num = self.config.ip_token_num)
        for name, param in self.transformer.named_parameters():
            # 10.10: add train attn2
            if "attn3" in name or "zero_linear" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def prepare_ip_adapter_module(self, n = 2, ip_token_num = 4):
        print("Replace IP-Adapter Module")
        inner_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        old_transformer_weights = self.transformer.state_dict()
        block_kwargs = dict(
            dim=inner_dim,
            num_attention_heads=self.transformer.config.num_attention_heads,
            attention_head_dim=self.transformer.config.attention_head_dim,
            num_cross_attention_heads=self.transformer.config.num_cross_attention_heads,
            cross_attention_head_dim=self.transformer.config.cross_attention_head_dim,
            cross_attention_dim=self.transformer.config.cross_attention_dim,
            attention_bias=self.transformer.config.attention_bias,
            norm_elementwise_affine=self.transformer.config.norm_elementwise_affine,
            norm_eps=self.transformer.config.norm_eps,
            mlp_ratio=self.transformer.config.mlp_ratio,
        )

        for i in range(self.transformer.config.num_layers):
            if i == 0 or i % n == 0:
                self.transformer.transformer_blocks[i] = SanaTransformerBlock_IP(
                    True, ip_token_num, 
                    **block_kwargs)
            else:
                self.transformer.transformer_blocks[i] = SanaTransformerBlock_IP(
                    False, ip_token_num,
                    **block_kwargs)
        
        self.transformer.forward = forward_c.__get__(self.transformer)
        missing_keys, unexpected_key = self.transformer.load_state_dict(old_transformer_weights, strict=False)
        print(missing_keys)
    
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
        # print(height, width, self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def forward(self, prompt_embeds, cond_image, target_image):
        prompt_embeds = self.con_proj(prompt_embeds)
        device = prompt_embeds.device
        clip_imgs = []
        bs = cond_image.shape[0]
        cond_image = cond_image.permute(0, 2, 3, 1)
        cond_image = cond_image.detach().cpu().numpy()
        for i in range(bs):
            img = cond_image[i]
            pil_img = Image.fromarray(img, mode="RGB")
            clip_img = self.clip_processor(images=pil_img, return_tensors="pt").pixel_values
            clip_imgs.append(clip_img)
        clip_imgs = torch.cat(clip_imgs, dim = 0).to(device=device)
        # for img_proj_model is clip
        image_embeds = self.image_encoder(clip_imgs).image_embeds # this is cls token embedding
        # for img_proj_model is resampler
        # image_embeds = self.image_encoder(clip_imgs, output_hidden_states=True).hidden_states[-2]
        # print(image_embeds.shape)
        ip_tokens = self.img_proj_model(image_embeds) # torch.Size([25, 4, 1152])
        # print(ip_tokens.shape)

        target_image = target_image / 255
        target_image = self.transform(target_image)
        target_latents = self.vae.encode(target_image).latent
        target_latents = target_latents * self.vae_config_scaling_factor
        noise = torch.randn_like(target_latents)

        bsz = target_latents.shape[0]

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

        sigmas = get_sigmas(self.noise_scheduler, timesteps, n_dim=target_latents.ndim, dtype=target_latents.dtype)
        noisy_model_input = (1.0 - sigmas) * target_latents + sigmas * noise

        # print(prompt_embds.shape, ip_tokens.shape)

        model_pred = self.transformer(
            hidden_states=noisy_model_input,
            # encoder_attention_mask=prompt_attention_mask,
            encoder_hidden_states=prompt_embeds,
            ip_tokens=ip_tokens,
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

        target = noise - target_latents
        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        return loss
    



