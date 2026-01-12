import torch.nn as nn
import torch
from torch import Tensor, nn
from diffusers import (
    AutoencoderKLWan,
    DPMSolverMultistepScheduler,
    SanaVideoPipeline,
    FlowMatchEulerDiscreteScheduler,
    SanaVideoTransformer3DModel,
)
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from lerobot.common.policies.world_model.modified_sana_transformer import Modified_SanaVideoTransformerBlock, Modified_SanaModulatedNorm, Modified_WanRotaryPosEmbed
from lerobot.common.utils.utils import get_safe_dtype
import math
import torch.nn.functional as F  # noqa: N812
from diffusers.video_processor import VideoProcessor
import imageio.v2 as imageio
import numpy as np
import time


latents_mean =  [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]
latents_std = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
]

def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device = "cuda"):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def prepare_encoder_attention_mask(
    N_V: int,
    N_A: int,
    M_H: int,
    M_LS: int,
    M_LM: int,
    *,
    batch_size: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
):
    """
    Build cross-attention mask for:
    Target: [Video | Action]
    Source: [History Video | Latent Scene | Latent Motion]

    Rules:
    - Video queries attend to History Video + Latent Scene
    - Action queries attend to Latent Motion only

    Returns:
    attn_mask:
        shape = (Q_len, K_len)                if batch_size is None
            or (B, Q_len, K_len)             if batch_size is not None
        values = 0 (allowed) or -inf (masked)
    """

    Q_len = N_V + N_A
    K_len = M_H + M_LS + M_LM

    # initialize all masked
    attn_mask = torch.full(
        (Q_len, K_len),
        float("-inf"),
        device=device,
        dtype=dtype,
    )

    # ----- Video queries -----
    video_q = slice(0, N_V)
    history_k = slice(0, M_H)
    scene_k = slice(M_H, M_H + M_LS)
    motion_k = slice(M_H + M_LS, M_H + M_LS + M_LM)

    attn_mask[video_q, history_k] = 0.0
    attn_mask[video_q, scene_k] = 0.0
    attn_mask[video_q, motion_k] = 0.0

    # ----- Action queries -----
    action_q = slice(N_V, N_V + N_A)
    motion_k = slice(M_H + M_LS, M_H + M_LS + M_LM)

    attn_mask[action_q, motion_k] = 0.0

    if batch_size is not None:
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1)

    return attn_mask

def prepare_attention_mask(
    N_V, 
    N_A,
    batch_size: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32
):
    Q_len = N_V + N_A
    # initialize all masked
    attn_mask = torch.full(
        (Q_len, Q_len),
        float("-inf"),
        device=device,
        dtype=dtype,
    )
    video_q = slice(0, N_V)
    action_q = slice(N_V, N_V + N_A)
    attn_mask[video_q, action_q] = 0.0
    attn_mask[action_q, video_q] = 0.0
    if batch_size is not None:
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1)

    return attn_mask

def forward_c(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples: Optional[Tuple[torch.Tensor]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        freqs_cos, freqs_sin, = self.rope(hidden_states)
        rotary_emb = (freqs_cos, freqs_sin)
        
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        if guidance is not None:
            timestep, embedded_timestep = self.time_embed(
                timestep.flatten(), guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        timestep = timestep.view(batch_size, -1, timestep.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # 2. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_num_frames,
                    post_patch_height,
                    post_patch_width,
                    rotary_emb,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        else:
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_num_frames,
                    post_patch_height,
                    post_patch_width,
                    rotary_emb,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        # 3. Normalization
        hidden_states = self.norm_out(hidden_states, embedded_timestep)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output)

        return Transformer2DModelOutput(sample=output)

class VideoWorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        print(f"Load diffusion model from {config.video_pred_model}")
        self.config = config
        self.vae = AutoencoderKLWan.from_pretrained(
            config.video_pred_model,
            subfolder="vae",
            local_files_only=True
        )
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial
        self.vae_mean = torch.tensor(latents_mean)
        self.vae_std = torch.tensor(latents_std)
        self.z_dim = self.vae.z_dim
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.video_pred_model, 
            subfolder="scheduler",
            local_files_only=True
        )
        
        self.transformer = SanaVideoTransformer3DModel.from_pretrained(
            config.video_pred_model, 
            subfolder="transformer", 
            locals_files_only=True
        )
        self.inner_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        self.transformer.norm_out = Modified_SanaModulatedNorm(self.inner_dim, 
                                                               elementwise_affine=False, 
                                                               eps=1e-6, 
                                                               inner_dim=self.inner_dim)
        
        self.prompt_proj = nn.Linear(self.config.vlm_token_dim, self.transformer.config.caption_channels)
        
        
        # 梯度检查点
        self.transformer.enable_gradient_checkpointing()
        self.vae.requires_grad_(False)
        
        # self.transformer.forward = forward_c.__get__(self.transformer)
        self.prepare_modules()
        self.dtype = torch.bfloat16
    
    def prepare_modules(self):

        self.transformer.forward = forward_c.__get__(self.transformer)
        # missing_keys, unexpected_key = self.transformer.load_state_dict(old_transformer_weights, strict=False)
        # print(missing_keys)
        
        print(f"Replace scale_shift_table...")
        old_table = self.transformer.scale_shift_table.data 
        old_table = old_table.view(2, -1)  # [2, 2240]

        new_table = self.transformer.norm_out.scale_shift_table.data  # [2, 2240]

        # 把旧权重扩展拷贝进去
        new_table[:, :old_table.shape[1]] = old_table

        # 更新 norm_out 的表
        self.transformer.norm_out.scale_shift_table.data = new_table
    
    def save_video(self, video_latents, save_name):
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(video_latents.device, video_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            video_latents.device, video_latents.dtype
        )
        video_latents = video_latents / latents_std + latents_mean
        try:
            video = self.vae.decode(video_latents, return_dict=False)[0]
        except Exception as e:
            print(
                f"{e}. \n"
                f"Try to use VAE tiling for large images. For example: \n"
                f"pipe.vae.enable_tiling(tile_sample_min_width=512, tile_sample_min_height=512)"
            )
        video = self.video_processor.postprocess_video(video.detach(), output_type="np")[0]
        if video.dtype != np.uint8:
            video = (video * 255).clip(0, 255).astype(np.uint8)
        imageio.mimsave(save_name, video, fps=10)
    
    def forward(self, img_embeds, sc_embeds, act_embeds, target_imgs, actions):
        # prepare image
        prompt_embeds = torch.cat([img_embeds, sc_embeds, act_embeds], dim = 1)
        prompt_embeds = self.prompt_proj(prompt_embeds)
        device = prompt_embeds.device
        # target_imgs: 10 3 T H W
        # print(target_imgs.shape) # 224 * 224
        target_z = self.vae.encode(target_imgs).latent_dist.mode().to(device)
        # print(target_z.shape) # torch.Size([2, 16, 8, 28, 28])
        vae_mean = self.vae_mean.to(device=device)
        vae_std = self.vae_std.to(device=device)
        # follow https://github.com/NVlabs/Sana/blob/main/diffusion/model/wan/vae.py#L497
        target_z = (target_z - vae_mean.view(1, self.z_dim, 1, 1, 1)) / vae_std.view(1, self.z_dim, 1, 1, 1)
        # print(target_z.shape) # [10, 16, 8, 28, 28]
        clean_images = target_z
        bs = clean_images.shape[0]
        # for logit_normal
        u = torch.normal(mean=0.0, std=1.0, size=(bs,), device=device)
        u = torch.nn.functional.sigmoid(u)
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long().to(device=device)
        sch_timesteps = self.noise_scheduler.timesteps.to(device=device)
        timesteps = sch_timesteps[indices].to(device=device)
        noise = torch.randn_like(clean_images)
        # self.noise_scheduler.add_noise()
        sigmas = get_sigmas(self.noise_scheduler, timesteps, n_dim=clean_images.ndim, dtype=clean_images.dtype)
        # noisy_model_input = (1.0 - sigmas) * clean_images + sigmas * noise
        noisy_model_input = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        model_output = self.transformer(
            hidden_states=noisy_model_input,
            # encoder_attention_mask=prompt_attention_mask,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            return_dict=False,
            # mask_index = mask_index
        )
        video_pred = model_output
        # torch.Size([10, 16, 8, 28, 28]) torch.Size([10, 30, 2240])
        # calculate loss
        weighting = torch.ones_like(sigmas)
        video_target = noise - clean_images
        # Compute regular loss.
        video_loss = torch.mean(
            (weighting.float() * (video_pred.float() - video_target.float()) ** 2).reshape(video_target.shape[0], -1),
            1,
        )
        video_loss = video_loss.mean()
        # print(video_loss.shape, action_loss.shape)
        loss = {}
        loss["video_loss"] = video_loss
        loss["action_loss"] = torch.tensor([0.0, 0.0], device=device)
        
        
        # video_latents = noise - video_pred
        # # decode video
        
        # self.save_video(video_latents, "pred.mp4")
        # self.save_video(clean_images, "gt.mp4")
        # time.sleep(5)
        return loss
        
        
        
        
        
        