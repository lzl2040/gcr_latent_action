import torch.nn as nn
import torch
from torch import Tensor
from lerobot.common.policies.latent_action.image_decoder_sana_ip_adapter import ImagePredictionModel as SANAModel
from lerobot.common.policies.latent_action.action_decoder import PaliGemmaWithExpertConfig, ActionDecoderModel
from lerobot.common.utils.utils import get_safe_dtype
import math
from PIL import Image
from pytest import Cache
import torch.nn.functional as F

def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks

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

def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)

def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device = "cuda"):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def repeat_kv_for_gqa(key_states: torch.Tensor, value_states: torch.Tensor, num_att_heads: int, num_key_value_heads: int):
    """
    Expand key/value states for Grouped-Query Attention (GQA).

    Args:
        key_states: Tensor, shape [B, L, N_kv, D_h]
        value_states: Tensor, shape [B, L, N_kv, D_h]
        num_att_heads: total number of attention heads (N_q)
        num_key_value_heads: number of key/value heads (N_kv)

    Returns:
        key_states: Tensor, shape [B, L, N_q, D_h]
        value_states: Tensor, shape [B, L, N_q, D_h]
    """

    assert num_att_heads % num_key_value_heads == 0, \
        f"num_attention_heads ({num_att_heads}) must be divisible by num_key_value_heads ({num_key_value_heads})"

    num_key_value_groups = num_att_heads // num_key_value_heads
    B, L, _, D_h = key_states.shape

    # Expand along new group dimension, then merge it back
    key_states = key_states[:, :, :, None, :].expand(
        B, L, num_key_value_heads, num_key_value_groups, D_h
    ).reshape(B, L, num_att_heads, D_h)

    value_states = value_states[:, :, :, None, :].expand(
        B, L, num_key_value_heads, num_key_value_groups, D_h
    ).reshape(B, L, num_att_heads, D_h)

    return key_states, value_states


def sana_attention(query, key, value):
    query = F.relu(query)
    key = F.relu(key)
    query = query.to(dtype=torch.float32)
    key = key.to(dtype=torch.float32)
    value = value.to(dtype=torch.float32)

    key = key.transpose(2, 3)
    scores = torch.matmul(value, key)
    hidden_states = torch.matmul(scores, query)
    # print(torch.max(hidden_states), torch.min(hidden_states))

    hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + 1e-15)
    # print(torch.max(hidden_states), torch.min(hidden_states))
    hidden_states = hidden_states.flatten(1, 2).transpose(1, 2)
    return hidden_states

class UniDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        paligemma_with_expert_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
        )
        self.action_decoder = ActionDecoderModel(paligemma_with_expert_config, config.action_expert_path)

        # Projections are float32
        self.con_proj = nn.Linear(self.config.vlm_token_dim, self.config.img_dim)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        # image decoder
        # self.image_decoder = ImagePredictionModel(config)
        self.image_decoder = SANAModel(config)

        self.dtype = torch.bfloat16
        self.decoder_proj_type = config.ip_token_gen_type

    
    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=self.dtype,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=self.dtype, device=device)

    def prepare_condition_for_image_decoder(self, prompt_embs, cond_image):
        prompt_embs = self.image_decoder.con_proj(prompt_embs)
        device = prompt_embs.device
        clip_imgs = []
        bs = cond_image.shape[0]
        cond_image = cond_image.permute(0, 2, 3, 1)
        cond_image = cond_image.detach().cpu().numpy()
        for i in range(bs):
            img = cond_image[i]
            pil_img = Image.fromarray(img, mode="RGB")
            clip_img = self.image_decoder.clip_processor(images=pil_img, return_tensors="pt").pixel_values
            clip_imgs.append(clip_img)
        clip_imgs = torch.cat(clip_imgs, dim = 0).to(device=device)
        # for img_proj_model is clip
        if self.image_decoder.img_proj_type == "cls_proj":
            image_embs = self.image_decoder.image_encoder(clip_imgs).image_embeds # this is cls token embedding
        # for img_proj_model is resampler
        else:
            image_embs = self.image_decoder.image_encoder(clip_imgs, output_hidden_states=True).hidden_states[-1]
        # print(image_embeds.shape)
        ip_tokens = self.image_decoder.img_proj_model(image_embs) # torch.Size([25, 4, 1152])
        return prompt_embs, ip_tokens

    def prepare_noise_image_latent_for_image_decoder(self, image):
        image = image / 255
        image = self.image_decoder.transform(image)
        latents = self.image_decoder.vae.encode(image).latent
        latents = latents * self.image_decoder.vae_config_scaling_factor
        noise = torch.randn_like(latents)

        bsz = latents.shape[0]

        if self.image_decoder.weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            u = torch.normal(mean=self.image_decoder.logit_mean, std=self.image_decoder.logit_std, size=(bsz,), device="cpu")
            u = torch.nn.functional.sigmoid(u)
        elif self.image_decoder.weighting_scheme == "mode":
            u = torch.rand(size=(bsz,), device="cpu")
            u = 1 - u - self.image_decoder.mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=(bsz,), device="cpu")

        indices = (u * self.image_decoder.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.image_decoder.noise_scheduler.timesteps[indices].to(device=latents.device)

        sigmas = get_sigmas(self.image_decoder.noise_scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        noisy_latent = (1.0 - sigmas) * latents + sigmas * noise
        return noisy_latent, timesteps, sigmas, latents, noise

    def embed_image_latent_for_image_decoder(self, first_image, last_image, prompt_embs):
        noise_latent_image, timestep, sigmas, latents, noise = self.prepare_noise_image_latent_for_image_decoder(last_image)
        prompt_embs, ip_tokens = self.prepare_condition_for_image_decoder(prompt_embs, first_image)

        batch_size, num_channels, height, width = noise_latent_image.shape
        p = self.image_decoder.transformer.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        image_latent_embs = self.image_decoder.transformer.patch_embed(noise_latent_image)

        timestep, embedded_timestep = self.image_decoder.transformer.time_embed(
            timestep, batch_size=batch_size, hidden_dtype=image_latent_embs.dtype
        )

        prompt_embs = self.image_decoder.transformer.caption_projection(prompt_embs)
        prompt_embs = prompt_embs.view(batch_size, -1, image_latent_embs.shape[-1])

        prompt_embs = self.image_decoder.transformer.caption_norm(prompt_embs)
        return image_latent_embs, prompt_embs, timestep, ip_tokens, \
            post_patch_height, post_patch_width, embedded_timestep, sigmas, latents, noise


    def embed_prefix_for_action(self, con_embeddings):
        embs = []
        pad_masks = []
        att_masks = []
        bsize = con_embeddings.shape[0]

        # torch.Size([2, 256, 2048]) torch.Size([2, 66, 1024])
        # print(embs[0].shape, con_embeddings.shape)
        con_embeddings = self.con_proj(con_embeddings)
        embs.append(con_embeddings)
        num_con = con_embeddings.shape[1]
        att_masks += [0] * num_con
        con_mask = torch.ones(bsize, num_con, dtype=torch.bool, device=con_embeddings.device)
        pad_masks.append(con_mask)
        # print(pad_masks[0].shape, pad_masks[-1].shape)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks
    
    def embed_suffix_for_action(self, noisy_actions, timestep):
        embs = []
        pad_masks = []
        att_masks = []
        dtype = noisy_actions.dtype
        device = noisy_actions.device

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        # print(f"pad mask shape is: {pad_masks.shape}")
        # print(f"att mask shape is: {att_masks.shape}")
        return embs, pad_masks, att_masks
    
    def forward(self, first_image, last_image, sc_embedding, act_embeddings, 
        actions, action_noise=None, action_time=None):
        if action_noise is None:
            action_noise = self.sample_noise(actions.shape, actions.device).to(dtype=self.dtype)

        if action_time is None:
            action_time = self.sample_time(actions.shape[0], actions.device).to(dtype=self.dtype)

        action_time_expanded = action_time[:, None, None]
        actions = actions.to(dtype=self.dtype)
        x_t = action_time_expanded * action_noise + (1 - action_time_expanded) * actions
        u_t = action_noise - actions
        con_embeddings = torch.cat([sc_embedding, act_embeddings], dim = 1)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix_for_action(
            con_embeddings=con_embeddings
        )
        # torch.Size([2, 578, 2048])
        # print(prefix_embs.shape)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_for_action(x_t, action_time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1


        image_latent_embs, prompt_embs_for_id, t_for_id, ip_tokens_for_id, \
            h, w, embedded_timestep, sigmas, latents, noise = self.embed_image_latent_for_image_decoder(first_image=first_image,
                                                                                                                      last_image=last_image,
                                                                                                                      prompt_embs=sc_embedding)

        outputs, _ = self.forward_multi_model(inputs_embeds=[image_latent_embs, prefix_embs, suffix_embs],
                                 use_cache=False,
                                 fill_kv_cache=False,
                                 past_key_values=None,
                                 prompt_embs_for_id=prompt_embs_for_id,
                                 t_for_id=t_for_id,
                                 ip_tokens_for_id=ip_tokens_for_id,
                                 height=h,
                                 width=w,
                                 embedded_t=embedded_timestep,
                                 attention_mask=att_2d_masks)

        # calculate action loss
        suffix_out = outputs[-1][:, -self.config.n_action_steps :]
        # Original openpi code, upcast attention output
        # suffix_out = suffix_out.to(dtype=torch.float32)
        suffix_out = suffix_out.to(dtype=self.dtype)
        v_t = self.action_out_proj(suffix_out)

        losses = {}
        # print(u_t, v_t)
        losses["action_loss"] = F.mse_loss(u_t, v_t, reduction="none")
        # losses["action_loss"] = torch.tensor(0.0)

        # calculate img loss
        if self.image_decoder.weighting_scheme == "sigma_sqrt":
            weighting = (sigmas ** -2.0).float()
        elif self.image_decoder.weighting_scheme == "cosmap":
            bot = 1 - 2 * sigmas + 2 * sigmas ** 2
            weighting = 2 / (math.pi * bot)
        else:
            weighting = torch.ones_like(sigmas)

        target = noise - latents
        img_loss = torch.mean(
            (weighting.float() * (outputs[0].float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        img_loss = img_loss.mean()
        # losses["image_loss"] = torch.tensor(0.0)
        losses["image_loss"] = img_loss
        return losses

    def forward_multi_model(self, 
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        prompt_embs_for_id: torch.FloatTensor | None = None,
        t_for_id: torch.FloatTensor | None = None, 
        embedded_t: torch.FloatTensor | None = None,
        ip_tokens_for_id: torch.FloatTensor | None = None,
        height: int | None = None,
        width: int | None = None,
        attention_mask: torch.BoolTensor | None = None):

        encoder_hidden_states_for_id = torch.cat([prompt_embs_for_id, ip_tokens_for_id], dim = 1)
        
        encoder_attention_mask_for_id = torch.ones(encoder_hidden_states_for_id.shape[0], 
                                                   encoder_hidden_states_for_id.shape[1], 
                                                   dtype=torch.long, 
                                                   device=encoder_hidden_states_for_id.device)
        if inputs_embeds[0] is not None:
            if encoder_attention_mask_for_id is not None and encoder_attention_mask_for_id.ndim == 2:
                encoder_attention_mask_for_id = (1 - encoder_attention_mask_for_id.to(inputs_embeds[0].dtype)) * -10000.0
                encoder_attention_mask_for_id = encoder_attention_mask_for_id.unsqueeze(1)
        
        # print(encoder_hidden_states_for_id.shape, encoder_attention_mask_for_id.shape)
        
        models = [self.image_decoder.transformer, # 20
                  self.action_decoder.latent_action_layers,  # 18
                  self.action_decoder.gemma_expert.model] # 18
        
        for hidden_states in inputs_embeds:
            # print(hidden_states.shape)
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]
        
        num_layers = self.action_decoder.gemma_expert.config.num_hidden_layers
        head_dim = self.action_decoder.gemma_expert.config.head_dim
        num_att_heads = self.action_decoder.gemma_expert.config.num_attention_heads
        num_key_value_heads = self.action_decoder.gemma_expert.config.num_key_value_heads
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                if i == 0:
                    layer = models[i].transformer_blocks[layer_idx]
                    # 1. Modulation
                    print(layer.scale_shift_table[None].shape, t_for_id.shape)
                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                        layer.scale_shift_table[None] + t_for_id.reshape(batch_size, 6, -1)
                    ).chunk(6, dim=1)

                    # 2. Self Attention
                    norm_hidden_states = layer.norm1(hidden_states)
                    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                    norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

                    query = layer.attn1.to_q(norm_hidden_states)
                    key = layer.attn1.to_k(norm_hidden_states)
                    value = layer.attn1.to_v(norm_hidden_states)
                    query = query.transpose(1, 2).unflatten(1, (layer.attn1.heads, -1))
                    key = key.transpose(1, 2).unflatten(1, (layer.attn1.heads, -1))
                    value = value.transpose(1, 2).unflatten(1, (layer.attn1.heads, -1))
                    # torch.Size([8, 70, 32, 256]) torch.Size([8, 70, 256, 32]) torch.Size([8, 70, 32, 256])
                    # print(query.shape, key.shape, value.shape)
                    query, key, value = query.float(), key.float(), value.float()

                    value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1.0)
                    # print("image", torch.isnan(query).any(), torch.isnan(key).any(), torch.isnan(value).any())
                elif i == 2:
                    layer = models[i].layers[layer_idx]
                    # normalizer = torch.tensor(models[i].config.hidden_size**0.5, dtype=hidden_states.dtype)
                    # hidden_states = hidden_states * normalizer
                    # print(hidden_states.shape)
                    hidden_states = layer.input_layernorm(hidden_states)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    # print(hidden_shape)

                    hidden_states = hidden_states.to(dtype=torch.bfloat16)
                    # print(layer.self_attn.q_proj)
                    query = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                    key = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                    value = layer.self_attn.v_proj(hidden_states).view(hidden_shape)
                    key, value = repeat_kv_for_gqa(key, value, num_att_heads, num_key_value_heads)
                    query = query.permute(0, 2, 3, 1)
                    key = key.permute(0, 2, 3, 1)
                    value = value.permute(0, 2, 3, 1)
                    value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1.0)
                    # print("action", torch.isnan(query).any(), torch.isnan(key).any(), torch.isnan(value).any())
                    # repeat
                    # print(key_state.shape)
                # print(f"{i}", query_state.shape, key_state.shape, value_state.shape, hidden_states.shape)
                else:
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, head_dim)
                    layer = models[i][layer_idx]
                    query = layer.q_proj(hidden_states).view(hidden_shape)
                    key = layer.k_proj(hidden_states).view(hidden_shape)
                    value = layer.v_proj(hidden_states).view(hidden_shape)
                    key, value = repeat_kv_for_gqa(key, value, num_att_heads, num_key_value_heads)
                    query = query.permute(0, 2, 3, 1)
                    key = key.permute(0, 2, 3, 1)
                    value = value.permute(0, 2, 3, 1)
                    value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1.0)
                    # print("sc", torch.isnan(query).any(), torch.isnan(key).any(), torch.isnan(value).any())
                    # print("cond:", query_state.shape, key_state.shape, value_state.shape, hidden_states.shape)

                query_states.append(query)
                key_states.append(key)
                value_states.append(value)
                # print(i, query.shape, key.shape, value.shape)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=-1)
            key_states = torch.cat(key_states, dim=-1)
            value_states = torch.cat(value_states, dim=-1)
            
            
            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                else:
                    # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                    # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                    # the max len, then we (for instance) double the cache size. This implementation already exists
                    # in `transformers`. (molbap)
                    key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                    value_states = torch.cat(
                        [past_key_values[layer_idx]["value_states"], value_states], dim=1
                    )
            # print(torch.isnan(query_states).any(), torch.isnan(key_states).any(), torch.isnan(value_states).any())
            
            att_output = sana_attention(
                query_states, key_states, value_states
            ) # B L D

            # attention_interface = self.action_decoder.get_attention_interface()
            # att_output = attention_interface(
            #     attention_mask, batch_size, head_dim, query_states, key_states, value_states
            # )

            # print(torch.isnan(att_output).any())
            att_output = att_output.to(dtype=torch.bfloat16)
            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):

                if hidden_states is not None:
                    if i == 0:
                        end = start + hidden_states.shape[1]
                        layer = models[i].transformer_blocks[layer_idx]
                        feats = att_output[:, start:end]
                        out_emb = layer.attn1.to_out[0](feats)
                        out_emb = layer.attn1.to_out[1](out_emb)
                        hidden_states = hidden_states + gate_msa * out_emb

                        # cross-attention
                        num_tokens = layer.ip_token_num
                        end_pos = encoder_hidden_states_for_id.shape[1] - num_tokens
                        encoder_hidden_states, ip_encoder_hidden_states = (
                            encoder_hidden_states_for_id[:, :end_pos, :],
                            encoder_hidden_states_for_id[:, end_pos:, :],
                        )
                        encoder_attention_mask, ip_encoder_attention_mask = (
                            encoder_attention_mask_for_id[:, :, :end_pos],
                            encoder_attention_mask_for_id[:, :, end_pos:],
                        )
                        # print(encoder_attention_mask_for_id.shape, end_pos, encoder_hidden_states.shape, encoder_attention_mask.shape)

                        # 3. Cross Attention
                        if layer.attn2 is not None:
                            # print(hidden_states.shape, encoder_hidden_states.shape, encoder_attention_mask.shape)
                            output = layer.attn2(
                                hidden_states,
                                encoder_hidden_states=encoder_hidden_states,
                                attention_mask=encoder_attention_mask
                            )
                            # hidden_states = attn_output + hidden_states

                            if layer.is_ip_adapter:
                                # follow https://github.com/rotem154154/ControlNet-Sana/blob/main/models/finetuner_ip_adapter.py#L19
                                out_emb_ip = layer.attn3(
                                    hidden_states,
                                    encoder_hidden_states = ip_encoder_hidden_states,
                                    attention_mask=ip_encoder_attention_mask
                                )
                                out_emb_ip = layer.zero_linear(out_emb_ip)
                                # print(torch.max(attn_output), torch.max(attn_output_ip))
                                output = output + out_emb_ip
                        hidden_states = output + hidden_states
                        # 4. Feed-forward
                        norm_hidden_states = layer.norm2(hidden_states)
                        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

                        norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)
                        ff_output = layer.ff(norm_hidden_states)
                        ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
                        hidden_states = hidden_states + gate_mlp * ff_output
                        outputs_embeds.append(hidden_states)
                        # print(hidden_states.shape)
                        start = end
                    elif i == 2:
                        layer = models[i].layers[layer_idx]
                        end = start + hidden_states.shape[1]

                        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                        out_emb = layer.self_attn.o_proj(att_output[:, start:end])

                        # TODO: first dropout (by default 0.0)

                        # first residual
                        out_emb += hidden_states
                        after_first_residual = out_emb.clone()

                        out_emb = layer.post_attention_layernorm(out_emb)
                        out_emb = layer.mlp(out_emb)

                        # TODO: second dropout (by default 0.0)

                        # second residual
                        out_emb += after_first_residual

                        outputs_embeds.append(out_emb)

                        start = end
                    else:
                        end = start + hidden_states.shape[1]
                        outputs_embeds.append(hidden_states)
                        start = end
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        p = self.image_decoder.transformer.config.patch_size
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                if i == 0:
                    # 3. Normalization
                    hidden_states = models[i].norm_out(hidden_states, 
                                                       embedded_t, 
                                                       models[i].scale_shift_table)

                    hidden_states = models[i].proj_out(hidden_states)

                    # 5. Unpatchify
                    hidden_states = hidden_states.reshape(
                        batch_size, height, width, models[i].config.patch_size, models[i].config.patch_size, -1
                    )
                    hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
                    out_emb = hidden_states.reshape(batch_size, -1, height * p, width * p)
                elif i == 2:
                    out_emb = models[i].norm(hidden_states)
                else:
                    out_emb = hidden_states
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        
        return outputs_embeds, past_key_values

        
