import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from diffusers.models.transformers.transformer_sana_video import SanaVideoTransformerBlock
from diffusers.utils import deprecate, logging
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, LinearActivation, SwiGLU
import copy

def check_norm(name, tensor):
    # 计算平均 L2 范数（在 embedding 维度上）
    l2_norm = tensor.norm(p=2, dim=-1).mean().item()
    # 计算标准差，看分布是否过大
    std = tensor.std().item()
    print(f"[{name}] L2 Norm: {l2_norm:.4f} | Std: {std:.4f} | Shape: {list(tensor.shape)}")

class Modified_ExpertQKV_SanaAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("SanaAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_image_token: int = None
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # print(attention_mask.shape)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        hidden_len = hidden_states.shape[1]
        hidden_states, hidden_states_action = hidden_states[:, :num_image_token], hidden_states[:, num_image_token:]
        query = attn.to_q(hidden_states)
        query_action = attn.to_q_action(hidden_states_action)
        query = torch.cat([query, query_action], dim=1)
        
        #print(gate_score.shape)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        encoder_hidden_states_video, encoder_hidden_states_action = encoder_hidden_states[:, :num_image_token], encoder_hidden_states[:, num_image_token:hidden_len]
        # encoder_hidden_states_action = torch.zeros_like(encoder_hidden_states_action) # add this become normal
        encoder_hidden_condition = encoder_hidden_states[:, hidden_len:]
        # 在你的逻辑中插入
        # check_norm("Video Tokens ", encoder_hidden_states_video)
        # check_norm("Action Tokens", encoder_hidden_states_action)
        # check_norm("Condition Tkns", encoder_hidden_condition)
        encoder_hidden_states_action = F.layer_norm(
            encoder_hidden_states_action, 
            (encoder_hidden_states_action.size(-1),)
        )
        encoder_hidden_states_video = F.layer_norm(
            encoder_hidden_states_video, 
            (encoder_hidden_states_video.size(-1),)
        )
        
        key = attn.to_k(encoder_hidden_condition)
        value = attn.to_v(encoder_hidden_condition)
        key_video = attn.to_k(encoder_hidden_states_video)
        value_video = attn.to_v(encoder_hidden_states_video)
        
        key_action = attn.to_k_action(encoder_hidden_states_action)
        value_action = attn.to_v_action(encoder_hidden_states_action)
        # print(key_action.shape)
        key = torch.cat([key_video, key_action, key], dim=1)
        value = torch.cat([value_video, value_action, value], dim=1)
        # key = attn.to_k(encoder_hidden_states)
        # value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)
        
        # # (B, H, Lq, D)
        # q = query.permute(0, 2, 1, 3)

        # # (B, H, Lk, D)
        # k = key.permute(0, 2, 1, 3)

        # # (B, H, Lq, Lk)
        # attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        # # 对每个 query，取 top-10 key
        # topv, topk_idx = torch.topk(attn_scores, k=10, dim=-1)
        # print(topk_idx[0, 0, 0], num_image_token) # 784

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class Modified_SanaAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("SanaAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # print(attention_mask.shape)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class Modified_SanaLinearAttnProcessor3_0_Action:
    r"""
    Processor for implementing scaled dot-product linear attention.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        # B,N,H,C

        query = F.relu(query)
        key = F.relu(key)
        # B,H,C,N
        query = query.permute(0, 2, 3, 1)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 3, 1)

        query, key, value = query.float(), key.float(), value.float()

        z = 1 / (key.sum(dim=-1, keepdim=True).transpose(-2, -1) @ query + 1e-15)

        scores = torch.matmul(value, key.transpose(-1, -2))
        hidden_states = torch.matmul(scores, query)
        # print(z.shape, hidden_states.shape, query_rotate.shape)

        hidden_states = hidden_states * z
        # B,H,C,N
        hidden_states = hidden_states.flatten(1, 2).transpose(1, 2)
        hidden_states = hidden_states.to(original_dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class Modified_SanaLinearAttnProcessor3_0:
    r"""
    Processor for implementing scaled dot-product linear attention.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        num_image_token = rotary_emb[0].shape[1]
        # if rotary_emb is not None:
        #     num_image_token = rotary_emb[0].shape[1]
        #     hidden_states, action_hidden_states = hidden_states[:, :num_image_token], hidden_states[:, num_image_token:] 
        # num_action_token = hidden_states.shape[1] - num_image_token
        # # num_action_token = rotary_emb_action[0].shape[1]
        # print(f"Action token:{num_action_token} Image token:{num_image_token}")

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        # B,N,H,C

        query = F.relu(query)
        key = F.relu(key)

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)
            query, action_query = query[:, :num_image_token], query[:, num_image_token:]
            key, action_key = key[:, :num_image_token], key[:, num_image_token:]
            
            query_rotate = apply_rotary_emb(query, *rotary_emb)
            key_rotate = apply_rotary_emb(key, *rotary_emb)
            
            query_rotate = torch.cat([query_rotate, action_query], dim = 1)
            key_rotate = torch.cat([key_rotate, action_key], dim = 1)
            query = torch.cat([query, action_query], dim = 1)
            key = torch.cat([key, action_query], dim = 1)

        # B,H,C,N
        query = query.permute(0, 2, 3, 1)
        key = key.permute(0, 2, 3, 1)
        query_rotate = query_rotate.permute(0, 2, 3, 1)
        key_rotate = key_rotate.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 3, 1)

        query_rotate, key_rotate, value = query_rotate.float(), key_rotate.float(), value.float()

        z = 1 / (key.sum(dim=-1, keepdim=True).transpose(-2, -1) @ query + 1e-15)

        scores = torch.matmul(value, key_rotate.transpose(-1, -2))
        hidden_states = torch.matmul(scores, query_rotate)
        # print(z.shape, hidden_states.shape, query_rotate.shape)

        hidden_states = hidden_states * z
        # B,H,C,N
        hidden_states = hidden_states.flatten(1, 2).transpose(1, 2)
        hidden_states = hidden_states.to(original_dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # add
        # hidden_states = torch.cat([hidden_states, action_hidden_states], dim=1)

        return hidden_states


class GLUMBTempConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4,
        norm_type: Optional[str] = None,
        residual_connection: bool = True,
    ) -> None:
        super().__init__()

        hidden_channels = int(expand_ratio * in_channels)
        self.norm_type = norm_type
        self.residual_connection = residual_connection

        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(in_channels, hidden_channels * 2, 1, 1, 0)
        self.conv_depth = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, 1, 1, groups=hidden_channels * 2)
        self.conv_point = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)

        self.norm = None
        if norm_type == "rms_norm":
            self.norm = RMSNorm(out_channels, eps=1e-5, elementwise_affine=True, bias=True)

        self.conv_temp = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.residual_connection:
            residual = hidden_states
        batch_size, num_frames, height, width, num_channels = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size * num_frames, height, width, num_channels).permute(0, 3, 1, 2)

        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)

        hidden_states = self.conv_point(hidden_states)

        # Temporal aggregation
        hidden_states_temporal = hidden_states.view(batch_size, num_frames, num_channels, height * width).permute(
            0, 2, 1, 3
        )
        hidden_states = hidden_states_temporal + self.conv_temp(hidden_states_temporal)
        hidden_states = hidden_states.permute(0, 2, 3, 1).view(batch_size, num_frames, height, width, num_channels)

        if self.norm_type == "rms_norm":
            # move channel to the last dimension so we apply RMSnorm across channel dimension
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states

class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "linear-silu":
            act_fn = LinearActivation(dim, inner_dim, bias=bias, activation="silu")

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

class MLP(nn.Module):
    """Multilayer perceptron with two hidden layers."""

    def __init__(self, in_dim, hidden_dim, out_dim, act=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Modified_SanaVideoTransformerBlock(SanaVideoTransformerBlock):
    r"""
    Transformer block introduced in [Sana-Video](https://huggingface.co/papers/2509.24695).
    """

    def __init__(
        self,
        action_video_fusion,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.attn1 = Attention(
            query_dim=kwargs.get("dim", 2240),
            heads=kwargs.get("num_attention_heads", 20),
            dim_head=kwargs.get("attention_head_dim", 112),
            kv_heads=kwargs.get("num_attention_heads", 20) if kwargs.get("qk_norm", "rms_norm_across_heads") is not None else None,
            qk_norm=kwargs.get("qk_norm", "rms_norm_across_heads"),
            dropout=kwargs.get("dropout", 0.0),
            bias=kwargs.get("attention_bias", True),
            cross_attention_dim=None,
            processor=Modified_SanaLinearAttnProcessor3_0(),
        )
        self.ff = GLUMBTempConv(
            kwargs.get("dim", 2240), 
            kwargs.get("dim", 2240), 
            kwargs.get("mlp_ratio", 3.0), 
            norm_type=None, 
            residual_connection=False
        )
        self.ff_action = FeedForward(
            dim=kwargs.get("dim", 2240),
            dropout=0.0,
            final_dropout=0.0,
            activation_fn="geglu",
            bias=True
        )
        
        
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        img_encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        embedded_timestep: Optional[torch.LongTensor] = None,
        frames: int = None,
        height: int = None,
        width: int = None,
        rotary_emb: Optional[torch.Tensor] = None,
        rotary_emb_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        
        num_image_token = rotary_emb[0].shape[1]
        encoder_hidden_states = torch.cat([img_encoder_hidden_states, encoder_hidden_states], dim = 1)

        # 1. Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None] + timestep.reshape(batch_size, timestep.shape[1], 6, -1)
        ).unbind(dim=2)

        # 2. Self Attention
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

        attn_output = self.attn1(norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = hidden_states + gate_msa * attn_output

        # 3. Cross Attention
        if self.attn2 is not None:
            encoder_hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim = 1)
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        # preprocess
        norm_hidden_states, norm_hidden_states_action = norm_hidden_states[:, :num_image_token], norm_hidden_states[:, num_image_token:]
        # print(norm_hidden_states.shape)
        norm_hidden_states = norm_hidden_states.unflatten(1, (frames, height, width))
        # print(norm_hidden_states.shape)
        ff_output = self.ff(norm_hidden_states)
        # need a ffn layer for action
        # print(norm_hidden_states_action.shape)
        ff_action_output = self.ff_action(norm_hidden_states_action)
        
        ff_output = ff_output.flatten(1, 3)
        ff_output = torch.cat([ff_output, ff_action_output], dim = 1)
        
        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states
        

class Modified_SanaVideoTransformerBlock_V2(SanaVideoTransformerBlock):
    r"""
    Transformer block introduced in [Sana-Video](https://huggingface.co/papers/2509.24695).
    """

    def __init__(
        self,
        action_video_fusion,
        **kwargs
    ):
        super().__init__(**kwargs)
        # self.attn1 = Attention(
        #     query_dim=kwargs.get("dim", 2240),
        #     heads=kwargs.get("num_attention_heads", 20),
        #     dim_head=kwargs.get("attention_head_dim", 112),
        #     kv_heads=kwargs.get("num_attention_heads", 20) if kwargs.get("qk_norm", "rms_norm_across_heads") is not None else None,
        #     qk_norm=kwargs.get("qk_norm", "rms_norm_across_heads"),
        #     dropout=kwargs.get("dropout", 0.0),
        #     bias=kwargs.get("attention_bias", True),
        #     cross_attention_dim=None,
        #     processor=Modified_SanaLinearAttnProcessor3_0(),
        # )
        self.ff = GLUMBTempConv(
            kwargs.get("dim", 2240), 
            kwargs.get("dim", 2240), 
            kwargs.get("mlp_ratio", 3.0), 
            norm_type=None, 
            residual_connection=False
        )
        self.ff_action = FeedForward(
            dim=kwargs.get("dim", 2240),
            dropout=0.0,
            final_dropout=0.0,
            activation_fn="gelu",
            bias=True
        )

        self.attn1_for_action = Attention(
            query_dim=kwargs.get("dim", 2240),
            qk_norm=kwargs.get("qk_norm", "rms_norm_across_heads"),
            kv_heads=kwargs.get("num_cross_attention_heads", 20),
            cross_attention_dim=None,
            heads=kwargs.get("num_attention_heads", 20),
            dim_head=kwargs.get("attention_head_dim", 112),
            dropout=kwargs.get("dropout", 0.0),
            bias=True,
            out_bias=kwargs.get("attention_out_bias", True),
            processor=Modified_SanaAttnProcessor2_0(),
        )
        
        # for image condition
        cross_attention_dim = kwargs.get("cross_attention_dim", None)
        dim = kwargs.get("dim", 2240)
        
        # add non-linear
        # self.linear_attn_1 = nn.Sequential(
        #     nn.Linear(cross_attention_dim, cross_attention_dim // 4),
        #     nn.GELU(),
        #     nn.Linear(cross_attention_dim // 4, cross_attention_dim),
        # )
        # self.linear_attn_2 = nn.Sequential(
        #     nn.Linear(cross_attention_dim, cross_attention_dim // 2),
        #     nn.GELU(),
        #     nn.Linear(cross_attention_dim // 2, cross_attention_dim),
        # )
        # self.linear_attn_fusion = nn.Linear(cross_attention_dim * 2, cross_attention_dim)
        # self.gate_ca = nn.Parameter(torch.ones(1, dim) / dim**0.5) 
        self.attn2 = Attention(
            query_dim=kwargs.get("dim", 2240),
            qk_norm=kwargs.get("qk_norm", "rms_norm_across_heads"),
            kv_heads=kwargs.get("num_cross_attention_heads", 20),
            cross_attention_dim=kwargs.get("cross_attention_dim", 2240),
            heads=kwargs.get("num_attention_heads", 20),
            dim_head=kwargs.get("attention_head_dim", 112),
            dropout=kwargs.get("dropout", 0.0),
            bias=True,
            out_bias=kwargs.get("attention_out_bias", True),
            processor=Modified_ExpertQKV_SanaAttnProcessor2_0(),
            # processor=Modified_SanaAttnProcessor2_0()
        )
        self.attn2.to_q_action = nn.Linear(self.attn2.query_dim, self.attn2.inner_dim, bias=True)
        self.attn2.to_k_action = nn.Linear(self.attn2.cross_attention_dim, self.attn2.inner_kv_dim, bias=True)
        self.attn2.to_v_action = nn.Linear(self.attn2.cross_attention_dim, self.attn2.inner_kv_dim, bias=True)
        self.action_adaln = nn.Linear(dim, 6 * dim, bias=True)
        self.silu = nn.SiLU()
        
    # wo action adaln params
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     encoder_hidden_states: Optional[torch.Tensor] = None,
    #     img_encoder_hidden_states: Optional[torch.Tensor] = None,
    #     encoder_attention_mask: Optional[torch.Tensor] = None,
    #     timestep: Optional[torch.LongTensor] = None,
    #     embedded_timestep: Optional[torch.LongTensor] = None,
    #     frames: int = None,
    #     height: int = None,
    #     width: int = None,
    #     rotary_emb: Optional[torch.Tensor] = None,
    #     rotary_emb_action: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     batch_size = hidden_states.shape[0]
        
    #     num_image_token = rotary_emb[0].shape[1]
    #     encoder_hidden_states = torch.cat([img_encoder_hidden_states, encoder_hidden_states], dim = 1)

    #     # 1. Modulation
    #     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
    #         self.scale_shift_table[None, None] + timestep.reshape(batch_size, timestep.shape[1], 6, -1)
    #     ).unbind(dim=2)
    #     # 2. Self Attention
    #     norm_hidden_states = self.norm1(hidden_states)
    #     norm_hidden_states, norm_action_hidden_states = norm_hidden_states[:, :num_image_token], norm_hidden_states[:, num_image_token:]
        
    #     norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
    #     norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)
        
    #     norm_action_hidden_states = norm_action_hidden_states.to(hidden_states.dtype)
        
    #     attn_output = self.attn1(norm_hidden_states, rotary_emb=rotary_emb)
        
    #     attn_output_action = self.attn1_for_action(norm_action_hidden_states)
    #     attn_output = gate_msa * attn_output
        
    #     attn_output = torch.cat([attn_output, attn_output_action], dim = 1)
    #     hidden_states = hidden_states + attn_output

    #     # 3. Cross Attention
    #     if self.attn2 is not None:
    #         # if self.action_video_fusion:
    #         #     encoder_attention_mask[:, :, :hidden_states.shape[1]] = float("-inf")
    #         encoder_hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim = 1)
    #         attn_output = self.attn2(
    #             hidden_states,
    #             encoder_hidden_states=encoder_hidden_states,
    #             attention_mask=encoder_attention_mask,
    #             # num_image_token=num_image_token
    #         ) # very large lead to nan: maybe action tend to fuse video
            
    #         hidden_states = attn_output + hidden_states

    #     # 4. Feed-forward
    #     norm_hidden_states = self.norm2(hidden_states)
    #     norm_hidden_states, norm_action_hidden_states = norm_hidden_states[:, :num_image_token], norm_hidden_states[:, num_image_token:]
    #     norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        
    #     norm_hidden_states = norm_hidden_states.unflatten(1, (frames, height, width))
    #     ff_output = self.ff(norm_hidden_states)
    #     # need a ffn layer for action
    #     ff_action_output = self.ff_action(norm_action_hidden_states)
        
    #     ff_output = ff_output.flatten(1, 3)
        
    #     ff_output = ff_output * gate_mlp
        
    #     ff_output = torch.cat([ff_output, ff_action_output], dim = 1)
        
    #     hidden_states = hidden_states + ff_output

    #     return hidden_states
    
    # action ada params
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        img_encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        embedded_timestep: Optional[torch.LongTensor] = None,
        frames: int = None,
        height: int = None,
        width: int = None,
        rotary_emb: Optional[torch.Tensor] = None,
        rotary_emb_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        
        num_image_token = rotary_emb[0].shape[1]
        # encoder_hidden_states = torch.cat([img_encoder_hidden_states, encoder_hidden_states], dim = 1)

        # 1. Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None] + timestep.reshape(batch_size, timestep.shape[1], 6, -1)
        ).unbind(dim=2)
        
        action_ada_params = self.silu(self.action_adaln(embedded_timestep).reshape(batch_size, timestep.shape[1], 6, -1))
        # print(action_ada_params.shape) # 4 1 6 2240
        shift_msa_action, scale_msa_action, gate_msa_action, shift_mlp_action, scale_mlp_action, gate_mlp_action = action_ada_params.unbind(dim = 2)
        # print(self.scale_shift_table[0, :10])
        # gate_ca = self.gate_ca.unsqueeze(0).repeat(batch_size, 1, 1)

        # 2. Self Attention
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states, norm_action_hidden_states = norm_hidden_states[:, :num_image_token], norm_hidden_states[:, num_image_token:]
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)
        
        norm_action_hidden_states = norm_action_hidden_states * (1 + scale_msa_action) + shift_msa_action
        norm_action_hidden_states = norm_action_hidden_states.to(hidden_states.dtype)
        
        attn_output = self.attn1(norm_hidden_states, rotary_emb=rotary_emb)
        
        attn_output_action = self.attn1_for_action(norm_action_hidden_states)
        # non-linear
        # attn_output_action_1 = self.linear_attn_1(attn_output_action)
        # attn_output_action_2 = self.linear_attn_2(attn_output_action)
        # # attn_output_action = torch.cat([attn_output_action_1, attn_output_action_2], dim = -1)
        # attn_output_action = (attn_output_action_1 + attn_output_action_2) / 2.0
        # attn_output_action = self.linear_attn_fusion(attn_output_action)
        attn_output = gate_msa * attn_output
        attn_output_action = gate_msa_action * attn_output_action
        
        attn_output = torch.cat([attn_output, attn_output_action], dim = 1)
        hidden_states = hidden_states + attn_output

        # 3. Cross Attention
        if self.attn2 is not None:
            # if self.action_video_fusion:
            #     encoder_attention_mask[:, :, :hidden_states.shape[1]] = float("-inf")
            # print(torch.max(hidden_states[:, :num_image_token]), torch.min(hidden_states[:, :num_image_token]), torch.max(hidden_states[:, num_image_token:]), torch.min(hidden_states[:, num_image_token:]))
            encoder_hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim = 1)
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                num_image_token=num_image_token
            ) # very large lead to nan: maybe action tend to fuse video
            # hidden_states = attn_output + hidden_states
            # hidden_states = torch.sigmoid(attn_output) * hidden_states + hidden_states
            # hidden_states = attn_output * gate_ca + hidden_states
            # no sense: 2.3
            # if self.attn3 is not None:
            #     attn_output_2 = self.attn3(
            #         hidden_states,
            #         attention_mask=attention_mask,
            #     )
            #     attn_output_2 = self.zero_linear(attn_output_2)
            #     attn_output = attn_output + attn_output_2
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states, norm_action_hidden_states = norm_hidden_states[:, :num_image_token], norm_hidden_states[:, num_image_token:]
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        
        norm_action_hidden_states = norm_action_hidden_states * (1 + scale_mlp_action) + shift_mlp_action
        
        # print(norm_hidden_states.shape)
        norm_hidden_states = norm_hidden_states.unflatten(1, (frames, height, width))
        ff_output = self.ff(norm_hidden_states)
        # need a ffn layer for action
        ff_action_output = self.ff_action(norm_action_hidden_states)
        
        ff_output = ff_output.flatten(1, 3)
        
        ff_output = ff_output * gate_mlp
        ff_action_output = ff_action_output * gate_mlp_action
        
        ff_output = torch.cat([ff_output, ff_action_output], dim = 1)
        
        hidden_states = hidden_states + ff_output

        return hidden_states
        
class Modified_SanaVideoTransformerBlock_Action(SanaVideoTransformerBlock):
    r"""
    Transformer block introduced in [Sana-Video](https://huggingface.co/papers/2509.24695).
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.attn1 = Attention(
            query_dim=kwargs.get("dim", 2240),
            heads=kwargs.get("num_attention_heads", 20),
            dim_head=kwargs.get("attention_head_dim", 112),
            kv_heads=kwargs.get("num_attention_heads", 20) if kwargs.get("qk_norm", "rms_norm_across_heads") is not None else None,
            qk_norm=kwargs.get("qk_norm", "rms_norm_across_heads"),
            dropout=kwargs.get("dropout", 0.0),
            bias=kwargs.get("attention_bias", True),
            cross_attention_dim=None,
            processor=Modified_SanaLinearAttnProcessor3_0_Action(),
        )
        # self.ff = GLUMBTempConv(
        #     kwargs.get("dim", 2240), 
        #     kwargs.get("dim", 2240), 
        #     kwargs.get("mlp_ratio", 3.0), 
        #     norm_type=None, 
        #     residual_connection=False
        # )
        self.ff_action = FeedForward(
            dim=kwargs.get("dim", 2240),
            dropout=0.0,
            final_dropout=0.0,
            activation_fn="geglu",
            bias=True
        )
        # our design: cross attention between image latents, noised action
        # self.norm3 = nn.LayerNorm(kwargs.get("dim", 2240), 
        #                           elementwise_affine=kwargs.get("norm_elementwise_affine", False), 
        #                           eps=kwargs.get("norm_eps", 1e-6)
        #                           )
        
        
        self.attn3 = Attention(
                query_dim=kwargs.get("dim", 2240),
                qk_norm=kwargs.get("qk_norm", "rms_norm_across_heads"),
                kv_heads=kwargs.get("num_cross_attention_heads", 20),
                cross_attention_dim=kwargs.get("cross_attention_dim", 2240),
                heads=kwargs.get("num_attention_heads", 20),
                dim_head=kwargs.get("attention_head_dim", 112),
                dropout=kwargs.get("dropout", 0.0),
                bias=True,
                out_bias=kwargs.get("attention_out_bias", True),
                processor=Modified_SanaAttnProcessor2_0(),
        )
        self.attn2_action = copy.deepcopy(self.attn2)
        
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        frames: int = None,
        height: int = None,
        width: int = None,
        rotary_emb: Optional[torch.Tensor] = None,
        rotary_emb_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # 1. Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None] + timestep.reshape(batch_size, timestep.shape[1], 6, -1)
        ).unbind(dim=2)

        # 2. Self Attention
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

        # attn_output = self.attn1(norm_hidden_states) # no rotary for action
        # hidden_states = hidden_states + gate_msa * attn_output

        # 3. Cross Attention
        if self.attn3 is not None:
            attn_output = self.attn3(
                hidden_states=norm_hidden_states,
                attention_mask=attention_mask
            )
            hidden_states = attn_output + hidden_states
        
        if self.attn2_action is not None:
            attn_output = self.attn2_action(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        # preprocess
        ff_output = self.ff_action(norm_hidden_states)
        
        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states

class Modified_WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor, action_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [self.t_dim, self.h_dim, self.w_dim] # 40 36 36
        # print(split_sizes, self.freqs_cos.shape) # 40 36 36, 1024 112

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)
        # print(freqs_cos[0].shape) # 1024 40

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        
        freqs_cos_action = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).reshape(1, ppf, 1, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        
        freqs_sin_action = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).reshape(1, ppf, 1, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        
        # print(freqs_cos.shape) # 1 1568 1 112
        return freqs_cos, freqs_sin, freqs_cos_action, freqs_sin_action

class Modified_SanaModulatedNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6, inner_dim: int = None):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=eps)
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        shift, scale = (self.scale_shift_table[None, None] + temb[:, :, None].to(self.scale_shift_table.device)).unbind(dim=2)
        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states

class AdaLNTime(nn.Module):
    def __init__(self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6, inner_dim: int = None):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=eps)
        self.linear = nn.Linear(dim, 2 * dim, bias=True)
        self.act = nn.SiLU()
    
    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        batch_size = hidden_states.shape[0]
        shift, scale = self.act(self.linear(temb)).reshape(batch_size, temb.shape[1], 2, -1).unbind(dim=2)
        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states