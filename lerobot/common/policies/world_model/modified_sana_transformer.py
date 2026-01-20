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


class Modified_Gated_SanaAttnProcessor2_0:
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

        query = attn.to_q(hidden_states)
        gate_score_video = attn.gate_g1_video(hidden_states)
        gate_score_action = attn.gate_g1_action(hidden_states)
        gate_score_video = gate_score_video[:, :num_image_token, :]
        gate_score_action = gate_score_action[:, num_image_token:, :]
        gate_score = torch.cat([gate_score_video, gate_score_action], dim=1)
        #print(gate_score.shape)

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
        
        hidden_states = hidden_states * torch.sigmoid(gate_score)

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
            # query, action_query = query[:, :num_image_token], query[:, num_image_token:]
            # key, action_key = key[:, :num_image_token], key[:, num_image_token:]
            
            query_rotate = apply_rotary_emb(query, *rotary_emb)
            key_rotate = apply_rotary_emb(key, *rotary_emb)
            
            # query_rotate = torch.cat([query_rotate, action_query], dim = 1)
            # key_rotate = torch.cat([key_rotate, action_key], dim = 1)
            # query = torch.cat([query, action_query], dim = 1)
            # key = torch.cat([key, action_query], dim = 1)

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
        # self.ff_action = MLP(
        #     in_dim=kwargs.get("dim", 2240),
        #     hidden_dim=kwargs.get("dim", 2240) * 2,
        #     out_dim=kwargs.get("dim", 2240),
        #     act=nn.GELU,
        #     drop=0.0
        # )
        
        # our design: cross attention between image latents, noised action
        # self.norm3 = nn.LayerNorm(kwargs.get("dim", 2240), 
        #                           elementwise_affine=kwargs.get("norm_elementwise_affine", False), 
        #                           eps=kwargs.get("norm_eps", 1e-6)
        #                           )

        self.action_video_fusion = action_video_fusion
        if self.action_video_fusion:
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
        else:
            self.attn3 = None
        
        
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
        
        num_image_token = rotary_emb[0].shape[1]

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
        
        if self.attn3 is not None:
            attn_output = self.attn3(
                hidden_states=hidden_states,
                attention_mask=attention_mask
            )
            hidden_states = attn_output + hidden_states

        # 3. Cross Attention
        if self.attn2 is not None:
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
            activation_fn="geglu",
            bias=True
        )

        self.attn1_for_action = Attention(
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
        
        # for image condition
        cross_attention_dim = kwargs.get("cross_attention_dim", None)
        dim = kwargs.get("dim", 2240)
        # self.zero_linear_1 = nn.Linear(cross_attention_dim, cross_attention_dim)
        # self.zero_linear = nn.Linear(cross_attention_dim, cross_attention_dim)
        # nn.init.zeros_(self.zero_linear.weight)
        # nn.init.zeros_(self.zero_linear.bias)
        # nn.init.zeros_(self.zero_linear_1.weight)
        # nn.init.zeros_(self.zero_linear_1.bias)
        # self.attn3 = copy.deepcopy(self.attn2)
        
        # add non-linear
        self.linear_attn_1 = nn.Sequential(
            nn.Linear(cross_attention_dim, cross_attention_dim // 4),
            nn.GELU(),
            nn.Linear(cross_attention_dim // 4, cross_attention_dim),
        )
        self.linear_attn_2 = nn.Sequential(
            nn.Linear(cross_attention_dim, cross_attention_dim // 2),
            nn.GELU(),
            nn.Linear(cross_attention_dim // 2, cross_attention_dim),
        )
        # self.linear_attn_fusion = nn.Linear(cross_attention_dim * 2, cross_attention_dim)
        # prevent nan
        self.gate_ca = nn.Parameter(torch.zeros(1, dim) / dim**0.5) 
        # gated attention for cross attention
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
                processor=Modified_Gated_SanaAttnProcessor2_0(),
        )
        self.attn2.gate_g1_action = nn.Linear(self.attn2.query_dim, self.attn2.inner_dim, bias=True)
        self.attn2.gate_g1_video = nn.Linear(self.attn2.query_dim, self.attn2.inner_dim, bias=True)
        
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        img_encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
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
        # gate_ca = self.gate_ca.unsqueeze(0).repeat(batch_size, 1, 1)

        # 2. Self Attention
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)
        norm_hidden_states, norm_action_hidden_states = norm_hidden_states[:, :num_image_token], norm_hidden_states[:, num_image_token:]
        attn_output = self.attn1(norm_hidden_states, rotary_emb=rotary_emb)
        
        attn_output_action = self.attn1_for_action(norm_action_hidden_states)
        # non-linear
        attn_output_action_1 = self.linear_attn_1(attn_output_action)
        attn_output_action_2 = self.linear_attn_2(attn_output_action)
        # attn_output_action = torch.cat([attn_output_action_1, attn_output_action_2], dim = -1)
        attn_output_action = (attn_output_action_1 + attn_output_action_2) / 2.0
        # attn_output_action = self.linear_attn_fusion(attn_output_action)
        
        attn_output = torch.cat([attn_output, attn_output_action], dim = 1)
        hidden_states = hidden_states + gate_msa * attn_output

        # 3. Cross Attention
        if self.attn2 is not None:
            # if self.action_video_fusion:
            #     encoder_attention_mask[:, :, :hidden_states.shape[1]] = float("-inf")
            encoder_hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim = 1)
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                num_image_token=num_image_token
            ) # very large lead to nan
            
            # print("hidden:", hidden_states.norm().item(), "attn2:", attn_output.norm().item(), "encoder:", encoder_hidden_states.norm().item())
            # attn_output = self.zero_linear_1(attn_output)
            
            # if self.attn3 is not None:
            #     img_attn_output = self.attn3(
            #         hidden_states,
            #         encoder_hidden_states=img_encoder_hidden_states
            #     )
            #     img_attn_output = self.zero_linear(img_attn_output)
            #     attn_output = attn_output * gate_ca + img_attn_output
            # attn_output_video = attn_output[:, :num_image_token]
            
            # hidden_states = attn_output * gate_ca + hidden_states
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