"""Action Chunk Encoder (ACE) Model.

This module encodes action chunks into compact embeddings using:
1. Padding action dimensions to 32
2. Grouping actions (G actions per group)
3. Projecting to hidden dimension
4. Bidirectional self-attention with RoPE (similar to BERT)
5. Output compact embedding
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .configuration_robo_clip import ACEConfig
except ImportError:
    from lerobot.common.policies.ace.configuration_robo_clip import ACEConfig


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_position: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos and sin
        self._set_cos_sin_cache(max_position)
    
    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper but similar implementation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get RoPE embeddings for given sequence length.
        
        Args:
            x: Input tensor (unused, kept for API compatibility)
            seq_len: Sequence length
            
        Returns:
            Tuple of (cos, sin) embeddings of shape (seq_len, dim)
        """
        return (
            self.cos_cached[:seq_len].unsqueeze(0),
            self.sin_cached[:seq_len].unsqueeze(0)
        )


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor.
    
    Args:
        x: Input tensor of shape (batch, seq_len, num_heads, head_dim)
        cos: Cosine embeddings of shape (1, seq_len, head_dim)
        sin: Sine embeddings of shape (1, seq_len, head_dim)
        
    Returns:
        Tensor with rotary embeddings applied
    """
    # x: (batch, seq_len, num_heads, head_dim)
    # cos, sin: (1, seq_len, head_dim)
    
    # Split x into two halves
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    
    # Rotate
    # For the rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    cos = cos.unsqueeze(2)  # (1, seq_len, 1, head_dim)
    sin = sin.unsqueeze(2)  # (1, seq_len, 1, head_dim)
    
    # Need to match dimensions
    cos_half = cos[..., : x1.shape[-1]]
    sin_half = sin[..., : x2.shape[-1]]
    
    rotated_x1 = x1 * cos_half - x2 * sin_half
    rotated_x2 = x1 * sin_half + x2 * cos_half
    
    return torch.cat([rotated_x1, rotated_x2], dim=-1)


class ACEAttention(nn.Module):
    """Multi-head self-attention with RoPE."""
    
    def __init__(self, config: ACEConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_dim // config.num_attention_heads
        self.hidden_dim = config.hidden_dim
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        return self.o_proj(attn_output)


class ACEMLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, config: ACEConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.fc2 = nn.Linear(config.intermediate_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ACELayer(nn.Module):
    """Single transformer layer with self-attention and FFN."""
    
    def __init__(self, config: ACEConfig):
        super().__init__()
        self.attention = ACEAttention(config)
        self.mlp = ACEMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, cos, sin, attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class ActionChunkEncoder(nn.Module):
    """Action Chunk Encoder (ACE) model.
    
    Encodes action chunks into compact embeddings using a transformer architecture
    with rotary position embeddings.
    
    Args:
        config: ACEConfig object containing model hyperparameters
    """
    
    def __init__(self, config: ACEConfig):
        super().__init__()
        self.config = config
        
        # Action dimension padding
        self.action_dim = config.action_dim
        self.action_dim_padded = config.max_action_dim
        self.chunk_size = config.chunk_size
        self.group_size = config.group_size
        self.hidden_dim = config.hidden_dim
        
        # Calculate grouped dimensions
        # After grouping G actions: (chunk_size / G) groups, each with G * action_dim_padded dims
        assert config.chunk_size % config.group_size == 0, \
            f"chunk_size ({config.chunk_size}) must be divisible by group_size ({config.group_size})"
        
        self.num_groups = config.chunk_size // config.group_size
        self.group_dim = config.group_size * config.max_action_dim
        
        # Input projection: from group_dim to hidden_dim
        self.input_proj = nn.Linear(self.group_dim, config.hidden_dim)
        
        # RoPE
        self.rope = RotaryPositionEmbedding(
            config.hidden_dim // config.num_attention_heads,
            config.max_position_embeddings
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([ACELayer(config) for _ in range(config.num_hidden_layers)])
        
        # Output projection
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Sample rate embedding (optional positional info)
        self.sample_rate_embed = nn.Embedding(50, config.hidden_dim)
        
    def _pad_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Pad action dimensions to action_dim_padded.
        
        Args:
            actions: Action tensor of shape (batch, chunk_size, action_dim)
            
        Returns:
            Padded actions of shape (batch, chunk_size, action_dim_padded)
        """
        if self.action_dim >= self.action_dim_padded:
            return actions[..., :self.action_dim_padded]
        
        padding = torch.zeros(
            *actions.shape[:-1],
            self.action_dim_padded - self.action_dim,
            device=actions.device,
            dtype=actions.dtype
        )
        return torch.cat([actions, padding], dim=-1)
    
    def _group_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Group actions along chunk dimension.
        
        Args:
            actions: Padded actions of shape (batch, chunk_size, action_dim_padded)
            
        Returns:
            Grouped actions of shape (batch, num_groups, group_size * action_dim_padded)
        """
        batch_size = actions.shape[0]
        
        # Reshape: (batch, chunk_size, action_dim_padded) -> 
        #          (batch, num_groups, group_size, action_dim_padded)
        # print(actions.shape)
        actions = actions.view(batch_size, self.num_groups, self.group_size, self.action_dim_padded)
        
        # Flatten last two dims: (batch, num_groups, group_size * action_dim_padded)
        actions = actions.view(batch_size, self.num_groups, self.group_dim)
        
        return actions
    
    def forward(
        self,
        actions: torch.Tensor,
        sample_rate: int = 0,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode action chunks into compact embeddings.
        
        Args:
            actions: Action chunk tensor of shape (batch, chunk_size, action_dim)
            sample_rate: Sample rate index for positional embedding (0-9)
            attention_mask: Optional attention mask
            
        Returns:
            Compact embedding of shape (batch, output_dim)
        """
        batch_size = actions.shape[0] # (B, chunk_size, action_dim), has pad
        
        # Step 2: Group actions along chunk dimension
        actions = self._group_actions(actions)  # (B, num_groups, group_dim)
        
        # Step 3: Concatenate sample rate embedding at the beginning (position 0)
        sample_rate_embed = self.sample_rate_embed(sample_rate.long())
        
        # Step 4: Project actions to hidden dimension
        hidden_states = self.input_proj(actions)  # (B, num_groups, hidden_dim)
        
        # Step 5: Concatenate sample rate embedding at the beginning along chunk dimension
        # (B, num_groups + 1, hidden_dim), where position 0 is sample_rate_embed
        hidden_states = torch.cat([sample_rate_embed.unsqueeze(1), hidden_states], dim=1)
        
        # Step 6: Get RoPE embeddings
        cos, sin = self.rope(hidden_states, hidden_states.shape[1])
        
        # Step 7: Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attention_mask)
        
        # Step 8: Output projection
        hidden_states = self.output_norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)
        
        # Step 9: Take the output at sample_rate_embed position (position 0)
        embedding = hidden_states[:, 0, :]  # (B, output_dim)
        
        return embedding
