"""RobotCLIP model for aligning action embeddings with vision embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from torch import Tensor, nn
from transformers import SiglipModel, AutoProcessor
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.ace.configuration_robo_clip import ACEConfig, RobotCLIPConfig
from lerobot.common.policies.ace.modeling_ace import ActionChunkEncoder
from collections import deque
from PIL import Image
import math
from torch import distributed as dist
from lerobot.common.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
    OBS_ROBOT
)
import torch.distributed.nn.functional as dist_nn


def concat_all_gather(tensor):
    """
    这是一个既稳定又支持梯度的 gather 方式。
    """
    with torch.no_grad():
        world_size = dist.get_world_size()
        tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensors_gather, tensor)
    
    # 关键点：将本卡的 tensor 替换回去，保留本卡的梯度链条
    rank = dist.get_rank()
    tensors_gather[rank] = tensor
    
    # 返回拼接后的结果
    return torch.cat(tensors_gather, dim=0)

class SmallConvBottleneck(nn.Module):
    """
    Small bottleneck:
    [B, D, H, W] -> [B, C_out, H_out, W_out]
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        out_hw: Tuple[int, int],
    ):
        super().__init__()
        self.out_hw = out_hw

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=min(8, mid_channels), num_channels=mid_channels),
            nn.SiLU(),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=min(8, mid_channels), num_channels=mid_channels),
            nn.SiLU(),

            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, H, W]
        x = self.net(x)
        x = F.adaptive_avg_pool2d(x, self.out_hw)  # force target shape
        return x


class TokenFusionHead(nn.Module):
    """
    Fuse original cls token and pooled dense token.
    """
    def __init__(self, cls_dim: int, pooled_dim: int, output_dim: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.LayerNorm(cls_dim + pooled_dim),
            nn.Linear(cls_dim + pooled_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, cls_token: torch.Tensor, pooled_token: torch.Tensor) -> torch.Tensor:
        # cls_token:   [B, cls_dim]
        # pooled_token:[B, pooled_dim]
        x = torch.cat([cls_token, pooled_token], dim=-1)
        return self.fuse(x)


class VisionEncoder(nn.Module):
    """
    Vision encoder using SigLIP2 or similar CLIP-like model.

    Outputs:
        - final_token: fused token for action alignment
        - vae_feature: dense VAE-like feature map
        - cls_token: original cls token from SigLIP
        - pooled_token: avg pooled token from dense feature
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        output_dim: int = 768,
        vae_shape: Tuple[int, int, int] = (16, 28, 28),   # (C_out, H_out, W_out) # for image_size=224, wan 2.1 vae
        bottleneck_mid: int = 256,
        dtype=torch.bfloat16,
    ):
        super().__init__()

        self.model = SiglipModel.from_pretrained(
            model_name,
            dtype=torch.float32
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        # hidden size
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "hidden_size"):
                vision_hidden_size = self.model.config.hidden_size
            elif hasattr(self.model.config, "vision_config"):
                vision_hidden_size = self.model.config.vision_config.hidden_size
            else:
                vision_hidden_size = 768
        else:
            vision_hidden_size = 768

        self.hidden_size = vision_hidden_size
        self.dtype = dtype

        # target dense latent shape
        self.vae_channels, self.vae_h, self.vae_w = vae_shape

        # projection for cls token if needed
        if vision_hidden_size != output_dim:
            self.cls_projection = nn.Linear(vision_hidden_size, output_dim)
        else:
            self.cls_projection = nn.Identity()

        # projection for patch tokens before bottleneck
        self.patch_projection = nn.Identity()

        # bottleneck to produce VAE-like feature
        self.bottleneck = SmallConvBottleneck(
            in_channels=vision_hidden_size,
            mid_channels=bottleneck_mid,
            out_channels=self.vae_channels,
            out_hw=(self.vae_h, self.vae_w),
        )

        # fuse [original cls] + [pooled dense token]
        # self.fusion_head = TokenFusionHead(
        #     cls_dim=output_dim,
        #     pooled_dim=self.vae_channels,
        #     output_dim=output_dim,
        # )
        self.vae_proj = nn.Linear(self.vae_channels, output_dim)
        self.tanh = nn.Tanh()

    def _infer_patch_grid(self, num_patch_tokens: int) -> Tuple[int, int]:
        """
        Infer patch grid from N.
        Assumes square grid.
        """
        side = int(math.sqrt(num_patch_tokens))
        if side * side != num_patch_tokens:
            raise ValueError(
                f"Patch token count {num_patch_tokens} is not a perfect square, "
                "cannot infer 2D grid automatically."
            )
        return side, side

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: image tensor, usually [B, C, H, W] or list of PIL images

        Returns:
            {
                "final_token": [B, output_dim],
                "vae_feature": [B, C_out, H_out, W_out],
                "cls_token":   [B, output_dim],
                "pooled_token":[B, C_out],
                "patch_tokens":[B, N, D],
            }
        """
        device = next(self.parameters()).device

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {
            k: v.to(device=device, dtype=self.dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()
        }

        # vision_outputs.last_hidden_state:
        # often [B, 1+N, D], where first token is cls
        vision_outputs = self.model.vision_model(**inputs)
        patch_tokens = vision_outputs.last_hidden_state  # [B, L, D]
        cls_token = vision_outputs.pooler_output

        # split cls token and patch tokens
        # cls_token = hidden[:, 0]         # [B, D]
        # patch_tokens = hidden[:, 1:]     # [B, N, D]

        # original cls -> projection
        cls_token = self.cls_projection(cls_token)  # [B, output_dim]

        # patch tokens -> 2D map
        B, N, D = patch_tokens.shape
        H_patch, W_patch = self._infer_patch_grid(N)

        patch_tokens_2d = self.patch_projection(patch_tokens)         # [B, N, D]
        patch_tokens_2d = patch_tokens_2d.view(B, H_patch, W_patch, D)
        patch_tokens_2d = patch_tokens_2d.permute(0, 3, 1, 2).contiguous()  # [B, D, H_patch, W_patch]

        # bottleneck -> VAE-like feature
        vae_feature = self.bottleneck(patch_tokens_2d)  # [B, C_out, H_out, W_out]
        vae_feature = self.tanh(vae_feature)

        # average pool -> token
        pooled_token = F.adaptive_avg_pool2d(vae_feature, output_size=1).flatten(1)  # [B, C_out]

        # # fuse pooled token with original cls token
        # final_token = self.fusion_head(cls_token, pooled_token)  # [B, output_dim]
        final_token = self.vae_proj(pooled_token)

        return {
            "final_token": final_token,
            "vae_feature": vae_feature,
            "cls_token": cls_token,
            "pooled_token": pooled_token,
            "patch_tokens": patch_tokens,
        }


class RobotCLIP(PreTrainedPolicy):
    """RobotCLIP: Contrastive learning for robot actions and visual observations.
    
    This model aligns action embeddings with visual observation embeddings
    using contrastive learning, similar to CLIP.
    """
    config_class = RobotCLIPConfig
    name = "robo_clip"
    
    def __init__(self, 
                 config: RobotCLIPConfig,
                 dataset_stats: dict[str, dict[str, Tensor]] | None = None,):
        super().__init__(config)
        self.config = config
        self.dtype = torch.bfloat16
        config.validate_features()
        # Vision encoder using SigLIP2
        self.vision_model = VisionEncoder(
            model_name=config.vision_model_name,
            output_dim=config.projection_dim,
        )
        
        # Action encoder using ACE
        action_config = ACEConfig(
            action_dim=config.action_dim,
            chunk_size=config.chunk_size,
            group_size=config.group_size,
            hidden_dim=config.hidden_dim,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            output_dim=config.output_dim,
            max_action_dim=config.max_action_dim,
        )
        self.action_encoder = ActionChunkEncoder(action_config)
        
        # Projection layers to align embeddings
        self.image_projection = nn.Linear(config.projection_dim, config.projection_dim)
        self.action_projection = nn.Linear(config.hidden_dim, config.projection_dim)
        
        # Temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / config.temperature))
        
        # Layer norm for stability
        self.image_ln = nn.LayerNorm(config.projection_dim)
        self.action_ln = nn.LayerNorm(config.hidden_dim)
        self.tanh = nn.Tanh()
    
    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        
    def select_action(self, batch):
        return super().select_action(batch)
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to normalized embeddings.
        
        Args:
            images: Image tensor of shape (B, C, H, W) or (B, H, W, C)
            
        Returns:
            Normalized image embeddings of shape (B, projection_dim)
        """
        image_embeddings = self.vision_model(images)["final_token"]  # (B, projection_dim)
        image_embeddings = self.image_ln(image_embeddings)
        # print(f"Image embeddings shape after vision model: {image_embeddings.shape}")
        image_embeddings = self.image_projection(image_embeddings)
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        # print(image_embeddings.shape)
        return image_embeddings
    
    def encode_actions(self, actions: torch.Tensor, sample_rate: int = 0) -> torch.Tensor:
        """Encode actions to normalized embeddings.
        
        Args:
            actions: Action tensor of shape (B, chunk_size, action_dim)
            sample_rate: Sample rate for action encoding
            
        Returns:
            Normalized action embeddings of shape (B, projection_dim)
        """
        action_embeddings = self.action_encoder(actions, sample_rate)  # (B, output_dim)
        action_embeddings = self.action_ln(action_embeddings)
        action_embeddings = self.action_projection(action_embeddings)
        action_embeddings = F.normalize(action_embeddings, dim=-1)
        # action_embeddings = self.tanh(action_embeddings)
        return action_embeddings
    
    def compute_contrastive_loss(
        self,
        image_embeddings: torch.Tensor,
        action_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss between image and action embeddings.
        
        Uses symmetric InfoNCE loss similar to CLIP.
        
        Args:
            image_embeddings: Normalized image embeddings of shape (B, D)
            action_embeddings: Normalized action embeddings of shape (B, D)
            
        Returns:
            Contrastive loss value
        """
        # print(f"Image embeddings shape: {image_embeddings.shape}, Action embeddings shape: {action_embeddings.shape}")
        batch_size = image_embeddings.shape[0]
        
        # Compute similarity matrix
        # logits: (B, B)
        # print("image", torch.max(image_embeddings), torch.min(image_embeddings))
        # print("action", torch.max(action_embeddings), torch.min(action_embeddings))
        logits = (image_embeddings @ action_embeddings.T) * self.logit_scale.exp()
        # print(torch.max(logits), torch.min(logits))
        # print(logits.shape)
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # Symmetric cross-entropy loss
        loss_i2a = F.cross_entropy(logits, labels)
        loss_a2i = F.cross_entropy(logits.T, labels)
        # print(F"loss_i2a: {loss_i2a.item():.4f}, loss_a2i: {loss_a2i.item():.4f}")
        
        # Average both directions
        loss = (loss_i2a + loss_a2i) / 2
        
        return loss
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass computing contrastive loss.
        
        Args:
            batch: Dictionary containing:
                - 'images': Image tensor of shape (B, C, H, W)
                - 'actions': Action tensor of shape (B, chunk_size, action_dim)
                - 'sample_rate': Optional sample rate index (default: 0)
                
        Returns:
            Contrastive loss value
        """
        # print(batch.keys())
        images = batch['observation.images.primary'].to(dtype=torch.float32)  # (B, C, H, W), [0, 1]
        actions = batch['action']
        states = batch['observation.state']
        sample_rate = batch.get('sample_rate', 0)
        # print(sample_rate)
        # print(torch.max(images), torch.min(images)) # 0-1
        images = images.squeeze()
        pil_images = [
            # in lerobot dataset, images are already in [0, 1] range, so we can directly convert to PIL without scaling
            Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
            for image in images
        ]
        # Encode images and actions
        # print(torch.max(actions), torch.min(actions), torch.max(sample_rate), torch.min(sample_rate))
        image_embeddings = self.encode_images(pil_images)  # (B, D)
        action_embeddings = self.encode_actions(actions, sample_rate)  # (B, D)
        
        # Compute contrastive loss
        loss = self.compute_contrastive_loss(image_embeddings, action_embeddings)
        
        # local_bs = image_embeddings.size(0)
        # rank = dist.get_rank()
        
        # # gather 全局特征
        # all_image_embeddings = concat_all_gather(image_embeddings)    # [global_bs, D]
        # all_action_embeddings = concat_all_gather(action_embeddings)  # [global_bs, D]
        # # print(all_action_embeddings.shape)
        # # print(torch.max(all_image_embeddings), torch.min(all_image_embeddings))
        
        # # 本卡 query，对全局 candidates 做分类
        # logits_i2a = self.logit_scale.exp() * image_embeddings @ all_action_embeddings.t()  # [local_bs, global_bs]
        # logits_a2i = self.logit_scale.exp() * action_embeddings @ all_image_embeddings.t()   # [local_bs, global_bs]

        # # 本地样本在全局中的正样本位置
        # labels = torch.arange(local_bs, device=image_embeddings.device) + rank * local_bs
        # # print(labels)
        # # print(logits_i2a.shape, labels.shape)

        # loss_i2a = F.cross_entropy(logits_i2a, labels)
        # loss_a2i = F.cross_entropy(logits_a2i, labels)
        # # print(all_image_embeddings.shape, all_action_embeddings.shape)

        # loss = 0.5 * (loss_i2a + loss_a2i)
        
        # print(F"Contrastive loss: {loss.item():.4f}")
        loss_dict = {"contrastive_loss": loss.item()}
        return loss, loss_dict
        # return image_embeddings, action_embeddings, self.logit_scale
    
    def get_similarity(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        sample_rate: int = 0
    ) -> torch.Tensor:
        """Get similarity scores between images and actions.
        
        Args:
            images: Image tensor of shape (B, C, H, W)
            actions: Action tensor of shape (B, chunk_size, action_dim)
            sample_rate: Sample rate for action encoding
            
        Returns:
            Similarity matrix of shape (B, B)
        """
        image_embeddings = self.encode_images(images)
        action_embeddings = self.encode_actions(actions, sample_rate)
        
        similarity = image_embeddings @ action_embeddings.T
        
        return similarity


def create_robot_clip_model(
    action_dim: int = 7,
    chunk_size: int = 16,
    group_size: int = 4,
    hidden_dim: int = 768,
    num_attention_heads: int = 12,
    num_hidden_layers: int = 12,
    output_dim: int = 768,
    vision_model_name: str = "google/siglip2-base-patch16-224",
    projection_dim: int = 768,
    temperature: float = 0.07,
) -> RobotCLIP:
    """Create a RobotCLIP model with specified parameters.
    
    Args:
        action_dim: Dimension of action vectors
        chunk_size: Number of actions in a chunk
        group_size: Number of actions per group
        hidden_dim: Hidden dimension for action encoder
        num_attention_heads: Number of attention heads
        num_hidden_layers: Number of transformer layers
        output_dim: Output dimension for action encoder
        vision_model_name: Name of the vision model to use
        projection_dim: Dimension for contrastive learning
        temperature: Temperature for contrastive loss
        
    Returns:
        RobotCLIP model
    """
    config = RobotCLIPConfig(
        action_dim=action_dim,
        chunk_size=chunk_size,
        group_size=group_size,
        hidden_dim=hidden_dim,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        output_dim=output_dim,
        vision_model_name=vision_model_name,
        projection_dim=projection_dim,
        temperature=temperature,
    )
    return RobotCLIP(config)