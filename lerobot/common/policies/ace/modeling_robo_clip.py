"""RobotCLIP model for aligning action embeddings with vision embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from torch import Tensor, nn
from transformers import SiglipModel, AutoProcessor
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.ace.configuration_robo_clip import ACEConfig, RobotCLIPConfig
from lerobot.common.policies.ace.modeling_ace import ActionChunkEncoder
from collections import deque
from PIL import Image
from lerobot.common.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
    OBS_ROBOT
)

class VisionEncoder(nn.Module):
    """Vision encoder using SigLIP2 or similar CLIP-like model."""
    
    def __init__(self, model_name: str = "google/siglip2-base-patch16-224", 
                 output_dim: int = 768,
                 dtype=torch.bfloat16):
        super().__init__()
        # try:
        self.model = SiglipModel.from_pretrained(
            model_name,
            # attn_implementation="flash_attention_2",
            dtype=torch.float32
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Get the hidden size from the vision model
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'hidden_size'):
                vision_hidden_size = self.model.config.hidden_size
            elif hasattr(self.model.config, 'vision_config'):
                vision_hidden_size = self.model.config.vision_config.hidden_size
            else:
                vision_hidden_size = 768  # default
        else:
            vision_hidden_size = 768
        
        self.hidden_size = vision_hidden_size
            
        # except ImportError:
        #     raise ImportError(
        #         "transformers library is required for VisionEncoder. "
        #         "Install it with: pip install transformers"
        #     )
        
        # Projection layer to match output_dim
        if vision_hidden_size != output_dim:
            self.projection = nn.Linear(vision_hidden_size, output_dim)
        else:
            self.projection = nn.Identity()
        self.dtype = dtype
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings.
        
        Args:
            images: Image tensor of shape (B, C, H, W) or (B, H, W, C)
            
        Returns:
            Image embeddings of shape (B, output_dim)
        """
        # Handle different image formats
        # if images.dim() == 4:
        #     if images.shape[1] == 3:  # (B, C, H, W)
        #         # Already in correct format
        #         pass
        #     elif images.shape[-1] == 3:  # (B, H, W, C)
        #         images = images.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
        
        inputs = self.processor(images=images, return_tensors="pt")
        # print(f"VisionEncoder: pixel_values shape = {inputs['pixel_values'].shape}") # torch.Size([1, 3, 224, 224])
        inputs = {k: v.to(dtype=self.dtype, device="cuda") for k, v in inputs.items()}
        # Get vision embeddings - use only the vision model
        vision_outputs = self.model.vision_model(**inputs)
        embeddings = vision_outputs.pooler_output
        
        # # Use pooler output or last hidden state's [CLS] token
        # if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
        #     embeddings = vision_outputs.pooler_output
        # else:
        #     # Use mean pooling over all patches
        #     embeddings = vision_outputs.last_hidden_state.mean(dim=1)
        
        # Project to output dimension
        embeddings = self.projection(embeddings)
        
        return embeddings


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
            output_dim=config.projection_dim
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
        image_embeddings = self.vision_model(images)  # (B, projection_dim)
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
        # print(F"Contrastive loss: {loss.item():.4f}")
        loss_dict = {"contrastive_loss": loss.item()}
        return loss, loss_dict
    
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
