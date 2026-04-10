"""Configuration for Action Chunk Encoder (ACE)."""
from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
import torch


@dataclass
class ACEConfig:
    """Configuration for Action Chunk Encoder.
    
    Args:
        action_dim: Dimension of action vectors (will be padded to action_dim_padded)
        action_dim_padded: Padded action dimension (default: 32)
        chunk_size: Number of actions in a chunk
        group_size: Number of actions per group for processing (default: 4)
        hidden_dim: Hidden dimension for the encoder (default: 768)
        num_attention_heads: Number of attention heads (default: 12)
        num_hidden_layers: Number of transformer layers (default: 12)
        intermediate_dim: Intermediate dimension in FFN (default: 3072)
        dropout: Dropout probability (default: 0.1)
        max_position_embeddings: Maximum position for RoPE (default: 512)
        output_dim: Output embedding dimension (optional, defaults to hidden_dim)
    """
    action_dim: int = 7
    max_action_dim: int = 32
    chunk_size: int = 16
    group_size: int = 4
    hidden_dim: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_dim: int = 3072
    dropout: float = 0.1
    max_position_embeddings: int = 512
    output_dim: int = None
    
    def __post_init__(self):
        if self.output_dim is None:
            self.output_dim = self.hidden_dim
        # Validate that hidden_dim is divisible by num_attention_heads
        assert self.hidden_dim % self.num_attention_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_attention_heads ({self.num_attention_heads})"


@PreTrainedConfig.register_subclass("robo_clip")
@dataclass
class RobotCLIPConfig(PreTrainedConfig):
    """Configuration for RobotCLIP model.
    
    RobotCLIP aligns action embeddings with vision embeddings using contrastive learning.
    
    Args:
        action_dim: Dimension of action vectors
        chunk_size: Number of actions in a chunk
        group_size: Number of actions per group
        hidden_dim: Hidden dimension for action encoder
        num_attention_heads: Number of attention heads
        num_hidden_layers: Number of transformer layers
        output_dim: Output dimension for action encoder
        vision_model_name: Name of the vision model to use (default: SigLIP2)
        projection_dim: Dimension for contrastive learning
        temperature: Temperature for contrastive loss
        freeze_vision_encoder: Whether to freeze vision encoder weights
    """
    pretrained_path: str = ""
    
    action_dim: int = 7
    chunk_size: int = 32
    group_size: int = 4
    hidden_dim: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    output_dim: int = None
    # vision_model_name: str = "/Data/lzl/huggingface/siglip2-base-patch16-224"
    vision_model_name: str = "/mnt/wangxiaofa/pt_weights/siglip2-base-patch16-224"
    projection_dim: int = 768 # siglip2 output dim
    temperature: float = 1.0
    freeze_vision_encoder: bool = True
    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32
    
    
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    
    n_action_steps: int = 16

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Add empty images. Used by pi0_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    # tokenizer_max_length: int = 100

    # Projector, default=1024
    proj_width: int = 1024

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True
    attention_implementation: str = "eager"  # or eager, flex

    # Training presets
    optimizer_lr: float = 1e-4
    # optimizer_beta2_decay: float = -0.8
    # optimizer_eps: tuple[float | None, float] = (None, 0.001)
    # optimizer_d: float = 1.0
    optimizer_weight_decay: float = 1e-10
    
    optimizer_eps: float = 1e-8
    optimizer_betas: tuple[float, float] = (0.9, 0.95)

    scheduler_warmup_steps: int = 3_000
    scheduler_decay_steps: int = -1
    scheduler_platform_steps: int = 20_000
    scheduler_decay_lr: float = 2.5e-6

    # TODO: Add EMA

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by pi0 for aloha real models. It is not ported yet in LeRobot."
            )
        
        if self.output_dim is None:
            self.output_dim = self.hidden_dim
        # Validate that hidden_dim is divisible by num_attention_heads
        assert self.hidden_dim % self.num_attention_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_attention_heads ({self.num_attention_heads})"


    def validate_features(self) -> None:
        # TODO: implement value error
        # if not self.image_features and not self.env_state_feature:
        #     raise ValueError("You must provide at least one image or the environment state among the inputs.")

        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_platform_steps=self.scheduler_platform_steps,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
    