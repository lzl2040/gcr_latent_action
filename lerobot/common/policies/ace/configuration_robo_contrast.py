"""Configuration for RoboContrast: perception <-> physical contrastive pre-training."""

from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWNormConfig
from lerobot.common.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("robo_contrast")
@dataclass
class RoboContrastConfig(PreTrainedConfig):
    """Contrastive alignment between the *perception* side and the *physical* side of a robot.

    Perception side (global scene information)
        vision at two time steps, language, and — once available — masks / optical flow.
    Physical side (robot-intrinsic information)
        proprioceptive state, action chunk, and tactile sensing.

    The two sides are encoded independently and pulled together with an InfoNCE objective,
    so the physical encoder is forced to explain *what visibly changed* over the horizon.
    """

    pretrained_path: str = ""

    # ------------------------------------------------------------------ perception
    vision_model_name: str = "/Data/lzl/huggingface/siglip2-base-patch16-224"
    freeze_vision_encoder: bool = True
    freeze_text_encoder: bool = True
    text_max_length: int = 32
    # Number of latent "what changed?" queries used to read the two-frame visual evidence.
    num_change_queries: int = 8
    num_fusion_layers: int = 3
    fusion_num_heads: int = 8
    # Keep every k-th patch token of each frame when building the evidence bank (1 = keep all).
    patch_token_stride: int = 1

    # ------------------------------------------------------------------ physical
    chunk_size: int = 16
    n_action_steps: int = 16
    group_size: int = 4
    hidden_dim: int = 768
    num_attention_heads: int = 12
    num_physical_layers: int = 6
    max_action_dim: int = 40
    max_state_dim: int = 40
    max_tactile_signal_dim: int = 32
    max_tactile_views: int = 4
    tactile_img_size: int = 64
    tactile_feat_dim: int = 128
    # Temporal distance (in frames) between the two perception frames. Defaults to chunk_size.
    frame_horizon: int | None = None
    use_wrist_image: bool = False

    # ------------------------------------------------------------------ regularisation
    # Per-sample probability of hiding a modality during training. Tactile is dropped most
    # aggressively because a 4-view image stream would otherwise dominate the physical token
    # budget and let the model ignore state/action entirely.
    dropout: float = 0.0
    modality_dropout_tactile: float = 0.3
    modality_dropout_state: float = 0.15
    modality_dropout_action: float = 0.1

    # ------------------------------------------------------------------ contrastive
    projection_dim: int = 512
    temperature: float = 0.07
    logit_scale_max: float = 100.0
    # Same-episode frames closer than this many frames are treated as false negatives.
    false_negative_frame_gap: int = 32

    # ------------------------------------------------------------------ batch shaping
    same_dataset_frac: float = 0.75
    # Drawing several frames of the *same* episode is both the strongest source of hard
    # negatives and, incidentally, the cheapest thing to read: the videos live on a spinning
    # disk and random seeks into 100-200 MB files dominate the wall clock. Raising the group
    # size from 4 to 8 doubled dataloader throughput (80 -> 164 samples/s).
    episode_group_frac: float = 0.75
    episode_group_size: int = 8
    min_frame_gap: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    resize_imgs_with_padding: tuple[int, int] = (224, 224)
    empty_cameras: int = 0

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 0.05
    optimizer_eps: float = 1e-8
    optimizer_betas: tuple[float, float] = (0.9, 0.95)

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = -1
    scheduler_platform_steps: int = 20_000
    scheduler_decay_lr: float = 2.5e-6

    # kept for CLI compatibility with the previous ACE pipeline
    frozen_ace: bool = False
    ace_pretrained_path: str = ""

    def __post_init__(self):
        super().__post_init__()
        if self.frame_horizon is None:
            self.frame_horizon = self.chunk_size
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) cannot exceed `chunk_size` ({self.chunk_size})."
            )
        if self.chunk_size % self.group_size != 0:
            raise ValueError(
                f"`chunk_size` ({self.chunk_size}) must be divisible by `group_size` ({self.group_size})."
            )
        if self.hidden_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"`hidden_dim` ({self.hidden_dim}) must be divisible by "
                f"`num_attention_heads` ({self.num_attention_heads})."
            )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))

    def get_optimizer_preset(self) -> AdamWNormConfig:
        return AdamWNormConfig(
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
