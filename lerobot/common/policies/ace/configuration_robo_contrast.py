"""Configuration for RoboContrast: perception <-> physical contrastive pre-training."""

from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWNormConfig
from lerobot.common.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.common.policies.ace.ftp1_tactile import FTP1_IMAGE_SIZE, FTP1_SENSOR_NAMES
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
    # DINOv3 rather than SigLIP2 for vision. SigLIP's visual features are trained to match a
    # caption, so they keep what language describes and discard the rest; the change between
    # two frames is mostly *not* describable, and DINOv3's self-supervised features preserve
    # far more spatial and geometric detail. DINOv3 has no text tower, so language still comes
    # from SigLIP2 -- its text embedding is already aligned to a visual space, which is what
    # makes "the instruction selects which change matters" work.
    vision_model_name: str = "/Data/lzl/huggingface/dinov3-vitb16-pretrain-lvd1689m"
    text_model_name: str = "/Data/lzl/huggingface/siglip2-base-patch16-224"
    freeze_vision_encoder: bool = True
    freeze_text_encoder: bool = True
    text_max_length: int = 32
    # Number of latent "what changed?" queries used to read the two-frame visual evidence.
    num_change_queries: int = 16
    # Self-attention layers run over the evidence bank ([v0, v1, v1-v0, text]) *before* the
    # change queries read it. This is where most of the perception capacity lives: it is the
    # only place where a parameter sees all ~620 tokens rather than 16 queries.
    num_evidence_layers: int = 5
    num_fusion_layers: int = 5
    fusion_num_heads: int = 16
    # Feature-space reconstruction: predict the frame-(t+H) DINOv3 features from the frame-t
    # features, the instruction and the change queries. The queries are the only route by
    # which anything about frame t+H can reach the prediction, so the objective forces them to
    # actually encode the change instead of whatever shortcut the contrastive loss tolerates.
    num_predictor_layers: int = 3
    perception_recon_weight: float = 1.0
    # Keep every k-th patch token of each frame when building the evidence bank (1 = keep all).
    patch_token_stride: int = 1

    # ------------------------------------------------------------------ physical
    # ``chunk_size`` is the number of *resampled* timesteps handed to the model, i.e. the token
    # grid, not a raw frame count. How many raw frames that grid spans is decided per dataset
    # from ``chunk_seconds`` -- see the note on ``chunk_seconds`` below.
    chunk_size: int = 32
    n_action_steps: int = 16
    # Frames per grouped token. ``chunk_size / group_size`` tokens are emitted for each of
    # state, action and tactile signal, so this sets the physical sequence length:
    # ``1 + 3 * chunk_size / group_size + max_tactile_views``.
    #
    # Only the *ratio* matters for the token budget. An earlier experiment found group_size 4
    # worse than 2 (contrastive loss 4.14 vs 3.71, doc/results.md S9.4), but that was at a fixed
    # chunk_size=16, where raising group_size halved the number of groups (8 -> 4) and shrank
    # the budget. Here chunk_size doubled at the same time, so ``num_groups`` is 8 either way
    # and the change is token neutral. Do not read S9.4 as "group_size 4 is bad" -- read it as
    # "count the budget after the change".
    group_size: int = 4
    hidden_dim: int = 1024
    num_attention_heads: int = 16
    # Sized so that the *learnable* capacity of the two branches is roughly equal
    # (~200M each); the frozen DINOv3 and SigLIP2 towers are feature extractors, not capacity
    # the model is free to use. Depth is cheap here because the physical side is only ~15
    # tokens wide, unlike the ~420-token evidence bank.
    num_physical_layers: int = 14
    max_action_dim: int = 40
    max_state_dim: int = 40
    max_tactile_signal_dim: int = 32
    # Must not exceed canonical_space.MAX_TACTILE_VIEWS, which the dataset clamps against.
    # 6 covers `ftp_1_sharpa`'s three pads per hand; datasets with fewer fill the spare slots
    # with the learned `missing` token.
    max_tactile_views: int = 6
    # ResNet-18 downsamples by 32, so 112 gives a 4x4 map (64 would give a useless 2x2).
    # Forced to 224 by ``__post_init__`` when ``tactile_backbone="ftp1"``, which is the
    # resolution its positional embedding was trained at.
    tactile_img_size: int = 112
    tactile_feat_dim: int = 512
    # A tactile pad whose per-channel *spatial* std stays below this (on a [0, 1] scale, so one
    # 8-bit grey level is 0.0039) at both `t` and `t+H` is treated as absent rather than as a
    # blank reading. Roughly half of our tactile pad-frames are dead -- see `doc/results.md`
    # §10.5. Set to 0 to disable the check.
    tactile_dead_std: float = 0.002
    # Length of the window in *seconds*, converted to a raw frame count per dataset.
    #
    # A fixed frame count is the wrong unit for this mixture. Its fps spans 10x (fractal is
    # 3 fps, most of the rest is 30), so a 16-frame window means 5.3 s on fractal and 0.53 s on
    # everything else -- a 10x difference in physical meaning, purely by accident of fps. The
    # 30 fps datasets were being asked to explain almost no visual change, while fractal, whose
    # episodes are only ~43 frames, was spending a third of an episode per sample and losing
    # 37% of its pairs to episode ends.
    #
    # The window is therefore ``clamp(round(chunk_seconds * fps), min, max)`` raw frames,
    # resampled onto the fixed ``chunk_size`` token grid. Measured per-dataset visual change
    # and pair validity behind these numbers are in doc/results.md S12.
    chunk_seconds: float = 1.6
    # Floors and caps the raw window. The floor matters for fractal: 1.6 s at 3 fps is 5 frames,
    # and 8 frames (2.67 s) measured strictly better -- more visual change *and* more valid
    # pairs than the 16 frames it used before.
    chunk_frames_min: int = 8
    chunk_frames_max: int = 48
    # Overrides the duration-based window with a fixed frame count on every dataset. ``None``
    # means "derive it from chunk_seconds", which is what you want; this exists for ablations.
    frame_horizon: int | None = None
    use_wrist_image: bool = False

    # ------------------------------------------------------------------ regularisation
    # Per-sample probability of hiding a modality during training. Tactile is dropped most
    # aggressively because a 4-view image stream would otherwise dominate the physical token
    # budget and let the model ignore state/action entirely.
    # Tactile image encoder, following UniVTAC (`UniVTAC/encoder/network.py`): a plain
    # ImageNet-pretrained ResNet-18 with a 512-d output, plus a reconstruction head. UniVTAC
    # supervises marker positions / depth / contact pose from simulation, which we do not
    # have for real sensors, so we keep only the RGB reconstruction head. Giving tactile its
    # own objective stops its features from being shaped purely by the contrastive loss.
    #
    # Set to "ftp1" to swap the ResNet for the *pretrained* FTP-1 tactile tower instead
    # (`lerobot/common/policies/ace/ftp1_tactile.py`). That tower is frozen, so it replaces
    # ~11.7M trainable parameters with 0, and it was trained on ~3000 h of tactile data
    # including our own sharpa / VisuoTactile / RDP datasets. Which of the two is better is an
    # empirical question: judge it on the *windowed contrastive loss*, not retrieval accuracy,
    # which cannot resolve tactile changes at all (doc/results.md S9).
    tactile_backbone: str = "resnet18"
    tactile_pretrained: bool = True
    # Where the FTP-1 `hpt_tokenizer/*.safetensors` files live. Only read for backbone="ftp1".
    ftp1_tactile_dir: str = "/Data/lzl/huggingface/ftp1_v0426_50kstep"
    # Which per-sensor tokenizers to load. Empty means "every sensor we have a mapping for",
    # which is 7 x 22.0M = 154M frozen parameters. `debug_research_data` only reaches three of
    # them (SharpaWave, OpenLoongVTouch, GelSightMini), so narrowing this list saves ~88M
    # parameters of GPU memory and some start-up time; a sensor that is needed but not listed
    # falls back to zero features, so only narrow it deliberately.
    ftp1_tactile_sensors: tuple[str, ...] = ()
    # The reconstruction head only exists to shape the tactile features. A frozen FTP-1 tower
    # has no features to shape, so `__post_init__` switches this off for backbone="ftp1" --
    # otherwise we would pay for a decoder whose gradient reaches nothing.
    tactile_recon_weight: float = 0.1
    tactile_recon_size: int = 28
    # UniVTAC trains its tactile backbone with a dedicated (much lower) learning rate; the
    # same trick keeps a 11.7M-parameter CNN from racing ahead of the rest of the model.
    tactile_lr_scale: float = 0.1

    # Recompute trunk activations in the backward pass instead of storing them. Costs ~30%
    # compute and saves ~15 GB at batch 256, which is worth it while the disk is the ceiling.
    gradient_checkpointing: bool = True

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
        # `frame_horizon` deliberately stays None here: the window is derived per dataset from
        # `chunk_seconds`, and defaulting it to `chunk_size` would silently pin every dataset
        # back to a fixed frame count.
        if self.chunk_size % self.group_size != 0:
            raise ValueError(
                f"`chunk_size` ({self.chunk_size}) must be divisible by `group_size` "
                f"({self.group_size}); it is split into chunk_size/group_size grouped tokens."
            )
        if self.chunk_frames_min > self.chunk_frames_max:
            raise ValueError(
                f"`chunk_frames_min` ({self.chunk_frames_min}) exceeds `chunk_frames_max` "
                f"({self.chunk_frames_max})."
            )
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
        if self.tactile_backbone not in ("resnet18", "ftp1"):
            raise ValueError(
                f"`tactile_backbone` must be 'resnet18' or 'ftp1', got {self.tactile_backbone!r}."
            )
        if self.tactile_backbone == "ftp1":
            # Nested `--policy.ftp1_tactile_sensors="['A','B']"` is not parsed as a list:
            # draccus decodes the literal into a tuple of its *characters*. Detect that (any
            # element that is not a valid sensor name) and rebuild the list from the joined
            # text, so both the CLI literal and a real tuple in code work.
            raw = self.ftp1_tactile_sensors
            names = [str(x) for x in ([raw] if isinstance(raw, str) else raw)]
            if any(name not in FTP1_SENSOR_NAMES for name in names):
                text = "".join(names).strip().strip("[]()")
                names = [part.strip().strip("\"'") for part in text.split(",") if part.strip()]
            self.ftp1_tactile_sensors = tuple(names)
            unknown = set(self.ftp1_tactile_sensors) - set(FTP1_SENSOR_NAMES)
            if unknown:
                raise ValueError(
                    f"Unknown FTP-1 tactile sensor(s) {sorted(unknown)}. "
                    f"Valid names: {FTP1_SENSOR_NAMES}."
                )
            # The published positional embedding is (1, 197, 768), i.e. exactly 14x14 patches
            # of 16 pixels. Anything else has to be resampled, which puts the input
            # off-distribution for weights we are not training.
            self.tactile_img_size = FTP1_IMAGE_SIZE
            # A frozen tower cannot be shaped by a reconstruction loss.
            self.tactile_recon_weight = 0.0

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
