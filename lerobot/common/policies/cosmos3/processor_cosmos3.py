# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native Cosmos3 canvas/prompt/rotation algorithms, adapted to flat LeRobot batches."""

import json
import logging
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file, save_file
from torch import Tensor
from transformers import AutoTokenizer

from lerobot.configs.types import NormalizationMode

from ._compat import ACTION, OBS_STATE, FlatBatchStep, TransitionKey, require_package
from .configuration_cosmos3 import RESOLUTION_TIER_LIMITS

ROTATION_WIDTHS = {"rot6d": 6, "rot9d": 9, "axisangle": 3, "euler_xyz": 3, "quat_xyzw": 4, "quat_wxyz": 4}


CHANNEL_TYPES = ("pos", *ROTATION_WIDTHS, "gripper", "joint")

VIDEO_CANVAS_SIZES = {
    "256": {"1,1": (256, 256), "4,3": (320, 256), "3,4": (256, 320), "16,9": (320, 192), "9,16": (192, 320)},
    "480": {
        "1,1": (640, 640),
        "4,3": (736, 544),
        "3,4": (544, 736),
        "16,9": (832, 480),
        "9,16": (480, 832),
        "2,3": (512, 768),
    },
    "720": {
        "1,1": (960, 960),
        "4,3": (1104, 832),
        "3,4": (832, 1104),
        "16,9": (1280, 720),
        "9,16": (720, 1280),
    },
}

CONCAT_VIEW_TEMPLATE = "This video contains concatenated views from multiple camera perspectives."


START_OF_GENERATION = "<|vision_start|>"


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Rotation matrices `[..., 3, 3]` of axis-angle vectors `[..., 3]` (Rodrigues' formula, in float64)."""
    rotvec = axis_angle.double()
    angle = rotvec.norm(dim=-1, keepdim=True)  # [...,1]
    axis = rotvec / angle.clamp(min=1e-12)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    cross = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).view(*x.shape, 3, 3)  # [K]_x
    sin, cos = angle.sin()[..., None], angle.cos()[..., None]
    identity = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    matrix = identity + sin * cross + (1 - cos) * (cross @ cross)
    return matrix.to(axis_angle.dtype)


def matrix_to_axis_angle(matrix: Tensor) -> Tensor:
    """Axis-angle vectors `[..., 3]` of rotation matrices `[..., 3, 3]` (in float64), for angles below pi."""
    m = matrix.double()
    skew = torch.stack(
        [m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1
    )
    sin_angle = skew.norm(dim=-1) / 2  # [...]
    cos_angle = (m.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    angle = torch.atan2(sin_angle, cos_angle)
    # angle / sin(angle), 1 at the identity.
    scale = torch.where(sin_angle > 1e-8, angle / sin_angle.clamp(min=1e-8), torch.ones_like(angle))
    return (skew * scale[..., None] / 2).to(matrix.dtype)


def matrix_to_rot6d(matrix: Tensor) -> Tensor:
    """The first two columns `[R00, R10, R20, R01, R11, R21]` of rotation matrices `[..., 3, 3]` (native rot6d)."""
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def rot6d_to_matrix(rot6d: Tensor) -> Tensor:
    """
    Nearest rotation matrices `[..., 3, 3]` of rot6d vectors `[..., 6]`, as the native LIBERO decoder builds
    them: the cross product of the two columns completes the matrix, which SVD projects onto SO(3).
    """
    first, second = rot6d[..., :3].double(), rot6d[..., 3:6].double()
    matrix = torch.stack([first, second, torch.linalg.cross(first, second)], dim=-1)  # columns
    u, _, vh = torch.linalg.svd(matrix)
    reflection = torch.linalg.det(u @ vh) < 0
    u = torch.where(reflection[..., None, None], u * u.new_tensor([1.0, 1.0, -1.0]), u)
    return (u @ vh).to(rot6d.dtype)


def rotation_angle(rotation: Tensor, rotation_format: str) -> Tensor:
    """
    Angles `[...]` of rotations `[..., width]` from the identity (native `_rotation_angle_per_arm`). For Euler
    angles the norm, an upper bound of the angle.
    """
    if rotation_format in ("quat_xyzw", "quat_wxyz"):
        w = rotation[..., 3] if rotation_format == "quat_xyzw" else rotation[..., 0]
        return 2.0 * torch.arccos(w.abs().clamp(max=1.0))
    if rotation_format in ("axisangle", "euler_xyz"):
        return rotation.norm(dim=-1)
    if rotation_format == "rot6d":
        a, b = rotation[..., :3], rotation[..., 3:6]
        trace = a[..., 0] + b[..., 1] + a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]  # of [a, b, a x b]
    else:  # rot9d, the flattened matrix
        trace = rotation[..., 0] + rotation[..., 4] + rotation[..., 8]
    return torch.arccos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))


@dataclass
class Cosmos3IdleFrameProcessorStep(FlatBatchStep):
    """
    Count the idle frames of a training action chunk for the `idle_frame` of the prompt, as the native
    datasets do (`compute_idle_frames`), on the action before the normalizer. The channel types come from
    `action_feature_names`, `[<prefix>_]<type>_<index>` with a prefix per arm (`left_rot6d_0`,
    `right_quat_wxyz_3`). A frame is still when the norm of all `pos` channels is below
    `translation_threshold`, every rotation (`rot6d`, `rot9d`, `axisangle`, `euler_xyz`, `quat_xyzw`,
    `quat_wxyz`) is closer to the identity than `rotation_threshold`, and every `gripper` and `joint` channel
    is closer to the previous frame than `gripper_threshold` and `joint_threshold`. Other channel types are
    ignored. A still frame counts inside a run of at least `min_streak` still frames. The speed thresholds are
    per second and divided by `fps`.

    Writes the counts `[B]` as `idle_frames` into the complementary data. Without an action (inference) or
    without a channel of these types nothing is written, and the prompt writes 0.

    Args:
        action_feature_names: Names of the action channels.
        fps: Frame rate of the data.
        use_state: The state action goes in front of the actions.
        idle_state: Count over the state action too, as the native DROID recipe does.
        translation_threshold: Translation speed of a still frame.
        rotation_threshold: Rotation speed of a still frame, in radians.
        gripper_threshold: Gripper change of a still frame from the previous frame.
        joint_threshold: Joint speed of a still frame, in radians.
        min_streak: Shortest run of still frames that counts.
    """

    action_feature_names: list[str] | None
    fps: float | None
    use_state: bool = False
    idle_state: bool = False
    translation_threshold: float = 5e-3
    rotation_threshold: float = math.radians(1.5)
    gripper_threshold: float = 1e-2
    joint_threshold: float = 5e-3
    min_streak: int = 3

    def __post_init__(self):
        if self.action_feature_names and not any(self.channel_mask(*CHANNEL_TYPES)):
            logging.warning(
                f"The action channels {self.action_feature_names} are not named [<prefix>_]<type>_<index> with a "
                f"type of {CHANNEL_TYPES}, so the prompt counts no idle frame."
            )

    def complementary_data(self, complementary_data):
        action = self.transition.get(TransitionKey.ACTION)
        if action is not None and any(self.channel_mask(*CHANNEL_TYPES)):
            observation = self.transition.get(TransitionKey.OBSERVATION) or {}
            complementary_data["idle_frames"] = self.count(action, observation.get(OBS_STATE), self.fps)
        return complementary_data

    def channel_mask(self, *types: str) -> list[bool]:
        """Which action channels have one of `types`, from their names `[<prefix>_]<type>_<index>`."""
        pattern = rf"(?:.+_)?(?:{'|'.join(types)})_\d+"
        return [re.fullmatch(pattern, name) is not None for name in self.action_feature_names or []]

    def count(self, action: Tensor, state: Tensor | None, fps: float) -> Tensor:
        """
        Idle frames `[B]` of dataset action chunks `[B, T, action_dim]` with the states `[B, action_dim]` at the
        frame rate `fps`.
        """
        rows = action
        if self.idle_state and self.use_state and state is not None:  # the native DROID recipe
            rows = torch.cat([state[:, None], action], dim=1)
        still = torch.ones(rows.shape[:2], dtype=torch.bool, device=rows.device)
        positions = rows[..., self.channel_mask("pos")]  # the norm of all translation channels together
        still &= positions.norm(dim=-1) < self.translation_threshold / fps
        for rotation_format, width in ROTATION_WIDTHS.items():  # every rotation of every arm
            rotations = rows[..., self.channel_mask(rotation_format)].unflatten(-1, (-1, width))
            still &= (rotation_angle(rotations, rotation_format) < self.rotation_threshold / fps).all(dim=-1)
        for kind, threshold in (("gripper", self.gripper_threshold), ("joint", self.joint_threshold / fps)):
            values = rows[..., self.channel_mask(kind)]  # the change of every channel from the previous frame
            still &= (torch.diff(values, dim=1, prepend=values[:, :1]).abs() < threshold).all(dim=-1)
        # A frame counts when a run of min_streak still frames covers it.
        k = self.min_streak
        runs = F.pad(still.float(), (0, k - 1)).unfold(-1, k, 1).amin(dim=-1)  # the k frames from t are still
        return F.pad(runs, (k - 1, 0)).unfold(-1, k, 1).amax(dim=-1).sum(dim=-1).long()

    def get_config(self) -> dict[str, Any]:
        return {
            "action_feature_names": self.action_feature_names,
            "fps": self.fps,
            "use_state": self.use_state,
            "idle_state": self.idle_state,
            "translation_threshold": self.translation_threshold,
            "rotation_threshold": self.rotation_threshold,
            "gripper_threshold": self.gripper_threshold,
            "joint_threshold": self.joint_threshold,
            "min_streak": self.min_streak,
        }


@dataclass
class Cosmos3VideoCanvasProcessorStep(FlatBatchStep):
    """
    Compose the cameras into the video canvas of the generation pathway, the native multi view layout fit to a
    resolution tier:

    1. The cameras are `camera_keys` in their order, the cameras of `input_features` in the order of the
       dataset info.json: `[third person, wrist]` for LIBERO and RoboCasa, `[wrist, exterior 1, exterior 2]`
       for DROID.
    2. Every view is resized to `view_size` (bilinear with antialias, rounded to uint8 levels) when it has
       another size.
    3. One view is used as is, two go side by side, three put the first on top and the others below it at half
       its height, side by side and splitting its width.
    4. The canvas is quantized to uint8 and fit into the canvas of its resolution tier (the shorter side picks
       the tier, the closest aspect ratio the canvas) without enlarging it: bicubic resizing with antialias,
       then reflection padding on the right and at the bottom (edge replication when the padding is wider than
       the content).
    5. It is mapped to [-1, 1].

    Training batches carry the clip `[batch, chunk_size + 1, 3, H, W]` of every camera, inference batches the
    condition frame `[batch, 3, H, W]`. The latent grid always spans the whole clip, `1 + chunk_size /
    temporal_compression` latent frames. The step writes `pixel_values_videos_gen` `[batch, 3, frames,
    canvas_h, canvas_w]`, `video_grid_thw_gen` (the latent grid of the content), `second_per_grid_ts_gen` and
    `video_canvas_size` (canvas height and width, for the prompt) into the complementary data.

    Args:
        camera_keys: Observation keys of the cameras, one to three.
        view_size: Height and width of every view.
        fps: Frame rate of the clip.
        chunk_size: Actions per chunk, the clip has chunk_size + 1 frames.
        temporal_compression: Frames per latent frame of the VAE after the first one.
        spatial_compression: Pixels per latent pixel of the VAE.
    """

    camera_keys: list[str]
    view_size: tuple[int, int] | None
    fps: float | None
    chunk_size: int
    temporal_compression: int = 4
    spatial_compression: int = 16

    def complementary_data(self, complementary_data):
        observation = self.transition.get(TransitionKey.OBSERVATION) or {}
        views = [self.prepare_view(observation[key], self.view_size) for key in self.camera_keys]
        # Compose in [0, 1] and truncate to uint8, as the native datasets do.
        frames = (self.compose(views) * 255.0).clamp(0, 255).to(torch.uint8)  # [B,T,3,H,W]
        canvas, (content_h, content_w) = self.fit_canvas(frames)

        batch_size, device = canvas.shape[0], canvas.device
        num_latent_frames = 1 + self.chunk_size // self.temporal_compression
        grid = [
            num_latent_frames,
            content_h // self.spatial_compression,
            content_w // self.spatial_compression,
        ]
        seconds_per_latent_frame = self.temporal_compression / self.fps
        complementary_data["pixel_values_videos_gen"] = canvas.permute(0, 2, 1, 3, 4).float() / 127.5 - 1.0
        complementary_data["video_grid_thw_gen"] = torch.tensor(grid, device=device).expand(batch_size, 3)
        complementary_data["second_per_grid_ts_gen"] = torch.full(
            (batch_size,), seconds_per_latent_frame, device=device
        )
        complementary_data["video_canvas_size"] = [canvas.shape[-2], canvas.shape[-1]]
        return complementary_data

    def prepare_view(self, view: Tensor, view_size: tuple[int, int]) -> Tensor:
        """A camera view `[B,3,H,W]` or `[B,T,3,H,W]` in [0, 1] as `[B,T,3,h,w]` in [0, 1] at `view_size`."""
        if view.dtype != torch.float32:
            raise TypeError(f"Camera views reach the canvas as float32 in [0, 1], got {view.dtype}.")
        if view.ndim == 4:
            view = view[:, None]
        height, width = view_size
        if tuple(view.shape[-2:]) != (height, width):
            # Bilinear with antialias, rounded to uint8 levels like a PIL resize.
            frames = view.flatten(0, 1) * 255.0
            frames = F.interpolate(
                frames, size=(height, width), mode="bilinear", align_corners=False, antialias=True
            )
            view = (frames.round().clamp(0, 255) / 255.0).unflatten(0, view.shape[:2])
        return view

    def compose(self, views: list[Tensor]) -> Tensor:
        """The multi view layout of the native datasets on `[B,T,3,h,w]` views in [0, 1]."""
        if len(views) == 1:
            return views[0]
        if len(views) == 2:
            return torch.cat(views, dim=-1)
        top, *bottom = views
        height, width = top.shape[-2:]
        halves = []
        for view, half_width in zip(bottom, (width // 2, width - width // 2), strict=True):
            half = F.interpolate(
                view.flatten(0, 1), size=(height // 2, half_width), mode="bilinear", align_corners=False
            )
            halves.append(half.unflatten(0, view.shape[:2]))
        return torch.cat([top, torch.cat(halves, dim=-1)], dim=-2)

    def fit_canvas(self, frames: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Fit uint8 frames `[B,T,3,H,W]` into the canvas of their resolution tier, returns it and the content size."""
        height, width = frames.shape[-2:]
        canvas_h, canvas_w = self.canvas_size(height, width)
        scale = min(canvas_w / width, canvas_h / height, 1.0)
        content_h, content_w = int(scale * height + 0.5), int(scale * width + 0.5)
        batch_size, num_frames = frames.shape[:2]
        canvas = frames.flatten(0, 1)
        if (content_h, content_w) != (height, width):
            content_size = (content_h, content_w)
            canvas = F.interpolate(
                canvas.float(), content_size, mode="bicubic", align_corners=False, antialias=True
            )
            canvas = canvas.clamp(0, 255).round().to(torch.uint8)
        pad_right, pad_bottom = canvas_w - content_w, canvas_h - content_h
        if pad_right or pad_bottom:
            mode = "replicate" if pad_right >= content_w or pad_bottom >= content_h else "reflect"
            canvas = F.pad(canvas.float(), (0, pad_right, 0, pad_bottom), mode=mode).to(torch.uint8)
        return canvas.unflatten(0, (batch_size, num_frames)), (content_h, content_w)

    def canvas_size(self, height: int, width: int) -> tuple[int, int]:
        """Height and width of the canvas of the resolution tier of a `height` x `width` frame."""
        shorter_side = min(height, width)
        tiers = [tier for tier, limit in RESOLUTION_TIER_LIMITS if shorter_side <= limit]
        if not tiers:
            raise ValueError(
                f"Unsupported {height}x{width} canvas, the shorter side exceeds {RESOLUTION_TIER_LIMITS[-1][1]}."
            )
        # The first aspect ratio wins a tie.
        sizes = VIDEO_CANVAS_SIZES[tiers[0]].values()
        return min(sizes, key=lambda size: abs(height / width - size[1] / size[0]))[::-1]

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_keys": list(self.camera_keys),
            "view_size": None if self.view_size is None else list(self.view_size),
            "fps": self.fps,
            "chunk_size": self.chunk_size,
            "temporal_compression": self.temporal_compression,
            "spatial_compression": self.spatial_compression,
        }


@dataclass
class Cosmos3PromptProcessorStep(FlatBatchStep):
    """
    Write the JSON prompt of the understanding pathway for every sample as `prompt` into the complementary
    data, the native `ActionPromptJsonFormatter` of the action policy recipes:

        {"cinematography": {"framing": <multi view template> <view description>},
         "actions": [{"time": "0:00-0:01", "description": <task>., "idle_frame": "0 out of 16."}],
         "duration": "0s", "fps": 20.0, "resolution": {"H": <canvas h>, "W": <canvas w>},
         "aspect_ratio": "16,9"}

    serialized with `json.dumps`. The timing covers the chunk_size + 1 frames of the clip at `fps`, the
    resolution is the video canvas. The idle frames come from `Cosmos3IdleFrameProcessorStep` (0 at inference)
    and count out of the chunk and the state action. An empty task gives an empty prompt.

    Args:
        view_description: Layout of the canvas after the multi view template.
        fps: Frame rate of the clip.
        chunk_size: Actions per chunk.
        use_state: The state action goes in front of the actions.
    """

    view_description: str
    fps: float | None
    chunk_size: int
    use_state: bool

    def complementary_data(self, complementary_data):
        tasks = complementary_data.get("task")
        if tasks is None:
            raise ValueError("Cosmos3PromptProcessorStep needs the task of every sample.")
        if isinstance(tasks, str):
            tasks = [tasks]
        canvas_h, canvas_w = complementary_data.pop("video_canvas_size")
        idle_frames = complementary_data.pop("idle_frames", None)
        counts = [0] * len(tasks) if idle_frames is None else idle_frames.tolist()
        prompts = []
        for task, idle in zip(tasks, counts, strict=True):
            prompts.append(self.caption(task, canvas_h, canvas_w, idle, self.fps))
        complementary_data["prompt"] = prompts
        return complementary_data

    def caption(self, task: str, canvas_h: int, canvas_w: int, idle_frames: int, fps: float) -> str:
        """The JSON prompt of a task at the frame rate `fps`, empty for an empty task."""
        task = task.strip()
        if not task:
            return ""
        if not task.endswith((".", "!", "?")):
            task = f"{task}."
        duration = (self.chunk_size + 1) / fps
        rounded = round(duration)
        # The ratio name of a tier canvas, the reduced width and height of another canvas.
        ratios = [ratio for sizes in VIDEO_CANVAS_SIZES.values() for ratio, size in sizes.items()
                  if size == (canvas_w, canvas_h)]  # fmt: skip
        divisor = math.gcd(canvas_w, canvas_h)
        aspect_ratio = ratios[0] if ratios else f"{canvas_w // divisor},{canvas_h // divisor}"
        framing = " ".join(text for text in (CONCAT_VIEW_TEMPLATE, self.view_description.strip()) if text)
        prompt = {
            "cinematography": {"framing": framing},
            "actions": [
                {
                    "time": f"0:00-{rounded // 60}:{rounded % 60:02d}",
                    "description": task,
                    "idle_frame": f"{idle_frames} out of {self.chunk_size + int(self.use_state)}.",
                }
            ],
            "duration": f"{int(duration)}s",
            "fps": float(int(fps)),
            "resolution": {"H": canvas_h, "W": canvas_w},
            "aspect_ratio": aspect_ratio,
        }
        return json.dumps(prompt)

    def get_config(self) -> dict[str, Any]:
        return {
            "view_description": self.view_description,
            "fps": self.fps,
            "chunk_size": self.chunk_size,
            "use_state": self.use_state,
        }


@dataclass
class Cosmos3TokenizerProcessorStep(FlatBatchStep):
    """
    Tokenize the prompts with the tokenizer and chat template of the understanding pathway, as the native
    sequence packer does: one user turn (the Cosmos3 Edge template puts an empty system turn before it), the
    generation prompt, then the end of turn token and `<|vision_start|>`. No special tokens are added, the
    batch is right padded and `mm_token_type_ids` is all zero.

    Training (an action and gradients) replaces each prompt by the empty prompt with probability
    `cfg_dropout_rate`. Inference (no action) also writes the empty prompt as the negative prompt of
    classifier-free guidance (`negative_input_ids`, `negative_attention_mask`, `negative_mm_token_type_ids`).

    The tokenizer is saved with the pipeline, as `tokenizer/` next to the policy config.

    Args:
        tokenizer_name: Hub repo or local directory of the tokenizer.
        cfg_dropout_rate: Probability of training a sample on the empty prompt.
    """

    tokenizer_name: str
    cfg_dropout_rate: float = 0.1

    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        require_package("transformers", extra="cosmos3")
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self._tokenizer.convert_tokens_to_ids(START_OF_GENERATION) == self._tokenizer.unk_token_id:
            raise ValueError(f"The tokenizer of {self.tokenizer_name} has no {START_OF_GENERATION} token.")

    def complementary_data(self, complementary_data):
        prompts = complementary_data.pop("prompt")
        has_action = self.transition.get(TransitionKey.ACTION) is not None
        if has_action and torch.is_grad_enabled() and self.cfg_dropout_rate > 0:
            prompts = ["" if random.random() < self.cfg_dropout_rate else prompt for prompt in prompts]
        input_ids, attention_mask = self.encode(prompts)
        complementary_data["input_ids"] = input_ids
        complementary_data["attention_mask"] = attention_mask
        complementary_data["mm_token_type_ids"] = torch.zeros_like(input_ids)
        if not has_action:
            batch_size = len(prompts)
            negative_ids, negative_mask = self.encode([""])
            complementary_data["negative_input_ids"] = negative_ids.expand(batch_size, -1)
            complementary_data["negative_attention_mask"] = negative_mask.expand(batch_size, -1)
            complementary_data["negative_mm_token_type_ids"] = torch.zeros_like(negative_ids).expand(
                batch_size, -1
            )
        return complementary_data

    def encode(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
        """`input_ids` and `attention_mask` `[batch, text_len]` of a batch of prompts."""
        suffix = self._tokenizer.eos_token + START_OF_GENERATION
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text + suffix)
        tokens = self._tokenizer(
            texts, add_special_tokens=False, padding=True, padding_side="right", return_tensors="pt"
        )
        return tokens["input_ids"], tokens["attention_mask"]

    def get_config(self) -> dict[str, Any]:
        return {"tokenizer_name": self.tokenizer_name, "cfg_dropout_rate": self.cfg_dropout_rate}

    def save_artifacts(self, save_directory: Path) -> dict[str, str]:
        """Save the tokenizer with the pipeline as `tokenizer/`."""
        self._tokenizer.save_pretrained(save_directory / "tokenizer")
        return {"tokenizer_name": "tokenizer"}


def cosmos3_libero_action(action: Tensor) -> Tensor:
    """Native 10D column-rot6d/open-gripper actions to LIBERO's 7D controls."""
    if action.shape[-1] != 10 or not torch.isfinite(action).all():
        raise ValueError("Cosmos3 LIBERO requires finite 10D native actions.")
    axis_angle = matrix_to_axis_angle(rot6d_to_matrix(action[..., 3:9]))
    gripper = (1.0 - 2.0 * action[..., 9:]).clamp(-1.0, 1.0)
    return torch.cat([action[..., :3], axis_angle, gripper], dim=-1)


class Cosmos3Processor:
    """Native Cosmos3 pre/post-processing embedded in the legacy policy interface."""

    def __init__(self, config, dataset_stats=None, pretrained_path=None):
        self.config = config
        self.stats = {}
        self.eps = 1e-8
        root = Path(pretrained_path) if pretrained_path else None
        if root is not None:
            for name in ("policy_preprocessor", "policy_postprocessor"):
                manifest = root / f"{name}.json"
                if not manifest.is_file():
                    raise FileNotFoundError(f"Missing Cosmos3 processor: {manifest}")
                pipeline = json.loads(manifest.read_text())
                allowed = {
                    "rename_observations_processor",
                    "to_batch_processor",
                    "device_processor",
                    "cosmos3_idle_frame_processor",
                    "normalizer_processor",
                    "cosmos3_video_canvas_processor",
                    "cosmos3_prompt_processor",
                    "cosmos3_tokenizer_processor",
                    "unnormalizer_processor",
                }
                for step in pipeline["steps"]:
                    if step["registry_name"] not in allowed:
                        raise ValueError(f"Unsupported Cosmos3 processor: {step['registry_name']}.")
                    if step["registry_name"] == "rename_observations_processor" and step["config"].get(
                        "rename_map"
                    ):
                        raise ValueError("Apply checkpoint observation renaming before the Cosmos3 policy.")
                    if step["registry_name"] not in ("normalizer_processor", "unnormalizer_processor"):
                        continue
                    self.eps = step["config"].get("eps", self.eps)
                    state_file = (root / step["state_file"]).resolve()
                    if not state_file.is_relative_to(root.resolve()):
                        raise ValueError("Processor state must be inside its checkpoint.")
                    loaded = {}
                    for key, value in load_file(str(state_file)).items():
                        feature, statistic = key.rsplit(".", 1)
                        loaded.setdefault(feature, {})[statistic] = value.float()
                    if name == "policy_preprocessor":
                        self.stats = loaded
                    else:
                        for feature, values in loaded.items():
                            if feature in self.stats:
                                for stat, value in values.items():
                                    if stat in self.stats[feature] and not torch.equal(
                                        value.reshape(-1), self.stats[feature][stat].reshape(-1)
                                    ):
                                        raise ValueError(
                                            f"Inconsistent pre/post statistics: {feature}.{stat}."
                                        )
        if dataset_stats is not None:
            for key, values in dataset_stats.items():
                self.stats[key] = {k: torch.as_tensor(v).float() for k, v in values.items()}
        self._validate_stats()
        self.idle = Cosmos3IdleFrameProcessorStep(
            config.action_feature_names, config.fps, config.use_state, config.idle_state
        )
        self.canvas = Cosmos3VideoCanvasProcessorStep(
            list(config.image_features),
            config.view_size,
            config.fps,
            config.chunk_size,
            config.vae.scale_factor_temporal or 1,
            config.vae.scale_factor_spatial or 1,
        )
        self.prompt = Cosmos3PromptProcessorStep(
            config.view_description, config.fps, config.chunk_size, config.use_state
        )
        tokenizer = root / "tokenizer" if root is not None else None
        tokenizer_name = (
            str(tokenizer) if tokenizer is not None and tokenizer.is_dir() else config.tokenizer_name
        )
        self.tokenizer = Cosmos3TokenizerProcessorStep(tokenizer_name, config.cfg_dropout_rate)

    def _validate_stats(self):
        features = {**self.config.input_features, **self.config.output_features}
        required = {
            NormalizationMode.QUANTILES: ("q01", "q99"),
            NormalizationMode.QUANTILE10: ("q10", "q90"),
            NormalizationMode.MEAN_STD: ("mean", "std"),
            NormalizationMode.MIN_MAX: ("min", "max"),
        }
        for key, feature in features.items():
            mode = self.config.normalization_mapping.get(feature.type.value, NormalizationMode.IDENTITY)
            if mode == NormalizationMode.IDENTITY:
                continue
            for statistic in required[mode]:
                value = self.stats.get(key, {}).get(statistic)
                if (
                    value is None
                    or value.numel() != math.prod(feature.shape)
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(f"Missing/nonfinite/wrong-width Cosmos3 statistic: {key}.{statistic}.")
            lo, hi = required[mode]
            if mode != NormalizationMode.MEAN_STD and torch.any(self.stats[key][hi] < self.stats[key][lo]):
                raise ValueError(f"Reversed Cosmos3 normalization bounds: {key}.")
            if mode == NormalizationMode.MEAN_STD and torch.any(self.stats[key]["std"] < 0):
                raise ValueError(f"Negative Cosmos3 standard deviation: {key}.")

    def transform(self, tensor: Tensor, key: str, *, inverse=False) -> Tensor:
        feature = {**self.config.input_features, **self.config.output_features}[key]
        mode = self.config.normalization_mapping.get(feature.type.value, NormalizationMode.IDENTITY)
        if mode == NormalizationMode.IDENTITY:
            return tensor
        stats = {k: v.to(device=tensor.device, dtype=tensor.dtype) for k, v in self.stats[key].items()}
        if mode == NormalizationMode.MEAN_STD:
            return (
                tensor * stats["std"] + stats["mean"]
                if inverse
                else (tensor - stats["mean"]) / (stats["std"] + self.eps)
            )
        low, high = {
            NormalizationMode.QUANTILES: ("q01", "q99"),
            NormalizationMode.QUANTILE10: ("q10", "q90"),
            NormalizationMode.MIN_MAX: ("min", "max"),
        }[mode]
        denominator = stats[high] - stats[low]
        denominator = torch.where(denominator == 0, self.eps, denominator)
        return (
            (tensor + 1) * denominator / 2 + stats[low]
            if inverse
            else (2 * (tensor - stats[low]) / denominator - 1)
        )

    def __call__(self, batch: dict) -> dict:
        batch = dict(batch)
        for key in self.config.image_features:
            image = batch[key]
            if image.ndim == 3:
                image = image[None]
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            if image.dtype != torch.float32 or not torch.isfinite(image).all():
                raise ValueError("Cosmos3 cameras must be uint8 or finite float32 in [0,1].")
            if image.amin() < 0 or image.amax() > 1:
                raise ValueError("Cosmos3 float32 cameras must be in [0,1].")
            batch[key] = image
        device = batch[next(iter(self.config.image_features))].device
        batch = self.idle(batch)
        for key in (*self.config.input_features, *self.config.output_features):
            if key in batch:
                batch[key] = self.transform(batch[key], key)
        batch = self.tokenizer(self.prompt(self.canvas(batch)))
        return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}

    def unnormalize_action(self, action: Tensor) -> Tensor:
        return self.transform(action, ACTION, inverse=True)

    def save_pretrained(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        features = {**self.config.input_features, **self.config.output_features}
        feature_json = {k: {"type": v.type.value, "shape": list(v.shape)} for k, v in features.items()}
        normalization = {
            "eps": self.eps,
            "features": feature_json,
            "norm_map": {k: v.value for k, v in self.config.normalization_mapping.items()},
        }
        pre = [
            {"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
            {"registry_name": "to_batch_processor", "config": {}},
            {"registry_name": "device_processor", "config": {"device": self.config.device}},
            {"registry_name": "cosmos3_idle_frame_processor", "config": self.idle.get_config()},
            {"registry_name": "normalizer_processor", "config": normalization},
            {"registry_name": "cosmos3_video_canvas_processor", "config": self.canvas.get_config()},
            {"registry_name": "cosmos3_prompt_processor", "config": self.prompt.get_config()},
            {
                "registry_name": "cosmos3_tokenizer_processor",
                "config": {
                    **self.tokenizer.get_config(),
                    "tokenizer_name": "tokenizer",
                },
                "artifacts": self.tokenizer.save_artifacts(directory),
            },
            {"registry_name": "device_processor", "config": {"device": self.config.device}},
        ]
        post = [
            {"registry_name": "unnormalizer_processor", "config": normalization},
            {"registry_name": "device_processor", "config": {"device": self.config.device}},
        ]
        tensors = {
            f"{key}.{stat}": value.detach().cpu().contiguous().clone()
            for key, values in self.stats.items()
            for stat, value in values.items()
        }
        for name, steps, index in (("policy_preprocessor", pre, 4), ("policy_postprocessor", post, 0)):
            step = steps[index]
            state_name = f"{name}_step_{index}_{step['registry_name']}.safetensors"
            step["state_file"] = state_name
            save_file(tensors, str(directory / state_name))
            (directory / f"{name}.json").write_text(json.dumps({"name": name, "steps": steps}, indent=2))
