"""Native Cosmos3 windows over this repository's LIBERO fine-tuning datasets."""

import logging

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot.common.datasets.contrastive_dataset import MultiModalContrastiveDataset

LOGGER = logging.getLogger(__name__)
CAMERAS = ("observation.images.image", "observation.images.image2")
ACTION_NAMES = [*(f"pos_{i}" for i in range(3)), *(f"rot6d_{i}" for i in range(6)), "gripper_0"]


def libero_actions_to_cosmos3(actions, layout="libero_euler_interleaved"):
    """Undo the old Euler encoding, then encode native axis-angle rotations as column rot6d."""
    array = np.asarray(actions, dtype=np.float64)
    if array.shape[-1] != 10 or not np.isfinite(array).all():
        raise ValueError("Cosmos3 LIBERO fine-tuning requires finite 10D source actions.")
    if layout == "cosmos3_native":
        return array.astype(np.float32)
    if layout != "libero_euler_interleaved":
        raise ValueError(f"Unknown Cosmos3 training action layout: {layout!r}.")
    flat = array.reshape(-1, 10)
    columns = flat[:, 3:9].reshape(-1, 3, 2)
    matrix = np.concatenate((columns, np.cross(columns[:, :, 0], columns[:, :, 1])[:, :, None]), axis=-1)
    controls = Rotation.from_matrix(matrix).as_euler("xyz")
    native = Rotation.from_rotvec(controls).as_matrix()
    result = flat.copy()
    result[:, 3:9] = np.concatenate((native[:, :, 0], native[:, :, 1]), axis=-1)
    return result.reshape(array.shape).astype(np.float32)


class Cosmos3LiberoDataset(MultiModalContrastiveDataset):
    """Reuse local v2/v3 loaders and full-coverage sampling, without canonical normalization."""

    def __init__(self, cfg, **kwargs):
        if cfg.stage != "finetune":
            raise ValueError("The Cosmos3 local mixture adapter supports the finetune stage.")
        if cfg.policy.use_state:
            raise ValueError(
                "The local LIBERO adapter follows native use_state=false; state actions are unsupported."
            )
        super().__init__(cfg, **kwargs)
        if not all(name.startswith("libero_") for name in self.dataset_names):
            raise ValueError("The Cosmos3 local mixture adapter currently supports LIBERO datasets only.")
        self.total_source_frames = sum(self.dataset_sizes)
        LOGGER.info(
            "Cosmos3: %d unpadded %d-action/%d-image windows; source action layout=%s; native quantile statistics.",
            self.total_pool_anchors,
            cfg.policy.chunk_size,
            cfg.policy.chunk_size + 1,
            cfg.policy.training_action_layout,
        )

    def _split_heldout_eval_episodes(self, requested, count, seed):
        if requested or count:
            raise ValueError("The MoT generation evaluator is not a Cosmos3 evaluator.")
        trimmed = []
        for ranges in self.all_episode_ranges:
            spans = ranges.copy()
            spans[:, 1] = np.maximum(spans[:, 0], spans[:, 1] - self.chunk_size)
            trimmed.append(spans)
        return trimmed, [np.empty((0, 2), dtype=np.int64) for _ in trimmed], []

    def _build_unified_meta(self, cfg, meta_features):
        meta = super()._build_unified_meta(cfg, meta_features)
        features = meta.info["features"]
        primary = features["observation.images.primary"]
        wrist = features["observation.images.wrist"]
        for key in list(features):
            if key.startswith("observation."):
                del features[key]
        features[CAMERAS[0]] = primary
        features[CAMERAS[1]] = wrist
        features["action"] = {"dtype": "float32", "shape": (10,), "names": ACTION_NAMES}
        if any(fps != self.true_fps[0] for fps in self.true_fps):
            raise ValueError("Cosmos3 local LIBERO datasets must share one real frame rate.")
        meta.info["fps"] = float(self.true_fps[0])
        rows = []
        for dataset in self.datasets:
            column = dataset.hf_dataset.select_columns(["action"]).with_format("numpy")["action"]
            rows.append(libero_actions_to_cosmos3(column, cfg.policy.training_action_layout))
        actions = np.concatenate(rows, axis=0)
        meta.stats = {
            "action": {
                "q01": np.quantile(actions, 0.01, axis=0).astype(np.float32),
                "q99": np.quantile(actions, 0.99, axis=0).astype(np.float32),
                "mean": actions.mean(axis=0),
                "std": actions.std(axis=0),
                "min": actions.min(axis=0),
                "max": actions.max(axis=0),
            }
        }
        return meta

    def _to_canonical(self, item, ds_idx, frame_idx):
        result = super()._to_canonical(item, ds_idx, frame_idx)
        if not result["video_view_mask"].all():
            raise ValueError("Cosmos3 LIBERO needs both real camera clips.")
        for key in (self.image_key_maps[ds_idx][role] for role in self.rgb_views):
            if item.get(f"{key}_is_pad", torch.zeros(1, dtype=torch.bool)).any():
                raise ValueError("A padded video reached the native Cosmos3 training adapter.")
        action = item["action"][: self.chunk_size]
        if (
            action.shape != (self.chunk_size, 10)
            or item.get("action_is_pad", torch.zeros(1, dtype=torch.bool)).any()
        ):
            raise ValueError("A padded/wrong-width action reached the native Cosmos3 training adapter.")
        output = {
            key: result[key]
            for key in (
                "task",
                "has_physical",
                "dataset_id",
                "episode_uid",
                "frame_index",
                "sample_rate",
                "tactile_image_mask",
            )
        }
        output["action"] = torch.from_numpy(
            libero_actions_to_cosmos3(action.numpy(), self.cfg.policy.training_action_layout)
        )
        for key, video in zip(CAMERAS, result["video_views"], strict=True):
            output[key] = video
        return output

    def _augment_rgb_window(self, result):
        for key in CAMERAS:
            result[key] = self.rgb_augmentation(result[key])
        return result
