"""Model-side adaptation of the shared, unnormalized canonical dataset."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot.common.datasets.canonical_space import CANON_DIM

CAMERAS = ("observation.images.image", "observation.images.image2", "observation.images.image3")


def libero_actions_to_cosmos3(actions, layout="libero_euler_interleaved"):
    """Undo legacy LIBERO's Euler-control encoding before writing native column rot6d."""
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


def canonical_to_cosmos3(values, mask, *, libero=False, native=False):
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    result = values.copy()
    if not native:
        for start in (3, 13):
            valid = mask[start : start + 6]
            if valid.any() and not valid.all():
                raise ValueError("Cosmos3 needs complete six-channel rotation groups.")
            if valid.all():
                result[..., start : start + 6] = values[..., start + np.array([0, 2, 4, 1, 3, 5])]
        if libero:
            if not mask[:10].all() or mask[10:].any():
                raise ValueError("Legacy LIBERO actions must occupy exactly canonical eef0.")
            result[..., :10] = libero_actions_to_cosmos3(values[..., :10])
    return result


def _channel_names():
    names = []
    for arm in range(2):
        prefix = "" if arm == 0 else "arm1_"
        names += [f"{prefix}pos_{i}" for i in range(3)]
        names += [f"{prefix}rot6d_{i}" for i in range(6)]
        names += [f"{prefix}gripper_0"]
    for arm in range(2):
        names += [f"arm{arm}_joint_{i}" for i in range(7)]
        names += [f"arm{arm}_joint_gripper_0"]
    return names


class Cosmos3DataAdapter:
    """Translate shared samples and metadata, without owning readers, indexing or sampling."""

    def __init__(self, dataset, config):
        if dataset.cfg.stage != "finetune":
            raise ValueError("The Cosmos3 shared-data adapter requires stage=finetune.")
        if dataset.normalize_physical or not dataset.require_full_window:
            raise ValueError("Cosmos3 needs unnormalized physical data and complete windows.")
        self.names = list(dataset.dataset_names)
        self.fps = list(dataset.true_fps)
        self.config = config
        self.views = tuple(config.rgb_views)
        self.legacy = []
        slots = set()
        for index, name in enumerate(self.names):
            keys = [dataset.image_key_maps[index].get(role) for role in self.views]
            if None in keys or len(set(keys)) != len(keys):
                raise ValueError(f"{name} is missing distinct cameras for {self.views}.")
            legacy = name.startswith("libero_") and config.training_action_layout != "cosmos3_native"
            if config.training_action_layout == "libero_euler_interleaved" and not legacy:
                raise ValueError(f"{name} is not legacy LIBERO; use training_action_layout=auto.")
            self.legacy.append(legacy)
            for kind in ("action", "state") if config.use_state else ("action",):
                features = dataset.datasets[index].meta.features
                for src, start, end, destination in dataset.specs[index].get(kind, []):
                    if src in features and np.prod(features[src]["shape"]) >= end:
                        slots.update(range(destination, destination + end - start))
        self.slots = sorted(slots)
        if not self.slots or max(self.slots) >= 36 or len(self.slots) > config.max_action_dim:
            raise ValueError("Cosmos3 needs nonempty supported canonical slots within max_action_dim.")
        saved = config.training_canonical_slots
        if saved is not None and list(saved) != self.slots:
            raise ValueError("The dataset's canonical slots changed from the saved Cosmos3 contract.")
        config.training_canonical_slots = self.slots
        self._adapt_metadata(dataset)

    def _convert(self, values, mask, dataset_idx, *, action):
        return canonical_to_cosmos3(
            values,
            mask,
            libero=action and self.legacy[dataset_idx],
            native=self.config.training_action_layout == "cosmos3_native",
        )

    def _adapt_metadata(self, dataset):
        fields = {"action": "action"}
        if self.config.use_state:
            fields["observation.state"] = "state"
        stats = {}
        for key, kind in fields.items():
            columns = [[] for _ in self.slots]
            for index, name in enumerate(self.names):
                raw, mask = dataset.get_raw_physical_values(index, kind)
                if not mask.any():
                    raise ValueError(f"{name} has no usable {kind} features.")
                if raw.shape[0] == 0:
                    continue
                values = self._convert(raw.numpy(), mask.numpy(), index, action=kind == "action")
                for channel, slot in enumerate(self.slots):
                    if mask[slot]:
                        columns[channel].append(values[:, slot].copy())
                del raw, values
            projected = {name: [] for name in ("q01", "q99", "mean", "std", "min", "max")}
            for parts in columns:
                values = np.concatenate(parts) if parts else np.array([-1.0, 1.0])
                low, high = np.quantile(values, [0.01, 0.99])
                summaries = (
                    low,
                    high,
                    values.mean(),
                    values.std(),
                    values.min(),
                    values.max(),
                )
                for statistic, value in zip(projected.values(), summaries, strict=True):
                    statistic.append(value)
            stats[key] = {name: np.asarray(values, dtype=np.float32) for name, values in projected.items()}
        meta = dataset.meta
        features = meta.info["features"]
        image = features["observation.images.primary"]
        for key in list(features):
            if key.startswith("observation."):
                del features[key]
        for camera in CAMERAS[: len(self.views)]:
            features[camera] = dict(image)
        names = [_channel_names()[slot] for slot in self.slots]
        features["action"] = {"dtype": "float32", "shape": (len(self.slots),), "names": names}
        if self.config.use_state:
            features["observation.state"] = dict(features["action"])
        meta.stats = stats
        meta.info["fps"] = float(self.fps[0])

    def __call__(self, sample):
        index = int(sample["dataset_id"])
        output = {
            key: sample[key]
            for key in (
                "task",
                "has_physical",
                "dataset_id",
                "episode_uid",
                "frame_index",
                "sample_rate",
                "tactile_image_mask",
                "data_read_fallback",
                "requested_frame_index",
            )
            if key in sample
        }
        videos = sample.get("video_views")
        valid = sample.get("video_view_mask")
        if videos is None:
            videos = sample["video"].unsqueeze(0)
            valid = sample["pair_is_valid"]
        if not valid.all():
            raise ValueError(f"{self.names[index]} has missing/incomplete RGB clips.")
        for camera, video in zip(CAMERAS[: len(self.views)], videos, strict=True):
            if video.shape[0] != self.config.chunk_size + 1:
                raise ValueError("Cosmos3 needs chunk_size + 1 real video frames.")
            output[camera] = video
        for key, mask_key, action in (
            ("action", "action_mask", True),
            ("observation.state", "state_mask", False),
        ):
            if not action and not self.config.use_state:
                continue
            values, mask = sample[key], sample[mask_key]
            if action and values.shape != (self.config.chunk_size, CANON_DIM):
                raise ValueError("Cosmos3 needs a complete canonical action chunk.")
            if not mask.any():
                raise ValueError(f"{self.names[index]} has no valid {key}.")
            if not action:
                values = values[0]
            native = self._convert(values.numpy(), mask.numpy(), index, action=action)
            output[key] = torch.from_numpy(native[..., self.slots])
            output[mask_key] = mask[self.slots].bool()
        output["sample_fps"] = torch.tensor(self.fps[index], dtype=torch.float32)
        return output
