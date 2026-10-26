import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as F  # noqa: N812

from lerobot.common.datasets.contrastive_dataset import MultiModalContrastiveDataset
from lerobot.common.datasets.transforms import ImageTransformsConfig, make_temporal_rgb_transform
from lerobot.configs.default import DatasetConfig
from lerobot.scripts.fsdp_train_contrast import FSDPTrainPipelineConfig


def _rgb_sample(multiview):
    image = (torch.arange(3 * 32 * 32).reshape(3, 32, 32) % 201 + 20).to(torch.uint8)
    video = image.unsqueeze(0).repeat(9, 1, 1, 1)
    sample = {
        "image_t0": image.clone(),
        "image_t1": image.clone(),
        "video": video.clone(),
        "action": torch.randn(4, 40),
        "observation.state": torch.randn(4, 40),
        "action_mask": torch.ones(40),
        "state_mask": torch.ones(40),
        "pair_is_valid": torch.tensor(1.0),
        "tactile_image": torch.randint(256, (2, 2, 3, 16, 16), dtype=torch.uint8),
        "tactile_image_mask": torch.ones(2),
        "task": "move the red cup",
    }
    if multiview:
        sample.update({
            "images_t0": torch.stack([image, torch.zeros_like(image), image]),
            "video_views": torch.stack([video, torch.zeros_like(video), video]),
            "image_view_mask": torch.tensor([True, False, True]),
            "video_view_mask": torch.tensor([True, False, True]),
        })
    return sample


def _dataset(sample, enabled):
    dataset = MultiModalContrastiveDataset.__new__(MultiModalContrastiveDataset)
    dataset.dataset_frame_ends = np.array([1])
    dataset.dataset_len = 1
    dataset.datasets = [[sample]]
    dataset.dataset_names = ["synthetic"]
    dataset._to_canonical = lambda item, *_: copy.deepcopy(item)
    dataset.heldout_sample_pool = object()
    dataset.heldout_episode_ranges = [np.array([[0, 1]])]
    dataset.rgb_augmentation = make_temporal_rgb_transform(32) if enabled else None
    return dataset


@pytest.mark.parametrize("multiview", [False, True])
def test_rgb_augmentation_shares_parameters_across_frames_and_aliases(multiview):
    sample = _rgb_sample(multiview)
    original = copy.deepcopy(sample)
    dataset = _dataset(sample, enabled=True)
    transform = dataset.rgb_augmentation
    windows = []

    def record_transform(frames):
        windows.append(frames.clone())
        return transform(frames)

    dataset.rgb_augmentation = record_transform
    torch.manual_seed(47)
    augmented = dataset[0]
    assert len(windows) == (2 if multiview else 1)
    assert windows[0].shape == (11, 3, 32, 32)
    assert not torch.equal(augmented["image_t0"], original["image_t0"])
    torch.testing.assert_close(augmented["video"], augmented["image_t0"].expand_as(augmented["video"]))
    torch.testing.assert_close(augmented["image_t1"], augmented["video"][-1])
    if multiview:
        assert windows[1].shape == (10, 3, 32, 32)
        torch.testing.assert_close(augmented["images_t0"][0], augmented["image_t0"])
        torch.testing.assert_close(augmented["video_views"][0], augmented["video"])
        for view in (0, 2):
            torch.testing.assert_close(
                augmented["video_views"][view],
                augmented["images_t0"][view].expand_as(augmented["video_views"][view]),
            )
        assert not torch.equal(augmented["images_t0"][0], augmented["images_t0"][2])
        assert not augmented["images_t0"][1].any()
        assert not augmented["video_views"][1].any()
        for key in ("image_view_mask", "video_view_mask"):
            torch.testing.assert_close(augmented[key], original[key])
    for key, value in original.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(sample[key], value)
        if key.startswith(("image", "video")):
            assert augmented[key].dtype == value.dtype
            assert augmented[key].shape == value.shape
        else:
            assert augmented[key] is not None
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(augmented[key], value)
            else:
                assert augmented[key] == value


@pytest.mark.parametrize("multiview", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_rgb_validation_is_unaugmented_even_when_training_augmentation_is_enabled(multiview, enabled):
    sample = _rgb_sample(multiview)
    dataset = _dataset(sample, enabled)
    before = torch.random.get_rng_state()
    first = dataset.get_heldout_item(0, 0)
    second = dataset.get_heldout_item(0, 0)
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(first[key], value)
            torch.testing.assert_close(second[key], value)
        else:
            assert first[key] == second[key] == value


@pytest.mark.parametrize("multiview", [False, True])
def test_disabling_rgb_augmentation_preserves_training_data_and_rng(multiview):
    sample = _rgb_sample(multiview)
    dataset = _dataset(sample, enabled=False)
    before = torch.random.get_rng_state()
    result = dataset[0]
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(result[key], value)
        else:
            assert result[key] == value


def test_rgb_augmentation_has_only_crop_and_color_jitter():
    crop, jitter = make_temporal_rgb_transform(32).transforms
    assert isinstance(crop, v2.RandomResizedCrop)
    assert crop.scale == (0.9, 1.0)
    assert crop.ratio == (1.0, 1.0)
    assert isinstance(jitter, v2.ColorJitter)
    assert jitter.brightness == (0.9, 1.1)
    assert jitter.contrast == (0.9, 1.1)
    assert jitter.saturation == (0.95, 1.05)
    assert jitter.hue is None


def test_rgb_geometry_only_crops_the_entire_window(monkeypatch):
    transform = make_temporal_rgb_transform(32)
    crop, _ = transform.transforms
    monkeypatch.setattr(crop, "make_params", lambda _: {"top": 1, "left": 0, "height": 31, "width": 31})
    transform.transforms[1] = v2.Identity()
    original = _rgb_sample(False)["video"]
    augmented = transform(original)
    expected = F.resized_crop(
        original, 1, 0, 31, 31, size=[32, 32], interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    torch.testing.assert_close(augmented, expected)
    assert not torch.equal(augmented, original)
    torch.testing.assert_close(augmented, augmented[0].expand_as(augmented))


def test_rgb_augmentation_preserves_missing_primary_camera():
    sample = _rgb_sample(True)
    for key in ("image_t0", "image_t1", "video"):
        sample[key].zero_()
    sample["images_t0"][0].zero_()
    sample["video_views"][0].zero_()
    sample["image_view_mask"][0] = sample["video_view_mask"][0] = False
    augmented = _dataset(sample, enabled=True)[0]
    for key in ("image_t0", "image_t1", "video"):
        assert not augmented[key].any()
    assert not augmented["images_t0"][:2].any()
    assert not augmented["video_views"][:2].any()
    torch.testing.assert_close(augmented["images_t0"][2], augmented["video_views"][2, 0])


@pytest.mark.parametrize("multiview", [False, True])
@pytest.mark.parametrize("floating_decoder", [False, True])
def test_rgb_augmentation_runs_after_real_canonical_extraction(multiview, floating_decoder):
    video = _rgb_sample(False)["video"]
    source = {
        "camera": video.float() / 255 if floating_decoder else video.clone(),
        "wrist": video.float() / 255 if floating_decoder else video.clone(),
        "action": torch.randn(32, 4),
        "observation.state": torch.randn(32, 4),
        "task": "move the red cup",
    }
    dataset = _dataset(source, enabled=True)
    del dataset._to_canonical
    dataset.cfg = SimpleNamespace(dataset=DatasetConfig(
        repo_id="test", image_transforms=ImageTransformsConfig(enable=True, img_size=32),
    ))
    dataset.chunk_size = 32
    dataset.world_video_frames = 9
    dataset.rgb_views = ("primary", "secondary", "wrist") if multiview else ("primary",)
    dataset.image_key_maps = [{"primary": "camera", "secondary": None, "wrist": "wrist"}]
    dataset.specs = [{"action": [("action", 0, 4, 0)], "state": [("observation.state", 0, 4, 0)]}]
    vector_stats = {
        "mean": torch.zeros(40), "std": torch.ones(40),
        "mask": torch.cat([torch.ones(4), torch.zeros(36)]),
    }
    dataset.norm_stats = [{"action": vector_stats, "state": vector_stats, "tactile_signal": {}}]
    dataset.true_fps = [20]
    dataset.max_tactile_views = dataset.tactile_frames = 2
    dataset.tactile_img_size = 16
    dataset.tactile_dead_std = 0
    dataset.tactile_sensor_meta = [(
        torch.full((2,), -1), torch.zeros(2, 3), torch.ones(2, 3),
    )]
    torch.manual_seed(47)
    training = dataset[0]
    validation = dataset.get_heldout_item(0, 0)
    torch.testing.assert_close(validation["video"], video)
    assert training["video"].dtype == torch.uint8
    assert not torch.equal(training["video"], validation["video"])
    torch.testing.assert_close(training["image_t0"], training["video"][0])
    torch.testing.assert_close(training["image_t1"], training["video"][-1])
    for key in ("action", "action_mask", "observation.state", "state_mask", "tactile_image"):
        torch.testing.assert_close(training[key], validation[key])
    if multiview:
        torch.testing.assert_close(training["video"], training["video_views"][0])
        torch.testing.assert_close(training["image_t0"], training["images_t0"][0])
        assert not training["video_views"][1].any()
        torch.testing.assert_close(validation["video_views"][2], video)


@pytest.mark.parametrize("enabled", [False, True])
def test_rgb_augmentation_uses_the_existing_dataset_enable_flag(enabled):
    import draccus

    cfg = draccus.parse(FSDPTrainPipelineConfig, args=[
        "--dataset.repo_id=test", "--policy.type=qwen3vl_mot",
        f"--dataset.image_transforms.enable={str(enabled).lower()}",
    ])
    assert cfg.dataset.image_transforms.enable is enabled
    assert DatasetConfig(repo_id="test").image_transforms.enable is False


@pytest.mark.parametrize("launcher", ["train_qwen3vl_mot_fsdp.sh", "train_qwen3vl_mot_local.sh"])
@pytest.mark.parametrize("setting", [None, "false", "true", "invalid"])
def test_rgb_augmentation_launcher_flag(tmp_path, launcher, setting):
    root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "model-cache-stub"
    python_stub.write_text("#!/usr/bin/env bash\nprintf '%s\\n' /mock/qwen /mock/cosmos /mock/generation\n")
    python_stub.chmod(0o755)
    deepspeed_stub = bin_dir / "deepspeed"
    deepspeed_stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    deepspeed_stub.chmod(0o755)
    env = {
        **os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PYTHON_BIN": sys.executable if "fsdp" in launcher else str(python_stub),
        "CHECK_PATHS": "false", "DRY_RUN": "true", "WEIGHT_RESUME": "false",
        "STAGE1_CHECKPOINT": str(tmp_path / "stage1"), "OUTPUT_DIR": str(tmp_path / "run"),
        "NNODES": "1", "NPROC_PER_NODE": "1", "MASTER_PORT": "29599",
        "STAGE2_TASK_NAMES": "", "STAGE2_TASK_WEIGHTS": "",
    }
    env.pop("RGB_AUG_ENABLED", None)
    if setting is not None:
        env["RGB_AUG_ENABLED"] = setting
    result = subprocess.run(
        ["bash", str(root / launcher)], cwd=root, env=env,
        capture_output=True, text=True, timeout=30,
    )
    if setting == "invalid":
        assert result.returncode != 0
        assert "RGB_AUG_ENABLED must be true or false" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert f"--dataset.image_transforms.enable={setting or 'false'}" in result.stdout
        if "fsdp" in launcher:
            assert "crop 90-100%, mild color jitter; no rotation" in result.stdout
