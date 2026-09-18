import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.common.datasets.contrastive_dataset import MultiModalContrastiveDataset
from lerobot.common.datasets import contrastive_eval
from lerobot.common.datasets.mixtures import OXE_NAMED_MIXTURES
from lerobot.common.policies.ace import modeling_robo_contrast
from lerobot.common.policies.ace.configuration_robo_contrast import RoboContrastConfig
from lerobot.common.policies.ace.modeling_robo_contrast import RoboContrast


class _DummyPerception(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, image_t0, image_t1, texts, has_text=None, probe=False):
        embedding = self.proj(image_t0.float())
        reconstruction = (self.proj(image_t1.float()) - image_t1.float()).square().mean()
        return embedding, reconstruction, {}


class _DummyPhysical(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.tactile_signal_gate = nn.Parameter(torch.zeros(()))
        self.tactile_image_gate = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return self.proj(batch["physical_input"]), None


def _model(perception_recon_weight: float = 1.0) -> RoboContrast:
    model = RoboContrast.__new__(RoboContrast)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        perception_recon_weight=perception_recon_weight,
        false_negative_frame_gap=0,
        query_probe_freq=0,
        tactile_recon_weight=0.0,
    )
    model.perception_encoder = _DummyPerception()
    model.physical_encoder = _DummyPhysical()
    model.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
    model._max_logit_scale = math.log(100.0)
    return model


def _batch(has_physical: list[float]) -> dict:
    batch_size = len(has_physical)
    generator = torch.Generator().manual_seed(7)
    return {
        "image_t0": torch.randn(batch_size, 4, generator=generator, requires_grad=True),
        "image_t1": torch.randn(batch_size, 4, generator=generator),
        "physical_input": torch.randn(
            batch_size,
            4,
            generator=generator,
            requires_grad=True,
        ),
        "task": [""] * batch_size,
        "has_text": torch.zeros(batch_size),
        "has_physical": torch.tensor(has_physical),
        "episode_uid": torch.arange(batch_size),
        "frame_index": torch.arange(batch_size) * 100,
        "tactile_image_mask": torch.zeros(batch_size, 1),
        "tactile_signal_mask": torch.zeros(batch_size),
    }


def test_pure_video_batch_skips_physical_encoder_and_trains_perception() -> None:
    model = _model()
    batch = _batch([0, 0, 0, 0])

    loss, metrics = model(batch, task_type="train_contrastive")
    loss.backward()

    assert model.physical_encoder.calls == 0
    assert all(parameter.grad is None for parameter in model.physical_encoder.parameters())
    assert model.perception_encoder.proj.weight.grad is not None
    assert metrics["contrastive_loss"] == 0.0
    assert metrics["physical_rows"] == 0.0
    assert metrics["video_only_rows"] == 4.0


def test_video_rows_are_excluded_from_both_sides_of_contrastive_loss() -> None:
    model = _model(perception_recon_weight=0.0)
    batch = _batch([1, 0, 1, 0])

    loss, metrics = model(batch, task_type="train_contrastive")
    loss.backward()

    assert model.physical_encoder.calls == 1
    assert batch["physical_input"].grad[[0, 2]].abs().sum() > 0
    assert batch["physical_input"].grad[[1, 3]].abs().sum() == 0
    assert batch["image_t0"].grad[[0, 2]].abs().sum() > 0
    assert batch["image_t0"].grad[[1, 3]].abs().sum() == 0
    assert metrics["physical_rows"] == 2.0
    assert metrics["video_only_rows"] == 2.0


def test_all_physical_batch_preserves_original_symmetric_infonce() -> None:
    model = _model(perception_recon_weight=0.0)
    batch = _batch([1, 1, 1, 1])
    del batch["has_physical"]

    perception, _, _ = model.encode_perception(batch)
    physical, _ = model.encode_physical(batch)
    scale = model.logit_scale.clamp(max=model._max_logit_scale).exp()
    labels = torch.arange(4)
    expected = 0.5 * (
        F.cross_entropy(scale * perception @ physical.t(), labels)
        + F.cross_entropy(scale * physical @ perception.t(), labels)
    )

    actual, metrics = model(batch, task_type="train_contrastive")

    torch.testing.assert_close(actual, expected)
    assert metrics["physical_rows"] == 4.0
    assert metrics["video_only_rows"] == 0.0


def test_pure_video_batch_requires_perception_reconstruction() -> None:
    model = _model(perception_recon_weight=0.0)

    with pytest.raises(RuntimeError, match="perception reconstruction"):
        model(_batch([0, 0]), task_type="train_contrastive")


def test_perception_only_model_freezes_reconstruction_unused_heads(
    monkeypatch,
) -> None:
    class TinyPerception(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.shared = nn.Linear(2, 2)
            self.out_proj = nn.Linear(2, 2)
            self.query_pool = nn.Parameter(torch.ones(2, 1))

    monkeypatch.setattr(modeling_robo_contrast, "PerceptionEncoder", TinyPerception)
    model = RoboContrast(RoboContrastConfig(perception_only=True))

    assert model.physical_encoder is None
    assert model.perception_encoder.shared.weight.requires_grad
    assert not any(
        parameter.requires_grad
        for parameter in model.perception_encoder.out_proj.parameters()
    )
    assert not model.perception_encoder.query_pool.requires_grad
    assert not model.logit_scale.requires_grad


def test_contrastive_dataset_keeps_perception_only_sources() -> None:
    class FakeDataset:
        def __init__(self):
            self.meta = SimpleNamespace(stats={})

        def __len__(self):
            return 100

    class DatasetUnderTest(MultiModalContrastiveDataset):
        def _build_dataset(self, cfg, dataset_name, data_root, version, spec):
            meta = SimpleNamespace(
                features={
                    "observation.images.ego": {
                        "dtype": "video",
                        "shape": (32, 32, 3),
                        "names": ["height", "width", "channel"],
                    }
                },
                video_keys=["observation.images.ego"],
                image_keys=[],
                fps=10,
            )
            dataset = FakeDataset()
            return (
                dataset,
                meta,
                {"primary": "observation.images.ego", "secondary": None, "wrist": None},
                4,
                10.0,
            )

        @staticmethod
        def _build_episode_ranges(dataset, version):
            return np.asarray([[0, 100]], dtype=np.int64)

        def _build_unified_meta(self, cfg, meta_features):
            return SimpleNamespace(features={})

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset_root = root / "Ego10K_part1"
        (dataset_root / "meta").mkdir(parents=True)
        (dataset_root / "meta" / "info.json").write_text(
            json.dumps({"codebase_version": "v2.1", "features": {}})
        )
        roots = root / "vla2root.json"
        roots.write_text(json.dumps({"ego10k_part1": "Ego10K_part1"}))
        cfg = SimpleNamespace(
            policy=SimpleNamespace(
                chunk_size=4,
                window_mode="frames",
                chunk_seconds=1.0,
                chunk_frames_min=1,
                chunk_frames_max=8,
                frame_horizon=None,
                tactile_img_size=32,
                tactile_frames=2,
                tactile_dead_std=0.0,
                max_tactile_views=1,
                use_wrist_image=False,
            ),
            dataset=SimpleNamespace(
                video_backend="pyav",
                parent_dir_v21=str(root),
                parent_dir_v30="",
                parent_dir_extra="",
                image_transforms=SimpleNamespace(img_size=32),
            ),
        )

        mix_name = "__video_only_test"
        OXE_NAMED_MIXTURES[mix_name] = [("ego10k_part1", 1.0)]
        try:
            dataset = DatasetUnderTest(
                cfg,
                data_mix=mix_name,
                vla2root_json=str(roots),
                dataset_size_one_epoch=16,
            )
        finally:
            del OXE_NAMED_MIXTURES[mix_name]

    assert dataset.dataset_names == ["ego10k_part1"]
    assert dataset.has_physical.tolist() == [False]
    assert dataset.dataset_statistics[0]["training"] == "perception-only"


def test_fixed_contrastive_eval_excludes_video_only_datasets() -> None:
    captured = {}
    original = contrastive_eval._draw_batches

    def capture(dataset, weights, *args, **kwargs):
        captured[len(captured)] = weights.copy()
        return []

    contrastive_eval._draw_batches = capture
    try:
        dataset = SimpleNamespace(
            sample_weights=np.asarray([0.2, 0.3, 0.5]),
            has_physical=np.asarray([True, False, True]),
            has_tactile=np.asarray([False, False, False]),
            frame_horizons=[4, 4, 4],
        )
        contrastive_eval.build_eval_loaders(
            dataset=dataset,
            policy_cfg=SimpleNamespace(
                same_dataset_frac=0.75,
                episode_group_frac=0.75,
                episode_group_size=8,
                min_frame_gap=32,
            ),
            collate_fn=lambda batch: batch,
            num_workers=0,
        )
    finally:
        contrastive_eval._draw_batches = original

    np.testing.assert_allclose(captured[0], [2 / 7, 0, 5 / 7])
