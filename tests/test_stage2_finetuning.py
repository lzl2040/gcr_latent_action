import json
from collections import Counter
from itertools import chain
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.common.datasets.contrastive_dataset import MultiModalContrastiveDataset
from lerobot.common.datasets.contrastive_sampler import ContrastiveBatchSampler, FullCoverageBatchSampler
from lerobot.common.datasets.mixtures import expand_dataset_names, resolve_stage2_training_stage
from lerobot.common.datasets.transforms import ImageTransformsConfig
from lerobot.common.policies.ace.configuration_robo_contrast import RoboContrastConfig
from lerobot.common.policies.qwen3vl_mot.configuration_qwen3vl_mot import Qwen3VLMoTConfig
from lerobot.common.policies.qwen3vl_mot.generation_eval import (
    GenerationEvalConfig,
    GenerationEvaluator,
    _write_sample_index,
)
from lerobot.configs.default import DatasetConfig
from lerobot.scripts.dps_train_contrast import (
    ContrastiveTrainPipelineConfig,
    _make_training_sampler,
    _training_sampler_checkpoint_state,
    _validate_training_sampler_resume,
)
from lerobot.scripts.fsdp_train_contrast import FSDPTrainPipelineConfig

LIBERO_DATASETS = (
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
    "libero_10_no_noops",
)


def _build_libero_dataset(
    tmp_path,
    monkeypatch,
    *,
    stage="auto",
    validation=True,
    episodes=2,
    episode_length=64,
    policy=None,
    missing_dataset=None,
    rgb_aug=False,
):
    roots = tmp_path / "data"
    mapping = {}
    for name in LIBERO_DATASETS:
        mapping[name] = f"{name}_lerobot"
        meta = roots / mapping[name] / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "info.json").write_text(
            json.dumps(
                {
                    "codebase_version": "v3.0",
                    "features": {"action": {"shape": [7]}, "observation.state": {"shape": [8]}},
                }
            )
        )
    if missing_dataset is not None:
        del mapping[missing_dataset]
    mapping_path = tmp_path / "vla2root.json"
    mapping_path.write_text(json.dumps(mapping))
    cfg = SimpleNamespace(
        stage=stage,
        seed=17,
        policy=policy or Qwen3VLMoTConfig(),
        dataset=DatasetConfig(
            repo_id="test",
            image_transforms=ImageTransformsConfig(enable=rgb_aug),
            parent_dir_v21=str(roots),
            parent_dir_v30=str(roots),
            sample_pool_enabled=True,
            sample_pool_root=str(tmp_path / "pools"),
            sample_pool_cache_dir=str(tmp_path / "pool_cache"),
            dataset_size_one_epoch=1,
        ),
    )
    ranges = np.asarray(
        [[index * episode_length, (index + 1) * episode_length] for index in range(episodes)],
        dtype=np.int64,
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(
        MultiModalContrastiveDataset, "_resolve_video_backend", staticmethod(lambda name: name)
    )
    monkeypatch.setattr(
        MultiModalContrastiveDataset,
        "_build_dataset",
        lambda *args: (
            list(range(episodes * episode_length)),
            SimpleNamespace(features={}, fps=20, video_keys=[]),
            {"primary": "image", "wrist": "wrist_image"},
            31,
            20.0,
        ),
    )
    monkeypatch.setattr(MultiModalContrastiveDataset, "_build_norm_stats", lambda *args: {})
    monkeypatch.setattr(MultiModalContrastiveDataset, "_build_episode_ranges", lambda *args: ranges.copy())
    monkeypatch.setattr(MultiModalContrastiveDataset, "_build_unified_meta", lambda *args: SimpleNamespace())
    dataset = MultiModalContrastiveDataset(
        cfg,
        data_mix="libero",
        vla2root_json=str(mapping_path),
        dataset_size_one_epoch=1,
        heldout_eval_dataset_names=("libero",) if validation else (),
        heldout_eval_episodes_per_dataset=2 if validation else 0,
        heldout_eval_seed=17,
    )
    return cfg, dataset


def test_libero_validation_alias_expands_and_deduplicates_actual_members():
    assert expand_dataset_names(("libero", "libero_object_no_noops", "")) == LIBERO_DATASETS
    assert expand_dataset_names(("unknown_dataset",)) == ("unknown_dataset",)
    config = GenerationEvalConfig(enabled=True, datasets="libero,libero_10")
    config.validate(100)
    assert config.dataset_names == LIBERO_DATASETS


def test_named_mixture_expansion_does_not_recurse_into_self_named_members(monkeypatch):
    from lerobot.common.datasets.mixtures import OXE_NAMED_MIXTURES

    monkeypatch.setitem(OXE_NAMED_MIXTURES, "self_named", [("self_named", 1.0)])
    assert expand_dataset_names(("self_named", "self_named")) == ("self_named",)


@pytest.mark.parametrize("config_type", [ContrastiveTrainPipelineConfig, FSDPTrainPipelineConfig])
def test_both_training_entrypoints_default_to_auto(config_type):
    import draccus

    cfg = draccus.parse(
        config_type,
        args=["--dataset.repo_id=test", "--policy.type=qwen3vl_mot"],
    )
    assert cfg.stage == "auto"


@pytest.mark.parametrize(
    "stage,mix,expected",
    [
        ("auto", "libero", "finetune"),
        ("auto", "libero_10", "finetune"),
        ("auto", "debug_research_data", "pretrain"),
        ("pretrain", "libero", "pretrain"),
        ("finetune", "debug_research_data", "finetune"),
    ],
)
def test_training_stage_defaults_preserve_pretraining(stage, mix, expected):
    assert resolve_stage2_training_stage(stage, mix) == expected


def test_auto_validation_uses_downstream_training_mixture():
    finetune = GenerationEvalConfig(enabled=True)
    finetune.resolve_datasets("libero", "finetune")
    assert finetune.dataset_names == LIBERO_DATASETS
    pretrain = GenerationEvalConfig(enabled=True)
    expected = pretrain.dataset_names
    pretrain.resolve_datasets("debug_research_data", "pretrain")
    assert pretrain.dataset_names == expected


def test_libero_finetuning_uses_all_windows_and_never_writes_sample_pools(tmp_path, monkeypatch):
    cfg, dataset = _build_libero_dataset(tmp_path, monkeypatch)
    assert cfg.stage == "finetune"
    assert cfg.dataset.sample_pool_enabled is False
    assert dataset.sample_pool is None
    assert dataset.sample_pool_fingerprint == "disabled"
    assert not (tmp_path / "pools").exists()
    assert not (tmp_path / "pool_cache").exists()
    np.testing.assert_array_equal(dataset.anchor_strides, [1, 1, 1, 1])
    assert dataset.total_pool_anchors == 4 * 2 * 64
    assert cfg.dataset.dataset_size_one_epoch == dataset.total_pool_anchors
    assert dataset.total_source_episodes == 8
    sampler = _make_training_sampler(cfg, dataset, batch_size=8, world_size=1, rank=0)
    assert isinstance(sampler, FullCoverageBatchSampler)
    assert sampler.total_anchors == dataset.total_pool_anchors
    assert all(stat["anchor_ratio"] == 1.0 for stat in dataset.dataset_statistics)
    expected = {(index, frame) for index in range(4) for frame in range(128)}
    sampled = list(chain.from_iterable(sampler))
    assert set(sampled) == expected
    assert len(sampled) == len(expected)


def test_training_validation_does_not_remove_any_episode_and_marks_overlap(tmp_path, monkeypatch):
    _, dataset = _build_libero_dataset(tmp_path, monkeypatch)
    for actual, original in zip(dataset.episode_ranges, dataset.all_episode_ranges, strict=True):
        np.testing.assert_array_equal(actual, original)
    samples = dataset.heldout_validation_samples()
    assert len(samples) == 8
    assert all(sample["validation_split"] == "training" for sample in samples)
    assert all(sample["anchor_stride"] == 1 for sample in samples)
    manifest = dataset.heldout_validation_manifest()
    assert manifest["validation_split"] == "training"
    assert manifest["dataset_names"] == list(LIBERO_DATASETS)
    assert dataset.heldout_sample_pool.storage_dir is None
    assert dataset.heldout_validation_samples() == samples


def test_training_validation_accepts_one_episode_without_holding_it_out(tmp_path, monkeypatch, caplog):
    _, dataset = _build_libero_dataset(tmp_path, monkeypatch, episodes=1)
    assert dataset.total_source_episodes == 4
    assert len(dataset.heldout_validation_samples()) == 4
    assert "training-sample validation uses all of them" in caplog.text


def test_finetuning_and_validation_accept_single_frame_episodes(tmp_path, monkeypatch):
    cfg, dataset = _build_libero_dataset(tmp_path, monkeypatch, episodes=1, episode_length=1)
    assert dataset.total_pool_anchors == 4
    np.testing.assert_array_equal(dataset.heldout_anchor_counts, np.ones(4, dtype=np.int64))
    samples = dataset.heldout_validation_samples()
    assert len(samples) == 4
    assert all(sample["frame_index"] == 0 for sample in samples)
    sampler = _make_training_sampler(cfg, dataset, batch_size=1, world_size=1, rank=0)
    assert set(chain.from_iterable(sampler)) == {(index, 0) for index in range(4)}


def test_explicit_pretraining_keeps_sparse_pools_and_contrastive_sampler(tmp_path, monkeypatch):
    cfg, dataset = _build_libero_dataset(tmp_path, monkeypatch, stage="pretrain", validation=False)
    assert not dataset.full_training_coverage
    assert dataset.sample_pool is not None
    assert (tmp_path / "pools").is_dir()
    assert (dataset.anchor_strides > 1).all()
    sampler = _make_training_sampler(cfg, dataset, batch_size=8, world_size=1, rank=0)
    assert isinstance(sampler, ContrastiveBatchSampler)


def test_stage_one_keeps_contrastive_sampling_even_with_generic_finetune_stage(tmp_path, monkeypatch):
    cfg, dataset = _build_libero_dataset(
        tmp_path,
        monkeypatch,
        stage="finetune",
        validation=False,
        policy=RoboContrastConfig(),
    )
    assert not dataset.full_training_coverage
    assert dataset.sample_pool is not None
    assert isinstance(_make_training_sampler(cfg, dataset, 8, 1, 0), ContrastiveBatchSampler)


@pytest.mark.parametrize("stage_one", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_temporal_rgb_augmentation_initializes_only_for_opted_in_stage_two(
    tmp_path, monkeypatch, stage_one, enabled
):
    _, dataset = _build_libero_dataset(
        tmp_path, monkeypatch, validation=False, rgb_aug=enabled,
        policy=RoboContrastConfig() if stage_one else Qwen3VLMoTConfig(),
    )
    assert (dataset.rgb_augmentation is not None) is (enabled and not stage_one)


def test_pretraining_still_removes_validation_episodes_and_preserves_legacy_manifest(tmp_path, monkeypatch):
    _, dataset = _build_libero_dataset(tmp_path, monkeypatch, stage="pretrain", episodes=3)
    assert dataset.total_source_episodes == 4
    assert dataset.validation_split == "heldout"
    assert "validation_split" not in dataset.heldout_validation_manifest()
    assert all("validation_split" not in sample for sample in dataset.heldout_validation_samples())
    for train, heldout in zip(dataset.episode_ranges, dataset.heldout_episode_ranges, strict=True):
        assert set(map(tuple, train)).isdisjoint(set(map(tuple, heldout)))


@pytest.mark.parametrize("validation", [False, True])
def test_missing_libero_member_is_not_silently_filtered(tmp_path, monkeypatch, validation):
    with pytest.raises(RuntimeError, match="libero_goal_no_noops"):
        _build_libero_dataset(
            tmp_path,
            monkeypatch,
            validation=validation,
            missing_dataset=LIBERO_DATASETS[0],
        )


def _full_sampler(rank=0, world_size=3, seed=17, batch_size=2):
    return FullCoverageBatchSampler(
        episode_ranges=[
            np.asarray([[0, 7], [10, 16]]),
            np.asarray([[0, 9]]),
            np.asarray([[0, 100]]),
        ],
        batch_size=batch_size,
        horizon=[3, 4, 1],
        sample_weights=np.asarray([0.01, 0.99, 0.0]),
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        dataset_names=["a", "b", "disabled"],
    )


def test_distributed_full_epoch_covers_all_anchors_without_weighted_resampling():
    samplers = [_full_sampler(rank) for rank in range(3)]
    batches = [list(sampler) for sampler in samplers]
    global_samples = [
        sample
        for batch_index in range(len(batches[0]))
        for rank in range(3)
        for sample in batches[rank][batch_index]
    ]
    expected = {(0, frame) for frame in [*range(7), *range(10, 16)]} | {(1, frame) for frame in range(9)}
    assert set(global_samples) == expected
    assert len(global_samples) == 24
    assert Counter(global_samples[:22]) == Counter(expected)
    assert global_samples[22:] == global_samples[:2]
    assert all(len(batch) == 2 for rank_batches in batches for batch in rank_batches)


def test_distributed_padding_repeats_only_at_end_of_global_epoch():
    batches = [list(_full_sampler(rank=rank, batch_size=3)) for rank in range(3)]
    global_samples = [
        sample
        for batch_index in range(len(batches[0]))
        for rank in range(3)
        for sample in batches[rank][batch_index]
    ]
    assert len(global_samples) == 27
    assert len(set(global_samples[:22])) == 22
    assert global_samples[22:] == global_samples[:5]


def test_full_epoch_pads_only_final_batch_and_resumes_exactly():
    sampler = FullCoverageBatchSampler(
        [np.asarray([[0, 8]])],
        3,
        horizon=[3],
        sample_weights=np.ones(1),
        seed=17,
    )
    original = list(sampler)
    sampled = list(chain.from_iterable(original))
    assert len(sampled) == 9
    assert len(set(sampled)) == 8
    assert len(set(sampled[:8])) == 8
    assert sampled[8:] == sampled[:1]
    sampler.set_start_batch(1)
    assert list(sampler) == original[1:]
    sampler.set_epoch(1)
    assert sampler.start_batch == 0
    assert len(list(sampler)) == len(original)
    assert list(sampler) != original
    with pytest.raises(ValueError, match="start_batch"):
        sampler.set_start_batch(4)


def test_full_sampler_resume_warns_about_changed_sampling_geometry(caplog):
    sampler = _full_sampler()
    state = _training_sampler_checkpoint_state(sampler)
    _validate_training_sampler_resume(sampler, state)
    assert not caplog.records
    for current, saved in [
        (sampler, {}),
        (_full_sampler(seed=18), state),
        (_full_sampler(world_size=2), state),
        (_full_sampler(batch_size=3), state),
    ]:
        caplog.clear()
        _validate_training_sampler_resume(current, saved)
        assert "sampler changed" in caplog.text
        assert "Continuing with the current sampler" in caplog.text


def test_empty_episode_dataset_fails_explicitly():
    with pytest.raises(ValueError, match="no nonempty"):
        FullCoverageBatchSampler(
            [np.asarray([[0, 0]])],
            2,
            horizon=[3],
            sample_weights=np.ones(1),
        )


@pytest.mark.parametrize(
    "ranges,horizon,weight",
    [
        ([[0, 8]], 3, float("nan")),
        ([[0, 8]], 3, -1.0),
        ([[-1, 8]], 3, 1.0),
        ([[8, 0]], 3, 1.0),
        ([[0, 8]], -1, 1.0),
    ],
)
def test_full_sampler_rejects_invalid_sources(ranges, horizon, weight):
    with pytest.raises(ValueError):
        FullCoverageBatchSampler(
            [np.asarray(ranges)],
            2,
            horizon=[horizon],
            sample_weights=np.asarray([weight]),
        )


def test_validation_artifacts_label_training_set_diagnostics(tmp_path):
    sample = {
        "dataset": "libero_object_no_noops",
        "episode_index": 0,
        "frame_index": 0,
        "fps": 20.0,
        "validation_split": "training",
    }
    _write_sample_index(tmp_path, sample, {})
    assert "Training-set diagnostic" in (tmp_path / "index.html").read_text()
    evaluator = GenerationEvaluator.__new__(GenerationEvaluator)
    evaluator.dataset = SimpleNamespace(validation_split="training")
    evaluator._write_step_index(tmp_path, [])
    assert "not held-out generalization" in (tmp_path / "index.html").read_text()
