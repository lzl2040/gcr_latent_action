import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.common.datasets.contrastive_dataset import MultiModalContrastiveDataset
from lerobot.common.datasets.physical_history import PhysicalHistoryBuffer
from lerobot.common.datasets.visual_history import VisualHistoryBuffer
from lerobot.common.policies.qwen3vl_mot.configuration_qwen3vl_mot import Qwen3VLMoTConfig
from lerobot.common.policies.qwen3vl_mot.modeling_qwen3vl_mot import Qwen3VLMoTPolicy
from lerobot.scripts.dps_train_contrast import _load_stage2_resume_policy_config, move_batch


def history_config(**kwargs):
    return Qwen3VLMoTConfig(
        **{
            "physical_encoder_type": "mlp",
            "physical_flow_space": "raw",
            "physical_tuning_mode": "full",
            "use_tactile_conditioning": False,
            "task_names": ("va_prediction",),
            "task_weights": (1.0,),
            "latent_action_loss_weight": 0,
            "rgb_views": ("primary", "wrist"),
            "visual_history_offsets": (-4, -2),
            "visual_history_include_first": True,
            **kwargs,
        }
    )


def pixel(index):
    return torch.full((3, 4, 4), index, dtype=torch.uint8)


@pytest.fixture(params=[False, True], ids=["image", "video"])
def history_dataset(request):
    class Table(list):
        selected_columns = None

        def select_columns(self, columns):
            self.selected_columns = columns
            return [{key: row[key] for key in columns} for row in self]

    class RawDataset:
        def __init__(self):
            self.hf_dataset = Table(
                [
                    {
                        "episode_index": torch.tensor(index // 10),
                        "timestamp": torch.tensor((index % 10) / 12.5),
                        "camera": pixel(index),
                    }
                    for index in range(20)
                ]
            )
            self.meta = SimpleNamespace(video_keys=["camera"] if request.param else [])
            self.reads, self.queries = [], []
            self.fail_index = None

        def __len__(self):
            return 20

        def __getitem__(self, index):
            self.reads.append(index)
            if index == self.fail_index:
                raise RuntimeError("broken current frame")
            return self.hf_dataset[index]

        def _query_videos(self, queries, episode):
            self.queries.append(queries)
            return {
                key: torch.stack([pixel(episode * 10 + round(stamp * 12.5)) for stamp in stamps]).squeeze(0)
                for key, stamps in queries.items()
            }

    dataset = MultiModalContrastiveDataset.__new__(MultiModalContrastiveDataset)
    dataset.cfg = SimpleNamespace(
        policy=history_config(), dataset=SimpleNamespace(image_transforms=SimpleNamespace(img_size=4))
    )
    dataset.visual_history_enabled = True
    dataset.rgb_views = ("primary", "wrist")
    dataset.datasets = [RawDataset()]
    dataset.dataset_names = ["history_test"]
    dataset.image_key_maps = [{"primary": "camera", "wrist": None}]
    dataset.all_episode_ranges = [np.array([[0, 10], [10, 20]])]
    dataset.true_fps = [12.5]
    dataset.rgb_augmentation = None

    def canonical(item, ds_idx, frame_idx):
        image = item["camera"]
        end = min(frame_idx + 8, (frame_idx // 10 + 1) * 10 - 1)
        video = torch.stack([pixel(min(frame_idx + offset, end)) for offset in range(9)])
        images = torch.stack([image, torch.zeros_like(image)])
        return {
            "image_t0": image,
            "image_t1": pixel(end),
            "pair_is_valid": torch.tensor(1.0),
            "images_t0": images,
            "image_view_mask": torch.tensor([True, False]),
            "video": video,
            "video_views": torch.stack([video, torch.zeros_like(video)]),
            "frame_index": torch.tensor(frame_idx),
        }

    dataset._to_canonical = canonical
    return dataset


def read(dataset, index, **kwargs):
    return dataset._get_item(0, index, fallback_pool=None, fallback_ranges=None, **kwargs)


def test_training_context_matches_per_frame_inference_and_episode_padding(history_dataset):
    buffer = VisualHistoryBuffer((-4, -2), True)
    for step in range(8):
        sample = read(history_dataset, 10 + step)
        current = {key: sample[key][None] for key in ("image_t0", "images_t0", "image_view_mask")}
        online = buffer.update(current)
        for key in ("context_images", "context_image_mask", "context_frame_offsets"):
            torch.testing.assert_close(sample[key], online[key][0])
        assert sample["context_fps"].item() == 12.5
        torch.testing.assert_close(sample["video"][0], sample["image_t0"])
        assert sample["video"][-1, 0, 0, 0] == min(10 + step + 8, 19)
    assert sample["context_images"][:, 0, 0, 0, 0].tolist() == [10, 13, 15, 17]
    assert sample["context_frame_offsets"].tolist() == [-7, -4, -2, 0]
    assert history_dataset.datasets[0].reads == list(range(10, 18))
    assert len(buffer.frames) == 5


def test_random_access_context_caches_first_frame_without_reading_full_past_samples(history_dataset):
    sample = read(history_dataset, 6)
    assert sample["context_images"][:, 0, 0, 0, 0].tolist() == [0, 2, 4, 6]
    assert sample["context_image_mask"][:, 0].all()
    assert not sample["context_image_mask"][:, 1].any()
    read(history_dataset, 7)
    raw = history_dataset.datasets[0]
    assert raw.reads == [6, 7]
    if raw.meta.video_keys:
        assert raw.hf_dataset.selected_columns == ["episode_index", "timestamp"]
        np.testing.assert_allclose(raw.queries[0]["camera"], [0, 2 / 12.5, 4 / 12.5])
        np.testing.assert_allclose(raw.queries[1]["camera"], [3 / 12.5, 5 / 12.5])
    else:
        assert raw.hf_dataset.selected_columns == ["episode_index", "timestamp", "camera"]


def test_fallback_context_uses_actual_returned_episode(history_dataset):
    history_dataset.datasets[0].fail_index = 3
    history_dataset._random_fallback_index = lambda *args, **kwargs: 12
    sample = read(history_dataset, 3)
    assert sample["requested_frame_index"] == 3
    assert sample["data_read_fallback"] == 1
    assert sample["context_images"][:, 0, 0, 0, 0].tolist() == [10, 10, 10, 12]
    assert sample["context_image_mask"][:, 0].tolist() == [True, False, True, True]


def test_history_column_projection_works_with_real_arrow_rows(history_dataset):
    from datasets import Dataset

    raw = history_dataset.datasets[0]
    raw.hf_dataset = Dataset.from_list(raw.hf_dataset).with_format("torch")
    sample = read(history_dataset, 16)
    assert sample["context_images"][:, 0, 0, 0, 0].tolist() == [10, 12, 14, 16]


def test_temporal_augmentation_covers_context_and_future_but_not_first_frame_cache(history_dataset):
    calls = []

    def augment(images):
        calls.append(images.clone())
        return images + 5

    history_dataset.rgb_augmentation = augment
    sample = read(history_dataset, 6, augment_rgb=True)
    assert sample["context_images"][:, 0, 0, 0, 0].tolist() == [5, 7, 9, 11]
    torch.testing.assert_close(sample["image_t0"], sample["context_images"][-1, 0])
    torch.testing.assert_close(sample["images_t0"], sample["context_images"][-1])
    torch.testing.assert_close(sample["video"], sample["video_views"][0])
    assert sample["video"][-1, 0, 0, 0] == 14
    assert len(calls) == 2
    assert read(history_dataset, 7)["context_images"][0, 0, 0, 0, 0] == 0


def test_single_camera_history_and_augmentation_do_not_require_multiview_fields(history_dataset):
    history_dataset.rgb_views = ("primary",)
    history_dataset.cfg.policy.rgb_views = ("primary",)
    original = history_dataset._to_canonical

    def single_camera(*args):
        sample = original(*args)
        for key in ("images_t0", "image_view_mask", "video_views", "video_view_mask"):
            sample.pop(key, None)
        return sample

    history_dataset._to_canonical = single_camera
    history_dataset.rgb_augmentation = lambda images: images + 1
    sample = read(history_dataset, 6, augment_rgb=True)
    assert sample["context_images"].shape == (4, 1, 3, 4, 4)
    assert sample["context_images"][:, 0, 0, 0, 0].tolist() == [1, 3, 5, 7]
    assert sample["context_image_mask"].all()
    torch.testing.assert_close(sample["image_t0"], sample["video"][0])


@pytest.mark.parametrize("offsets", [(0,), (1,), (-2, -4), (-2, -2), (-2.0,), (True,)])
def test_history_rejects_noncausal_or_ambiguous_offsets(offsets):
    with pytest.raises(ValueError, match="visual_history_offsets"):
        Qwen3VLMoTConfig(visual_history_offsets=offsets)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("field", ["visual_history_offsets", "visual_history_include_first"])
def test_checkpoint_history_contract_and_legacy_resume(tmp_path, enabled, field):
    saved = (
        history_config()
        if enabled
        else history_config(visual_history_offsets=(), visual_history_include_first=False)
    )
    saved._save_pretrained(tmp_path)
    if not enabled:
        path = tmp_path / "config.json"
        payload = json.loads(path.read_text())
        del payload["visual_history_offsets"], payload["visual_history_include_first"]
        path.write_text(json.dumps(payload))
    cfg = SimpleNamespace(
        weight_resume=True,
        output_dir=tmp_path,
        policy=Qwen3VLMoTConfig(),
        use_policy_training_preset=False,
    )
    _load_stage2_resume_policy_config(cfg, args=[])
    assert cfg.policy.visual_history_enabled == enabled
    changed = (() if enabled else (-4, -2)) if field == "visual_history_offsets" else not enabled
    fields = {
        "visual_history_offsets": saved.visual_history_offsets,
        "visual_history_include_first": saved.visual_history_include_first,
        field: changed,
    }
    cfg.policy = history_config(**fields)
    with pytest.raises(ValueError, match=field):
        _load_stage2_resume_policy_config(cfg, args=[f"--policy.{field}={json.dumps(changed)}"])
    with pytest.raises(ValueError, match=field):
        Qwen3VLMoTPolicy.from_pretrained(tmp_path, config=cfg.policy)


def test_buffer_rejects_missing_or_stale_current_observation():
    buffer = VisualHistoryBuffer((-4, -2), True)
    batch = {"image_t0": pixel(1)[None]}
    with pytest.raises(ValueError, match="every observation"):
        buffer.attach(batch)
    buffer.update(batch)
    with pytest.raises(ValueError, match="current observation"):
        buffer.attach({"image_t0": pixel(2)[None]})


def test_training_transfer_preserves_history_images_indices_and_exact_source_fps():
    batch = {
        "context_images": pixel(123)[None, None, None],
        "context_frame_offsets": torch.tensor([[-6, -4, -2, 0]]),
        "context_fps": torch.tensor([29.97]),
        "action": torch.ones(1, 4, 40),
    }
    moved = move_batch(batch, "cpu", torch.bfloat16)
    for key in ("context_images", "context_frame_offsets", "context_fps"):
        torch.testing.assert_close(moved[key], batch[key])
    assert moved["action"].dtype == torch.bfloat16


@pytest.mark.parametrize("plus", [False, True])
def test_libero_rollouts_observe_every_control_frame_after_settling(plus):
    from lerobot.evaluation import eval_qwen3vl_mot_libero as original
    from lerobot.evaluation import eval_qwen3vl_mot_libero_plus as extended

    settling = 10 if plus else 2
    buffer = VisualHistoryBuffer((-4, -2), True)
    planned, observed = [], []

    class CountingEnv:
        count = 0

        def reset(self):
            self.count = 0

        def set_init_state(self, state):
            return self.count

        def check_success(self):
            return False

        def step(self, action):
            self.count += 1
            return self.count, 0, False, {}

    def observe(frame):
        observed.append(frame - settling)
        buffer.update({"image_t0": pixel(frame - settling)[None]})

    def plan(frame, **kwargs):
        planned.append(buffer.latest["context_images"][0, :, 0, 0, 0, 0].tolist())
        return np.zeros((3, 7))

    runner = extended.rollout_episode if plus else original.rollout_episode
    kwargs = {} if plus else {"wait_steps": settling}
    runner(
        CountingEnv(), None, plan, max_steps=7, replan_steps=3, save_video=False, observe=observe, **kwargs
    )
    assert observed == list(range(7))
    assert len(planned) == 3
    assert planned[-1] == [0, 2, 4, 6]


@pytest.mark.parametrize("physical_mode", [None, "state", "action", "state_action"])
def test_history_launcher_isolated_recipe_and_real_parser(tmp_path, physical_mode):
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "WEIGHT_RESUME": "false",
        "DRY_RUN": "true",
        "CHECK_PATHS": "false",
        "MODEL_AUTO_DOWNLOAD": "false",
        "FP8_ENABLED": "false",
        "LOG_DIR": str(tmp_path),
        "OUTPUT_ROOT": str(tmp_path),
        "DATA_MIX": "robodojo_fps12",
    }
    arguments = (
        ["--policy.physical_history_steps=5", f"--policy.physical_history_mode={physical_mode}"]
        if physical_mode
        else []
    )
    result = subprocess.run(
        ["bash", str(root / "train_qwen3vl_mot_baseline_history_fsdp.sh"), *arguments],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--policy.physical_encoder_type=mlp" in result.stdout
    assert "--policy.generation_backbone=nano" in result.stdout
    assert "--policy.visual_history_offsets=" in result.stdout
    assert "qwen3vl_mot_baseline_history" in result.stdout
    import draccus

    command = next(
        line for line in result.stdout.splitlines() if "lerobot/scripts/fsdp_train_contrast.py" in line
    )
    policy_args = [
        argument.removeprefix("--policy.")
        for argument in shlex.split(command)
        if argument.startswith("--policy.") and not argument.startswith("--policy.type=")
    ]
    parsed = draccus.parse(Qwen3VLMoTConfig, args=[f"--{argument}" for argument in policy_args])
    assert parsed.visual_history_offsets == (-4, -2)
    assert parsed.visual_history_include_first
    assert parsed.physical_encoder_type == "mlp"
    assert parsed.task_weights == (0.15, 0.15, 0.7)
    assert parsed.physical_history_steps == (5 if physical_mode else 0)
    assert parsed.physical_history_modalities == (tuple(physical_mode.split("_")) if physical_mode else ())


@pytest.mark.parametrize("mode", ["state", "action", "state_action"])
def test_physical_training_history_matches_executed_step_buffer_without_future_or_episode_leaks(mode):
    from datasets import Dataset

    dataset = MultiModalContrastiveDataset.__new__(MultiModalContrastiveDataset)
    cfg = history_config(physical_history_steps=3, physical_history_mode=mode)
    dataset.cfg = SimpleNamespace(policy=cfg)
    table = Dataset.from_dict(
        {
            "episode_index": [0] * 5 + [1] * 6,
            "state": [[index, index + 0.5] for index in range(11)],
            "action": [[index * 2, index * 3] for index in range(11)],
            "irrelevant_video": ["never decoded"] * 11,
        }
    ).with_format("torch")
    dataset.datasets = [SimpleNamespace(hf_dataset=table)]
    dataset.all_episode_ranges = [np.array([[0, 5], [5, 11]])]
    dataset.specs = [{"state": [("state", 0, 2, 0)], "action": [("action", 0, 2, 0)]}]
    norm = {
        "mean": torch.ones(40),
        "std": torch.full((40,), 2.0),
        "mask": torch.tensor([1.0, 1.0] + [0.0] * 38),
    }
    dataset.norm_stats = [{"state": norm, "action": norm}]
    buffer = PhysicalHistoryBuffer(3, cfg.physical_history_modalities)
    for frame in range(5, 10):
        state, state_mask = dataset._build_canonical_vector(
            table[frame], dataset.specs[0]["state"], norm, False
        )
        current = {
            "observation.state": state[None],
            "state_mask": state_mask[None],
            "action_mask": state_mask[None],
        }
        action = None
        if frame > 5 and "action" in cfg.physical_history_modalities:
            action = dataset._build_canonical_vector(
                table[frame - 1], dataset.specs[0]["action"], norm, False
            )[0][None]
        online = buffer.update(current, executed_action=action)
        offline = dataset._physical_history(0, frame)
        for key, value in offline.items():
            torch.testing.assert_close(value, online[key][0])
        if frame == 5:
            assert all(not value.any() for value in offline.values())
        if frame == 6:
            for name in cfg.physical_history_modalities:
                assert offline[f"history_{name}_mask"][:, 0].tolist() == [False, False, True]
    for name in cfg.physical_history_modalities:
        source = [6, 7, 8] if name == "state" else [12, 14, 16]
        torch.testing.assert_close(offline[f"history_{name}"][:, 0], (torch.tensor(source) - 1) / 2)
    dataset.specs[0][cfg.physical_history_modalities[0]] = [("missing_source", 0, 2, 0)]
    missing = dataset._physical_history(0, 9)
    assert not missing[f"history_{cfg.physical_history_modalities[0]}_mask"].any()


def test_physical_buffer_rejects_unacknowledged_actions_and_stale_observations():
    buffer = PhysicalHistoryBuffer(3, ("state", "action"))
    batch = {
        "observation.state": torch.zeros(1, 4),
        "state_mask": torch.ones(1, 4),
        "action_mask": torch.ones(1, 4),
    }
    with pytest.raises(ValueError, match="every control observation"):
        buffer.attach(batch)
    buffer.update(batch)
    changed = {**batch, "observation.state": torch.ones(1, 4)}
    with pytest.raises(ValueError, match="actually executed"):
        buffer.update(changed)
    with pytest.raises(ValueError, match="every control observation"):
        buffer.attach(changed)
    buffer.update(changed, executed_action=torch.full((1, 4), 0.5))
    result = buffer.attach(changed)
    assert result["history_action"][0, :, 0].tolist() == [0, 0, 0.5]
    assert not result["history_state"].any()
    assert result["history_state_mask"][0, :, 0].tolist() == [False, False, True]


@pytest.mark.parametrize(
    "field,changed", [("physical_history_steps", 4), ("physical_history_mode", "action")]
)
@pytest.mark.parametrize("enabled", [False, True])
def test_physical_history_checkpoint_contract_and_legacy_defaults(tmp_path, field, changed, enabled):
    saved = history_config(physical_history_steps=3 if enabled else 0)
    saved._save_pretrained(tmp_path)
    if not enabled:
        path = tmp_path / "config.json"
        payload = json.loads(path.read_text())
        del payload["physical_history_steps"], payload["physical_history_mode"]
        path.write_text(json.dumps(payload))
    cfg = SimpleNamespace(
        weight_resume=True,
        output_dir=tmp_path,
        policy=history_config(),
        use_policy_training_preset=False,
    )
    _load_stage2_resume_policy_config(cfg, args=[])
    assert cfg.policy.physical_history_steps == (3 if enabled else 0)
    cfg.policy = (
        history_config(physical_history_steps=saved.physical_history_steps, **{field: changed})
        if field != "physical_history_steps"
        else history_config(physical_history_steps=changed)
    )
    with pytest.raises(ValueError, match=field):
        _load_stage2_resume_policy_config(cfg, args=[f"--policy.{field}={json.dumps(changed)}"])
    with pytest.raises(ValueError, match=field):
        Qwen3VLMoTPolicy.from_pretrained(tmp_path, config=cfg.policy)


@pytest.mark.parametrize(
    "options",
    [
        {"physical_history_steps": -1},
        {"physical_history_steps": True},
        {"physical_history_mode": "video"},
    ],
)
def test_physical_history_invalid_options(options):
    with pytest.raises(ValueError, match="physical_history"):
        history_config(**options)


@pytest.mark.parametrize("plus", [False, True])
def test_libero_physical_history_records_only_executed_prefix_after_settling(plus):
    from lerobot.evaluation import eval_qwen3vl_mot_libero as original
    from lerobot.evaluation import eval_qwen3vl_mot_libero_plus as extended

    class Adapter:
        def observation_batch(self, observation, instruction, config, device):
            return {
                "observation.state": torch.full((1, 2), float(observation)),
                "state_mask": torch.ones(1, 2),
                "action_mask": torch.ones(1, 2),
            }

        def normalize_executed_action(self, action):
            return torch.tensor(action[:2])[None]

    class Env:
        count = 0

        def reset(self):
            self.count = 0

        def set_init_state(self, initial):
            return self.count

        def check_success(self):
            return False

        def step(self, action):
            self.count += 1
            return self.count, 0, False, {}

    history = original.LiberoHistory(
        Adapter(),
        history_config(
            physical_history_steps=3,
            visual_history_offsets=(),
            visual_history_include_first=False,
        ),
        "move",
    )
    planned = []

    def plan(observation, **kwargs):
        batch = history.adapter.observation_batch(observation, "", None, None)
        planned.append(history.attach(batch))
        return np.array([[0.2] * 7, [0.3] * 7, [0.4] * 7, [99.0] * 7])

    runner = extended.rollout_episode if plus else original.rollout_episode
    runner(
        Env(),
        None,
        plan,
        max_steps=7,
        replan_steps=3,
        save_video=False,
        observe=history.observe,
        observe_action=history.observe_action,
        **({} if plus else {"wait_steps": 2}),
    )
    assert len(planned) == 3
    assert not planned[0]["history_action_mask"].any()
    torch.testing.assert_close(planned[1]["history_action"][0, :, 0], torch.tensor([0.2, 0.3, 0.4]))
    settling = 10 if plus else 2
    assert planned[1]["history_state"][0, :, 0].tolist() == [settling, settling + 1, settling + 2]
    assert not any((batch["history_action"] == 99).any() for batch in planned)
