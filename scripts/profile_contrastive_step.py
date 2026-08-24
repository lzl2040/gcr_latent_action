"""Where does a contrastive training step spend its time?

``updt_s`` and ``data_s`` in the training log only say *whether* the step or the loader is
slow. When a mixture change makes training slower, the useful question is which module got
more work, and that needs a breakdown inside the forward pass.

This script runs the real dataset, sampler and model, and wraps the interesting submodules
in CUDA-synchronised timers. It also reports the *shapes* those modules actually saw --
notably how many tactile pads survived masking -- because the cost of the tactile branch is
driven by a count that varies per batch, not by anything in the config.

DeepSpeed is deliberately not used: it would add ZeRO bookkeeping to every measurement
without changing which module dominates, and it makes the script much harder to run. The
absolute numbers are therefore a little optimistic; the *ratios* are what to read.

Example, comparing a tactile-heavy mixture against the default one::

    python scripts/profile_contrastive_step.py \
        --datasets ftp_1_sharpa:3.0 ftp_1_VisuoTactile_D-WHEEL:3.0 taco_play:1.0 fractal20220817_data:1.0 \
        --batch_size 128 --steps 6

    python scripts/profile_contrastive_step.py --mix debug_research_data --batch_size 128 --steps 6
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from lerobot.common.datasets.contrastive_dataset import (  # noqa: E402
    MultiModalContrastiveDataset,
    contrastive_collate_fn,
)
from lerobot.common.datasets.contrastive_sampler import ContrastiveBatchSampler  # noqa: E402
from lerobot.common.datasets.mixtures import OXE_NAMED_MIXTURES  # noqa: E402
from lerobot.common.policies.factory import make_policy  # noqa: E402
from lerobot.common.policies.ace.configuration_robo_contrast import RoboContrastConfig  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402

# Kept as a module constant because the loader-only measurement has to know how deep the
# prefetch buffer is in order to skip past it; see the comment there.
PREFETCH_FACTOR = 4


@dataclass
class _Cfg:
    """The subset of ``TrainPipelineConfig`` the dataset and the policy actually read."""

    policy: RoboContrastConfig = None
    dataset: DatasetConfig = None
    task_type: str = "contrast"
    seed: int = 1000
    device: str = "cuda"


class Timer:
    """CUDA-synchronised wall clock, accumulated per label.

    Synchronising inside nested hooks serialises the queue and so slightly inflates the
    total, but without it every measurement would just record the time to enqueue a kernel
    and the last section would absorb everyone else's work.
    """

    def __init__(self):
        self.total = defaultdict(float)
        self.count = defaultdict(int)

    @contextlib.contextmanager
    def __call__(self, label: str):
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.total[label] += time.perf_counter() - start
            self.count[label] += 1

    def hook(self, module: torch.nn.Module, label: str) -> None:
        state = {}

        def pre(_m, _inp):
            torch.cuda.synchronize()
            state["t"] = time.perf_counter()

        def post(_m, _inp, _out):
            torch.cuda.synchronize()
            self.total[label] += time.perf_counter() - state["t"]
            self.count[label] += 1

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    def reset(self):
        self.total.clear()
        self.count.clear()


def build_config(args) -> _Cfg:
    policy = RoboContrastConfig(
        vision_model_name=args.vision_model,
        text_model_name=args.text_model,
        chunk_size=args.chunk_size,
        group_size=args.group_size,
        chunk_seconds=args.chunk_seconds,
        tactile_backbone=args.tactile_backbone,
        ftp1_tactile_dir=args.ftp1_tactile_dir,
        tactile_frames=args.tactile_frames,
        tactile_tokens_per_pad=args.tactile_tokens_per_pad,
    )
    if args.max_tactile_views is not None:
        policy.max_tactile_views = args.max_tactile_views
    dataset = DatasetConfig(repo_id="profile")
    # ``--loader_only`` has to run past the prefetch buffer before it measures anything, so the
    # epoch must be long enough to supply those extra batches as well as the timed ones.
    steps = args.steps
    if args.loader_only:
        steps = max(steps, max(args.warmup, args.num_workers * PREFETCH_FACTOR + 2) + 20)
    dataset.dataset_size_one_epoch = args.batch_size * (steps + 2)
    dataset.parent_dir_v21 = args.parent_dir_v21
    dataset.parent_dir_v30 = args.parent_dir_v30
    if args.parent_dir_extra and hasattr(dataset, "parent_dir_extra"):
        dataset.parent_dir_extra = args.parent_dir_extra
    dataset.processor_path = args.processor_path
    if args.video_backend:
        dataset.video_backend = args.video_backend
    return _Cfg(policy=policy, dataset=dataset)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=None, help="name:weight pairs; overrides --mix")
    p.add_argument("--mix", default="debug_research_data")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2, help="steps excluded from the averages")
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--video_backend", default=None, help="override DatasetConfig.video_backend")
    p.add_argument("--max_tactile_views", type=int, default=None,
                   help="pads per sample; each one is an extra video stream to decode")
    p.add_argument("--loader_only", action="store_true",
                   help="skip the model; measure what the loader alone can deliver")
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--chunk_seconds", type=float, default=1.6)
    p.add_argument("--tactile_backbone", default="resnet18", choices=["resnet18", "ftp1"])
    p.add_argument("--tactile_frames", type=int, default=4,
                   help="frames read per pad; the loader decodes them and the CNN runs once per frame")
    p.add_argument("--tactile_tokens_per_pad", type=int, default=2,
                   help="tokens each pad contributes after the temporal fusion")
    p.add_argument("--ftp1_tactile_dir", default="/Data/lzl/huggingface/ftp1_v0426_50kstep")
    p.add_argument("--vision_model", default="/Data/lzl/huggingface/dinov3-vitb16-pretrain-lvd1689m")
    p.add_argument("--text_model", default="/Data/lzl/huggingface/siglip2-base-patch16-224")
    p.add_argument("--processor_path", default="/Data/lzl/huggingface/InternVL3_5-2B-HF")
    p.add_argument("--parent_dir_v21", default="/Data/lerobot_data_ort6d")
    p.add_argument("--parent_dir_v30", default="/Data/lerobot_data_ort6d/v30")
    p.add_argument("--parent_dir_extra", default="/media/v-wangxiaofa/新加卷/lerobot_data")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    os.environ.setdefault("DECORD_LOG_LEVEL", "error")

    if args.datasets:
        entries = []
        for item in args.datasets:
            name, _, weight = item.partition(":")
            entries.append((name, float(weight) if weight else 1.0))
        args.mix = "__profile_mixture__"
        OXE_NAMED_MIXTURES[args.mix] = entries
        print(f"mixture: {entries}")
    else:
        print(f"mixture: {args.mix}")

    cfg = build_config(args)
    dataset = MultiModalContrastiveDataset(
        cfg=cfg, data_mix=args.mix, seed=cfg.seed,
        dataset_size_one_epoch=cfg.dataset.dataset_size_one_epoch,
    )
    sampler = ContrastiveBatchSampler(
        episode_ranges=dataset.episode_ranges,
        sample_weights=dataset.sample_weights,
        batch_size=args.batch_size,
        num_replicas=1,
        rank=0,
        seed=cfg.seed,
        samples_per_epoch=cfg.dataset.dataset_size_one_epoch,
        horizon=dataset.frame_horizons,
        same_dataset_frac=cfg.policy.same_dataset_frac,
        episode_group_frac=cfg.policy.episode_group_frac,
        episode_group_size=cfg.policy.episode_group_size,
        min_frame_gap=cfg.policy.min_frame_gap,
    )
    loader = DataLoader(
        dataset=dataset, batch_sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, collate_fn=contrastive_collate_fn,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=PREFETCH_FACTOR if args.num_workers > 0 else None,
    )

    if args.loader_only:
        # The question "can the loader keep up?" is separable from "is the model slow?", and
        # on a network filesystem it is usually the only one that matters. Answer it without
        # paying for a 772M-parameter model or a GPU.
        #
        # The trap here is the prefetch buffer. The workers start filling it at ``iter()`` and
        # hold ``num_workers * prefetch_factor`` batches -- 48 at the defaults. A run of 10
        # steps therefore never touches a decoder at all: it drains a queue that was filled
        # while the process was still starting up, and reports memory bandwidth. That is not a
        # small bias, it is the difference between 900 and 1_300_000 samples/s (both measured).
        # Steady state only begins once the buffer is empty, which takes as many steps as the
        # buffer is deep, so the timed window has to start after that.
        prefetch_depth = args.num_workers * PREFETCH_FACTOR if args.num_workers > 0 else 0
        warmup = max(args.warmup, prefetch_depth + 2)
        if args.steps <= warmup:
            print(f"[loader_only] --steps {args.steps} is inside the {prefetch_depth}-batch prefetch "
                  f"buffer and would measure the queue, not the decoder; raising to {warmup + 20}.")
        steps = max(args.steps, warmup + 20)
        step, seen = 0, 0
        start = time.perf_counter()
        for batch in loader:
            seen += batch["observation.state"].shape[0]
            step += 1
            if step == warmup:  # buffer drained; from here the loader is producing, not replaying
                start, seen = time.perf_counter(), 0
            if step >= steps:
                break
        if step < steps:
            print(f"[loader_only] epoch ended after {step} batches; increase "
                  f"--steps or the mixture size for a steady-state number.")
        elapsed = time.perf_counter() - start
        print(f"\nloader only: {seen / max(elapsed, 1e-9):.1f} 样本/秒 "
              f"({1000 * elapsed / max(seen, 1):.1f} ms/样本, {args.num_workers} workers, "
              f"{step - warmup} steps timed after a {prefetch_depth}-batch warmup)")
        return 0

    policy = make_policy(cfg=cfg.policy, device="cpu", ds_meta=dataset.meta)
    device = torch.device("cuda")
    # Matches the deepspeed config: bf16 weights, no autocast wrapper.
    policy = policy.to(device=device, dtype=torch.bfloat16)
    policy.train()
    optim = torch.optim.AdamW(policy.parameters(), lr=1e-5)

    timer = Timer()
    phys = policy.physical_encoder
    percep = policy.perception_encoder
    timer.hook(percep, "perception encoder (总)")
    timer.hook(percep.vision_backbone, "  · DINOv3 vision")
    timer.hook(percep.text_backbone, "  · SigLIP2 text")
    timer.hook(phys, "physical encoder (总)")
    timer.hook(phys.tactile_cnn, "  · 触觉图像 backbone")
    if phys.tactile_recon is not None:
        timer.hook(phys.tactile_recon, "  · 触觉重建 head")

    # How many pad rows the tactile backbone was handed, which is what actually varies.
    pads = {"selected": 0.0, "rows": 0.0, "batches": 0}
    orig_tac_forward = phys.tactile_cnn.forward

    def counting_forward(images, *a, **k):
        pads["selected"] += images.shape[0]
        pads["batches"] += 1
        return orig_tac_forward(images, *a, **k)

    phys.tactile_cnn.forward = counting_forward

    print(f"\nparams: {sum(p.numel() for p in policy.parameters()) / 1e6:.0f}M  "
          f"batch={args.batch_size}  tactile_backbone={args.tactile_backbone}\n")

    step = 0
    wall = defaultdict(float)
    batch_ready = time.perf_counter()
    for batch in loader:
        wall["dataloading"] += time.perf_counter() - batch_ready
        if step == args.warmup:  # discard warm-up: cudnn autotune, allocator growth
            timer.reset()
            wall.clear()
            pads["selected"] = 0.0
            pads["rows"] = 0.0
            pads["batches"] = 0
            wall["dataloading"] = 0.0

        gpu_batch = {}
        with timer("H2D 拷贝"):
            for k, v in batch.items():
                if not isinstance(v, torch.Tensor):
                    gpu_batch[k] = v
                    continue
                v = v.to(device, non_blocking=True)
                if v.is_floating_point() and k not in ("image_t0", "image_t1", "tactile_image"):
                    v = v.to(torch.bfloat16)
                gpu_batch[k] = v
        pads["rows"] += float(gpu_batch["tactile_image_mask"].sum().item())

        with timer("forward (总)"):
            loss, out = policy(gpu_batch, task_type=cfg.task_type, step=step)
        with timer("backward"):
            loss.backward()
        with timer("optimizer"):
            optim.step()
            optim.zero_grad(set_to_none=True)

        step += 1
        print(f"  step {step}: loss={loss.item():.3f} "
              f"tac_pads={int(gpu_batch['tactile_image_mask'].sum().item())}/"
              f"{gpu_batch['tactile_image_mask'].numel()}")
        if step >= args.steps:
            break
        batch_ready = time.perf_counter()

    n = max(1, step - args.warmup)
    print(f"\n{'=' * 62}\n每步平均（{n} 步，已去掉 {args.warmup} 步 warmup）\n{'=' * 62}")
    order = ["dataloading", "H2D 拷贝", "forward (总)", "perception encoder (总)",
             "  · DINOv3 vision", "  · SigLIP2 text", "physical encoder (总)",
             "  · 触觉图像 backbone", "  · 触觉重建 head", "backward", "optimizer"]
    total_step = wall["dataloading"] / n + sum(
        timer.total[k] for k in ("H2D 拷贝", "forward (总)", "backward", "optimizer")
    ) / n
    for label in order:
        secs = (wall[label] if label in wall else timer.total[label]) / n
        if secs == 0:
            continue
        print(f"{label:<28s} {secs:7.3f} s  ({100 * secs / max(total_step, 1e-9):5.1f}%)")
    print(f"{'-' * 62}\n{'每步合计':<26s} {total_step:7.3f} s")
    if pads["batches"]:
        print(f"\n触觉 pad：每步送进 backbone {pads['selected'] / n:.0f} 个 pad "
              f"（每 pad {args.tactile_frames} 帧 = {args.tactile_frames * pads['selected'] / n:.0f} 张图），"
              f"mask 存活 {pads['rows'] / n:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
