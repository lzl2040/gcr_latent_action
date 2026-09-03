"""Train / benchmark the Phi-4-Multimodal MoT world model on the real loader.

Does two jobs deliberately kept in one file, because the second is only trustworthy if it runs
the first: a real training step (dataset -> Wan VAE -> MoT -> AdamW), and a per-task timing
breakdown used to project cluster wall-clock.

The timing is split into data / vae / model rather than reported as one number, because the
three scale differently: the loader is CPU-bound and parallelises over workers, the VAE is a
fixed cost per clip that a real run would pay once and cache, and only the model part responds
to a faster GPU. Quoting a single step time would make the cluster extrapolation meaningless.

Run (same args as train_ace_local.sh, minus deepspeed)::

    BATCH=8 STEPS=30 python -u scripts/train_mot_world.py --policy.type=robo_contrast ...
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.common.datasets.canonical_space import CANON_DIM  # noqa: E402
from lerobot.common.datasets.contrastive_dataset import (  # noqa: E402
    MultiModalContrastiveDataset,
    contrastive_collate_fn,
)
from lerobot.common.policies.mot.modeling_mot import MoTConfig  # noqa: E402
from lerobot.common.policies.mot.vae_latents import WanLatentEncoder  # noqa: E402
from lerobot.common.policies.mot.world_model import (  # noqa: E402
    TASK_SPECS,
    MoTWorldModel,
    WorldModelConfig,
    parse_mix,
    sample_task,
)
from lerobot.configs import parser  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402


def env(name, default, cast=int):
    return cast(os.environ.get(name, default))


class Timer:
    def __init__(self, device):
        self.device = device
        self.acc = defaultdict(float)
        self.n = defaultdict(int)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def add(self, key, dt):
        self.acc[key] += dt
        self.n[key] += 1

    def mean(self, key):
        return self.acc[key] / max(1, self.n[key])


@parser.wrap()
def main(cfg: TrainPipelineConfig):
    batch_size = env("BATCH", 8)
    steps = env("STEPS", 30)
    warmup = env("WARMUP", 5)
    latent_frames = env("LATENT_FRAMES", 3)
    stage = os.environ.get("STAGE", "3")
    phi_dir = os.environ.get("PHI_DIR", "/Data/lzl/huggingface/Phi-4-multimodal-instruct")
    vae_dir = os.environ.get("VAE_DIR", "/Data/lzl/huggingface/Cosmos3-Edge/vae")
    per_task = os.environ.get("PER_TASK", "1") == "1"
    scope = os.environ.get("SCOPE", "gen_only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Wan maps frame 0 to its own latent and compresses the rest 4x, so the clip length is
    # forced by the number of latent frames we want -- any other length silently truncates.
    rgb_frames = WanLatentEncoder.required_frames(latent_frames)
    cfg.policy.rgb_frames = rgb_frames
    cfg.policy.use_tactile = False
    print(f"clip: {rgb_frames} pixel frames -> {latent_frames} latent frames; tactile off")

    dataset = MultiModalContrastiveDataset(
        cfg=cfg,
        data_mix=cfg.data_mix,
        seed=cfg.seed,
        dataset_size_one_epoch=cfg.dataset.dataset_size_one_epoch,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=contrastive_collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(phi_dir)
    vae = WanLatentEncoder(vae_dir, dtype=dtype).to(device)

    # The dataset's canonical action vector is 40 wide; Cosmos's default of 64 would leave the
    # per-domain projection expecting columns the loader never fills.
    mot = MoTConfig.from_phi_dir(phi_dir, action_dim=CANON_DIM)
    assert mot.latent_channels == vae.latent_channels, (
        f"MoT expects {mot.latent_channels} latent channels, VAE produces {vae.latent_channels}"
    )
    model = MoTWorldModel(
        WorldModelConfig(mot=mot, trainable_scope=scope)
    ).to(device=device, dtype=dtype)
    model.load_pretrained()
    model.mot.gradient_checkpointing = True
    model.train()

    rep = model.param_report()
    print(
        f"params: total {rep['total'] / 1e9:.3f}B  trainable {rep['trainable'] / 1e9:.3f}B  "
        f"scope={scope}  "
        f"(vae {sum(p.numel() for p in vae.vae.parameters()) / 1e6:.0f}M frozen, not counted)"
    )

    # A 5.54B-trainable scope does not fit on one 48 GiB card with gradients and Adam state.
    # NO_OPT=1 drops the optimizer so the forward/backward compute -- the part
    # that a faster GPU or more GPUs actually changes -- can still be measured and compared
    # across scopes; the optimizer state is ZeRO-sharded in any real run of this size anyway.
    no_opt = os.environ.get("NO_OPT", "0") == "1"
    opt = (
        None
        if no_opt
        else torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
            weight_decay=0.01,
            fused=device.type == "cuda",
        )
    )
    if no_opt:
        print("NO_OPT=1: measuring forward+backward only, no optimizer state allocated")

    # MIX takes a preset name ("stage2", "stage3", "action_only", "stage3_joint_only") or an
    # explicit "policy=0.4,i2v=0.6" string; it defaults to the preset the stage implies.
    mix = parse_mix(env("MIX", "stage3" if stage == "3" else "stage2", str))
    print("task mix: " + "  ".join(f"{k}={v:.2f}" for k, v in sorted(mix.items())))
    task_cycle = list(TASK_SPECS) if per_task else None
    timer = Timer(device)
    losses = defaultdict(list)
    gen = torch.Generator().manual_seed(cfg.seed)

    it = iter(loader)
    t_last = time.perf_counter()
    for step in range(steps + warmup):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        timer.sync()
        t_data = time.perf_counter() - t_last

        # Cycle tasks deterministically when profiling, sample from the mix when training, so
        # the per-task numbers are not at the mercy of the sampler.
        task = task_cycle[step % len(task_cycle)] if task_cycle else sample_task(mix, gen)
        spec = TASK_SPECS[task]

        t0 = time.perf_counter()
        clip = batch["image_clip"].to(device, non_blocking=True)
        latents = vae(clip).to(dtype)
        timer.sync()
        t_vae = time.perf_counter() - t0

        t0 = time.perf_counter()
        images = None
        if spec.image:
            images = batch["image_t0"].to(device, non_blocking=True).to(dtype).div(255.0)
        text = tokenizer(
            [t or " " for t in batch["task"]],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=env("TEXT_LEN", 32),
        )
        text_ids = text["input_ids"].to(device)
        actions = batch["action"].to(device).to(dtype)
        domain_id = batch["dataset_id"].to(device)

        out = model(
            latents=latents,
            pixel_values=images,
            text_ids=text_ids,
            actions=actions,
            domain_id=domain_id,
            task=task,
        )
        out["loss"].backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        if opt is not None:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        else:
            for p in trainable:
                p.grad = None
        timer.sync()
        t_model = time.perf_counter() - t0

        if step >= warmup:
            timer.add(f"data/{task}", t_data)
            timer.add(f"vae/{task}", t_vae)
            timer.add(f"model/{task}", t_model)
            timer.add("data/all", t_data)
            timer.add("vae/all", t_vae)
            timer.add("model/all", t_model)
            for term in ("loss_video", "loss_action"):
                if term in out:
                    losses[f"{task}/{term}"].append(out[term].item())
        if step % 5 == 0:
            print(
                f"step {step:3d} [{task:12s}] loss {out['loss'].item():7.3f}  "
                f"data {t_data * 1e3:6.0f} ms  vae {t_vae * 1e3:6.0f} ms  model {t_model * 1e3:6.0f} ms"
                + ("  (warmup)" if step < warmup else "")
            )
        t_last = time.perf_counter()

    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    print(f"\npeak memory {peak:.1f} GiB at batch {batch_size}\n")
    print(f"{'task':13s} {'data':>9s} {'vae':>9s} {'model':>9s} {'step':>9s} {'clip/s':>9s} {'L_video':>9s} {'L_act':>9s}")
    for task in TASK_SPECS:
        if timer.n[f"model/{task}"] == 0:
            continue
        d, v, m = (timer.mean(f"{k}/{task}") for k in ("data", "vae", "model"))
        step_s = d + v + m
        cells = []
        for term in ("loss_video", "loss_action"):
            vals = losses[f"{task}/{term}"]
            cells.append(f"{sum(vals) / len(vals):9.3f}" if vals else f"{'-':>9s}")
        print(
            f"{task:13s} {d * 1e3:8.0f}m {v * 1e3:8.0f}m {m * 1e3:8.0f}m {step_s * 1e3:8.0f}m "
            f"{batch_size / step_s:9.2f} " + " ".join(cells)
        )
    d, v, m = (timer.mean(f"{k}/all") for k in ("data", "vae", "model"))
    print(
        f"{'ALL':13s} {d * 1e3:8.0f}m {v * 1e3:8.0f}m {m * 1e3:8.0f}m {(d + v + m) * 1e3:8.0f}m "
        f"{batch_size / (d + v + m):9.2f}"
    )
    print(
        f"\nframes/s (pixel frames through the model): "
        f"{batch_size * rgb_frames / (d + v + m):.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
