"""Fine-tune the local PI05 policy on RoboDojo v3 delta end-effector data."""

import argparse
import hashlib
import json
import logging
import os
import subprocess
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from lerobot.common.datasets.utils import cycle
from lerobot.common.datasets_v30.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.pi05.configuration_pi05 import PI05Config
from lerobot.common.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.common.utils.random_utils import get_rng_state, set_rng_state, set_seed
from lerobot.configs.types import FeatureType, PolicyFeature

STATE_KEY = "observation.ee_ort6d_pos"
ACTION_KEY = "delta_action.ee_ort6d_pos"
CAMERAS = {
    "primary": "observation.images.cam_high",
    "secondary": "observation.images.cam_left_wrist",
    "wrist": "observation.images.cam_right_wrist",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def wandb_logging(args, config, *, rank=0):
    if not args.wandb or rank != 0:
        yield None
        return
    identity_path = args.output_dir / "wandb_run.json"
    saved = {}
    if getattr(args, "resume", False) and identity_path.is_file():
        saved = json.loads(identity_path.read_text())
    project = args.wandb_project or saved.get("project") or "lerobot-pi05"
    entity = args.wandb_entity or saved.get("entity") or None
    if saved and (project != saved["project"] or entity != saved.get("entity")):
        raise ValueError("W&B resume requires the original project and entity.")
    rng = get_rng_state()
    try:
        import wandb

        run = wandb.init(
            id=saved.get("id"),
            project=project,
            entity=entity,
            name=args.wandb_run_name or args.output_dir.name,
            mode=args.wandb_mode,
            dir=str(args.output_dir),
            config=config,
            allow_val_change=True,
            resume="allow" if saved and args.wandb_mode == "online" else None,
            save_code=False,
            settings=wandb.Settings(disable_git=True, disable_code=True),
        )
    finally:
        set_rng_state(rng)
    try:
        # A custom axis preserves updates when the checkpoint trails uploaded history.
        run.define_metric("train/*", step_metric="train/step")
        identity = {"id": run.id, "project": run.project, "entity": run.entity or None}
        temporary = identity_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(identity, indent=2) + "\n")
        temporary.replace(identity_path)
        location = run.url if args.wandb_mode == "online" else args.output_dir / "wandb"
        logging.info("W&B %s run %s: %s", args.wandb_mode, run.id, location)
        yield run
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


class RoboDojoDeltaDataset(Dataset):
    def __init__(self, root, chunk_size=32, video_backend="pyav"):
        root = Path(root)
        info = json.loads((root / "meta/info.json").read_text())
        stats = json.loads((root / "meta/stats.json").read_text())
        if info["codebase_version"] != "v3.0" or chunk_size <= 0:
            raise ValueError("A RoboDojo v3 dataset and positive chunk_size are required.")
        for key in (STATE_KEY, ACTION_KEY):
            if info["features"][key]["shape"] != [20]:
                raise ValueError(f"{key} must have exactly 20 dimensions.")
        self.input_features = {
            f"observation.images.{role}": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)) for role in CAMERAS
        }
        self.input_features["observation.state"] = PolicyFeature(FeatureType.STATE, (20,))
        self.output_features = {"action": PolicyFeature(FeatureType.ACTION, (20,))}
        self.stats = {
            target: {
                name: torch.as_tensor(value, dtype=torch.float32) for name, value in stats[source].items()
            }
            for target, source in (("observation.state", STATE_KEY), ("action", ACTION_KEY))
        }
        for target, values in self.stats.items():
            for name in ("q01", "q99"):
                if values[name].shape != (20,) or not torch.isfinite(values[name]).all():
                    raise ValueError(f"{target}.{name} requires 20 finite statistics.")
            if torch.any(values["q99"] < values["q01"]):
                raise ValueError(f"{target} has reversed quantiles.")
        for key in CAMERAS.values():
            if info["features"][key]["dtype"] != "video":
                raise ValueError(f"Expected a RoboDojo RGB video feature: {key}")
        self.source = LeRobotDataset(
            repo_id="local/robodojo",
            root=root,
            delta_timestamps={ACTION_KEY: [index / info["fps"] for index in range(chunk_size)]},
            video_backend=video_backend,
            download_videos=False,
        )
        self.source.video_keys_to_decode = list(CAMERAS.values())
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = self.source[index]
        batch = {
            "observation.state": sample[STATE_KEY].float(),
            "action": sample[ACTION_KEY].float(),
            "action_is_pad": sample[f"{ACTION_KEY}_is_pad"],
            "task": sample["task"],
        }
        if batch["observation.state"].shape != (20,) or batch["action"].shape != (self.chunk_size, 20):
            raise ValueError("Unexpected RoboDojo state/action sample shape.")
        for key in ("observation.state", "action"):
            if not torch.isfinite(batch[key]).all():
                raise ValueError(f"Nonfinite {key} at dataset index {index}.")
        for role, key in CAMERAS.items():
            image = sample[key]
            if image.ndim != 3 or image.shape[0] != 3 or image.dtype != torch.float32:
                raise ValueError(f"{key} must decode to float32 RGB (3,H,W).")
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError(f"{key} must contain finite values in [0,1].")
            batch[f"observation.images.{role}"] = image
        if not isinstance(batch["task"], str) or not batch["task"].strip():
            raise ValueError("Every training sample requires a nonempty task instruction.")
        return batch


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pretrained-path", type=Path)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--validate-data-only", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Record training metrics with Weights & Biases.")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-name", default=os.environ.get("WANDB_RUN_NAME"))
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline"), default=os.environ.get("WANDB_MODE", "online")
    )
    return parser


def parse_args(parser=None):
    parser = build_parser() if parser is None else parser
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.chunk_size, args.save_freq) <= 0:
        parser.error("steps, batch-size, chunk-size, and save-freq must be positive.")
    if args.num_workers < 0 or args.warmup_steps < 0 or args.lr <= 0:
        parser.error("num-workers/warmup-steps must be nonnegative and lr must be positive.")
    if args.wandb and args.wandb_mode not in ("online", "offline"):
        parser.error("--wandb-mode must be online or offline.")
    if not args.validate_data_only and (args.pretrained_path is None or args.tokenizer_path is None):
        parser.error(
            "Training requires --pretrained-path and --tokenizer-path; random initialization is disabled."
        )
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset = RoboDojoDeltaDataset(args.dataset_root, args.chunk_size, args.video_backend)
    # Decode only inside workers when workers are requested; do not initialize a decoder before forking.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    iterator = cycle(loader)
    first_batch = next(iterator)
    contract = {
        "state_feature": STATE_KEY,
        "action_feature": ACTION_KEY,
        "camera_mapping": CAMERAS,
        "action_semantics": "Adjacent measured-state increments; world dp, left-multiplied dR, incremental grip.",
        "normalization": "QUANTILES",
        "frames": len(dataset),
        "chunk_size": args.chunk_size,
        "dataset_stats_sha256": file_hash(args.dataset_root / "meta/stats.json"),
        "sample_shapes": {
            key: list(value.shape) for key, value in first_batch.items() if isinstance(value, torch.Tensor)
        },
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "data_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    if args.validate_data_only:
        logging.info("Data validation completed: %s", contract["sample_shapes"])
        return
    device = torch.device(args.device)
    config = PI05Config(
        tokenizer_path=args.tokenizer_path,
        device="cpu",
        dtype="bfloat16" if device.type == "cuda" else "float32",
        chunk_size=args.chunk_size,
        n_action_steps=min(10, args.chunk_size),
        input_features=dataset.input_features,
        output_features=dataset.output_features,
        gradient_checkpointing=True,
        train_expert_only=False,
        optimizer_lr=args.lr,
        scheduler_warmup_steps=min(args.warmup_steps, max(0, args.steps - 1)),
        scheduler_decay_steps=args.steps,
    )
    policy = PI05Policy(config, dataset_stats=dataset.stats)
    base_file = (
        args.pretrained_path / "model.safetensors" if args.pretrained_path.is_dir() else args.pretrained_path
    )
    contract["base_sha256"] = file_hash(base_file)
    contract["tokenizer_sha256"] = (
        file_hash(args.tokenizer_path) if Path(args.tokenizer_path).is_file() else None
    )
    policy.load_base_weights(base_file)
    policy.to(device).train()
    policy.config.device = str(device)
    (args.output_dir / "data_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    optimizer = config.get_optimizer_preset().build(policy.parameters())
    scheduler_config = config.get_scheduler_preset()
    scheduler_config.num_platform_steps = 0
    scheduler_config.decay_from_platform_end = True
    scheduler_config.num_decay_steps = max(1, args.steps - config.scheduler_warmup_steps)
    scheduler = scheduler_config.build(optimizer, args.steps)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with wandb_logging(args, contract) as run, (args.output_dir / "training.jsonl").open("x") as log:
        for step in range(1, args.steps + 1):
            batch = first_batch if step == 1 else next(iterator)
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with context:
                loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                config.optimizer_grad_clip_norm,
                error_if_nonfinite=True,
            )
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            row = {"step": step, "loss": loss.item(), "grad_norm": grad_norm.item(), "lr": lr}
            log.write(json.dumps(row) + "\n")
            log.flush()
            logging.info("%s", row)
            if run is not None:
                run.log({f"train/{key}": value for key, value in row.items()})
            if step % args.save_freq == 0 or step == args.steps:
                checkpoint = args.output_dir / f"step{step}"
                policy.save_pretrained(checkpoint)
                torch.save(
                    {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
                    checkpoint / "training_state.pt",
                )
        summary = {"completed": True, "steps": args.steps, "final_loss": row["loss"]}
        if device.type == "cuda":
            summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if run is not None:
            run.summary.update(summary)


if __name__ == "__main__":
    main()
