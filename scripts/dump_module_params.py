"""Dump the per-module parameter breakdown used by ``doc/cosmos3_contra.md``.

Run it rather than editing the doc's tables by hand: the numbers move whenever a config
default changes, and a stale parameter table is worse than none.
"""

import argparse

import torch

from lerobot.common.policies.ace.configuration_robo_contrast import RoboContrastConfig
from lerobot.common.policies.ace.modeling_robo_contrast import RoboContrast


def counts(module) -> tuple[float, float]:
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total / 1e6, train / 1e6


def dump(model: RoboContrast, title: str):
    print(f"\n### {title}\n")
    print("| module | params | trainable |")
    print("|---|---:|---:|")
    for branch_name, branch in (
        ("perception_encoder", model.perception_encoder),
        ("physical_encoder", model.physical_encoder),
    ):
        b_tot, b_tr = counts(branch)
        print(f"| **{branch_name}** | **{b_tot:.1f}M** | **{b_tr:.1f}M** |")
        for name, child in branch.named_children():
            tot, tr = counts(child)
            if tot < 0.05:
                continue
            print(f"| &nbsp;&nbsp;`{name}` | {tot:.1f}M | {tr:.1f}M |")
    tot, tr = counts(model)
    print(f"| **total** | **{tot:.1f}M** | **{tr:.1f}M** |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--tactile_backbone",
        default="resnet18",
        choices=("resnet18", "ftp1", "anytouch"),
    )
    ap.add_argument(
        "--anytouch_checkpoint",
        default="/Data/lzl/huggingface/anytouch_encoder.pth",
    )
    ap.add_argument(
        "--vision_backbone",
        choices=("dinov3", "cosmos3", "qwen3vl"),
        help="dump only this vision backbone instead of the standard comparison matrix",
    )
    ap.add_argument(
        "--perception_recon_target",
        default="vision",
        choices=("vision", "vae"),
    )
    args = ap.parse_args()

    combinations = (
        [(args.vision_backbone, args.perception_recon_target)]
        if args.vision_backbone
        else [
            ("dinov3", "vision"),
            ("cosmos3", "vision"),
            ("cosmos3", "vae"),
            ("qwen3vl", "vision"),
        ]
    )
    for backbone, target in combinations:
        cfg = RoboContrastConfig(
            vision_backbone=backbone,
            perception_recon_target=target,
            tactile_backbone=args.tactile_backbone,
            anytouch_checkpoint=args.anytouch_checkpoint,
        )
        model = RoboContrast(cfg).to(args.device)
        dump(
            model,
            f"vision_backbone={backbone}, perception_recon_target={target}, "
            f"tactile_backbone={args.tactile_backbone}",
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
