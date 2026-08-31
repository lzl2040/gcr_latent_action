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
    args = ap.parse_args()

    for backbone, target in (
        ("dinov3", "vision"),
        ("cosmos3", "vision"),
        ("cosmos3", "vae"),
        ("qwen3vl", "vision"),
    ):
        cfg = RoboContrastConfig()
        cfg.vision_backbone = backbone
        cfg.perception_recon_target = target
        cfg.__post_init__()
        model = RoboContrast(cfg).to(args.device)
        dump(model, f"vision_backbone={backbone}, perception_recon_target={target}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
