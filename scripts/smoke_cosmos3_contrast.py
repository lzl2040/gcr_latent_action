"""Smoke-test the `cosmos3_contra` knobs: vision backbone, recon target, CLS token count.

Checks that every combination builds, runs a forward and a backward, and reports its
parameter split. The K=1 / dinov3 / vision row is the pre-existing configuration and must
still work unchanged -- that is the regression this script exists to catch.
"""

import argparse
import itertools

import torch

from lerobot.common.policies.ace.configuration_robo_contrast import RoboContrastConfig
from lerobot.common.policies.ace.modeling_robo_contrast import RoboContrast


def make_batch(cfg: RoboContrastConfig, batch_size: int, device: str) -> dict:
    b, c = batch_size, cfg.max_state_dim
    s, v = cfg.chunk_size, cfg.max_tactile_views
    return {
        "image_t0": torch.randint(0, 255, (b, 3, 224, 224), dtype=torch.uint8, device=device),
        "image_t1": torch.randint(0, 255, (b, 3, 224, 224), dtype=torch.uint8, device=device),
        "task": ["pick up the cube"] * b,
        "has_text": torch.ones(b, device=device),
        "observation.state": torch.randn(b, s, c, device=device),
        "state_mask": torch.ones(b, c, device=device),
        "action": torch.randn(b, s, c, device=device),
        "action_mask": torch.ones(b, c, device=device),
        "tactile_signal": torch.randn(b, s, cfg.max_tactile_signal_dim, device=device),
        "tactile_signal_mask": torch.ones(b, device=device),
        "tactile_image": torch.randint(
            0,
            255,
            (b, v, cfg.tactile_frames, 3, cfg.tactile_img_size, cfg.tactile_img_size),
            dtype=torch.uint8,
            device=device,
        ),
        "tactile_image_mask": torch.ones(b, v, device=device),
        "tactile_sensor_id": torch.zeros(b, v, dtype=torch.long, device=device),
        "tactile_img_mean": torch.zeros(b, v, 3, device=device),
        "tactile_img_std": torch.ones(b, v, 3, device=device),
        "sample_rate": torch.full((b,), 15, device=device),
        "pair_is_valid": torch.ones(b, device=device),
        "dataset_index": torch.zeros(b, dtype=torch.long, device=device),
        "episode_index": torch.arange(b, device=device),
        "episode_uid": torch.arange(b, device=device),
        "frame_index": torch.arange(b, device=device) * 1000,
    }


def param_report(model: RoboContrast) -> str:
    def count(module):
        total = sum(p.numel() for p in module.parameters())
        train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total / 1e6, train / 1e6

    p_tot, p_tr = count(model.perception_encoder)
    r_tot, r_tr = count(model.physical_encoder)
    a_tot, a_tr = count(model)
    return (
        f"perception {p_tot:6.1f}M ({p_tr:5.1f}M trainable) | "
        f"physical {r_tot:6.1f}M ({r_tr:5.1f}M trainable) | "
        f"total {a_tot:6.1f}M ({a_tr:5.1f}M trainable)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", nargs="*", default=["dinov3", "cosmos3", "qwen3vl"])
    ap.add_argument("--target", nargs="*", default=["vision", "vae"])
    ap.add_argument("--k", nargs="*", type=int, default=[1, 4])
    ap.add_argument(
        "--tactile_backbone",
        default="resnet18",
        choices=("resnet18", "ftp1", "anytouch"),
    )
    ap.add_argument(
        "--anytouch_checkpoint",
        default="/Data/lzl/huggingface/anytouch_encoder.pth",
    )
    ap.add_argument("--anytouch_forward_batch_size", type=int, default=128)
    args = ap.parse_args()

    combos = list(itertools.product(args.backbone, args.target, args.k))
    for backbone, target, k in combos:
        cfg = RoboContrastConfig(
            vision_backbone=backbone,
            perception_recon_target=target,
            num_cls_tokens=k,
            tactile_backbone=args.tactile_backbone,
            anytouch_checkpoint=args.anytouch_checkpoint,
            anytouch_forward_batch_size=args.anytouch_forward_batch_size,
        )

        model = RoboContrast(cfg).to(args.device)
        model.train()
        batch = make_batch(cfg, args.batch_size, args.device)
        loss, log = model.forward(batch)
        loss.backward()

        grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        wanted = sum(1 for p in model.parameters() if p.requires_grad)
        missing = [
            n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
        ]
        emb, _, _ = model.encode_perception(batch)
        phys, _ = model.encode_physical(batch)
        print(
            f"[{backbone:7s} recon={target:6s} K={k}] loss {loss.item():.4f} "
            f"contrastive {log['contrastive_loss']:.4f} percep_recon {log['percep_recon_loss']:.4f} "
            f"| percep emb {tuple(emb.shape)} phys emb {tuple(phys.shape)} "
            f"| grads {grads}/{wanted}"
        )
        print("             ", param_report(model))
        if missing:
            roots = sorted({".".join(n.split(".")[:3]) for n in missing})
            print(f"              no gradient ({len(missing)}): {roots}")
        del model, batch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
