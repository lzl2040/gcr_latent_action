"""Verify that ``trainable_scope`` actually controls which weights receive gradients.

Two things are checked, because a switch like this fails silently in both directions:

1. **Static**: the ``requires_grad`` flags match the scope, group by group.
2. **Dynamic**: after a real backward pass, every parameter that *should* train has a
   non-None gradient and every frozen one has none. This is the part that catches the
   ``torch.no_grad()`` bug -- a tower can have ``requires_grad=True`` on every weight and
   still receive nothing if an enclosing context manager detached the graph.

Run:
    python -u scripts/check_trainable_scope.py
"""

import argparse

import torch

from lerobot.common.policies.mot.modeling_mot import MoTConfig
from lerobot.common.policies.mot.world_model import (
    TRAINABLE_SCOPES,
    MoTWorldModel,
    WorldModelConfig,
)

CANON_DIM = 40


def groups(model: MoTWorldModel) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    """Bucket every parameter into vision / projector / und / gen."""
    out: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "vision": [],
        "projector": [],
        "und": [],
        "gen": [],
    }
    gen_marks = (
        "moe_gen",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
        "norm_added",
        "k_norm_und_for_gen",
        "proj_in",
        "proj_out",
        "time_embedder",
        "action_",
    )
    for name, p in model.named_parameters():
        if name.startswith("vision.img_processor."):
            out["vision"].append((name, p))
        elif name.startswith("vision."):
            out["projector"].append((name, p))
        elif any(m in name for m in gen_marks):
            out["gen"].append((name, p))
        else:
            out["und"].append((name, p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2, help="small model: the switch is per-group")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    for scope_name in TRAINABLE_SCOPES:
        mot_cfg = MoTConfig(
            num_hidden_layers=args.layers,
            gen_hidden_size=256,
            gen_num_attention_heads=8,
            gen_intermediate_size=512,
            action_dim=CANON_DIM,
        )
        cfg = WorldModelConfig(mot=mot_cfg, trainable_scope=scope_name)
        model = MoTWorldModel(cfg).to(device, dtype=torch.bfloat16)
        model.mot.gradient_checkpointing = True
        model.train()

        expected = cfg.scope()
        buckets = groups(model)

        # --- static ---
        static_ok = True
        counts = {}
        for g, params in buckets.items():
            want = getattr(expected, g)
            n_train = sum(1 for _, p in params if p.requires_grad)
            counts[g] = (n_train, len(params))
            if want and n_train == 0:
                print(f"  [FAIL] {g}: expected active trainable parameters, got 0")
                static_ok = False
            if not want and n_train != 0:
                print(f"  [FAIL] {g}: expected 0 trainable, got {n_train}")
                static_ok = False

        # --- dynamic: does a backward actually reach them? ---
        b = args.batch
        c, g = mot_cfg.latent_channels, cfg.latent_grid
        size = model.vision_image_size
        latents = torch.randn(b, c, 2, g, g, device=device, dtype=torch.bfloat16)
        images = torch.rand(b, 3, size, size, device=device, dtype=torch.bfloat16)
        text = torch.randint(0, 1000, (b, 8), device=device)
        actions = torch.randn(b, 8, CANON_DIM, device=device, dtype=torch.bfloat16)
        domain = torch.zeros(b, dtype=torch.long, device=device)

        out = model(
            latents=latents,
            pixel_values=images,
            text_ids=text,
            actions=actions,
            domain_id=domain,
            task="joint_action",
        )
        out["loss"].backward()

        dyn_ok = True
        got = {}
        for g, params in buckets.items():
            want = getattr(expected, g)
            n_grad = sum(1 for _, p in params if p.grad is not None)
            got[g] = n_grad
            if want and n_grad != counts[g][0]:
                print(
                    f"  [FAIL] {g}: {counts[g][0]} tensors are trainable but only "
                    f"{n_grad} received gradients"
                )
                dyn_ok = False
            if not want and n_grad != 0:
                print(f"  [FAIL] {g}: frozen but {n_grad} tensors received gradient")
                dyn_ok = False

        n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all_p = sum(p.numel() for p in model.parameters())
        flags = " ".join(
            f"{g}={int(getattr(expected, g))}"
            for g in ("vision", "projector", "und", "gen")
        )
        detail = " ".join(f"{g}:{counts[g][0]}/{counts[g][1]}->grad{got[g]}" for g in buckets)
        status = "PASS" if (static_ok and dyn_ok) else "FAIL"
        print(
            f"[{status}] scope={scope_name:<14} {flags}  "
            f"trainable={n_train_p / 1e6:8.1f}M / {n_all_p / 1e6:.1f}M  {detail}"
        )

        del model, out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
