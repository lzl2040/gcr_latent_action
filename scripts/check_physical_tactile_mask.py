#!/usr/bin/env python
"""Validate Physical Transformer masking for missing tactile tokens."""

import argparse

import torch

from lerobot.common.policies.ace.configuration_robo_contrast import RoboContrastConfig
from lerobot.common.policies.ace.modeling_robo_contrast import PhysicalEncoder


def make_batch(config: RoboContrastConfig, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = 2
    return {
        "observation.state": torch.randn(
            batch_size,
            config.chunk_size,
            config.max_state_dim,
            device=device,
        ),
        "state_mask": torch.ones(batch_size, config.max_state_dim, device=device),
        "action": torch.randn(
            batch_size,
            config.chunk_size,
            config.max_action_dim,
            device=device,
        ),
        "action_mask": torch.ones(batch_size, config.max_action_dim, device=device),
        "tactile_signal": torch.randn(
            batch_size,
            config.chunk_size,
            config.max_tactile_signal_dim,
            device=device,
        ),
        "tactile_signal_mask": torch.tensor([0.0, 1.0], device=device),
        "tactile_image": torch.randint(
            0,
            256,
            (
                batch_size,
                config.max_tactile_views,
                config.tactile_frames,
                3,
                config.tactile_img_size,
                config.tactile_img_size,
            ),
            dtype=torch.uint8,
            device=device,
        ),
        "tactile_image_mask": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
            device=device,
        ),
        "tactile_sensor_id": torch.zeros(
            batch_size,
            config.max_tactile_views,
            dtype=torch.long,
            device=device,
        ),
        "tactile_img_mean": torch.zeros(
            batch_size,
            config.max_tactile_views,
            3,
            device=device,
        ),
        "tactile_img_std": torch.ones(
            batch_size,
            config.max_tactile_views,
            3,
            device=device,
        ),
        "sample_rate": torch.full((batch_size,), 15, device=device),
    }


def clone_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in batch.items()}


def assert_changed(name: str, before: torch.Tensor, after: torch.Tensor) -> None:
    delta = (before - after).abs().max().item()
    if delta <= 1e-6:
        raise AssertionError(f"Valid {name} did not affect the physical embedding")


def check_mask(tokens_per_pad: int, device: torch.device) -> None:
    config = RoboContrastConfig(
        chunk_size=8,
        n_action_steps=4,
        group_size=4,
        hidden_dim=64,
        num_attention_heads=4,
        num_physical_layers=2,
        max_action_dim=4,
        max_state_dim=4,
        max_tactile_signal_dim=3,
        max_tactile_views=3,
        tactile_frames=4,
        tactile_tokens_per_pad=tokens_per_pad,
        tactile_img_size=32,
        tactile_pretrained=False,
        tactile_recon_weight=0.0,
        modality_dropout_tactile=0.0,
        modality_dropout_state=0.0,
        modality_dropout_action=0.0,
        dropout=0.0,
        projection_dim=32,
        gradient_checkpointing=False,
    )
    encoder = PhysicalEncoder(config).to(device).eval()
    batch = make_batch(config, device)

    with torch.no_grad():
        encoder.tactile_signal_gate.fill_(3.0)
        encoder.tactile_image_gate.fill_(3.0)
        baseline, _ = encoder(batch)

        perturbation = torch.linspace(
            -100.0,
            100.0,
            config.hidden_dim,
            device=device,
            dtype=encoder.missing_embed.weight.dtype,
        )
        encoder.missing_embed.weight[encoder.MOD_TAC_SIG].add_(perturbation)
        encoder.missing_embed.weight[encoder.MOD_TAC_IMG].sub_(perturbation)
        changed_missing, _ = encoder(batch)
        torch.testing.assert_close(
            baseline,
            changed_missing,
            rtol=0.0,
            atol=1e-6,
            msg="Masked tactile placeholder content changed the physical embedding",
        )

        changed_signal_batch = clone_batch(batch)
        changed_signal_batch["tactile_signal"][1].add_(10.0)
        changed_signal, _ = encoder(changed_signal_batch)
        assert_changed("tactile signal", changed_missing[1], changed_signal[1])

        changed_image_batch = clone_batch(batch)
        changed_image_batch["tactile_image"][1, 0].zero_()
        changed_image_batch["tactile_image"][1, 2].fill_(255)
        changed_image, _ = encoder(changed_image_batch)
        assert_changed("tactile image", changed_missing[1], changed_image[1])

    encoder.train()
    encoder.config.modality_dropout_tactile = 1.0
    with torch.no_grad():
        dropped_baseline, _ = encoder(batch)
        changed_tactile_batch = clone_batch(batch)
        changed_tactile_batch["tactile_signal"].add_(10.0)
        changed_tactile_batch["tactile_image"].bitwise_not_()
        dropped_changed, _ = encoder(changed_tactile_batch)
        torch.testing.assert_close(
            dropped_baseline,
            dropped_changed,
            rtol=0.0,
            atol=1e-6,
            msg="Dropped tactile content changed the physical embedding",
        )

    encoder.config.modality_dropout_tactile = 0.0
    encoder.train()
    encoder.zero_grad(set_to_none=True)
    no_tactile_batch = clone_batch(batch)
    no_tactile_batch["tactile_signal_mask"].zero_()
    no_tactile_batch["tactile_image_mask"].zero_()
    embedding, _ = encoder(no_tactile_batch)
    embedding.float().square().mean().backward()

    tactile_parameter_prefixes = (
        "signal_proj.",
        "tactile_cnn.",
        "tactile_temporal.",
        "tactile_img_proj.",
        "tactile_view_embed.",
        "tactile_token_embed.",
        "tactile_signal_gate",
        "tactile_image_gate",
    )
    missing_gradients = [
        name
        for name, parameter in encoder.named_parameters()
        if parameter.requires_grad
        and name.startswith(tactile_parameter_prefixes)
        and parameter.grad is None
    ]
    if missing_gradients:
        raise AssertionError(
            "No-tactile batch skipped tactile parameters needed by ZeRO: "
            f"{missing_gradients}"
        )

    print(
        "Physical tactile mask OK: "
        f"tokens_per_pad={tokens_per_pad} "
        f"embedding={tuple(embedding.shape)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2], choices=(1, 2))
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    for tokens_per_pad in args.tokens:
        check_mask(tokens_per_pad, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
