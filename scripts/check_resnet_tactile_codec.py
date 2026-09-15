#!/usr/bin/env python
"""Validate the reusable ResNet tactile patch codec and contrastive temporal head."""

import argparse

import torch

from lerobot.common.policies.ace.modeling_robo_contrast import (
    TactileImageEncoder,
    TactilePatchDecoder,
    TactilePatchTemporal,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    encoder = TactileImageEncoder(pretrained=False).to(device=device, dtype=dtype)
    temporal = TactilePatchTemporal(
        dim=512,
        num_frames=args.frames,
        num_tokens=args.tokens,
    ).to(device=device, dtype=dtype)
    decoder = TactilePatchDecoder(512, args.image_size).to(device=device, dtype=dtype)

    images = torch.randint(
        0,
        256,
        (args.batch_size, args.frames, 3, args.image_size, args.image_size),
        dtype=torch.uint8,
        device=device,
    )
    pooled, patches = encoder(images)
    expected_grid = args.image_size // encoder.output_stride
    expected_patches = (
        args.batch_size,
        args.frames,
        512,
        expected_grid,
        expected_grid,
    )
    if patches.shape != expected_patches:
        raise AssertionError(f"Expected patches {expected_patches}, got {tuple(patches.shape)}")
    if pooled.shape != (args.batch_size, args.frames, 512):
        raise AssertionError(f"Unexpected pooled shape {tuple(pooled.shape)}")
    pooled_from_patches = patches.mean(dim=(-1, -2))
    if not torch.equal(pooled, pooled_from_patches):
        raise AssertionError("Pooled tactile embedding is not the mean of the reusable patches")

    temporal_spatial = temporal.forward_spatial(patches)
    expected_spatial = (
        args.batch_size,
        args.tokens,
        512,
        expected_grid,
        expected_grid,
    )
    if temporal_spatial.shape != expected_spatial:
        raise AssertionError(
            f"Expected temporal spatial features {expected_spatial}, "
            f"got {tuple(temporal_spatial.shape)}"
        )
    physical_tokens = temporal.pool_spatial(temporal_spatial)
    expected_tokens = (args.batch_size, args.tokens, 512)
    if physical_tokens.shape != expected_tokens:
        raise AssertionError(
            f"Expected physical tokens {expected_tokens}, got {tuple(physical_tokens.shape)}"
        )
    torch.testing.assert_close(
        physical_tokens,
        temporal_spatial.float().mean(dim=(-1, -2)).to(temporal_spatial.dtype),
        msg="Zero-initialized learned spatial pooling does not match the previous mean",
    )

    decoded = decoder(patches[:, 0])
    if decoded.shape != (args.batch_size, 3, args.image_size, args.image_size):
        raise AssertionError(f"Unexpected decoded shape {tuple(decoded.shape)}")
    decoded_sequence = decoder(patches[:, :2])
    if decoded_sequence.shape != (
        args.batch_size,
        2,
        3,
        args.image_size,
        args.image_size,
    ):
        raise AssertionError(
            f"Unexpected sequence decode shape {tuple(decoded_sequence.shape)}"
        )

    loss = physical_tokens.float().square().mean() + decoded.float().square().mean()
    loss.backward()
    modules = {
        "encoder": encoder,
        "temporal": temporal,
        "decoder": decoder,
    }
    for name, module in modules.items():
        missing = [
            parameter_name
            for parameter_name, parameter in module.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if missing:
            raise AssertionError(f"{name} parameters without gradients: {missing}")
    learning_grads = {
        "spatial_mixing_gate": temporal.spatial_mixing_gate.grad,
        "spatial_pool_score": temporal.spatial_pool_score.weight.grad,
    }
    for name, gradient in learning_grads.items():
        if gradient is None or gradient.abs().max().item() <= 0:
            raise AssertionError(f"{name} did not receive a learning signal at initialization")

    with torch.no_grad():
        temporal.spatial_mixing_gate.fill_(3.0)
        base_spatial = temporal.forward_spatial(patches.detach())
        changed_patches = patches.detach().clone()
        changed_patches[:, -1, 0, 0, 0] += 10
        changed_spatial = temporal.forward_spatial(changed_patches)
        spatial_delta = (changed_spatial - base_spatial).abs()
        if spatial_delta[:, :, :, 0, 0].max().item() <= 0:
            raise AssertionError("Changing a future patch did not change its temporal feature")
        spatial_delta[:, :, :, 0, 0] = 0
        if spatial_delta.max().item() <= 0:
            raise AssertionError(
                "A future-patch change did not propagate across spatial locations"
            )

    temporal.zero_grad(set_to_none=True)
    opened_tokens = temporal(patches.detach())
    probe = torch.linspace(
        -1.0,
        1.0,
        opened_tokens.numel(),
        device=device,
    ).view_as(opened_tokens)
    (opened_tokens.float() * probe).sum().backward()
    spatial_gradients = {
        "coordinate_projection": temporal.spatial_coordinate_projection.weight.grad,
        "dilation_1": temporal.spatial_blocks[0].depthwise.weight.grad,
        "dilation_2": temporal.spatial_blocks[1].depthwise.weight.grad,
    }
    for name, gradient in spatial_gradients.items():
        if gradient is None or gradient.abs().max().item() <= 0:
            raise AssertionError(f"{name} did not receive gradient after opening the spatial gate")

    print(
        "ResNet tactile codec OK: "
        f"patches={tuple(patches.shape)} "
        f"temporal_spatial={tuple(temporal_spatial.shape)} "
        f"physical={tuple(physical_tokens.shape)} "
        f"decoded={tuple(decoded.shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
