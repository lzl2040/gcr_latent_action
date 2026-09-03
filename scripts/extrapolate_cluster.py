"""Project 16-GPU wall-clock from the measured Phi-4-Multimodal A6000 anchors.

The primary convention is the one requested for robot video: one optimizer sample represents
one source-video frame. Therefore ``hours * 3600 * fps`` samples are divided by
``16 * batch_per_gpu``. Raw-video rows include mounted-storage latency, decode and the online
Wan VAE; latent rows remove the VAE and use one small sequential read per sample.

Run:
    python scripts/extrapolate_cluster.py --hours 30000 --batch-per-gpu 32
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

FPS = 20
GPUS = 16
GPUS_PER_NODE = 8
BATCH_REF = 32

# Measured on one RTX A6000 with scripts/train_mot_world.py. Times are the model column, so
# they exclude data loading and Wan VAE. gen_only includes fused AdamW. freeze_vision uses
# NO_OPT=1 because the unsharded optimizer cannot fit; 30 ms is added below for the estimated
# per-rank ZeRO-2 optimizer shard.
TASK_MIX = {
    "policy": 0.20,
    "i2v": 0.20,
    "v2v": 0.15,
    "joint_action": 0.15,
    "inv_dyn": 0.10,
    "t2v": 0.10,
    "fwd_dyn": 0.05,
    "t2i": 0.05,
}


@dataclass(frozen=True)
class Scope:
    task_ms_b32: dict[str, float]
    trainable_params: float
    fixed_ms: float
    per_clip_ms: float
    backward_fraction: float
    optimizer_ms_b32: float
    activation_gib_per_clip: float
    strategy: str
    measured_memory: str

    def model_ms(self, batch: int) -> float:
        scale = batch / BATCH_REF
        return self.fixed_ms + self.per_clip_ms * batch + self.optimizer_ms_b32 * scale


GEN_B32 = {
    "t2i": 616.0,
    "t2v": 1351.0,
    "i2v": 3804.0,
    "v2v": 3607.0,
    "joint_action": 3810.0,
    "fwd_dyn": 3809.0,
    "inv_dyn": 3816.0,
    "policy": 3052.0,
}
GEN_B64 = {
    "t2i": 1093.0,
    "t2v": 2532.0,
    "i2v": 7244.0,
    "v2v": 7320.0,
    "joint_action": 7593.0,
    "fwd_dyn": 7596.0,
    "inv_dyn": 7589.0,
    "policy": 6138.0,
}
FREEZE_B32 = {
    "t2i": 767.0,
    "t2v": 1513.0,
    "i2v": 8883.0,
    "v2v": 8900.0,
    "joint_action": 9173.0,
    "fwd_dyn": 9092.0,
    "inv_dyn": 9080.0,
    "policy": 8351.0,
}


def weighted(values: dict[str, float]) -> float:
    return sum(TASK_MIX[name] * values[name] for name in TASK_MIX)


GEN_WEIGHTED_B32 = weighted(GEN_B32)
GEN_WEIGHTED_B64 = weighted(GEN_B64)
GEN_PER_CLIP = (GEN_WEIGHTED_B64 - GEN_WEIGHTED_B32) / (64 - BATCH_REF)
GEN_FIXED = GEN_WEIGHTED_B32 - GEN_PER_CLIP * BATCH_REF
FREEZE_WEIGHTED_B32 = weighted(FREEZE_B32)

SCOPES = {
    "gen_only": Scope(
        task_ms_b32=GEN_B32,
        trainable_params=1_420_254_656,
        fixed_ms=GEN_FIXED,
        per_clip_ms=GEN_PER_CLIP,
        backward_fraction=0.31,
        optimizer_ms_b32=0.0,
        activation_gib_per_clip=0.26,
        strategy="DDP",
        measured_memory="b32 26.9 GiB; b64 35.8 GiB; b128 36.5 GiB model-only",
    ),
    "freeze_vision": Scope(
        task_ms_b32=FREEZE_B32,
        trainable_params=5_543_981_760,
        fixed_ms=GEN_FIXED,
        per_clip_ms=(FREEZE_WEIGHTED_B32 - GEN_FIXED) / BATCH_REF,
        backward_fraction=0.72,
        optimizer_ms_b32=30.0,
        activation_gib_per_clip=0.12,
        strategy="ZeRO-2",
        measured_memory="b32 25.6 GiB; b64 34.3 GiB (NO_OPT=1)",
    ),
}

# A6000 anchors: online Wan VAE at batch 32 and mounted-storage assumptions.
VAE_MS_B32 = 1306.0
RAW_VIDEO_LATENCY_MS = 200.0
LATENT_LATENCY_MS = 60.0
WORKERS_PER_GPU = 12

# Total model excludes the frozen 705M Wan VAE. action_dim=40 is the real-loader model.
TOTAL_PARAMS = 6_066_925_184
BF16_BYTES = 2


@dataclass(frozen=True)
class Device:
    usable_bf16_tflops: float
    efficiency_low: float
    efficiency_high: float
    memory_gib: float


# Efficiency is relative to the measured A6000 kernel mix, not marketing peak. The B200 band
# is intentionally broad because the 2048-wide GEN matrices cannot use its tensor cores as
# efficiently as Phi's larger matrices.
DEVICES = {
    "RTX A6000": Device(77.4, 1.00, 1.00, 47.6),
    "A100-40GB": Device(312.0, 0.75, 0.90, 39.5),
    "A100-80GB": Device(312.0, 0.75, 0.90, 79.2),
    "B200": Device(2250.0, 0.30, 0.55, 180.0),
}

NVLINK_GBPS = 250.0
LATENCY_US = 8.0
FABRICS = {
    "8x200G IB": 200.0,
    "1x200G IB": 25.0,
    "100G Eth": 12.5,
}
MIN_EXPOSED = 0.05


def device_ratios(device: Device) -> tuple[float, float]:
    """Return slow/fast compute ratios relative to the A6000."""
    base = DEVICES["RTX A6000"].usable_bf16_tflops
    return (
        device.usable_bf16_tflops * device.efficiency_low / base,
        device.usable_bf16_tflops * device.efficiency_high / base,
    )


def communication_ms(trainable: float, node_gbps: float) -> float:
    """Hierarchical bf16 gradient all-reduce for two nodes of eight GPUs."""
    gradient_gb = trainable * BF16_BYTES / 1e9
    intra = (
        2
        * (GPUS_PER_NODE - 1)
        / GPUS_PER_NODE
        * gradient_gb
        / NVLINK_GBPS
        * 1000.0
    )
    nodes = GPUS // GPUS_PER_NODE
    inter_gb = 2 * (nodes - 1) / nodes * gradient_gb
    inter = inter_gb / node_gbps * 1000.0 + LATENCY_US * 2 * (GPUS - 1) / 1000.0
    return intra + inter


def exposed_communication_ms(total: float, compute: float, backward_fraction: float) -> float:
    return max(total - backward_fraction * compute, MIN_EXPOSED * total)


def loader_ms(batch: int, latency_ms: float) -> float:
    clips_per_second = WORKERS_PER_GPU / (latency_ms / 1000.0)
    return batch / clips_per_second * 1000.0


def state_gib(scope: Scope) -> float:
    """Resident state for the strategy used in the projection.

    DDP matches the measured native BF16 AdamW run: replicated BF16 params, gradients and two
    BF16 moments. ZeRO-2 conservatively assumes sharded BF16 gradients plus FP32 master params
    and two FP32 moments.
    """
    if scope.strategy == "DDP":
        byte_count = TOTAL_PARAMS * 2 + scope.trainable_params * 6
    else:
        byte_count = TOTAL_PARAMS * 2 + scope.trainable_params * 14 / GPUS
    return byte_count / 2**30


def memory_gib(scope: Scope, batch: int) -> float:
    return state_gib(scope) + scope.activation_gib_per_clip * batch


def compute_ms(scope: Scope, device: Device, batch: int, raw_video: bool) -> tuple[float, float]:
    """Return fast/slow compute endpoints, including online VAE for raw video."""
    slow_ratio, fast_ratio = device_ratios(device)
    a6000 = scope.model_ms(batch)
    if raw_video:
        a6000 += VAE_MS_B32 * batch / BATCH_REF
    fast = a6000 / fast_ratio
    slow = a6000 / slow_ratio
    return fast, slow


def step_range_ms(
    scope: Scope,
    device: Device,
    node_gbps: float,
    batch: int,
    raw_video: bool,
) -> tuple[float, float, str]:
    fast_compute, slow_compute = compute_ms(scope, device, batch, raw_video)
    total_comm = communication_ms(scope.trainable_params, node_gbps)
    latency = RAW_VIDEO_LATENCY_MS if raw_video else LATENT_LATENCY_MS
    io = loader_ms(batch, latency)

    endpoints = []
    bounds = set()
    for compute in (fast_compute, slow_compute):
        exposed = exposed_communication_ms(total_comm, compute, scope.backward_fraction)
        compute_path = compute + exposed
        step = max(compute_path, io)
        endpoints.append(step)
        if io >= compute_path:
            bounds.add("I/O")
        elif exposed > compute:
            bounds.add("comms")
        else:
            bounds.add("compute")
    return min(endpoints), max(endpoints), "/".join(sorted(bounds))


def fmt_ms_range(fast: float, slow: float) -> str:
    if abs(fast - slow) < 0.5:
        return f"{fast:.0f}"
    return f"{fast:.0f}-{slow:.0f}"


def fmt_days_range(fast: float, slow: float) -> str:
    if abs(fast - slow) < 0.05:
        return f"{fast:.1f} d"
    return f"{fast:.1f}-{slow:.1f} d"


def main(hours: int, batch: int) -> None:
    frames = hours * 3600 * FPS
    global_batch = GPUS * batch
    steps = frames / global_batch

    print(
        f"{hours:,} h x 3600 x {FPS} fps = {frames:.3g} frame-samples; "
        f"16 x {batch} = {global_batch} global batch; {steps:.3g} optimizer steps"
    )
    print(
        f"A6000 measured stage-3 model @ b32: gen_only={GEN_WEIGHTED_B32:.0f} ms, "
        f"freeze_vision={FREEZE_WEIGHTED_B32:.0f} ms (+30 ms estimated ZeRO optimizer)"
    )
    print(f"online Wan VAE @ b32: {VAE_MS_B32:.0f} ms\n")

    for data_name, raw_video in (("raw video", True), ("cached latent", False)):
        print(f"### {data_name}")
        print(
            f"{'scope':>14s} {'device':>11s} {'strategy':>9s} {'memory':>10s} "
            f"{'fabric':>11s} {'step ms':>11s} {'bound':>11s} {'wall time':>14s}"
        )
        for scope_name, scope in SCOPES.items():
            for device_name, device in DEVICES.items():
                need = memory_gib(scope, batch)
                if need > device.memory_gib:
                    print(
                        f"{scope_name:>14s} {device_name:>11s} {scope.strategy:>9s} "
                        f"{need:9.1f}G {'-':>11s} {'OOM':>11s} {'memory':>11s} {'-':>14s}"
                    )
                    continue
                for fabric_name, node_gbps in FABRICS.items():
                    fast_step, slow_step, bound = step_range_ms(
                        scope,
                        device,
                        node_gbps,
                        batch,
                        raw_video,
                    )
                    fast_days = steps * fast_step / 1000.0 / 86400.0
                    slow_days = steps * slow_step / 1000.0 / 86400.0
                    print(
                        f"{scope_name:>14s} {device_name:>11s} {scope.strategy:>9s} "
                        f"{need:9.1f}G {fabric_name:>11s} "
                        f"{fmt_ms_range(fast_step, slow_step):>11s} "
                        f"{bound:>11s} {fmt_days_range(fast_days, slow_days):>14s}"
                    )
        print()

    print("Measured memory anchors:")
    for name, scope in SCOPES.items():
        print(f"  {name:>14s}: {scope.measured_memory}")
    print(
        "\nA100/B200 values are projections. Ranges come from achievable-efficiency bands; "
        "raw-video I/O assumes 12 workers/GPU and 200 ms/clip on mounted storage."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=30_000)
    parser.add_argument("--batch-per-gpu", type=int, default=32)
    args = parser.parse_args()
    main(args.hours, args.batch_per_gpu)
