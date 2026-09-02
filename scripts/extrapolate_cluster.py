"""Project cluster wall-clock for the MoT world model from the measured RTX A6000 numbers.

Extrapolating across GPU generations is the part of a plan most often quoted as one confident
number and most often wrong. Two things are done here to avoid that:

* **Achievable throughput, not marketing peak.** The A6000 anchor matters because it is
  GA102, whose bf16 tensor cores run at *half* rate with FP32 accumulate -- usable peak ~77
  TFLOP/s, not the 155 on the datasheet. The earlier stage profile measured the und expert at
  81 TFLOP/s, i.e. essentially hardware peak, which is what makes the anchor trustworthy.
  Efficiency on the other devices is a *band*, not a point, because it is an assumption.
* **Communication is modelled, not fudged.** A 1.44B-parameter bf16 all-reduce is 2.88 GB per
  step. That is negligible against a 2.1 s A6000 step and decidedly not negligible against a
  ~70 ms B200 step, which is exactly the regime change a single "scaling efficiency" constant
  would hide. The script reports the exposed comms so the reader can see when the job stops
  being compute-bound.

The anchor is a real measurement: scripts/train_mot_world.py at batch 32 on one RTX A6000.

Run:  python scripts/extrapolate_cluster.py [--hours 10000]
"""

import argparse

# --- measured: scripts/train_mot_world.py, batch 32, 1x RTX A6000, 15 steps/task ------------
# Both scopes were measured through the same code path (fused AdamW, expandable_segments), so
# the ratio between them is like-for-like.
BATCH_REF = 32
SCOPES = {
    # scope -> (per-task model ms at BATCH_REF, trainable params, {batch: measured peak GiB})
    # The memory column is a *measurement*, including the OOM boundary: the largest batch that
    # ran and the smallest that did not. Extrapolating it linearly was wrong -- gen_only's peak
    # tracks activations while freeze_vision's is pinned by optimizer state, so they have
    # different slopes and only the measured points are trustworthy.
    "gen_only": (
        {"t2i": 614.0, "t2v": 1345.0, "i2v": 2147.0, "v2v": 2324.0, "action": 2338.0},
        1.445e9,
        {8: 20.4, 32: 26.1, 64: 35.0, 128: None},  # None = OOM on a 47.65 GiB A6000
    ),
    "freeze_vision": (
        {"t2i": 1173.0, "t2v": 2069.0, "i2v": 2613.0, "v2v": 2613.0, "action": 3018.0},
        5.281e9,
        {8: 40.8, 32: 40.9, 64: None},
    ),
}
# Second anchor at batch 64 (gen_only only; freeze_vision OOMs there). Used to fit the
# marginal per-clip cost instead of assuming the step time is proportional to the batch --
# it is not, there is a ~281 ms batch-independent floor of launch and optimizer overhead.
MODEL_MS_B64 = {"t2i": 1066.0, "t2v": 2479.0, "i2v": 4033.0, "v2v": 4170.0, "action": 4420.0}

VAE_MS = 1274.0  # task- and scope-independent: the same clip is encoded either way
NUM_WORKERS = 12

# --- I/O, measured by scripts/measure_io.py ---------------------------------------------------
# LOADER_MS_WARM is a *single* worker hitting the page cache on a local disk. It was the number
# used in earlier revisions of this script and it is wildly optimistic for a cluster: /Data here
# is /dev/sdc1, a Seagate ST16000NM000J with ROTA=1 -- a spinning disk. Reading cold, one worker
# at a time, the same clip costs 780.8 ms, i.e. 21x more. A blobfuse/NFS mount sits in between:
# no seek penalty, but a network round trip per read, and there are 6 read syscalls per clip.
LOADER_MS_WARM = 37.2
LOADER_MS_COLD_HDD = 780.8
LOADER_MS_MOUNT = 200.0  # assumption for a blobfuse/NFS mount; the two above are measured
BYTES_PER_CLIP = 677 * 1024
READS_PER_CLIP = 6

MIX = {"action": 0.5, "i2v": 0.2, "v2v": 0.15, "t2v": 0.1, "t2i": 0.05}  # world_model.STAGE3_MIX

# bf16 bytes per trainable parameter that must live on the card, by parallelism strategy.
# ddp: params(all) + grads + 2 Adam moments.  zero2: grads and moments sharded over GPUS.
BYTES_PARAM = 2

# --- dataset --------------------------------------------------------------------------------
HOURS = 10_000  # overridden by --hours
FPS = 20
CHUNK_SECONDS = 1.6
PIXEL_FRAMES_PER_CLIP = 9
LATENT_BYTES_PER_CLIP = 48 * 3 * 16 * 16 * 2  # C x T x H x W, bf16

# --- hardware ---------------------------------------------------------------------------------
# (usable dense bf16 TFLOP/s, (efficiency_low, efficiency_high), usable GiB)
# Efficiency is relative to the A6000's measured behaviour, which is 1.0 by construction.
# B200's band is wide and starts low on purpose: its headline peak is ~29x the A6000's, but
# our gen expert reaches only 67% of usable peak even on the A6000 (52.1 vs the und expert's
# 78.5 TFLOP/s), because its matrices are smaller -- and that gap widens as tensor cores grow.
# A100-40GB and A100-80GB have identical compute; only capacity differs, and that is exactly
# what decides which scope and batch are reachable.
A6000_GIB = 47.6  # the card every measurement in SCOPES was taken on
DEVICES = {
    "RTX A6000": (77.4, (1.00, 1.00), A6000_GIB),
    "A100-40GB": (312.0, (0.75, 0.90), 39.5),
    "A100-80GB": (312.0, (0.75, 0.90), 79.2),
    "B200": (2250.0, (0.30, 0.55), 180.0),
}
GPUS = 16

# --- interconnect -----------------------------------------------------------------------------
# 16 GPUs is 2 nodes of 8, and that makes the *node's* NIC the bottleneck rather than the
# per-GPU bandwidth a flat model would use. NCCL runs a hierarchical all-reduce: reduce-scatter
# over NVLink inside the node, all-reduce the 1/8 shards across nodes, all-gather back over
# NVLink. So the traffic crossing one node's NIC is 2*(nodes-1)/nodes * G -- for 2 nodes, the
# whole gradient -- shared by all 8 local GPUs. A flat "20 GB/s per GPU" both overstates a
# commodity VM and understates a properly wired ND-series node, by an order of magnitude each
# way, which is why this is now a scenario rather than a constant.
GPUS_PER_NODE = 8
NVLINK_GBPS = 250.0  # per GPU, bidirectional; A100 NVLink3, conservative for B200
LATENCY_US = 8.0     # per ring hop; 2*(GPUS-1) hops. Included for completeness -- at 30 hops
                     # it is 0.24 ms, i.e. never the issue at these message sizes.
FABRICS = {
    "8x200G IB": 200.0,  # ND A100 v4 / ND H100 v5 class: one HDR NIC per GPU, 200 GB/s/node
    "1x200G IB": 25.0,   # a single HDR NIC serving all 8 GPUs
    "100G Eth": 12.5,    # commodity cloud VM
}
# How much of the all-reduce can hide behind the backward pass. A flat "70% is hidden" is wrong
# in both directions: it under-hides a fast fabric and, worse, it silently claims to hide 919 ms
# of 100 GbE traffic inside a 1.4 s step. The honest bound is that overlap can only use the
# *backward* window, and the final bucket can never overlap at all.
# measure_forward.py: fwd 840 ms of a 3052 ms fwd+bwd, so backward is ~72% of the step.
BWD_FRACTION = 0.72
MIN_EXPOSED = 0.05  # the last gradient bucket, which starts only after backward ends

# --- data loading -----------------------------------------------------------------------------
# The loader runs in parallel with compute, so it only matters when it cannot keep up:
# step = max(compute + exposed comms, batch / loader throughput). Throughput is capped by the
# number of worker processes a node can afford -- an 8-GPU node with 96 vCPU gives 12 per GPU,
# and that ceiling is what turns a fast GPU into an I/O-bound one.
WORKERS_PER_GPU = 12
IO_SCENARIOS = {
    "local warm": LOADER_MS_WARM,     # measured, page cache hot -- what earlier revisions used
    "blob video": LOADER_MS_MOUNT,    # 6 reads/clip, one network round trip each
    "blob slow": 500.0,               # throttled or cold container
    "blob latents": 60.0,             # 1 sequential 72 KiB read, no decode
}

TOTAL_PARAMS = 5.588e9
# Fitted on gen_only's three measured peaks (20.4 / 26.1 / 35.0 GiB at batch 8 / 32 / 64),
# which are self-consistent to 0.24-0.28 GiB per clip. freeze_vision's measured peak is far
# below what this predicts, so for that scope the number below is a conservative bound and
# the measured boundary (batch 32 fits, batch 64 OOMs on 47.6 GiB) is the real answer.
ACT_GIB_PER_CLIP = 0.26


def comms_ms(trainable: float, node_gbps: float) -> tuple[float, float]:
    """Return ``(intra_node_ms, inter_node_ms)`` for one hierarchical all-reduce.

    ``trainable`` is a parameter count; gradients are bf16. NCCL reduce-scatters over NVLink
    inside the node, all-reduces the 1/GPUS_PER_NODE shards across nodes, then all-gathers back.
    The inter-node term is the volume crossing *one node's* NIC -- all local GPUs share it,
    which is the step a flat per-GPU bandwidth model skips and the reason a 100G VM is 16x
    worse than an ND-series node.
    """
    grad_gb = trainable * BYTES_PARAM / 1e9
    nodes = max(1, GPUS // GPUS_PER_NODE)
    intra = 2 * (GPUS_PER_NODE - 1) / GPUS_PER_NODE * grad_gb / NVLINK_GBPS * 1000.0
    nic_gb = 2 * (nodes - 1) / nodes * grad_gb
    inter = nic_gb / node_gbps * 1000.0 + LATENCY_US * 2 * (GPUS - 1) / 1000.0
    return intra, inter


def exposed_comms(total_comms: float, compute: float) -> float:
    """Comms that cannot hide behind the backward pass, so it lands on the critical path."""
    return max(total_comms - BWD_FRACTION * compute, MIN_EXPOSED * total_comms)


def loader_ms(batch: int, latency_ms: float) -> float:
    """Wall-clock the loader needs per step, given a hard ceiling of WORKERS_PER_GPU."""
    return batch / (WORKERS_PER_GPU / (latency_ms / 1000.0)) * 1000.0
# Fitted on gen_only's three measured peaks (20.4 / 26.1 / 35.0 GiB at batch 8 / 32 / 64),
# which are self-consistent to 0.24-0.28 GiB per clip. freeze_vision's measured peak is far
# below what this predicts, so for that scope the number below is a conservative bound and
# the measured boundary (batch 32 fits, batch 64 OOMs on 47.6 GiB) is the real answer.
ACT_GIB_PER_CLIP = 0.26


def state_gib(trainable: float, shard: int) -> float:
    """Params + grads + two bf16 Adam moments. ``shard`` = 1 for DDP, GPUS for ZeRO-2."""
    return (TOTAL_PARAMS * BYTES_PARAM + trainable * BYTES_PARAM * 3 / shard) / 2**30


def fmt_time(seconds: float) -> str:
    d = seconds / 86400
    return f"{d:.1f} d" if d >= 1 else f"{seconds / 3600:.1f} h"


def marginal_fit(scope: str) -> tuple[float, float]:
    """Return ``(fixed_ms, ms_per_clip)`` on the A6000 for the stage-3 mix.

    Step time is *not* proportional to the batch: there is a batch-independent floor of kernel
    launch and optimizer overhead. gen_only measured 2112 ms at batch 32 and 3943 ms at batch
    64 -- doubling the batch cost 1.87x, not 2x. Assuming proportionality would overstate a
    batch-128 step by ~13%. Only gen_only has a second anchor, so freeze_vision reuses
    gen_only's fixed floor, which is the conservative choice (it attributes more of the step
    to the batch-dependent part).
    """
    model_ms = SCOPES[scope][0]
    w32 = sum(MIX[t] * model_ms[t] for t in MIX)
    if scope == "gen_only":
        w64 = sum(MIX[t] * MODEL_MS_B64[t] for t in MIX)
        per_clip = (w64 - w32) / (64 - BATCH_REF)
        return w32 - per_clip * BATCH_REF, per_clip
    fixed = marginal_fit("gen_only")[0]
    return fixed, (w32 - fixed) / BATCH_REF


def samples_by_convention() -> dict[str, tuple[float, str]]:
    """How many training samples is "10,000 hours"? Three defensible answers, 32x apart.

    Our sample is a 1.6 s window that READS 9 pixel frames and SPANS 32 of them, so the
    conversion from "hours of video" to "optimizer steps" is a modelling choice, not a fact.
    """
    frames = HOURS * 3600 * FPS
    clips = HOURS * 3600 / CHUNK_SECONDS
    return {
        "frames": (frames, f"1 sample = 1 frame ({frames:.3g} / global batch)"),
        "windows": (clips, f"1 sample = one non-overlapping {CHUNK_SECONDS}s window"),
        "coverage": (clips * frames / (clips * PIXEL_FRAMES_PER_CLIP) ,
                     "enough windows that every frame is read at least once"),
    }


def best_batch(
    trainable: float, mem: float, peaks: dict[int, float | None]
) -> tuple[int, float, str] | None:
    """Largest batch that fits, preferring DDP and falling back to ZeRO-2.

    Where a measurement exists it wins over the model. ``peaks`` was taken under DDP on a
    47.6 GiB A6000, so a measured OOM rules that batch out on any card of that capacity --
    the estimate must not quietly promote a batch that was observed to fail. OOM is treated as
    monotonic: freeze_vision was measured to OOM at 64, so 128 is out too even though nobody
    ran it.
    """
    smallest_oom = min((b for b, g in peaks.items() if g is None), default=None)
    for batch in (128, 64, 32):
        ruled_out = (
            smallest_oom is not None and batch >= smallest_oom and mem <= A6000_GIB
        )
        if ruled_out:
            continue
        # A measured peak beats the model in both directions. ACT_GIB_PER_CLIP is fitted on
        # gen_only and over-predicts freeze_vision, which would otherwise demote a
        # configuration that was observed to run under plain DDP.
        measured = peaks.get(batch)
        if measured is not None and measured <= mem:
            return batch, measured, "DDP"
        need_ddp = state_gib(trainable, 1) + ACT_GIB_PER_CLIP * batch
        need_z2 = state_gib(trainable, GPUS) + ACT_GIB_PER_CLIP * batch
        if need_ddp <= mem:
            return batch, need_ddp, "DDP"
        if need_z2 <= mem:
            return batch, need_z2, "ZeRO-2*"
    return None


def main(hours: int) -> None:
    global HOURS
    HOURS = hours
    clips = HOURS * 3600 / CHUNK_SECONDS
    frames = HOURS * 3600 * FPS
    frames_seen = clips * PIXEL_FRAMES_PER_CLIP
    conventions = samples_by_convention()

    print(f"{HOURS:,} h @ {FPS} fps        : {frames:.3g} pixel frames on disk\n")
    print(f"One sample is a {CHUNK_SECONDS} s window that READS {PIXEL_FRAMES_PER_CLIP} "
          f"frames and SPANS {int(CHUNK_SECONDS * FPS)}, so 'how many steps")
    print(f"is {HOURS:,} hours' has three defensible answers and they differ by 32x:")
    for name, (n, why) in conventions.items():
        print(f"  {name:>9s} : {n:9.3g} samples   {why}")
    print(f"            'coverage' exists because one pass over the windows touches only "
          f"{frames_seen / frames * 100:.0f}%")
    print(f"            of the frames on disk ({frames_seen:.3g} of {frames:.3g}); full coverage "
          f"needs {frames / frames_seen:.1f}x.")
    print(f"\ncached-latent footprint     : {clips * LATENT_BYTES_PER_CLIP / 1e12:.2f} TB")
    print(f"per-clip I/O                : {BYTES_PER_CLIP / 1024:.0f} KiB in "
          f"{READS_PER_CLIP} reads; {LOADER_MS_WARM:.0f} ms warm-local / "
          f"{LOADER_MS_MOUNT:.0f} ms mount / {LOADER_MS_COLD_HDD:.0f} ms cold-HDD (1 worker)")
    print(f"loader ceiling              : {WORKERS_PER_GPU} workers/GPU -> "
          + ", ".join(f"{k} {WORKERS_PER_GPU / (v / 1000):.0f} clips/s"
                      for k, v in IO_SCENARIOS.items()))
    print(f"topology                    : {GPUS} GPUs = {GPUS // GPUS_PER_NODE} nodes x "
          f"{GPUS_PER_NODE}; NVLink {NVLINK_GBPS:.0f} GB/s/GPU, NIC per scenario\n")

    for scope, (model_ms, trainable, peaks) in SCOPES.items():
        w32 = sum(MIX[t] * model_ms[t] for t in MIX)
        fixed_ms, per_clip_ms = marginal_fit(scope)
        grad_gb = trainable * BYTES_PARAM / 1e9
        meas = "  ".join(
            f"b{b}={'OOM' if g is None else f'{g:.1f}G'}" for b, g in sorted(peaks.items())
        )

        print(f"##### scope = {scope}  ({trainable / 1e9:.2f}B trainable) #####")
        print(f"A6000 step (batch {BATCH_REF})     : {w32:.0f} ms "
              f"= {fixed_ms:.0f} ms fixed + {per_clip_ms:.1f} ms/clip")
        print(f"measured peak memory        : {meas}   (47.6 GiB card)")
        print(f"resident state (no acts)    : DDP {state_gib(trainable, 1):.1f} GiB   "
              f"ZeRO-2 x{GPUS} {state_gib(trainable, GPUS):.1f} GiB")
        print(f"gradient all-reduce         : {grad_gb:.2f} GB/GPU; NVLink phase "
              f"{comms_ms(trainable, 1e9)[0]:.0f} ms, NIC phase per fabric below\n")

        print(f"{'device':>11s} {'b/GPU':>6s} {'mem':>15s} {'fabric':>11s} {'io':>13s} "
              f"{'compute':>8s} {'comms':>7s} {'io':>7s} {'step':>7s} {'bound':>8s} "
              f"{'frames':>9s}")
        for dev, (peak, (eff_lo, eff_hi), mem) in DEVICES.items():
            fit = best_batch(trainable, mem, peaks)
            if fit is None:
                print(f"{dev:>11s} {'-':>6s} {'>' + f'{mem:.0f}G even ZeRO-2':>15s}")
                continue
            batch, need, tag = fit
            ratio = (peak * eff_hi) / DEVICES["RTX A6000"][0]
            compute = (fixed_ms + per_clip_ms * batch) / ratio
            for fabric, node_gbps in FABRICS.items():
                intra, inter = comms_ms(trainable, node_gbps)
                exposed = exposed_comms(intra + inter, compute)
                for io_name, io_lat in (("blob video", IO_SCENARIOS["blob video"]),
                                        ("blob latents", IO_SCENARIOS["blob latents"])):
                    io = loader_ms(batch, io_lat)
                    step = max(compute + exposed, io)
                    bound = ("io" if io > compute + exposed
                             else "comms" if exposed > compute else "compute")
                    thru = batch / (step / 1000.0) * GPUS
                    t = fmt_time(conventions["frames"][0] / thru)
                    print(f"{dev:>11s} {batch:6d} {f'{need:.0f}G {tag}':>15s} {fabric:>11s} "
                          f"{io_name:>13s} {compute:7.0f}ms {exposed:6.0f}ms {io:6.0f}ms "
                          f"{step:6.0f}ms {bound:>8s} {t:>9s}")
        print()

    print("##### where the time actually goes #####")
    print("Three things can bind, and which one does changes with the GPU:")
    print("  compute  -- the A6000 and, on a good fabric, the A100")
    print("  comms    -- the gradient all-reduce stops hiding behind the backward pass once")
    print("              the step gets short, and on a 100G VM it dominates outright")
    print(f"  io       -- {WORKERS_PER_GPU} workers/GPU at 200 ms/clip is "
          f"{WORKERS_PER_GPU / 0.2:.0f} clips/s/GPU; a B200 wants more")
    print()
    fv = SCOPES["freeze_vision"][1]
    for fabric, node_gbps in FABRICS.items():
        intra, inter = comms_ms(fv, node_gbps)
        print(f"  freeze_vision all-reduce on {fabric:>11s}: NVLink {intra:5.0f} ms + "
              f"NIC {inter:6.0f} ms = {intra + inter:6.0f} ms "
              f"(raw)")
    print()
    print("Times use the optimistic end of each efficiency band; A100/B200 efficiency and the")
    print("blob latencies are assumptions, the A6000 step and the 677 KiB/clip are measurements.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=HOURS)
    main(ap.parse_args().hours)
