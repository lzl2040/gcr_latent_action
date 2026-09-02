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

Run:  python scripts/extrapolate_cluster.py
"""

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
LOADER_MS_PER_SAMPLE = 37.2  # single worker, measured by check_multiframe_dataset.py
NUM_WORKERS = 12

MIX = {"action": 0.5, "i2v": 0.2, "v2v": 0.15, "t2v": 0.1, "t2i": 0.05}  # world_model.STAGE3_MIX

# bf16 bytes per trainable parameter that must live on the card, by parallelism strategy.
# ddp: params(all) + grads + 2 Adam moments.  zero2: grads and moments sharded over GPUS.
BYTES_PARAM = 2

# --- dataset --------------------------------------------------------------------------------
HOURS = 10_000
FPS = 20
CHUNK_SECONDS = 1.6
PIXEL_FRAMES_PER_CLIP = 9
LATENT_BYTES_PER_CLIP = 48 * 3 * 16 * 16 * 2  # C x T x H x W, bf16

# --- hardware ---------------------------------------------------------------------------------
# (usable dense bf16 TFLOP/s, (efficiency_low, efficiency_high), inter-node GB/s, usable GiB)
# Efficiency is relative to the A6000's measured behaviour, which is 1.0 by construction.
# B200's band is wide and starts low on purpose: its headline peak is ~29x the A6000's, but
# our gen expert reaches only 67% of usable peak even on the A6000 (52.1 vs the und expert's
# 78.5 TFLOP/s), because its matrices are smaller -- and that gap widens as tensor cores grow.
# A100-40GB and A100-80GB have identical compute; only capacity differs, and that is exactly
# what decides which scope and batch are reachable.
DEVICES = {
    "RTX A6000": (77.4, (1.00, 1.00), 12.0, 47.6),
    "A100-40GB": (312.0, (0.75, 0.90), 20.0, 39.5),
    "A100-80GB": (312.0, (0.75, 0.90), 20.0, 79.2),
    "B200": (2250.0, (0.30, 0.55), 50.0, 180.0),
}
GPUS = 16
OVERLAP = 0.7  # fraction of the all-reduce hidden behind the backward pass
TOTAL_PARAMS = 5.588e9
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


def main() -> None:
    clips = HOURS * 3600 / CHUNK_SECONDS
    frames = HOURS * 3600 * FPS
    loader_ceiling = NUM_WORKERS / (LOADER_MS_PER_SAMPLE / 1000.0)
    frames_seen = clips * PIXEL_FRAMES_PER_CLIP

    print(f"10,000 h @ {FPS} fps          : {frames:.3g} pixel frames on disk")
    print(f"1 'epoch'                   : {clips:.3g} non-overlapping {CHUNK_SECONDS}s windows "
          f"(= {frames:.3g} / {int(CHUNK_SECONDS * FPS)})")
    print(f"                              but each window READS only {PIXEL_FRAMES_PER_CLIP} of "
          f"the {int(CHUNK_SECONDS * FPS)} frames it spans,")
    print(f"                              so one epoch touches {frames_seen:.3g} frames = "
          f"{frames_seen / frames * 100:.0f}% of the corpus.")
    print(f"                              Seeing every frame needs ~"
          f"{frames / frames_seen:.1f} epochs.")
    print(f"cached-latent footprint     : {clips * LATENT_BYTES_PER_CLIP / 1e12:.2f} TB")
    print(f"loader ceiling              : {loader_ceiling:.0f} clips/s/GPU")
    print(f"online VAE                  : {VAE_MS:.0f} ms per {BATCH_REF} clips\n")

    for scope, (model_ms, trainable, peaks) in SCOPES.items():
        w32 = sum(MIX[t] * model_ms[t] for t in MIX)
        fixed_ms, per_clip_ms = marginal_fit(scope)
        grad_gb = trainable * 2 / 1e9
        ring = 2 * (GPUS - 1) / GPUS * grad_gb
        meas = "  ".join(
            f"b{b}={'OOM' if g is None else f'{g:.1f}G'}" for b, g in sorted(peaks.items())
        )

        print(f"##### scope = {scope}  ({trainable / 1e9:.2f}B trainable) #####")
        print(f"A6000 step (batch {BATCH_REF})     : {w32:.0f} ms "
              f"= {fixed_ms:.0f} ms fixed + {per_clip_ms:.1f} ms/clip")
        print(f"measured peak memory        : {meas}   (47.6 GiB card)")
        print(f"resident state (no acts)    : DDP {state_gib(trainable, 1):.1f} GiB   "
              f"ZeRO-2 x{GPUS} {state_gib(trainable, GPUS):.1f} GiB")
        print(f"gradient all-reduce         : {grad_gb:.2f} GB -> {ring:.2f} GB moved per GPU\n")

        print(f"{'device':>11s} {'batch/GPU':>10s} {'mem':>18s} {'step':>9s} "
              f"{'clips/s':>9s} {'1 epoch (cached)':>20s}")
        for dev, (peak, (eff_lo, eff_hi), bw, mem) in DEVICES.items():
            for batch in (32, 64, 128):
                need_ddp = state_gib(trainable, 1) + ACT_GIB_PER_CLIP * batch
                need_z2 = state_gib(trainable, GPUS) + ACT_GIB_PER_CLIP * batch
                if need_ddp <= mem:
                    tag, need = "DDP", need_ddp
                elif need_z2 <= mem:
                    tag, need = "ZeRO-2", need_z2
                else:
                    print(f"{dev:>11s} {batch:10d} {'>' + f'{mem:.0f}G even w/ ZeRO-2':>18s}"
                          f" {'-':>9s} {'-':>9s} {'infeasible':>20s}")
                    continue
                row = []
                for eff in (eff_hi, eff_lo):
                    ratio = (peak * eff) / DEVICES["RTX A6000"][0]
                    compute_ms = (fixed_ms + per_clip_ms * batch) / ratio
                    comms_ms = ring / bw * 1000.0 * (1 - OVERLAP)
                    step_ms = compute_ms + comms_ms
                    row.append((step_ms, batch / (step_ms / 1000.0) * GPUS))
                (s_hi, t_hi), (s_lo, t_lo) = row
                span = "" if eff_lo == eff_hi else f" - {fmt_time(clips / t_lo)}"
                print(f"{dev:>11s} {batch:10d} {f'{need:.1f}G {tag}':>18s} {s_hi:7.0f}ms "
                      f"{t_hi:9.0f} {fmt_time(clips / t_hi) + span:>20s}")
        print()

    fv, go = SCOPES["freeze_vision"][1], SCOPES["gen_only"][1]
    print(f"Two things decide whether freeze_vision is reachable at all:")
    print(f"  * Its resident state under DDP is {state_gib(fv, 1):.1f} GiB -- params, grads and")
    print(f"    two bf16 Adam moments -- which does not fit a 40 GB A100 at ANY batch size.")
    print(f"    ZeRO-2 across {GPUS} GPUs cuts that to {state_gib(fv, GPUS):.1f} GiB and it fits easily.")
    print(f"  * Its all-reduce is {fv / go:.1f}x gen_only's, so the scope costs 1.27x compute on")
    print("    one GPU but more than that on a cluster, and the gap widens as GPUs get faster.")
    print()
    print("'1 epoch' means one pass over the 1.6 s windows, NOT over every frame -- see the")
    print("frame-coverage line at the top. Ranges show the efficiency band, fastest first;")
    print("A100/B200 efficiency is an assumption, the A6000 anchor is a measurement.")


if __name__ == "__main__":
    main()
