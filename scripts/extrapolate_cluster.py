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

# --- measured: scripts/train_mot_world.py, batch 32, 1x RTX A6000, 8 steps/task ------------
BATCH_REF = 32
MODEL_MS = {"t2i": 801.0, "t2v": 1353.0, "i2v": 2244.0, "v2v": 2253.0, "action": 2365.0}
VAE_MS = 1276.0  # task-independent: same clip either way
LOADER_MS_PER_SAMPLE = 37.2  # single worker, measured by check_multiframe_dataset.py
NUM_WORKERS = 12

MIX = {"action": 0.5, "i2v": 0.2, "v2v": 0.15, "t2v": 0.1, "t2i": 0.05}  # world_model.STAGE3_MIX
TRAINABLE_PARAMS = 1.442e9

# --- dataset --------------------------------------------------------------------------------
HOURS = 10_000
FPS = 20
CHUNK_SECONDS = 1.6
PIXEL_FRAMES_PER_CLIP = 9
LATENT_BYTES_PER_CLIP = 48 * 3 * 16 * 16 * 2  # C x T x H x W, bf16

# --- hardware ---------------------------------------------------------------------------------
# (usable dense bf16 TFLOP/s, (efficiency_low, efficiency_high), inter-node GB/s)
# Efficiency is relative to the A6000's measured behaviour, which is 1.0 by construction.
# B200's band is wide and starts low on purpose: its headline peak is ~29x the A6000's, but
# our gen expert reaches only 67% of usable peak even on the A6000 (52.1 vs the und expert's
# 78.5 TFLOP/s), because its matrices are smaller -- and that gap widens as tensor cores grow.
DEVICES = {
    "RTX A6000": (77.4, (1.00, 1.00), 12.0),
    "A100-80GB": (312.0, (0.75, 0.90), 20.0),
    "B200": (2250.0, (0.30, 0.55), 50.0),
}
GPUS = 16
OVERLAP = 0.7  # fraction of the all-reduce hidden behind the backward pass


def fmt_time(seconds: float) -> str:
    d = seconds / 86400
    return f"{d:.1f} d" if d >= 1 else f"{seconds / 3600:.1f} h"


def main() -> None:
    weighted_ms = sum(MIX[t] * MODEL_MS[t] for t in MIX)
    clips = HOURS * 3600 / CHUNK_SECONDS
    frames = HOURS * 3600 * FPS
    loader_ceiling = NUM_WORKERS / (LOADER_MS_PER_SAMPLE / 1000.0)
    grad_gb = TRAINABLE_PARAMS * 2 / 1e9
    ring = 2 * (GPUS - 1) / GPUS * grad_gb  # ring all-reduce moves this much per GPU

    print(f"stage-3 weighted model step : {weighted_ms:.0f} ms / {BATCH_REF} clips (A6000)")
    print(f"online VAE                  : {VAE_MS:.0f} ms "
          f"({VAE_MS / (weighted_ms + VAE_MS) * 100:.0f}% of the step)")
    print(f"10,000 h @ {FPS} fps          : {frames:.3g} pixel frames "
          f"-> {clips:.3g} windows of {CHUNK_SECONDS}s")
    print(f"                              each window reads {PIXEL_FRAMES_PER_CLIP} of the "
          f"{int(CHUNK_SECONDS * FPS)} frames it spans")
    print(f"cached-latent footprint     : {clips * LATENT_BYTES_PER_CLIP / 1e12:.2f} TB")
    print(f"loader ceiling              : {loader_ceiling:.0f} clips/s/GPU")
    print(f"gradient all-reduce         : {grad_gb:.2f} GB -> {ring:.2f} GB moved per GPU\n")

    for latents in ("online VAE", "cached latents"):
        base_ms = weighted_ms + (VAE_MS if latents == "online VAE" else 0.0)
        print(f"=== {latents}, {GPUS} GPUs ===")
        print(f"{'device':>11s} {'batch/GPU':>10s} {'compute':>9s} {'comms':>9s} "
              f"{'step':>9s} {'clips/s':>10s} {'1 epoch':>16s}")
        for dev, (peak, (eff_lo, eff_hi), bw) in DEVICES.items():
            for batch in (BATCH_REF, 128):
                row = []
                for eff in (eff_hi, eff_lo):  # fast case first
                    ratio = (peak * eff) / (DEVICES["RTX A6000"][0] * 1.0)
                    compute_ms = base_ms * (batch / BATCH_REF) / ratio
                    comms_ms = ring / bw * 1000.0 * (1 - OVERLAP)
                    step_ms = compute_ms + comms_ms
                    per_gpu = batch / (step_ms / 1000.0)
                    if latents == "online VAE":
                        # Cached latents are read, not decoded, so only the online path is
                        # constrained by video decoding.
                        per_gpu = min(per_gpu, loader_ceiling)
                    row.append((compute_ms, comms_ms, step_ms, per_gpu * GPUS))
                (c_hi, m_hi, s_hi, t_hi), (c_lo, m_lo, s_lo, t_lo) = row
                span = "" if eff_lo == eff_hi else f" - {fmt_time(clips / t_lo)}"
                print(f"{dev:>11s} {batch:10d} {c_hi:7.0f}ms {m_hi:7.0f}ms {s_hi:7.0f}ms "
                      f"{t_hi:10.0f} {fmt_time(clips / t_hi) + span:>16s}")
        print()

    print("Reading the table: the 'comms' column is the part of the all-reduce not hidden by")
    print("the backward pass. It is fixed per step, so it is ~4% of a cached A6000 step and")
    print("~19% of a cached B200 step at batch 32 -- which is why the larger per-GPU batch buys")
    print("B200 ~17% and the A6000 almost nothing. Ranges show the efficiency band, fastest")
    print("figure first; the B200 band is wide because it is an assumption, not a measurement.")


if __name__ == "__main__":
    main()
