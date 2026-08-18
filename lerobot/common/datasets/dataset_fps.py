"""True capture rates for datasets whose declared fps is wrong.

A LeRobot dataset carries exactly one fps, in `meta/info.json`, and the codebase used it for
two jobs that are only accidentally the same number:

* **Index fps** -- the time base the `timestamp` column was written on. Frame `i` is stored at
  `i / index_fps` seconds, so any `delta_timestamps` we ask the loader for *must* be built with
  this fps or it will match the wrong frame (or none, and trip the tolerance check).
* **True fps** -- the rate the data was actually captured at. This is what decides how much
  wall-clock time a window of `H` frames covers, and it is what `sample_rate` should tell the
  model.

For most datasets the two agree. For the FTP-1 sets they do not: `info.json` declares 30 while
the real capture rate is 10-15 Hz, and the stored timestamps are exactly 1/30 s apart, i.e.
synthesised from the declared rate rather than measured. Consecutive rows in several of them
repeat (D-WHEEL runs of 2.34 identical frames on average, RDP_Bimanual 2.10), which is the
signature of a lower-rate stream written onto a 30 Hz index.

Taking the declared 30 at face value makes `chunk_seconds` mean the wrong thing there -- a
"1.6 s" window would really span 3-5 s of robot motion -- and feeds a wrong `sample_rate`
embedding. Overriding it here fixes both without touching the timestamp arithmetic.
"""

from __future__ import annotations

# dataset name -> true capture rate in Hz.
DATASET_TRUE_FPS: dict[str, float] = {
    # FTP-1: info.json says 30, actual capture is 10-15 Hz.
    "ftp_1_exUMI": 15.0,
    "ftp_1_RH20TCfg5Franka": 15.0,
    "ftp_1_sharpa": 15.0,
    "ftp_1_sharpa_split_0": 15.0,
    "ftp_1_VisuoTactile_D-WHEEL": 15.0,
    "ftp_1_VisuoTactile_D-WHEEL_split_0": 15.0,
    "ftp_1_RDP_Bimanual": 15.0,
}


def resolve_true_fps(dataset_name: str, index_fps: float) -> float:
    """Capture rate for ``dataset_name``, falling back to the declared ``index_fps``."""
    return float(DATASET_TRUE_FPS.get(dataset_name, index_fps))
