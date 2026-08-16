"""Canonical physical (robot-side) space definition for cross-embodiment contrastive learning.

Different datasets expose heterogeneous proprioceptive / action layouts:

* single-arm end-effector pose  ``xyz(3) + ort6d(6) + gripper(1)``  -> 10 dims
* dual-arm end-effector pose                                        -> 20 dims
* arm joint positions ``joint_0..joint_6 + gripper``                -> 8 dims per arm
* dexterous-hand joints, tactile signals, tactile images, ...

Instead of blindly right-padding every vector to a fixed width (which puts the
``xyz`` of a joint-only dataset on top of the ``xyz`` of an end-effector dataset),
we map every dataset into a **slotted canonical vector** where a given index always
carries the same physical meaning, together with a **validity mask** telling the
model which slots actually exist for this sample.

Canonical layout (``CANON_DIM`` = 40)::

    [ 0: 3]  arm-0 eef translation      (x, y, z)
    [ 3: 9]  arm-0 eef rotation 6d
    [ 9:10]  arm-0 gripper
    [10:13]  arm-1 eef translation      (x, y, z)
    [13:19]  arm-1 eef rotation 6d
    [19:20]  arm-1 gripper
    [20:27]  arm-0 joint positions      (joint_0 .. joint_6)
    [27:28]  arm-0 joint-space gripper
    [28:35]  arm-1 joint positions      (joint_0 .. joint_6)
    [35:36]  arm-1 joint-space gripper
    [36:40]  reserved (always masked out for now)

A dataset that only ships joint positions therefore leaves ``[0:20]`` masked out,
and a dataset that only ships end-effector poses leaves ``[20:36]`` masked out.
Both live in the *same* space, so a single encoder can consume all of them and the
mask prevents the model from reading zeros as if they were real measurements.
"""

from __future__ import annotations

CANON_DIM = 40

# name -> (start, end) half-open canonical slice
CANON_SLOTS: dict[str, tuple[int, int]] = {
    "eef0": (0, 10),
    "eef0_xyz": (0, 3),
    "eef0_rot6d": (3, 9),
    "eef0_gripper": (9, 10),
    "eef1": (10, 20),
    "eef1_xyz": (10, 13),
    "eef1_rot6d": (13, 19),
    "eef1_gripper": (19, 20),
    "joint0": (20, 28),
    "joint1": (28, 36),
    "reserved": (36, 40),
}

# Translation dims: the only ones that carry an unbounded metric scale.
CANON_XYZ_INDICES: list[int] = [0, 1, 2, 10, 11, 12]

# Maximum width of a flattened tactile *signal* vector (forces / torques / taxels).
MAX_TACTILE_SIGNAL_DIM = 32
# Maximum number of tactile *image* views kept per sample.
MAX_TACTILE_VIEWS = 4


def _seg(src_key: str, src_from: int, src_to: int, dst_from: int) -> tuple[str, int, int, int]:
    """One ``source[src_from:src_to] -> canonical[dst_from:...]`` copy instruction."""
    return (src_key, src_from, src_to, dst_from)


# ---------------------------------------------------------------------------
# Per-dataset specs
# ---------------------------------------------------------------------------
# Each spec is a dict with the optional entries:
#   "action"          : list of copy instructions for the action chunk
#   "state"           : list of copy instructions for the proprioceptive state
#   "tactile_signal"  : list of low-dimensional tactile keys (concatenated)
#   "tactile_image"   : list of tactile camera keys (at most MAX_TACTILE_VIEWS used)
# Datasets absent from this table fall back to :func:`default_spec`.
PHYSICAL_SPECS: dict[str, dict] = {
    # --- Open-X style datasets already converted to xyz + ort6d + gripper -----
    "fractal20220817_data": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "taco_play": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    # --- YAM / xdof bimanual stations: eef ort6d (20) + joints (14 = 7 + 7) ---
    "ms_data_xdof_1": {
        "action": [
            _seg("action.ee_ort6d_pos", 0, 20, 0),
            _seg("action.joint_position", 0, 7, 20),
            _seg("action.joint_position", 7, 14, 28),
        ],
        "state": [
            _seg("observations.ee_ort6d_pos", 0, 20, 0),
            _seg("observations.joint_position", 0, 7, 20),
            _seg("observations.joint_position", 7, 14, 28),
        ],
    },
    # --- FTP-1 tactile datasets ---------------------------------------------
    "ftp_1_RH20TCfg5Franka": {
        # single Franka arm, eef pose only, 6-axis gripper torque as tactile signal
        "action": [_seg("action.eef_pose", 0, 10, 0)],
        "state": [_seg("observation.state.eef_pose", 0, 10, 0)],
        "tactile_signal": ["observation.state.tactile_right_grippertorque"],
    },
    "ftp_1_sharpa_split_0": {
        # bimanual: eef pose (20) + arm joints (16 = 8 + 8) + 6 tactile cameras
        "action": [
            _seg("action.eef_pose", 0, 20, 0),
            _seg("action.arm_joint", 0, 8, 20),
            _seg("action.arm_joint", 8, 16, 28),
        ],
        "state": [
            _seg("observation.state.eef_pose", 0, 20, 0),
            _seg("observation.state.arm_joint", 0, 8, 20),
            _seg("observation.state.arm_joint", 8, 16, 28),
        ],
        "tactile_image": [
            "observation.images.tactile_left_0",
            "observation.images.tactile_left_1",
            "observation.images.tactile_right_0",
            "observation.images.tactile_right_1",
        ],
    },
    "ftp_1_VisuoTactile_D-WHEEL_split_0": {
        # bimanual, *joint space only* (no eef pose at all) + 4 tactile cameras
        "action": [
            _seg("action.arm_joint", 0, 8, 20),
            _seg("action.arm_joint", 8, 16, 28),
        ],
        "state": [
            _seg("observation.state.arm_joint", 0, 8, 20),
            _seg("observation.state.arm_joint", 8, 16, 28),
        ],
        "tactile_image": [
            "observation.images.tactile_left_0",
            "observation.images.tactile_left_1",
            "observation.images.tactile_right_0",
            "observation.images.tactile_right_1",
        ],
    },
}

# ``ms_data_xdof_{2,3,4}`` share the layout of ``ms_data_xdof_1``.
for _idx in (2, 3, 4):
    PHYSICAL_SPECS[f"ms_data_xdof_{_idx}"] = PHYSICAL_SPECS["ms_data_xdof_1"]


def default_spec(action_dim: int | None, state_dim: int | None) -> dict:
    """Best-effort spec for datasets without an explicit entry.

    ``xyz + ort6d + gripper`` is the repository-wide convention, so a width of 10
    maps to arm-0 and a width of 20 maps to both arms. Anything else is dropped
    into the joint slots, which is the only honest interpretation available.
    """

    def _rule(key: str, dim: int | None) -> list[tuple[str, int, int, int]]:
        if dim is None:
            return []
        if dim >= 20:
            return [_seg(key, 0, 20, 0)]
        if dim in (14, 16):
            half = dim // 2
            return [_seg(key, 0, half, 20), _seg(key, half, dim, 28)]
        if dim >= 10:
            return [_seg(key, 0, 10, 0)]
        # Short / unknown vector: treat as a single-arm joint block.
        return [_seg(key, 0, min(dim, 8), 20)]

    return {
        "action": _rule("action", action_dim),
        "state": _rule("observation.state", state_dim),
    }


def get_spec(dataset_name: str, item_keys=None, action_dim=None, state_dim=None) -> dict:
    """Return the canonical spec for ``dataset_name``, falling back to heuristics."""
    spec = PHYSICAL_SPECS.get(dataset_name)
    if spec is not None:
        return spec
    return default_spec(action_dim, state_dim)


def resolve_source_keys(spec: dict) -> set[str]:
    """All raw dataset keys a spec reads from (used to prune video decoding etc.)."""
    keys: set[str] = set()
    for field in ("action", "state"):
        for src_key, *_ in spec.get(field, []):
            keys.add(src_key)
    keys.update(spec.get("tactile_signal", []))
    keys.update(spec.get("tactile_image", []))
    return keys


def tactile_image_keys(spec: dict) -> list[str]:
    return list(spec.get("tactile_image", []))[:MAX_TACTILE_VIEWS]


def tactile_signal_keys(spec: dict) -> list[str]:
    return list(spec.get("tactile_signal", []))
