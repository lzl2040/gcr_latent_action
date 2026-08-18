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
# Maximum number of tactile *image* views kept per sample. 6 because `ftp_1_sharpa` has six
# pads (three per hand); at 4 the excess was silently dropped by list truncation, and since the
# keys are ordered left-first that removed two of the three *right*-hand pads rather than
# thinning both hands evenly.
MAX_TACTILE_VIEWS = 6


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
    "kuka": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "bridge_orig": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "jaco_play": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "taco_play": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "roboturk": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "berkeley_autolab_ur5": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "stanford_hydra_dataset_converted_externally_to_rlds": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "language_table": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "cmu_stretch": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "bc_z": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "fmb_dataset": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "dobbe": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "agibot_alpha": {
        "action": [_seg("action", 0, 20, 0)],
        "state": [_seg("observation.state", 0, 20, 0)],
    },
    "robomind_franka_dual_arm": {
        "action": [_seg("action", 0, 20, 0)],
        "state": [_seg("observation.state", 0, 20, 0)],
    },
    "robomind_franka_3rgb": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "robomind_franka_1rgb": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "robomind_agilex_3rgb": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "robomind_ur_1rgb": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    "interna1_dual_arm_0": {
        "action": [_seg("action", 0, 20, 0)],
        "state": [_seg("observation.state", 0, 20, 0)],
    },
    "interna1_dual_arm_1": {
        "action": [_seg("action", 0, 20, 0)],
        "state": [_seg("observation.state", 0, 20, 0)],
    },
    "interna1_dual_arm_2": {
        "action": [_seg("action", 0, 20, 0)],
        "state": [_seg("observation.state", 0, 20, 0)],
    },
    "interna1_dual_arm_3": {
        "action": [_seg("action", 0, 20, 0)],
        "state": [_seg("observation.state", 0, 20, 0)],
    },
    "interna1_dual_arm_4": {
        "action": [_seg("action", 0, 20, 0)],
        "state": [_seg("observation.state", 0, 20, 0)],
    },
    "interna1_single_arm": {
        "action": [_seg("action", 0, 10, 0)],
        "state": [_seg("observation.state", 0, 10, 0)],
    },
    # Micro Data
    ## --- YAM / xdof bimanual stations: eef ort6d (20) + joints (14 = 7 + 7) ---
    ## XMI_MERGED do not have joint positions, so we only use the eef ort6d (20) part.
    "ms_data_xdof_1": {
        "action": [
            _seg("action.ee_ort6d_pos", 0, 20, 0),
        ],
        "state": [
            _seg("observations.ee_ort6d_pos", 0, 20, 0),
        ],
    },
    ## YAM_Box_MERGED, same as YAM_Station_MERGED
    "ms_data_xdof_2": {
        "action": [
            _seg("action.ee_ort6d_pos", 0, 20, 0),
            _seg("action.joint_position", 0, 7, 20), # a little error, because final pos is gripper
            _seg("action.joint_position", 7, 14, 28),
        ],
        "state": [
            _seg("observations.ee_ort6d_pos", 0, 20, 0),
            _seg("observations.joint_position", 0, 7, 20),
            _seg("observations.joint_position", 7, 14, 28),
        ],
    },
    ## YAM_Station_MERGED
    # "ms_data_xdof_3": {
    #     "action": [
    #         _seg("action.ee_ort6d_pos", 0, 20, 0),
    #         _seg("action.joint_position", 0, 7, 20),
    #         _seg("action.joint_position", 7, 14, 28),
    #     ],
    #     "state": [
    #         _seg("observations.ee_ort6d_pos", 0, 20, 0),
    #         _seg("observations.joint_position", 0, 7, 20),
    #         _seg("observations.joint_position", 7, 14, 28),
    #     ],
    # },
    ## FR3_Duo_MERGED has 7 joints per arm + gripper
    "ms_data_xdof_4": {
        "action": [
            _seg("action.ee_ort6d_pos", 0, 20, 0),
            _seg("action.joint_position", 0, 8, 20),
            _seg("action.joint_position", 8, 16, 28),
        ],
        "state": [
            _seg("observations.ee_ort6d_pos", 0, 20, 0),
            _seg("observations.joint_position", 0, 8, 20),
            _seg("observations.joint_position", 8, 16, 28),
        ],
    },
    ## UR_AI_Trainer_MERGED is the same as YAM_Station_MERGED, but with 7 joints per arm + gripper
    # "ms_data_xdof_5": {
    #     "action": [
    #         _seg("action.ee_ort6d_pos", 0, 20, 0),
    #         _seg("action.joint_position", 0, 7, 20),
    #         _seg("action.joint_position", 7, 14, 28),
    #     ],
    #     "state": [
    #         _seg("observations.ee_ort6d_pos", 0, 20, 0),
    #         _seg("observations.joint_position", 0, 7, 20),
    #         _seg("observations.joint_position", 7, 14, 28),
    #     ],
    # },
    ## Trossen_Stationary_AI_480x640_padded_MERGED, whose joint positions are not like joint_0, joint_1
    "ms_data_scale": {
        "action": [
            _seg("action.ee_ort6d_pos", 0, 20, 0),
            # _seg("observations.joint_position", 0, 7, 20), 
            # _seg("observations.joint_position", 7, 14, 28),
        ],
        "state": [
            _seg("observations.ee_ort6d_pos", 0, 20, 0),
            # _seg("observations.joint_position", 0, 7, 20),
            # _seg("observations.joint_position", 7, 14, 28),
        ],
    },
    # --- FTP-1 tactile datasets ---------------------------------------------
    # "ftp_1_RH20TCfg5Franka": {
    #     # single Franka arm, eef pose only, 6-axis gripper torque as tactile signal
    #     "action": [_seg("action.eef_pose", 0, 10, 0)],
    #     "state": [_seg("observation.state.eef_pose", 0, 10, 0)],
    #     "tactile_signal": ["observation.state.tactile_right_grippertorque"],
    # },
    "ftp_1_FreeTacMan": {
        # 单右臂，仅 eef pose；触觉是 2 路 FreeTacMan 指尖触觉相机
        # 注意：该数据集的 info.json 没有 tactile_info 字段，触觉键取自 features
        "action": [_seg("action.eef_pose", 0, 10, 0)],
        "state": [_seg("observation.state.eef_pose", 0, 10, 0)],
        "tactile_image": [
            "observation.images.tactile_right_0",
            "observation.images.tactile_right_1",
        ],
    },
    "ftp_1_MotionTrans": {
        # 单右臂 eef pose + InspireHand 6 自由度手；触觉是力矩数值，非图像
        "action": [
            _seg("action.eef_pose", 0, 10, 0),
            # *_faas_segs("action.hand_joints", MOTIONTRANS_HAND_FAAS, 0, HAND_RIGHT), # TO DO: not use hand joints
        ],
        "state": [
            _seg("observation.state.eef_pose", 0, 10, 0),
            # *_faas_segs("observation.state.hand_joints", MOTIONTRANS_HAND_FAAS, 0, HAND_RIGHT),
        ],
        "tactile_signal": [
            "observation.state.tactile_right_fingertorque",   # (4, 1) FAAS 区域 20/19/18/17
            "observation.state.tactile_right_thumbtorque",    # (1, 2) FAAS 区域 16 两通道
        ],
    },
    
    "ftp_1_RDP": {
        # 单 *左* 臂 eef pose；MCTac 触觉相机 + Flexiv 夹爪力
        "action": [_seg("action.eef_pose", 0, 10, 0)],
        "state": [_seg("observation.state.eef_pose", 0, 10, 0)],
        "tactile_image": ["observation.images.tactile_left_0"],
        "tactile_signal": [
            "observation.state.tactile_left_gripperforce_flexivgripper",   # (1, 1)
        ],
    },
    
    "ftp_1_RDP_Bimanual": {
        # 双臂 eef pose (20)；每侧 1 路 MCTac 触觉相机 + 1 路夹爪力
        "action": [
            _seg("action.eef_pose", 0, 10, 0),
            _seg("action.eef_pose", 10, 20, 10),
        ],
        "state": [
            _seg("observation.state.eef_pose", 0, 10, 0),
            _seg("observation.state.eef_pose", 10, 20, 10),
        ],
        "tactile_image": [
            "observation.images.tactile_left_0",
            "observation.images.tactile_right_0",
        ],
        "tactile_signal": [
            "observation.state.tactile_left_gripperforce_flexivgripper",   # (1, 1)
            "observation.state.tactile_right_gripperforce_flexivgripper",  # (1, 1)
        ],
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
            "observation.images.tactile_left_2",
            "observation.images.tactile_right_0",
            "observation.images.tactile_right_1",
            "observation.images.tactile_right_2",
        ],
    },
    
    "ftp_1_sharpa": {
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
            "observation.images.tactile_left_2",
            "observation.images.tactile_right_0",
            "observation.images.tactile_right_1",
            "observation.images.tactile_right_2",
        ],
    },
    
    "ftp_1_RH20TCfg5Franka": {
        # 单右臂 Franka，仅 eef pose；触觉为 Franka 关节力矩换算的 6 轴夹爪力/力矩
        "action": [_seg("action.eef_pose", 0, 10, 0)],
        "state": [_seg("observation.state.eef_pose", 0, 10, 0)],
        "tactile_signal": [
            "observation.state.tactile_right_grippertorque",   # (1, 6) FAAS 区域 15
        ],
    },
    "ftp_1_RH20TCfg6ATIAxia": {
        # 单右臂，仅 eef pose；ATI Axia80-M20 六维力/力矩传感器
        "action": [_seg("action.eef_pose", 0, 10, 0)],
        "state": [_seg("observation.state.eef_pose", 0, 10, 0)],
        "tactile_signal": [
            "observation.state.tactile_right_grippertorque",   # (1, 6) FAAS 区域 15
        ],
    },
    "ftp_1_RH20TCfg7Tactile": {
        # 单右臂，仅 eef pose；ATI Axia80-M20 六维力/力矩传感器
        "action": [_seg("action.eef_pose", 0, 10, 0)],
        "state": [_seg("observation.state.eef_pose", 0, 10, 0)],
        "tactile_signal": [
            "observation.state.tactile_right_grippertorque",   # (1, 6) FAAS 区域 15
        ],
    },
    "ftp_1_Unit": {
        # 单右臂，*仅关节空间*（无 eef pose）；1 路 GelSight Mini 触觉相机
        "action": [_seg("action.arm_joint", 0, 8, 20)],
        "state": [_seg("observation.state.arm_joint", 0, 8, 20)],
        "tactile_image": ["observation.images.tactile_right_0"],
    },
    "ftp_1_VLA_touch": {
        # 单右臂，同时提供 eef pose 和关节；1 路 GelSight Mini 触觉相机
        "action": [
            _seg("action.eef_pose", 0, 10, 0),
            _seg("action.arm_joint", 0, 8, 20),
        ],
        "state": [
            _seg("observation.state.eef_pose", 0, 10, 0),
            _seg("observation.state.arm_joint", 0, 8, 20),
        ],
        "tactile_image": ["observation.images.tactile_right_0"],
    },
    "ftp_1_ViTaMIn": {
        # 单右臂，仅 eef pose；夹爪两指各 1 路 ViTaMIn 触觉相机
        "action": [_seg("action.eef_pose", 0, 10, 0)],
        "state": [_seg("observation.state.eef_pose", 0, 10, 0)],
        "tactile_image": [
            "observation.images.tactile_right_0",
            "observation.images.tactile_right_1",
        ],
    },
    "ftp_1_VisuoTactile_D-WHEEL": {
        # 双臂，*仅关节空间*（无 eef pose）；每侧 2 路 OpenLoong VTouch 触觉相机
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
    "ftp_1_VisuoTactile_QINGLOONG": {
        # 双臂，*仅关节空间*（无 eef pose）；每侧 2 路 OpenLoong VTouch 触觉相机
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
    "ftp_1_exUMI": {
        # 单右臂手持式采集，仅 eef pose；1 路 exUMI 触觉相机 (460x680)
        "action": [_seg("action.eef_pose", 0, 10, 0)],
        "state": [_seg("observation.state.eef_pose", 0, 10, 0)],
        "tactile_image": ["observation.images.tactile_right_0"],
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
    # open_neo_data
    "open_neo_aloha": {
        "action": [
            _seg("action.eef_pose", 0, 20, 0),
            _seg("action", 0, 7, 20),
            _seg("action", 7, 14, 28),
        ],
        "state": [
            _seg("observation.eef_pose", 0, 20, 0),
            _seg("observation.state", 0, 7, 20),
            _seg("observation.state", 7, 14, 28),
        ],
        "tactile_image": [
            "observation.images.left_wrist_left_tactile",
            "observation.images.left_wrist_right_tactile",
            "observation.images.right_wrist_left_tactile",
            "observation.images.right_wrist_right_tactile"
        ],
    },
    "open_neo_arx5_single": {
        "action": [
            _seg("action.eef_pose", 0, 10, 0),
            _seg("action", 0, 7, 20),
        ],
        "state": [
            _seg("observation.eef_pose", 0, 10, 0),
            _seg("observation.state", 0, 7, 20),
        ],
        "tactile_image": [
            "observation.images.left_wrist_left_tactile",
            "observation.images.left_wrist_right_tactile"
        ],
    }
}

# ms_data_xdof_5, ms_data_xdof_2,  ms_data_xdof_3
# ``ms_data_xdof_{2,3,4}`` share the layout of ``ms_data_xdof_1``.
for _idx in (3, 5):
    PHYSICAL_SPECS[f"ms_data_xdof_{_idx}"] = PHYSICAL_SPECS["ms_data_xdof_2"]


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
