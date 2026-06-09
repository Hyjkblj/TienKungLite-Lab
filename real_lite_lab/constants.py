from __future__ import annotations

import json
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = PACKAGE_DIR.parent
WORKSPACE_DIR = PIPELINE_DIR.parent

ASSETS_DIR = PACKAGE_DIR / "assets"
DATASETS_DIR = PACKAGE_DIR / "datasets"
LOGS_DIR = PIPELINE_DIR / "logs"
MJCF_DIR = PIPELINE_DIR / "mjcf"

LEFT_LEG_JOINT_NAMES = [
    "hip_roll_l_joint",
    "hip_pitch_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
]
RIGHT_LEG_JOINT_NAMES = [
    "hip_roll_r_joint",
    "hip_pitch_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
]
LEFT_ARM_JOINT_NAMES = [
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_l_joint",
]
RIGHT_ARM_JOINT_NAMES = [
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_r_joint",
]
ANKLE_JOINT_NAMES = [
    "ankle_pitch_l_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_l_joint",
    "ankle_roll_r_joint",
]
FEET_LINK_NAMES = ["ankle_roll_l_link", "ankle_roll_r_link"]
ELBOW_LINK_NAMES = ["elbow_l_link", "elbow_r_link"]

POLICY_JOINT_NAMES = (
    LEFT_LEG_JOINT_NAMES + RIGHT_LEG_JOINT_NAMES + LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
)
HIP_PITCH_JOINT_NAMES = tuple(name for name in POLICY_JOINT_NAMES if "hip_pitch" in name)
KNEE_PITCH_JOINT_NAMES = tuple(name for name in POLICY_JOINT_NAMES if "knee_pitch" in name)
ANKLE_PITCH_JOINT_NAMES = tuple(name for name in POLICY_JOINT_NAMES if "ankle_pitch" in name)
ANKLE_ROLL_JOINT_NAMES = tuple(name for name in POLICY_JOINT_NAMES if "ankle_roll" in name)
POLICY_JOINT_COUNT = len(POLICY_JOINT_NAMES)

DEFAULT_JOINT_POS = {
    "hip_roll_l_joint": 0.0,
    "hip_yaw_l_joint": 0.0,
    "hip_pitch_l_joint": -0.5,
    "knee_pitch_l_joint": 1.0,
    "ankle_pitch_l_joint": -0.5,
    "ankle_roll_l_joint": 0.0,
    "hip_roll_r_joint": 0.0,
    "hip_yaw_r_joint": 0.0,
    "hip_pitch_r_joint": -0.5,
    "knee_pitch_r_joint": 1.0,
    "ankle_pitch_r_joint": -0.5,
    "ankle_roll_r_joint": 0.0,
    "shoulder_pitch_l_joint": 0.0,
    "shoulder_roll_l_joint": 0.1,
    "shoulder_yaw_l_joint": 0.0,
    "elbow_l_joint": -0.3,
    "shoulder_pitch_r_joint": 0.0,
    "shoulder_roll_r_joint": -0.1,
    "shoulder_yaw_r_joint": 0.0,
    "elbow_r_joint": -0.3,
}
DEFAULT_DOF_POS = [DEFAULT_JOINT_POS[name] for name in POLICY_JOINT_NAMES]

ROOT_POS_DIM = 3
ROOT_EULER_DIM = 3
ROOT_LIN_VEL_DIM = 3
ROOT_ANG_VEL_DIM = 3
END_EFFECTOR_POS_DIM = 12

OBS_ANG_VEL_DIM = 3
OBS_PROJECTED_GRAVITY_DIM = 3
OBS_COMMAND_DIM = 3
OBS_GAIT_PHASE_DIM = 2
OBS_GAIT_RATIO_DIM = 2
OBS_PER_STEP_DIM = (
    OBS_ANG_VEL_DIM
    + OBS_PROJECTED_GRAVITY_DIM
    + OBS_COMMAND_DIM
    + POLICY_JOINT_COUNT
    + POLICY_JOINT_COUNT
    + POLICY_JOINT_COUNT
    + OBS_GAIT_PHASE_DIM
    + OBS_GAIT_PHASE_DIM
    + OBS_GAIT_RATIO_DIM
)

# Visualization frame layout (52 total):
#   root_pos(3) + root_euler(3) +
#   left_leg_pos(6) + right_leg_pos(6) + left_arm_pos(4) + right_arm_pos(4) +
#   root_lin_vel(3) + root_ang_vel(3) +
#   left_leg_vel(6) + right_leg_vel(6) + left_arm_vel(4) + right_arm_vel(4)
VIS_LEFT_LEG_POS_SLICE = slice(ROOT_POS_DIM + ROOT_EULER_DIM, ROOT_POS_DIM + ROOT_EULER_DIM + len(LEFT_LEG_JOINT_NAMES))
VIS_RIGHT_LEG_POS_SLICE = slice(VIS_LEFT_LEG_POS_SLICE.stop, VIS_LEFT_LEG_POS_SLICE.stop + len(RIGHT_LEG_JOINT_NAMES))
VIS_LEFT_ARM_POS_SLICE = slice(VIS_RIGHT_LEG_POS_SLICE.stop, VIS_RIGHT_LEG_POS_SLICE.stop + len(LEFT_ARM_JOINT_NAMES))
VIS_RIGHT_ARM_POS_SLICE = slice(VIS_LEFT_ARM_POS_SLICE.stop, VIS_LEFT_ARM_POS_SLICE.stop + len(RIGHT_ARM_JOINT_NAMES))
VIS_ROOT_LIN_VEL_SLICE = slice(VIS_RIGHT_ARM_POS_SLICE.stop, VIS_RIGHT_ARM_POS_SLICE.stop + ROOT_LIN_VEL_DIM)
VIS_ROOT_ANG_VEL_SLICE = slice(VIS_ROOT_LIN_VEL_SLICE.stop, VIS_ROOT_LIN_VEL_SLICE.stop + ROOT_ANG_VEL_DIM)
VIS_LEFT_LEG_VEL_SLICE = slice(VIS_ROOT_ANG_VEL_SLICE.stop, VIS_ROOT_ANG_VEL_SLICE.stop + len(LEFT_LEG_JOINT_NAMES))
VIS_RIGHT_LEG_VEL_SLICE = slice(VIS_LEFT_LEG_VEL_SLICE.stop, VIS_LEFT_LEG_VEL_SLICE.stop + len(RIGHT_LEG_JOINT_NAMES))
VIS_LEFT_ARM_VEL_SLICE = slice(VIS_RIGHT_LEG_VEL_SLICE.stop, VIS_RIGHT_LEG_VEL_SLICE.stop + len(LEFT_ARM_JOINT_NAMES))
VIS_RIGHT_ARM_VEL_SLICE = slice(VIS_LEFT_ARM_VEL_SLICE.stop, VIS_LEFT_ARM_VEL_SLICE.stop + len(RIGHT_ARM_JOINT_NAMES))
VISUALIZATION_FRAME_DIM = VIS_RIGHT_ARM_VEL_SLICE.stop
# AMP expert frame layout (52 total): joint_pos(20) + joint_vel(20) + end_effector_pos(12)
AMP_EXPERT_FRAME_DIM = POLICY_JOINT_COUNT * 2 + END_EFFECTOR_POS_DIM

if len(DEFAULT_JOINT_POS) != POLICY_JOINT_COUNT:
    raise ValueError("DEFAULT_JOINT_POS and POLICY_JOINT_NAMES must describe the same joints.")


def _infer_motion_cycle_seconds(motion_file: Path, default: float) -> float:
    """Infer one loop duration from a visualization motion file when available."""
    if not motion_file.is_file():
        return default

    try:
        motion_data = json.loads(motion_file.read_text(encoding="utf-8"))
        frame_duration = float(motion_data["FrameDuration"])
        num_frames = len(motion_data["Frames"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return default

    if frame_duration <= 0.0 or num_frames <= 0:
        return default
    return frame_duration * num_frames

TASK_PRESETS = {
    "walk_real_lite": {
        "gait_air_ratio_l": 0.38,
        "gait_air_ratio_r": 0.38,
        "gait_phase_offset_l": 0.38,
        "gait_phase_offset_r": 0.88,
        "gait_cycle": 0.85,
        "amp_motion_file": DATASETS_DIR / "motion_amp_expert" / "walk.txt",
        "display_motion_file": DATASETS_DIR / "motion_visualization" / "walk.txt",
    },
    "run_real_lite": {
        "gait_air_ratio_l": 0.6,
        "gait_air_ratio_r": 0.6,
        "gait_phase_offset_l": 0.6,
        "gait_phase_offset_r": 0.1,
        "gait_cycle": 0.5,
        "amp_motion_file": DATASETS_DIR / "motion_amp_expert" / "run.txt",
        "display_motion_file": DATASETS_DIR / "motion_visualization" / "run.txt",
    },
    "upper_body_real_lite": {
        "gait_air_ratio_l": 0.5,
        "gait_air_ratio_r": 0.5,
        "gait_phase_offset_l": 0.0,
        "gait_phase_offset_r": 0.0,
        # Keep the phase signal aligned with the actual upper-body clip once the dataset exists.
        "gait_cycle": _infer_motion_cycle_seconds(DATASETS_DIR / "motion_visualization" / "upper_body.txt", 1.0),
        "amp_motion_file": DATASETS_DIR / "motion_amp_expert" / "upper_body.txt",
        "display_motion_file": DATASETS_DIR / "motion_visualization" / "upper_body.txt",
    },
    "stand_real_lite": {
        "gait_air_ratio_l": 0.0,
        "gait_air_ratio_r": 0.0,
        "gait_phase_offset_l": 0.0,
        "gait_phase_offset_r": 0.0,
        "gait_cycle": 1.0,
        "display_motion_file": DATASETS_DIR / "motion_visualization" / "upper_body.txt",
    },
}

TASK_NAMES = tuple(TASK_PRESETS.keys())
