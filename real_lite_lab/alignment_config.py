from __future__ import annotations

import re

from .constants import POLICY_JOINT_NAMES

DEFAULT_ACTION_SCALE = 0.25
DEFAULT_CLIP_ACTIONS = 100.0
DEFAULT_CLIP_OBSERVATIONS = 100.0
DEFAULT_OBS_SCALES = {
    "ang_vel": 1.0,
    "projected_gravity": 1.0,
    "commands": 1.0,
    "joint_pos": 1.0,
    "joint_vel": 1.0,
    "actions": 1.0,
}
SOFT_JOINT_POS_LIMIT_FACTOR = 0.9

TASK_COMMAND_RANGES = {
    "walk_real_lite": {
        "lin_vel_x": (-0.6, 1.0),
        "lin_vel_y": (-0.5, 0.5),
        "ang_vel_z": (-1.57, 1.57),
    },
    "walk_forward_real_lite": {
        "lin_vel_x": (0.10, 0.35),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (0.0, 0.0),
    },
    "walk_gmr_crawl_real_lite": {
        "lin_vel_x": (0.08, 0.22),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (0.0, 0.0),
    },
    "run_real_lite": {
        "lin_vel_x": (-0.6, 1.0),
        "lin_vel_y": (-0.5, 0.5),
        "ang_vel_z": (-1.57, 1.57),
    },
    "upper_body_real_lite": {
        "lin_vel_x": (0.0, 0.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (0.0, 0.0),
    },
    "stand_real_lite": {
        "lin_vel_x": (0.0, 0.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (0.0, 0.0),
    },
}

JOINT_STIFFNESS_PATTERNS = {
    "hip_roll_.*_joint": 700.0,
    "hip_yaw_.*_joint": 500.0,
    "hip_pitch_.*_joint": 700.0,
    "knee_pitch_.*_joint": 700.0,
    "ankle_pitch_.*_joint": 30.0,
    "ankle_roll_.*_joint": 16.8,
    "shoulder_pitch_.*_joint": 60.0,
    "shoulder_roll_.*_joint": 20.0,
    "shoulder_yaw_.*_joint": 10.0,
    "elbow_.*_joint": 10.0,
}
JOINT_DAMPING_PATTERNS = {
    "hip_roll_.*_joint": 10.0,
    "hip_yaw_.*_joint": 5.0,
    "hip_pitch_.*_joint": 10.0,
    "knee_pitch_.*_joint": 10.0,
    "ankle_pitch_.*_joint": 2.5,
    "ankle_roll_.*_joint": 1.4,
    "shoulder_pitch_.*_joint": 3.0,
    "shoulder_roll_.*_joint": 1.5,
    "shoulder_yaw_.*_joint": 1.0,
    "elbow_.*_joint": 1.0,
}
JOINT_EFFORT_LIMIT_PATTERNS = {
    "hip_roll_.*_joint": 150.0,
    "hip_yaw_.*_joint": 90.0,
    "hip_pitch_.*_joint": 150.0,
    "knee_pitch_.*_joint": 150.0,
    "ankle_pitch_.*_joint": 60.0,
    "ankle_roll_.*_joint": 30.0,
    "shoulder_pitch_.*_joint": 36.0,
    "shoulder_roll_.*_joint": 36.0,
    "shoulder_yaw_.*_joint": 36.0,
    "elbow_.*_joint": 36.0,
}
JOINT_VELOCITY_LIMIT_PATTERNS = {
    "hip_roll_.*_joint": 12.0,
    "hip_yaw_.*_joint": 14.0,
    "hip_pitch_.*_joint": 12.0,
    "knee_pitch_.*_joint": 12.0,
    "ankle_pitch_.*_joint": 7.8,
    "ankle_roll_.*_joint": 7.8,
    "shoulder_pitch_.*_joint": 7.8,
    "shoulder_roll_.*_joint": 7.8,
    "shoulder_yaw_.*_joint": 7.8,
    "elbow_.*_joint": 7.8,
}


def resolve_joint_parameter_map(pattern_values: dict[str, float]) -> dict[str, float]:
    resolved: dict[str, float] = {}
    for joint_name in POLICY_JOINT_NAMES:
        for pattern, value in pattern_values.items():
            if re.fullmatch(pattern, joint_name):
                resolved[joint_name] = float(value)
                break
        else:
            raise ValueError(f"No alignment value configured for joint '{joint_name}'.")
    return resolved


JOINT_STIFFNESS = resolve_joint_parameter_map(JOINT_STIFFNESS_PATTERNS)
JOINT_DAMPING = resolve_joint_parameter_map(JOINT_DAMPING_PATTERNS)
JOINT_EFFORT_LIMIT = resolve_joint_parameter_map(JOINT_EFFORT_LIMIT_PATTERNS)
JOINT_VELOCITY_LIMIT = resolve_joint_parameter_map(JOINT_VELOCITY_LIMIT_PATTERNS)
