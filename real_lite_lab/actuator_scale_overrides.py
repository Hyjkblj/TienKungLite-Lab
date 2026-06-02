from __future__ import annotations

from .constants import (
    ANKLE_PITCH_JOINT_NAMES,
    ANKLE_ROLL_JOINT_NAMES,
    HIP_PITCH_JOINT_NAMES,
    KNEE_PITCH_JOINT_NAMES,
)


def _joint_name_scales(joint_names: tuple[str, ...], scale: float) -> dict[str, float]:
    if abs(float(scale) - 1.0) <= 1e-12:
        return {}
    return {joint_name: float(scale) for joint_name in joint_names}


def build_actuator_scale_overrides(
    *,
    hip_pitch_kp_scale: float = 1.0,
    hip_pitch_kv_scale: float = 1.0,
    knee_pitch_kp_scale: float = 1.0,
    knee_pitch_kv_scale: float = 1.0,
    ankle_pitch_kp_scale: float = 1.0,
    ankle_pitch_kv_scale: float = 1.0,
    ankle_roll_kp_scale: float = 1.0,
    ankle_roll_kv_scale: float = 1.0,
) -> tuple[dict[str, float], dict[str, float]]:
    joint_kp_scales = (
        _joint_name_scales(HIP_PITCH_JOINT_NAMES, hip_pitch_kp_scale)
        | _joint_name_scales(KNEE_PITCH_JOINT_NAMES, knee_pitch_kp_scale)
        | _joint_name_scales(ANKLE_PITCH_JOINT_NAMES, ankle_pitch_kp_scale)
        | _joint_name_scales(ANKLE_ROLL_JOINT_NAMES, ankle_roll_kp_scale)
    )
    joint_kv_scales = (
        _joint_name_scales(HIP_PITCH_JOINT_NAMES, hip_pitch_kv_scale)
        | _joint_name_scales(KNEE_PITCH_JOINT_NAMES, knee_pitch_kv_scale)
        | _joint_name_scales(ANKLE_PITCH_JOINT_NAMES, ankle_pitch_kv_scale)
        | _joint_name_scales(ANKLE_ROLL_JOINT_NAMES, ankle_roll_kv_scale)
    )
    return joint_kp_scales, joint_kv_scales
