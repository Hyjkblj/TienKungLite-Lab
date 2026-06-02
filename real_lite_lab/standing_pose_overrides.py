from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constants import ANKLE_PITCH_JOINT_NAMES, HIP_PITCH_JOINT_NAMES, KNEE_PITCH_JOINT_NAMES


def apply_symmetric_standing_pitch_targets(
    default_dof_pos: Sequence[float],
    joint_names: Sequence[str],
    *,
    hip_pitch_target: float | None = None,
    knee_pitch_target: float | None = None,
    ankle_pitch_target: float | None = None,
    hip_pitch_offset: float = 0.0,
    knee_pitch_offset: float = 0.0,
    ankle_pitch_offset: float = 0.0,
) -> np.ndarray:
    adjusted = np.asarray(default_dof_pos, dtype=np.float64).copy()
    joint_name_to_idx = {joint_name: idx for idx, joint_name in enumerate(joint_names)}

    for joint_name in HIP_PITCH_JOINT_NAMES:
        joint_idx = joint_name_to_idx.get(joint_name)
        if joint_idx is not None:
            if hip_pitch_target is not None:
                adjusted[joint_idx] = float(hip_pitch_target)
            adjusted[joint_idx] += float(hip_pitch_offset)
    for joint_name in KNEE_PITCH_JOINT_NAMES:
        joint_idx = joint_name_to_idx.get(joint_name)
        if joint_idx is not None:
            if knee_pitch_target is not None:
                adjusted[joint_idx] = float(knee_pitch_target)
            adjusted[joint_idx] += float(knee_pitch_offset)
    for joint_name in ANKLE_PITCH_JOINT_NAMES:
        joint_idx = joint_name_to_idx.get(joint_name)
        if joint_idx is not None:
            if ankle_pitch_target is not None:
                adjusted[joint_idx] = float(ankle_pitch_target)
            adjusted[joint_idx] += float(ankle_pitch_offset)

    return adjusted


def apply_symmetric_standing_pitch_offsets(
    default_dof_pos: Sequence[float],
    joint_names: Sequence[str],
    *,
    hip_pitch_offset: float = 0.0,
    knee_pitch_offset: float = 0.0,
    ankle_pitch_offset: float = 0.0,
) -> np.ndarray:
    return apply_symmetric_standing_pitch_targets(
        default_dof_pos,
        joint_names,
        hip_pitch_offset=hip_pitch_offset,
        knee_pitch_offset=knee_pitch_offset,
        ankle_pitch_offset=ankle_pitch_offset,
    )
