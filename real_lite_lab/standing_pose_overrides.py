from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constants import ANKLE_PITCH_JOINT_NAMES, HIP_PITCH_JOINT_NAMES, KNEE_PITCH_JOINT_NAMES


def apply_symmetric_standing_pitch_offsets(
    default_dof_pos: Sequence[float],
    joint_names: Sequence[str],
    *,
    hip_pitch_offset: float = 0.0,
    knee_pitch_offset: float = 0.0,
    ankle_pitch_offset: float = 0.0,
) -> np.ndarray:
    adjusted = np.asarray(default_dof_pos, dtype=np.float64).copy()
    joint_name_to_idx = {joint_name: idx for idx, joint_name in enumerate(joint_names)}

    for joint_name in HIP_PITCH_JOINT_NAMES:
        joint_idx = joint_name_to_idx.get(joint_name)
        if joint_idx is not None:
            adjusted[joint_idx] += float(hip_pitch_offset)
    for joint_name in KNEE_PITCH_JOINT_NAMES:
        joint_idx = joint_name_to_idx.get(joint_name)
        if joint_idx is not None:
            adjusted[joint_idx] += float(knee_pitch_offset)
    for joint_name in ANKLE_PITCH_JOINT_NAMES:
        joint_idx = joint_name_to_idx.get(joint_name)
        if joint_idx is not None:
            adjusted[joint_idx] += float(ankle_pitch_offset)

    return adjusted
