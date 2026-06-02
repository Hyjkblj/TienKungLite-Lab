from __future__ import annotations

from typing import Callable

import numpy as np


def apply_default_joint_state(
    *,
    model,
    data,
    joint_names: list[str] | tuple[str, ...],
    default_joint_pos: np.ndarray,
    joint_name_to_id: Callable[[str], int],
    root_pos: tuple[float, float, float] = (0.0, 0.0, 1.0),
    root_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    if len(joint_names) != int(default_joint_pos.shape[0]):
        raise ValueError(
            f"default joint position size mismatch: expected {len(joint_names)}, got {default_joint_pos.shape[0]}."
        )

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    data.qpos[0:3] = np.asarray(root_pos, dtype=np.float64)
    data.qpos[3:7] = np.asarray(root_quat_wxyz, dtype=np.float64)

    for joint_name, joint_pos in zip(joint_names, default_joint_pos, strict=True):
        joint_id = int(joint_name_to_id(joint_name))
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qvel_adr = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_adr] = float(joint_pos)
        data.qvel[qvel_adr] = 0.0
