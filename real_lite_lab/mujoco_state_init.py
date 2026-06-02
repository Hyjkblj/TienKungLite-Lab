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


def snap_root_height_to_ground(*, model, data, ground_z: float = 0.0, clearance: float = 1e-4) -> float:
    """Shift the free root vertically so the lowest collidable support geom sits on the ground."""

    import mujoco

    support_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) != 0
        and int(model.geom_contype[geom_id]) != 0
        and int(model.geom_conaffinity[geom_id]) != 0
    ]
    if not support_geom_ids:
        return 0.0

    def geom_lowest_z(geom_id: int) -> float:
        geom_type = int(model.geom_type[geom_id])
        center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        size = np.asarray(model.geom_size[geom_id], dtype=np.float64)

        if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            z_extent = size[0]
        elif geom_type in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            axis_z = abs(float(rot[2, 2]))
            radial_z = np.sqrt(max(0.0, 1.0 - axis_z**2))
            z_extent = size[1] * axis_z + size[0] * radial_z
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            z_extent = (
                abs(float(rot[2, 0])) * size[0]
                + abs(float(rot[2, 1])) * size[1]
                + abs(float(rot[2, 2])) * size[2]
            )
        else:
            # Conservative fallback for uncommon support geom types.
            z_extent = float(model.geom_rbound[geom_id])

        return float(center[2] - z_extent)

    lowest_support_z = min(geom_lowest_z(geom_id) for geom_id in support_geom_ids)
    target_root_shift = ground_z + clearance - lowest_support_z
    data.qpos[2] += target_root_shift
    mujoco.mj_forward(model, data)
    return float(target_root_shift)
