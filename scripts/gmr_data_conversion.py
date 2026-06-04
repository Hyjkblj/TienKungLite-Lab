import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from real_lite_lab.constants import (  # noqa: E402
    DEFAULT_DOF_POS,
    LEFT_ARM_JOINT_NAMES,
    POLICY_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
)


GMR_TIENKUNGLITE_JOINT_NAMES = (
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "hip_pitch_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "hip_pitch_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_r_joint",
)
SOURCE_JOINT_ORDERS = {
    "policy": tuple(POLICY_JOINT_NAMES),
    "gmr_tienkunglite": GMR_TIENKUNGLITE_JOINT_NAMES,
}
UPPER_BODY_JOINT_NAMES = tuple(LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES)


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load motion PKLs produced with NumPy module path variants."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_motion_data(input_pkl: str):
    with open(input_pkl, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as exc:
            if exc.name != "numpy._core":
                raise
            f.seek(0)
            return _NumpyCompatUnpickler(f).load()


def _resolve_fps(motion_data: dict, fps: float | None) -> float:
    if fps is not None:
        return float(fps)
    source_fps = motion_data.get("fps")
    if source_fps is None:
        return 30.0
    return float(source_fps)


def _resolve_source_order(motion_data: dict, source_order: str) -> str:
    if source_order != "auto":
        return source_order

    link_body_list = motion_data.get("link_body_list") or []
    if (
        motion_data.get("dof_pos") is not None
        and np.asarray(motion_data["dof_pos"]).shape[-1] == len(GMR_TIENKUNGLITE_JOINT_NAMES)
        and "left_hand" in link_body_list
        and "right_hand" in link_body_list
        and "elbow_pitch_l_link" in link_body_list
        and "elbow_pitch_r_link" in link_body_list
    ):
        return "gmr_tienkunglite"

    return "policy"


def _reorder_joint_matrix(
    joint_matrix: np.ndarray,
    *,
    source_joint_names: tuple[str, ...],
    target_joint_names: tuple[str, ...] = tuple(POLICY_JOINT_NAMES),
) -> np.ndarray:
    joint_matrix = np.asarray(joint_matrix, dtype=np.float64)
    if joint_matrix.ndim != 2:
        raise ValueError(f"Expected joint_matrix with shape [T, J], got {joint_matrix.shape}.")
    if joint_matrix.shape[1] != len(source_joint_names):
        raise ValueError(
            f"Expected {len(source_joint_names)} joints for source order, got {joint_matrix.shape[1]}."
        )
    source_index = {name: idx for idx, name in enumerate(source_joint_names)}
    return np.stack([joint_matrix[:, source_index[name]] for name in target_joint_names], axis=1)


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    conjugate = quat.copy()
    conjugate[..., 1:] *= -1.0
    return conjugate


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def quat_wxyz_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    normalized_quat = quat / np.clip(norms, a_min=1e-12, a_max=None)
    quat_xyzw = normalized_quat[..., [1, 2, 3, 0]]
    return Rotation.from_quat(quat_xyzw).as_rotvec()


def _normalize_initial_yaw(root_pos: np.ndarray, root_rot_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_rot_wxyz = np.asarray(root_rot_wxyz, dtype=np.float64)

    if root_pos.shape[0] == 0:
        return root_pos.copy(), root_rot_wxyz.copy(), 0.0

    rotations = Rotation.from_quat(root_rot_wxyz[:, [1, 2, 3, 0]])
    forward_world = rotations[0].apply(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    forward_xy = forward_world[:2]
    forward_xy_norm = np.linalg.norm(forward_xy)
    if forward_xy_norm < 1e-8:
        return root_pos.copy(), root_rot_wxyz.copy(), 0.0

    initial_yaw = float(np.arctan2(forward_xy[1], forward_xy[0]))
    yaw_correction = Rotation.from_euler("z", -initial_yaw, degrees=False)

    anchor = root_pos[0].copy()
    normalized_root_pos = anchor + yaw_correction.apply(root_pos - anchor)
    normalized_rotations = yaw_correction * rotations
    normalized_root_rot_wxyz = normalized_rotations.as_quat()[:, [3, 0, 1, 2]]
    return normalized_root_pos, normalized_root_rot_wxyz, initial_yaw


def _apply_motion_profile(
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    *,
    motion_profile: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if motion_profile == "full_body":
        return root_pos, root_rot_wxyz, dof_pos
    if motion_profile != "upper_body":
        raise ValueError(f"Unsupported motion_profile: {motion_profile!r}")

    if root_pos.shape[0] == 0:
        return root_pos.copy(), root_rot_wxyz.copy(), dof_pos.copy()

    profiled_root_pos = np.repeat(root_pos[:1], root_pos.shape[0], axis=0)
    profiled_root_rot = np.repeat(root_rot_wxyz[:1], root_rot_wxyz.shape[0], axis=0)

    profiled_dof_pos = np.repeat(
        np.asarray(DEFAULT_DOF_POS, dtype=np.float64)[np.newaxis, :],
        dof_pos.shape[0],
        axis=0,
    )
    target_index = {name: idx for idx, name in enumerate(POLICY_JOINT_NAMES)}
    for joint_name in UPPER_BODY_JOINT_NAMES:
        joint_idx = target_index[joint_name]
        profiled_dof_pos[:, joint_idx] = dof_pos[:, joint_idx]

    return profiled_root_pos, profiled_root_rot, profiled_dof_pos


def _write_visualization_motion(output_path: Path, frames: np.ndarray, fps: float) -> None:
    payload = {
        "FrameType": "visualization",
        "LoopMode": "Wrap",
        "FrameDuration": round(1.0 / fps, 3),
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": 0.5,
        "Frames": frames.tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def convert_pkl_to_custom(
    input_pkl,
    output_txt,
    fps=None,
    normalize_initial_yaw=True,
    source_order="auto",
    motion_profile="full_body",
):
    motion_data = _load_motion_data(input_pkl)
    fps = _resolve_fps(motion_data, fps)
    source_order = _resolve_source_order(motion_data, source_order)
    dt = 1.0 / fps

    root_pos = np.asarray(motion_data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion_data["root_rot"], dtype=np.float64)[:, [3, 0, 1, 2]]  # xyzw -> wxyz
    dof_pos = _reorder_joint_matrix(
        np.asarray(motion_data["dof_pos"], dtype=np.float64),
        source_joint_names=SOURCE_JOINT_ORDERS[source_order],
    )

    applied_initial_yaw = 0.0
    if normalize_initial_yaw:
        root_pos, root_rot, applied_initial_yaw = _normalize_initial_yaw(root_pos, root_rot)
    root_pos, root_rot, dof_pos = _apply_motion_profile(
        root_pos,
        root_rot,
        dof_pos,
        motion_profile=motion_profile,
    )

    root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt
    q1_conj = quat_conjugate_wxyz(root_rot[:-1])
    dq = quat_mul_wxyz(q1_conj, root_rot[1:])
    root_ang_vel = quat_wxyz_to_rotvec(dq) / dt
    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt

    euler_angles = Rotation.from_quat(root_rot[:-1, [1, 2, 3, 0]]).as_euler("XYZ", degrees=False)
    euler_angles = np.unwrap(euler_angles, axis=0)

    data_output = np.concatenate(
        (
            root_pos[:-1],
            euler_angles,
            dof_pos[:-1],
            root_lin_vel,
            root_ang_vel,
            dof_vel,
        ),
        axis=1,
    )

    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_visualization_motion(output_path, data_output, fps)
    if normalize_initial_yaw:
        print(f"Normalized initial yaw by {-np.degrees(applied_initial_yaw):.3f} degrees.")
    print(f"Using fps={fps:.6f}, source_order={source_order}, motion_profile={motion_profile}.")
    print(f"Successfully converted {input_pkl} to {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--output_txt", type=str, required=True)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target frame rate. Defaults to the PKL fps field when available.",
    )
    parser.add_argument(
        "--source_order",
        type=str,
        default="auto",
        choices=("auto", *sorted(SOURCE_JOINT_ORDERS)),
        help="Joint order used inside the PKL dof_pos array.",
    )
    parser.add_argument(
        "--motion_profile",
        type=str,
        default="full_body",
        choices=("full_body", "upper_body"),
        help="full_body keeps the retargeted whole-body motion; upper_body freezes root and legs at the default stance.",
    )
    parser.add_argument(
        "--disable_initial_yaw_normalization",
        action="store_true",
        help="Keep the original world heading from the retargeted motion instead of aligning the first frame to +x.",
    )
    args = parser.parse_args()

    convert_pkl_to_custom(
        args.input_pkl,
        args.output_txt,
        args.fps,
        normalize_initial_yaw=not args.disable_initial_yaw_normalization,
        source_order=args.source_order,
        motion_profile=args.motion_profile,
    )
