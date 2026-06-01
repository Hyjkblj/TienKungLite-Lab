import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


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


def convert_pkl_to_custom(input_pkl, output_txt, fps):
    dt = 1.0 / fps

    motion_data = _load_motion_data(input_pkl)

    root_pos = np.asarray(motion_data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion_data["root_rot"], dtype=np.float64)[:, [3, 0, 1, 2]]  # xyzw -> wxyz
    dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float64)

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
    print(f"Successfully converted {input_pkl} to {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--output_txt", type=str, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    convert_pkl_to_custom(args.input_pkl, args.output_txt, args.fps)
