from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load PKLs saved with NumPy module paths from newer versions."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        try:
            data = pickle.load(f)
        except ModuleNotFoundError as exc:
            if exc.name != "numpy._core":
                raise
            f.seek(0)
            data = _NumpyCompatUnpickler(f).load()
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict PKL, got {type(data).__name__}.")
    return data


def _load_visualization(path: Path) -> tuple[np.ndarray, float, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = np.asarray(payload["Frames"], dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] < 29:
        raise ValueError(f"Expected visualization frames with at least 29 columns, got {frames.shape}.")
    frame_duration = float(payload["FrameDuration"])
    return frames, frame_duration, str(payload.get("FrameType", "unknown"))


def _motion_from_pkl(path: Path) -> tuple[np.ndarray, np.ndarray, float, str]:
    data = _load_pickle(path)
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos with shape [T, 3], got {root_pos.shape}.")
    fps = float(data.get("fps") or 30.0)
    dt = 1.0 / fps
    root_lin_vel = np.zeros_like(root_pos)
    if root_pos.shape[0] > 1:
        root_lin_vel[:-1] = np.diff(root_pos, axis=0) / dt
        root_lin_vel[-1] = root_lin_vel[-2]
    return root_pos, root_lin_vel, dt, "gmr_pkl"


def _motion_from_txt(path: Path) -> tuple[np.ndarray, np.ndarray, float, str]:
    frames, dt, frame_type = _load_visualization(path)
    root_pos = frames[:, 0:3]
    root_lin_vel = frames[:, 26:29]
    return root_pos, root_lin_vel, dt, frame_type


def _format_vec(vec: np.ndarray) -> str:
    return "(" + ", ".join(f"{value:+.4f}" for value in vec.tolist()) + ")"


def _safe_ratio(num: float, den: float) -> float:
    if abs(den) < 1.0e-8:
        return 0.0
    return num / den


def analyze_motion(path: Path, *, min_forward_speed: float, max_lateral_ratio: float) -> int:
    if path.suffix.lower() == ".pkl":
        root_pos, root_lin_vel, dt, frame_type = _motion_from_pkl(path)
    else:
        root_pos, root_lin_vel, dt, frame_type = _motion_from_txt(path)

    frame_count = root_pos.shape[0]
    duration = max(frame_count - 1, 0) * dt
    delta = root_pos[-1] - root_pos[0] if frame_count else np.zeros(3)
    planar_delta = delta[:2]
    planar_distance = float(np.linalg.norm(planar_delta))
    path_length_xy = (
        float(np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1).sum()) if frame_count > 1 else 0.0
    )
    mean_velocity = np.mean(root_lin_vel, axis=0) if frame_count else np.zeros(3)
    median_velocity = np.median(root_lin_vel, axis=0) if frame_count else np.zeros(3)
    travel_speed = planar_distance / duration if duration > 0.0 else 0.0
    x_speed = float(mean_velocity[0])
    lateral_ratio = abs(float(delta[1])) / max(abs(float(delta[0])), 1.0e-8)

    forward_ok = x_speed >= min_forward_speed and float(delta[0]) > 0.0
    lateral_ok = lateral_ratio <= max_lateral_ratio
    verdict = "PASS" if forward_ok and lateral_ok else "CHECK"

    print(f"[INFO] motion={path}")
    print(f"[INFO] frame_type={frame_type}, frames={frame_count}, dt={dt:.6f}, duration_s={duration:.3f}")
    print(f"[INFO] root_start={_format_vec(root_pos[0])}, root_end={_format_vec(root_pos[-1])}")
    print(f"[INFO] root_delta={_format_vec(delta)}, planar_distance={planar_distance:.4f}m")
    print(f"[INFO] path_length_xy={path_length_xy:.4f}m, travel_speed={travel_speed:.4f}m/s")
    print(f"[INFO] mean_root_lin_vel={_format_vec(mean_velocity)}, median_root_lin_vel={_format_vec(median_velocity)}")
    print(f"[INFO] root_z_range=({root_pos[:, 2].min():.4f}, {root_pos[:, 2].max():.4f})")
    print(
        f"[INFO] forward_x_check: mean_vx={x_speed:.4f}m/s, min_required={min_forward_speed:.4f}, "
        f"delta_x={delta[0]:.4f}"
    )
    print(
        f"[INFO] lateral_check: abs(delta_y/delta_x)={lateral_ratio:.4f}, "
        f"max_allowed={max_lateral_ratio:.4f}"
    )
    print(f"[RESULT] forward_motion_verdict={verdict}")

    if path.suffix.lower() == ".pkl":
        print("[HINT] Convert this PKL before training, then audit the converted visualization txt:")
        print(
            "       python scripts/gmr_data_conversion.py "
            f"--input_pkl {path.as_posix()} "
            "--output_txt real_lite_lab/datasets/motion_visualization/walk_gmr_forward.txt "
            "--source_order auto --motion_profile full_body --align_travel_direction"
        )
    elif verdict != "PASS":
        print("[HINT] If this came from a GMR PKL, reconvert with --align_travel_direction and audit again.")

    return 0 if verdict == "PASS" else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GMR PKL or visualization motion for usable forward walking.")
    parser.add_argument("motion", type=Path, help="Path to a GMR .pkl or visualization .txt motion file.")
    parser.add_argument("--min_forward_speed", type=float, default=0.10)
    parser.add_argument("--max_lateral_ratio", type=float, default=0.20)
    args = parser.parse_args()
    raise SystemExit(
        analyze_motion(
            args.motion,
            min_forward_speed=args.min_forward_speed,
            max_lateral_ratio=args.max_lateral_ratio,
        )
    )


if __name__ == "__main__":
    main()
