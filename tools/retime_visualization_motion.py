from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from real_lite_lab.constants import (
    VIS_LEFT_ARM_VEL_SLICE,
    VIS_LEFT_LEG_VEL_SLICE,
    VIS_RIGHT_ARM_VEL_SLICE,
    VIS_RIGHT_LEG_VEL_SLICE,
    VIS_ROOT_ANG_VEL_SLICE,
    VIS_ROOT_LIN_VEL_SLICE,
)


VELOCITY_SLICES = (
    VIS_ROOT_LIN_VEL_SLICE,
    VIS_ROOT_ANG_VEL_SLICE,
    VIS_LEFT_LEG_VEL_SLICE,
    VIS_RIGHT_LEG_VEL_SLICE,
    VIS_LEFT_ARM_VEL_SLICE,
    VIS_RIGHT_ARM_VEL_SLICE,
)


def _load_motion(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = np.asarray(payload["Frames"], dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != VIS_RIGHT_ARM_VEL_SLICE.stop:
        raise ValueError(f"Expected visualization frames [N, {VIS_RIGHT_ARM_VEL_SLICE.stop}], got {frames.shape}.")
    return payload


def _resolve_time_scale(frames: np.ndarray, explicit_scale: float | None, target_mean_vx: float | None) -> float:
    if explicit_scale is not None:
        return float(explicit_scale)
    if target_mean_vx is None:
        raise ValueError("Either --time_scale or --target_mean_vx must be provided.")
    mean_vx = float(np.mean(frames[:, VIS_ROOT_LIN_VEL_SLICE][:, 0]))
    if abs(target_mean_vx) < 1.0e-8:
        raise ValueError("--target_mean_vx must be non-zero.")
    if mean_vx <= 0.0:
        raise ValueError(f"Source mean root vx must be positive for forward retiming, got {mean_vx:.6f}.")
    return mean_vx / float(target_mean_vx)


def retime_motion(input_path: Path, output_path: Path, *, time_scale: float | None, target_mean_vx: float | None) -> None:
    payload = _load_motion(input_path)
    frames = np.asarray(payload["Frames"], dtype=np.float64)
    resolved_scale = _resolve_time_scale(frames, time_scale, target_mean_vx)
    if resolved_scale <= 0.0:
        raise ValueError(f"time_scale must be positive, got {resolved_scale}.")

    retimed_frames = frames.copy()
    for velocity_slice in VELOCITY_SLICES:
        retimed_frames[:, velocity_slice] /= resolved_scale

    source_dt = float(payload["FrameDuration"])
    payload["FrameDuration"] = round(source_dt * resolved_scale, 6)
    payload["Frames"] = retimed_frames.tolist()
    payload["RetimedFrom"] = input_path.as_posix()
    payload["TimeScale"] = round(resolved_scale, 6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    duration = max(retimed_frames.shape[0] - 1, 0) * payload["FrameDuration"]
    mean_vx = float(np.mean(retimed_frames[:, VIS_ROOT_LIN_VEL_SLICE][:, 0]))
    print(f"[INFO] Wrote: {output_path}")
    print(f"[INFO] time_scale={resolved_scale:.6f}, frame_duration={payload['FrameDuration']:.6f}s")
    print(f"[INFO] frames={retimed_frames.shape[0]}, duration_s={duration:.3f}, mean_root_vx={mean_vx:.4f}m/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retime a visualization motion by scaling its timeline and velocities.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time_scale", type=float, default=None)
    parser.add_argument("--target_mean_vx", type=float, default=None)
    args = parser.parse_args()
    retime_motion(args.input, args.output, time_scale=args.time_scale, target_mean_vx=args.target_mean_vx)


if __name__ == "__main__":
    main()
