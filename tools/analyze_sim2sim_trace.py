from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def _load_trace(trace_path: Path) -> dict[str, np.ndarray]:
    with np.load(trace_path) as data:
        return {key: data[key] for key in data.files}


def _tilt_deg_from_projected_gravity(projected_gravity: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(projected_gravity, axis=1)
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    cos_theta = np.clip(-projected_gravity[:, 2] / safe_norms, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _first_index(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    return int(indices[0])


def _format_event(label: str, index: int | None, times: np.ndarray) -> str:
    if index is None:
        return f"{label}: not reached"
    return f"{label}: step={index}, time={times[index]:.3f}s"


def analyze_trace(trace_path: Path, *, height_drop_threshold: float, tilt_threshold_deg: float) -> list[str]:
    trace = _load_trace(trace_path)

    required = (
        "sim_time",
        "root_pos",
        "projected_gravity",
        "angular_velocity",
        "joint_vel_isaac",
    )
    missing = [key for key in required if key not in trace]
    if missing:
        raise KeyError(f"Trace file is missing required keys: {', '.join(missing)}")

    times = np.asarray(trace["sim_time"], dtype=np.float64)
    root_pos = np.asarray(trace["root_pos"], dtype=np.float64)
    projected_gravity = np.asarray(trace["projected_gravity"], dtype=np.float64)
    angular_velocity = np.asarray(trace["angular_velocity"], dtype=np.float64)
    joint_vel = np.asarray(trace["joint_vel_isaac"], dtype=np.float64)

    tilt_deg = _tilt_deg_from_projected_gravity(projected_gravity)
    root_z = root_pos[:, 2]
    root_xy_disp = np.linalg.norm(root_pos[:, :2] - root_pos[0, :2], axis=1)
    ang_speed = np.linalg.norm(angular_velocity, axis=1)
    joint_speed = np.max(np.abs(joint_vel), axis=1)

    min_root_z_idx = int(np.argmin(root_z))
    max_tilt_idx = int(np.argmax(tilt_deg))
    max_ang_speed_idx = int(np.argmax(ang_speed))
    max_joint_speed_idx = int(np.argmax(joint_speed))

    start_root_z = float(root_z[0])
    drop_event_idx = _first_index(root_z <= start_root_z - height_drop_threshold)
    tilt_event_idx = _first_index(tilt_deg >= tilt_threshold_deg)
    severe_tilt_idx = _first_index(tilt_deg >= 45.0)

    lines = [
        f"trace: {trace_path}",
        f"frames: {len(times)}",
        f"duration: {times[-1]:.3f}s",
        f"root_z: start={root_z[0]:.4f}, end={root_z[-1]:.4f}, min={root_z[min_root_z_idx]:.4f} at {times[min_root_z_idx]:.3f}s",
        f"root_xy_disp: end={root_xy_disp[-1]:.4f}m, max={np.max(root_xy_disp):.4f}m",
        f"tilt_deg: start={tilt_deg[0]:.2f}, end={tilt_deg[-1]:.2f}, max={tilt_deg[max_tilt_idx]:.2f} at {times[max_tilt_idx]:.3f}s",
        f"ang_speed: start={ang_speed[0]:.4f}, end={ang_speed[-1]:.4f}, max={ang_speed[max_ang_speed_idx]:.4f} at {times[max_ang_speed_idx]:.3f}s",
        f"joint_speed_abs_max: start={joint_speed[0]:.4f}, end={joint_speed[-1]:.4f}, max={joint_speed[max_joint_speed_idx]:.4f} at {times[max_joint_speed_idx]:.3f}s",
        _format_event(f"root height drop >= {height_drop_threshold:.3f}m", drop_event_idx, times),
        _format_event(f"tilt >= {tilt_threshold_deg:.1f}deg", tilt_event_idx, times),
        _format_event("tilt >= 45.0deg", severe_tilt_idx, times),
    ]

    if "action" in trace:
        action = np.asarray(trace["action"], dtype=np.float64)
        action_abs = np.max(np.abs(action), axis=1)
        max_action_idx = int(np.argmax(action_abs))
        lines.append(
            f"action_abs_max: start={action_abs[0]:.4f}, end={action_abs[-1]:.4f}, "
            f"max={action_abs[max_action_idx]:.4f} at {times[max_action_idx]:.3f}s"
        )

    if "ctrl" in trace:
        ctrl = np.asarray(trace["ctrl"], dtype=np.float64)
        ctrl_range = np.max(ctrl, axis=1) - np.min(ctrl, axis=1)
        max_ctrl_range_idx = int(np.argmax(ctrl_range))
        lines.append(
            f"ctrl_range: start={ctrl_range[0]:.4f}, end={ctrl_range[-1]:.4f}, "
            f"max={ctrl_range[max_ctrl_range_idx]:.4f} at {times[max_ctrl_range_idx]:.3f}s"
        )

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a sim2sim trace exported by sim2sim_real_lite.py.")
    parser.add_argument("trace", help="Path to sim2sim trace .npz")
    parser.add_argument(
        "--height_drop_threshold",
        type=float,
        default=0.05,
        help="Mark the first time the root drops by at least this many meters from its start height.",
    )
    parser.add_argument(
        "--tilt_threshold_deg",
        type=float,
        default=20.0,
        help="Mark the first time the body tilt reaches at least this many degrees.",
    )
    args = parser.parse_args()

    trace_path = Path(args.trace).resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    for line in analyze_trace(
        trace_path,
        height_drop_threshold=args.height_drop_threshold,
        tilt_threshold_deg=args.tilt_threshold_deg,
    ):
        print(line)


if __name__ == "__main__":
    main()
