from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from real_lite_lab.constants import DEFAULT_DOF_POS, POLICY_JOINT_NAMES


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


def _top_joint_table(values: np.ndarray, *, topk: int = 5) -> str:
    abs_values = np.abs(values)
    order = np.argsort(abs_values)[::-1][:topk]
    return ", ".join(f"{POLICY_JOINT_NAMES[idx]}={values[idx]:+.4f}" for idx in order)


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
    joint_pos = np.asarray(trace.get("joint_pos_isaac"), dtype=np.float64) if "joint_pos_isaac" in trace else None
    policy_target = np.asarray(trace.get("policy_target_isaac"), dtype=np.float64) if "policy_target_isaac" in trace else None
    clamped_target = (
        np.asarray(trace.get("clamped_target_isaac"), dtype=np.float64) if "clamped_target_isaac" in trace else None
    )
    support_margin = np.asarray(trace["support_margin"], dtype=np.float64) if "support_margin" in trace else None
    foot_normal_forces = np.asarray(trace["foot_normal_forces"], dtype=np.float64) if "foot_normal_forces" in trace else None
    left_load_share = np.asarray(trace["left_load_share"], dtype=np.float64) if "left_load_share" in trace else None
    double_support = np.asarray(trace["double_support"], dtype=np.int32) if "double_support" in trace else None

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
    support_loss_idx = _first_index(support_margin < 0.0) if support_margin is not None else None
    single_support_idx = _first_index(double_support == 0) if double_support is not None else None

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

    if support_margin is not None:
        min_support_margin_idx = int(np.argmin(support_margin))
        lines.append(
            f"support_margin: start={support_margin[0]:+.4f}, end={support_margin[-1]:+.4f}, "
            f"min={support_margin[min_support_margin_idx]:+.4f} at {times[min_support_margin_idx]:.3f}s"
        )
        lines.append(_format_event("support margin < 0.0m", support_loss_idx, times))

    if foot_normal_forces is not None:
        total_foot_force = np.sum(foot_normal_forces, axis=1)
        min_total_foot_force_idx = int(np.argmin(total_foot_force))
        lines.append(
            f"foot_normal_force_total: start={total_foot_force[0]:.2f}, end={total_foot_force[-1]:.2f}, "
            f"min={total_foot_force[min_total_foot_force_idx]:.2f} at {times[min_total_foot_force_idx]:.3f}s"
        )

    if left_load_share is not None:
        finite_left_load_share = np.where(np.isfinite(left_load_share), left_load_share, 0.5)
        left_load_imbalance = np.abs(finite_left_load_share - 0.5)
        max_left_load_imbalance_idx = int(np.argmax(left_load_imbalance))
        lines.append(
            f"left_load_share: start={finite_left_load_share[0]:.3f}, end={finite_left_load_share[-1]:.3f}, "
            f"max_imbalance={left_load_imbalance[max_left_load_imbalance_idx]:.3f} at {times[max_left_load_imbalance_idx]:.3f}s"
        )

    if double_support is not None:
        double_support_ratio = float(np.mean(double_support != 0))
        lines.append(f"double_support_ratio: {double_support_ratio:.3f}")
        lines.append(_format_event("double support lost", single_support_idx, times))

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

    target_clip = None
    target_clip_event_idx = None
    if policy_target is not None and clamped_target is not None:
        target_clip = clamped_target - policy_target
        target_clip_abs = np.max(np.abs(target_clip), axis=1)
        max_target_clip_idx = int(np.argmax(target_clip_abs))
        target_clip_event_idx = _first_index(target_clip_abs > 1e-6)
        lines.append(
            f"target_clip_abs_max: start={target_clip_abs[0]:.4f}, end={target_clip_abs[-1]:.4f}, "
            f"max={target_clip_abs[max_target_clip_idx]:.4f} at {times[max_target_clip_idx]:.3f}s"
        )
        lines.append(_format_event("target clipped by soft limit", target_clip_event_idx, times))

    event_indices: list[tuple[str, int]] = []
    for label, index in (
        ("tilt_event", tilt_event_idx),
        ("drop_event", drop_event_idx),
        ("severe_tilt", severe_tilt_idx),
        ("max_ang_speed", max_ang_speed_idx),
        ("max_joint_speed", max_joint_speed_idx),
        ("target_clip_event", target_clip_event_idx),
        ("support_loss", support_loss_idx),
        ("single_support", single_support_idx),
    ):
        if index is not None:
            event_indices.append((label, index))

    seen_indices: set[int] = set()
    for label, index in event_indices:
        if index in seen_indices:
            continue
        seen_indices.add(index)

        lines.append(
            f"{label}@{times[index]:.3f}s: root_z={root_z[index]:.4f}, tilt={tilt_deg[index]:.2f}deg, "
            f"ang_vel={angular_velocity[index]}"
        )
        if support_margin is not None:
            lines.append(f"{label} support_margin: {support_margin[index]:+.4f}m")
        if foot_normal_forces is not None:
            lines.append(
                f"{label} foot_normal_forces: left={foot_normal_forces[index, 0]:.2f}N, "
                f"right={foot_normal_forces[index, 1]:.2f}N"
            )
        if left_load_share is not None:
            lines.append(f"{label} left_load_share: {left_load_share[index]:.3f}")
        lines.append(f"{label} top_joint_vel: {_top_joint_table(joint_vel[index])}")
        if joint_pos is not None:
            default_dof_pos = np.asarray(DEFAULT_DOF_POS, dtype=np.float64)
            joint_pos_error = joint_pos[index] - default_dof_pos
            lines.append(f"{label} top_joint_pos_error: {_top_joint_table(joint_pos_error)}")
        if "action" in trace:
            lines.append(f"{label} top_action: {_top_joint_table(action[index])}")
        if target_clip is not None:
            lines.append(f"{label} top_target_clip: {_top_joint_table(target_clip[index])}")

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
