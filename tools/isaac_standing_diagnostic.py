from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def _first_run_index(mask: np.ndarray, run_length: int) -> int | None:
    if run_length <= 1:
        return _first_index(mask)

    active_length = 0
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value:
            active_length += 1
            if active_length >= run_length:
                return int(index - run_length + 1)
        else:
            active_length = 0
    return None


def _format_event(label: str, index: int | None, times: np.ndarray) -> str:
    if index is None:
        return f"{label}: not reached"
    return f"{label}: step={index}, time={times[index]:.3f}s"


def _format_xy(values: np.ndarray) -> str:
    return f"({float(values[0]):+.4f}, {float(values[1]):+.4f})"


def _format_xyz(values: np.ndarray) -> str:
    return f"({float(values[0]):+.4f}, {float(values[1]):+.4f}, {float(values[2]):+.4f})"


def _detect_fixed_root_markers(physics_usd_path: Path) -> dict[str, bool]:
    if not physics_usd_path.is_file():
        return {"physics_usd_exists": False, "has_root_joint": False, "has_fixed_token": False}

    usd_bytes = physics_usd_path.read_bytes()
    return {
        "physics_usd_exists": True,
        "has_root_joint": b"root_joint" in usd_bytes,
        "has_fixed_token": b"Fixed" in usd_bytes,
    }


def _resolve_physics_usd_path(usd_path: Path) -> Path:
    candidates = (
        usd_path.parent / "configuration" / f"{usd_path.stem}_physics.usd",
        usd_path.parent / "configuration" / "humanoid_publish_physics.usd",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _joint_value_summary(values: np.ndarray, indices: np.ndarray, joint_names: tuple[str, ...] | None) -> str:
    parts = []
    for index in indices:
        label = f"j{int(index)}" if joint_names is None else joint_names[int(index)]
        parts.append(f"{label}={values[int(index)]:+.4f}")
    return ", ".join(parts)


def _optional_joint_signal_summary(
    trace: dict[str, np.ndarray],
    *,
    trace_key: str,
    label: str,
    times: np.ndarray,
    joint_names: tuple[str, ...] | None,
) -> list[str]:
    if trace_key not in trace:
        return []

    values = np.asarray(trace[trace_key], dtype=np.float64)
    if values.ndim != 2 or not np.any(np.isfinite(values)):
        return []

    abs_values = np.abs(values)
    abs_values = np.where(np.isfinite(abs_values), abs_values, np.nan)
    max_per_frame = np.nanmax(abs_values, axis=1)
    max_index = int(np.nanargmax(max_per_frame))
    start_value = float(max_per_frame[0])
    end_value = float(max_per_frame[-1])
    max_value = float(max_per_frame[max_index])
    lines = [
        f"{label}_abs_max: start={start_value:.4f}, end={end_value:.4f}, "
        f"max={max_value:.4f} at {times[max_index]:.3f}s"
    ]
    return lines


def _optional_joint_signal_event_lines(
    trace: dict[str, np.ndarray],
    *,
    trace_key: str,
    event_label: str,
    output_label: str,
    index: int,
    joint_names: tuple[str, ...] | None,
) -> list[str]:
    if trace_key not in trace:
        return []

    values = np.asarray(trace[trace_key], dtype=np.float64)
    if values.ndim != 2 or index >= values.shape[0] or not np.any(np.isfinite(values[index])):
        return []

    finite_abs = np.where(np.isfinite(values[index]), np.abs(values[index]), -np.inf)
    top_indices = np.argsort(finite_abs)[::-1][:5]
    return [
        f"{event_label} top_joint_{output_label}: "
        + _joint_value_summary(values[index], top_indices, joint_names)
    ]


def _scale_param_map(
    param_map: dict[str, float],
    *,
    pattern_scales: dict[str, float],
) -> dict[str, float]:
    updated: dict[str, float] = {}
    for joint_pattern, value in param_map.items():
        scale = 1.0
        for scale_pattern, candidate_scale in pattern_scales.items():
            if re.search(scale_pattern, joint_pattern):
                scale *= float(candidate_scale)
        updated[joint_pattern] = float(value) * scale
    return updated


def apply_isaac_actuator_scales(
    robot_cfg,
    *,
    hip_pitch_kp_scale: float = 1.0,
    hip_pitch_kd_scale: float = 1.0,
    knee_pitch_kp_scale: float = 1.0,
    knee_pitch_kd_scale: float = 1.0,
    ankle_pitch_kp_scale: float = 1.0,
    ankle_pitch_kd_scale: float = 1.0,
    ankle_roll_kp_scale: float = 1.0,
    ankle_roll_kd_scale: float = 1.0,
) -> dict[str, dict[str, float]]:
    stiffness_scales = {
        "hip_pitch": hip_pitch_kp_scale,
        "knee_pitch": knee_pitch_kp_scale,
        "ankle_pitch": ankle_pitch_kp_scale,
        "ankle_roll": ankle_roll_kp_scale,
    }
    damping_scales = {
        "hip_pitch": hip_pitch_kd_scale,
        "knee_pitch": knee_pitch_kd_scale,
        "ankle_pitch": ankle_pitch_kd_scale,
        "ankle_roll": ankle_roll_kd_scale,
    }

    effective: dict[str, dict[str, float]] = {"stiffness": {}, "damping": {}}
    for actuator_cfg in robot_cfg.actuators.values():
        stiffness = getattr(actuator_cfg, "stiffness", None)
        if isinstance(stiffness, dict):
            updated_stiffness = _scale_param_map(stiffness, pattern_scales=stiffness_scales)
            actuator_cfg.stiffness = updated_stiffness
            effective["stiffness"].update(updated_stiffness)

        damping = getattr(actuator_cfg, "damping", None)
        if isinstance(damping, dict):
            updated_damping = _scale_param_map(damping, pattern_scales=damping_scales)
            actuator_cfg.damping = updated_damping
            effective["damping"].update(updated_damping)
    return effective


def summarize_standing_trace(
    trace: dict[str, np.ndarray],
    *,
    height_drop_threshold: float,
    tilt_threshold_deg: float,
    support_force_threshold: float,
    support_hold_steps: int,
    joint_names: tuple[str, ...] | None = None,
) -> list[str]:
    required = (
        "sim_time",
        "root_pos",
        "projected_gravity",
        "joint_vel_policy",
        "joint_pos_error_policy",
        "foot_normal_forces",
        "termination_contact",
    )
    missing = [key for key in required if key not in trace]
    if missing:
        raise KeyError(f"Trace is missing required keys: {', '.join(missing)}")

    times = np.asarray(trace["sim_time"], dtype=np.float64)
    root_pos = np.asarray(trace["root_pos"], dtype=np.float64)
    projected_gravity = np.asarray(trace["projected_gravity"], dtype=np.float64)
    joint_vel = np.asarray(trace["joint_vel_policy"], dtype=np.float64)
    joint_pos_error = np.asarray(trace["joint_pos_error_policy"], dtype=np.float64)
    foot_normal_forces = np.asarray(trace["foot_normal_forces"], dtype=np.float64)
    termination_contact = np.asarray(trace["termination_contact"], dtype=bool)

    tilt_deg = _tilt_deg_from_projected_gravity(projected_gravity)
    root_z = root_pos[:, 2]
    root_xy_disp = np.linalg.norm(root_pos[:, :2] - root_pos[0, :2], axis=1)
    joint_speed = np.max(np.abs(joint_vel), axis=1)
    total_foot_force = np.sum(foot_normal_forces, axis=1)
    left_load_share = np.full_like(total_foot_force, np.nan, dtype=np.float64)
    np.divide(foot_normal_forces[:, 0], total_foot_force, out=left_load_share, where=total_foot_force > 1e-9)
    loaded_double_support = np.all(foot_normal_forces >= support_force_threshold, axis=1)

    min_root_z_idx = int(np.argmin(root_z))
    max_tilt_idx = int(np.argmax(tilt_deg))
    max_joint_speed_idx = int(np.argmax(joint_speed))
    min_total_foot_force_idx = int(np.argmin(total_foot_force))
    start_root_z = float(root_z[0])

    drop_event_idx = _first_index(root_z <= start_root_z - height_drop_threshold)
    tilt_event_idx = _first_index(tilt_deg >= tilt_threshold_deg)
    severe_tilt_idx = _first_index(tilt_deg >= 45.0)
    termination_contact_idx = _first_index(termination_contact)
    loaded_single_support_idx = _first_run_index(~loaded_double_support, support_hold_steps)

    lines = [
        f"frames: {len(times)}",
        f"duration: {times[-1]:.3f}s",
        f"root_z: start={root_z[0]:.4f}, end={root_z[-1]:.4f}, min={root_z[min_root_z_idx]:.4f} at {times[min_root_z_idx]:.3f}s",
        f"root_xy_disp: end={root_xy_disp[-1]:.4f}m, max={np.max(root_xy_disp):.4f}m",
        f"tilt_deg: start={tilt_deg[0]:.2f}, end={tilt_deg[-1]:.2f}, max={tilt_deg[max_tilt_idx]:.2f} at {times[max_tilt_idx]:.3f}s",
        f"joint_speed_abs_max: start={joint_speed[0]:.4f}, end={joint_speed[-1]:.4f}, max={joint_speed[max_joint_speed_idx]:.4f} at {times[max_joint_speed_idx]:.3f}s",
        _format_event(f"root height drop >= {height_drop_threshold:.3f}m", drop_event_idx, times),
        _format_event(f"tilt >= {tilt_threshold_deg:.1f}deg", tilt_event_idx, times),
        _format_event("tilt >= 45.0deg", severe_tilt_idx, times),
        _format_event("termination contact", termination_contact_idx, times),
        (
            f"foot_normal_force_total: start={total_foot_force[0]:.2f}, end={total_foot_force[-1]:.2f}, "
            f"min={total_foot_force[min_total_foot_force_idx]:.2f} at {times[min_total_foot_force_idx]:.3f}s"
        ),
        _format_event(f"loaded double support lost for {support_hold_steps} frames", loaded_single_support_idx, times),
    ]

    finite_left_load_share = np.where(np.isfinite(left_load_share), left_load_share, 0.5)
    max_left_load_imbalance_idx = int(np.argmax(np.abs(finite_left_load_share - 0.5)))
    lines.append(
        f"left_load_share: start={finite_left_load_share[0]:.3f}, end={finite_left_load_share[-1]:.3f}, "
        f"max_imbalance={np.abs(finite_left_load_share[max_left_load_imbalance_idx] - 0.5):.3f} at "
        f"{times[max_left_load_imbalance_idx]:.3f}s"
    )
    lines.extend(
        _optional_joint_signal_summary(
            trace,
            trace_key="joint_applied_torque_policy",
            label="joint_applied_torque",
            times=times,
            joint_names=joint_names,
        )
    )
    lines.extend(
        _optional_joint_signal_summary(
            trace,
            trace_key="joint_computed_torque_policy",
            label="joint_computed_torque",
            times=times,
            joint_names=joint_names,
        )
    )

    if "feet_pos_w" in trace:
        feet_pos_w = np.asarray(trace["feet_pos_w"], dtype=np.float64)
        foot_z = feet_pos_w[:, :, 2]
        lines.append(
            f"feet_z_w: start=({foot_z[0, 0]:.4f}, {foot_z[0, 1]:.4f}), "
            f"end=({foot_z[-1, 0]:.4f}, {foot_z[-1, 1]:.4f}), "
            f"min=({np.min(foot_z[:, 0]):.4f}, {np.min(foot_z[:, 1]):.4f})"
        )
        feet_center_xy = np.mean(feet_pos_w[:, :, :2], axis=1)
        root_to_feet_center_xy = root_pos[:, :2] - feet_center_xy
        root_to_feet_center_norm = np.linalg.norm(root_to_feet_center_xy, axis=1)
        max_root_to_feet_center_idx = int(np.argmax(root_to_feet_center_norm))
        lines.append(
            f"root_xy_minus_feet_center_xy: start={_format_xy(root_to_feet_center_xy[0])}, "
            f"end={_format_xy(root_to_feet_center_xy[-1])}, "
            f"max_norm={root_to_feet_center_norm[max_root_to_feet_center_idx]:.4f}m at "
            f"{times[max_root_to_feet_center_idx]:.3f}s"
        )
        if "system_com_pos_w" in trace:
            system_com_pos = np.asarray(trace["system_com_pos_w"], dtype=np.float64)
            finite_com = np.all(np.isfinite(system_com_pos), axis=1)
            if np.any(finite_com):
                com_to_feet_center_xy = system_com_pos[:, :2] - feet_center_xy
                com_to_feet_center_norm = np.linalg.norm(com_to_feet_center_xy, axis=1)
                finite_norm = np.where(finite_com, com_to_feet_center_norm, -np.inf)
                max_com_to_feet_center_idx = int(np.argmax(finite_norm))
                lines.append(
                    f"system_com_pos_w: start={_format_xyz(system_com_pos[0])}, "
                    f"end={_format_xyz(system_com_pos[-1])}"
                )
                lines.append(
                    f"com_xy_minus_feet_center_xy: start={_format_xy(com_to_feet_center_xy[0])}, "
                    f"end={_format_xy(com_to_feet_center_xy[-1])}, "
                    f"max_norm={com_to_feet_center_norm[max_com_to_feet_center_idx]:.4f}m at "
                    f"{times[max_com_to_feet_center_idx]:.3f}s"
                )

    for label, index in (
        ("start_state", 0),
        ("termination_contact", termination_contact_idx),
        ("tilt_event", tilt_event_idx),
        ("drop_event", drop_event_idx),
        ("severe_tilt", severe_tilt_idx),
        ("loaded_single_support", loaded_single_support_idx),
    ):
        if index is None:
            continue
        lines.append(
            f"{label}@{times[index]:.3f}s: root_z={root_z[index]:.4f}, tilt={tilt_deg[index]:.2f}deg, "
            f"foot_forces=({foot_normal_forces[index, 0]:.2f}, {foot_normal_forces[index, 1]:.2f})"
        )
        if "feet_pos_w" in trace:
            feet_pos_w = np.asarray(trace["feet_pos_w"], dtype=np.float64)
            feet_center_xy = np.mean(feet_pos_w[index, :, :2], axis=0)
            root_to_feet_center_xy = root_pos[index, :2] - feet_center_xy
            lines.append(f"{label} root_xy_minus_feet_center_xy: {_format_xy(root_to_feet_center_xy)}")
            if "system_com_pos_w" in trace:
                system_com_pos = np.asarray(trace["system_com_pos_w"], dtype=np.float64)
                if index < system_com_pos.shape[0] and np.all(np.isfinite(system_com_pos[index])):
                    com_to_feet_center_xy = system_com_pos[index, :2] - feet_center_xy
                    lines.append(f"{label} com_xy_minus_feet_center_xy: {_format_xy(com_to_feet_center_xy)}")
        top_vel_idx = np.argsort(np.abs(joint_vel[index]))[::-1][:5]
        top_err_idx = np.argsort(np.abs(joint_pos_error[index]))[::-1][:5]
        lines.append(
            f"{label} top_joint_vel: " + _joint_value_summary(joint_vel[index], top_vel_idx, joint_names)
        )
        lines.append(
            f"{label} top_joint_pos_error: " + _joint_value_summary(joint_pos_error[index], top_err_idx, joint_names)
        )
        lines.extend(
            _optional_joint_signal_event_lines(
                trace,
                trace_key="joint_applied_torque_policy",
                event_label=label,
                output_label="applied_torque",
                index=index,
                joint_names=joint_names,
            )
        )
        lines.extend(
            _optional_joint_signal_event_lines(
                trace,
                trace_key="joint_computed_torque_policy",
                event_label=label,
                output_label="computed_torque",
                index=index,
                joint_names=joint_names,
            )
        )

    return lines


def _max_contact_force_by_body(env, body_ids) -> tuple[float, str, np.ndarray]:
    import torch

    forces = env.contact_sensor.data.net_forces_w_history[:, :, body_ids, :3]
    force_norms = torch.norm(forces, dim=-1)
    max_per_body = torch.max(force_norms[0], dim=0)[0]
    max_index = int(torch.argmax(max_per_body).item())
    max_force = float(max_per_body[max_index].item())
    body_id = int(body_ids[max_index])
    body_names = getattr(env.robot, "body_names", [])
    body_name = body_names[body_id] if body_id < len(body_names) else f"body_{body_id}"
    return max_force, body_name, max_per_body.detach().cpu().numpy().astype(np.float64).copy()


def _compute_termination_contact(env) -> tuple[bool, float, str, np.ndarray]:
    max_force, body_name, max_per_body = _max_contact_force_by_body(env, env.termination_contact_cfg.body_ids)
    return max_force > 1.0, max_force, body_name, max_per_body


def _first_env_tensor_value(env, attr_names: tuple[str, ...]) -> np.ndarray | None:
    import torch

    for attr_name in attr_names:
        values = getattr(env.robot.data, attr_name, None)
        if values is None or not torch.is_tensor(values) or values.ndim < 2:
            continue
        return values[0].detach().cpu().numpy().astype(np.float64).copy()
    return None


def _system_com_pos_or_nan(env) -> np.ndarray:
    body_com = _first_env_tensor_value(env, ("body_com_pos_w", "body_com_pose_w", "body_pos_w", "body_state_w"))
    body_mass = _first_env_tensor_value(env, ("body_mass", "default_mass", "body_masses", "default_body_mass"))
    if body_com is None or body_mass is None:
        return np.full(3, np.nan, dtype=np.float64)

    body_com_pos = body_com[:, :3]
    body_mass = np.asarray(body_mass, dtype=np.float64).reshape(-1)
    if body_com_pos.shape[0] != body_mass.shape[0]:
        return np.full(3, np.nan, dtype=np.float64)

    valid = np.isfinite(body_mass) & (body_mass > 0.0) & np.all(np.isfinite(body_com_pos), axis=1)
    if not np.any(valid):
        return np.full(3, np.nan, dtype=np.float64)

    valid_mass = body_mass[valid]
    total_mass = float(np.sum(valid_mass))
    if total_mass <= 0.0:
        return np.full(3, np.nan, dtype=np.float64)
    return np.sum(body_com_pos[valid] * valid_mass[:, None], axis=0) / total_mass


def _step_pd_hold_without_env_reset(env, joint_position_targets, *, headless: bool) -> tuple[np.ndarray, bool, float, str, np.ndarray]:
    import torch

    foot_force_sum = torch.zeros(len(env.feet_cfg.body_ids), dtype=torch.float, device=env.device)
    for _ in range(env.cfg.sim.decimation):
        env.sim_step_counter += 1
        env.robot.set_joint_position_target(joint_position_targets)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        foot_force_sum += torch.norm(
            env.contact_sensor.data.net_forces_w[:, env.feet_cfg.body_ids, :3],
            dim=-1,
        )[0]

    if not headless:
        env.sim.render()

    foot_forces = (foot_force_sum / env.cfg.sim.decimation).detach().cpu().numpy().astype(np.float64).copy()
    termination_contact, termination_force, termination_body, termination_forces_by_body = _compute_termination_contact(env)
    return foot_forces, termination_contact, termination_force, termination_body, termination_forces_by_body


def _write_initial_root_state(env, *, root_z: float | None) -> None:
    if root_z is None:
        return

    import torch

    env_ids = torch.arange(env.num_envs, device=env.device)
    root_state = env.robot.data.root_state_w.clone()
    root_state[:, 2] = float(root_z)
    root_state[:, 7:13] = 0.0
    env.robot.write_root_state_to_sim(root_state, env_ids)


def _policy_joint_signal_or_nan(env, attr_names: tuple[str, ...]) -> np.ndarray:
    import torch

    for attr_name in attr_names:
        values = getattr(env.robot.data, attr_name, None)
        if values is None or not torch.is_tensor(values):
            continue
        try:
            selected = values[0, env.policy_joint_ids]
        except (IndexError, RuntimeError, TypeError):
            continue
        return selected.detach().cpu().numpy().astype(np.float64).copy()
    return np.full(env.num_actions, np.nan, dtype=np.float64)


def evaluate_standing_stability(
    trace: dict[str, np.ndarray],
    *,
    height_drop_threshold: float,
    tilt_threshold_deg: float,
    support_force_threshold: float,
    support_hold_steps: int,
) -> list[str]:
    required = (
        "sim_time",
        "root_pos",
        "projected_gravity",
        "foot_normal_forces",
        "termination_contact",
    )
    missing = [key for key in required if key not in trace]
    if missing:
        raise KeyError(f"Trace is missing required keys: {', '.join(missing)}")

    times = np.asarray(trace["sim_time"], dtype=np.float64)
    root_pos = np.asarray(trace["root_pos"], dtype=np.float64)
    projected_gravity = np.asarray(trace["projected_gravity"], dtype=np.float64)
    foot_normal_forces = np.asarray(trace["foot_normal_forces"], dtype=np.float64)
    termination_contact = np.asarray(trace["termination_contact"], dtype=bool)

    tilt_deg = _tilt_deg_from_projected_gravity(projected_gravity)
    root_z = root_pos[:, 2]
    start_root_z = float(root_z[0])
    loaded_double_support = np.all(foot_normal_forces >= support_force_threshold, axis=1)

    failures: list[str] = []
    termination_idx = _first_index(termination_contact)
    root_drop_idx = _first_index(root_z <= start_root_z - height_drop_threshold)
    tilt_idx = _first_index(tilt_deg >= tilt_threshold_deg)
    loaded_single_support_idx = _first_run_index(~loaded_double_support, support_hold_steps)

    if termination_idx is not None:
        failures.append(f"termination contact at {times[termination_idx]:.3f}s")
    if root_drop_idx is not None:
        failures.append(
            f"root dropped by >= {height_drop_threshold:.3f}m at {times[root_drop_idx]:.3f}s"
        )
    if tilt_idx is not None:
        failures.append(f"tilt reached >= {tilt_threshold_deg:.1f}deg at {times[tilt_idx]:.3f}s")
    if loaded_single_support_idx is not None:
        failures.append(
            "loaded double support was lost "
            f"for {support_hold_steps} frames at {times[loaded_single_support_idx]:.3f}s"
        )
    if float(np.max(np.sum(foot_normal_forces, axis=1))) <= 1e-6:
        failures.append("foot normal forces stayed at 0 across the rollout")
    return failures


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Run an Isaac-side standing hold diagnostic for the Real Lite asset.")
    parser.add_argument("--task", type=str, default="walk_real_lite")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--trace_out", default=None, help="Optional output .npz trace path.")
    parser.add_argument("--settle_time", type=float, default=0.6, help="Seconds of PD hold before trace capture starts.")
    parser.add_argument("--height_drop_threshold", type=float, default=0.05)
    parser.add_argument("--tilt_threshold_deg", type=float, default=20.0)
    parser.add_argument("--support_force_threshold", type=float, default=20.0)
    parser.add_argument("--support_hold_steps", type=int, default=3)
    parser.add_argument(
        "--root_z",
        type=float,
        default=None,
        help="Optional absolute world root height override applied after env reset and before PD hold.",
    )
    parser.add_argument(
        "--continue_after_termination",
        action="store_true",
        help="Keep stepping after a termination contact so the trace captures the full fall.",
    )
    parser.add_argument(
        "--require_stable",
        action="store_true",
        help="Exit non-zero if the hold trace drops, tilts, terminates, or loses loaded double support.",
    )
    parser.add_argument("--hip_pitch_target", type=float, default=None)
    parser.add_argument("--knee_pitch_target", type=float, default=None)
    parser.add_argument("--ankle_pitch_target", type=float, default=None)
    parser.add_argument("--hip_pitch_kp_scale", type=float, default=1.0)
    parser.add_argument("--hip_pitch_kd_scale", type=float, default=1.0)
    parser.add_argument("--knee_pitch_kp_scale", type=float, default=1.0)
    parser.add_argument("--knee_pitch_kd_scale", type=float, default=1.0)
    parser.add_argument("--ankle_pitch_kp_scale", type=float, default=1.0)
    parser.add_argument("--ankle_pitch_kd_scale", type=float, default=1.0)
    parser.add_argument("--ankle_roll_kp_scale", type=float, default=1.0)
    parser.add_argument("--ankle_roll_kd_scale", type=float, default=1.0)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from real_lite_lab import register_tasks
    from real_lite_lab.constants import DEFAULT_DOF_POS, POLICY_JOINT_NAMES
    from real_lite_lab.standing_pose_overrides import apply_symmetric_standing_pitch_targets
    from real_lite_lab.task_registry import task_registry

    register_tasks()
    env_cfg, _ = task_registry.get_cfgs(args_cli.task)

    env_cfg.noise.add_noise = False
    env_cfg.commands.rel_standing_envs = 1.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.debug_vis = False
    env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(env_cfg.commands.ranges, "heading"):
        env_cfg.commands.ranges.heading = (0.0, 0.0)

    env_cfg.scene.max_episode_length_s = max(float(args_cli.duration) + 1.0, float(env_cfg.scene.max_episode_length_s))
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    env_cfg.domain_rand.events.physics_material = None
    env_cfg.domain_rand.events.add_base_mass = None
    env_cfg.domain_rand.events.reset_base = None
    env_cfg.domain_rand.events.reset_robot_joints = None
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.action_delay.enable = False
    effective_actuator_params = apply_isaac_actuator_scales(
        env_cfg.scene.robot,
        hip_pitch_kp_scale=args_cli.hip_pitch_kp_scale,
        hip_pitch_kd_scale=args_cli.hip_pitch_kd_scale,
        knee_pitch_kp_scale=args_cli.knee_pitch_kp_scale,
        knee_pitch_kd_scale=args_cli.knee_pitch_kd_scale,
        ankle_pitch_kp_scale=args_cli.ankle_pitch_kp_scale,
        ankle_pitch_kd_scale=args_cli.ankle_pitch_kd_scale,
        ankle_roll_kp_scale=args_cli.ankle_roll_kp_scale,
        ankle_roll_kd_scale=args_cli.ankle_roll_kd_scale,
    )

    env_class = task_registry.get_task_class(args_cli.task)
    env = None
    try:
        env = env_class(env_cfg, args_cli.headless)
        physics_usd_path = _resolve_physics_usd_path(Path(env.robot.cfg.spawn.usd_path).resolve())
        fixed_root_markers = _detect_fixed_root_markers(physics_usd_path)

        standing_target = apply_symmetric_standing_pitch_targets(
            DEFAULT_DOF_POS,
            POLICY_JOINT_NAMES,
            hip_pitch_target=args_cli.hip_pitch_target,
            knee_pitch_target=args_cli.knee_pitch_target,
            ankle_pitch_target=args_cli.ankle_pitch_target,
        )

        joint_position_targets = env.robot.data.default_joint_pos.clone()
        joint_velocity_targets = env.robot.data.default_joint_vel.clone()
        joint_velocity_targets[:] = 0.0

        import torch

        standing_target_tensor = torch.tensor(standing_target, dtype=joint_position_targets.dtype, device=env.device)
        standing_target_batch = standing_target_tensor.unsqueeze(0).repeat(env.num_envs, 1)
        joint_position_targets[:, env.policy_joint_ids] = standing_target_batch

        # Make zero-action env.step() hold the diagnostic standing target instead of the training default.
        env.default_joint_pos_policy = standing_target_batch.clone()
        env.robot.data.default_joint_pos[:, env.policy_joint_ids] = standing_target_batch
        env.robot.data.default_joint_vel[:, env.policy_joint_ids] = 0.0

        env.robot.write_joint_position_to_sim(joint_position_targets)
        env.robot.write_joint_velocity_to_sim(joint_velocity_targets)
        _write_initial_root_state(env, root_z=args_cli.root_z)
        env.scene.write_data_to_sim()
        env.sim.forward()

        settle_policy_steps = max(0, int(round(args_cli.settle_time / env.step_dt)))
        for _ in range(settle_policy_steps):
            _step_pd_hold_without_env_reset(env, joint_position_targets, headless=args_cli.headless)

        total_policy_steps = max(1, int(round(args_cli.duration / env.step_dt)))
        trace: dict[str, list[np.ndarray | float | bool]] = {
            "sim_time": [],
            "root_pos": [],
            "root_quat_wxyz": [],
            "root_lin_vel": [],
            "root_ang_vel": [],
            "system_com_pos_w": [],
            "projected_gravity": [],
            "joint_pos_policy": [],
            "joint_vel_policy": [],
            "joint_pos_error_policy": [],
            "joint_applied_torque_policy": [],
            "joint_computed_torque_policy": [],
            "foot_normal_forces": [],
            "feet_pos_w": [],
            "termination_contact": [],
            "termination_force": [],
            "termination_body": [],
            "termination_forces_by_body": [],
        }

        for step_idx in range(total_policy_steps):
            (
                foot_forces,
                termination_contact,
                termination_force,
                termination_body,
                termination_forces_by_body,
            ) = _step_pd_hold_without_env_reset(env, joint_position_targets, headless=args_cli.headless)

            current_time = (step_idx + 1) * env.step_dt
            root_state_w = env.robot.data.root_state_w[0].detach().cpu().numpy().copy()
            root_pos = root_state_w[0:3]
            root_quat = root_state_w[3:7]
            root_lin_vel = root_state_w[7:10]
            root_ang_vel = root_state_w[10:13]
            system_com_pos = _system_com_pos_or_nan(env)
            projected_gravity = env.robot.data.projected_gravity_b[0].detach().cpu().numpy().copy()
            joint_pos = env.robot.data.joint_pos[0, env.policy_joint_ids].detach().cpu().numpy().copy()
            joint_vel = env.robot.data.joint_vel[0, env.policy_joint_ids].detach().cpu().numpy().copy()
            joint_applied_torque = _policy_joint_signal_or_nan(env, ("applied_torque", "joint_effort"))
            joint_computed_torque = _policy_joint_signal_or_nan(env, ("computed_torque",))
            feet_pos_w = env.robot.data.body_pos_w[0, env.feet_body_ids, :].detach().cpu().numpy().astype(np.float64).copy()

            trace["sim_time"].append(float(current_time))
            trace["root_pos"].append(root_pos)
            trace["root_quat_wxyz"].append(root_quat)
            trace["root_lin_vel"].append(root_lin_vel)
            trace["root_ang_vel"].append(root_ang_vel)
            trace["system_com_pos_w"].append(system_com_pos)
            trace["projected_gravity"].append(projected_gravity)
            trace["joint_pos_policy"].append(joint_pos)
            trace["joint_vel_policy"].append(joint_vel)
            trace["joint_pos_error_policy"].append((joint_pos - standing_target).astype(np.float64))
            trace["joint_applied_torque_policy"].append(joint_applied_torque)
            trace["joint_computed_torque_policy"].append(joint_computed_torque)
            trace["foot_normal_forces"].append(foot_forces)
            trace["feet_pos_w"].append(feet_pos_w)
            trace["termination_contact"].append(termination_contact)
            trace["termination_force"].append(float(termination_force))
            trace["termination_body"].append(termination_body)
            trace["termination_forces_by_body"].append(termination_forces_by_body)

            if termination_contact and not args_cli.continue_after_termination:
                break

        stacked_trace = {
            key: np.asarray(values) if key in {"sim_time", "termination_contact"} else np.stack(values, axis=0)
            for key, values in trace.items()
        }
        if args_cli.trace_out:
            trace_path = Path(args_cli.trace_out).resolve()
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(trace_path, **stacked_trace, standing_target_policy=standing_target.astype(np.float64))
            print(f"[INFO] Saved Isaac standing trace to: {trace_path}")

        print(
            "[INFO] Standing target: "
            f"hip_pitch={args_cli.hip_pitch_target if args_cli.hip_pitch_target is not None else 'default'}, "
            f"knee_pitch={args_cli.knee_pitch_target if args_cli.knee_pitch_target is not None else 'default'}, "
            f"ankle_pitch={args_cli.ankle_pitch_target if args_cli.ankle_pitch_target is not None else 'default'}"
        )
        print(
            "[INFO] Diagnostic init: "
            f"root_z={args_cli.root_z if args_cli.root_z is not None else 'asset default'}, "
            f"settle_time={args_cli.settle_time:g}s, "
            f"continue_after_termination={args_cli.continue_after_termination}"
        )
        print(
            "[INFO] Actuator scales: "
            f"hip_pitch(kp={args_cli.hip_pitch_kp_scale:g}, kd={args_cli.hip_pitch_kd_scale:g}), "
            f"knee_pitch(kp={args_cli.knee_pitch_kp_scale:g}, kd={args_cli.knee_pitch_kd_scale:g}), "
            f"ankle_pitch(kp={args_cli.ankle_pitch_kp_scale:g}, kd={args_cli.ankle_pitch_kd_scale:g}), "
            f"ankle_roll(kp={args_cli.ankle_roll_kp_scale:g}, kd={args_cli.ankle_roll_kd_scale:g})"
        )
        print(f"[INFO] Effective stiffness: {effective_actuator_params['stiffness']}")
        print(f"[INFO] Effective damping: {effective_actuator_params['damping']}")
        if fixed_root_markers["physics_usd_exists"]:
            print(
                "[INFO] Physics USD markers: "
                f"root_joint={fixed_root_markers['has_root_joint']}, "
                f"Fixed={fixed_root_markers['has_fixed_token']}, "
                f"path={physics_usd_path}"
            )
            if fixed_root_markers["has_root_joint"] and fixed_root_markers["has_fixed_token"]:
                print(
                    "[WARN] Physics USD contains both 'root_joint' and 'Fixed' markers; "
                    "this Isaac asset may be fixed-root and not directly comparable to MuJoCo free-base standing."
                )
        for line in summarize_standing_trace(
            stacked_trace,
            height_drop_threshold=args_cli.height_drop_threshold,
            tilt_threshold_deg=args_cli.tilt_threshold_deg,
            support_force_threshold=args_cli.support_force_threshold,
            support_hold_steps=args_cli.support_hold_steps,
            joint_names=POLICY_JOINT_NAMES,
        ):
            print(line)
        termination_indices = np.flatnonzero(np.asarray(stacked_trace["termination_contact"], dtype=bool))
        if termination_indices.size:
            termination_index = int(termination_indices[0])
            print(
                "[INFO] First termination body: "
                f"{stacked_trace['termination_body'][termination_index]} "
                f"force={float(stacked_trace['termination_force'][termination_index]):.3f}N"
            )
        stability_failures = evaluate_standing_stability(
            stacked_trace,
            height_drop_threshold=args_cli.height_drop_threshold,
            tilt_threshold_deg=args_cli.tilt_threshold_deg,
            support_force_threshold=args_cli.support_force_threshold,
            support_hold_steps=args_cli.support_hold_steps,
        )
        if args_cli.require_stable and stability_failures:
            print("[ERROR] Isaac standing stability gate failed:")
            for failure in stability_failures:
                print(f"[ERROR]   {failure}")
            raise RuntimeError("Isaac standing diagnostic did not satisfy the requested stability gate.")
        if float(np.max(np.sum(stacked_trace["foot_normal_forces"], axis=1))) <= 1e-6:
            print(
                "[WARN] Foot normal forces stayed at 0 across the rollout; "
                "this trace likely does not represent grounded stance contact."
            )
    finally:
        if env is not None:
            close_env = getattr(env, "close", None)
            if callable(close_env):
                close_env()
        simulation_app.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    sys.exit(exit_code)
