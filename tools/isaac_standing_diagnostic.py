from __future__ import annotations

import argparse
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


def _detect_fixed_root_markers(physics_usd_path: Path) -> dict[str, bool]:
    if not physics_usd_path.is_file():
        return {"physics_usd_exists": False, "has_root_joint": False, "has_fixed_token": False}

    usd_bytes = physics_usd_path.read_bytes()
    return {
        "physics_usd_exists": True,
        "has_root_joint": b"root_joint" in usd_bytes,
        "has_fixed_token": b"Fixed" in usd_bytes,
    }


def summarize_standing_trace(
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

    if "feet_pos_w" in trace:
        feet_pos_w = np.asarray(trace["feet_pos_w"], dtype=np.float64)
        foot_z = feet_pos_w[:, :, 2]
        lines.append(
            f"feet_z_w: start=({foot_z[0, 0]:.4f}, {foot_z[0, 1]:.4f}), "
            f"end=({foot_z[-1, 0]:.4f}, {foot_z[-1, 1]:.4f}), "
            f"min=({np.min(foot_z[:, 0]):.4f}, {np.min(foot_z[:, 1]):.4f})"
        )

    for label, index in (
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
        top_vel_idx = np.argsort(np.abs(joint_vel[index]))[::-1][:5]
        top_err_idx = np.argsort(np.abs(joint_pos_error[index]))[::-1][:5]
        lines.append(
            f"{label} top_joint_vel: "
            + ", ".join(f"j{int(i)}={joint_vel[index, i]:+.4f}" for i in top_vel_idx)
        )
        lines.append(
            f"{label} top_joint_pos_error: "
            + ", ".join(f"j{int(i)}={joint_pos_error[index, i]:+.4f}" for i in top_err_idx)
        )

    return lines


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
    parser.add_argument("--hip_pitch_target", type=float, default=None)
    parser.add_argument("--knee_pitch_target", type=float, default=None)
    parser.add_argument("--ankle_pitch_target", type=float, default=None)
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

    env_class = task_registry.get_task_class(args_cli.task)
    env = None
    try:
        env = env_class(env_cfg, args_cli.headless)
        physics_usd_path = Path(env.robot.cfg.spawn.usd_path).resolve().parent / "configuration" / "humanoid_publish_physics.usd"
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
        env.scene.write_data_to_sim()
        env.sim.forward()

        settle_policy_steps = max(0, int(round(args_cli.settle_time / env.step_dt)))
        zero_action = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device)
        for _ in range(settle_policy_steps):
            env.step(zero_action)

        total_policy_steps = max(1, int(round(args_cli.duration / env.step_dt)))
        trace: dict[str, list[np.ndarray | float | bool]] = {
            "sim_time": [],
            "root_pos": [],
            "projected_gravity": [],
            "joint_pos_policy": [],
            "joint_vel_policy": [],
            "joint_pos_error_policy": [],
            "foot_normal_forces": [],
            "feet_pos_w": [],
            "termination_contact": [],
        }

        for step_idx in range(total_policy_steps):
            _, _, reset_buf, _ = env.step(zero_action)

            current_time = (step_idx + 1) * env.step_dt
            root_pos = env.robot.data.root_pos_w[0].detach().cpu().numpy().copy()
            projected_gravity = env.robot.data.projected_gravity_b[0].detach().cpu().numpy().copy()
            joint_pos = env.robot.data.joint_pos[0, env.policy_joint_ids].detach().cpu().numpy().copy()
            joint_vel = env.robot.data.joint_vel[0, env.policy_joint_ids].detach().cpu().numpy().copy()
            foot_forces = env.avg_feet_force_per_step[0].detach().cpu().numpy().astype(np.float64).copy()
            feet_pos_w = env.robot.data.body_pos_w[0, env.feet_body_ids, :].detach().cpu().numpy().astype(np.float64).copy()
            termination_contact = bool(reset_buf[0].item())

            trace["sim_time"].append(float(current_time))
            trace["root_pos"].append(root_pos)
            trace["projected_gravity"].append(projected_gravity)
            trace["joint_pos_policy"].append(joint_pos)
            trace["joint_vel_policy"].append(joint_vel)
            trace["joint_pos_error_policy"].append((joint_pos - standing_target).astype(np.float64))
            trace["foot_normal_forces"].append(foot_forces)
            trace["feet_pos_w"].append(feet_pos_w)
            trace["termination_contact"].append(termination_contact)

            if termination_contact:
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
        ):
            print(line)
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
