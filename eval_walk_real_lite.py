import argparse
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from debug_signals import install_stack_dump_signal
from isaaclab.app import AppLauncher

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

_stack_dump_signal = install_stack_dump_signal()
if _stack_dump_signal is not None:
    print(f"[INFO] Send SIGUSR1 to PID {os.getpid()} to dump Python stack traces.")

import real_lite_lab.cli_args as cli_args
from real_lite_lab.constants import TASK_NAMES


parser = argparse.ArgumentParser(description="Evaluate a trained Real Lite walk policy in Isaac Lab.")
parser.add_argument("--task", type=str, default="walk_real_lite", choices=TASK_NAMES)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--duration_s", type=float, default=30.0)
parser.add_argument("--command_vx", type=float, default=0.2)
parser.add_argument("--command_vy", type=float, default=0.0)
parser.add_argument("--command_wz", type=float, default=0.0)
parser.add_argument("--keep_reset_noise", action="store_true")
parser.add_argument("--keep_domain_rand", action="store_true")
parser.add_argument("--enable_noise", action="store_true")
parser.add_argument("--enable_terrain", action="store_true")
parser.add_argument("--trace_out", default=None, help="Optional .npz trace path for rendering a walk-policy video.")
parser.add_argument(
    "--min_linear_speed_ratio",
    type=float,
    default=0.5,
    help="Minimum mean velocity along the commanded linear direction, as a fraction of command speed.",
)
parser.add_argument(
    "--min_linear_progress_ratio",
    type=float,
    default=0.5,
    help="Minimum final displacement along the commanded linear direction, as a fraction of expected progress.",
)
parser.add_argument(
    "--max_lateral_drift_ratio",
    type=float,
    default=0.5,
    help="Maximum mean lateral drift as a fraction of expected linear progress.",
)
parser.add_argument(
    "--max_lateral_drift_abs",
    type=float,
    default=0.25,
    help="Absolute lower bound for allowed mean lateral drift in meters.",
)
parser.add_argument(
    "--min_yaw_rate_ratio",
    type=float,
    default=0.5,
    help="Minimum mean yaw rate as a fraction of commanded yaw rate when command_wz is non-zero.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

from real_lite_lab import register_tasks
from real_lite_lab.cli_args import apply_headless_env_cfg_overrides, update_rsl_rl_cfg
from real_lite_lab.isaaclab_compat import get_checkpoint_path
from real_lite_lab.motion_files import validate_motion_files
from real_lite_lab.task_registry import task_registry

_RUNNERS = {"OnPolicyRunner": OnPolicyRunner, "AmpOnPolicyRunner": AmpOnPolicyRunner}


def _configure_fixed_command(env_cfg) -> None:
    env_cfg.commands.resampling_time_range = (args_cli.duration_s, args_cli.duration_s)
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.heading_control_stiffness = 0.0
    env_cfg.commands.debug_vis = False
    env_cfg.commands.ranges.lin_vel_x = (args_cli.command_vx, args_cli.command_vx)
    env_cfg.commands.ranges.lin_vel_y = (args_cli.command_vy, args_cli.command_vy)
    env_cfg.commands.ranges.ang_vel_z = (args_cli.command_wz, args_cli.command_wz)
    env_cfg.commands.ranges.heading = (0.0, 0.0)


def _disable_reset_noise(env_cfg) -> None:
    reset_base = env_cfg.domain_rand.events.reset_base
    if reset_base is not None:
        reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    reset_robot_joints = env_cfg.domain_rand.events.reset_robot_joints
    if reset_robot_joints is not None:
        reset_robot_joints.params["position_range"] = (1.0, 1.0)
        reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


def _tilt_deg(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    gravity_xy = torch.linalg.norm(projected_gravity_b[:, :2], dim=1)
    gravity_z = torch.clamp(projected_gravity_b[:, 2].abs(), min=1.0e-6)
    return torch.atan2(gravity_xy, gravity_z) * (180.0 / math.pi)


def _bad_contact_force(env) -> torch.Tensor:
    forces = env.contact_sensor.data.net_forces_w_history[:, :, env.termination_contact_cfg.body_ids, :]
    return torch.linalg.norm(forces, dim=-1).amax(dim=(1, 2))


def _policy_torque_abs_max(env) -> torch.Tensor:
    applied_torque = getattr(env.robot.data, "applied_torque", None)
    if applied_torque is None:
        return torch.zeros(env.num_envs, device=env.device)
    return applied_torque[:, env.policy_joint_ids].abs().amax(dim=1)


def _print_shutdown(message: str) -> None:
    print(f"[SHUTDOWN] {message}", flush=True)


def _run_shutdown_step(step_name: str, callback) -> None:
    if not callable(callback):
        _print_shutdown(f"Skipping {step_name}: not available.")
        return

    _print_shutdown(f"Starting {step_name}")
    callback()
    _print_shutdown(f"Completed {step_name}")


def _mean_or_zero(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.detach().mean().cpu().item())


def _max_or_zero(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.detach().max().cpu().item())


def main() -> None:
    if args_cli.duration_s <= 0.0:
        raise ValueError("--duration_s must be positive.")
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")
    if args_cli.min_linear_speed_ratio < 0.0:
        raise ValueError("--min_linear_speed_ratio must be non-negative.")
    if args_cli.min_linear_progress_ratio < 0.0:
        raise ValueError("--min_linear_progress_ratio must be non-negative.")
    if args_cli.max_lateral_drift_ratio < 0.0:
        raise ValueError("--max_lateral_drift_ratio must be non-negative.")
    if args_cli.max_lateral_drift_abs < 0.0:
        raise ValueError("--max_lateral_drift_abs must be non-negative.")
    if args_cli.min_yaw_rate_ratio < 0.0:
        raise ValueError("--min_yaw_rate_ratio must be non-negative.")
    if args_cli.task not in {"walk_real_lite", "walk_forward_real_lite", "walk_gmr_forward_real_lite", "run_real_lite"}:
        raise ValueError(
            "eval_walk_real_lite.py expects walk_real_lite, walk_forward_real_lite, "
            "walk_gmr_forward_real_lite, or run_real_lite."
        )

    register_tasks()
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_class = task_registry.get_task_class(args_cli.task)

    env_cfg.scene.num_envs = args_cli.num_envs
    # Keep the env from resetting on the final evaluation frame, so final displacement is measured pre-reset.
    env_cfg.scene.max_episode_length_s = args_cli.duration_s + 1.0
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.height_scanner.enable_height_scan = False
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    if not args_cli.enable_terrain:
        env_cfg.scene.terrain_type = "plane"
        env_cfg.scene.terrain_generator = None
        env_cfg.scene.max_init_terrain_level = 0
    if not args_cli.enable_noise:
        env_cfg.noise.add_noise = False
    if not args_cli.keep_domain_rand:
        env_cfg.domain_rand.events.physics_material = None
        env_cfg.domain_rand.events.add_base_mass = None
        env_cfg.domain_rand.events.push_robot = None
    if not args_cli.keep_reset_noise:
        _disable_reset_noise(env_cfg)
    _configure_fixed_command(env_cfg)

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg = apply_headless_env_cfg_overrides(env_cfg, args_cli.headless)
    env_cfg.scene.seed = agent_cfg.seed
    validate_motion_files(
        task_name=args_cli.task,
        display_motion_files=getattr(env_cfg, "amp_motion_files_display", None),
        amp_motion_files=getattr(agent_cfg, "amp_motion_files", None),
    )

    env = None
    runner = None
    try:
        env = env_class(env_cfg, args_cli.headless)
        log_root_path = os.path.abspath(os.path.join(PIPELINE_DIR, "logs", agent_cfg.experiment_name))
        checkpoint_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")

        runner_class = _RUNNERS[agent_cfg.runner_class_name]
        runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint_path, load_optimizer=False)
        policy = runner.get_inference_policy(device=agent_cfg.device)

        obs, _ = env.get_observations()
        max_steps = math.ceil(args_cli.duration_s / env.step_dt)
        target_duration = max_steps * env.step_dt
        first_reset_time = torch.full((env.num_envs,), target_duration, device=env.device)
        first_reset_was_timeout = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        max_tilt = torch.zeros(env.num_envs, device=env.device)
        min_root_z = torch.full((env.num_envs,), float("inf"), device=env.device)
        max_root_xy_drift = torch.zeros(env.num_envs, device=env.device)
        max_bad_contact = torch.zeros(env.num_envs, device=env.device)
        max_torque = torch.zeros(env.num_envs, device=env.device)
        vx_error_sum = torch.zeros(env.num_envs, device=env.device)
        vy_error_sum = torch.zeros(env.num_envs, device=env.device)
        wz_error_sum = torch.zeros(env.num_envs, device=env.device)
        vx_sum = torch.zeros(env.num_envs, device=env.device)
        vy_sum = torch.zeros(env.num_envs, device=env.device)
        wz_sum = torch.zeros(env.num_envs, device=env.device)
        world_vx_sum = torch.zeros(env.num_envs, device=env.device)
        world_vy_sum = torch.zeros(env.num_envs, device=env.device)
        metric_samples = torch.zeros(env.num_envs, device=env.device)
        last_root_delta = torch.zeros(env.num_envs, 2, device=env.device)
        trace = {
            "time": [],
            "root_pos": [],
            "root_quat_wxyz": [],
            "joint_pos_policy": [],
        }

        def append_trace(time_s: float) -> None:
            if args_cli.trace_out is None:
                return
            trace["time"].append(float(time_s))
            trace["root_pos"].append(env.robot.data.root_pos_w[0].detach().cpu().numpy().copy())
            trace["root_quat_wxyz"].append(env.robot.data.root_quat_w[0].detach().cpu().numpy().copy())
            trace["joint_pos_policy"].append(
                env.robot.data.joint_pos[0, env.policy_joint_ids].detach().cpu().numpy().copy()
            )

        append_trace(0.0)

        with torch.inference_mode():
            for step_idx in range(max_steps):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                elapsed_s = min((step_idx + 1) * env.step_dt, target_duration)
                append_trace(elapsed_s)

                just_done = dones & alive
                if torch.any(just_done):
                    first_reset_time[just_done] = elapsed_s
                    first_reset_was_timeout[just_done] = env.time_out_buf[just_done]
                    alive[just_done] = False

                if torch.any(alive):
                    root_pos = env.robot.data.root_pos_w
                    root_delta = root_pos[:, :2] - env.scene.env_origins[:, :2]
                    root_xy_drift = torch.linalg.norm(root_delta, dim=1)
                    command = env.command_generator.command
                    lin_vel_b = env.robot.data.root_lin_vel_b
                    lin_vel_w = env.robot.data.root_lin_vel_w
                    ang_vel_w = env.robot.data.root_ang_vel_w

                    last_root_delta[alive] = root_delta[alive]
                    max_tilt[alive] = torch.maximum(max_tilt[alive], _tilt_deg(env.robot.data.projected_gravity_b)[alive])
                    min_root_z[alive] = torch.minimum(min_root_z[alive], root_pos[:, 2][alive])
                    max_root_xy_drift[alive] = torch.maximum(max_root_xy_drift[alive], root_xy_drift[alive])
                    max_bad_contact[alive] = torch.maximum(max_bad_contact[alive], _bad_contact_force(env)[alive])
                    max_torque[alive] = torch.maximum(max_torque[alive], _policy_torque_abs_max(env)[alive])
                    vx_error_sum[alive] += torch.abs(command[:, 0] - lin_vel_b[:, 0])[alive]
                    vy_error_sum[alive] += torch.abs(command[:, 1] - lin_vel_b[:, 1])[alive]
                    wz_error_sum[alive] += torch.abs(command[:, 2] - ang_vel_w[:, 2])[alive]
                    vx_sum[alive] += lin_vel_b[:, 0][alive]
                    vy_sum[alive] += lin_vel_b[:, 1][alive]
                    wz_sum[alive] += ang_vel_w[:, 2][alive]
                    world_vx_sum[alive] += lin_vel_w[:, 0][alive]
                    world_vy_sum[alive] += lin_vel_w[:, 1][alive]
                    metric_samples[alive] += 1.0

                if not torch.any(alive):
                    break

        survived_full = first_reset_time >= target_duration - 1.0e-6
        clean_timeout = survived_full & first_reset_was_timeout
        failed = ~clean_timeout
        samples = torch.clamp(metric_samples, min=1.0)
        failed_times = first_reset_time[failed]
        first_failure = float(failed_times.min().cpu().item()) if failed_times.numel() else target_duration
        mean_body_vx = vx_sum / samples
        mean_body_vy = vy_sum / samples
        mean_wz = wz_sum / samples
        mean_world_vx = world_vx_sum / samples
        mean_world_vy = world_vy_sum / samples

        command_xy = torch.tensor([args_cli.command_vx, args_cli.command_vy], dtype=torch.float32, device=env.device)
        command_speed = float(torch.linalg.norm(command_xy).detach().cpu().item())
        expected_progress = command_speed * target_duration
        has_linear_command = command_speed > 1.0e-6
        linear_speed_ok = True
        linear_progress_ok = True
        lateral_drift_ok = True
        linear_speed_ratio = 1.0
        linear_progress_ratio = 1.0
        mean_world_velocity_along_command = 0.0
        mean_progress_along_command = 0.0
        mean_lateral_drift = 0.0
        lateral_drift_limit = max(args_cli.max_lateral_drift_abs, args_cli.max_lateral_drift_ratio * expected_progress)

        if has_linear_command:
            command_dir = command_xy / command_speed
            progress_along_command = last_root_delta @ command_dir
            lateral_direction = torch.stack((-command_dir[1], command_dir[0]))
            lateral_drift = last_root_delta @ lateral_direction
            world_velocity_along_command = mean_world_vx * command_dir[0] + mean_world_vy * command_dir[1]
            mean_progress_along_command = _mean_or_zero(progress_along_command)
            mean_lateral_drift = _mean_or_zero(torch.abs(lateral_drift))
            mean_world_velocity_along_command = _mean_or_zero(world_velocity_along_command)
            linear_speed_ratio = mean_world_velocity_along_command / command_speed
            linear_progress_ratio = mean_progress_along_command / max(expected_progress, 1.0e-6)
            linear_speed_ok = linear_speed_ratio >= args_cli.min_linear_speed_ratio
            linear_progress_ok = linear_progress_ratio >= args_cli.min_linear_progress_ratio
            lateral_drift_ok = mean_lateral_drift <= lateral_drift_limit

        has_yaw_command = abs(args_cli.command_wz) > 1.0e-6
        yaw_rate_ok = True
        yaw_rate_ratio = 1.0
        mean_yaw_rate = _mean_or_zero(mean_wz)
        if has_yaw_command:
            yaw_rate_ratio = mean_yaw_rate / args_cli.command_wz
            yaw_rate_ok = yaw_rate_ratio >= args_cli.min_yaw_rate_ratio

        survival_ok = not failed.any()
        walking_ok = survival_ok and linear_speed_ok and linear_progress_ok and lateral_drift_ok and yaw_rate_ok

        print("[RESULT] walk policy evaluation")
        print(f"[RESULT] checkpoint={checkpoint_path}")
        print(f"[RESULT] num_envs={env.num_envs}, duration_s={target_duration:.3f}, step_dt={env.step_dt:.5f}")
        print(
            f"[RESULT] command=(vx={args_cli.command_vx:.3f}, vy={args_cli.command_vy:.3f}, "
            f"wz={args_cli.command_wz:.3f}), terrain={args_cli.enable_terrain}, "
            f"reset_noise={args_cli.keep_reset_noise}, obs_noise={args_cli.enable_noise}, "
            f"domain_rand={args_cli.keep_domain_rand}"
        )
        print(
            f"[RESULT] survived_full={int(clean_timeout.sum().cpu().item())}/{env.num_envs}, "
            f"success_rate={_mean_or_zero(clean_timeout.float()):.3f}, first_failure_s={first_failure:.3f}"
        )
        print(
            f"[RESULT] survival_time_s: mean={_mean_or_zero(first_reset_time):.3f}, "
            f"min={float(first_reset_time.min().cpu().item()):.3f}, "
            f"max={float(first_reset_time.max().cpu().item()):.3f}"
        )
        print(
            f"[RESULT] tracking_abs_error_mean: vx={_mean_or_zero(vx_error_sum / samples):.3f}, "
            f"vy={_mean_or_zero(vy_error_sum / samples):.3f}, "
            f"wz={_mean_or_zero(wz_error_sum / samples):.3f}"
        )
        print(
            f"[RESULT] measured_velocity_mean: vx={_mean_or_zero(mean_body_vx):.3f}, "
            f"vy={_mean_or_zero(mean_body_vy):.3f}, "
            f"wz={mean_yaw_rate:.3f}"
        )
        print(
            f"[RESULT] measured_world_velocity_mean: vx={_mean_or_zero(mean_world_vx):.3f}, "
            f"vy={_mean_or_zero(mean_world_vy):.3f}"
        )
        print(
            f"[RESULT] final_displacement: x_mean={_mean_or_zero(last_root_delta[:, 0]):.3f}, "
            f"x_min={float(last_root_delta[:, 0].min().cpu().item()):.3f}, "
            f"x_max={float(last_root_delta[:, 0].max().cpu().item()):.3f}, "
            f"abs_y_mean={_mean_or_zero(torch.abs(last_root_delta[:, 1])):.3f}"
        )
        print(
            f"[RESULT] alive_env_metrics: max_tilt_deg={_max_or_zero(max_tilt):.2f}, "
            f"min_root_z={float(min_root_z.min().cpu().item()):.4f}, "
            f"max_root_xy_drift={_max_or_zero(max_root_xy_drift):.4f}, "
            f"max_bad_contact_force={_max_or_zero(max_bad_contact):.2f}, "
            f"max_policy_torque={_max_or_zero(max_torque):.2f}"
        )
        if has_linear_command:
            print(
                f"[RESULT] linear_progress: command_speed={command_speed:.3f}, "
                f"expected={expected_progress:.3f}m, along_mean={mean_progress_along_command:.3f}m, "
                f"progress_ratio={linear_progress_ratio:.3f}, world_speed_along={mean_world_velocity_along_command:.3f}, "
                f"speed_ratio={linear_speed_ratio:.3f}, lateral_abs_mean={mean_lateral_drift:.3f}m, "
                f"lateral_limit={lateral_drift_limit:.3f}m"
            )
        else:
            print("[RESULT] linear_progress: skipped because commanded linear velocity is zero.")
        if has_yaw_command:
            print(
                f"[RESULT] yaw_tracking: command_wz={args_cli.command_wz:.3f}, "
                f"measured_wz={mean_yaw_rate:.3f}, yaw_rate_ratio={yaw_rate_ratio:.3f}"
            )
        print(
            "[RESULT] walking_checks: "
            f"survival={'PASS' if survival_ok else 'FAIL'}, "
            f"linear_speed={'PASS' if linear_speed_ok else 'FAIL'}, "
            f"linear_progress={'PASS' if linear_progress_ok else 'FAIL'}, "
            f"lateral_drift={'PASS' if lateral_drift_ok else 'FAIL'}, "
            f"yaw_rate={'PASS' if yaw_rate_ok else 'FAIL'}"
        )
        print(f"[RESULT] walking_verdict={'PASS' if walking_ok else 'FAIL'}")
        if args_cli.trace_out is not None:
            trace_path = Path(args_cli.trace_out)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                trace_path,
                time=np.asarray(trace["time"], dtype=np.float64),
                root_pos=np.asarray(trace["root_pos"], dtype=np.float64),
                root_quat_wxyz=np.asarray(trace["root_quat_wxyz"], dtype=np.float64),
                joint_pos_policy=np.asarray(trace["joint_pos_policy"], dtype=np.float64),
                policy_joint_names=np.asarray(env.policy_joint_names),
                command=np.asarray([args_cli.command_vx, args_cli.command_vy, args_cli.command_wz], dtype=np.float64),
            )
            print(f"[RESULT] trace_out={trace_path}")
        if not walking_ok:
            print("[RESULT] verdict=FAIL")
            raise RuntimeError("Walk policy did not satisfy the walking evaluation criteria.")
        print("[RESULT] verdict=PASS")
    finally:
        if runner is not None:
            close_runner = getattr(runner, "close", None)
            _run_shutdown_step("runner.close()", close_runner)
        if env is not None:
            close_env = getattr(env, "close", None)
            _run_shutdown_step("env.close()", close_env)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        _print_shutdown("Starting simulation_app.close()")
        simulation_app.close()
        _print_shutdown("Completed simulation_app.close()")
    sys.exit(exit_code)
