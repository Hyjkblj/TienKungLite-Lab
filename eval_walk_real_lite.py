import argparse
import math
import os
import sys
import traceback
from pathlib import Path

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
    if args_cli.task not in {"walk_real_lite", "run_real_lite"}:
        raise ValueError("eval_walk_real_lite.py expects walk_real_lite or run_real_lite.")

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

        with torch.inference_mode():
            for step_idx in range(max_steps):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                elapsed_s = min((step_idx + 1) * env.step_dt, target_duration)

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
            f"[RESULT] measured_velocity_mean: vx={_mean_or_zero(vx_sum / samples):.3f}, "
            f"vy={_mean_or_zero(vy_sum / samples):.3f}, "
            f"wz={_mean_or_zero(wz_sum / samples):.3f}"
        )
        print(
            f"[RESULT] measured_world_velocity_mean: vx={_mean_or_zero(world_vx_sum / samples):.3f}, "
            f"vy={_mean_or_zero(world_vy_sum / samples):.3f}"
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
        if failed.any():
            print("[RESULT] verdict=FAIL")
            raise RuntimeError("Walk policy did not survive the full evaluation duration in every environment.")
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
