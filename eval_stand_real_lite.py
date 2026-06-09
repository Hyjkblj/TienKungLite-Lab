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


parser = argparse.ArgumentParser(description="Evaluate a trained Real Lite stand policy.")
parser.add_argument("--task", type=str, default="stand_real_lite", choices={"stand_real_lite"})
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--duration_s", type=float, default=30.0)
parser.add_argument(
    "--keep_reset_noise",
    action="store_true",
    help="Keep the stand task's reset pose/joint randomization for a robustness check.",
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


def _configure_zero_command(env_cfg) -> None:
    env_cfg.commands.rel_standing_envs = 1.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.heading_control_stiffness = 0.0
    env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
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


def _foot_force_total(env) -> torch.Tensor:
    forces = env.contact_sensor.data.net_forces_w[:, env.feet_cfg.body_ids, :]
    return torch.linalg.norm(forces, dim=-1).sum(dim=1)


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


def main() -> None:
    if args_cli.duration_s <= 0.0:
        raise ValueError("--duration_s must be positive.")
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")

    register_tasks()
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_class = task_registry.get_task_class(args_cli.task)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.max_episode_length_s = args_cli.duration_s
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain_type = "plane"
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.max_init_terrain_level = 0
    env_cfg.scene.height_scanner.enable_height_scan = False
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    _configure_zero_command(env_cfg)
    if not args_cli.keep_reset_noise:
        _disable_reset_noise(env_cfg)

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
        min_foot_force = torch.full((env.num_envs,), float("inf"), device=env.device)
        max_torque = torch.zeros(env.num_envs, device=env.device)

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
                    root_xy_drift = torch.linalg.norm(root_pos[:, :2] - env.scene.env_origins[:, :2], dim=1)
                    max_tilt[alive] = torch.maximum(max_tilt[alive], _tilt_deg(env.robot.data.projected_gravity_b)[alive])
                    min_root_z[alive] = torch.minimum(min_root_z[alive], root_pos[:, 2][alive])
                    max_root_xy_drift[alive] = torch.maximum(max_root_xy_drift[alive], root_xy_drift[alive])
                    max_bad_contact[alive] = torch.maximum(max_bad_contact[alive], _bad_contact_force(env)[alive])
                    min_foot_force[alive] = torch.minimum(min_foot_force[alive], _foot_force_total(env)[alive])
                    max_torque[alive] = torch.maximum(max_torque[alive], _policy_torque_abs_max(env)[alive])

                if not torch.any(alive):
                    break

        survived_full = first_reset_time >= target_duration - 1.0e-6
        clean_timeout = survived_full & first_reset_was_timeout
        failed = ~clean_timeout

        def mean_value(values: torch.Tensor) -> float:
            return float(values.detach().mean().cpu().item())

        def min_value(values: torch.Tensor) -> float:
            return float(values.detach().min().cpu().item())

        def max_value(values: torch.Tensor) -> float:
            return float(values.detach().max().cpu().item())

        failed_times = first_reset_time[failed]
        first_failure = float(failed_times.min().cpu().item()) if failed_times.numel() else target_duration
        print("[RESULT] stand policy evaluation")
        print(f"[RESULT] checkpoint={checkpoint_path}")
        print(f"[RESULT] num_envs={env.num_envs}, duration_s={target_duration:.3f}, step_dt={env.step_dt:.5f}")
        print(f"[RESULT] keep_reset_noise={args_cli.keep_reset_noise}")
        print(
            f"[RESULT] survived_full={int(clean_timeout.sum().cpu().item())}/{env.num_envs}, "
            f"success_rate={mean_value(clean_timeout.float()):.3f}, first_failure_s={first_failure:.3f}"
        )
        print(
            f"[RESULT] survival_time_s: mean={mean_value(first_reset_time):.3f}, "
            f"min={min_value(first_reset_time):.3f}, max={max_value(first_reset_time):.3f}"
        )
        print(
            f"[RESULT] alive_env_metrics: max_tilt_deg={max_value(max_tilt):.2f}, "
            f"min_root_z={min_value(min_root_z):.4f}, max_root_xy_drift={max_value(max_root_xy_drift):.4f}, "
            f"min_foot_force_total={min_value(min_foot_force):.2f}, "
            f"max_bad_contact_force={max_value(max_bad_contact):.2f}, "
            f"max_policy_torque={max_value(max_torque):.2f}"
        )
        if failed.any():
            print("[RESULT] verdict=FAIL")
            raise RuntimeError("Stand policy did not survive the full evaluation duration in every environment.")
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
