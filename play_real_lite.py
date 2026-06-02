import argparse
import os
import sys
import traceback
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import real_lite_lab.cli_args as cli_args
from real_lite_lab.constants import TASK_NAMES


parser = argparse.ArgumentParser(description="Export Real Lite policies.")
parser.add_argument("--task", type=str, required=True, choices=TASK_NAMES)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

from real_lite_lab import register_tasks
from real_lite_lab.cli_args import update_rsl_rl_cfg
from real_lite_lab.isaaclab_compat import export_policy_as_jit, export_policy_as_onnx, get_checkpoint_path
from real_lite_lab.task_registry import task_registry

_RUNNERS = {"OnPolicyRunner": OnPolicyRunner, "AmpOnPolicyRunner": AmpOnPolicyRunner}


def _print_shutdown(message: str) -> None:
    print(f"[SHUTDOWN] {message}", flush=True)


def _run_shutdown_step(step_name: str, callback) -> None:
    if not callable(callback):
        _print_shutdown(f"Skipping {step_name}: not available.")
        return

    _print_shutdown(f"Starting {step_name}")
    callback()
    _print_shutdown(f"Completed {step_name}")


def main():
    register_tasks()
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    if args_cli.task == "upper_body_real_lite":
        env_cfg.commands.rel_standing_envs = 1.0
        env_cfg.commands.rel_heading_envs = 0.0
        env_cfg.commands.heading_command = False
        env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
        env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    else:
        env_cfg.commands.rel_standing_envs = 0.0
        env_cfg.commands.ranges.lin_vel_x = (1.0, 1.0)
        env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(args_cli.task)
    env = None
    runner = None
    try:
        env = env_class(env_cfg, args_cli.headless)

        log_root_path = os.path.abspath(os.path.join(PIPELINE_DIR, "logs", agent_cfg.experiment_name))
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)

        if agent_cfg.runner_class_name not in _RUNNERS:
            raise ValueError(
                f"Unknown runner_class_name: {agent_cfg.runner_class_name!r}, expected one of {list(_RUNNERS)}"
            )
        runner_class = _RUNNERS[agent_cfg.runner_class_name]
        runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner.load(resume_path, load_optimizer=False)

        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )
        print(f"[INFO] Exported policy to: {export_model_dir}")
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
