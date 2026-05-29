import argparse
import os
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import torch
from isaaclab.app import AppLauncher
from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

_RUNNERS = {"OnPolicyRunner": OnPolicyRunner, "AmpOnPolicyRunner": AmpOnPolicyRunner}

from real_lite_lab import register_tasks, task_registry
from real_lite_lab import cli_args
from real_lite_lab.cli_args import update_rsl_rl_cfg
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
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.abspath(os.path.join(PIPELINE_DIR, "logs", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    if agent_cfg.runner_class_name not in _RUNNERS:
        raise ValueError(f"Unknown runner_class_name: {agent_cfg.runner_class_name!r}, expected one of {list(_RUNNERS)}")
    runner_class = _RUNNERS[agent_cfg.runner_class_name]
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )
    print(f"[INFO] Exported policy to: {export_model_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
