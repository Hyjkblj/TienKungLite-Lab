import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import torch
from isaaclab.app import AppLauncher
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

_RUNNERS = {"OnPolicyRunner": OnPolicyRunner, "AmpOnPolicyRunner": AmpOnPolicyRunner}

from real_lite_lab import register_tasks, task_registry
from real_lite_lab import cli_args
from real_lite_lab.cli_args import update_rsl_rl_cfg


parser = argparse.ArgumentParser(description="Train Real Lite RL tasks.")
parser.add_argument("--task", type=str, required=True, choices=["walk_real_lite", "run_real_lite"])
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    register_tasks()
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_class = task_registry.get_task_class(args_cli.task)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.abspath(os.path.join(PIPELINE_DIR, "logs", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if agent_cfg.runner_class_name not in _RUNNERS:
        raise ValueError(f"Unknown runner_class_name: {agent_cfg.runner_class_name!r}, expected one of {list(_RUNNERS)}")
    runner_class = _RUNNERS[agent_cfg.runner_class_name]
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
