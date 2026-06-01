import argparse
import os
import sys
from datetime import datetime
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


parser = argparse.ArgumentParser(description="Train Real Lite RL tasks.")
parser.add_argument("--task", type=str, required=True, choices=TASK_NAMES)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

from real_lite_lab import register_tasks, task_registry
from real_lite_lab.cli_args import apply_headless_env_cfg_overrides, update_rsl_rl_cfg
from real_lite_lab.isaaclab_compat import dump_yaml, get_checkpoint_path

_RUNNERS = {"OnPolicyRunner": OnPolicyRunner, "AmpOnPolicyRunner": AmpOnPolicyRunner}

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


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
    env_class = task_registry.get_task_class(args_cli.task)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg = apply_headless_env_cfg_overrides(env_cfg, args_cli.headless)
    env_cfg.scene.seed = agent_cfg.seed

    env = None
    runner = None
    try:
        env = env_class(env_cfg, args_cli.headless)

        log_root_path = os.path.abspath(os.path.join(PIPELINE_DIR, "logs", agent_cfg.experiment_name))
        print(f"[INFO] Logging experiment in directory: {log_root_path}")

        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        if agent_cfg.runner_class_name not in _RUNNERS:
            raise ValueError(
                f"Unknown runner_class_name: {agent_cfg.runner_class_name!r}, expected one of {list(_RUNNERS)}"
            )
        runner_class = _RUNNERS[agent_cfg.runner_class_name]
        runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

        if agent_cfg.resume:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        _print_shutdown("runner.learn() returned.")
    finally:
        if runner is not None:
            close_runner = getattr(runner, "close", None)
            _run_shutdown_step("runner.close()", close_runner)
        if env is not None:
            close_env = getattr(env, "close", None)
            _run_shutdown_step("env.close()", close_env)


if __name__ == "__main__":
    try:
        main()
    finally:
        _print_shutdown("Starting simulation_app.close()")
        simulation_app.close()
        _print_shutdown("Completed simulation_app.close()")
