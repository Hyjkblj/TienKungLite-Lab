import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# local imports
import real_lite_lab.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize AMP motion and export expert data.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--save_path", type=str, default=None, help="Path to save the txt file")
parser.add_argument("--fps", type=float, default=30.0, help="Target fps")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if args_cli.task and "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from real_lite_lab import register_tasks, task_registry
from real_lite_lab.cli_args import update_rsl_rl_cfg

register_tasks()


def _print_shutdown(message: str) -> None:
    print(f"[SHUTDOWN] {message}", flush=True)


def _write_amp_expert_motion(save_path: str, frames: list[np.ndarray], fps: float) -> None:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "FrameType": "amp_expert",
        "LoopMode": "Wrap",
        "FrameDuration": round(1.0 / fps, 3),
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": 0.5,
        "Frames": np.stack(frames, axis=0).tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Successfully saved AMP expert data to {save_path}")


def play_amp_animation():
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"
    env_cfg.commands.debug_vis = False

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = None
    try:
        env = env_class(env_cfg, args_cli.headless)

        frame_cnt = 0
        all_frames = []
        while simulation_app.is_running():
            while True:
                time = (frame_cnt % env.motion_len) * (1.0 / args_cli.fps)
                frame = env.visualize_motion(time)
                if args_cli.save_path:
                    all_frames.append(frame.cpu().numpy().reshape(-1))
                frame_cnt += 1
                if frame_cnt >= (env.motion_len - 1):
                    break
            break

        if args_cli.save_path:
            _write_amp_expert_motion(args_cli.save_path, all_frames, args_cli.fps)
    finally:
        if env is not None:
            _print_shutdown("Starting env.close()")
            env.close()
            _print_shutdown("Completed env.close()")


if __name__ == "__main__":
    exit_code = 0
    try:
        play_amp_animation()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        _print_shutdown("Starting simulation_app.close()")
        simulation_app.close()
        _print_shutdown("Completed simulation_app.close()")
    sys.exit(exit_code)
