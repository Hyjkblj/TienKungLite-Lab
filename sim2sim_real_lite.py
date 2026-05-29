import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco_viewer
import numpy as np
import torch

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from real_lite_lab.constants import (
    DEFAULT_DOF_POS,
    MJCF_DIR,
    OBS_PER_STEP_DIM,
    POLICY_JOINT_COUNT,
    POLICY_JOINT_NAMES,
    TASK_NAMES,
    TASK_PRESETS,
)


MJCF_PATH = MJCF_DIR / "real_lite.xml"
ISAAC_POLICY_ORDER = POLICY_JOINT_NAMES
MUJOCO_SENSOR_ORDER = POLICY_JOINT_NAMES
# If these differ, mujoco_to_isaac_idx / isaac_to_mujoco_idx become non-trivial index remappings.
assert ISAAC_POLICY_ORDER == MUJOCO_SENSOR_ORDER, (
    f"Joint order mismatch: Isaac and MuJoCo must agree, got {ISAAC_POLICY_ORDER} vs {MUJOCO_SENSOR_ORDER}"
)


class RealLiteSim2SimCfg:
    class sim:
        sim_duration = 100.0
        num_action = POLICY_JOINT_COUNT
        num_obs_per_step = OBS_PER_STEP_DIM
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25
        enable_keyboard_commands = True

    class robot:
        gait_air_ratio_l = TASK_PRESETS["walk_real_lite"]["gait_air_ratio_l"]
        gait_air_ratio_r = TASK_PRESETS["walk_real_lite"]["gait_air_ratio_r"]
        gait_phase_offset_l = TASK_PRESETS["walk_real_lite"]["gait_phase_offset_l"]
        gait_phase_offset_r = TASK_PRESETS["walk_real_lite"]["gait_phase_offset_r"]
        gait_cycle = TASK_PRESETS["walk_real_lite"]["gait_cycle"]


class RealLiteMujocoRunner:
    def __init__(self, cfg: RealLiteSim2SimCfg, policy_path: Path, model_path: Path):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = self.cfg.sim.dt
        self.policy = torch.jit.load(str(policy_path))
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False
        self.init_variables()

    def init_variables(self):
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)
        self.default_dof_pos = np.array(DEFAULT_DOF_POS, dtype=np.float64)
        if self.default_dof_pos.shape[0] != self.cfg.sim.num_action:
            raise ValueError(
                f"default_dof_pos size mismatch: expected {self.cfg.sim.num_action}, got {self.default_dof_pos.shape[0]}."
            )
        self.episode_length_buf = 0
        self.gait_phase = np.zeros(2)
        self.gait_cycle = self.cfg.robot.gait_cycle
        self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r])
        self.phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r])
        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )
        self.sensor_joint_name_to_idx = {name: idx for idx, name in enumerate(MUJOCO_SENSOR_ORDER)}
        self.policy_joint_name_to_idx = {name: idx for idx, name in enumerate(ISAAC_POLICY_ORDER)}
        self.mujoco_to_isaac_idx = [self.sensor_joint_name_to_idx[name] for name in ISAAC_POLICY_ORDER]
        self.isaac_to_mujoco_idx = [self.policy_joint_name_to_idx[name] for name in MUJOCO_SENSOR_ORDER]

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a - b + c

    def calculate_gait_para(self):
        t = self.episode_length_buf * self.dt / self.gait_cycle
        self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
        self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0

    def get_obs(self) -> np.ndarray:
        n = self.cfg.sim.num_action
        self.dof_pos = self.data.sensordata[0:n]
        self.dof_vel = self.data.sensordata[n:2*n]
        obs = np.concatenate(
            [
                self.data.sensor("angular-velocity").data.astype(np.double),
                self.quat_rotate_inverse(
                    self.data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double),
                    np.array([0, 0, -1], dtype=np.double),
                ),
                self.command_vel,
                (self.dof_pos - self.default_dof_pos)[self.mujoco_to_isaac_idx],
                self.dof_vel[self.mujoco_to_isaac_idx],
                np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions),
                np.sin(2 * np.pi * self.gait_phase),
                np.cos(2 * np.pi * self.gait_phase),
                self.phase_ratio,
            ],
            axis=0,
        ).astype(np.float32)
        if obs.shape[0] != self.cfg.sim.num_obs_per_step:
            raise ValueError(
                f"Observation size mismatch: expected {self.cfg.sim.num_obs_per_step}, got {obs.shape[0]}."
            )
        self.obs_history[:-self.cfg.sim.num_obs_per_step] = self.obs_history[self.cfg.sim.num_obs_per_step :]
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs.copy()
        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def position_control(self) -> np.ndarray:
        actions_scaled = self.action * self.cfg.sim.action_scale
        return actions_scaled[self.isaac_to_mujoco_idx] + self.default_dof_pos

    def adjust_command_vel(self, idx: int, increment: float):
        self.command_vel[idx] += increment
        self.command_vel[idx] = np.clip(self.command_vel[idx], -1.0, 1.0)

    def setup_keyboard_listener(self):
        if not self.cfg.sim.enable_keyboard_commands:
            self.listener = None
            return

        try:
            from pynput import keyboard
        except ImportError:
            self.listener = None
            return

        def on_press(key):
            try:
                if key.char == "8":
                    self.adjust_command_vel(0, 0.2)
                elif key.char == "2":
                    self.adjust_command_vel(0, -0.2)
                elif key.char == "4":
                    self.adjust_command_vel(1, -0.2)
                elif key.char == "6":
                    self.adjust_command_vel(1, 0.2)
                elif key.char == "7":
                    self.adjust_command_vel(2, -0.2)
                elif key.char == "9":
                    self.adjust_command_vel(2, 0.2)
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)

    def run(self):
        self.setup_keyboard_listener()
        if self.listener is not None:
            self.listener.start()
        while self.data.time < self.cfg.sim.sim_duration:
            self.obs_history = self.get_obs()
            action_tensor = self.policy(torch.tensor(self.obs_history, dtype=torch.float32))
            self.action[:] = action_tensor.detach().cpu().numpy()[:self.cfg.sim.num_action]
            self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)

            for _ in range(self.cfg.sim.decimation):
                step_start_time = time.time()
                self.data.ctrl = self.position_control()
                mujoco.mj_step(self.model, self.data)
                self.viewer.render()
                elapsed = time.time() - step_start_time
                sleep_time = self.cfg.sim.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.episode_length_buf += 1
            self.calculate_gait_para()

        if self.listener is not None:
            self.listener.stop()
        self.viewer.close()


def build_cfg(task_name: str, duration: float) -> RealLiteSim2SimCfg:
    cfg = RealLiteSim2SimCfg()
    cfg.sim.sim_duration = duration
    cfg.sim.enable_keyboard_commands = task_name != "upper_body_real_lite"
    preset = TASK_PRESETS[task_name]
    cfg.robot.gait_air_ratio_l = preset["gait_air_ratio_l"]
    cfg.robot.gait_air_ratio_r = preset["gait_air_ratio_r"]
    cfg.robot.gait_phase_offset_l = preset["gait_phase_offset_l"]
    cfg.robot.gait_phase_offset_r = preset["gait_phase_offset_r"]
    cfg.robot.gait_cycle = preset["gait_cycle"]
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run Real Lite sim2sim Mujoco controller.")
    parser.add_argument("--task", required=True, choices=TASK_NAMES)
    parser.add_argument("--policy", required=True, help="Path to exported policy.pt")
    parser.add_argument("--model", default=str(MJCF_PATH), help="Path to Real Lite MuJoCo XML")
    parser.add_argument("--duration", type=float, default=100.0)
    args = parser.parse_args()

    policy_path = Path(args.policy).resolve()
    model_path = Path(args.model).resolve()

    if not policy_path.is_file():
        print(f"[ERROR] Policy file not found: {policy_path}")
        sys.exit(1)
    if not model_path.is_file():
        print(f"[ERROR] MuJoCo model file not found: {model_path}")
        print("[INFO] Run generate_real_lite_mjcf.py first.")
        sys.exit(1)

    cfg = build_cfg(args.task, args.duration)
    runner = RealLiteMujocoRunner(cfg=cfg, policy_path=policy_path, model_path=model_path)
    runner.run()


if __name__ == "__main__":
    main()
