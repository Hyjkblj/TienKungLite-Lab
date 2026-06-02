import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _cli_requests_offscreen(argv: list[str]) -> bool:
    for arg in argv:
        if arg == "--save_video" or arg.startswith("--save_video="):
            return True
    return False


if _cli_requests_offscreen(sys.argv[1:]):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    imageio = None

try:
    import mujoco_viewer
except ModuleNotFoundError:
    mujoco_viewer = None

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from real_lite_lab.constants import (
    DEFAULT_DOF_POS,
    LEFT_ARM_JOINT_NAMES,
    MJCF_DIR,
    OBS_PER_STEP_DIM,
    POLICY_JOINT_COUNT,
    POLICY_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    TASK_NAMES,
    TASK_PRESETS,
)
from real_lite_lab.joint_order import build_target_order_indices
from real_lite_lab.mjcf_mesh_fallback import build_mesh_safe_model, ensure_offscreen_framebuffer_size
from real_lite_lab.mujoco_state_init import apply_default_joint_state, snap_root_height_to_ground
from real_lite_lab.render_camera import camera_preset_names, get_camera_preset


MJCF_PATH = MJCF_DIR / "real_lite.xml"
ISAAC_POLICY_ORDER = POLICY_JOINT_NAMES
MUJOCO_SENSOR_ORDER = POLICY_JOINT_NAMES
# If these differ, mujoco_to_isaac_idx / isaac_to_mujoco_idx become non-trivial index remappings.
assert ISAAC_POLICY_ORDER == MUJOCO_SENSOR_ORDER, (
    f"Joint order mismatch: Isaac and MuJoCo must agree, got {ISAAC_POLICY_ORDER} vs {MUJOCO_SENSOR_ORDER}"
)


def _actuator_joint_names(model: mujoco.MjModel) -> list[str]:
    joint_names: list[str] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            raise ValueError(f"Unable to resolve joint name for actuator {actuator_id}.")
        joint_names.append(joint_name)
    return joint_names


def _model_camera_names(model: mujoco.MjModel) -> list[str]:
    camera_names: list[str] = []
    for camera_id in range(model.ncam):
        camera_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        if camera_name is not None:
            camera_names.append(camera_name)
    return camera_names


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


class _ImageioVideoSink:
    def __init__(self, output_path: Path, fps: float):
        if imageio is None:
            raise RuntimeError("imageio is not installed.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264")

    def write_frame(self, frame: np.ndarray) -> None:
        self._writer.append_data(frame)

    def close(self) -> None:
        self._writer.close()


class _FFmpegVideoSink:
    def __init__(self, output_path: Path, fps: float, width: int, height: int):
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg is not available on PATH.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write_frame(self, frame: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available.")
        self._process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        stderr_output = b""
        if self._process.stderr is not None:
            stderr_output = self._process.stderr.read()
        return_code = self._process.wait()
        if return_code != 0:
            stderr_text = stderr_output.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr_text}")


def _create_video_sink(output_path: Path, fps: float, width: int, height: int):
    errors: list[str] = []
    try:
        return _ImageioVideoSink(output_path=output_path, fps=fps)
    except Exception as exc:
        errors.append(f"imageio backend failed: {exc}")

    try:
        return _FFmpegVideoSink(output_path=output_path, fps=fps, width=width, height=height)
    except Exception as exc:
        errors.append(f"ffmpeg backend failed: {exc}")

    raise RuntimeError(
        "Unable to create an mp4 writer. Install imageio/ffmpeg or add ffmpeg to PATH.\n"
        + "\n".join(errors)
    )


class RealLiteMujocoRunner:
    def __init__(
        self,
        cfg: RealLiteSim2SimCfg,
        policy_path: Path | None,
        model_path: Path,
        *,
        save_video: Path | None = None,
        video_fps: float = 30.0,
        width: int = 1280,
        height: int = 720,
        camera: str | None = None,
        command_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
        control_mode: str = "policy",
        trace_path: Path | None = None,
        trace_steps: int = 0,
        lock_arms: bool = False,
    ):
        self.cfg = cfg
        self.model = self._load_model_with_mesh_fallback(model_path)
        self.model.opt.timestep = self.cfg.sim.dt
        self.control_mode = control_mode
        self.policy = torch.jit.load(str(policy_path)) if self.control_mode == "policy" and policy_path is not None else None
        self.data = mujoco.MjData(self.model)

        self.save_video = save_video
        self.video_fps = video_fps
        self.camera = camera
        self.camera_preset = get_camera_preset(camera)
        self.initial_command_vel = np.array(command_vel, dtype=np.float64)
        self.frame_interval = 1.0 / video_fps if save_video is not None else None
        self.next_frame_time = 0.0
        self.viewer = None
        self.renderer = None
        self.render_camera = None
        self.video_sink = None
        self.model_camera_names = tuple(_model_camera_names(self.model))
        self.trace_path = trace_path
        self.trace_steps = max(0, int(trace_steps))
        self.trace_records: list[dict[str, np.ndarray | float | int]] = []
        self.lock_arms = lock_arms

        if self.camera is not None and self.camera_preset is None and self.camera not in self.model_camera_names:
            available_model_cameras = ", ".join(self.model_camera_names) if self.model_camera_names else "none"
            raise ValueError(
                f"Unknown camera '{self.camera}'. "
                f"Available presets: {', '.join(camera_preset_names())}. "
                f"Available model cameras: {available_model_cameras}."
            )

        if self.save_video is None:
            if mujoco_viewer is None:
                raise RuntimeError(
                    "mujoco_viewer is not installed. Use --save_video for headless export, "
                    "or install mujoco_viewer for interactive visualization."
                )
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer._render_every_frame = False
        else:
            framebuffer_size = ensure_offscreen_framebuffer_size(self.model, width=width, height=height)
            if framebuffer_size is not None:
                print(f"[INFO] Offscreen framebuffer resized to: {framebuffer_size[0]}x{framebuffer_size[1]}")
            if self.camera_preset is not None:
                self.render_camera = mujoco.MjvCamera()
                self.render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                print(f"[INFO] Using camera preset: {self.camera}")
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
            self.video_sink = _create_video_sink(self.save_video, fps=video_fps, width=width, height=height)

        self.init_variables()
        self._initialize_sim_state()

    def _load_model_with_mesh_fallback(self, model_path: Path) -> mujoco.MjModel:
        try:
            return mujoco.MjModel.from_xml_path(str(model_path))
        except ValueError as exc:
            error_text = str(exc)
            if "decoder failed for mesh file" not in error_text:
                raise

            fallback_result = build_mesh_safe_model(model_path)
            if fallback_result is None:
                raise

            stripped_meshes = ", ".join(fallback_result.stripped_mesh_names)
            print(
                "[WARN] MuJoCo could not load one or more visual meshes. "
                f"Retrying with incompatible meshes removed: {stripped_meshes}"
            )
            print(f"[WARN] Using fallback model: {fallback_result.model_path}")
            return mujoco.MjModel.from_xml_path(str(fallback_result.model_path))

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
        self.command_vel = self.initial_command_vel.copy()
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )
        self.latest_obs_step = np.zeros(self.cfg.sim.num_obs_per_step, dtype=np.float32)
        self.actuator_joint_names = _actuator_joint_names(self.model)
        self.joint_pos_sensor_names, self.joint_pos_sensor_adr = self._joint_sensor_layout(mujoco.mjtSensor.mjSENS_JOINTPOS)
        self.joint_vel_sensor_names, self.joint_vel_sensor_adr = self._joint_sensor_layout(mujoco.mjtSensor.mjSENS_JOINTVEL)
        self.sensor_joint_name_to_idx = {name: idx for idx, name in enumerate(self.joint_pos_sensor_names)}
        self.vel_sensor_joint_name_to_idx = {name: idx for idx, name in enumerate(self.joint_vel_sensor_names)}
        self.mujoco_to_isaac_idx = [self.sensor_joint_name_to_idx[name] for name in ISAAC_POLICY_ORDER]
        self.mujoco_vel_to_isaac_idx = [self.vel_sensor_joint_name_to_idx[name] for name in ISAAC_POLICY_ORDER]

        if len(self.actuator_joint_names) != self.cfg.sim.num_action:
            raise ValueError(
                f"Actuator count mismatch: expected {self.cfg.sim.num_action}, got {len(self.actuator_joint_names)}."
            )
        self.isaac_to_mujoco_actuator_idx = build_target_order_indices(ISAAC_POLICY_ORDER, self.actuator_joint_names)
        policy_joint_name_to_idx = {name: idx for idx, name in enumerate(ISAAC_POLICY_ORDER)}
        self.arm_action_ids = np.array(
            [policy_joint_name_to_idx[name] for name in (*LEFT_ARM_JOINT_NAMES, *RIGHT_ARM_JOINT_NAMES)],
            dtype=np.int64,
        )

    def _joint_sensor_layout(self, sensor_type: int) -> tuple[list[str], np.ndarray]:
        joint_names: list[str] = []
        sensor_adr: list[int] = []
        for sensor_id in range(self.model.nsensor):
            if int(self.model.sensor_type[sensor_id]) != int(sensor_type):
                continue
            joint_id = int(self.model.sensor_objid[sensor_id])
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is None:
                raise ValueError(f"Unable to resolve joint name for sensor {sensor_id}.")
            joint_names.append(joint_name)
            sensor_adr.append(int(self.model.sensor_adr[sensor_id]))

        if len(joint_names) != self.cfg.sim.num_action:
            raise ValueError(
                f"Sensor count mismatch for type {sensor_type}: expected {self.cfg.sim.num_action}, got {len(joint_names)}."
            )
        return joint_names, np.asarray(sensor_adr, dtype=np.int32)

    def _initialize_sim_state(self) -> None:
        apply_default_joint_state(
            model=self.model,
            data=self.data,
            joint_names=ISAAC_POLICY_ORDER,
            default_joint_pos=self.default_dof_pos,
            joint_name_to_id=lambda joint_name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name),
        )
        mujoco.mj_forward(self.model, self.data)
        root_height_shift = snap_root_height_to_ground(model=self.model, data=self.data)
        if abs(root_height_shift) > 1e-5:
            print(f"[INFO] Adjusted initial root height by {root_height_shift:+.4f} m to place support geoms on the floor.")
        self.data.ctrl[:] = self.position_control()
        mujoco.mj_forward(self.model, self.data)

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
        self.dof_pos = np.asarray(self.data.sensordata[self.joint_pos_sensor_adr], dtype=np.float64)
        self.dof_vel = np.asarray(self.data.sensordata[self.joint_vel_sensor_adr], dtype=np.float64)
        obs = np.concatenate(
            [
                self.data.sensor("angular-velocity").data.astype(np.double),
                self.quat_rotate_inverse(
                    self.data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double),
                    np.array([0, 0, -1], dtype=np.double),
                ),
                self.command_vel,
                (self.dof_pos - self.default_dof_pos)[self.mujoco_to_isaac_idx],
                self.dof_vel[self.mujoco_vel_to_isaac_idx],
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
        self.latest_obs_step = obs.copy()
        self.obs_history[: -self.cfg.sim.num_obs_per_step] = self.obs_history[self.cfg.sim.num_obs_per_step :]
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs.copy()
        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def position_control(self) -> np.ndarray:
        policy_targets = self.action * self.cfg.sim.action_scale + self.default_dof_pos
        actuator_targets = np.empty_like(policy_targets)
        actuator_targets[self.isaac_to_mujoco_actuator_idx] = policy_targets
        return actuator_targets

    def adjust_command_vel(self, idx: int, increment: float):
        self.command_vel[idx] += increment
        self.command_vel[idx] = np.clip(self.command_vel[idx], -1.0, 1.0)

    def setup_keyboard_listener(self):
        if not self.cfg.sim.enable_keyboard_commands or self.viewer is None:
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

    def _render_offscreen_frame(self) -> None:
        if self.renderer is None or self.video_sink is None:
            return
        if self.camera_preset is not None:
            self._update_preset_camera()
            self.renderer.update_scene(self.data, camera=self.render_camera)
        elif self.camera is None:
            self.renderer.update_scene(self.data)
        else:
            self.renderer.update_scene(self.data, camera=self.camera)
        frame = self.renderer.render()
        self.video_sink.write_frame(frame)

    def _update_preset_camera(self) -> None:
        if self.render_camera is None or self.camera_preset is None:
            return
        root_pos = np.asarray(self.data.qpos[:3], dtype=np.float64)
        lookat_offset = np.asarray(self.camera_preset["lookat_offset"], dtype=np.float64)
        self.render_camera.lookat[:] = root_pos + lookat_offset
        self.render_camera.distance = float(self.camera_preset["distance"])
        self.render_camera.azimuth = float(self.camera_preset["azimuth"])
        self.render_camera.elevation = float(self.camera_preset["elevation"])

    def _close_rendering(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.video_sink is not None:
            self.video_sink.close()
            self.video_sink = None
        if self.renderer is not None:
            close_renderer = getattr(self.renderer, "close", None)
            if callable(close_renderer):
                close_renderer()
            self.renderer = None

    def _append_trace_record(self) -> None:
        if self.trace_path is None or len(self.trace_records) >= self.trace_steps:
            return

        record = {
            "policy_step": int(self.episode_length_buf),
            "sim_time": float(self.data.time),
            "root_pos": np.asarray(self.data.qpos[:3], dtype=np.float64).copy(),
            "root_quat_wxyz": np.asarray(self.data.qpos[3:7], dtype=np.float64).copy(),
            "command_vel": self.command_vel.astype(np.float64).copy(),
            "angular_velocity": self.data.sensor("angular-velocity").data.astype(np.float64).copy(),
            "projected_gravity": self.quat_rotate_inverse(
                self.data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float64),
                np.array([0.0, 0.0, -1.0], dtype=np.float64),
            ),
            "joint_pos_isaac": self.dof_pos[self.mujoco_to_isaac_idx].astype(np.float64).copy(),
            "joint_vel_isaac": self.dof_vel[self.mujoco_vel_to_isaac_idx].astype(np.float64).copy(),
            "action": self.action.astype(np.float64).copy(),
            "ctrl": np.asarray(self.data.ctrl, dtype=np.float64).copy(),
            "obs_step": self.latest_obs_step.astype(np.float32).copy(),
        }
        self.trace_records.append(record)

    def _flush_trace(self) -> None:
        if self.trace_path is None or not self.trace_records:
            return

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        keys = tuple(self.trace_records[0].keys())
        stacked = {}
        for key in keys:
            values = [record[key] for record in self.trace_records]
            first_value = values[0]
            if np.isscalar(first_value):
                stacked[key] = np.asarray(values)
            else:
                stacked[key] = np.stack(values, axis=0)
        np.savez_compressed(self.trace_path, **stacked)
        print(f"[INFO] Saved sim2sim trace to: {self.trace_path}")

    def run(self):
        self.setup_keyboard_listener()
        if self.listener is not None:
            self.listener.start()

        try:
            if self.save_video is not None:
                self._render_offscreen_frame()
                self.next_frame_time = self.frame_interval

            while self.data.time < self.cfg.sim.sim_duration:
                self.obs_history = self.get_obs()
                if self.control_mode == "policy":
                    if self.policy is None:
                        raise RuntimeError("Policy mode requested but no policy is loaded.")
                    action_tensor = self.policy(torch.tensor(self.obs_history, dtype=torch.float32))
                    self.action[:] = action_tensor.detach().cpu().numpy()[: self.cfg.sim.num_action]
                    self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
                    if self.lock_arms:
                        self.action[self.arm_action_ids] = 0.0
                else:
                    self.action[:] = 0.0

                self._append_trace_record()

                for _ in range(self.cfg.sim.decimation):
                    step_start_time = time.time()
                    self.data.ctrl = self.position_control()
                    mujoco.mj_step(self.model, self.data)
                    if self.viewer is not None:
                        self.viewer.render()
                        elapsed = time.time() - step_start_time
                        sleep_time = self.cfg.sim.dt - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                if self.save_video is not None and self.frame_interval is not None:
                    while self.data.time + 1e-9 >= self.next_frame_time:
                        self._render_offscreen_frame()
                        self.next_frame_time += self.frame_interval

                self.episode_length_buf += 1
                self.calculate_gait_para()
        finally:
            if self.listener is not None:
                self.listener.stop()
            self._close_rendering()
            self._flush_trace()

        if self.save_video is not None:
            print(f"[INFO] Saved rollout video to: {self.save_video}")


def build_cfg(task_name: str, duration: float, enable_keyboard_commands: bool) -> RealLiteSim2SimCfg:
    cfg = RealLiteSim2SimCfg()
    cfg.sim.sim_duration = duration
    cfg.sim.enable_keyboard_commands = enable_keyboard_commands and task_name != "upper_body_real_lite"
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
    parser.add_argument("--policy", default=None, help="Path to exported policy.pt")
    parser.add_argument("--model", default=str(MJCF_PATH), help="Path to Real Lite MuJoCo XML")
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--save_video", default=None, help="Optional output mp4 path for headless offscreen rendering.")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS when --save_video is set.")
    parser.add_argument("--width", type=int, default=1280, help="Video width when --save_video is set.")
    parser.add_argument("--height", type=int, default=720, help="Video height when --save_video is set.")
    parser.add_argument(
        "--camera",
        default=None,
        help=(
            "Optional offscreen camera preset or MuJoCo camera name. "
            f"Built-in presets: {', '.join(camera_preset_names())}."
        ),
    )
    parser.add_argument("--command_vx", type=float, default=0.0, help="Initial commanded forward velocity.")
    parser.add_argument("--command_vy", type=float, default=0.0, help="Initial commanded lateral velocity.")
    parser.add_argument("--command_wz", type=float, default=0.0, help="Initial commanded yaw velocity.")
    parser.add_argument(
        "--control_mode",
        choices=("policy", "hold"),
        default="policy",
        help="Use the exported policy or hold the default pose to diagnose MuJoCo stability.",
    )
    parser.add_argument(
        "--trace_out",
        default=None,
        help="Optional .npz output path for sim2sim debug traces collected at each policy step.",
    )
    parser.add_argument(
        "--trace_steps",
        type=int,
        default=0,
        help="Maximum number of policy steps to store in --trace_out. Use 0 to disable tracing.",
    )
    parser.add_argument(
        "--lock_arms",
        action="store_true",
        help="Keep all arm joints at their default pose after policy inference for sim2sim debugging.",
    )
    args = parser.parse_args()

    policy_path = Path(args.policy).resolve() if args.policy else None
    model_path = Path(args.model).resolve()
    save_video_path = Path(args.save_video).resolve() if args.save_video else None
    trace_path = Path(args.trace_out).resolve() if args.trace_out else None

    if args.control_mode == "policy":
        if policy_path is None:
            print("[ERROR] --policy is required when --control_mode=policy.")
            sys.exit(1)
        if not policy_path.is_file():
            print(f"[ERROR] Policy file not found: {policy_path}")
            sys.exit(1)
    if not model_path.is_file():
        print(f"[ERROR] MuJoCo model file not found: {model_path}")
        print("[INFO] Run generate_real_lite_mjcf.py first.")
        sys.exit(1)
    if save_video_path is not None and args.fps <= 0.0:
        print(f"[ERROR] FPS must be positive, got {args.fps}.")
        sys.exit(1)
    if save_video_path is not None and (args.width <= 0 or args.height <= 0):
        print(f"[ERROR] Video size must be positive, got width={args.width}, height={args.height}.")
        sys.exit(1)
    if trace_path is not None and args.trace_steps <= 0:
        print("[ERROR] --trace_steps must be positive when --trace_out is set.")
        sys.exit(1)

    cfg = build_cfg(args.task, args.duration, enable_keyboard_commands=save_video_path is None)
    command_vel = (args.command_vx, args.command_vy, args.command_wz)
    print(f"[INFO] Initial command velocity: vx={command_vel[0]:.3f}, vy={command_vel[1]:.3f}, wz={command_vel[2]:.3f}")
    runner = RealLiteMujocoRunner(
        cfg=cfg,
        policy_path=policy_path,
        model_path=model_path,
        save_video=save_video_path,
        video_fps=args.fps,
        width=args.width,
        height=args.height,
        camera=args.camera,
        command_vel=command_vel,
        control_mode=args.control_mode,
        trace_path=trace_path,
        trace_steps=args.trace_steps,
        lock_arms=args.lock_arms,
    )
    runner.run()


if __name__ == "__main__":
    main()
