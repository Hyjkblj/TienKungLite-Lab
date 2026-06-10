from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - exercised on the server when imageio is unavailable.
    imageio = None

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from real_lite_lab.constants import MJCF_DIR, POLICY_JOINT_NAMES
from real_lite_lab.render_camera import camera_preset_alias_names, camera_preset_names, get_camera_preset


def _make_writer(output_path: Path, fps: float):
    if imageio is None:
        raise RuntimeError("imageio is not installed; cannot write mp4.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(output_path), fps=fps, codec="libx264")


def _configure_camera(camera_name: str | None) -> mujoco.MjvCamera | str | None:
    preset = get_camera_preset(camera_name)
    if preset is None:
        return camera_name
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = float(preset["distance"])
    camera.azimuth = float(preset["azimuth"])
    camera.elevation = float(preset["elevation"])
    return camera


def _update_camera(camera, camera_name: str | None, root_pos: np.ndarray) -> None:
    if not isinstance(camera, mujoco.MjvCamera):
        return
    preset = get_camera_preset(camera_name)
    if preset is None:
        return
    camera.lookat[:] = root_pos + np.asarray(preset["lookat_offset"], dtype=np.float64)


def _policy_joint_qpos_addresses(model: mujoco.MjModel) -> np.ndarray:
    qpos_addresses = []
    for joint_name in POLICY_JOINT_NAMES:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing policy joint: {joint_name}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(qpos_addresses, dtype=np.int32)


def _frame_indices(trace_time: np.ndarray, fps: float) -> np.ndarray:
    duration_s = float(trace_time[-1])
    frame_times = np.arange(0.0, duration_s + 1.0e-9, 1.0 / fps)
    indices = np.searchsorted(trace_time, frame_times, side="left")
    return np.clip(indices, 0, len(trace_time) - 1)


def render_trace_video(
    *,
    trace_path: Path,
    model_path: Path,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    camera_name: str | None,
) -> None:
    trace = np.load(trace_path, allow_pickle=True)
    root_pos = np.asarray(trace["root_pos"], dtype=np.float64)
    root_quat_wxyz = np.asarray(trace["root_quat_wxyz"], dtype=np.float64)
    joint_pos_policy = np.asarray(trace["joint_pos_policy"], dtype=np.float64)
    trace_time = np.asarray(trace["time"], dtype=np.float64)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Unexpected root_pos shape: {root_pos.shape}")
    if root_quat_wxyz.shape != (root_pos.shape[0], 4):
        raise ValueError(f"Unexpected root_quat_wxyz shape: {root_quat_wxyz.shape}")
    if joint_pos_policy.shape != (root_pos.shape[0], len(POLICY_JOINT_NAMES)):
        raise ValueError(f"Unexpected joint_pos_policy shape: {joint_pos_policy.shape}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    qpos_addresses = _policy_joint_qpos_addresses(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = _configure_camera(camera_name)
    writer = _make_writer(output_path, fps=fps)

    try:
        for frame_idx in _frame_indices(trace_time, fps):
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            data.qpos[0:3] = root_pos[frame_idx]
            data.qpos[3:7] = root_quat_wxyz[frame_idx]
            data.qpos[qpos_addresses] = joint_pos_policy[frame_idx]
            mujoco.mj_forward(model, data)
            _update_camera(camera, camera_name, root_pos[frame_idx])
            if camera is None:
                renderer.update_scene(data)
            else:
                renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        close_renderer = getattr(renderer, "close", None)
        if callable(close_renderer):
            close_renderer()

    print(f"[INFO] Rendered policy trace video: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an Isaac policy trace to mp4 using MuJoCo visuals only.")
    parser.add_argument("--trace", required=True, help="Input .npz trace from eval_stand_real_lite.py --trace_out.")
    parser.add_argument("--model", default=str(MJCF_DIR / "real_lite.xml"), help="MuJoCo XML model path.")
    parser.add_argument("--output", required=True, help="Output mp4 path.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--camera",
        default="follow_side",
        help=(
            "Camera preset or MuJoCo camera name. "
            f"Built-in presets: {', '.join(camera_preset_names())}. "
            f"Aliases: {', '.join(camera_preset_alias_names())}."
        ),
    )
    args = parser.parse_args()

    if args.fps <= 0.0:
        raise ValueError("--fps must be positive.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive.")

    render_trace_video(
        trace_path=Path(args.trace).expanduser().resolve(),
        model_path=Path(args.model).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera_name=args.camera,
    )


if __name__ == "__main__":
    main()
