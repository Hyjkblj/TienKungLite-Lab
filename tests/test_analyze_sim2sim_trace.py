from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from real_lite_lab.constants import POLICY_JOINT_NAMES


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "analyze_sim2sim_trace.py"


def load_trace_module():
    module_name = "test_analyze_sim2sim_trace_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class AnalyzeSim2SimTraceTests(unittest.TestCase):
    def test_analyze_trace_reports_support_margin_and_load_balance_events(self) -> None:
        module = load_trace_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace.npz"
            num_frames = 3
            num_joints = 20
            np.savez_compressed(
                trace_path,
                sim_time=np.array([0.0, 0.5, 1.0], dtype=np.float64),
                root_pos=np.array(
                    [
                        [0.0, 0.0, 1.00],
                        [0.0, 0.0, 0.98],
                        [0.0, 0.0, 0.90],
                    ],
                    dtype=np.float64,
                ),
                projected_gravity=np.array(
                    [
                        [0.0, 0.0, -1.0],
                        [0.2, 0.0, -0.98],
                        [0.8, 0.0, -0.2],
                    ],
                    dtype=np.float64,
                ),
                angular_velocity=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.2, 0.1, 0.0],
                        [3.0, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                ),
                joint_vel_isaac=np.zeros((num_frames, num_joints), dtype=np.float64),
                joint_pos_isaac=np.zeros((num_frames, num_joints), dtype=np.float64),
                action=np.zeros((num_frames, num_joints), dtype=np.float64),
                policy_target_isaac=np.zeros((num_frames, num_joints), dtype=np.float64),
                clamped_target_isaac=np.zeros((num_frames, num_joints), dtype=np.float64),
                ctrl=np.zeros((num_frames, num_joints), dtype=np.float64),
                support_margin=np.array([0.03, 0.01, -0.02], dtype=np.float64),
                foot_normal_forces=np.array(
                    [
                        [100.0, 100.0],
                        [130.0, 70.0],
                        [20.0, 0.0],
                    ],
                    dtype=np.float64,
                ),
                left_load_share=np.array([0.5, 0.65, 1.0], dtype=np.float64),
                double_support=np.array([1, 1, 0], dtype=np.int32),
            )

            lines = module.analyze_trace(
                trace_path,
                height_drop_threshold=0.05,
                tilt_threshold_deg=20.0,
                support_force_threshold=20.0,
                support_hold_steps=3,
            )

        joined = "\n".join(lines)
        self.assertIn("support_margin: start=+0.0300, end=-0.0200", joined)
        self.assertIn("support margin < 0.0m: step=2, time=1.000s", joined)
        self.assertIn("double_support_ratio(contact-count): 0.667", joined)
        self.assertIn("double support lost(contact-count): step=2, time=1.000s", joined)
        self.assertIn("loaded_double_support_ratio(>=20.0N per foot): 0.667", joined)
        self.assertIn("loaded double support lost for 3 frames: not reached", joined)
        self.assertIn("left_load_share: start=0.500, end=1.000", joined)
        self.assertIn("tilt_event support_margin: -0.0200m", joined)

    def test_analyze_trace_ignores_transient_loaded_support_dropouts(self) -> None:
        module = load_trace_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace_transient_dropout.npz"
            num_frames = 5
            num_joints = 20
            np.savez_compressed(
                trace_path,
                sim_time=np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64),
                root_pos=np.array(
                    [
                        [0.0, 0.0, 1.00],
                        [0.0, 0.0, 0.99],
                        [0.0, 0.0, 0.98],
                        [0.0, 0.0, 0.97],
                        [0.0, 0.0, 0.96],
                    ],
                    dtype=np.float64,
                ),
                projected_gravity=np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float64), (num_frames, 1)),
                angular_velocity=np.zeros((num_frames, 3), dtype=np.float64),
                joint_vel_isaac=np.zeros((num_frames, num_joints), dtype=np.float64),
                joint_pos_isaac=np.zeros((num_frames, num_joints), dtype=np.float64),
                action=np.zeros((num_frames, num_joints), dtype=np.float64),
                policy_target_isaac=np.zeros((num_frames, num_joints), dtype=np.float64),
                clamped_target_isaac=np.zeros((num_frames, num_joints), dtype=np.float64),
                ctrl=np.zeros((num_frames, num_joints), dtype=np.float64),
                support_margin=np.full(num_frames, 0.05, dtype=np.float64),
                foot_normal_forces=np.array(
                    [
                        [100.0, 100.0],
                        [0.0, 100.0],
                        [100.0, 100.0],
                        [0.0, 100.0],
                        [0.0, 100.0],
                    ],
                    dtype=np.float64,
                ),
                left_load_share=np.array([0.5, 0.0, 0.5, 0.0, 0.0], dtype=np.float64),
                double_support=np.array([1, 0, 1, 0, 0], dtype=np.int32),
            )

            lines = module.analyze_trace(
                trace_path,
                height_drop_threshold=0.05,
                tilt_threshold_deg=20.0,
                support_force_threshold=20.0,
                support_hold_steps=2,
            )

        joined = "\n".join(lines)
        self.assertIn("double support lost(contact-count): step=1, time=0.500s", joined)
        self.assertIn("loaded double support lost for 2 frames: step=3, time=1.500s", joined)

    def test_analyze_trace_uses_recorded_standing_target_for_joint_error(self) -> None:
        module = load_trace_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace_standing_target.npz"
            num_frames = 2
            num_joints = 20
            standing_target = np.zeros(num_joints, dtype=np.float64)
            hip_pitch_left_idx = POLICY_JOINT_NAMES.index("hip_pitch_l_joint")
            standing_target[hip_pitch_left_idx] = -0.4
            joint_pos = np.zeros((num_frames, num_joints), dtype=np.float64)
            joint_pos[1, hip_pitch_left_idx] = -0.1
            np.savez_compressed(
                trace_path,
                sim_time=np.array([0.0, 1.0], dtype=np.float64),
                root_pos=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.9]], dtype=np.float64),
                projected_gravity=np.array([[0.0, 0.0, -1.0], [0.8, 0.0, -0.2]], dtype=np.float64),
                angular_velocity=np.zeros((num_frames, 3), dtype=np.float64),
                joint_vel_isaac=np.zeros((num_frames, num_joints), dtype=np.float64),
                joint_pos_isaac=joint_pos,
                standing_target_isaac=standing_target,
            )

            lines = module.analyze_trace(
                trace_path,
                height_drop_threshold=0.05,
                tilt_threshold_deg=20.0,
                support_force_threshold=20.0,
                support_hold_steps=3,
            )

        joined = "\n".join(lines)
        self.assertIn("tilt_event top_joint_pos_error: hip_pitch_l_joint=+0.3000", joined)

    def test_extract_trace_metrics_reports_support_offsets(self) -> None:
        module = load_trace_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace_metrics.npz"
            num_frames = 3
            num_joints = 20
            np.savez_compressed(
                trace_path,
                sim_time=np.array([0.0, 0.5, 1.0], dtype=np.float64),
                root_pos=np.array(
                    [
                        [0.0, 0.0, 1.00],
                        [0.0, 0.0, 0.97],
                        [0.0, 0.0, 0.90],
                    ],
                    dtype=np.float64,
                ),
                projected_gravity=np.array(
                    [
                        [0.0, 0.0, -1.0],
                        [0.2, 0.0, -0.98],
                        [0.8, 0.0, -0.2],
                    ],
                    dtype=np.float64,
                ),
                angular_velocity=np.zeros((num_frames, 3), dtype=np.float64),
                joint_vel_isaac=np.zeros((num_frames, num_joints), dtype=np.float64),
                support_margin=np.array([0.03, 0.01, -0.02], dtype=np.float64),
                support_offset_xy=np.array(
                    [
                        [-0.02, 0.00],
                        [0.01, 0.00],
                        [0.09, 0.01],
                    ],
                    dtype=np.float64,
                ),
                foot_normal_forces=np.array(
                    [
                        [100.0, 100.0],
                        [100.0, 100.0],
                        [0.0, 100.0],
                    ],
                    dtype=np.float64,
                ),
                double_support=np.array([1, 1, 0], dtype=np.int32),
            )

            metrics = module.extract_trace_metrics(
                trace_path,
                height_drop_threshold=0.05,
                tilt_threshold_deg=20.0,
                support_force_threshold=20.0,
                support_hold_steps=2,
            )

        self.assertAlmostEqual(float(metrics["support_loss_time"]), 1.0)
        self.assertEqual(metrics["support_offset_xy_start"], [-0.02, 0.0])
        self.assertEqual(metrics["support_offset_xy_at_support_loss"], [0.09, 0.01])


if __name__ == "__main__":
    unittest.main()
