from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "run_isaac_standing_sweep.py"


def load_sweep_module():
    module_name = "test_run_isaac_standing_sweep_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class RunIsaacStandingSweepTests(unittest.TestCase):
    def test_build_variant_label_handles_root_z_and_pd_fields(self) -> None:
        module = load_sweep_module()
        config = module.SweepConfig(
            root_z=0.96,
            hip_pitch_target=-0.45,
            knee_pitch_target=0.9,
            ankle_pitch_target=-0.42,
            hip_pitch_kp_scale=1.0,
            hip_pitch_kd_scale=1.0,
            knee_pitch_kp_scale=1.0,
            knee_pitch_kd_scale=1.5,
            ankle_pitch_kp_scale=2.0,
            ankle_pitch_kd_scale=3.0,
            ankle_roll_kp_scale=1.0,
            ankle_roll_kd_scale=1.0,
        )

        label = module._build_variant_label(config, ("root_z", "ankle_pitch_kp_scale", "ankle_pitch_kd_scale"))

        self.assertEqual(label, "rz_0p96__apkp_2__apkd_3")

    def test_extract_metrics_reports_torque_maxima(self) -> None:
        module = load_sweep_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace.npz"
            np.savez_compressed(
                trace_path,
                sim_time=np.array([0.0, 0.5], dtype=np.float64),
                root_pos=np.array([[0.0, 0.0, 0.8], [0.0, 0.0, 0.7]], dtype=np.float64),
                projected_gravity=np.array([[0.0, 0.0, -1.0], [0.8, 0.0, -0.2]], dtype=np.float64),
                foot_normal_forces=np.array([[100.0, 100.0], [50.0, 60.0]], dtype=np.float64),
                feet_pos_w=np.array(
                    [
                        [[0.0, -0.1, 0.05], [0.0, 0.1, 0.05]],
                        [[0.1, -0.1, 0.05], [0.1, 0.1, 0.05]],
                    ],
                    dtype=np.float64,
                ),
                system_com_pos_w=np.array([[0.02, 0.0, 0.6], [0.18, 0.04, 0.5]], dtype=np.float64),
                termination_contact=np.array([False, True], dtype=bool),
                joint_vel_policy=np.array([[0.1, -2.5], [0.2, -3.0]], dtype=np.float64),
                joint_pos_error_policy=np.array([[0.05, -0.4], [0.1, -0.6]], dtype=np.float64),
                joint_applied_torque_policy=np.array([[1.0, -2.0], [3.0, -60.0]], dtype=np.float64),
                joint_computed_torque_policy=np.array([[1.0, -2.0], [3.0, -75.0]], dtype=np.float64),
            )

            metrics = module._extract_metrics(trace_path, height_drop_threshold=0.05, tilt_threshold_deg=20.0)

        self.assertEqual(metrics["applied_torque_abs_max"], 60.0)
        self.assertEqual(metrics["applied_torque_abs_max_time"], 0.5)
        self.assertEqual(metrics["applied_torque_abs_max_joint"], "hip_pitch_l_joint")
        self.assertEqual(metrics["computed_torque_abs_max"], 75.0)
        self.assertEqual(metrics["start_joint_speed_abs_max"], 2.5)
        self.assertEqual(metrics["start_joint_speed_abs_max_joint"], "hip_pitch_l_joint")
        self.assertEqual(metrics["start_joint_pos_error_abs_max"], 0.4)
        self.assertEqual(metrics["start_applied_torque_abs_max"], 2.0)
        self.assertAlmostEqual(metrics["com_x_minus_feet_center_start"], 0.02)
        self.assertAlmostEqual(metrics["com_x_minus_feet_center_end"], 0.08)
        self.assertAlmostEqual(metrics["com_y_minus_feet_center_end"], 0.04)
        self.assertAlmostEqual(metrics["com_x_minus_feet_center_tilt20"], 0.08)
        self.assertAlmostEqual(metrics["com_x_minus_feet_center_drop"], 0.08)
        self.assertAlmostEqual(metrics["com_x_minus_feet_center_termination"], 0.08)


if __name__ == "__main__":
    unittest.main()
