from __future__ import annotations

import importlib.util
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "isaac_standing_diagnostic.py"


def load_module():
    module_name = "test_isaac_standing_diagnostic_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class IsaacStandingDiagnosticTests(unittest.TestCase):
    def test_summarize_standing_trace_reports_termination_contact(self) -> None:
        module = load_module()
        trace = {
            "sim_time": np.array([0.0, 0.5, 1.0], dtype=np.float64),
            "root_pos": np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.98],
                    [0.0, 0.0, 0.90],
                ],
                dtype=np.float64,
            ),
            "projected_gravity": np.array(
                [
                    [0.0, 0.0, -1.0],
                    [0.1, 0.0, -0.99],
                    [0.8, 0.0, -0.2],
                ],
                dtype=np.float64,
            ),
            "joint_vel_policy": np.zeros((3, 20), dtype=np.float64),
            "joint_pos_error_policy": np.zeros((3, 20), dtype=np.float64),
            "foot_normal_forces": np.array(
                [
                    [200.0, 200.0],
                    [150.0, 30.0],
                    [0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            "termination_contact": np.array([False, False, True], dtype=bool),
        }

        lines = module.summarize_standing_trace(
            trace,
            height_drop_threshold=0.05,
            tilt_threshold_deg=20.0,
            support_force_threshold=20.0,
            support_hold_steps=2,
        )
        joined = "\n".join(lines)
        self.assertIn("termination contact: step=2, time=1.000s", joined)
        self.assertIn("loaded double support lost for 2 frames: not reached", joined)

    def test_evaluate_standing_stability_reports_failures(self) -> None:
        module = load_module()
        trace = {
            "sim_time": np.array([0.0, 0.5, 1.0], dtype=np.float64),
            "root_pos": np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.98],
                    [0.0, 0.0, 0.90],
                ],
                dtype=np.float64,
            ),
            "projected_gravity": np.array(
                [
                    [0.0, 0.0, -1.0],
                    [0.1, 0.0, -0.99],
                    [0.8, 0.0, -0.2],
                ],
                dtype=np.float64,
            ),
            "foot_normal_forces": np.array(
                [
                    [200.0, 200.0],
                    [150.0, 30.0],
                    [0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            "termination_contact": np.array([False, False, True], dtype=bool),
        }

        failures = module.evaluate_standing_stability(
            trace,
            height_drop_threshold=0.05,
            tilt_threshold_deg=20.0,
            support_force_threshold=20.0,
            support_hold_steps=2,
        )

        joined = "\n".join(failures)
        self.assertIn("termination contact", joined)
        self.assertIn("root dropped", joined)
        self.assertIn("tilt reached", joined)

    def test_summarize_standing_trace_can_report_joint_names(self) -> None:
        module = load_module()
        trace = {
            "sim_time": np.array([0.0, 0.5], dtype=np.float64),
            "root_pos": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.9]], dtype=np.float64),
            "projected_gravity": np.array([[0.0, 0.0, -1.0], [0.8, 0.0, -0.2]], dtype=np.float64),
            "joint_vel_policy": np.array([[0.0, 0.0], [0.2, -0.7]], dtype=np.float64),
            "joint_pos_error_policy": np.array([[0.0, 0.0], [0.1, -0.4]], dtype=np.float64),
            "foot_normal_forces": np.array([[100.0, 100.0], [0.0, 0.0]], dtype=np.float64),
            "termination_contact": np.array([False, True], dtype=bool),
        }

        lines = module.summarize_standing_trace(
            trace,
            height_drop_threshold=0.05,
            tilt_threshold_deg=20.0,
            support_force_threshold=20.0,
            support_hold_steps=1,
            joint_names=("hip_pitch_l_joint", "ankle_pitch_l_joint"),
        )

        joined = "\n".join(lines)
        self.assertIn("termination_contact top_joint_vel: ankle_pitch_l_joint=-0.7000", joined)
        self.assertIn("termination_contact top_joint_pos_error: ankle_pitch_l_joint=-0.4000", joined)

    def test_apply_isaac_actuator_scales_updates_stiffness_and_damping(self) -> None:
        module = load_module()
        robot_cfg = SimpleNamespace(
            actuators={
                "legs": SimpleNamespace(
                    stiffness={"hip_pitch_.*_joint": 700.0, "knee_pitch_.*_joint": 700.0},
                    damping={"hip_pitch_.*_joint": 10.0, "knee_pitch_.*_joint": 10.0},
                ),
                "feet": SimpleNamespace(
                    stiffness={"ankle_pitch_.*_joint": 30.0, "ankle_roll_.*_joint": 16.8},
                    damping={"ankle_pitch_.*_joint": 2.5, "ankle_roll_.*_joint": 1.4},
                ),
            }
        )

        effective = module.apply_isaac_actuator_scales(
            robot_cfg,
            knee_pitch_kd_scale=1.5,
            ankle_pitch_kp_scale=2.0,
            ankle_pitch_kd_scale=3.0,
        )

        self.assertEqual(robot_cfg.actuators["legs"].damping["knee_pitch_.*_joint"], 15.0)
        self.assertEqual(robot_cfg.actuators["feet"].stiffness["ankle_pitch_.*_joint"], 60.0)
        self.assertEqual(robot_cfg.actuators["feet"].damping["ankle_pitch_.*_joint"], 7.5)
        self.assertEqual(effective["stiffness"]["ankle_roll_.*_joint"], 16.8)


if __name__ == "__main__":
    unittest.main()
