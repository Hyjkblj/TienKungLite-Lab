from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
