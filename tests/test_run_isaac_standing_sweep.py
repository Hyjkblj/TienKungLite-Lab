from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
