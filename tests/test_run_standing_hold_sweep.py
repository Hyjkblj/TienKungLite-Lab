from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "run_standing_hold_sweep.py"


def load_sweep_module():
    module_name = "test_run_standing_hold_sweep_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class RunStandingHoldSweepTests(unittest.TestCase):
    def test_build_variant_label_only_includes_varying_fields(self) -> None:
        module = load_sweep_module()
        config = module.SweepConfig(
            hip_pitch_target=-0.4925,
            knee_pitch_target=0.9850,
            ankle_pitch_target=-0.4625,
            knee_pitch_kv_scale=1.5,
            ankle_pitch_kp_scale=2.5,
            ankle_pitch_kv_scale=2.5,
            ankle_roll_kp_scale=1.2,
            ankle_roll_kv_scale=1.2,
        )

        label = module._build_variant_label(config, ("ankle_pitch_target", "ankle_pitch_kv_scale"))
        self.assertEqual(label, "ap_m0p4625__apkv_2p5")

    def test_rank_sweep_results_prefers_later_failure_times(self) -> None:
        module = load_sweep_module()
        ranked = module.rank_sweep_results(
            [
                {
                    "tag": "early",
                    "duration": 6.0,
                    "support_loss_time": 1.5,
                    "loaded_single_support_time": 1.6,
                    "tilt_20_time": 1.7,
                    "tilt_45_time": 1.9,
                },
                {
                    "tag": "late",
                    "duration": 6.0,
                    "support_loss_time": 4.2,
                    "loaded_single_support_time": 4.3,
                    "tilt_20_time": 4.4,
                    "tilt_45_time": 4.7,
                },
                {
                    "tag": "not_reached",
                    "duration": 6.0,
                    "support_loss_time": None,
                    "loaded_single_support_time": None,
                    "tilt_20_time": None,
                    "tilt_45_time": None,
                },
            ]
        )

        self.assertEqual([row["tag"] for row in ranked], ["not_reached", "late", "early"])


if __name__ == "__main__":
    unittest.main()
