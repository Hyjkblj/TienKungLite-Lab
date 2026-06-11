from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTANTS_PATH = REPO_ROOT / "real_lite_lab" / "constants.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WalkForwardTaskMetadataTests(unittest.TestCase):
    def test_walk_forward_task_uses_walk_motion_files(self) -> None:
        constants = load_module(CONSTANTS_PATH, "test_walk_forward_constants_module")

        self.assertIn("walk_forward_real_lite", constants.TASK_NAMES)
        preset = constants.TASK_PRESETS["walk_forward_real_lite"]
        self.assertEqual(preset["gait_cycle"], constants.TASK_PRESETS["walk_real_lite"]["gait_cycle"])
        self.assertEqual(preset["amp_motion_file"].parts[-2:], ("motion_amp_expert", "walk.txt"))
        self.assertEqual(preset["display_motion_file"].parts[-2:], ("motion_visualization", "walk.txt"))

    def test_walk_forward_command_ranges_are_low_speed_forward_only(self) -> None:
        from real_lite_lab import alignment_config

        self.assertEqual(
            alignment_config.TASK_COMMAND_RANGES["walk_forward_real_lite"],
            {
                "lin_vel_x": (0.10, 0.35),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (0.0, 0.0),
            },
        )

    def test_walk_forward_cfg_does_not_copy_configclass_fields_from_class(self) -> None:
        cfg_text = (REPO_ROOT / "real_lite_lab" / "walk_forward_cfg.py").read_text(encoding="utf-8")

        self.assertNotIn("RealLiteWalkEnvCfg.scene", cfg_text)
        self.assertNotIn("RealLiteWalkEnvCfg.robot", cfg_text)

    def test_walk_forward_policy_uses_log_std(self) -> None:
        cfg_text = (REPO_ROOT / "real_lite_lab" / "walk_forward_cfg.py").read_text(encoding="utf-8")

        self.assertIn('noise_std_type="log"', cfg_text)
        self.assertIn("learning_rate=3.0e-4", cfg_text)

    def test_walk_forward_v2_penalizes_velocity_shortfall(self) -> None:
        cfg_text = (REPO_ROOT / "real_lite_lab" / "walk_forward_cfg.py").read_text(encoding="utf-8")
        rewards_text = (REPO_ROOT / "real_lite_lab" / "rewards.py").read_text(encoding="utf-8")

        self.assertIn("lin_vel_x_shortfall", cfg_text)
        self.assertIn("weight=-8.0", cfg_text)
        self.assertIn("def lin_vel_x_shortfall_l1", rewards_text)
        self.assertIn("amp_task_reward_lerp = 0.97", cfg_text)


if __name__ == "__main__":
    unittest.main()
