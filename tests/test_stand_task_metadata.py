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


class StandTaskMetadataTests(unittest.TestCase):
    def test_stand_task_is_available_without_amp_motion_requirement(self) -> None:
        constants = load_module(CONSTANTS_PATH, "test_stand_constants_module")

        self.assertIn("stand_real_lite", constants.TASK_NAMES)
        stand_preset = constants.TASK_PRESETS["stand_real_lite"]
        self.assertNotIn("amp_motion_file", stand_preset)
        self.assertEqual(stand_preset["gait_cycle"], 1.0)
        self.assertTrue(str(stand_preset["display_motion_file"]).endswith("upper_body.txt"))

    def test_stand_command_ranges_are_zero_velocity(self) -> None:
        from real_lite_lab import alignment_config

        self.assertEqual(
            alignment_config.TASK_COMMAND_RANGES["stand_real_lite"],
            {
                "lin_vel_x": (0.0, 0.0),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (0.0, 0.0),
            },
        )


if __name__ == "__main__":
    unittest.main()
