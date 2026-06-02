from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("real_lite_lab.actuator_scale_overrides")


class ActuatorScaleOverrideTests(unittest.TestCase):
    def test_build_actuator_scale_overrides_populates_selected_joint_groups(self) -> None:
        module = load_module()

        joint_kp_scales, joint_kv_scales = module.build_actuator_scale_overrides(
            hip_pitch_kp_scale=1.2,
            knee_pitch_kp_scale=0.9,
            ankle_pitch_kp_scale=2.5,
            hip_pitch_kv_scale=1.1,
            knee_pitch_kv_scale=0.8,
            ankle_roll_kv_scale=1.3,
        )

        self.assertEqual(
            joint_kp_scales,
            {
                "hip_pitch_l_joint": 1.2,
                "hip_pitch_r_joint": 1.2,
                "knee_pitch_l_joint": 0.9,
                "knee_pitch_r_joint": 0.9,
                "ankle_pitch_l_joint": 2.5,
                "ankle_pitch_r_joint": 2.5,
            },
        )
        self.assertEqual(
            joint_kv_scales,
            {
                "hip_pitch_l_joint": 1.1,
                "hip_pitch_r_joint": 1.1,
                "knee_pitch_l_joint": 0.8,
                "knee_pitch_r_joint": 0.8,
                "ankle_roll_l_joint": 1.3,
                "ankle_roll_r_joint": 1.3,
            },
        )

    def test_build_actuator_scale_overrides_omits_unity_scales(self) -> None:
        module = load_module()

        joint_kp_scales, joint_kv_scales = module.build_actuator_scale_overrides()

        self.assertEqual(joint_kp_scales, {})
        self.assertEqual(joint_kv_scales, {})


if __name__ == "__main__":
    unittest.main()
