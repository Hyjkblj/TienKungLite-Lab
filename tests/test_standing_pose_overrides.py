from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("real_lite_lab.standing_pose_overrides")


class StandingPoseOverrideTests(unittest.TestCase):
    def test_apply_symmetric_standing_pitch_offsets_updates_only_pitch_chain(self) -> None:
        module = load_module()
        joint_names = (
            "hip_roll_l_joint",
            "hip_pitch_l_joint",
            "knee_pitch_l_joint",
            "ankle_pitch_l_joint",
            "hip_roll_r_joint",
            "hip_pitch_r_joint",
            "knee_pitch_r_joint",
            "ankle_pitch_r_joint",
            "shoulder_pitch_l_joint",
        )
        default_dof_pos = np.array([0.0, -0.5, 1.0, -0.5, 0.0, -0.5, 1.0, -0.5, 0.0], dtype=np.float64)

        adjusted = module.apply_symmetric_standing_pitch_offsets(
            default_dof_pos,
            joint_names,
            hip_pitch_offset=0.03,
            knee_pitch_offset=-0.06,
            ankle_pitch_offset=0.02,
        )

        np.testing.assert_allclose(
            adjusted,
            np.array([0.0, -0.47, 0.94, -0.48, 0.0, -0.47, 0.94, -0.48, 0.0], dtype=np.float64),
        )

    def test_apply_symmetric_standing_pitch_offsets_is_noop_by_default(self) -> None:
        module = load_module()
        joint_names = ("hip_pitch_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint")
        default_dof_pos = np.array([-0.5, 1.0, -0.5], dtype=np.float64)

        adjusted = module.apply_symmetric_standing_pitch_offsets(default_dof_pos, joint_names)

        np.testing.assert_allclose(adjusted, default_dof_pos)


if __name__ == "__main__":
    unittest.main()
