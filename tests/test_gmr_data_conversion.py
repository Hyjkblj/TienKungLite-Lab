from __future__ import annotations

import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "gmr_data_conversion.py"


def load_conversion_module():
    spec = importlib.util.spec_from_file_location("test_gmr_data_conversion_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GmrDataConversionTests(unittest.TestCase):
    @staticmethod
    def _make_policy_sized_dof_rows(values0: tuple[float, float], values1: tuple[float, float], values2: tuple[float, float]):
        row0 = np.zeros(20, dtype=np.float64)
        row1 = np.zeros(20, dtype=np.float64)
        row2 = np.zeros(20, dtype=np.float64)
        row0[:2] = values0
        row1[:2] = values1
        row2[:2] = values2
        return np.stack([row0, row1, row2], axis=0)

    def test_load_motion_data_falls_back_for_numpy_core_pickles(self) -> None:
        module = load_conversion_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_pkl = tmp_path / "motion.pkl"
            payload = {"value": np.array([1.0, 2.0], dtype=np.float64)}
            with input_pkl.open("wb") as f:
                pickle.dump(payload, f)

            fallback_error = ModuleNotFoundError("No module named 'numpy._core'")
            fallback_error.name = "numpy._core"

            with mock.patch.object(
                module.pickle,
                "load",
                side_effect=fallback_error,
            ):
                result = module._load_motion_data(str(input_pkl))

        np.testing.assert_allclose(result["value"], payload["value"])

    def test_convert_pkl_to_custom_writes_visualization_payload(self) -> None:
        module = load_conversion_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_pkl = tmp_path / "motion.pkl"
            output_txt = tmp_path / "motion.json"

            motion_data = {
                "root_pos": np.array(
                    [
                        [0.0, 0.0, 1.0],
                        [0.1, 0.0, 1.0],
                        [0.2, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
                "root_rot": np.array(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
                "dof_pos": np.array(
                    self._make_policy_sized_dof_rows((0.0, 0.1), (0.2, 0.3), (0.4, 0.5)),
                ),
            }

            with input_pkl.open("wb") as f:
                pickle.dump(motion_data, f)

            module.convert_pkl_to_custom(str(input_pkl), str(output_txt), fps=30.0)

            payload = json.loads(output_txt.read_text(encoding="utf-8"))
            self.assertEqual(payload["FrameType"], "visualization")
            self.assertEqual(payload["LoopMode"], "Wrap")
            self.assertEqual(payload["FrameDuration"], round(1.0 / 30.0, 3))
            self.assertEqual(len(payload["Frames"]), 2)
            self.assertEqual(len(payload["Frames"][0]), 52)

    def test_convert_pkl_to_custom_normalizes_initial_yaw_to_positive_x(self) -> None:
        module = load_conversion_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_pkl = tmp_path / "motion.pkl"
            output_txt = tmp_path / "motion.json"

            yaw90_xyzw = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=np.float64)
            motion_data = {
                "root_pos": np.array(
                    [
                        [0.0, 0.0, 1.0],
                        [0.0, 0.1, 1.0],
                        [0.0, 0.2, 1.0],
                    ],
                    dtype=np.float64,
                ),
                "root_rot": np.stack([yaw90_xyzw, yaw90_xyzw, yaw90_xyzw], axis=0),
                "dof_pos": np.array(
                    self._make_policy_sized_dof_rows((0.0, 0.1), (0.2, 0.3), (0.4, 0.5)),
                ),
            }

            with input_pkl.open("wb") as f:
                pickle.dump(motion_data, f)

            module.convert_pkl_to_custom(str(input_pkl), str(output_txt), fps=30.0)

            payload = json.loads(output_txt.read_text(encoding="utf-8"))
            first_frame = np.asarray(payload["Frames"][0], dtype=np.float64)
            second_frame = np.asarray(payload["Frames"][1], dtype=np.float64)
            dof_dim = motion_data["dof_pos"].shape[1]
            root_lin_vel_slice = slice(6 + dof_dim, 9 + dof_dim)
            first_root_euler = first_frame[3:6]
            first_root_lin_vel = first_frame[root_lin_vel_slice]
            second_root_lin_vel = second_frame[root_lin_vel_slice]

            self.assertTrue(np.allclose(first_root_euler[2], 0.0, atol=1e-6))
            self.assertGreater(first_root_lin_vel[0], 0.0)
            self.assertTrue(np.allclose(first_root_lin_vel[1], 0.0, atol=1e-6))
            self.assertGreater(second_root_lin_vel[0], 0.0)
            self.assertTrue(np.allclose(second_root_lin_vel[1], 0.0, atol=1e-6))

    def test_convert_pkl_to_custom_can_disable_initial_yaw_normalization(self) -> None:
        module = load_conversion_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_pkl = tmp_path / "motion.pkl"
            output_txt = tmp_path / "motion.json"

            yaw90_xyzw = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=np.float64)
            motion_data = {
                "root_pos": np.array(
                    [
                        [0.0, 0.0, 1.0],
                        [0.0, 0.1, 1.0],
                        [0.0, 0.2, 1.0],
                    ],
                    dtype=np.float64,
                ),
                "root_rot": np.stack([yaw90_xyzw, yaw90_xyzw, yaw90_xyzw], axis=0),
                "dof_pos": np.array(
                    self._make_policy_sized_dof_rows((0.0, 0.1), (0.2, 0.3), (0.4, 0.5)),
                ),
            }

            with input_pkl.open("wb") as f:
                pickle.dump(motion_data, f)

            module.convert_pkl_to_custom(
                str(input_pkl),
                str(output_txt),
                fps=30.0,
                normalize_initial_yaw=False,
            )

            payload = json.loads(output_txt.read_text(encoding="utf-8"))
            first_frame = np.asarray(payload["Frames"][0], dtype=np.float64)
            dof_dim = motion_data["dof_pos"].shape[1]
            root_lin_vel_slice = slice(6 + dof_dim, 9 + dof_dim)
            first_root_euler = first_frame[3:6]
            first_root_lin_vel = first_frame[root_lin_vel_slice]

            self.assertTrue(np.allclose(first_root_euler[2], np.pi / 2.0, atol=1e-6))
            self.assertGreater(first_root_lin_vel[1], 0.0)
            self.assertTrue(np.allclose(first_root_lin_vel[0], 0.0, atol=1e-6))

    def test_convert_pkl_to_custom_reorders_gmr_tienkunglite_joints(self) -> None:
        module = load_conversion_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_pkl = tmp_path / "motion.pkl"
            output_txt = tmp_path / "motion.json"

            first_frame = np.arange(20, dtype=np.float64)
            second_frame = first_frame + 1.0
            motion_data = {
                "fps": 30.0,
                "root_pos": np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]], dtype=np.float64),
                "root_rot": np.array(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
                "dof_pos": np.stack([first_frame, second_frame, second_frame + 1.0], axis=0),
                "link_body_list": [
                    "Base_link",
                    "hip_roll_l_link",
                    "hip_pitch_l_link",
                    "hip_yaw_l_link",
                    "knee_pitch_l_link",
                    "ankle_pitch_l_link",
                    "ankle_roll_l_link",
                    "hip_roll_r_link",
                    "hip_pitch_r_link",
                    "hip_yaw_r_link",
                    "knee_pitch_r_link",
                    "ankle_pitch_r_link",
                    "ankle_roll_r_link",
                    "shoulder_pitch_l_link",
                    "shoulder_roll_l_link",
                    "shoulder_yaw_l_link",
                    "elbow_pitch_l_link",
                    "left_hand",
                    "shoulder_pitch_r_link",
                    "shoulder_roll_r_link",
                    "shoulder_yaw_r_link",
                    "elbow_pitch_r_link",
                    "right_hand",
                ],
            }

            with input_pkl.open("wb") as f:
                pickle.dump(motion_data, f)

            module.convert_pkl_to_custom(str(input_pkl), str(output_txt))

            payload = json.loads(output_txt.read_text(encoding="utf-8"))
            pose = np.asarray(payload["Frames"][0][6:26], dtype=np.float64)
            expected = np.asarray(
                [
                    0.0,
                    2.0,
                    1.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    8.0,
                    7.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                    17.0,
                    18.0,
                    19.0,
                ],
                dtype=np.float64,
            )
            np.testing.assert_allclose(pose, expected)

    def test_convert_pkl_to_custom_upper_body_profile_freezes_root_and_legs(self) -> None:
        module = load_conversion_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_pkl = tmp_path / "motion.pkl"
            output_txt = tmp_path / "motion.json"

            dof0 = np.zeros(20, dtype=np.float64)
            dof1 = np.zeros(20, dtype=np.float64)
            dof2 = np.zeros(20, dtype=np.float64)
            dof1[12:20] = np.arange(8, dtype=np.float64) + 1.0
            dof2[12:20] = np.arange(8, dtype=np.float64) + 2.0
            motion_data = {
                "fps": 30.0,
                "root_pos": np.array([[0.0, 0.0, 1.0], [0.5, 0.1, 1.1], [1.0, 0.2, 1.2]], dtype=np.float64),
                "root_rot": np.array(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
                "dof_pos": np.stack([dof0, dof1, dof2], axis=0),
                "link_body_list": [
                    "Base_link",
                    "hip_roll_l_link",
                    "hip_pitch_l_link",
                    "hip_yaw_l_link",
                    "knee_pitch_l_link",
                    "ankle_pitch_l_link",
                    "ankle_roll_l_link",
                    "hip_roll_r_link",
                    "hip_pitch_r_link",
                    "hip_yaw_r_link",
                    "knee_pitch_r_link",
                    "ankle_pitch_r_link",
                    "ankle_roll_r_link",
                    "shoulder_pitch_l_link",
                    "shoulder_roll_l_link",
                    "shoulder_yaw_l_link",
                    "elbow_pitch_l_link",
                    "left_hand",
                    "shoulder_pitch_r_link",
                    "shoulder_roll_r_link",
                    "shoulder_yaw_r_link",
                    "elbow_pitch_r_link",
                    "right_hand",
                ],
            }

            with input_pkl.open("wb") as f:
                pickle.dump(motion_data, f)

            module.convert_pkl_to_custom(str(input_pkl), str(output_txt), motion_profile="upper_body")

            payload = json.loads(output_txt.read_text(encoding="utf-8"))
            frame0 = np.asarray(payload["Frames"][0], dtype=np.float64)
            frame1 = np.asarray(payload["Frames"][1], dtype=np.float64)

            np.testing.assert_allclose(frame0[:6], np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
            np.testing.assert_allclose(frame1[:6], np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
            np.testing.assert_allclose(frame0[6:18], np.asarray(module.DEFAULT_DOF_POS[:12], dtype=np.float64))
            np.testing.assert_allclose(frame1[6:18], np.asarray(module.DEFAULT_DOF_POS[:12], dtype=np.float64))
            self.assertTrue(np.allclose(frame1[26:32], 0.0))
            self.assertTrue(np.allclose(frame1[32:38], 0.0))
            self.assertFalse(np.allclose(frame1[18:26], frame0[18:26]))


if __name__ == "__main__":
    unittest.main()
