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
                    [
                        [0.0, 0.1],
                        [0.2, 0.3],
                        [0.4, 0.5],
                    ],
                    dtype=np.float64,
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
            self.assertEqual(len(payload["Frames"][0]), 16)

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
                    [
                        [0.0, 0.1],
                        [0.2, 0.3],
                        [0.4, 0.5],
                    ],
                    dtype=np.float64,
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
                    [
                        [0.0, 0.1],
                        [0.2, 0.3],
                        [0.4, 0.5],
                    ],
                    dtype=np.float64,
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


if __name__ == "__main__":
    unittest.main()
