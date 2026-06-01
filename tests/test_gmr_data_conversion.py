from __future__ import annotations

import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
