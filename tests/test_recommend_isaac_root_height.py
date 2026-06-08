from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "recommend_isaac_root_height.py"


def load_module():
    module_name = "test_recommend_isaac_root_height_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class RecommendIsaacRootHeightTests(unittest.TestCase):
    def test_recommend_root_height_uses_lowest_initial_foot_z(self) -> None:
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace.npz"
            np.savez_compressed(
                trace_path,
                root_pos=np.array([[0.9475, 0.0, 0.9475], [0.0, 0.0, 0.80]], dtype=np.float64),
                feet_pos_w=np.array(
                    [
                        [[0.0, 0.0, 0.2128], [0.0, 0.0, 0.2100]],
                        [[0.0, 0.0, 0.0487], [0.0, 0.0, 0.0495]],
                    ],
                    dtype=np.float64,
                ),
            )

            recommendation = module.recommend_root_height(trace_path, target_foot_z=0.05)

        self.assertAlmostEqual(recommendation.feet_z_start_min, 0.2100)
        self.assertAlmostEqual(recommendation.root_z_delta, -0.1600)
        self.assertAlmostEqual(recommendation.suggested_root_z, 0.7875)

    def test_trace_paths_accepts_directory_inputs(self) -> None:
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.npz").write_bytes(b"")
            (root / "b.txt").write_text("", encoding="utf-8")

            paths = module._trace_paths([str(root)])

        self.assertEqual([path.name for path in paths], ["a.npz"])


if __name__ == "__main__":
    unittest.main()
