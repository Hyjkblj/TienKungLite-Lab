from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "analyze_isaac_standing_trace.py"


def load_module():
    module_name = "test_analyze_isaac_standing_trace_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class AnalyzeIsaacStandingTraceTests(unittest.TestCase):
    def test_load_trace_reads_npz_arrays(self) -> None:
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace.npz"
            np.savez_compressed(trace_path, sim_time=np.array([0.0, 0.5], dtype=np.float64))

            trace = module._load_trace(trace_path)

        np.testing.assert_allclose(trace["sim_time"], np.array([0.0, 0.5], dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
