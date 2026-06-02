from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "real_lite_lab" / "mujoco_state_init.py"


def load_state_init_module():
    module_name = "test_mujoco_state_init_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class MujocoStateInitTests(unittest.TestCase):
    def test_apply_default_joint_state_populates_root_and_joint_defaults(self) -> None:
        module = load_state_init_module()

        model = SimpleNamespace(
            jnt_qposadr=np.array([7, 8, 9], dtype=np.int32),
            jnt_dofadr=np.array([0, 1, 2], dtype=np.int32),
        )
        data = SimpleNamespace(
            qpos=np.full(10, -999.0, dtype=np.float64),
            qvel=np.full(3, -999.0, dtype=np.float64),
        )
        joint_names = ("a", "b", "c")
        default_joint_pos = np.array([0.1, -0.2, 0.3], dtype=np.float64)
        joint_name_to_id = {"a": 0, "b": 1, "c": 2}.__getitem__

        module.apply_default_joint_state(
            model=model,
            data=data,
            joint_names=joint_names,
            default_joint_pos=default_joint_pos,
            joint_name_to_id=joint_name_to_id,
        )

        np.testing.assert_allclose(data.qpos[:7], np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(data.qpos[7:], default_joint_pos)
        np.testing.assert_allclose(data.qvel, np.zeros(3, dtype=np.float64))

    def test_apply_default_joint_state_validates_joint_count(self) -> None:
        module = load_state_init_module()
        model = SimpleNamespace(jnt_qposadr=np.array([7], dtype=np.int32), jnt_dofadr=np.array([0], dtype=np.int32))
        data = SimpleNamespace(qpos=np.zeros(8, dtype=np.float64), qvel=np.zeros(1, dtype=np.float64))

        with self.assertRaises(ValueError):
            module.apply_default_joint_state(
                model=model,
                data=data,
                joint_names=("a",),
                default_joint_pos=np.array([0.1, 0.2], dtype=np.float64),
                joint_name_to_id={"a": 0}.__getitem__,
            )


if __name__ == "__main__":
    unittest.main()
