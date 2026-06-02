from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "real_lite_lab" / "mujoco_standing_diagnostics.py"


def load_diagnostics_module():
    module_name = "test_mujoco_standing_diagnostics_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class MujocoStandingDiagnosticsTests(unittest.TestCase):
    def test_signed_distance_to_convex_polygon_is_positive_inside_and_negative_outside(self) -> None:
        module = load_diagnostics_module()
        polygon_xy = np.array(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.assertAlmostEqual(module.signed_distance_to_convex_polygon(np.array([0.0, 0.0]), polygon_xy), 1.0)
        self.assertAlmostEqual(module.signed_distance_to_convex_polygon(np.array([2.0, 0.0]), polygon_xy), -1.0)

    def test_compute_nominal_support_diagnostics_reports_support_margin_and_extents(self) -> None:
        module = load_diagnostics_module()
        identity_rot = np.eye(3, dtype=np.float64).reshape(1, 9)
        model = SimpleNamespace(
            geom_size=np.array(
                [
                    [0.10, 0.05, 0.01],
                    [0.10, 0.05, 0.01],
                ],
                dtype=np.float64,
            ),
            body_mass=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
        data = SimpleNamespace(
            geom_xpos=np.array(
                [
                    [-0.10, 0.00, 0.00],
                    [0.10, 0.00, 0.00],
                ],
                dtype=np.float64,
            ),
            geom_xmat=np.repeat(identity_rot, repeats=2, axis=0),
            xipos=np.array(
                [
                    [0.00, 0.00, 0.00],
                    [-0.02, 0.00, 0.40],
                    [0.02, 0.00, 0.80],
                ],
                dtype=np.float64,
            ),
        )

        diagnostics = module.compute_nominal_support_diagnostics(
            model,
            data,
            {"sole_left": 0, "sole_right": 1},
        )

        np.testing.assert_allclose(diagnostics["support_center_xy"], np.array([0.0, 0.0]))
        np.testing.assert_allclose(diagnostics["support_extents_xy"], np.array([0.4, 0.1]), atol=1e-9)
        np.testing.assert_allclose(diagnostics["com_world"], np.array([0.0033333333, 0.0, 0.5333333333]), atol=1e-8)
        self.assertGreater(float(diagnostics["support_margin"]), 0.0)

    def test_compute_support_polygon_xy_accepts_toe_rail_cylinders(self) -> None:
        module = load_diagnostics_module()
        identity_rot = np.eye(3, dtype=np.float64)
        cylinder_rot = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        fake_mujoco = SimpleNamespace(
            mjtGeom=SimpleNamespace(
                mjGEOM_BOX=0,
                mjGEOM_SPHERE=1,
                mjGEOM_CAPSULE=3,
                mjGEOM_CYLINDER=5,
            )
        )
        model = SimpleNamespace(
            geom_type=np.array([5, 5, 5, 5], dtype=np.int32),
            geom_size=np.array(
                [
                    [0.015, 0.115, 0.0],
                    [0.015, 0.115, 0.0],
                    [0.015, 0.115, 0.0],
                    [0.015, 0.115, 0.0],
                ],
                dtype=np.float64,
            ),
            geom_rbound=np.full(4, 0.13, dtype=np.float64),
        )
        data = SimpleNamespace(
            geom_xpos=np.array(
                [
                    [0.035, 0.025, -0.042],
                    [0.035, -0.025, -0.042],
                    [0.035, 0.025, -0.042],
                    [0.035, -0.025, -0.042],
                ],
                dtype=np.float64,
            ),
            geom_xmat=np.stack([cylinder_rot.reshape(9)] * 4, axis=0),
        )

        with mock.patch.dict(sys.modules, {"mujoco": fake_mujoco}):
            polygon_xy = module.compute_support_polygon_xy(
                model,
                data,
                {
                    "left": (0, 1),
                    "right": (2, 3),
                },
            )

        np.testing.assert_allclose(np.min(polygon_xy, axis=0), np.array([-0.08, -0.04]), atol=1e-9)
        np.testing.assert_allclose(np.max(polygon_xy, axis=0), np.array([0.15, 0.04]), atol=1e-9)

    def test_compute_support_contact_summary_counts_per_foot_contacts(self) -> None:
        module = load_diagnostics_module()
        contact_normal_forces = [120.0, 80.0, 10.0]

        def fake_mj_contact_force(_model, _data, contact_id, force_buffer) -> None:
            force_buffer[:] = 0.0
            force_buffer[0] = contact_normal_forces[contact_id]

        fake_mujoco = SimpleNamespace(mj_contactForce=fake_mj_contact_force)
        model = SimpleNamespace()
        data = SimpleNamespace(
            ncon=3,
            contact=[
                SimpleNamespace(geom1=10, geom2=0),
                SimpleNamespace(geom1=11, geom2=0),
                SimpleNamespace(geom1=10, geom2=11),
            ],
        )

        with mock.patch.dict(sys.modules, {"mujoco": fake_mujoco}):
            summary = module.compute_support_contact_summary(
                model,
                data,
                {"sole_left": 10, "sole_right": 11},
            )

        np.testing.assert_array_equal(summary["foot_contact_counts"], np.array([2, 2], dtype=np.int32))
        np.testing.assert_allclose(summary["foot_normal_forces"], np.array([130.0, 90.0]))
        self.assertEqual(int(summary["double_support"]), 1)
        self.assertAlmostEqual(float(summary["left_load_share"]), 130.0 / 220.0)


if __name__ == "__main__":
    unittest.main()
