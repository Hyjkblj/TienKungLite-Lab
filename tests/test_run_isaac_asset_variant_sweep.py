from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "run_isaac_asset_variant_sweep.py"


def load_module():
    module_name = "test_run_isaac_asset_variant_sweep_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class RunIsaacAssetVariantSweepTests(unittest.TestCase):
    def test_reference_feet_mass_zero_fixed_align_command_preserves_sweep_scope(self) -> None:
        module = load_module()
        variant = module.ASSET_VARIANTS["reference_feet_mass_zero_fixed"]

        command = module._build_align_command(
            variant,
            candidate_urdf=Path("/asset/urdf/humanoid_publish.urdf"),
            output_urdf=Path("/asset/urdf/humanoid_publish.reference_feet_mass_zero_fixed.urdf"),
            reference_asset_root=None,
            reference_urdf=None,
        )

        self.assertIsNotNone(command)
        joined = " ".join(command)
        self.assertIn("--replace-ankle-roll-collisions-with-reference", joined)
        self.assertIn("--sync-link-mass", joined)
        self.assertIn("--zero-candidate-only-fixed-link-mass", joined)
        self.assertIn("--no-sync-collision-topology", joined)
        self.assertIn("--no-sync-joint-limits", joined)

    def test_baseline_does_not_generate_aligned_urdf(self) -> None:
        module = load_module()

        command = module._build_align_command(
            module.ASSET_VARIANTS["baseline"],
            candidate_urdf=Path("/asset/urdf/humanoid_publish.urdf"),
            output_urdf=Path("/asset/urdf/humanoid_publish.baseline.urdf"),
            reference_asset_root=None,
            reference_urdf=None,
        )

        self.assertIsNone(command)

    def test_variant_usd_subdir_is_unique(self) -> None:
        module = load_module()

        self.assertEqual(
            module._variant_usd_subdir("humanoid_publish_asset_variant", "reference_feet"),
            "humanoid_publish_asset_variant_reference_feet",
        )

    def test_diagnostic_command_uses_current_best_pose_defaults(self) -> None:
        module = load_module()

        command = module._build_diagnostic_command(
            task="walk_real_lite",
            duration=8.0,
            settle_time=0.0,
            trace_path=Path("/logs/trace.npz"),
            root_z=0.782,
            hip_pitch_target=-0.55,
            knee_pitch_target=1.0,
            ankle_pitch_target=-0.5,
            hip_pitch_kp_scale=1.0,
            hip_pitch_kd_scale=1.0,
            knee_pitch_kp_scale=1.0,
            knee_pitch_kd_scale=1.0,
            ankle_pitch_kp_scale=1.0,
            ankle_pitch_kd_scale=3.0,
            ankle_roll_kp_scale=1.0,
            ankle_roll_kd_scale=1.0,
            continue_after_termination=False,
            headless=True,
        )

        joined = " ".join(command)
        self.assertIn("--root_z 0.782", joined)
        self.assertIn("--hip_pitch_target -0.55", joined)
        self.assertIn("--knee_pitch_target 1", joined)
        self.assertIn("--ankle_pitch_target -0.5", joined)
        self.assertIn("--ankle_pitch_kd_scale 3", joined)
        self.assertIn("--headless", joined)


if __name__ == "__main__":
    unittest.main()
