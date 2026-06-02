from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "real_lite_lab" / "render_camera.py"


def load_render_camera_module():
    spec = importlib.util.spec_from_file_location("test_render_camera_module", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RenderCameraTests(unittest.TestCase):
    def test_camera_preset_names_are_stable(self) -> None:
        module = load_render_camera_module()

        self.assertEqual(
            module.camera_preset_names(),
            ("follow_diag", "follow_side", "follow_front", "follow_topdiag"),
        )

    def test_camera_presets_have_required_fields(self) -> None:
        module = load_render_camera_module()

        for preset_name in module.camera_preset_names():
            with self.subTest(preset_name=preset_name):
                preset = module.get_camera_preset(preset_name)
                self.assertIsNotNone(preset)
                assert preset is not None
                self.assertGreater(preset["distance"], 0.0)
                self.assertEqual(len(preset["lookat_offset"]), 3)
                self.assertIsInstance(preset["azimuth"], float)
                self.assertIsInstance(preset["elevation"], float)

    def test_get_camera_preset_returns_none_for_unknown_name(self) -> None:
        module = load_render_camera_module()

        self.assertIsNone(module.get_camera_preset("not_a_camera"))
        self.assertIsNone(module.get_camera_preset(None))

    def test_camera_aliases_resolve_to_canonical_presets(self) -> None:
        module = load_render_camera_module()

        self.assertEqual(module.resolve_camera_preset_name("side"), "follow_side")
        self.assertEqual(module.resolve_camera_preset_name("front"), "follow_front")
        self.assertEqual(module.resolve_camera_preset_name("diag"), "follow_diag")
        self.assertEqual(module.resolve_camera_preset_name("topdiag"), "follow_topdiag")
        self.assertEqual(module.get_camera_preset("side"), module.get_camera_preset("follow_side"))


if __name__ == "__main__":
    unittest.main()
