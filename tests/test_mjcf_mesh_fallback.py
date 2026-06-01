from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "real_lite_lab" / "mjcf_mesh_fallback.py"


def load_mesh_fallback_module():
    module_name = "test_mjcf_mesh_fallback_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_binary_stl(mesh_path: Path, triangle_count: int) -> None:
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    triangle_record = (b"\0" * 50)
    mesh_path.write_bytes(b"binary".ljust(80, b"\0") + triangle_count.to_bytes(4, "little") + triangle_record * triangle_count)


class MjcfMeshFallbackTests(unittest.TestCase):
    def test_ensure_offscreen_framebuffer_size_grows_only_when_needed(self) -> None:
        module = load_mesh_fallback_module()
        model = SimpleNamespace(vis=SimpleNamespace(global_=SimpleNamespace(offwidth=640, offheight=480)))

        resized = module.ensure_offscreen_framebuffer_size(model, width=1280, height=720)
        self.assertEqual(resized, (1280, 720))
        self.assertEqual(model.vis.global_.offwidth, 1280)
        self.assertEqual(model.vis.global_.offheight, 720)

        unchanged = module.ensure_offscreen_framebuffer_size(model, width=640, height=480)
        self.assertIsNone(unchanged)

    def test_binary_stl_triangle_count_rejects_mismatched_payload(self) -> None:
        module = load_mesh_fallback_module()
        invalid_binary = b"binary".ljust(80, b"\0") + (999).to_bytes(4, "little") + (b"\0" * 50)
        self.assertIsNone(module._binary_stl_triangle_count(invalid_binary))

    def test_build_mesh_safe_model_strips_incompatible_visual_meshes(self) -> None:
        module = load_mesh_fallback_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mesh_dir = tmp_path / "_mesh_cache"
            _write_binary_stl(mesh_dir / "good.STL", triangle_count=1)
            _write_binary_stl(mesh_dir / "too_large.STL", triangle_count=module.MAX_MUJOCO_STL_TRIANGLES + 1)

            model_path = tmp_path / "model.xml"
            model_path.write_text(
                """<mujoco model="real_lite">
  <compiler meshdir="_mesh_cache/"/>
  <asset>
    <mesh name="good_mesh" file="good.STL"/>
    <mesh name="waist_link" file="too_large.STL"/>
  </asset>
  <worldbody>
    <body name="waist_link">
      <geom type="mesh" mesh="waist_link"/>
    </body>
    <body name="other">
      <geom type="mesh" mesh="good_mesh"/>
    </body>
  </worldbody>
</mujoco>
""",
                encoding="utf-8",
            )

            result = module.build_mesh_safe_model(model_path)

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.stripped_mesh_names, ("waist_link",))
            self.assertTrue(result.model_path.is_file())

            root = ET.parse(result.model_path).getroot()
            asset_mesh_names = [mesh.get("name") for mesh in root.findall("./asset/mesh")]
            self.assertEqual(asset_mesh_names, ["good_mesh"])

            remaining_geom_meshes = [geom.get("mesh") for geom in root.findall(".//geom[@type='mesh']")]
            self.assertEqual(remaining_geom_meshes, ["good_mesh"])

            waist_body = root.find(".//body[@name='waist_link']")
            self.assertIsNotNone(waist_body)
            assert waist_body is not None
            self.assertIsNotNone(waist_body.find("./geom[@name='waist_link_placeholder']"))


if __name__ == "__main__":
    unittest.main()
