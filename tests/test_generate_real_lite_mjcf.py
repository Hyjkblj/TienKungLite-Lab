from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "generate_real_lite_mjcf.py"


def load_generate_module(asset_root: Path):
    spec = importlib.util.spec_from_file_location("test_generate_real_lite_mjcf_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, {"TIENKUNG_LITE_ASSET_ROOT": str(asset_root)}, clear=False):
        spec.loader.exec_module(module)
    return module


class GenerateRealLiteMjcfTests(unittest.TestCase):
    def test_prepare_mujoco_mesh_cache_converts_ascii_stl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            asset_root = tmp_path / "asset_root"
            meshes_dir = asset_root / "meshes"
            urdf_dir = asset_root / "urdf"
            meshes_dir.mkdir(parents=True)
            urdf_dir.mkdir(parents=True)
            (urdf_dir / "humanoid_publish.urdf").write_text("<robot />", encoding="utf-8")

            ascii_stl = """solid triangle
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid triangle
"""
            (meshes_dir / "triangle.STL").write_text(ascii_stl, encoding="utf-8")

            module = load_generate_module(asset_root)
            generated_dir = tmp_path / "generated_meshes"

            with mock.patch.object(module, "GENERATED_MESH_DIR", generated_dir):
                module._prepare_mujoco_mesh_cache()

            output_path = generated_dir / "triangle.STL"
            output_bytes = output_path.read_bytes()
            self.assertEqual(len(output_bytes), 84 + 50)
            self.assertFalse(output_bytes.startswith(b"solid"))


if __name__ == "__main__":
    unittest.main()
