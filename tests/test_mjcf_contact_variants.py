from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "real_lite_lab" / "mjcf_contact_variants.py"


def load_contact_variants_module():
    module_name = "test_mjcf_contact_variants_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class MjcfContactVariantsTests(unittest.TestCase):
    def test_build_toe_rail_contact_model_replaces_sole_boxes(self) -> None:
        module = load_contact_variants_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "real_lite.xml"
            output_path = Path(tmp_dir) / "real_lite.toe_rails.xml"
            model_path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
  <worldbody>
    <body name="ankle_roll_l_link">
      <geom name="sole_left" contype="2" conaffinity="1" size="0.115 0.040 0.015" pos="0.035 0 -0.042" type="box" />
    </body>
    <body name="ankle_roll_r_link">
      <geom name="sole_right" contype="2" conaffinity="1" size="0.115 0.040 0.015" pos="0.035 0 -0.042" type="box" />
    </body>
  </worldbody>
</mujoco>
""",
                encoding="utf-8",
            )

            result = module.build_toe_rail_contact_model(model_path, output_path=output_path)

            self.assertEqual(result.variant_name, "toe_rails")
            self.assertEqual(result.model_path, output_path)

            tree = ET.parse(output_path)
            root = tree.getroot()
            self.assertIsNone(root.find(".//geom[@name='sole_left']"))
            self.assertIsNone(root.find(".//geom[@name='sole_right']"))
            self.assertIsNotNone(root.find(".//geom[@name='toe1_left']"))
            self.assertIsNotNone(root.find(".//geom[@name='toe2_left']"))
            self.assertIsNotNone(root.find(".//geom[@name='toe1_right']"))
            self.assertIsNotNone(root.find(".//geom[@name='toe2_right']"))


if __name__ == "__main__":
    unittest.main()
