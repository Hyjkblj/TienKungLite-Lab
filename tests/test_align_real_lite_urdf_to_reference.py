from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "align_real_lite_urdf_to_reference.py"


def load_module():
    module_name = "test_align_real_lite_urdf_to_reference_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_candidate_urdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
<robot name="real_lite">
  <link name="pelvis"/>
  <link name="ankle_pitch_l_link"/>
  <link name="ankle_roll_l_link">
    <collision name="mesh_foot_l"><geometry><mesh filename="../meshes/ankle_roll_l_link.STL"/></geometry></collision>
  </link>
  <link name="ankle_pitch_r_link"/>
  <link name="ankle_roll_r_link">
    <collision name="mesh_foot_r"><geometry><mesh filename="../meshes/ankle_roll_r_link.STL"/></geometry></collision>
  </link>
  <joint name="ankle_roll_l_joint" type="revolute">
    <parent link="ankle_pitch_l_link"/>
    <child link="ankle_roll_l_link"/>
    <limit lower="-0.1" upper="0.1" effort="1" velocity="2"/>
  </joint>
  <joint name="ankle_roll_r_joint" type="revolute">
    <parent link="ankle_pitch_r_link"/>
    <child link="ankle_roll_r_link"/>
    <limit lower="-0.2" upper="0.2" effort="3" velocity="4"/>
  </joint>
</robot>
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _collision_tags(root: ET.Element, link_name: str) -> list[str]:
    link = root.find(f"./link[@name='{link_name}']")
    if link is None:
        raise AssertionError(f"link not found: {link_name}")
    tags = []
    for collision in link.findall("collision"):
        geometry = collision.find("geometry")
        if geometry is not None and len(geometry):
            tags.append(geometry[0].tag)
    return tags


def _collision_elements(root: ET.Element, link_name: str) -> list[ET.Element]:
    link = root.find(f"./link[@name='{link_name}']")
    if link is None:
        raise AssertionError(f"link not found: {link_name}")
    return list(link.findall("collision"))


class AlignRealLiteUrdfToReferenceTests(unittest.TestCase):
    def test_reference_feet_only_replaces_feet_without_syncing_joint_limits(self) -> None:
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            candidate_urdf = tmp_path / "asset" / "urdf" / "humanoid_publish.urdf"
            output_urdf = tmp_path / "asset" / "urdf" / "humanoid_publish.reference_feet.urdf"
            _write_candidate_urdf(candidate_urdf)

            argv = [
                "align_real_lite_urdf_to_reference.py",
                "--reference-feet-only",
                "--candidate-urdf",
                str(candidate_urdf),
                "--output-urdf",
                str(output_urdf),
            ]
            with mock.patch.object(sys, "argv", argv):
                module.main()

            root = ET.parse(output_urdf).getroot()

        self.assertEqual(_collision_tags(root, "ankle_roll_l_link"), ["cylinder", "cylinder"])
        self.assertEqual(_collision_tags(root, "ankle_roll_r_link"), ["cylinder", "cylinder"])

        left_limit = root.find("./joint[@name='ankle_roll_l_joint']/limit")
        right_limit = root.find("./joint[@name='ankle_roll_r_joint']/limit")
        self.assertIsNotNone(left_limit)
        self.assertIsNotNone(right_limit)
        self.assertEqual(left_limit.get("lower"), "-0.1")
        self.assertEqual(left_limit.get("upper"), "0.1")
        self.assertEqual(right_limit.get("lower"), "-0.2")
        self.assertEqual(right_limit.get("upper"), "0.2")

    def test_reference_feet_only_default_output_name_is_explicit(self) -> None:
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            candidate_urdf = Path(tmp_dir) / "asset" / "urdf" / "humanoid_publish.urdf"
            _write_candidate_urdf(candidate_urdf)

            argv = [
                "align_real_lite_urdf_to_reference.py",
                "--reference-feet-only",
                "--candidate-urdf",
                str(candidate_urdf),
            ]
            with mock.patch.object(sys, "argv", argv):
                module.main()

            self.assertTrue(candidate_urdf.with_name("humanoid_publish.reference_feet.urdf").is_file())

    def test_flat_sole_feet_only_default_output_name_is_explicit(self) -> None:
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            candidate_urdf = Path(tmp_dir) / "asset" / "urdf" / "humanoid_publish.urdf"
            _write_candidate_urdf(candidate_urdf)

            argv = [
                "align_real_lite_urdf_to_reference.py",
                "--flat-sole-feet-only",
                "--candidate-urdf",
                str(candidate_urdf),
            ]
            with mock.patch.object(sys, "argv", argv):
                module.main()

            self.assertTrue(candidate_urdf.with_name("humanoid_publish.flat_sole.urdf").is_file())

    def test_reference_feet_support_overrides_update_cylinder_x_and_length(self) -> None:
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            candidate_urdf = tmp_path / "asset" / "urdf" / "humanoid_publish.urdf"
            output_urdf = tmp_path / "asset" / "urdf" / "humanoid_publish.reference_feet.urdf"
            _write_candidate_urdf(candidate_urdf)

            argv = [
                "align_real_lite_urdf_to_reference.py",
                "--reference-feet-only",
                "--reference-feet-support-x",
                "0.07",
                "--reference-feet-support-length",
                "0.34",
                "--candidate-urdf",
                str(candidate_urdf),
                "--output-urdf",
                str(output_urdf),
            ]
            with mock.patch.object(sys, "argv", argv):
                module.main()

            root = ET.parse(output_urdf).getroot()

        for collision in _collision_elements(root, "ankle_roll_l_link") + _collision_elements(root, "ankle_roll_r_link"):
            origin = collision.find("origin")
            cylinder = collision.find("./geometry/cylinder")
            self.assertIsNotNone(origin)
            self.assertIsNotNone(cylinder)
            self.assertEqual(origin.get("xyz").split()[0], "0.07")
            self.assertEqual(cylinder.get("length"), "0.34")

    def test_flat_sole_feet_only_replaces_feet_with_box_soles_without_syncing_limits(self) -> None:
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            candidate_urdf = tmp_path / "asset" / "urdf" / "humanoid_publish.urdf"
            output_urdf = tmp_path / "asset" / "urdf" / "humanoid_publish.flat_sole.urdf"
            _write_candidate_urdf(candidate_urdf)

            argv = [
                "align_real_lite_urdf_to_reference.py",
                "--flat-sole-feet-only",
                "--flat-sole-size",
                "0.24",
                "0.10",
                "0.03",
                "--flat-sole-origin",
                "0.04",
                "0.0",
                "-0.045",
                "--candidate-urdf",
                str(candidate_urdf),
                "--output-urdf",
                str(output_urdf),
            ]
            with mock.patch.object(sys, "argv", argv):
                module.main()

            root = ET.parse(output_urdf).getroot()

        self.assertEqual(_collision_tags(root, "ankle_roll_l_link"), ["box"])
        self.assertEqual(_collision_tags(root, "ankle_roll_r_link"), ["box"])

        for collision in _collision_elements(root, "ankle_roll_l_link") + _collision_elements(root, "ankle_roll_r_link"):
            origin = collision.find("origin")
            box = collision.find("./geometry/box")
            self.assertIsNotNone(origin)
            self.assertIsNotNone(box)
            self.assertEqual(origin.get("xyz"), "0.04 0 -0.045")
            self.assertEqual(box.get("size"), "0.24 0.1 0.03")

        left_limit = root.find("./joint[@name='ankle_roll_l_joint']/limit")
        self.assertIsNotNone(left_limit)
        self.assertEqual(left_limit.get("lower"), "-0.1")
        self.assertEqual(left_limit.get("upper"), "0.1")


if __name__ == "__main__":
    unittest.main()
