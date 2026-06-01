from __future__ import annotations

import importlib.util
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
JOINT_ORDER_MODULE_PATH = REPO_ROOT / "real_lite_lab" / "joint_order.py"
CONSTANTS_MODULE_PATH = REPO_ROOT / "real_lite_lab" / "constants.py"


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class JointOrderTests(unittest.TestCase):
    def test_build_target_order_indices_reorders_policy_to_actuator_order(self) -> None:
        joint_order_module = load_module(JOINT_ORDER_MODULE_PATH, "test_joint_order_module")
        source_order = ("hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint")
        target_order = ("hip_roll_l_joint", "hip_yaw_l_joint", "hip_pitch_l_joint")

        indices = joint_order_module.build_target_order_indices(source_order, target_order)

        self.assertEqual(indices, [0, 2, 1])

    def test_mjcf_actuator_order_differs_from_policy_order(self) -> None:
        constants_module = load_module(CONSTANTS_MODULE_PATH, "test_constants_module")
        mjcf_path = REPO_ROOT / "mjcf" / "real_lite.xml"
        mjcf_root = ET.parse(mjcf_path).getroot()

        actuator_order = [elem.attrib["joint"] for elem in mjcf_root.findall("./actuator/position")]
        policy_order = list(constants_module.POLICY_JOINT_NAMES)

        self.assertNotEqual(actuator_order, policy_order)
        self.assertEqual(set(actuator_order), set(policy_order))
        self.assertEqual(actuator_order[:3], ["hip_roll_l_joint", "hip_yaw_l_joint", "hip_pitch_l_joint"])
        self.assertEqual(policy_order[:3], ["hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint"])


if __name__ == "__main__":
    unittest.main()
