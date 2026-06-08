from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "audit_real_lite_resource_pipeline.py"


def load_module():
    module_name = "test_audit_real_lite_resource_pipeline_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_minimal_urdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    policy_joint_xml = []
    for joint_name in load_module().POLICY_JOINT_NAMES:
        child = joint_name.replace("_joint", "_link")
        policy_joint_xml.append(
            f"""
  <link name="{child}">
    <inertial><mass value="1.0"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="{joint_name}" type="revolute">
    <parent link="pelvis"/>
    <child link="{child}"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.2" upper="3.2" effort="100" velocity="10"/>
  </joint>
"""
        )
    path.write_text(
        """
<robot name="real_lite">
  <link name="pelvis">
    <inertial><mass value="5.0"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="ankle_roll_l_link"><collision><geometry><box size="0.2 0.1 0.03"/></geometry></collision></link>
  <link name="ankle_roll_r_link"><collision><geometry><box size="0.2 0.1 0.03"/></geometry></collision></link>
"""
        + "\n".join(policy_joint_xml)
        + "\n</robot>\n",
        encoding="utf-8",
    )


class AuditRealLiteResourcePipelineTests(unittest.TestCase):
    def test_audit_reports_missing_freebase_usd_as_blocker(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            asset_root = Path(tmp_dir)
            (asset_root / "meshes").mkdir()
            urdf_path = asset_root / "urdf" / "humanoid_publish.urdf"
            _write_minimal_urdf(urdf_path)

            report = module.build_audit_report(asset_root, urdf_path, None)

            issue_codes = {issue["code"] for issue in report["issues"]}
            self.assertIn("FREE_BASE_USD_MISSING", issue_codes)

    def test_markdown_report_contains_server_commands(self) -> None:
        module = load_module()
        report = {
            "asset_root": "asset",
            "urdf": {"path": "robot.urdf"},
            "mjcf": None,
            "issues": [],
            "status": {"blocker_count": 0, "warning_count": 0},
            "usd": {
                "free_base": {"usd_exists": True, "has_root_joint": False, "has_fixed_token": False},
                "fixed_base": {"usd_exists": True, "has_root_joint": True, "has_fixed_token": True},
            },
            "mass": {"urdf_total_mass": 1.0},
            "policy": {"policy_joint_count": 20},
        }

        markdown = module.format_markdown_report(report)

        self.assertIn("tools/reexport_real_lite_usd.py", markdown)
        self.assertIn("tools/isaac_standing_diagnostic.py", markdown)


if __name__ == "__main__":
    unittest.main()
