from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_lite_lab.assets import resolve_real_lite_asset_root
from real_lite_lab.constants import DEFAULT_JOINT_POS, POLICY_JOINT_NAMES
from real_lite_lab.usd_asset_validation import (
    FIXED_BASE_USD_REL_PATH,
    FREE_BASE_USD_REL_PATH,
    detect_fixed_root_markers,
    resolve_physics_usd_path,
)


@dataclass(frozen=True)
class Issue:
    level: str
    code: str
    message: str


def _issue(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_urdf(urdf_path: Path) -> dict[str, Any]:
    root = ET.parse(urdf_path).getroot()
    links: dict[str, dict[str, Any]] = {}
    joints: dict[str, dict[str, Any]] = {}

    for link in root.findall("link"):
        name = link.get("name")
        if not name:
            continue

        mass_value = None
        inertial = link.find("inertial")
        if inertial is not None:
            mass_elem = inertial.find("mass")
            mass_value = _float_or_none(None if mass_elem is None else mass_elem.get("value"))

        collision_specs = []
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            geom_tag = None
            geom_attrib = {}
            if geometry is not None and len(geometry):
                geom_tag = geometry[0].tag
                geom_attrib = dict(geometry[0].attrib)
            origin = collision.find("origin")
            collision_specs.append(
                {
                    "name": collision.get("name"),
                    "geometry": {"tag": geom_tag, "attrib": geom_attrib},
                    "origin": {} if origin is None else dict(origin.attrib),
                }
            )

        links[name] = {
            "mass": mass_value,
            "collision_count": len(collision_specs),
            "collision_specs": collision_specs,
        }

    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name:
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        limit = joint.find("limit")
        origin = joint.find("origin")
        joints[name] = {
            "type": joint.get("type"),
            "parent": None if parent is None else parent.get("link"),
            "child": None if child is None else child.get("link"),
            "axis": None if axis is None else axis.get("xyz"),
            "origin": {} if origin is None else dict(origin.attrib),
            "lower": _float_or_none(None if limit is None else limit.get("lower")),
            "upper": _float_or_none(None if limit is None else limit.get("upper")),
            "effort": _float_or_none(None if limit is None else limit.get("effort")),
            "velocity": _float_or_none(None if limit is None else limit.get("velocity")),
        }

    total_mass = sum(float(link["mass"]) for link in links.values() if link["mass"] is not None)
    fixed_joint_count = sum(1 for joint in joints.values() if joint["type"] == "fixed")
    revolute_joint_count = sum(1 for joint in joints.values() if joint["type"] == "revolute")
    return {
        "path": str(urdf_path),
        "links": links,
        "joints": joints,
        "link_count": len(links),
        "joint_count": len(joints),
        "fixed_joint_count": fixed_joint_count,
        "revolute_joint_count": revolute_joint_count,
        "total_mass": total_mass,
        "collision_count": sum(int(link["collision_count"]) for link in links.values()),
        "mesh_collision_count": sum(
            1
            for link in links.values()
            for collision in link["collision_specs"]
            if collision["geometry"]["tag"] == "mesh"
        ),
    }


def _parse_mjcf(mjcf_path: Path) -> dict[str, Any]:
    root = ET.parse(mjcf_path).getroot()
    joints: dict[str, dict[str, Any]] = {}
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if not name or name == "root":
            continue
        range_text = joint.get("range", "")
        range_values = [_float_or_none(item) for item in range_text.split()]
        lower = range_values[0] if len(range_values) >= 1 else None
        upper = range_values[1] if len(range_values) >= 2 else None
        joints[name] = {
            "axis": joint.get("axis"),
            "lower": lower,
            "upper": upper,
            "range": range_text,
        }

    body_masses = []
    for inertial in root.findall(".//inertial"):
        mass = _float_or_none(inertial.get("mass"))
        if mass is not None:
            body_masses.append(mass)

    geoms = [geom.get("name") for geom in root.findall(".//geom") if geom.get("name")]
    return {
        "path": str(mjcf_path),
        "joints": joints,
        "actuator_order": [elem.get("joint") for elem in root.findall("./actuator/position") if elem.get("joint")],
        "jointpos_order": [elem.get("joint") for elem in root.findall("./sensor/jointpos") if elem.get("joint")],
        "jointvel_order": [elem.get("joint") for elem in root.findall("./sensor/jointvel") if elem.get("joint")],
        "total_mass": sum(body_masses),
        "geom_names": geoms,
    }


def _audit_usd(asset_root: Path, issues: list[dict[str, str]]) -> dict[str, Any]:
    results = {}
    for label, rel_path in (("free_base", FREE_BASE_USD_REL_PATH), ("fixed_base", FIXED_BASE_USD_REL_PATH)):
        usd_path = (asset_root / rel_path).resolve()
        physics_path = resolve_physics_usd_path(usd_path)
        markers = detect_fixed_root_markers(physics_path)
        results[label] = {
            "usd_path": str(usd_path),
            "usd_exists": usd_path.is_file(),
            "physics_usd_path": str(physics_path),
            "physics_usd_exists": markers["exists"],
            "has_root_joint": markers["has_root_joint"],
            "has_fixed_token": markers["has_fixed_token"],
        }

    free_base = results["free_base"]
    fixed_base = results["fixed_base"]
    if not free_base["usd_exists"]:
        issues.append(
            _issue(
                "blocker",
                "FREE_BASE_USD_MISSING",
                "Free-base USD is missing; server must export humanoid_publish_free_base before training.",
            )
        )
    elif free_base["has_root_joint"] and free_base["has_fixed_token"]:
        issues.append(
            _issue(
                "blocker",
                "FREE_BASE_USD_FIXED",
                "Free-base USD physics file still contains fixed-root markers.",
            )
        )

    if fixed_base["usd_exists"] and fixed_base["has_root_joint"] and fixed_base["has_fixed_token"]:
        issues.append(
            _issue(
                "warning",
                "LEGACY_FIXED_USD_PRESENT",
                "Legacy fixed-base USD is present; do not use it for free-base standing/walking.",
            )
        )
    return results


def _audit_policy_joints(urdf: dict[str, Any], mjcf: dict[str, Any] | None, issues: list[dict[str, str]]) -> dict[str, Any]:
    urdf_joints = urdf["joints"]
    missing_policy_joints = [name for name in POLICY_JOINT_NAMES if name not in urdf_joints]
    if missing_policy_joints:
        issues.append(
            _issue("blocker", "URDF_POLICY_JOINTS_MISSING", f"URDF missing policy joints: {', '.join(missing_policy_joints)}")
        )

    default_limit_violations = []
    for joint_name in POLICY_JOINT_NAMES:
        joint = urdf_joints.get(joint_name)
        if joint is None:
            continue
        lower = joint["lower"]
        upper = joint["upper"]
        default = float(DEFAULT_JOINT_POS[joint_name])
        if lower is not None and default < lower:
            default_limit_violations.append(f"{joint_name}={default:+.4f} < lower {lower:+.4f}")
        if upper is not None and default > upper:
            default_limit_violations.append(f"{joint_name}={default:+.4f} > upper {upper:+.4f}")
    if default_limit_violations:
        issues.append(
            _issue(
                "blocker",
                "DEFAULT_POSE_OUTSIDE_URDF_LIMITS",
                "Default pose violates URDF joint limits: " + "; ".join(default_limit_violations),
            )
        )

    mjcf_summary = {}
    if mjcf is not None:
        for order_name in ("actuator_order", "jointpos_order", "jointvel_order"):
            order = mjcf[order_name]
            if set(order) != set(POLICY_JOINT_NAMES):
                issues.append(
                    _issue(
                        "blocker",
                        f"MJCF_{order_name.upper()}_MISMATCH",
                        f"MJCF {order_name} does not contain exactly the policy joints.",
                    )
                )
            mjcf_summary[f"{order_name}_same_as_policy"] = order == list(POLICY_JOINT_NAMES)

        range_diffs = []
        for joint_name in POLICY_JOINT_NAMES:
            urdf_joint = urdf_joints.get(joint_name)
            mjcf_joint = mjcf["joints"].get(joint_name)
            if urdf_joint is None or mjcf_joint is None:
                continue
            for key in ("lower", "upper"):
                urdf_value = urdf_joint[key]
                mjcf_value = mjcf_joint[key]
                if urdf_value is None or mjcf_value is None:
                    continue
                if abs(float(urdf_value) - float(mjcf_value)) > 1e-6:
                    range_diffs.append(f"{joint_name}.{key}: URDF={urdf_value:g}, MJCF={mjcf_value:g}")
        if range_diffs:
            issues.append(
                _issue(
                    "blocker",
                    "MJCF_JOINT_LIMITS_DIVERGE",
                    "MJCF joint limits differ from canonical URDF: " + "; ".join(range_diffs[:12]),
                )
            )

    return {
        "policy_joint_count": len(POLICY_JOINT_NAMES),
        "missing_policy_joints": missing_policy_joints,
        "default_limit_violations": default_limit_violations,
        **mjcf_summary,
    }


def _audit_collision_model(urdf: dict[str, Any], mjcf: dict[str, Any] | None, issues: list[dict[str, str]]) -> dict[str, Any]:
    mesh_collision_count = int(urdf["mesh_collision_count"])
    if mesh_collision_count:
        issues.append(
            _issue(
                "warning",
                "URDF_USES_MESH_COLLISIONS",
                f"URDF has {mesh_collision_count} mesh collision elements; prefer simplified primitives for training.",
            )
        )

    foot_links = ("ankle_roll_l_link", "ankle_roll_r_link")
    foot_collision_summary = {}
    for link_name in foot_links:
        link = urdf["links"].get(link_name)
        specs = [] if link is None else link["collision_specs"]
        tags = [spec["geometry"]["tag"] for spec in specs]
        names = [spec["name"] for spec in specs]
        foot_collision_summary[link_name] = {"count": len(specs), "tags": tags, "names": names}
        if not specs:
            issues.append(_issue("blocker", "FOOT_COLLISION_MISSING", f"{link_name} has no collision geometry."))
        elif all(tag == "mesh" for tag in tags):
            issues.append(
                _issue(
                    "warning",
                    "FOOT_COLLISION_IS_MESH",
                    f"{link_name} uses mesh-only foot collision; add toe/sole primitives for stable contact.",
                )
            )

    mjcf_support_geoms = {}
    if mjcf is not None:
        geom_names = set(mjcf["geom_names"])
        mjcf_support_geoms = {
            "sole_left": "sole_left" in geom_names,
            "sole_right": "sole_right" in geom_names,
            "toe1_left": "toe1_left" in geom_names,
            "toe2_left": "toe2_left" in geom_names,
            "toe1_right": "toe1_right" in geom_names,
            "toe2_right": "toe2_right" in geom_names,
        }
        has_sole_pair = mjcf_support_geoms["sole_left"] and mjcf_support_geoms["sole_right"]
        has_toe_pairs = all(
            mjcf_support_geoms[name]
            for name in ("toe1_left", "toe2_left", "toe1_right", "toe2_right")
        )
        if not has_sole_pair and not has_toe_pairs:
            issues.append(
                _issue("blocker", "MJCF_SUPPORT_GEOMS_MISSING", "MJCF is missing recognized foot support geoms.")
            )

    return {
        "urdf_collision_count": urdf["collision_count"],
        "urdf_mesh_collision_count": mesh_collision_count,
        "foot_collision_summary": foot_collision_summary,
        "mjcf_support_geoms": mjcf_support_geoms,
    }


def _audit_mass(urdf: dict[str, Any], mjcf: dict[str, Any] | None, issues: list[dict[str, str]]) -> dict[str, Any]:
    summary = {"urdf_total_mass": float(urdf["total_mass"])}
    if mjcf is None:
        return summary

    mjcf_mass = float(mjcf["total_mass"])
    urdf_mass = float(urdf["total_mass"])
    delta = mjcf_mass - urdf_mass
    summary.update({"mjcf_total_mass": mjcf_mass, "mjcf_minus_urdf_mass": delta})
    if abs(delta) > 1e-5:
        issues.append(
            _issue(
                "blocker",
                "MJCF_TOTAL_MASS_DIVERGES",
                f"MJCF total mass differs from URDF by {delta:+.6f} kg.",
            )
        )
    return summary


def build_audit_report(asset_root: Path, urdf_path: Path, mjcf_path: Path | None) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    urdf = _parse_urdf(urdf_path)
    mjcf = _parse_mjcf(mjcf_path) if mjcf_path is not None and mjcf_path.is_file() else None
    if mjcf_path is not None and not mjcf_path.is_file():
        issues.append(_issue("warning", "MJCF_MISSING", f"MJCF file not found: {mjcf_path}"))

    usd = _audit_usd(asset_root, issues)
    policy = _audit_policy_joints(urdf, mjcf, issues)
    collision = _audit_collision_model(urdf, mjcf, issues)
    mass = _audit_mass(urdf, mjcf, issues)
    blocker_count = sum(1 for item in issues if item["level"] == "blocker")
    warning_count = sum(1 for item in issues if item["level"] == "warning")

    return {
        "asset_root": str(asset_root),
        "urdf": {
            "path": urdf["path"],
            "link_count": urdf["link_count"],
            "joint_count": urdf["joint_count"],
            "fixed_joint_count": urdf["fixed_joint_count"],
            "revolute_joint_count": urdf["revolute_joint_count"],
        },
        "mjcf": None if mjcf is None else {"path": mjcf["path"]},
        "usd": usd,
        "policy": policy,
        "collision": collision,
        "mass": mass,
        "issues": issues,
        "status": {
            "blocker_count": blocker_count,
            "warning_count": warning_count,
            "ready_for_server_freebase_export": blocker_count == 0 or all(
                issue["code"] in {"FREE_BASE_USD_MISSING", "LEGACY_FIXED_USD_PRESENT"}
                for issue in issues
            ),
        },
    }


def _markdown_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Item | Value |", "| --- | --- |"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines)


def format_markdown_report(report: dict[str, Any]) -> str:
    issues = report["issues"]
    blocker_count = report["status"]["blocker_count"]
    warning_count = report["status"]["warning_count"]
    lines = [
        "# Real Lite Resource Pipeline Audit",
        "",
        _markdown_table(
            [
                ("Asset root", f"`{report['asset_root']}`"),
                ("URDF", f"`{report['urdf']['path']}`"),
                ("MJCF", "missing" if report["mjcf"] is None else f"`{report['mjcf']['path']}`"),
                ("Blockers", str(blocker_count)),
                ("Warnings", str(warning_count)),
            ]
        ),
        "",
        "## Issues",
    ]
    if not issues:
        lines.append("")
        lines.append("No blockers or warnings detected.")
    else:
        for issue in issues:
            lines.append(f"- **{issue['level'].upper()} `{issue['code']}`**: {issue['message']}")

    usd = report["usd"]
    lines.extend(
        [
            "",
            "## USD Candidates",
            _markdown_table(
                [
                    ("Free-base USD exists", str(usd["free_base"]["usd_exists"])),
                    ("Free-base physics fixed markers", str(usd["free_base"]["has_root_joint"] and usd["free_base"]["has_fixed_token"])),
                    ("Fixed-base USD exists", str(usd["fixed_base"]["usd_exists"])),
                    ("Fixed-base physics fixed markers", str(usd["fixed_base"]["has_root_joint"] and usd["fixed_base"]["has_fixed_token"])),
                ]
            ),
            "",
            "## Mass",
            _markdown_table([(key, f"{value:.6f}" if isinstance(value, float) else str(value)) for key, value in report["mass"].items()]),
            "",
            "## Policy Joint Order",
            _markdown_table([(key, str(value)) for key, value in report["policy"].items() if not isinstance(value, list)]),
            "",
            "## Next Local Command",
            "```bash",
            "python tools/audit_real_lite_resource_pipeline.py --strict",
            "```",
            "",
            "## Next Server Commands",
            "```bash",
            "python tools/reexport_real_lite_usd.py --headless --force",
            "export TIENKUNG_LITE_USD_REL_PATH=urdf/humanoid_publish_free_base/humanoid_publish_free_base.usd",
            "python tools/isaac_standing_diagnostic.py --task walk_real_lite --headless --duration 6 --trace_out logs/standing/isaac_freebase_baseline.npz",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local Real Lite resources before server-side Isaac export/tests.")
    parser.add_argument("--asset-root", default=None, help="Asset root containing meshes/ and urdf/.")
    parser.add_argument("--urdf", default=None, help="Canonical URDF path. Defaults to <asset-root>/urdf/humanoid_publish.urdf.")
    parser.add_argument("--mjcf", default=str(ROOT / "mjcf" / "real_lite.xml"), help="Optional MJCF path to audit.")
    parser.add_argument("--report-md", default=str(ROOT / "logs" / "resource_audit" / "real_lite_resource_audit.md"))
    parser.add_argument("--report-json", default=str(ROOT / "logs" / "resource_audit" / "real_lite_resource_audit.json"))
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when blockers remain.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asset_root = Path(args.asset_root).expanduser().resolve() if args.asset_root else resolve_real_lite_asset_root()
    urdf_path = Path(args.urdf).expanduser().resolve() if args.urdf else asset_root / "urdf" / "humanoid_publish.urdf"
    mjcf_path = Path(args.mjcf).expanduser().resolve() if args.mjcf else None
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Canonical URDF not found: {urdf_path}")

    report = build_audit_report(asset_root, urdf_path, mjcf_path)
    report_md_path = Path(args.report_md).expanduser().resolve()
    report_json_path = Path(args.report_json).expanduser().resolve()
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(format_markdown_report(report), encoding="utf-8")
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[INFO] Wrote Markdown report: {report_md_path}")
    print(f"[INFO] Wrote JSON report: {report_json_path}")
    print(
        "[INFO] Audit status: "
        f"blockers={report['status']['blocker_count']}, warnings={report['status']['warning_count']}"
    )
    for issue in report["issues"]:
        print(f"[{issue['level'].upper()}] {issue['code']}: {issue['message']}")

    if args.strict and report["status"]["blocker_count"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
