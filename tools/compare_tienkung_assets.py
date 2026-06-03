from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_lite_lab.assets import resolve_real_lite_asset_root


JOINT_ALIAS_MAP = {
    "elbow_pitch_l_joint": "elbow_l_joint",
    "elbow_pitch_r_joint": "elbow_r_joint",
}
LINK_ALIAS_MAP = {
    "elbow_pitch_l_link": "elbow_l_link",
    "elbow_pitch_r_link": "elbow_r_link",
}


def _log(message: str) -> None:
    print(message, flush=True)


def _detect_fixed_root_markers(physics_usd_path: Path) -> dict[str, bool]:
    if not physics_usd_path.is_file():
        return {"exists": False, "has_root_joint": False, "has_fixed_token": False}

    usd_bytes = physics_usd_path.read_bytes()
    return {
        "exists": True,
        "has_root_joint": b"root_joint" in usd_bytes,
        "has_fixed_token": b"Fixed" in usd_bytes,
    }


def _parse_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key and not key.startswith("-"):
            values[key.strip()] = value.strip()
    return values


def _parse_urdf(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()

    links: dict[str, dict[str, object]] = {}
    joints: dict[str, dict[str, object]] = {}

    total_mass = 0.0
    total_collision_elems = 0
    nonzero_collision_links = 0

    for link in root.findall("link"):
        link_name = link.get("name")
        if link_name is None:
            continue

        inertial = link.find("inertial")
        mass_value = None
        if inertial is not None:
            mass_elem = inertial.find("mass")
            if mass_elem is not None and mass_elem.get("value") is not None:
                mass_value = float(mass_elem.get("value"))
                total_mass += mass_value

        collision_elems = link.findall("collision")
        collision_count = len(collision_elems)
        total_collision_elems += collision_count
        if collision_count > 0:
            nonzero_collision_links += 1

        links[link_name] = {
            "mass": mass_value,
            "collision_count": collision_count,
        }

    fixed_joints = 0
    revolute_joints = 0
    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        if joint_name is None:
            continue

        joint_type = joint.get("type")
        if joint_type == "fixed":
            fixed_joints += 1
        elif joint_type == "revolute":
            revolute_joints += 1

        axis_elem = joint.find("axis")
        limit_elem = joint.find("limit")
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")

        joints[joint_name] = {
            "type": joint_type,
            "axis": None if axis_elem is None else axis_elem.get("xyz"),
            "lower": None if limit_elem is None else limit_elem.get("lower"),
            "upper": None if limit_elem is None else limit_elem.get("upper"),
            "effort": None if limit_elem is None else limit_elem.get("effort"),
            "velocity": None if limit_elem is None else limit_elem.get("velocity"),
            "parent": None if parent_elem is None else parent_elem.get("link"),
            "child": None if child_elem is None else child_elem.get("link"),
        }

    return {
        "path": path,
        "links": links,
        "joints": joints,
        "link_count": len(links),
        "joint_count": len(joints),
        "fixed_joint_count": fixed_joints,
        "revolute_joint_count": revolute_joints,
        "total_mass": total_mass,
        "nonzero_collision_links": nonzero_collision_links,
        "total_collision_elems": total_collision_elems,
    }


def _resolve_physics_usd_path(usd_path: Path) -> Path:
    candidates = [
        usd_path.parent / "configuration" / f"{usd_path.stem}_physics.usd",
        usd_path.parent / "configuration" / "humanoid_publish_physics.usd",
        usd_path.parent / "configuration" / "tienkung2_lite_physics.usd",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _canonical_joint_name(name: str, *, side: str) -> str:
    if side == "reference":
        return JOINT_ALIAS_MAP.get(name, name)
    return name


def _canonical_link_name(name: str, *, side: str) -> str:
    if side == "reference":
        return LINK_ALIAS_MAP.get(name, name)
    return name


def _compare_joint_properties(reference: dict[str, object], candidate: dict[str, object]) -> list[dict[str, object]]:
    ref_joints = {
        _canonical_joint_name(name, side="reference"): payload for name, payload in reference["joints"].items()  # type: ignore[index]
    }
    cand_joints = {
        _canonical_joint_name(name, side="candidate"): payload for name, payload in candidate["joints"].items()  # type: ignore[index]
    }

    rows: list[dict[str, object]] = []
    fields = ("type", "axis", "lower", "upper", "effort", "velocity", "parent", "child")
    for name in sorted(set(ref_joints) & set(cand_joints)):
        ref_payload = ref_joints[name]
        cand_payload = cand_joints[name]
        changed_fields = [field for field in fields if ref_payload.get(field) != cand_payload.get(field)]
        if changed_fields:
            rows.append(
                {
                    "name": name,
                    "changed_fields": changed_fields,
                    "reference": ref_payload,
                    "candidate": cand_payload,
                }
            )
    return rows


def _compare_link_properties(reference: dict[str, object], candidate: dict[str, object]) -> tuple[list[dict[str, object]], list[str], list[str]]:
    ref_links = {
        _canonical_link_name(name, side="reference"): payload for name, payload in reference["links"].items()  # type: ignore[index]
    }
    cand_links = {
        _canonical_link_name(name, side="candidate"): payload for name, payload in candidate["links"].items()  # type: ignore[index]
    }

    mass_rows: list[dict[str, object]] = []
    for name in sorted(set(ref_links) & set(cand_links)):
        ref_mass = ref_links[name].get("mass")
        cand_mass = cand_links[name].get("mass")
        if ref_mass is None or cand_mass is None:
            continue
        delta = float(cand_mass) - float(ref_mass)
        if abs(delta) > 1e-6:
            mass_rows.append(
                {
                    "name": name,
                    "reference_mass": float(ref_mass),
                    "candidate_mass": float(cand_mass),
                    "delta": delta,
                }
            )

    only_reference = sorted(set(ref_links) - set(cand_links))
    only_candidate = sorted(set(cand_links) - set(ref_links))
    return sorted(mass_rows, key=lambda row: abs(float(row["delta"])), reverse=True), only_reference, only_candidate


def _compare_collision_counts(reference: dict[str, object], candidate: dict[str, object]) -> list[dict[str, object]]:
    ref_links = {
        _canonical_link_name(name, side="reference"): payload for name, payload in reference["links"].items()  # type: ignore[index]
    }
    cand_links = {
        _canonical_link_name(name, side="candidate"): payload for name, payload in candidate["links"].items()  # type: ignore[index]
    }

    rows: list[dict[str, object]] = []
    for name in sorted(set(ref_links) | set(cand_links)):
        ref_collision_count = int(ref_links.get(name, {}).get("collision_count", 0))
        cand_collision_count = int(cand_links.get(name, {}).get("collision_count", 0))
        if ref_collision_count != cand_collision_count:
            rows.append(
                {
                    "name": name,
                    "reference_collision_count": ref_collision_count,
                    "candidate_collision_count": cand_collision_count,
                }
            )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the original TienKung-Lab asset against the current Real Lite asset.")
    parser.add_argument(
        "--reference-asset-root",
        type=str,
        default=str((ROOT.parent / "TienKung-Lab" / "legged_lab" / "assets" / "tienkung2_lite").resolve()),
        help="Path to the reference TienKung-Lab asset root.",
    )
    parser.add_argument(
        "--candidate-asset-root",
        type=str,
        default=None,
        help="Path to the candidate Real Lite asset root. Defaults to resolve_real_lite_asset_root().",
    )
    parser.add_argument(
        "--reference-urdf",
        type=str,
        default=None,
        help="Explicit reference URDF path. Defaults to <reference-asset-root>/urdf/tienkung2_lite.urdf.",
    )
    parser.add_argument(
        "--candidate-urdf",
        type=str,
        default=None,
        help="Explicit candidate URDF path. Defaults to <candidate-asset-root>/urdf/humanoid_publish.urdf.",
    )
    parser.add_argument(
        "--reference-usd",
        type=str,
        default=None,
        help="Explicit reference USD path. Defaults to <reference-asset-root>/usd/tienkung2_lite.usd.",
    )
    parser.add_argument(
        "--candidate-default-usd",
        type=str,
        default=None,
        help="Explicit candidate default USD path. Defaults to <candidate-asset-root>/urdf/humanoid_publish/humanoid_publish.usd.",
    )
    parser.add_argument(
        "--candidate-freebase-usd",
        type=str,
        default=None,
        help="Explicit candidate free-base USD path. Defaults to <candidate-asset-root>/urdf/humanoid_publish_free_base/humanoid_publish_free_base.usd.",
    )
    parser.add_argument(
        "--top-mass-deltas",
        type=int,
        default=12,
        help="How many largest per-link mass deltas to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    reference_root = Path(args.reference_asset_root).expanduser().resolve()
    candidate_root = Path(args.candidate_asset_root).expanduser().resolve() if args.candidate_asset_root else resolve_real_lite_asset_root()

    reference_urdf = Path(args.reference_urdf).expanduser().resolve() if args.reference_urdf else reference_root / "urdf" / "tienkung2_lite.urdf"
    candidate_urdf = Path(args.candidate_urdf).expanduser().resolve() if args.candidate_urdf else candidate_root / "urdf" / "humanoid_publish.urdf"

    reference_usd = Path(args.reference_usd).expanduser().resolve() if args.reference_usd else reference_root / "usd" / "tienkung2_lite.usd"
    candidate_default_usd = (
        Path(args.candidate_default_usd).expanduser().resolve()
        if args.candidate_default_usd
        else candidate_root / "urdf" / "humanoid_publish" / "humanoid_publish.usd"
    )
    candidate_freebase_usd = (
        Path(args.candidate_freebase_usd).expanduser().resolve()
        if args.candidate_freebase_usd
        else candidate_root / "urdf" / "humanoid_publish_free_base" / "humanoid_publish_free_base.usd"
    )

    reference = _parse_urdf(reference_urdf)
    candidate = _parse_urdf(candidate_urdf)

    reference_cfg = _parse_simple_yaml(reference_root / "usd" / "config.yaml")
    candidate_default_cfg = _parse_simple_yaml(candidate_root / "urdf" / "humanoid_publish" / "config.yaml")
    candidate_freebase_cfg = _parse_simple_yaml(candidate_root / "urdf" / "humanoid_publish_free_base" / "config.yaml")

    reference_markers = _detect_fixed_root_markers(_resolve_physics_usd_path(reference_usd))
    candidate_default_markers = _detect_fixed_root_markers(_resolve_physics_usd_path(candidate_default_usd))
    candidate_freebase_markers = _detect_fixed_root_markers(_resolve_physics_usd_path(candidate_freebase_usd))

    mass_rows, only_reference_links, only_candidate_links = _compare_link_properties(reference, candidate)
    joint_rows = _compare_joint_properties(reference, candidate)
    collision_rows = _compare_collision_counts(reference, candidate)

    _log("[SUMMARY]")
    _log(f"reference_asset_root: {reference_root}")
    _log(f"candidate_asset_root: {candidate_root}")
    _log(f"reference_urdf: {reference_urdf}")
    _log(f"candidate_urdf: {candidate_urdf}")
    _log(
        "reference_urdf_summary: "
        f"links={reference['link_count']}, joints={reference['joint_count']}, revolute={reference['revolute_joint_count']}, "
        f"fixed={reference['fixed_joint_count']}, total_mass={float(reference['total_mass']):.6f}, "
        f"collision_links={reference['nonzero_collision_links']}, collision_elems={reference['total_collision_elems']}"
    )
    _log(
        "candidate_urdf_summary: "
        f"links={candidate['link_count']}, joints={candidate['joint_count']}, revolute={candidate['revolute_joint_count']}, "
        f"fixed={candidate['fixed_joint_count']}, total_mass={float(candidate['total_mass']):.6f}, "
        f"collision_links={candidate['nonzero_collision_links']}, collision_elems={candidate['total_collision_elems']}"
    )
    _log(f"total_mass_delta(candidate-reference): {float(candidate['total_mass']) - float(reference['total_mass']):+.6f}")

    _log("[USD]")
    _log(
        "reference_usd_markers: "
        f"path={_resolve_physics_usd_path(reference_usd)}, exists={reference_markers['exists']}, "
        f"root_joint={reference_markers['has_root_joint']}, Fixed={reference_markers['has_fixed_token']}"
    )
    _log(
        "candidate_default_usd_markers: "
        f"path={_resolve_physics_usd_path(candidate_default_usd)}, exists={candidate_default_markers['exists']}, "
        f"root_joint={candidate_default_markers['has_root_joint']}, Fixed={candidate_default_markers['has_fixed_token']}"
    )
    _log(
        "candidate_freebase_usd_markers: "
        f"path={_resolve_physics_usd_path(candidate_freebase_usd)}, exists={candidate_freebase_markers['exists']}, "
        f"root_joint={candidate_freebase_markers['has_root_joint']}, Fixed={candidate_freebase_markers['has_fixed_token']}"
    )

    _log("[USD_CONFIG]")
    if reference_cfg:
        _log(
            "reference_usd_config: "
            f"fix_base={reference_cfg.get('fix_base')}, merge_fixed_joints={reference_cfg.get('merge_fixed_joints')}, "
            f"self_collision={reference_cfg.get('self_collision')}, collision_from_visuals={reference_cfg.get('collision_from_visuals')}"
        )
    if candidate_default_cfg:
        _log(
            "candidate_default_usd_config: "
            f"fix_base={candidate_default_cfg.get('fix_base')}, merge_fixed_joints={candidate_default_cfg.get('merge_fixed_joints')}, "
            f"self_collision={candidate_default_cfg.get('self_collision')}, collision_from_visuals={candidate_default_cfg.get('collision_from_visuals')}"
        )
    if candidate_freebase_cfg:
        _log(
            "candidate_freebase_usd_config: "
            f"fix_base={candidate_freebase_cfg.get('fix_base')}, merge_fixed_joints={candidate_freebase_cfg.get('merge_fixed_joints')}, "
            f"self_collision={candidate_freebase_cfg.get('self_collision')}, collision_from_visuals={candidate_freebase_cfg.get('collision_from_visuals')}"
        )

    _log("[LINK_SET]")
    _log(f"links_only_in_reference({len(only_reference_links)}): {', '.join(only_reference_links) if only_reference_links else 'none'}")
    _log(f"links_only_in_candidate({len(only_candidate_links)}): {', '.join(only_candidate_links) if only_candidate_links else 'none'}")

    _log("[MASS_DELTAS]")
    for row in mass_rows[: args.top_mass_deltas]:
        _log(
            f"{row['name']}: reference_mass={float(row['reference_mass']):.6f}, "
            f"candidate_mass={float(row['candidate_mass']):.6f}, delta={float(row['delta']):+.6f}"
        )

    _log("[JOINT_DIFFS]")
    _log(f"differing_joint_count: {len(joint_rows)}")
    for row in joint_rows:
        changed_fields = ", ".join(row["changed_fields"])
        _log(
            f"{row['name']}: changed={changed_fields}; "
            f"reference={row['reference']}; candidate={row['candidate']}"
        )

    _log("[COLLISION_DIFFS]")
    _log(f"differing_collision_link_count: {len(collision_rows)}")
    for row in collision_rows:
        _log(
            f"{row['name']}: "
            f"reference_collision_count={row['reference_collision_count']}, "
            f"candidate_collision_count={row['candidate_collision_count']}"
        )


if __name__ == "__main__":
    main()
