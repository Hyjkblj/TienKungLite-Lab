from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_lite_lab.assets import resolve_real_lite_asset_root
from real_lite_lab.reference_tienkung2_lite import REFERENCE_TIENKUNG2_LITE_SNAPSHOT


REFERENCE_ASSET_ROOT_ENV_VAR = "TIENKUNG_REFERENCE_ASSET_ROOT"
JOINT_ALIAS_MAP = {
    "elbow_pitch_l_joint": "elbow_l_joint",
    "elbow_pitch_r_joint": "elbow_r_joint",
}
LINK_ALIAS_MAP = {
    "elbow_pitch_l_link": "elbow_l_link",
    "elbow_pitch_r_link": "elbow_r_link",
}
INERTIA_ATTRS = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
REFERENCE_ANKLE_ROLL_COLLISION_SPECS = {
    "ankle_roll_l_link": (
        {
            "origin": {"xyz": "0.035 0.025 -0.042", "rpy": "0 1.5708 0"},
            "geometry": ("cylinder", {"length": "0.23", "radius": "0.015"}),
        },
        {
            "origin": {"xyz": "0.035 -0.025 -0.042", "rpy": "0 1.5708 0"},
            "geometry": ("cylinder", {"length": "0.23", "radius": "0.015"}),
        },
    ),
    "ankle_roll_r_link": (
        {
            "origin": {"xyz": "0.035 0.025 -0.042", "rpy": "0 1.5708 0"},
            "geometry": ("cylinder", {"length": "0.23", "radius": "0.015"}),
        },
        {
            "origin": {"xyz": "0.035 -0.025 -0.042", "rpy": "0 1.5708 0"},
            "geometry": ("cylinder", {"length": "0.23", "radius": "0.015"}),
        },
    ),
}


def _log(message: str) -> None:
    print(message, flush=True)


def _default_reference_asset_root() -> Path:
    env_text = os.environ.get(REFERENCE_ASSET_ROOT_ENV_VAR)
    if env_text:
        return Path(env_text).expanduser().resolve()
    return (ROOT.parent / "TienKung-Lab" / "legged_lab" / "assets" / "tienkung2_lite").resolve()


def _cli_flag_present(flag_name: str) -> bool:
    return any(argument == flag_name or argument.startswith(f"{flag_name}=") for argument in sys.argv[1:])


def _reference_override_requested() -> bool:
    return (
        _cli_flag_present("--reference-asset-root")
        or _cli_flag_present("--reference-urdf")
        or bool(os.environ.get(REFERENCE_ASSET_ROOT_ENV_VAR))
    )


def _canonical_joint_name(name: str, *, side: str) -> str:
    if side == "reference":
        return JOINT_ALIAS_MAP.get(name, name)
    return name


def _canonical_link_name(name: str, *, side: str) -> str:
    if side == "reference":
        return LINK_ALIAS_MAP.get(name, name)
    return name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a safer candidate URDF by aligning selected properties from the original "
            "TienKung-Lab asset into the current Real Lite URDF."
        )
    )
    parser.add_argument(
        "--reference-asset-root",
        type=str,
        default=str(_default_reference_asset_root()),
        help=f"Path to the reference TienKung-Lab asset root. Can also be set with {REFERENCE_ASSET_ROOT_ENV_VAR}.",
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
        "--output-urdf",
        type=str,
        default=None,
        help="Output URDF path. Defaults to <candidate-urdf stem>.reference_aligned.urdf.",
    )
    parser.add_argument(
        "--sync-collision-topology",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove candidate collisions from links that have no collision in the reference asset.",
    )
    parser.add_argument(
        "--sync-joint-limits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy lower/upper/effort/velocity values from matching reference joints.",
    )
    parser.add_argument(
        "--sync-link-mass",
        action="store_true",
        help="Copy reference mass values and scale candidate inertia tensors by the same ratio.",
    )
    parser.add_argument(
        "--zero-candidate-only-fixed-link-mass",
        action="store_true",
        help=(
            "Set mass and inertia to zero for candidate-only fixed links such as waist_link. "
            "Useful after --sync-link-mass to avoid double-counting mass that is already folded into "
            "the reference root chain."
        ),
    )
    parser.add_argument(
        "--replace-ankle-roll-collisions-with-reference",
        action="store_true",
        help=(
            "Replace ankle_roll link collision geometry with the original TienKung-Lab dual-cylinder "
            "collision layout instead of the imported mesh collision."
        ),
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    reference_root = Path(args.reference_asset_root).expanduser().resolve()
    candidate_root = Path(args.candidate_asset_root).expanduser().resolve() if args.candidate_asset_root else resolve_real_lite_asset_root()
    reference_urdf = (
        Path(args.reference_urdf).expanduser().resolve()
        if args.reference_urdf
        else reference_root / "urdf" / "tienkung2_lite.urdf"
    )
    candidate_urdf = (
        Path(args.candidate_urdf).expanduser().resolve()
        if args.candidate_urdf
        else candidate_root / "urdf" / "humanoid_publish.urdf"
    )
    output_urdf = (
        Path(args.output_urdf).expanduser().resolve()
        if args.output_urdf
        else candidate_urdf.with_name(f"{candidate_urdf.stem}.reference_aligned.urdf")
    )

    if not reference_urdf.is_file() and _reference_override_requested():
        raise FileNotFoundError(f"Reference URDF not found: {reference_urdf}")
    if not candidate_urdf.is_file():
        raise FileNotFoundError(f"Candidate URDF not found: {candidate_urdf}")

    return reference_urdf, candidate_urdf, output_urdf


def _build_reference_link_collision_counts(root: ET.Element) -> dict[str, int]:
    counts: dict[str, int] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if not name:
            continue
        counts[_canonical_link_name(name, side="reference")] = len(link.findall("collision"))
    return counts


def _build_reference_joint_limits(root: ET.Element) -> dict[str, dict[str, str]]:
    values: dict[str, dict[str, str]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        payload = {}
        for attr in ("lower", "upper", "effort", "velocity"):
            value = limit.get(attr)
            if value is not None:
                payload[attr] = value
        if payload:
            values[_canonical_joint_name(name, side="reference")] = payload
    return values


def _build_reference_link_masses(root: ET.Element) -> dict[str, float]:
    values: dict[str, float] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if not name:
            continue
        mass_elem = link.find("./inertial/mass")
        if mass_elem is None or mass_elem.get("value") is None:
            continue
        values[_canonical_link_name(name, side="reference")] = float(mass_elem.get("value"))
    return values


def _build_reference_link_collision_counts_from_snapshot() -> dict[str, int]:
    values: dict[str, int] = {}
    for name, payload in REFERENCE_TIENKUNG2_LITE_SNAPSHOT["urdf"]["links"].items():
        values[_canonical_link_name(name, side="reference")] = int(payload["collision_count"])
    return values


def _build_reference_joint_limits_from_snapshot() -> dict[str, dict[str, str]]:
    values: dict[str, dict[str, str]] = {}
    for name, payload in REFERENCE_TIENKUNG2_LITE_SNAPSHOT["urdf"]["joints"].items():
        limit_payload: dict[str, str] = {}
        for attr in ("lower", "upper", "effort", "velocity"):
            value = payload.get(attr)
            if value is not None:
                limit_payload[attr] = str(value)
        if limit_payload:
            values[_canonical_joint_name(name, side="reference")] = limit_payload
    return values


def _build_reference_link_masses_from_snapshot() -> dict[str, float]:
    values: dict[str, float] = {}
    for name, payload in REFERENCE_TIENKUNG2_LITE_SNAPSHOT["urdf"]["links"].items():
        mass_value = payload.get("mass")
        if mass_value is None:
            continue
        values[_canonical_link_name(name, side="reference")] = float(mass_value)
    return values


def _build_reference_link_names(root: ET.Element) -> set[str]:
    values: set[str] = set()
    for link in root.findall("link"):
        name = link.get("name")
        if not name:
            continue
        values.add(_canonical_link_name(name, side="reference"))
    return values


def _build_reference_link_names_from_snapshot() -> set[str]:
    values: set[str] = set()
    for name in REFERENCE_TIENKUNG2_LITE_SNAPSHOT["urdf"]["links"].keys():
        values.add(_canonical_link_name(name, side="reference"))
    return values


def _build_fixed_joint_children(root: ET.Element) -> set[str]:
    values: set[str] = set()
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        child_elem = joint.find("child")
        child_link = None if child_elem is None else child_elem.get("link")
        if child_link:
            values.add(child_link)
    return values


def _read_link_mass(link: ET.Element) -> float | None:
    mass_elem = link.find("./inertial/mass")
    if mass_elem is None or mass_elem.get("value") is None:
        return None
    return float(mass_elem.get("value"))


def _sync_collision_topology(candidate_root: ET.Element, reference_collision_counts: dict[str, int]) -> tuple[int, list[str]]:
    removed_count = 0
    touched_links: list[str] = []
    for link in candidate_root.findall("link"):
        name = link.get("name")
        if not name:
            continue
        ref_collision_count = reference_collision_counts.get(_canonical_link_name(name, side="candidate"), 0)
        if ref_collision_count != 0:
            continue
        collisions = list(link.findall("collision"))
        if not collisions:
            continue
        for collision in collisions:
            link.remove(collision)
            removed_count += 1
        touched_links.append(name)
    return removed_count, touched_links


def _sync_joint_limits(candidate_root: ET.Element, reference_joint_limits: dict[str, dict[str, str]]) -> tuple[int, list[str]]:
    changed_joints: list[str] = []
    changed_fields = 0
    for joint in candidate_root.findall("joint"):
        name = joint.get("name")
        if not name:
            continue
        ref_limits = reference_joint_limits.get(_canonical_joint_name(name, side="candidate"))
        if not ref_limits:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        joint_changed = False
        for attr, ref_value in ref_limits.items():
            if limit.get(attr) == ref_value:
                continue
            limit.set(attr, ref_value)
            changed_fields += 1
            joint_changed = True
        if joint_changed:
            changed_joints.append(name)
    return changed_fields, changed_joints


def _sync_link_masses(candidate_root: ET.Element, reference_link_masses: dict[str, float]) -> tuple[int, list[str]]:
    changed_links: list[str] = []
    for link in candidate_root.findall("link"):
        name = link.get("name")
        if not name:
            continue
        ref_mass = reference_link_masses.get(_canonical_link_name(name, side="candidate"))
        if ref_mass is None:
            continue
        mass_elem = link.find("./inertial/mass")
        inertia_elem = link.find("./inertial/inertia")
        if mass_elem is None or mass_elem.get("value") is None:
            continue

        cand_mass = float(mass_elem.get("value"))
        if cand_mass <= 0.0:
            continue
        if abs(cand_mass - ref_mass) <= 1e-12:
            continue

        scale = ref_mass / cand_mass
        mass_elem.set("value", f"{ref_mass:.12g}")
        if inertia_elem is not None:
            for attr in INERTIA_ATTRS:
                value = inertia_elem.get(attr)
                if value is None:
                    continue
                inertia_elem.set(attr, f"{float(value) * scale:.12g}")
        changed_links.append(name)
    return len(changed_links), changed_links


def _zero_link_mass_and_inertia(link: ET.Element) -> bool:
    mass_elem = link.find("./inertial/mass")
    inertia_elem = link.find("./inertial/inertia")
    changed = False

    if mass_elem is not None and mass_elem.get("value") != "0":
        mass_elem.set("value", "0")
        changed = True
    if inertia_elem is not None:
        for attr in INERTIA_ATTRS:
            if inertia_elem.get(attr) != "0":
                inertia_elem.set(attr, "0")
                changed = True
    return changed


def _zero_candidate_only_fixed_link_masses(candidate_root: ET.Element, candidate_only_fixed_links: list[str]) -> tuple[int, list[str]]:
    changed_links: list[str] = []
    candidate_only_fixed_link_set = set(candidate_only_fixed_links)
    for link in candidate_root.findall("link"):
        name = link.get("name")
        if not name or name not in candidate_only_fixed_link_set:
            continue
        if _zero_link_mass_and_inertia(link):
            changed_links.append(name)
    return len(changed_links), changed_links


def _replace_ankle_roll_collisions_with_reference(candidate_root: ET.Element) -> tuple[int, list[str]]:
    changed_links: list[str] = []
    for link in candidate_root.findall("link"):
        name = link.get("name")
        if not name or name not in REFERENCE_ANKLE_ROLL_COLLISION_SPECS:
            continue

        for collision in list(link.findall("collision")):
            link.remove(collision)

        for collision_spec in REFERENCE_ANKLE_ROLL_COLLISION_SPECS[name]:
            collision_elem = ET.SubElement(link, "collision")
            ET.SubElement(collision_elem, "origin", attrib=dict(collision_spec["origin"]))
            geometry_elem = ET.SubElement(collision_elem, "geometry")
            geom_tag, geom_attrib = collision_spec["geometry"]
            ET.SubElement(geometry_elem, geom_tag, attrib=dict(geom_attrib))
        changed_links.append(name)
    return len(changed_links), changed_links


def _write_xml(output_path: Path, tree: ET.ElementTree) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indent = getattr(ET, "indent", None)
    if callable(indent):
        indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    args = _parse_args()
    reference_urdf, candidate_urdf, output_urdf = _resolve_paths(args)

    candidate_tree = ET.parse(candidate_urdf)
    candidate_root = candidate_tree.getroot()

    if reference_urdf.is_file():
        reference_tree = ET.parse(reference_urdf)
        reference_root = reference_tree.getroot()
        reference_collision_counts = _build_reference_link_collision_counts(reference_root)
        reference_joint_limits = _build_reference_joint_limits(reference_root)
        reference_link_masses = _build_reference_link_masses(reference_root)
        reference_link_names = _build_reference_link_names(reference_root)
        reference_source = f"filesystem:{reference_urdf}"
    else:
        reference_collision_counts = _build_reference_link_collision_counts_from_snapshot()
        reference_joint_limits = _build_reference_joint_limits_from_snapshot()
        reference_link_masses = _build_reference_link_masses_from_snapshot()
        reference_link_names = _build_reference_link_names_from_snapshot()
        reference_source = f"snapshot:{REFERENCE_TIENKUNG2_LITE_SNAPSHOT['source_hint']}"

    _log(f"[INFO] reference_source: {reference_source}")
    _log(f"[INFO] candidate_urdf: {candidate_urdf}")
    _log(f"[INFO] output_urdf: {output_urdf}")
    fixed_joint_children = _build_fixed_joint_children(candidate_root)

    candidate_only_fixed_links: list[str] = []
    candidate_only_fixed_mass = 0.0
    for link in candidate_root.findall("link"):
        name = link.get("name")
        if not name:
            continue
        canonical_name = _canonical_link_name(name, side="candidate")
        if canonical_name in reference_link_names:
            continue
        if name not in fixed_joint_children:
            continue
        candidate_only_fixed_links.append(name)
        link_mass = _read_link_mass(link)
        if link_mass is not None:
            candidate_only_fixed_mass += link_mass
    if candidate_only_fixed_links:
        _log(
            "[WARN] candidate_only_fixed_links: "
            f"{', '.join(candidate_only_fixed_links)} "
            f"(combined_mass={candidate_only_fixed_mass:.6f})"
        )

    removed_collisions = 0
    collision_links: list[str] = []
    if args.sync_collision_topology:
        removed_collisions, collision_links = _sync_collision_topology(candidate_root, reference_collision_counts)
        _log(
            f"[INFO] collision_topology_sync: removed {removed_collisions} collision elements from "
            f"{len(collision_links)} candidate links"
        )
        if collision_links:
            _log(f"[INFO] collision_links_touched: {', '.join(collision_links)}")

    changed_joint_fields = 0
    changed_joints: list[str] = []
    if args.sync_joint_limits:
        changed_joint_fields, changed_joints = _sync_joint_limits(candidate_root, reference_joint_limits)
        _log(
            f"[INFO] joint_limit_sync: updated {changed_joint_fields} limit attributes across "
            f"{len(changed_joints)} candidate joints"
        )
        if changed_joints:
            _log(f"[INFO] joint_limit_joints_touched: {', '.join(changed_joints)}")

    changed_mass_count = 0
    changed_mass_links: list[str] = []
    if args.sync_link_mass:
        changed_mass_count, changed_mass_links = _sync_link_masses(candidate_root, reference_link_masses)
        _log(f"[INFO] link_mass_sync: updated {changed_mass_count} candidate links")
        if changed_mass_links:
            _log(f"[INFO] link_mass_links_touched: {', '.join(changed_mass_links)}")
        if candidate_only_fixed_links and candidate_only_fixed_mass > 0.0:
            _log(
                "[WARN] link_mass_sync kept extra fixed-link mass in the candidate asset. "
                "If the converter merges fixed joints, this mass can be folded into the root chain and overshoot "
                f"the reference total by about {candidate_only_fixed_mass:.6f}."
            )

    zeroed_extra_fixed_link_count = 0
    zeroed_extra_fixed_links: list[str] = []
    if args.zero_candidate_only_fixed_link_mass:
        zeroed_extra_fixed_link_count, zeroed_extra_fixed_links = _zero_candidate_only_fixed_link_masses(
            candidate_root, candidate_only_fixed_links
        )
        _log(
            f"[INFO] zero_candidate_only_fixed_link_mass: updated {zeroed_extra_fixed_link_count} candidate-only fixed links"
        )
        if zeroed_extra_fixed_links:
            _log(f"[INFO] zeroed_candidate_only_fixed_links: {', '.join(zeroed_extra_fixed_links)}")

    replaced_ankle_collision_count = 0
    replaced_ankle_collision_links: list[str] = []
    if args.replace_ankle_roll_collisions_with_reference:
        replaced_ankle_collision_count, replaced_ankle_collision_links = _replace_ankle_roll_collisions_with_reference(
            candidate_root
        )
        _log(
            f"[INFO] replace_ankle_roll_collisions_with_reference: updated {replaced_ankle_collision_count} links"
        )
        if replaced_ankle_collision_links:
            _log(f"[INFO] replaced_ankle_roll_collision_links: {', '.join(replaced_ankle_collision_links)}")

    _write_xml(output_urdf, candidate_tree)
    _log("[INFO] Wrote aligned URDF copy.")


if __name__ == "__main__":
    main()
