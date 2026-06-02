from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from real_lite_lab.mujoco_standing_diagnostics import TOE_RAIL_SUPPORT_GEOM_GROUPS


TOE_RAIL_GEOM_ATTRIBUTES = {
    "sole_left": (
        {
            "name": "toe1_left",
            "contype": "2",
            "conaffinity": "1",
            "size": "0.015 0.115",
            "pos": "0.035 0.025 -0.042",
            "quat": "0.707105 0 0.707108 0",
            "type": "cylinder",
            "rgba": "0.752941 0.752941 0.752941 1",
        },
        {
            "name": "toe2_left",
            "contype": "2",
            "conaffinity": "1",
            "size": "0.015 0.115",
            "pos": "0.035 -0.025 -0.042",
            "quat": "0.707105 0 0.707108 0",
            "type": "cylinder",
            "rgba": "0.752941 0.752941 0.752941 1",
        },
    ),
    "sole_right": (
        {
            "name": "toe1_right",
            "contype": "2",
            "conaffinity": "1",
            "size": "0.015 0.115",
            "pos": "0.035 0.025 -0.042",
            "quat": "0.707105 0 0.707108 0",
            "type": "cylinder",
            "rgba": "0.752941 0.752941 0.752941 1",
        },
        {
            "name": "toe2_right",
            "contype": "2",
            "conaffinity": "1",
            "size": "0.015 0.115",
            "pos": "0.035 -0.025 -0.042",
            "quat": "0.707105 0 0.707108 0",
            "type": "cylinder",
            "rgba": "0.752941 0.752941 0.752941 1",
        },
    ),
}


@dataclass(frozen=True)
class ContactVariantResult:
    model_path: Path
    variant_name: str
    support_geom_groups: tuple[tuple[str, tuple[str, ...]], ...]


def _write_xml_tree(output_path: Path, tree: ET.ElementTree) -> None:
    indent = getattr(ET, "indent", None)
    if callable(indent):
        indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def build_toe_rail_contact_model(model_path: Path, output_path: Path | None = None) -> ContactVariantResult:
    tree = ET.parse(model_path)
    root = tree.getroot()
    parent_map = {child: parent for parent in root.iter() for child in parent}

    missing_geom_names: list[str] = []
    for source_geom_name, replacement_specs in TOE_RAIL_GEOM_ATTRIBUTES.items():
        geom_elem = root.find(f".//geom[@name='{source_geom_name}']")
        if geom_elem is None:
            missing_geom_names.append(source_geom_name)
            continue

        parent_elem = parent_map.get(geom_elem)
        if parent_elem is None:
            raise ValueError(f"Unable to locate parent element for geom '{source_geom_name}'.")

        insert_index = list(parent_elem).index(geom_elem)
        parent_elem.remove(geom_elem)
        for offset, attributes in enumerate(replacement_specs):
            parent_elem.insert(insert_index + offset, ET.Element("geom", dict(attributes)))

    if missing_geom_names:
        raise ValueError(f"Unable to find source support geoms: {', '.join(missing_geom_names)}")

    if output_path is None:
        output_path = model_path.with_name(f"{model_path.stem}.toe_rails.xml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_xml_tree(output_path, tree)

    return ContactVariantResult(
        model_path=output_path,
        variant_name="toe_rails",
        support_geom_groups=TOE_RAIL_SUPPORT_GEOM_GROUPS,
    )
