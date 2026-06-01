from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


MAX_MUJOCO_STL_TRIANGLES = 200_000


@dataclass(frozen=True)
class MeshFallbackResult:
    model_path: Path
    stripped_mesh_names: tuple[str, ...]


def ensure_offscreen_framebuffer_size(model, width: int, height: int) -> tuple[int, int] | None:
    visual = getattr(model, "vis", None)
    global_visual = getattr(visual, "global_", None)
    if global_visual is None:
        return None

    current_width = int(getattr(global_visual, "offwidth", 0))
    current_height = int(getattr(global_visual, "offheight", 0))
    target_width = max(current_width, int(width))
    target_height = max(current_height, int(height))

    if target_width == current_width and target_height == current_height:
        return None

    global_visual.offwidth = target_width
    global_visual.offheight = target_height
    return (target_width, target_height)


def _binary_stl_triangle_count(stl_bytes: bytes) -> int | None:
    if len(stl_bytes) < 84:
        return None

    triangle_data_length = len(stl_bytes) - 84
    if triangle_data_length <= 0 or triangle_data_length % 50 != 0:
        return None

    triangle_count = struct.unpack("<I", stl_bytes[80:84])[0]
    if triangle_count != triangle_data_length // 50:
        return None
    return triangle_count


def _mesh_is_mujoco_compatible(mesh_path: Path) -> bool:
    if not mesh_path.is_file():
        return False
    if mesh_path.suffix.lower() != ".stl":
        return True

    triangle_count = _binary_stl_triangle_count(mesh_path.read_bytes())
    return triangle_count is not None and 1 <= triangle_count <= MAX_MUJOCO_STL_TRIANGLES


def _append_waist_placeholder(body: ET.Element) -> None:
    existing_placeholder = body.find("./geom[@name='waist_link_placeholder']")
    if existing_placeholder is not None:
        return

    body.append(
        ET.Element(
            "geom",
            {
                "name": "waist_link_placeholder",
                "type": "box",
                "size": "0.09 0.12 0.22",
                "pos": "0 0 0.22",
                "rgba": "0.75 0.75 0.75 1",
                "contype": "0",
                "conaffinity": "0",
                "density": "0",
                "group": "1",
            },
        )
    )


def build_mesh_safe_model(model_path: Path, output_path: Path | None = None) -> MeshFallbackResult | None:
    tree = ET.parse(model_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_root = (model_path.parent / meshdir).resolve()

    asset = root.find("asset")
    if asset is None:
        return None

    incompatible_mesh_names: list[str] = []
    for mesh_elem in list(asset.findall("mesh")):
        mesh_name = mesh_elem.get("name")
        mesh_file = mesh_elem.get("file")
        if not mesh_name or not mesh_file:
            continue

        mesh_path = (mesh_root / mesh_file).resolve()
        if _mesh_is_mujoco_compatible(mesh_path):
            continue

        incompatible_mesh_names.append(mesh_name)
        asset.remove(mesh_elem)

    if not incompatible_mesh_names:
        return None

    incompatible_mesh_set = set(incompatible_mesh_names)
    parent_map = {child: parent for parent in root.iter() for child in parent}
    for geom in list(root.iter("geom")):
        if geom.get("type") != "mesh":
            continue
        if geom.get("mesh") not in incompatible_mesh_set:
            continue
        parent = parent_map.get(geom)
        if parent is not None:
            parent.remove(geom)

    if "waist_link" in incompatible_mesh_set:
        waist_body = root.find(".//body[@name='waist_link']")
        if waist_body is not None:
            _append_waist_placeholder(waist_body)

    if output_path is None:
        output_path = model_path.with_name(f"{model_path.stem}.mesh_safe.xml")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    indent = getattr(ET, "indent", None)
    if callable(indent):
        indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return MeshFallbackResult(
        model_path=output_path,
        stripped_mesh_names=tuple(incompatible_mesh_names),
    )
