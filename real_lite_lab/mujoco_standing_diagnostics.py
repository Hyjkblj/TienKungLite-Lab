from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


DEFAULT_SUPPORT_GEOM_NAMES = ("sole_left", "sole_right")


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError(f"Expected points_xy with shape (N, 2), got {points_xy.shape}.")

    unique_points = np.unique(points_xy, axis=0)
    if unique_points.shape[0] <= 1:
        return unique_points

    sorted_points = unique_points[np.lexsort((unique_points[:, 1], unique_points[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for point in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[np.ndarray] = []
    for point in sorted_points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull if hull.size else unique_points[:1]


def _distance_point_to_segment(point_xy: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray) -> float:
    segment = end_xy - start_xy
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return float(np.linalg.norm(point_xy - start_xy))
    t = float(np.clip(np.dot(point_xy - start_xy, segment) / denom, 0.0, 1.0))
    projection = start_xy + t * segment
    return float(np.linalg.norm(point_xy - projection))


def signed_distance_to_convex_polygon(point_xy: np.ndarray, polygon_xy: np.ndarray) -> float:
    polygon_xy = np.asarray(polygon_xy, dtype=np.float64)
    point_xy = np.asarray(point_xy, dtype=np.float64)

    if polygon_xy.ndim != 2 or polygon_xy.shape[1] != 2:
        raise ValueError(f"Expected polygon_xy with shape (N, 2), got {polygon_xy.shape}.")
    if point_xy.shape != (2,):
        raise ValueError(f"Expected point_xy with shape (2,), got {point_xy.shape}.")

    if polygon_xy.shape[0] == 0:
        return float("nan")
    if polygon_xy.shape[0] == 1:
        return -float(np.linalg.norm(point_xy - polygon_xy[0]))
    if polygon_xy.shape[0] == 2:
        return -_distance_point_to_segment(point_xy, polygon_xy[0], polygon_xy[1])

    edge_distances: list[float] = []
    inside = True
    for idx in range(polygon_xy.shape[0]):
        start_xy = polygon_xy[idx]
        end_xy = polygon_xy[(idx + 1) % polygon_xy.shape[0]]
        edge = end_xy - start_xy
        rel = point_xy - start_xy
        cross_z = float(edge[0] * rel[1] - edge[1] * rel[0])
        if cross_z < -1e-9:
            inside = False
        edge_distances.append(_distance_point_to_segment(point_xy, start_xy, end_xy))

    margin = min(edge_distances)
    return margin if inside else -margin


def _project_box_geom_xy_corners(model, data, geom_id: int) -> np.ndarray:
    center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)

    local_corners = np.asarray(
        [
            [sx * size[0], sy * size[1], sz * size[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    world_corners = center + local_corners @ rotation.T
    return world_corners[:, :2]


def compute_support_polygon_xy(model, data, support_geom_ids: Sequence[int]) -> np.ndarray:
    support_points_xy = np.concatenate(
        [_project_box_geom_xy_corners(model, data, int(geom_id)) for geom_id in support_geom_ids],
        axis=0,
    )
    return _convex_hull_xy(support_points_xy)


def compute_mass_weighted_com(model, data) -> np.ndarray:
    body_mass = np.asarray(model.body_mass, dtype=np.float64)
    body_com_world = np.asarray(data.xipos, dtype=np.float64)
    if body_mass.ndim != 1:
        raise ValueError(f"Expected body_mass with shape (N,), got {body_mass.shape}.")
    if body_com_world.ndim != 2 or body_com_world.shape[0] != body_mass.shape[0] or body_com_world.shape[1] != 3:
        raise ValueError(
            "Expected data.xipos with shape (N, 3) aligned with model.body_mass, "
            f"got {body_com_world.shape} for masses {body_mass.shape}."
        )

    positive_mass = body_mass > 0.0
    total_mass = float(np.sum(body_mass[positive_mass]))
    if total_mass <= 0.0:
        raise ValueError("Total articulated body mass must be positive.")

    return np.sum(body_com_world[positive_mass] * body_mass[positive_mass, None], axis=0) / total_mass


def compute_nominal_support_diagnostics(model, data, support_geom_name_to_id: Mapping[str, int]) -> dict[str, np.ndarray | float]:
    if not support_geom_name_to_id:
        raise ValueError("support_geom_name_to_id must not be empty.")

    support_names = tuple(support_geom_name_to_id.keys())
    support_ids = tuple(int(geom_id) for geom_id in support_geom_name_to_id.values())
    support_polygon_xy = compute_support_polygon_xy(model, data, support_ids)
    com_world = compute_mass_weighted_com(model, data)
    com_xy = com_world[:2].copy()
    support_center_xy = np.mean(support_polygon_xy, axis=0)
    support_bounds_min_xy = np.min(support_polygon_xy, axis=0)
    support_bounds_max_xy = np.max(support_polygon_xy, axis=0)

    foot_centers_world = np.stack(
        [np.asarray(data.geom_xpos[support_geom_name_to_id[name]], dtype=np.float64).copy() for name in support_names],
        axis=0,
    )
    support_margin = signed_distance_to_convex_polygon(com_xy, support_polygon_xy)

    return {
        "com_world": com_world.astype(np.float64),
        "support_center_xy": support_center_xy.astype(np.float64),
        "support_offset_xy": (com_xy - support_center_xy).astype(np.float64),
        "support_bounds_min_xy": support_bounds_min_xy.astype(np.float64),
        "support_bounds_max_xy": support_bounds_max_xy.astype(np.float64),
        "support_extents_xy": (support_bounds_max_xy - support_bounds_min_xy).astype(np.float64),
        "support_margin": float(support_margin),
        "foot_centers_world": foot_centers_world.astype(np.float64),
    }


def compute_support_contact_summary(model, data, support_geom_name_to_id: Mapping[str, int]) -> dict[str, np.ndarray | float]:
    import mujoco

    support_names = tuple(support_geom_name_to_id.keys())
    contact_counts = np.zeros(len(support_names), dtype=np.int32)
    normal_forces = np.zeros(len(support_names), dtype=np.float64)

    force_buffer = np.zeros(6, dtype=np.float64)
    for contact_id in range(int(data.ncon)):
        mujoco.mj_contactForce(model, data, contact_id, force_buffer)
        normal_force = abs(float(force_buffer[0]))
        if normal_force <= 0.0:
            continue

        contact = data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        for idx, name in enumerate(support_names):
            geom_id = int(support_geom_name_to_id[name])
            if geom1 == geom_id or geom2 == geom_id:
                contact_counts[idx] += 1
                normal_forces[idx] += normal_force

    total_normal_force = float(np.sum(normal_forces))
    left_load_share = float(normal_forces[0] / total_normal_force) if total_normal_force > 1e-9 else float("nan")
    return {
        "foot_contact_counts": contact_counts,
        "foot_normal_forces": normal_forces,
        "double_support": int(np.count_nonzero(contact_counts) == len(support_names)),
        "left_load_share": left_load_share,
    }


def collect_standing_diagnostics(model, data, support_geom_name_to_id: Mapping[str, int]) -> dict[str, np.ndarray | float]:
    diagnostics = compute_nominal_support_diagnostics(model, data, support_geom_name_to_id)
    diagnostics.update(compute_support_contact_summary(model, data, support_geom_name_to_id))
    return diagnostics


def format_standing_diagnostics_summary(
    diagnostics: Mapping[str, np.ndarray | float],
    *,
    support_names: Sequence[str] = DEFAULT_SUPPORT_GEOM_NAMES,
) -> list[str]:
    com_world = np.asarray(diagnostics["com_world"], dtype=np.float64)
    support_center_xy = np.asarray(diagnostics["support_center_xy"], dtype=np.float64)
    support_offset_xy = np.asarray(diagnostics["support_offset_xy"], dtype=np.float64)
    support_extents_xy = np.asarray(diagnostics["support_extents_xy"], dtype=np.float64)
    contact_counts = np.asarray(diagnostics["foot_contact_counts"], dtype=np.int32)
    normal_forces = np.asarray(diagnostics["foot_normal_forces"], dtype=np.float64)

    support_labels = ", ".join(
        f"{name}: ncon={int(contact_counts[idx])}, normal={normal_forces[idx]:.2f}N"
        for idx, name in enumerate(support_names[: contact_counts.shape[0]])
    )
    return [
        (
            "[INFO] Standing diagnostics: "
            f"com=({com_world[0]:+.4f}, {com_world[1]:+.4f}, {com_world[2]:+.4f}), "
            f"support_center=({support_center_xy[0]:+.4f}, {support_center_xy[1]:+.4f}), "
            f"support_offset=({support_offset_xy[0]:+.4f}, {support_offset_xy[1]:+.4f}), "
            f"support_margin={float(diagnostics['support_margin']):+.4f}m"
        ),
        (
            "[INFO] Support extents: "
            f"length={support_extents_xy[0]:.4f}m, width={support_extents_xy[1]:.4f}m, "
            f"double_support={int(diagnostics['double_support'])}, left_load_share={float(diagnostics['left_load_share']):.3f}"
        ),
        f"[INFO] Foot contacts: {support_labels}",
    ]
