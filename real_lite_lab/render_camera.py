from __future__ import annotations


CAMERA_PRESETS = {
    "follow_diag": {
        "distance": 2.8,
        "azimuth": 135.0,
        "elevation": -18.0,
        "lookat_offset": (0.0, 0.0, 0.9),
    },
    "follow_side": {
        "distance": 2.6,
        "azimuth": 90.0,
        "elevation": -15.0,
        "lookat_offset": (0.0, 0.0, 0.9),
    },
    "follow_front": {
        "distance": 2.4,
        "azimuth": 180.0,
        "elevation": -14.0,
        "lookat_offset": (0.0, 0.0, 0.9),
    },
    "follow_front_full_body": {
        "distance": 3.6,
        "azimuth": 180.0,
        "elevation": -8.0,
        "lookat_offset": (0.0, 0.0, 0.35),
    },
    "follow_topdiag": {
        "distance": 4.4,
        "azimuth": 135.0,
        "elevation": -30.0,
        "lookat_offset": (0.0, 0.0, 0.8),
    },
}

CAMERA_PRESET_ALIASES = {
    "diag": "follow_diag",
    "side": "follow_side",
    "front": "follow_front",
    "front_full": "follow_front_full_body",
    "front_full_body": "follow_front_full_body",
    "topdiag": "follow_topdiag",
}


def resolve_camera_preset_name(camera_name: str | None) -> str | None:
    if camera_name is None:
        return None
    if camera_name in CAMERA_PRESETS:
        return camera_name
    return CAMERA_PRESET_ALIASES.get(camera_name)


def get_camera_preset(camera_name: str | None) -> dict | None:
    resolved_name = resolve_camera_preset_name(camera_name)
    if resolved_name is None:
        return None
    return CAMERA_PRESETS.get(resolved_name)


def camera_preset_names() -> tuple[str, ...]:
    return tuple(CAMERA_PRESETS.keys())


def camera_preset_alias_names() -> tuple[str, ...]:
    return tuple(CAMERA_PRESET_ALIASES.keys())
