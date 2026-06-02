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
    "follow_topdiag": {
        "distance": 3.4,
        "azimuth": 135.0,
        "elevation": -32.0,
        "lookat_offset": (0.0, 0.0, 0.9),
    },
}


def get_camera_preset(camera_name: str | None) -> dict | None:
    if camera_name is None:
        return None
    return CAMERA_PRESETS.get(camera_name)


def camera_preset_names() -> tuple[str, ...]:
    return tuple(CAMERA_PRESETS.keys())
