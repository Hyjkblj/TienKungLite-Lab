from __future__ import annotations

from isaaclab.sensors import patterns
from isaaclab.utils import configclass


@configclass
class LidarCfg:
    enable_lidar: bool = False
    prim_body_name: str = "pelvis"
    offset: tuple = (0.15, 0.0, 0.6)
    rotation: tuple = (1.0, 0.0, 0.0, 0.0)
    pattern_cfg = patterns.LidarPatternCfg(
        channels=64,
        horizontal_fov_range=(0, 360),
        vertical_fov_range=(-90, 90),
        horizontal_res=0.5,
    )
    debug_vis: bool = False
    max_distance: float = 20.0
    mesh_prim_paths = ["/World"]
