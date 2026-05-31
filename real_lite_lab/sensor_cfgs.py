from __future__ import annotations

from isaaclab.sensors.camera import CameraCfg as BaseCameraCfg
from isaaclab.sim import PinholeCameraCfg
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


@configclass
class DepthCameraCfg:
    enable_depth_camera: bool = False
    prim_body_name: str = "pelvis/depth_camera"
    width: int = 480
    height: int = 270
    data_types: list[str] = ["distance_to_image_plane"]
    offset: BaseCameraCfg.OffsetCfg = BaseCameraCfg.OffsetCfg()
    spawn: PinholeCameraCfg = PinholeCameraCfg()
    debug_vis: bool = False
    visualizer_cfg = None
