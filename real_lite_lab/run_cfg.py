from __future__ import annotations

from isaaclab.utils import configclass

from .constants import TASK_PRESETS
from .walk_cfg import RealLiteRewardCfg, RealLiteWalkAgentCfg, RealLiteWalkEnvCfg


@configclass
class RealLiteRunGaitCfg:
    gait_air_ratio_l: float = TASK_PRESETS["run_real_lite"]["gait_air_ratio_l"]
    gait_air_ratio_r: float = TASK_PRESETS["run_real_lite"]["gait_air_ratio_r"]
    gait_phase_offset_l: float = TASK_PRESETS["run_real_lite"]["gait_phase_offset_l"]
    gait_phase_offset_r: float = TASK_PRESETS["run_real_lite"]["gait_phase_offset_r"]
    gait_cycle: float = TASK_PRESETS["run_real_lite"]["gait_cycle"]


@configclass
class RealLiteRunEnvCfg(RealLiteWalkEnvCfg):
    amp_motion_files_display = [str(TASK_PRESETS["run_real_lite"]["display_motion_file"])]
    reward = RealLiteRewardCfg()
    gait = RealLiteRunGaitCfg()


@configclass
class RealLiteRunAgentCfg(RealLiteWalkAgentCfg):
    experiment_name = "run_real_lite"
    neptune_project = "run_real_lite"
    wandb_project = "run_real_lite"
    amp_motion_files = [str(TASK_PRESETS["run_real_lite"]["amp_motion_file"])]
