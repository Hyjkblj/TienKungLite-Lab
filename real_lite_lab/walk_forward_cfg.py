from __future__ import annotations

import copy

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp as base_mdp
from . import rewards as rl_rewards
from .assets.tienkung2_lite_real import REAL_LITE_ARTICULATION_CFG
from .base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    NoiseCfg,
    NoiseScalesCfg,
)
from .constants import POLICY_JOINT_COUNT, TASK_PRESETS
from .stand_cfg import STAND_ROOT_Z
from .walk_cfg import RealLiteGaitCfg, RealLiteRewardCfg, RealLiteWalkAgentCfg, RealLiteWalkEnvCfg


WALK_FORWARD_ARTICULATION_CFG = copy.deepcopy(REAL_LITE_ARTICULATION_CFG)
WALK_FORWARD_ARTICULATION_CFG.init_state.pos = (0.0, 0.0, STAND_ROOT_Z)


@configclass
class RealLiteWalkForwardRewardCfg(RealLiteRewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=rl_rewards.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.2})
    track_lin_vel_x_exp = RewTerm(func=rl_rewards.track_lin_vel_x_yaw_frame_exp, weight=3.0, params={"std": 0.15})
    forward_velocity = RewTerm(func=rl_rewards.forward_velocity_yaw_frame, weight=1.0)
    backward_velocity_l2 = RewTerm(func=rl_rewards.backward_velocity_yaw_frame_l2, weight=-4.0)
    lin_vel_y_l2 = RewTerm(func=rl_rewards.lin_vel_y_yaw_frame_l2, weight=-1.0)
    track_ang_vel_z_exp = RewTerm(func=rl_rewards.track_ang_vel_z_world_exp, weight=0.5, params={"std": 0.25})
    body_orientation_l2 = RewTerm(
        func=rl_rewards.body_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")},
        weight=-3.0,
    )
    flat_orientation_l2 = RewTerm(func=rl_rewards.flat_orientation_l2, weight=-1.5)
    action_rate_l2 = RewTerm(func=rl_rewards.action_rate_l2, weight=-0.005)
    joint_deviation_legs = RewTerm(
        func=rl_rewards.joint_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_pitch_.*_joint",
                    "knee_pitch_.*_joint",
                    "ankle_pitch_.*_joint",
                    "ankle_roll_.*_joint",
                ],
            )
        },
    )
    hip_roll_action = RewTerm(func=rl_rewards.hip_roll_action, weight=-0.5)
    hip_yaw_action = RewTerm(func=rl_rewards.hip_yaw_action, weight=-0.5)


@configclass
class RealLiteWalkForwardEnvCfg(RealLiteWalkEnvCfg):
    amp_motion_files_display = [str(TASK_PRESETS["walk_forward_real_lite"]["display_motion_file"])]
    scene: BaseSceneCfg = copy.deepcopy(RealLiteWalkEnvCfg.scene)
    scene.robot = WALK_FORWARD_ARTICULATION_CFG
    scene.terrain_type = "plane"
    scene.terrain_generator = None
    scene.max_init_terrain_level = 0
    robot = copy.deepcopy(RealLiteWalkEnvCfg.robot)
    robot.action_scale = 0.18
    reward = RealLiteWalkForwardRewardCfg()
    gait = RealLiteGaitCfg()
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        debug_vis=False,
        ranges=CommandRangesCfg(
            lin_vel_x=(0.10, 0.35),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            ang_vel=0.05,
            projected_gravity=0.02,
            joint_pos=0.005,
            joint_vel=0.5,
            height_scan=0.0,
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=None,
            add_base_mass=None,
            reset_base=EventTerm(
                func=base_mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-0.05, 0.05)},
                    "velocity_range": {
                        "x": (0.0, 0.0),
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                        "roll": (-0.02, 0.02),
                        "pitch": (-0.02, 0.02),
                        "yaw": (-0.02, 0.02),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=base_mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.98, 1.02),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=None,
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )


@configclass
class RealLiteWalkForwardAgentCfg(RealLiteWalkAgentCfg):
    max_iterations = 8000
    experiment_name = "walk_forward_real_lite"
    neptune_project = "walk_forward_real_lite"
    wandb_project = "walk_forward_real_lite"
    amp_motion_files = [str(TASK_PRESETS["walk_forward_real_lite"]["amp_motion_file"])]
    amp_reward_coef = 0.15
    amp_task_reward_lerp = 0.9
    min_normalized_std = [0.03] * POLICY_JOINT_COUNT
