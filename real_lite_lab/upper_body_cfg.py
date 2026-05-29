from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp as base_mdp
from .assets.tienkung2_lite_real import REAL_LITE_ARTICULATION_CFG
from .base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
)
from .constants import (
    ANKLE_JOINT_NAMES,
    ELBOW_LINK_NAMES,
    FEET_LINK_NAMES,
    LEFT_ARM_JOINT_NAMES,
    LEFT_LEG_JOINT_NAMES,
    POLICY_JOINT_COUNT,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_LEG_JOINT_NAMES,
    TASK_PRESETS,
)
from . import rewards as rl_rewards
from .walk_cfg import RealLiteWalkAgentCfg, RealLiteWalkEnvCfg


@configclass
class RealLiteUpperBodyGaitCfg:
    gait_air_ratio_l: float = TASK_PRESETS["upper_body_real_lite"]["gait_air_ratio_l"]
    gait_air_ratio_r: float = TASK_PRESETS["upper_body_real_lite"]["gait_air_ratio_r"]
    gait_phase_offset_l: float = TASK_PRESETS["upper_body_real_lite"]["gait_phase_offset_l"]
    gait_phase_offset_r: float = TASK_PRESETS["upper_body_real_lite"]["gait_phase_offset_r"]
    gait_cycle: float = TASK_PRESETS["upper_body_real_lite"]["gait_cycle"]


@configclass
class RealLiteUpperBodyRewardCfg:
    track_lin_vel_xy_exp = RewTerm(func=rl_rewards.track_lin_vel_xy_yaw_frame_exp, weight=1.5, params={"std": 0.3})
    track_ang_vel_z_exp = RewTerm(func=rl_rewards.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.3})
    lin_vel_z_l2 = RewTerm(func=rl_rewards.lin_vel_z_l2, weight=-1.5)
    ang_vel_xy_l2 = RewTerm(func=rl_rewards.ang_vel_xy_l2, weight=-0.1)
    energy = RewTerm(func=rl_rewards.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=rl_rewards.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=rl_rewards.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=rl_rewards.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_.*", "pelvis"]
            ),
            "threshold": 1.0,
        },
    )
    body_orientation_l2 = RewTerm(
        func=rl_rewards.body_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")},
        weight=-4.0,
    )
    flat_orientation_l2 = RewTerm(func=rl_rewards.flat_orientation_l2, weight=-2.0)
    termination_penalty = RewTerm(func=rl_rewards.is_terminated, weight=-200.0)
    feet_slide = RewTerm(
        func=rl_rewards.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=rl_rewards.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=rl_rewards.feet_too_near_humanoid,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["ankle_roll.*"]), "threshold": 0.18},
    )
    feet_stumble = RewTerm(
        func=rl_rewards.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=base_mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=rl_rewards.joint_deviation_l1,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["hip_yaw_.*_joint", "hip_roll_.*_joint", "hip_pitch_.*_joint"],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=rl_rewards.joint_deviation_l1,
        weight=-0.12,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "knee_pitch_.*_joint",
                    "ankle_pitch_.*_joint",
                    "ankle_roll_.*_joint",
                ],
            )
        },
    )
    ankle_torque = RewTerm(func=rl_rewards.ankle_torque, weight=-0.0005)
    ankle_action = RewTerm(func=rl_rewards.ankle_action, weight=-0.001)
    hip_roll_action = RewTerm(func=rl_rewards.hip_roll_action, weight=-0.5)
    hip_yaw_action = RewTerm(func=rl_rewards.hip_yaw_action, weight=-0.5)
    feet_y_distance = RewTerm(func=rl_rewards.feet_y_distance, weight=-1.0)


@configclass
class RealLiteUpperBodyEnvCfg(RealLiteWalkEnvCfg):
    amp_motion_files_display = [str(TASK_PRESETS["upper_body_real_lite"]["display_motion_file"])]
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=15.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=REAL_LITE_ARTICULATION_CFG,
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),
        ),
    )
    robot = RealLiteWalkEnvCfg.robot.__class__(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,
        terminate_contacts_body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_.*", "pelvis"],
        feet_body_names=["ankle_roll.*"],
        left_leg_joint_names=LEFT_LEG_JOINT_NAMES,
        right_leg_joint_names=RIGHT_LEG_JOINT_NAMES,
        left_arm_joint_names=LEFT_ARM_JOINT_NAMES,
        right_arm_joint_names=RIGHT_ARM_JOINT_NAMES,
        ankle_joint_names=ANKLE_JOINT_NAMES,
        feet_link_names=FEET_LINK_NAMES,
        elbow_link_names=ELBOW_LINK_NAMES,
    )
    reward = RealLiteUpperBodyRewardCfg()
    gait = RealLiteUpperBodyGaitCfg()
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        debug_vis=False,
        ranges=CommandRangesCfg(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=base_mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=base_mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "mass_distribution_params": (-5.0, 5.0),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=base_mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-0.2, 0.2)},
                    "velocity_range": {
                        "x": (0.0, 0.0),
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                        "roll": (-0.1, 0.1),
                        "pitch": (-0.1, 0.1),
                        "yaw": (-0.1, 0.1),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=base_mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.95, 1.05),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=None,
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )


@configclass
class RealLiteUpperBodyAgentCfg(RealLiteWalkAgentCfg):
    experiment_name = "upper_body_real_lite"
    neptune_project = "upper_body_real_lite"
    wandb_project = "upper_body_real_lite"
    amp_reward_coef = 0.5
    amp_motion_files = [str(TASK_PRESETS["upper_body_real_lite"]["amp_motion_file"])]
    amp_task_reward_lerp = 0.4
    min_normalized_std = [0.05] * POLICY_JOINT_COUNT
