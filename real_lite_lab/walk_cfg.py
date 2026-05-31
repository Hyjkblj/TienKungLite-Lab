from __future__ import annotations

import math

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
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
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
from .isaaclab_compat import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from .terrain_cfg import REAL_LITE_GRAVEL_TERRAINS_CFG


@configclass
class RealLiteGaitCfg:
    gait_air_ratio_l: float = TASK_PRESETS["walk_real_lite"]["gait_air_ratio_l"]
    gait_air_ratio_r: float = TASK_PRESETS["walk_real_lite"]["gait_air_ratio_r"]
    gait_phase_offset_l: float = TASK_PRESETS["walk_real_lite"]["gait_phase_offset_l"]
    gait_phase_offset_r: float = TASK_PRESETS["walk_real_lite"]["gait_phase_offset_r"]
    gait_cycle: float = TASK_PRESETS["walk_real_lite"]["gait_cycle"]


@configclass
class RealLiteRewardCfg:
    track_lin_vel_xy_exp = RewTerm(func=rl_rewards.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=rl_rewards.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=rl_rewards.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=rl_rewards.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=rl_rewards.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=rl_rewards.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=rl_rewards.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=rl_rewards.undesired_contacts,
        weight=-1.0,
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
        weight=-2.0,
    )
    flat_orientation_l2 = RewTerm(func=rl_rewards.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=rl_rewards.is_terminated, weight=-200.0)
    feet_slide = RewTerm(
        func=rl_rewards.feet_slide,
        weight=-0.25,
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
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["ankle_roll.*"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=rl_rewards.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=base_mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=rl_rewards.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_yaw_.*_joint",
                    "hip_roll_.*_joint",
                    "shoulder_pitch_.*_joint",
                    "elbow_.*_joint",
                ],
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=rl_rewards.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_roll_.*_joint", "shoulder_yaw_.*_joint"])},
    )
    joint_deviation_legs = RewTerm(
        func=rl_rewards.joint_deviation_l1,
        weight=-0.02,
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
    gait_feet_frc_perio = RewTerm(func=rl_rewards.gait_feet_frc_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_spd_perio = RewTerm(func=rl_rewards.gait_feet_spd_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_frc_support_perio = RewTerm(
        func=rl_rewards.gait_feet_frc_support_perio, weight=0.6, params={"delta_t": 0.02}
    )
    ankle_torque = RewTerm(func=rl_rewards.ankle_torque, weight=-0.0005)
    ankle_action = RewTerm(func=rl_rewards.ankle_action, weight=-0.001)
    hip_roll_action = RewTerm(func=rl_rewards.hip_roll_action, weight=-1.0)
    hip_yaw_action = RewTerm(func=rl_rewards.hip_yaw_action, weight=-1.0)
    feet_y_distance = RewTerm(func=rl_rewards.feet_y_distance, weight=-2.0)


@configclass
class RealLiteWalkEnvCfg:
    amp_motion_files_display = [str(TASK_PRESETS["walk_real_lite"]["display_motion_file"])]
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=REAL_LITE_ARTICULATION_CFG,
        terrain_type="generator",
        terrain_generator=REAL_LITE_GRAVEL_TERRAINS_CFG,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),
        ),
    )
    robot: RobotCfg = RobotCfg(
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
    reward = RealLiteRewardCfg()
    gait = RealLiteGaitCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.57, 1.57),
            heading=(-math.pi, math.pi),
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
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
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=base_mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=base_mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass
class RealLiteWalkAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    clip_actions = 100.0
    save_interval = 100
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "walk_real_lite"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk_real_lite"
    wandb_project = "walk_real_lite"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    amp_reward_coef = 0.3
    amp_motion_files = [str(TASK_PRESETS["walk_real_lite"]["amp_motion_file"])]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * POLICY_JOINT_COUNT
