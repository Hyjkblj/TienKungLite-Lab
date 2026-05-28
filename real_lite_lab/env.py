from __future__ import annotations

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate
from scipy.spatial.transform import Rotation

from rsl_rl.env import VecEnv
from rsl_rl.utils import AMPLoaderDisplay

from .constants import (
    OBS_ANG_VEL_DIM,
    OBS_COMMAND_DIM,
    OBS_GAIT_PHASE_DIM,
    OBS_GAIT_RATIO_DIM,
    OBS_PER_STEP_DIM,
    OBS_PROJECTED_GRAVITY_DIM,
    VIS_LEFT_ARM_POS_SLICE,
    VIS_LEFT_ARM_VEL_SLICE,
    VIS_LEFT_LEG_POS_SLICE,
    VIS_LEFT_LEG_VEL_SLICE,
    VIS_RIGHT_ARM_POS_SLICE,
    VIS_RIGHT_ARM_VEL_SLICE,
    VIS_RIGHT_LEG_POS_SLICE,
    VIS_RIGHT_LEG_VEL_SLICE,
    VIS_ROOT_LIN_VEL_SLICE,
    VISUALIZATION_FRAME_DIM,
)
from .run_cfg import RealLiteRunEnvCfg
from .scene import SceneCfg
from .walk_cfg import RealLiteWalkEnvCfg


def _build_actor_obs_slices(num_actions: int) -> dict[str, slice]:
    offset = 0
    slices = {}
    slices["ang_vel"] = slice(offset, offset + OBS_ANG_VEL_DIM)
    offset = slices["ang_vel"].stop
    slices["projected_gravity"] = slice(offset, offset + OBS_PROJECTED_GRAVITY_DIM)
    offset = slices["projected_gravity"].stop
    slices["command"] = slice(offset, offset + OBS_COMMAND_DIM)
    offset = slices["command"].stop
    slices["joint_pos"] = slice(offset, offset + num_actions)
    offset = slices["joint_pos"].stop
    slices["joint_vel"] = slice(offset, offset + num_actions)
    offset = slices["joint_vel"].stop
    slices["actions"] = slice(offset, offset + num_actions)
    offset = slices["actions"].stop
    slices["sin_phase"] = slice(offset, offset + OBS_GAIT_PHASE_DIM)
    offset = slices["sin_phase"].stop
    slices["cos_phase"] = slice(offset, offset + OBS_GAIT_PHASE_DIM)
    offset = slices["cos_phase"].stop
    slices["phase_ratio"] = slice(offset, offset + OBS_GAIT_RATIO_DIM)
    if slices["phase_ratio"].stop != OBS_PER_STEP_DIM:
        raise ValueError(
            f"Actor observation layout mismatch: expected {OBS_PER_STEP_DIM}, got {slices['phase_ratio'].stop}."
        )
    return slices


class RealLiteEnv(VecEnv):
    def __init__(self, cfg: RealLiteRunEnvCfg | RealLiteWalkEnvCfg, headless):
        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]

        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]
        if self.cfg.scene.lidar.enable_lidar:
            self.lidar: RayCaster = self.scene.sensors["lidar"]
        if self.cfg.scene.depth_camera.enable_depth_camera:
            self.depth_camera: TiledCamera = self.scene.sensors["depth_camera"]

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

        self.amp_loader_display = AMPLoaderDisplay(
            motion_files=self.cfg.amp_motion_files_display, device=self.device, time_between_frames=self.physics_dt
        )
        self.motion_len = self.amp_loader_display.trajectory_num_frames[0]

    def init_buffers(self):
        self.extras = {}
        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations
        self.action_scale = self.cfg.robot.action_scale

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.feet_body_ids, _ = self.robot.find_bodies(name_keys=self.cfg.robot.feet_link_names, preserve_order=True)
        self.elbow_body_ids, _ = self.robot.find_bodies(
            name_keys=self.cfg.robot.elbow_link_names, preserve_order=True
        )
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=self.cfg.robot.left_leg_joint_names,
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=self.cfg.robot.right_leg_joint_names,
            preserve_order=True,
        )
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=self.cfg.robot.left_arm_joint_names,
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=self.cfg.robot.right_arm_joint_names,
            preserve_order=True,
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=self.cfg.robot.ankle_joint_names,
            preserve_order=True,
        )
        self.policy_joint_names = (
            list(self.cfg.robot.left_leg_joint_names)
            + list(self.cfg.robot.right_leg_joint_names)
            + list(self.cfg.robot.left_arm_joint_names)
            + list(self.cfg.robot.right_arm_joint_names)
        )
        self.policy_joint_ids, _ = self.robot.find_joints(name_keys=self.policy_joint_names, preserve_order=True)
        self.num_actions = len(self.policy_joint_ids)
        if self.num_actions != len(self.policy_joint_names):
            raise ValueError("Resolved policy joint count does not match policy joint name count.")
        self.default_joint_pos_policy = self.robot.data.default_joint_pos[:, self.policy_joint_ids]
        self.default_joint_vel_policy = self.robot.data.default_joint_vel[:, self.policy_joint_ids]
        default_feet_y = self.robot.data.body_pos_w[0, self.feet_body_ids[0], 1] - self.robot.data.body_pos_w[0, self.feet_body_ids[1], 1]
        self.default_feet_y_dist = torch.abs(default_feet_y).item()
        self.actor_obs_slices = _build_actor_obs_slices(self.num_actions)

        policy_joint_name_to_action_idx = {name: idx for idx, name in enumerate(self.policy_joint_names)}
        self.ankle_action_ids = [policy_joint_name_to_action_idx[name] for name in self.cfg.robot.ankle_joint_names]
        self.hip_roll_action_ids = [
            policy_joint_name_to_action_idx[self.cfg.robot.left_leg_joint_names[0]],
            policy_joint_name_to_action_idx[self.cfg.robot.right_leg_joint_names[0]],
        ]
        self.hip_yaw_action_ids = [
            policy_joint_name_to_action_idx[self.cfg.robot.left_leg_joint_names[2]],
            policy_joint_name_to_action_idx[self.cfg.robot.right_leg_joint_names[2]],
        ]

        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.left_arm_local_vec = torch.tensor(self.cfg.robot.left_arm_local_vec, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.right_arm_local_vec = torch.tensor(self.cfg.robot.right_arm_local_vec, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.gait_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_cycle = torch.full(
            (self.num_envs,), self.cfg.gait.gait_cycle, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.phase_ratio = torch.tensor(
            [self.cfg.gait.gait_air_ratio_l, self.cfg.gait.gait_air_ratio_r], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self.phase_offset = torch.tensor(
            [self.cfg.gait.gait_phase_offset_l, self.cfg.gait.gait_phase_offset_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.prev_delayed_action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.current_delayed_action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.init_obs_buffer()

    def visualize_motion(self, time):
        visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        if visual_motion_frame.shape[0] != VISUALIZATION_FRAME_DIM:
            raise ValueError(
                f"Unexpected visualization frame size {visual_motion_frame.shape[0]}; expected {VISUALIZATION_FRAME_DIM}."
            )
        device = self.device
        dof_pos = self.robot.data.default_joint_pos.clone()
        dof_vel = self.robot.data.default_joint_vel.clone()

        dof_pos[:, self.left_leg_ids] = visual_motion_frame[VIS_LEFT_LEG_POS_SLICE]
        dof_pos[:, self.right_leg_ids] = visual_motion_frame[VIS_RIGHT_LEG_POS_SLICE]
        dof_pos[:, self.left_arm_ids] = visual_motion_frame[VIS_LEFT_ARM_POS_SLICE]
        dof_pos[:, self.right_arm_ids] = visual_motion_frame[VIS_RIGHT_ARM_POS_SLICE]
        dof_vel[:, self.left_leg_ids] = visual_motion_frame[VIS_LEFT_LEG_VEL_SLICE]
        dof_vel[:, self.right_leg_ids] = visual_motion_frame[VIS_RIGHT_LEG_VEL_SLICE]
        dof_vel[:, self.left_arm_ids] = visual_motion_frame[VIS_LEFT_ARM_VEL_SLICE]
        dof_vel[:, self.right_arm_ids] = visual_motion_frame[VIS_RIGHT_ARM_VEL_SLICE]

        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(dof_vel)

        env_ids = torch.arange(self.num_envs, device=device)
        root_pos = visual_motion_frame[:3].clone()
        root_pos[2] += 0.3

        euler = visual_motion_frame[3:6].cpu().numpy()
        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=device
        )
        lin_vel = visual_motion_frame[VIS_ROOT_LIN_VEL_SLICE].clone()
        ang_vel = torch.zeros_like(lin_vel)

        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.sim.render()
        self.sim.step()
        self.scene.update(dt=self.step_dt)

        return self._build_amp_observation(dof_pos, dof_vel)

    def _compute_relative_end_effector_positions(self):
        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7], self.left_arm_local_vec)
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7], self.right_arm_local_vec)
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        right_foot_pos = self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)
        return left_hand_pos, right_hand_pos, left_foot_pos, right_foot_pos

    def _build_amp_observation(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        left_hand_pos, right_hand_pos, left_foot_pos, right_foot_pos = self._compute_relative_end_effector_positions()
        return torch.cat(
            (
                joint_pos[:, self.right_arm_ids],
                joint_pos[:, self.left_arm_ids],
                joint_pos[:, self.right_leg_ids],
                joint_pos[:, self.left_leg_ids],
                joint_vel[:, self.right_arm_ids],
                joint_vel[:, self.left_arm_ids],
                joint_vel[:, self.right_leg_ids],
                joint_vel[:, self.left_leg_ids],
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos,
            ),
            dim=-1,
        )

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos[:, self.policy_joint_ids] - self.default_joint_pos_policy
        joint_vel = robot.data.joint_vel[:, self.policy_joint_ids] - self.default_joint_vel_policy
        action = self.current_delayed_action
        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5

        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                torch.sin(2 * torch.pi * self.gait_phase),
                torch.cos(2 * torch.pi * self.gait_phase),
                self.phase_ratio,
            ],
            dim=-1,
        )
        current_critic_obs = torch.cat([current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1)
        if current_actor_obs.shape[1] != OBS_PER_STEP_DIM:
            raise ValueError(
                f"Actor observation size mismatch: expected {OBS_PER_STEP_DIM}, got {current_actor_obs.shape[1]}."
            )
        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)
        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        if self.cfg.scene.depth_camera.enable_depth_camera:
            depth_image = self.depth_camera.data.output["distance_to_image_plane"]
            flattened_depth = depth_image.view(self.num_envs, -1)
            actor_obs = torch.cat([actor_obs, flattened_depth], dim=-1)
            critic_obs = torch.cat([critic_obs, flattened_depth], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)
        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0
        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None and self.cfg.scene.terrain_generator.curriculum:
            terrain_levels = self.update_terrain_levels(env_ids)
            self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf
        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.prev_delayed_action[env_ids] = 0.0
        self.current_delayed_action[env_ids] = 0.0
        self.action[env_ids] = 0.0
        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)
        self.prev_delayed_action.copy_(self.current_delayed_action)
        self.current_delayed_action.copy_(delayed_actions)
        self.action = torch.clip(self.current_delayed_action, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = self.action * self.action_scale + self.default_joint_pos_policy
        joint_position_targets = self.robot.data.default_joint_pos.clone()
        joint_position_targets[:, self.policy_joint_ids] = processed_actions

        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(joint_position_targets)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation
        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self._calculate_gait_para()
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        reset_buf = torch.any(
            torch.max(
                torch.norm(net_contact_forces[:, :, self.termination_contact_cfg.body_ids], dim=-1),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[self.actor_obs_slices["ang_vel"]] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[self.actor_obs_slices["projected_gravity"]] = (
                noise_scales.projected_gravity * self.obs_scales.projected_gravity
            )
            noise_vec[self.actor_obs_slices["command"]] = 0.0
            noise_vec[self.actor_obs_slices["joint_pos"]] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[self.actor_obs_slices["joint_vel"]] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[self.actor_obs_slices["actions"]] = 0.0
            noise_vec[self.actor_obs_slices["sin_phase"]] = 0.0
            noise_vec[self.actor_obs_slices["cos_phase"]] = 0.0
            noise_vec[self.actor_obs_slices["phase_ratio"]] = 0.0
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        return {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    def get_amp_obs_for_expert_trans(self):
        return self._build_amp_observation(self.robot.data.joint_pos, self.robot.data.joint_vel)

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)

    def _calculate_gait_para(self) -> None:
        t = self.episode_length_buf * self.step_dt / self.gait_cycle
        self.gait_phase[:, 0] = (t + self.phase_offset[:, 0]) % 1.0
        self.gait_phase[:, 1] = (t + self.phase_offset[:, 1]) % 1.0
