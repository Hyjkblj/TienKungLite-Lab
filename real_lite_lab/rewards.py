from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from .env import RealLiteEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: "RealLiteEnv", std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: "RealLiteEnv", std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: "RealLiteEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: "RealLiteEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: "RealLiteEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)


def joint_acc_l2(env: "RealLiteEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: "RealLiteEnv") -> torch.Tensor:
    return torch.sum(torch.square(env.current_delayed_action - env.prev_delayed_action), dim=1)


def undesired_contacts(env: "RealLiteEnv", threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def flat_orientation_l2(env: "RealLiteEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: "RealLiteEnv") -> torch.Tensor:
    return env.reset_buf * ~env.time_out_buf


def feet_slide(
    env: "RealLiteEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def body_force(
    env: "RealLiteEnv",
    sensor_cfg: SceneEntityCfg,
    threshold: float = 500,
    max_reward: float = 400,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    return (reward - threshold).clamp(min=0, max=max_reward)


def joint_deviation_l1(
    env: "RealLiteEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def body_orientation_l2(
    env: "RealLiteEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: "RealLiteEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: "RealLiteEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def ankle_torque(env: "RealLiteEnv") -> torch.Tensor:
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def ankle_action(env: "RealLiteEnv") -> torch.Tensor:
    return torch.sum(torch.abs(env.action[:, env.ankle_action_ids]), dim=1)


def hip_roll_action(env: "RealLiteEnv") -> torch.Tensor:
    return torch.sum(torch.abs(env.action[:, env.hip_roll_action_ids]), dim=1)


def hip_yaw_action(env: "RealLiteEnv") -> torch.Tensor:
    return torch.sum(torch.abs(env.action[:, env.hip_yaw_action_ids]), dim=1)


def feet_y_distance(env: "RealLiteEnv") -> torch.Tensor:
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - env.default_feet_y_dist)
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


def gait_clock(phase: torch.Tensor, air_ratio: torch.Tensor, delta_t: float):
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))
    trans_flag1 = phase < delta_t
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
    trans_flag3 = phase > (1 - delta_t)

    i_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
    )
    i_spd = 1.0 - i_frc
    return i_frc, i_spd


def gait_feet_frc_perio(env: "RealLiteEnv", delta_t: float = 0.02) -> torch.Tensor:
    left_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_score = left_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_score = right_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_score + right_score


def gait_feet_spd_perio(env: "RealLiteEnv", delta_t: float = 0.02) -> torch.Tensor:
    left_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_score = left_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_score = right_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    return left_score + right_score


def gait_feet_frc_support_perio(env: "RealLiteEnv", delta_t: float = 0.02) -> torch.Tensor:
    left_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_score = left_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_score = right_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_score + right_score
