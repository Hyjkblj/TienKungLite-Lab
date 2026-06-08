import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from ...alignment_config import SOFT_JOINT_POS_LIMIT_FACTOR
from .. import resolve_real_lite_asset_root
from ...constants import DEFAULT_JOINT_POS
from ...usd_asset_validation import FREE_BASE_USD_REL_PATH, resolve_real_lite_usd_path

ASSET_DIR = resolve_real_lite_asset_root()
USD_REL_PATH_ENV_VAR = "TIENKUNG_LITE_USD_REL_PATH"
ALLOW_FIXED_BASE_USD_ENV_VAR = "TIENKUNG_LITE_ALLOW_FIXED_BASE_USD"
DEFAULT_USD_REL_PATH = FREE_BASE_USD_REL_PATH


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_usd_path() -> Path:
    configured_rel_path = os.getenv(USD_REL_PATH_ENV_VAR)
    return resolve_real_lite_usd_path(
        ASSET_DIR,
        configured_rel_path=configured_rel_path,
        allow_fixed_base=_env_flag_enabled(ALLOW_FIXED_BASE_USD_ENV_VAR),
    )


USD_PATH = _resolve_usd_path()

REAL_LITE_ARTICULATION_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos=dict(DEFAULT_JOINT_POS),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=SOFT_JOINT_POS_LIMIT_FACTOR,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_roll_.*_joint",
                "hip_yaw_.*_joint",
                "hip_pitch_.*_joint",
                "knee_pitch_.*_joint",
            ],
            effort_limit_sim={
                "hip_roll_.*_joint": 150,
                "hip_yaw_.*_joint": 90,
                "hip_pitch_.*_joint": 150,
                "knee_pitch_.*_joint": 150,
            },
            velocity_limit_sim={
                "hip_roll_.*_joint": 12.0,
                "hip_yaw_.*_joint": 14.0,
                "hip_pitch_.*_joint": 12.0,
                "knee_pitch_.*_joint": 12.0,
            },
            stiffness={
                "hip_roll_.*_joint": 700,
                "hip_yaw_.*_joint": 500,
                "hip_pitch_.*_joint": 700,
                "knee_pitch_.*_joint": 700,
            },
            damping={
                "hip_roll_.*_joint": 10,
                "hip_yaw_.*_joint": 5,
                "hip_pitch_.*_joint": 10,
                "knee_pitch_.*_joint": 10,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "ankle_pitch_.*_joint",
                "ankle_roll_.*_joint",
            ],
            effort_limit_sim={
                "ankle_pitch_.*_joint": 60,
                "ankle_roll_.*_joint": 30,
            },
            velocity_limit_sim={
                "ankle_pitch_.*_joint": 7.8,
                "ankle_roll_.*_joint": 7.8,
            },
            stiffness={
                "ankle_pitch_.*_joint": 30,
                "ankle_roll_.*_joint": 16.8,
            },
            damping={
                "ankle_pitch_.*_joint": 2.5,
                "ankle_roll_.*_joint": 1.4,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_.*_joint",
                "shoulder_roll_.*_joint",
                "shoulder_yaw_.*_joint",
                "elbow_.*_joint",
            ],
            effort_limit_sim={
                "shoulder_pitch_.*_joint": 36,
                "shoulder_roll_.*_joint": 36,
                "shoulder_yaw_.*_joint": 36,
                "elbow_.*_joint": 36,
            },
            velocity_limit_sim={
                "shoulder_pitch_.*_joint": 7.8,
                "shoulder_roll_.*_joint": 7.8,
                "shoulder_yaw_.*_joint": 7.8,
                "elbow_.*_joint": 7.8,
            },
            stiffness={
                "shoulder_pitch_.*_joint": 60,
                "shoulder_roll_.*_joint": 20,
                "shoulder_yaw_.*_joint": 10,
                "elbow_.*_joint": 10,
            },
            damping={
                "shoulder_pitch_.*_joint": 3,
                "shoulder_roll_.*_joint": 1.5,
                "shoulder_yaw_.*_joint": 1,
                "elbow_.*_joint": 1,
            },
        ),
    },
)

TIENKUNG2LITE_REAL_CFG = REAL_LITE_ARTICULATION_CFG
