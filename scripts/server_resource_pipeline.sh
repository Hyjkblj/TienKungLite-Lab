#!/usr/bin/env bash
set -euo pipefail

SERVER_REPO="${SERVER_REPO:-/ai/users/huangwy/exp2/TienKungLite-Lab}"
TASK="${TASK:-walk_real_lite}"
ISAAC_HOLD_DURATION="${ISAAC_HOLD_DURATION:-6}"
RUN_MUJOCO_HOLD="${RUN_MUJOCO_HOLD:-0}"
MUJOCO_HOLD_DURATION="${MUJOCO_HOLD_DURATION:-30}"
INSTALL_SIM2SIM="${INSTALL_SIM2SIM:-0}"
REQUIRE_STABLE="${REQUIRE_STABLE:-1}"
MUJOCO_VIDEO_PATH="${MUJOCO_VIDEO_PATH:-logs/standing/mujoco_hold_${MUJOCO_HOLD_DURATION}s.mp4}"
USE_REFERENCE_FEET_COLLISIONS="${USE_REFERENCE_FEET_COLLISIONS:-0}"
REFERENCE_FEET_URDF="${REFERENCE_FEET_URDF:-}"
USD_REL_PATH="${USD_REL_PATH:-urdf/humanoid_publish_free_base/humanoid_publish_free_base.usd}"

cd "${SERVER_REPO}"

echo "[INFO] repo: ${SERVER_REPO}"
echo "[INFO] task: ${TASK}"
echo "[INFO] exporting free-base USD"
REEXPORT_ARGS=(--headless --force)
AUDIT_ARGS=()
if [[ "${USE_REFERENCE_FEET_COLLISIONS}" == "1" ]]; then
  if [[ -z "${REFERENCE_FEET_URDF}" ]]; then
    REFERENCE_FEET_URDF="$(
      python - <<'PY'
from real_lite_lab.assets import resolve_real_lite_asset_root
print(resolve_real_lite_asset_root() / "urdf" / "humanoid_publish.reference_feet.urdf")
PY
    )"
  fi
  echo "[INFO] generating reference-feet URDF: ${REFERENCE_FEET_URDF}"
  python tools/align_real_lite_urdf_to_reference.py \
    --reference-feet-only \
    --output-urdf "${REFERENCE_FEET_URDF}"
  REEXPORT_ARGS+=(--urdf_path "${REFERENCE_FEET_URDF}")
  AUDIT_ARGS+=(--urdf "${REFERENCE_FEET_URDF}")
fi

python tools/reexport_real_lite_usd.py "${REEXPORT_ARGS[@]}"

export TIENKUNG_LITE_USD_REL_PATH="${USD_REL_PATH}"
echo "[INFO] TIENKUNG_LITE_USD_REL_PATH=${TIENKUNG_LITE_USD_REL_PATH}"

echo "[INFO] auditing resource pipeline"
python tools/audit_real_lite_resource_pipeline.py --strict "${AUDIT_ARGS[@]}"

echo "[INFO] running Isaac free-base standing diagnostic"
ISAAC_STABILITY_ARGS=()
if [[ "${REQUIRE_STABLE}" == "1" ]]; then
  ISAAC_STABILITY_ARGS+=(--require_stable)
fi
ISAAC_DIAGNOSTIC_ARGS=()
append_optional_arg() {
  local env_name="$1"
  local cli_name="$2"
  local env_value="${!env_name-}"
  if [[ -n "${env_value}" ]]; then
    ISAAC_DIAGNOSTIC_ARGS+=("${cli_name}" "${env_value}")
  fi
}
append_optional_arg ISAAC_ROOT_Z --root_z
append_optional_arg ISAAC_SETTLE_TIME --settle_time
append_optional_arg ISAAC_HIP_PITCH_TARGET --hip_pitch_target
append_optional_arg ISAAC_KNEE_PITCH_TARGET --knee_pitch_target
append_optional_arg ISAAC_ANKLE_PITCH_TARGET --ankle_pitch_target
append_optional_arg ISAAC_HIP_PITCH_KP_SCALE --hip_pitch_kp_scale
append_optional_arg ISAAC_HIP_PITCH_KD_SCALE --hip_pitch_kd_scale
append_optional_arg ISAAC_KNEE_PITCH_KP_SCALE --knee_pitch_kp_scale
append_optional_arg ISAAC_KNEE_PITCH_KD_SCALE --knee_pitch_kd_scale
append_optional_arg ISAAC_ANKLE_PITCH_KP_SCALE --ankle_pitch_kp_scale
append_optional_arg ISAAC_ANKLE_PITCH_KD_SCALE --ankle_pitch_kd_scale
append_optional_arg ISAAC_ANKLE_ROLL_KP_SCALE --ankle_roll_kp_scale
append_optional_arg ISAAC_ANKLE_ROLL_KD_SCALE --ankle_roll_kd_scale
if [[ "${ISAAC_CONTINUE_AFTER_TERMINATION:-0}" == "1" ]]; then
  ISAAC_DIAGNOSTIC_ARGS+=(--continue_after_termination)
fi
if [[ "${#ISAAC_DIAGNOSTIC_ARGS[@]}" -gt 0 ]]; then
  printf '[INFO] Isaac diagnostic extra args:'
  printf ' %q' "${ISAAC_DIAGNOSTIC_ARGS[@]}"
  printf '\n'
fi

python tools/isaac_standing_diagnostic.py \
  --task "${TASK}" \
  --headless \
  --duration "${ISAAC_HOLD_DURATION}" \
  --trace_out "logs/standing/isaac_freebase_baseline.npz" \
  "${ISAAC_DIAGNOSTIC_ARGS[@]}" \
  "${ISAAC_STABILITY_ARGS[@]}"

if [[ "${RUN_MUJOCO_HOLD}" == "1" ]]; then
  if [[ "${INSTALL_SIM2SIM}" == "1" ]]; then
    echo "[INFO] installing optional sim2sim dependencies"
    python -m pip install -e ".[sim2sim]"
  fi

  echo "[INFO] generating MuJoCo MJCF"
  python tools/generate_real_lite_mjcf.py

  echo "[INFO] running MuJoCo hold diagnostic"
  python sim2sim_real_lite.py \
    --task "${TASK}" \
    --control_mode hold \
    --duration "${MUJOCO_HOLD_DURATION}" \
    --trace_out "logs/standing/mujoco_hold_${MUJOCO_HOLD_DURATION}s.npz" \
    --trace_steps "$((MUJOCO_HOLD_DURATION * 50 + 1))" \
    --save_video "${MUJOCO_VIDEO_PATH}" \
    --camera follow_side \
    --fps 20 \
    --width 960 \
    --height 540 \
    --settle_steps 120
fi

echo "[INFO] resource pipeline completed"
