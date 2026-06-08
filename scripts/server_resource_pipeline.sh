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

cd "${SERVER_REPO}"

echo "[INFO] repo: ${SERVER_REPO}"
echo "[INFO] task: ${TASK}"
echo "[INFO] exporting free-base USD"
python tools/reexport_real_lite_usd.py --headless --force

export TIENKUNG_LITE_USD_REL_PATH="urdf/humanoid_publish_free_base/humanoid_publish_free_base.usd"
echo "[INFO] TIENKUNG_LITE_USD_REL_PATH=${TIENKUNG_LITE_USD_REL_PATH}"

echo "[INFO] auditing resource pipeline"
python tools/audit_real_lite_resource_pipeline.py --strict

echo "[INFO] running Isaac free-base standing diagnostic"
ISAAC_STABILITY_ARGS=()
if [[ "${REQUIRE_STABLE}" == "1" ]]; then
  ISAAC_STABILITY_ARGS+=(--require_stable)
fi

python tools/isaac_standing_diagnostic.py \
  --task "${TASK}" \
  --headless \
  --duration "${ISAAC_HOLD_DURATION}" \
  --trace_out "logs/standing/isaac_freebase_baseline.npz" \
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
