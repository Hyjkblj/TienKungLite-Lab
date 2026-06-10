# Real Lite Resource Pipeline

This repository should treat the URDF plus shared Python constants as the local source of truth. Server-side Isaac exports and MuJoCo sim2sim models should be regenerated and audited from that source instead of edited by hand.

## Local Work

Run this before using the server:

```bash
python tools/audit_real_lite_resource_pipeline.py
```

The audit writes:

```text
logs/resource_audit/real_lite_resource_audit.md
logs/resource_audit/real_lite_resource_audit.json
```

Use strict mode in CI or before a serious training run:

```bash
python tools/audit_real_lite_resource_pipeline.py --strict
```

Expected local responsibilities:

- Verify the canonical URDF exists under the asset root.
- Verify policy joints exist and `DEFAULT_JOINT_POS` is inside URDF limits.
- Detect missing or fixed-root free-base USD exports.
- Compare MJCF joint limits, actuator order, sensor order, mass, and support geoms against the URDF/policy assumptions.
- Flag mesh collision usage, especially mesh-only foot contacts.

## Server Work

The target server checkout is:

```text
/ai/users/huangwy/exp2/TienKungLite-Lab
```

Run the resource pipeline from the Isaac Lab Python environment:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
bash scripts/server_resource_pipeline.sh
```

Or run the steps manually. Generate the free-base USD on the Isaac Lab server:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
python tools/reexport_real_lite_usd.py --headless --force
export TIENKUNG_LITE_USD_REL_PATH=urdf/humanoid_publish_free_base/humanoid_publish_free_base.usd
python tools/audit_real_lite_resource_pipeline.py --strict
```

Run a free-base Isaac standing diagnostic before PPO:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
export TIENKUNG_LITE_USD_REL_PATH=urdf/humanoid_publish_free_base/humanoid_publish_free_base.usd
python tools/isaac_standing_diagnostic.py \
  --task walk_real_lite \
  --headless \
  --duration 6 \
  --trace_out logs/standing/isaac_freebase_baseline.npz \
  --require_stable
```

If the robot falls immediately in IsaacLab, do not start PPO yet. First capture a no-reset, no-settle diagnostic so the first unstable frame is not hidden by environment resets:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
export TIENKUNG_LITE_USD_REL_PATH=urdf/humanoid_publish_free_base/humanoid_publish_free_base.usd
python tools/isaac_standing_diagnostic.py \
  --task walk_real_lite \
  --headless \
  --duration 2 \
  --settle_time 0 \
  --continue_after_termination \
  --trace_out logs/standing/isaac_no_settle_raw_fall.npz
```

Then run a small coarse sweep for root height and sagittal standing pose:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
export TIENKUNG_LITE_USD_REL_PATH=urdf/humanoid_publish_free_base/humanoid_publish_free_base.usd
python tools/run_isaac_standing_sweep.py \
  --run-dir logs/standing/isaac_pose_sweep_$(date +%Y%m%d_%H%M%S) \
  --duration 3 \
  --settle-time 0 \
  --root-zs 0.95 1.00 \
  --hip-pitch-targets -0.45 -0.50 \
  --knee-pitch-targets 0.90 1.00 \
  --ankle-pitch-targets -0.42 -0.50
```

Use the generated traces to estimate a grounded root height. The current Real Lite USD tends to report grounded foot body origins around `z=0.05m`, so traces whose initial `feet_z_w` is much higher are still falling from the air:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
python tools/recommend_isaac_root_height.py \
  logs/standing/isaac_pose_sweep_*/isaac_standing_*.npz \
  --target-foot-z 0.05 \
  --csv-out logs/standing/isaac_root_height_recommendations.csv
```

After choosing the best pose row from `*_summary.csv`, sweep a small PD matrix around that pose:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
export TIENKUNG_LITE_USD_REL_PATH=urdf/humanoid_publish_free_base/humanoid_publish_free_base.usd
python tools/run_isaac_standing_sweep.py \
  --run-dir logs/standing/isaac_pd_sweep_$(date +%Y%m%d_%H%M%S) \
  --duration 3 \
  --settle-time 0 \
  --root-zs 1.00 \
  --hip-pitch-targets -0.45 \
  --knee-pitch-targets 0.90 \
  --ankle-pitch-targets -0.42 \
  --knee-pitch-kd-scales 1.0 1.5 2.0 \
  --ankle-pitch-kp-scales 1.0 2.0 3.0 \
  --ankle-pitch-kd-scales 1.0 2.0 3.0
```

Use the best rows in the generated `*_summary.csv` for a longer hold:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
export TIENKUNG_LITE_USD_REL_PATH=urdf/humanoid_publish_free_base/humanoid_publish_free_base.usd
python tools/isaac_standing_diagnostic.py \
  --task walk_real_lite \
  --headless \
  --duration 30 \
  --root_z 1.00 \
  --hip_pitch_target -0.45 \
  --knee_pitch_target 0.90 \
  --ankle_pitch_target -0.42 \
  --ankle_pitch_kp_scale 2.0 \
  --ankle_pitch_kd_scale 2.0 \
  --trace_out logs/standing/isaac_candidate_30s.npz \
  --require_stable
```

If the audit still reports mesh-only foot contact, generate a safer URDF copy with primitive reference foot collisions, review it, then use it as the export source:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
python tools/align_real_lite_urdf_to_reference.py \
  --reference-feet-only
python tools/reexport_real_lite_usd.py \
  --headless \
  --force \
  --urdf_path "$(python - <<'PY'
from real_lite_lab.assets import resolve_real_lite_asset_root
print(resolve_real_lite_asset_root() / "urdf" / "humanoid_publish.reference_feet.urdf")
PY
)"
```

For the current IsaacLab standing issue, use the script switch below to run the same export, audit, and no-action hold without changing joint limits or mass:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
USE_REFERENCE_FEET_COLLISIONS=1 \
ISAAC_HOLD_DURATION=8 \
ISAAC_SETTLE_TIME=0 \
ISAAC_ROOT_Z=0.785 \
ISAAC_HIP_PITCH_TARGET=-0.50 \
ISAAC_KNEE_PITCH_TARGET=0.90 \
ISAAC_ANKLE_PITCH_TARGET=-0.50 \
ISAAC_ANKLE_PITCH_KD_SCALE=2.0 \
REQUIRE_STABLE=0 \
bash scripts/server_resource_pipeline.sh
```

If this run still falls forward, pull the latest diagnostic code and rerun once more before changing pose/PD. The newer trace prints both pelvis/root and whole-body COM relative to the foot center, which separates a true COM support-polygon failure from a root-origin geometry offset:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
git pull --ff-only origin main
USE_REFERENCE_FEET_COLLISIONS=1 \
ISAAC_HOLD_DURATION=8 \
ISAAC_SETTLE_TIME=0 \
ISAAC_ROOT_Z=0.785 \
ISAAC_HIP_PITCH_TARGET=-0.50 \
ISAAC_KNEE_PITCH_TARGET=0.90 \
ISAAC_ANKLE_PITCH_TARGET=-0.50 \
ISAAC_ANKLE_PITCH_KD_SCALE=2.0 \
REQUIRE_STABLE=0 \
bash scripts/server_resource_pipeline.sh
```

In the output, compare `root_xy_minus_feet_center_xy` with `com_xy_minus_feet_center_xy` at `tilt_event` and `drop_event`.

If COM itself moves forward past the support center, run a clearly marked support-polygon diagnostic. This does not claim to be the final physical foot model; it only tests whether a longer/more-forward primitive support contact can stop the forward fall:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
git pull --ff-only origin main
USE_REFERENCE_FEET_COLLISIONS=1 \
REFERENCE_FEET_SUPPORT_X=0.07 \
REFERENCE_FEET_SUPPORT_LENGTH=0.34 \
ISAAC_HOLD_DURATION=8 \
ISAAC_SETTLE_TIME=0 \
ISAAC_ROOT_Z=0.785 \
ISAAC_HIP_PITCH_TARGET=-0.50 \
ISAAC_KNEE_PITCH_TARGET=0.90 \
ISAAC_ANKLE_PITCH_TARGET=-0.50 \
ISAAC_ANKLE_PITCH_KD_SCALE=2.0 \
REQUIRE_STABLE=0 \
bash scripts/server_resource_pipeline.sh
```

If this increases the hold time materially, the next asset task is to define a realistic sole/toe collision model instead of continuing to tune PPO or PD blindly.

If the support-polygon diagnostic does not improve the hold, inspect the saved trace offline before running more sweeps:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
git pull --ff-only origin main
python tools/analyze_isaac_standing_trace.py logs/standing/isaac_freebase_baseline.npz
```

Focus on `start_state top_joint_vel`, `start_state top_joint_pos_error`, and `start_state top_joint_applied_torque`; large initial sagittal velocities/errors mean the pose is not a static equilibrium even before the visible fall.

For the current forward-fall case, run a small static-equilibrium sweep and rank rows by both hold time and the new `start_*` columns in the CSV:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
git pull --ff-only origin main
USE_REFERENCE_FEET_COLLISIONS=1 REQUIRE_STABLE=0 ISAAC_HOLD_DURATION=1 bash scripts/server_resource_pipeline.sh
export TIENKUNG_LITE_USD_REL_PATH=urdf/humanoid_publish_free_base/humanoid_publish_free_base.usd
python tools/run_isaac_standing_sweep.py \
  --run-dir logs/standing/isaac_static_equilibrium_$(date +%Y%m%d_%H%M%S) \
  --duration 3 \
  --settle-time 0 \
  --root-zs 0.785 \
  --hip-pitch-targets -0.55 -0.50 -0.45 \
  --knee-pitch-targets 0.80 0.90 1.00 \
  --ankle-pitch-targets -0.60 -0.55 -0.50 -0.45 \
  --ankle-pitch-kd-scales 2.0 3.0 4.0
```

Prefer candidates that reduce `start_joint_speed_abs_max`, `start_joint_pos_error_abs_max`, and `start_applied_torque_abs_max` without making `tilt_20_time` earlier.

If a zero-start-speed candidate also reports `foot_force_total_start=0`, do not treat it as grounded static balance yet. First sweep `root_z` around that pose and prefer rows with non-zero initial foot force and lower start-state velocity/torque:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
export TIENKUNG_LITE_USD_REL_PATH=urdf/humanoid_publish_free_base/humanoid_publish_free_base.usd
python tools/run_isaac_standing_sweep.py \
  --run-dir logs/standing/isaac_rootz_static_contact_$(date +%Y%m%d_%H%M%S) \
  --duration 3 \
  --settle-time 0 \
  --root-zs 0.760 0.770 0.780 0.785 0.790 \
  --hip-pitch-targets -0.55 \
  --knee-pitch-targets 1.00 \
  --ankle-pitch-targets -0.50 \
  --ankle-pitch-kd-scales 2.0 3.0 4.0
```

In the summary, reject rows whose `foot_force_total_start` is near zero even if their `start_*` metrics look perfect.

When one root height falls forward and a lower height falls backward, run a narrow root/ankle sweep and compare the signed `com_x_minus_feet_center_tilt20/drop/termination` columns:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
export TIENKUNG_LITE_USD_REL_PATH=urdf/humanoid_publish_free_base/humanoid_publish_free_base.usd
python tools/run_isaac_standing_sweep.py \
  --run-dir logs/standing/isaac_rootz_ankle_balance_$(date +%Y%m%d_%H%M%S) \
  --duration 5 \
  --settle-time 0 \
  --root-zs 0.772 0.774 0.776 0.778 0.780 0.782 \
  --hip-pitch-targets -0.55 \
  --knee-pitch-targets 1.00 \
  --ankle-pitch-targets -0.52 -0.50 -0.48 \
  --ankle-pitch-kd-scales 3.0
```

Prefer rows with non-zero `foot_force_total_start`, later/no `tilt_20_time`, and smaller signed COM drift at the first failure event.

If the best pose remains stuck around a few seconds and neighboring `root_z` values flip between forward and backward falls, stop broad pose sweeps and test asset physics variants at the same pose. This isolates whether the blocker is foot collision, reference mass/inertia alignment, or candidate-only fixed-link mass folded into the pelvis during USD import:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
git pull --ff-only origin main
python tools/run_isaac_asset_variant_sweep.py \
  --run-dir logs/standing/isaac_asset_variant_$(date +%Y%m%d_%H%M%S) \
  --duration 8 \
  --settle-time 0 \
  --root-z 0.782 \
  --hip-pitch-target -0.55 \
  --knee-pitch-target 1.00 \
  --ankle-pitch-target -0.50 \
  --ankle-pitch-kd-scale 3.0
```

Ranked output is written to `isaac_asset_variant_summary.csv`. If `reference_feet_mass_zero_fixed` or `reference_aligned_mass_zero_fixed` clearly outperforms `reference_feet`, treat the standing issue as an asset mass/inertia/import problem before doing more PPO or pose sweeps.

If `reference_feet` remains best but the hold time is extremely sensitive to sub-millimeter `root_z`, run a flat-sole diagnostic before sweeping more pose values. This replaces the two round toe rails with one box sole per foot, matching the MuJoCo sole dimensions:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
git pull --ff-only origin main
python tools/run_isaac_asset_variant_sweep.py \
  --run-dir logs/standing/isaac_flat_sole_contact_$(date +%Y%m%d_%H%M%S) \
  --variants reference_feet flat_sole \
  --duration 10 \
  --settle-time 0 \
  --root-z 0.76984 \
  --hip-pitch-target -0.55 \
  --knee-pitch-target 1.00 \
  --ankle-pitch-target -0.50 \
  --ankle-pitch-kd-scale 3.0
```

If flat-sole and PD sweeps still do not exceed a few seconds, stop pure zero-action PD search and train the dedicated stand policy. The current seed pose for that task is the best grounded static candidate found so far: `root_z=0.76984`, `hip=-0.55`, `knee=1.00`, `ankle=-0.50`, reference-feet USD, and default hip/knee damping.

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
git pull --ff-only origin main
export TIENKUNG_LITE_USD_REL_PATH=urdf/humanoid_publish_asset_variant_reference_feet/humanoid_publish_asset_variant_reference_feet.usd
python train_real_lite.py \
  --task stand_real_lite \
  --headless \
  --logger tensorboard \
  --num_envs 4096 \
  --max_iterations 3000 \
  --run_name stand_seed_rz_0p76984_reference_feet
```

Evaluate the trained stand checkpoint before starting locomotion. The deterministic check should pass first, then the
reset-noise check should also pass:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
git pull --ff-only origin main
export TIENKUNG_LITE_USD_REL_PATH=urdf/humanoid_publish_asset_variant_reference_feet/humanoid_publish_asset_variant_reference_feet.usd
RUN_DIR=$(ls -td logs/stand_real_lite/*stand_seed_rz_0p76984_reference_feet* | head -1)

python eval_stand_real_lite.py \
  --task stand_real_lite \
  --headless \
  --num_envs 128 \
  --duration_s 30 \
  --load_run "$(basename "$RUN_DIR")"

python eval_stand_real_lite.py \
  --task stand_real_lite \
  --headless \
  --num_envs 128 \
  --duration_s 30 \
  --keep_reset_noise \
  --load_run "$(basename "$RUN_DIR")"
```

After both stand evaluations pass, warm-start walking from the stand actor-critic weights. This is not a normal
`--resume`: walk uses `AmpOnPolicyRunner`, so the AMP discriminator and optimizer state must start fresh.

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
export TIENKUNG_LITE_USD_REL_PATH=urdf/humanoid_publish_asset_variant_reference_feet/humanoid_publish_asset_variant_reference_feet.usd
STAND_RUN=$(ls -td logs/stand_real_lite/*stand_seed_rz_0p76984_reference_feet* | head -1)
STAND_CKPT=$(ls "$STAND_RUN"/model_*.pt | sort -V | tail -1)

python train_real_lite.py \
  --task walk_real_lite \
  --headless \
  --logger tensorboard \
  --num_envs 4096 \
  --max_iterations 5000 \
  --run_name walk_warmstart_from_stand_reference_feet \
  --init_policy_checkpoint "$STAND_CKPT"
```

After walk training finishes, export and evaluate the policy in Isaac Lab before sim2sim:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
export TIENKUNG_LITE_USD_REL_PATH=urdf/humanoid_publish_asset_variant_reference_feet/humanoid_publish_asset_variant_reference_feet.usd
RUN_DIR=$(ls -td logs/walk_real_lite/*walk_resume_from_1900* logs/walk_real_lite/*walk_warmstart_from_stand_reference_feet_retry_clean_gpu* 2>/dev/null | head -1)
CKPT=$(ls "$RUN_DIR"/model_*.pt | sort -V | tail -1)

python play_real_lite.py \
  --task walk_real_lite \
  --headless \
  --num_envs 50 \
  --load_run "$(basename "$RUN_DIR")" \
  --checkpoint "$(basename "$CKPT")"

python eval_walk_real_lite.py \
  --task walk_real_lite \
  --headless \
  --num_envs 128 \
  --duration_s 30 \
  --command_vx 0.2 \
  --load_run "$(basename "$RUN_DIR")" \
  --checkpoint "$(basename "$CKPT")"
```

Only after Isaac free-base hold is stable should MuJoCo hold be used as a sim2sim check:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
export TIENKUNG_LITE_USD_REL_PATH=urdf/humanoid_publish_free_base/humanoid_publish_free_base.usd
pip install -e ".[sim2sim]"
python tools/generate_real_lite_mjcf.py
python sim2sim_real_lite.py \
  --task walk_real_lite \
  --control_mode hold \
  --duration 30 \
  --trace_out logs/standing/mujoco_hold_30s.npz \
  --trace_steps 1501 \
  --save_video logs/standing/mujoco_hold_30s.mp4 \
  --camera follow_side \
  --settle_steps 120
```

The same MuJoCo phase can be run through the script:

```bash
cd /ai/users/huangwy/exp2/TienKungLite-Lab
RUN_MUJOCO_HOLD=1 bash scripts/server_resource_pipeline.sh
```

## Acceptance Criteria

- Local audit has no blockers except `FREE_BASE_USD_MISSING` before the server export step.
- Server-generated free-base USD does not contain fixed-root markers.
- Isaac free-base hold has no early termination, no root drop, and no tilt event during the diagnostic window.
- MuJoCo hold uses the same default pose and actuator assumptions and holds for 30 seconds before policy training continues.
