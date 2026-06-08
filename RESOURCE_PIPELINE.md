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
  --trace_out logs/standing/isaac_freebase_baseline.npz
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
