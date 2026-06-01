# TienKungLiteLab

`TienKungLiteLab` is an independent Isaac Lab training pipeline for the TienKung Lite humanoid robot.

This repository keeps the new architecture self-contained except for the large robot geometry assets:
- motion datasets live under `real_lite_lab/datasets`
- task configs, environment code, and rewards live under `real_lite_lab`
- training, export, and MuJoCo sim2sim entrypoints live at the repository root
- the local AMP runtime is vendored under `rsl_rl`

The internal Python package name remains `real_lite_lab` for compatibility with the current codebase.

## Versions

- Isaac Sim: `4.5.0`
- Isaac Lab: `2.1.0`

## Installation

Clone the repository outside the Isaac Lab source tree, then install both the main package and the vendored `rsl_rl` package with the Python environment that already has Isaac Lab installed.

```bash
cd TienKungLiteLab
pip install -e .
pip install -e ./rsl_rl
```

## External Robot Assets

The large robot resource files are not stored in this repository.

Expected external asset directory contents:
- `meshes/`
- `urdf/`

By default, the code looks for assets at:

```text
../lite_urdf_publish/x_humanoid_0430_newfeet_newbody_publish
```

relative to the repository parent directory.

If your assets live somewhere else, set:

```bash
export TIENKUNG_LITE_ASSET_ROOT=/path/to/x_humanoid_0430_newfeet_newbody_publish
```

On Windows PowerShell:

```powershell
$env:TIENKUNG_LITE_ASSET_ROOT="D:\path\to\x_humanoid_0430_newfeet_newbody_publish"
```

## Repository Layout

- `train_real_lite.py`
  Training entrypoint for `walk_real_lite` and `run_real_lite`
- `play_real_lite.py`
  Loads a checkpoint and exports `policy.pt` and `policy.onnx`
- `sim2sim_real_lite.py`
  Runs MuJoCo-side closed-loop validation with an exported policy
- `scripts/gmr_data_conversion.py`
  Converts retargeted PKL motion data into visualization motion format
- `scripts/play_amp_animation.py`
  Replays visualization motion and optionally exports AMP expert motion
- `tools/generate_real_lite_mjcf.py`
  Generates `mjcf/real_lite.xml` from the repository asset definitions

## Training

```bash
python train_real_lite.py --task walk_real_lite --headless --logger tensorboard --num_envs 4096
python train_real_lite.py --task run_real_lite --headless --logger tensorboard --num_envs 4096
```

## Export

```bash
python play_real_lite.py --task walk_real_lite --headless --load_run <run_dir>
python play_real_lite.py --task run_real_lite --headless --load_run <run_dir>
```

Exported policies are written to:

```text
logs/<experiment>/<run>/exported/policy.pt
logs/<experiment>/<run>/exported/policy.onnx
```

## Motion Data Workflow

Step 1: convert retargeted PKL data into visualization motion data.

```bash
python scripts/gmr_data_conversion.py --input_pkl <motion.pkl> --output_txt real_lite_lab/datasets/motion_visualization/walk.txt
```

Step 2: replay the visualization motion and export AMP expert motion if needed.

```bash
python scripts/play_amp_animation.py --task walk_real_lite --num_envs 1
python scripts/play_amp_animation.py --task walk_real_lite --num_envs 1 --save_path real_lite_lab/datasets/motion_amp_expert/walk.txt --fps 30.0
```

The same workflow applies to `run_real_lite`.

## MuJoCo Sim2Sim

Generate the MuJoCo model first:

```bash
python tools/generate_real_lite_mjcf.py
```

Then run closed-loop validation:

```bash
python sim2sim_real_lite.py --task walk_real_lite --policy <policy.pt>
python sim2sim_real_lite.py --task run_real_lite --policy <policy.pt>
```

On headless servers, export an offscreen rollout video instead of opening the interactive viewer:

```bash
python sim2sim_real_lite.py --task walk_real_lite --policy <policy.pt> --save_video logs/walk_real_lite/rollout.mp4
```

## Notes

- This repository currently covers training, export, motion visualization, and MuJoCo sim2sim validation.
- The real-robot deployment path is intentionally kept outside this repository for now.
- If you change the URDF, joint limits, or inertial parameters, regenerate and revalidate `mjcf/real_lite.xml`.
