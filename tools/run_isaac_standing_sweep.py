from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_lite_lab.constants import POLICY_JOINT_NAMES  # noqa: E402


@dataclass(frozen=True)
class SweepConfig:
    root_z: float | None
    hip_pitch_target: float
    knee_pitch_target: float
    ankle_pitch_target: float
    hip_pitch_kp_scale: float
    hip_pitch_kd_scale: float
    knee_pitch_kp_scale: float
    knee_pitch_kd_scale: float
    ankle_pitch_kp_scale: float
    ankle_pitch_kd_scale: float
    ankle_roll_kp_scale: float
    ankle_roll_kd_scale: float


FIELD_ORDER = (
    "root_z",
    "hip_pitch_target",
    "knee_pitch_target",
    "ankle_pitch_target",
    "hip_pitch_kp_scale",
    "hip_pitch_kd_scale",
    "knee_pitch_kp_scale",
    "knee_pitch_kd_scale",
    "ankle_pitch_kp_scale",
    "ankle_pitch_kd_scale",
    "ankle_roll_kp_scale",
    "ankle_roll_kd_scale",
)
FIELD_LABELS = {
    "root_z": "rz",
    "hip_pitch_target": "hp",
    "knee_pitch_target": "kp",
    "ankle_pitch_target": "ap",
    "hip_pitch_kp_scale": "hpkp",
    "hip_pitch_kd_scale": "hpkd",
    "knee_pitch_kp_scale": "kkp",
    "knee_pitch_kd_scale": "kkd",
    "ankle_pitch_kp_scale": "apkp",
    "ankle_pitch_kd_scale": "apkd",
    "ankle_roll_kp_scale": "arkp",
    "ankle_roll_kd_scale": "arkd",
}


def _format_float_tag(value: float | None) -> str:
    if value is None:
        return "default"
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _build_variant_label(config: SweepConfig, varying_fields: tuple[str, ...]) -> str:
    if not varying_fields:
        return "base"
    return "__".join(f"{FIELD_LABELS[field]}_{_format_float_tag(getattr(config, field))}" for field in varying_fields)


def _first_index(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    return int(indices[0])


def _tilt_deg_from_projected_gravity(projected_gravity: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(projected_gravity, axis=1)
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    cos_theta = np.clip(-projected_gravity[:, 2] / safe_norms, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _joint_signal_metrics(
    trace: dict[str, np.ndarray],
    *,
    trace_key: str,
    label: str,
    times: np.ndarray,
) -> dict[str, object]:
    if trace_key not in trace:
        return {
            f"{label}_abs_max": None,
            f"{label}_abs_max_time": None,
            f"{label}_abs_max_joint": None,
        }

    values = np.asarray(trace[trace_key], dtype=np.float64)
    if values.ndim != 2 or not np.any(np.isfinite(values)):
        return {
            f"{label}_abs_max": None,
            f"{label}_abs_max_time": None,
            f"{label}_abs_max_joint": None,
        }

    finite_abs = np.where(np.isfinite(values), np.abs(values), np.nan)
    flat_index = int(np.nanargmax(finite_abs))
    frame_idx, joint_idx = np.unravel_index(flat_index, finite_abs.shape)
    joint_name = POLICY_JOINT_NAMES[joint_idx] if joint_idx < len(POLICY_JOINT_NAMES) else f"joint_{joint_idx}"
    return {
        f"{label}_abs_max": float(finite_abs[frame_idx, joint_idx]),
        f"{label}_abs_max_time": float(times[frame_idx]),
        f"{label}_abs_max_joint": joint_name,
    }


def _extract_metrics(trace_path: Path, *, height_drop_threshold: float, tilt_threshold_deg: float) -> dict[str, object]:
    with np.load(trace_path) as data:
        trace = {key: data[key] for key in data.files}

    times = np.asarray(trace["sim_time"], dtype=np.float64)
    root_pos = np.asarray(trace["root_pos"], dtype=np.float64)
    projected_gravity = np.asarray(trace["projected_gravity"], dtype=np.float64)
    foot_normal_forces = np.asarray(trace["foot_normal_forces"], dtype=np.float64)
    termination_contact = np.asarray(trace["termination_contact"], dtype=bool)
    feet_pos_w = np.asarray(trace["feet_pos_w"], dtype=np.float64) if "feet_pos_w" in trace else None
    termination_force = np.asarray(trace["termination_force"], dtype=np.float64) if "termination_force" in trace else None
    termination_body = np.asarray(trace["termination_body"]) if "termination_body" in trace else None

    root_z = root_pos[:, 2]
    tilt_deg = _tilt_deg_from_projected_gravity(projected_gravity)
    total_foot_force = np.sum(foot_normal_forces, axis=1)

    start_root_z = float(root_z[0])
    root_drop_idx = _first_index(root_z <= start_root_z - height_drop_threshold)
    tilt_20_idx = _first_index(tilt_deg >= tilt_threshold_deg)
    tilt_45_idx = _first_index(tilt_deg >= 45.0)
    termination_idx = _first_index(termination_contact)

    metrics: dict[str, object] = {
        "duration": float(times[-1]),
        "root_z_start": float(root_z[0]),
        "root_z_end": float(root_z[-1]),
        "root_z_min": float(np.min(root_z)),
        "root_drop_time": None if root_drop_idx is None else float(times[root_drop_idx]),
        "tilt_20_time": None if tilt_20_idx is None else float(times[tilt_20_idx]),
        "tilt_45_time": None if tilt_45_idx is None else float(times[tilt_45_idx]),
        "termination_contact_time": None if termination_idx is None else float(times[termination_idx]),
        "termination_force": None if termination_idx is None or termination_force is None else float(termination_force[termination_idx]),
        "termination_body": None if termination_idx is None or termination_body is None else str(termination_body[termination_idx]),
        "foot_force_total_start": float(total_foot_force[0]),
        "foot_force_total_end": float(total_foot_force[-1]),
        "foot_force_total_min": float(np.min(total_foot_force)),
    }
    metrics.update(
        _joint_signal_metrics(
            trace,
            trace_key="joint_applied_torque_policy",
            label="applied_torque",
            times=times,
        )
    )
    metrics.update(
        _joint_signal_metrics(
            trace,
            trace_key="joint_computed_torque_policy",
            label="computed_torque",
            times=times,
        )
    )
    if feet_pos_w is not None:
        foot_z = feet_pos_w[:, :, 2]
        metrics["feet_z_start"] = foot_z[0].astype(np.float64).tolist()
        metrics["feet_z_end"] = foot_z[-1].astype(np.float64).tolist()
        metrics["feet_z_min"] = np.min(foot_z, axis=0).astype(np.float64).tolist()
    return metrics


def _ranking_value(event_time: object, duration: float) -> float:
    if event_time is None or event_time == "":
        return duration + 1.0
    return float(event_time)


def _stringify_metric(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return ",".join(f"{float(item):.6f}" for item in value)
    return str(value)


def _rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def sort_key(row: dict[str, object]) -> tuple[float, float, float, float]:
        duration = float(row["duration"])
        return (
            _ranking_value(row.get("termination_contact_time"), duration),
            _ranking_value(row.get("root_drop_time"), duration),
            _ranking_value(row.get("tilt_20_time"), duration),
            _ranking_value(row.get("tilt_45_time"), duration),
        )

    return sorted(rows, key=sort_key, reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and rank Isaac free-base standing diagnostics across pose targets.")
    parser.add_argument("--run-dir", required=True, help="Directory where traces, logs, and summaries are written.")
    parser.add_argument("--task", default="walk_real_lite")
    parser.add_argument("--prefix", default="isaac_standing")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--settle-time", type=float, default=0.6)
    parser.add_argument("--height-drop-threshold", type=float, default=0.05)
    parser.add_argument("--tilt-threshold-deg", type=float, default=20.0)
    parser.add_argument("--root-zs", nargs="+", type=float, default=None)
    parser.add_argument("--hip-pitch-targets", nargs="+", type=float, default=[-0.5])
    parser.add_argument("--knee-pitch-targets", nargs="+", type=float, default=[1.0])
    parser.add_argument("--ankle-pitch-targets", nargs="+", type=float, default=[-0.5])
    parser.add_argument("--hip-pitch-kp-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--hip-pitch-kd-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--knee-pitch-kp-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--knee-pitch-kd-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--ankle-pitch-kp-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--ankle-pitch-kd-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--ankle-roll-kp-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--ankle-roll-kd-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--continue-after-termination", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    field_values = {
        "root_z": [None] if args.root_zs is None else args.root_zs,
        "hip_pitch_target": args.hip_pitch_targets,
        "knee_pitch_target": args.knee_pitch_targets,
        "ankle_pitch_target": args.ankle_pitch_targets,
        "hip_pitch_kp_scale": args.hip_pitch_kp_scales,
        "hip_pitch_kd_scale": args.hip_pitch_kd_scales,
        "knee_pitch_kp_scale": args.knee_pitch_kp_scales,
        "knee_pitch_kd_scale": args.knee_pitch_kd_scales,
        "ankle_pitch_kp_scale": args.ankle_pitch_kp_scales,
        "ankle_pitch_kd_scale": args.ankle_pitch_kd_scales,
        "ankle_roll_kp_scale": args.ankle_roll_kp_scales,
        "ankle_roll_kd_scale": args.ankle_roll_kd_scales,
    }
    varying_fields = tuple(field for field in FIELD_ORDER if len(field_values[field]) > 1)
    combos = [SweepConfig(*values) for values in itertools.product(*(field_values[field] for field in FIELD_ORDER))]

    rows: list[dict[str, object]] = []
    for index, config in enumerate(combos, start=1):
        label = _build_variant_label(config, varying_fields)
        tag = f"{args.prefix}_{index:02d}_{label}"
        trace_path = run_dir / f"{tag}.npz"
        log_path = run_dir / f"{tag}.log"

        command = [
            sys.executable,
            str(ROOT / "tools" / "isaac_standing_diagnostic.py"),
            "--task",
            args.task,
            "--headless",
            "--duration",
            f"{args.duration:g}",
            "--settle_time",
            f"{args.settle_time:g}",
            "--trace_out",
            str(trace_path),
        ]
        if config.root_z is not None:
            command.extend(["--root_z", f"{config.root_z:g}"])
        command.extend(
            [
            "--hip_pitch_target",
            f"{config.hip_pitch_target:g}",
            "--knee_pitch_target",
            f"{config.knee_pitch_target:g}",
            "--ankle_pitch_target",
            f"{config.ankle_pitch_target:g}",
            "--hip_pitch_kp_scale",
            f"{config.hip_pitch_kp_scale:g}",
            "--hip_pitch_kd_scale",
            f"{config.hip_pitch_kd_scale:g}",
            "--knee_pitch_kp_scale",
            f"{config.knee_pitch_kp_scale:g}",
            "--knee_pitch_kd_scale",
            f"{config.knee_pitch_kd_scale:g}",
            "--ankle_pitch_kp_scale",
            f"{config.ankle_pitch_kp_scale:g}",
            "--ankle_pitch_kd_scale",
            f"{config.ankle_pitch_kd_scale:g}",
            "--ankle_roll_kp_scale",
            f"{config.ankle_roll_kp_scale:g}",
            "--ankle_roll_kd_scale",
            f"{config.ankle_roll_kd_scale:g}",
            ]
        )
        if args.continue_after_termination:
            command.append("--continue_after_termination")

        print(f"[INFO] [{index}/{len(combos)}] {tag}")
        print(f"[INFO] Command: {' '.join(command)}")
        if args.dry_run:
            continue

        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"{tag} failed with exit code {completed.returncode}. See {log_path}")

        metrics = _extract_metrics(
            trace_path,
            height_drop_threshold=args.height_drop_threshold,
            tilt_threshold_deg=args.tilt_threshold_deg,
        )
        row: dict[str, object] = {
            "tag": tag,
            "trace_path": str(trace_path),
            "log_path": str(log_path),
        }
        for field in FIELD_ORDER:
            row[field] = getattr(config, field)
        row.update(metrics)
        rows.append(row)

    if args.dry_run:
        return

    ranked_rows = _rank_rows(rows)
    summary_path = run_dir / f"{args.prefix}_summary.csv"
    summary_fields = (
        "tag",
        *FIELD_ORDER,
        "termination_contact_time",
        "termination_force",
        "termination_body",
        "root_drop_time",
        "tilt_20_time",
        "tilt_45_time",
        "root_z_start",
        "root_z_end",
        "root_z_min",
        "foot_force_total_start",
        "foot_force_total_end",
        "foot_force_total_min",
        "applied_torque_abs_max",
        "applied_torque_abs_max_time",
        "applied_torque_abs_max_joint",
        "computed_torque_abs_max",
        "computed_torque_abs_max_time",
        "computed_torque_abs_max_joint",
        "feet_z_start",
        "feet_z_end",
        "feet_z_min",
        "duration",
        "trace_path",
        "log_path",
    )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow({field: _stringify_metric(row.get(field)) for field in summary_fields})

    print(f"[INFO] Wrote sweep summary: {summary_path}")
    print("[INFO] Ranked results:")
    for row in ranked_rows:
        print(
            "[INFO]   "
            f"{row['tag']}: "
            f"termination={_stringify_metric(row.get('termination_contact_time')) or 'not_reached'}s, "
            f"drop={_stringify_metric(row.get('root_drop_time')) or 'not_reached'}s, "
            f"tilt20={_stringify_metric(row.get('tilt_20_time')) or 'not_reached'}s, "
            f"tilt45={_stringify_metric(row.get('tilt_45_time')) or 'not_reached'}s, "
            f"applied_tau_max={_stringify_metric(row.get('applied_torque_abs_max')) or 'n/a'}"
            f"@{row.get('applied_torque_abs_max_joint') or 'n/a'}"
        )


if __name__ == "__main__":
    main()
