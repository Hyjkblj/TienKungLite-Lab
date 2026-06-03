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


@dataclass(frozen=True)
class SweepConfig:
    hip_pitch_target: float
    knee_pitch_target: float
    ankle_pitch_target: float


FIELD_ORDER = (
    "hip_pitch_target",
    "knee_pitch_target",
    "ankle_pitch_target",
)
FIELD_LABELS = {
    "hip_pitch_target": "hp",
    "knee_pitch_target": "kp",
    "ankle_pitch_target": "ap",
}


def _format_float_tag(value: float) -> str:
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


def _extract_metrics(trace_path: Path, *, height_drop_threshold: float, tilt_threshold_deg: float) -> dict[str, object]:
    with np.load(trace_path) as data:
        trace = {key: data[key] for key in data.files}

    times = np.asarray(trace["sim_time"], dtype=np.float64)
    root_pos = np.asarray(trace["root_pos"], dtype=np.float64)
    projected_gravity = np.asarray(trace["projected_gravity"], dtype=np.float64)
    foot_normal_forces = np.asarray(trace["foot_normal_forces"], dtype=np.float64)
    termination_contact = np.asarray(trace["termination_contact"], dtype=bool)
    feet_pos_w = np.asarray(trace["feet_pos_w"], dtype=np.float64) if "feet_pos_w" in trace else None

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
        "foot_force_total_start": float(total_foot_force[0]),
        "foot_force_total_end": float(total_foot_force[-1]),
        "foot_force_total_min": float(np.min(total_foot_force)),
    }
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
    parser.add_argument("--hip-pitch-targets", nargs="+", type=float, default=[-0.5])
    parser.add_argument("--knee-pitch-targets", nargs="+", type=float, default=[1.0])
    parser.add_argument("--ankle-pitch-targets", nargs="+", type=float, default=[-0.5])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    field_values = {
        "hip_pitch_target": args.hip_pitch_targets,
        "knee_pitch_target": args.knee_pitch_targets,
        "ankle_pitch_target": args.ankle_pitch_targets,
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
            "--hip_pitch_target",
            f"{config.hip_pitch_target:g}",
            "--knee_pitch_target",
            f"{config.knee_pitch_target:g}",
            "--ankle_pitch_target",
            f"{config.ankle_pitch_target:g}",
        ]

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
        "root_drop_time",
        "tilt_20_time",
        "tilt_45_time",
        "root_z_start",
        "root_z_end",
        "root_z_min",
        "foot_force_total_start",
        "foot_force_total_end",
        "foot_force_total_min",
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
            f"tilt45={_stringify_metric(row.get('tilt_45_time')) or 'not_reached'}s"
        )


if __name__ == "__main__":
    main()
