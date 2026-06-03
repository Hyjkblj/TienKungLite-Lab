from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import analyze_sim2sim_trace as trace_tools


@dataclass(frozen=True)
class SweepConfig:
    hip_pitch_target: float
    knee_pitch_target: float
    ankle_pitch_target: float
    knee_pitch_kv_scale: float
    ankle_pitch_kp_scale: float
    ankle_pitch_kv_scale: float
    ankle_roll_kp_scale: float
    ankle_roll_kv_scale: float


FIELD_ORDER = (
    "hip_pitch_target",
    "knee_pitch_target",
    "ankle_pitch_target",
    "knee_pitch_kv_scale",
    "ankle_pitch_kp_scale",
    "ankle_pitch_kv_scale",
    "ankle_roll_kp_scale",
    "ankle_roll_kv_scale",
)
FIELD_LABELS = {
    "hip_pitch_target": "hp",
    "knee_pitch_target": "kp",
    "ankle_pitch_target": "ap",
    "knee_pitch_kv_scale": "kkv",
    "ankle_pitch_kp_scale": "apkp",
    "ankle_pitch_kv_scale": "apkv",
    "ankle_roll_kp_scale": "arkp",
    "ankle_roll_kv_scale": "arkv",
}


def _format_float_tag(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _build_variant_label(config: SweepConfig, varying_fields: tuple[str, ...]) -> str:
    if not varying_fields:
        return "base"
    return "__".join(f"{FIELD_LABELS[field]}_{_format_float_tag(getattr(config, field))}" for field in varying_fields)


def _ranking_value(event_time: float | None, duration: float) -> float:
    if event_time is None:
        return duration + 1.0
    return float(event_time)


def rank_sweep_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def sort_key(row: dict[str, object]) -> tuple[float, float, float, float]:
        duration = float(row["duration"])
        return (
            _ranking_value(row["support_loss_time"], duration),
            _ranking_value(row["loaded_single_support_time"], duration),
            _ranking_value(row["tilt_20_time"], duration),
            _ranking_value(row["tilt_45_time"], duration),
        )

    return sorted(rows, key=sort_key, reverse=True)


def _write_analysis_file(trace_path: Path, analysis_path: Path, args) -> dict[str, object]:
    lines = trace_tools.analyze_trace(
        trace_path,
        height_drop_threshold=args.height_drop_threshold,
        tilt_threshold_deg=args.tilt_threshold_deg,
        support_force_threshold=args.support_force_threshold,
        support_hold_steps=args.support_hold_steps,
    )
    analysis_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    metrics = trace_tools.extract_trace_metrics(
        trace_path,
        height_drop_threshold=args.height_drop_threshold,
        tilt_threshold_deg=args.tilt_threshold_deg,
        support_force_threshold=args.support_force_threshold,
        support_hold_steps=args.support_hold_steps,
    )
    return metrics


def _stringify_metric(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return ",".join(f"{float(item):.6f}" for item in value)
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and rank a standing hold sweep for sim2sim_real_lite.py.")
    parser.add_argument("--run-dir", required=True, help="Directory where traces, logs, videos, and summaries are written.")
    parser.add_argument("--task", default="walk_real_lite", help="Task name passed to sim2sim_real_lite.py.")
    parser.add_argument("--prefix", default="hold_sweep", help="Filename prefix for generated artifacts.")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--trace-steps", type=int, default=301)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--camera", default="follow_side")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--height-drop-threshold", type=float, default=0.05)
    parser.add_argument("--tilt-threshold-deg", type=float, default=20.0)
    parser.add_argument("--support-force-threshold", type=float, default=20.0)
    parser.add_argument("--support-hold-steps", type=int, default=3)
    parser.add_argument("--hip-pitch-targets", nargs="+", type=float, default=[-0.4925])
    parser.add_argument("--knee-pitch-targets", nargs="+", type=float, default=[0.9850])
    parser.add_argument("--ankle-pitch-targets", nargs="+", type=float, default=[-0.4700])
    parser.add_argument("--knee-pitch-kv-scales", nargs="+", type=float, default=[1.5])
    parser.add_argument("--ankle-pitch-kp-scales", nargs="+", type=float, default=[2.5])
    parser.add_argument("--ankle-pitch-kv-scales", nargs="+", type=float, default=[2.5])
    parser.add_argument("--ankle-roll-kp-scales", nargs="+", type=float, default=[1.2])
    parser.add_argument("--ankle-roll-kv-scales", nargs="+", type=float, default=[1.2])
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    field_values = {
        "hip_pitch_target": args.hip_pitch_targets,
        "knee_pitch_target": args.knee_pitch_targets,
        "ankle_pitch_target": args.ankle_pitch_targets,
        "knee_pitch_kv_scale": args.knee_pitch_kv_scales,
        "ankle_pitch_kp_scale": args.ankle_pitch_kp_scales,
        "ankle_pitch_kv_scale": args.ankle_pitch_kv_scales,
        "ankle_roll_kp_scale": args.ankle_roll_kp_scales,
        "ankle_roll_kv_scale": args.ankle_roll_kv_scales,
    }
    varying_fields = tuple(field for field in FIELD_ORDER if len(field_values[field]) > 1)

    combos = [
        SweepConfig(*values)
        for values in itertools.product(*(field_values[field] for field in FIELD_ORDER))
    ]

    rows: list[dict[str, object]] = []
    for index, config in enumerate(combos, start=1):
        label = _build_variant_label(config, varying_fields)
        tag = f"{args.prefix}_{index:02d}_{label}"
        trace_path = run_dir / f"{tag}.npz"
        video_path = run_dir / f"{tag}.mp4"
        log_path = run_dir / f"{tag}.log"
        analysis_path = run_dir / f"{tag}_analysis.txt"

        command = [
            sys.executable,
            str(ROOT / "sim2sim_real_lite.py"),
            "--task",
            args.task,
            "--control_mode",
            "hold",
            "--duration",
            f"{args.duration:g}",
            "--trace_out",
            str(trace_path),
            "--trace_steps",
            str(args.trace_steps),
            "--save_video",
            str(video_path),
            "--camera",
            args.camera,
            "--fps",
            f"{args.fps:g}",
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--settle_steps",
            str(args.settle_steps),
            "--hip_pitch_target",
            f"{config.hip_pitch_target:g}",
            "--knee_pitch_target",
            f"{config.knee_pitch_target:g}",
            "--ankle_pitch_target",
            f"{config.ankle_pitch_target:g}",
            "--knee_pitch_kv_scale",
            f"{config.knee_pitch_kv_scale:g}",
            "--ankle_pitch_kp_scale",
            f"{config.ankle_pitch_kp_scale:g}",
            "--ankle_pitch_kv_scale",
            f"{config.ankle_pitch_kv_scale:g}",
            "--ankle_roll_kp_scale",
            f"{config.ankle_roll_kp_scale:g}",
            "--ankle_roll_kv_scale",
            f"{config.ankle_roll_kv_scale:g}",
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

        metrics = _write_analysis_file(trace_path, analysis_path, args)
        row: dict[str, object] = {
            "tag": tag,
            "trace_path": str(trace_path),
            "analysis_path": str(analysis_path),
            "log_path": str(log_path),
        }
        for field in FIELD_ORDER:
            row[field] = getattr(config, field)
        row.update(metrics)
        rows.append(row)

    if args.dry_run:
        return

    ranked_rows = rank_sweep_results(rows)
    summary_path = run_dir / f"{args.prefix}_summary.csv"
    summary_fields = (
        "tag",
        *FIELD_ORDER,
        "support_loss_time",
        "loaded_single_support_time",
        "tilt_20_time",
        "tilt_45_time",
        "root_drop_time",
        "support_margin_min",
        "support_offset_xy_at_support_loss",
        "support_offset_xy_start",
        "duration",
        "trace_path",
        "analysis_path",
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
            f"support_loss={_stringify_metric(row.get('support_loss_time')) or 'not_reached'}s, "
            f"loaded_single_support={_stringify_metric(row.get('loaded_single_support_time')) or 'not_reached'}s, "
            f"tilt20={_stringify_metric(row.get('tilt_20_time')) or 'not_reached'}s, "
            f"tilt45={_stringify_metric(row.get('tilt_45_time')) or 'not_reached'}s, "
            f"offset_at_loss={_stringify_metric(row.get('support_offset_xy_at_support_loss')) or 'n/a'}"
        )


if __name__ == "__main__":
    main()
