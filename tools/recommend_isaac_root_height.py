from __future__ import annotations

import argparse
import csv
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RootHeightRecommendation:
    trace_path: Path
    root_z_start: float
    feet_z_start_min: float
    feet_z_start_mean: float
    feet_z_min: float
    suggested_root_z: float
    root_z_delta: float


def _trace_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            paths.extend(sorted(candidate for candidate in path.glob("*.npz") if candidate.is_file()))
        else:
            matches = [Path(match) for match in sorted(glob.glob(item))] if any(char in item for char in "*?[]") else [path]
            paths.extend(candidate for candidate in matches if candidate.is_file())
    return [path.resolve() for path in paths]


def recommend_root_height(trace_path: Path, *, target_foot_z: float = 0.05) -> RootHeightRecommendation:
    with np.load(trace_path) as data:
        if "root_pos" not in data or "feet_pos_w" not in data:
            raise KeyError(f"{trace_path} must contain root_pos and feet_pos_w arrays.")
        root_pos = np.asarray(data["root_pos"], dtype=np.float64)
        feet_pos_w = np.asarray(data["feet_pos_w"], dtype=np.float64)

    if root_pos.ndim != 2 or root_pos.shape[1] < 3:
        raise ValueError(f"{trace_path} root_pos must have shape [T, >=3].")
    if feet_pos_w.ndim != 3 or feet_pos_w.shape[1] < 1 or feet_pos_w.shape[2] < 3:
        raise ValueError(f"{trace_path} feet_pos_w must have shape [T, feet, >=3].")

    root_z_start = float(root_pos[0, 2])
    feet_z = feet_pos_w[:, :, 2]
    feet_z_start = feet_z[0]
    feet_z_start_min = float(np.min(feet_z_start))
    feet_z_start_mean = float(np.mean(feet_z_start))
    feet_z_min = float(np.min(feet_z))
    root_z_delta = float(target_foot_z - feet_z_start_min)
    suggested_root_z = root_z_start + root_z_delta
    return RootHeightRecommendation(
        trace_path=trace_path,
        root_z_start=root_z_start,
        feet_z_start_min=feet_z_start_min,
        feet_z_start_mean=feet_z_start_mean,
        feet_z_min=feet_z_min,
        suggested_root_z=suggested_root_z,
        root_z_delta=root_z_delta,
    )


def _format_row(row: RootHeightRecommendation) -> dict[str, str]:
    return {
        "trace": str(row.trace_path),
        "root_z_start": f"{row.root_z_start:.6f}",
        "feet_z_start_min": f"{row.feet_z_start_min:.6f}",
        "feet_z_start_mean": f"{row.feet_z_start_mean:.6f}",
        "feet_z_min": f"{row.feet_z_min:.6f}",
        "suggested_root_z": f"{row.suggested_root_z:.6f}",
        "root_z_delta": f"{row.root_z_delta:+.6f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend Isaac root_z from standing traces by placing the lowest initial foot body near a target z."
    )
    parser.add_argument("traces", nargs="+", help="Trace .npz files, directories, or glob patterns.")
    parser.add_argument(
        "--target-foot-z",
        type=float,
        default=0.05,
        help="Desired initial foot body z in meters. For the current Real Lite USD, 0.05m matches grounded contact.",
    )
    parser.add_argument("--csv-out", default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    paths = _trace_paths(args.traces)
    if not paths:
        raise FileNotFoundError("No trace .npz files matched the provided inputs.")

    rows = [recommend_root_height(path, target_foot_z=args.target_foot_z) for path in paths]
    formatted_rows = [_format_row(row) for row in rows]
    fieldnames = tuple(formatted_rows[0].keys())

    if args.csv_out:
        csv_path = Path(args.csv_out).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(formatted_rows)
        print(f"[INFO] Wrote root height recommendations: {csv_path}")

    header = ",".join(fieldnames)
    print(header)
    for row in formatted_rows:
        print(",".join(row[field] for field in fieldnames))


if __name__ == "__main__":
    main()
