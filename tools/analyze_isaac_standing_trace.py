from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_lite_lab.constants import POLICY_JOINT_NAMES  # noqa: E402
from tools.isaac_standing_diagnostic import evaluate_standing_stability, summarize_standing_trace  # noqa: E402


def _load_trace(trace_path: Path) -> dict[str, np.ndarray]:
    with np.load(trace_path) as data:
        return {key: data[key] for key in data.files}


def _print_trace_summary(
    trace_path: Path,
    *,
    height_drop_threshold: float,
    tilt_threshold_deg: float,
    support_force_threshold: float,
    support_hold_steps: int,
) -> None:
    trace = _load_trace(trace_path)
    print(f"[INFO] Trace: {trace_path}")
    for line in summarize_standing_trace(
        trace,
        height_drop_threshold=height_drop_threshold,
        tilt_threshold_deg=tilt_threshold_deg,
        support_force_threshold=support_force_threshold,
        support_hold_steps=support_hold_steps,
        joint_names=POLICY_JOINT_NAMES,
    ):
        print(line)

    failures = evaluate_standing_stability(
        trace,
        height_drop_threshold=height_drop_threshold,
        tilt_threshold_deg=tilt_threshold_deg,
        support_force_threshold=support_force_threshold,
        support_hold_steps=support_hold_steps,
    )
    if failures:
        print("[INFO] Stability failures:")
        for failure in failures:
            print(f"[INFO]   {failure}")
    else:
        print("[INFO] Stability failures: none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved Isaac standing diagnostic .npz traces without rerunning Isaac.")
    parser.add_argument("traces", nargs="+", help="One or more Isaac standing trace .npz files.")
    parser.add_argument("--height-drop-threshold", type=float, default=0.05)
    parser.add_argument("--tilt-threshold-deg", type=float, default=20.0)
    parser.add_argument("--support-force-threshold", type=float, default=20.0)
    parser.add_argument("--support-hold-steps", type=int, default=3)
    args = parser.parse_args()

    for index, trace_text in enumerate(args.traces):
        if index:
            print("")
        trace_path = Path(trace_text).expanduser().resolve()
        _print_trace_summary(
            trace_path,
            height_drop_threshold=args.height_drop_threshold,
            tilt_threshold_deg=args.tilt_threshold_deg,
            support_force_threshold=args.support_force_threshold,
            support_hold_steps=args.support_hold_steps,
        )


if __name__ == "__main__":
    main()
