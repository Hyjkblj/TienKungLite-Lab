from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_lite_lab.assets import resolve_real_lite_asset_root  # noqa: E402
from tools.run_isaac_standing_sweep import _extract_metrics, _rank_rows, _stringify_metric  # noqa: E402


@dataclass(frozen=True)
class AssetVariant:
    name: str
    description: str
    align_args: tuple[str, ...] | None


ASSET_VARIANTS = {
    "baseline": AssetVariant(
        name="baseline",
        description="Current candidate URDF, re-exported into an isolated USD folder.",
        align_args=None,
    ),
    "reference_feet": AssetVariant(
        name="reference_feet",
        description="Only replace ankle-roll foot mesh collisions with reference primitive toe rails.",
        align_args=("--reference-feet-only",),
    ),
    "reference_feet_mass": AssetVariant(
        name="reference_feet_mass",
        description="Reference primitive feet plus reference link masses, without joint-limit or collision-topology sync.",
        align_args=(
            "--no-sync-collision-topology",
            "--no-sync-joint-limits",
            "--replace-ankle-roll-collisions-with-reference",
            "--sync-link-mass",
        ),
    ),
    "reference_feet_mass_zero_fixed": AssetVariant(
        name="reference_feet_mass_zero_fixed",
        description=(
            "Reference primitive feet plus reference link masses, then zero candidate-only fixed-link mass "
            "such as waist_link."
        ),
        align_args=(
            "--no-sync-collision-topology",
            "--no-sync-joint-limits",
            "--replace-ankle-roll-collisions-with-reference",
            "--sync-link-mass",
            "--zero-candidate-only-fixed-link-mass",
        ),
    ),
    "reference_aligned_mass_zero_fixed": AssetVariant(
        name="reference_aligned_mass_zero_fixed",
        description=(
            "Full reference-style primitive collision topology and joint limits, reference link masses, "
            "and zero candidate-only fixed-link mass."
        ),
        align_args=(
            "--replace-all-primitive-collisions-with-reference",
            "--sync-link-mass",
            "--zero-candidate-only-fixed-link-mass",
        ),
    ),
}


SUMMARY_FIELDS = (
    "variant",
    "description",
    "urdf_path",
    "usd_rel_path",
    "align_log_path",
    "export_log_path",
    "diagnostic_log_path",
    "termination_contact_time",
    "termination_force",
    "termination_body",
    "root_drop_time",
    "tilt_20_time",
    "tilt_45_time",
    "root_z_start",
    "root_z_end",
    "root_z_min",
    "start_joint_speed_abs_max",
    "start_joint_speed_abs_max_joint",
    "start_joint_pos_error_abs_max",
    "start_joint_pos_error_abs_max_joint",
    "start_applied_torque_abs_max",
    "start_applied_torque_abs_max_joint",
    "foot_force_total_start",
    "foot_force_total_end",
    "foot_force_total_min",
    "applied_torque_abs_max",
    "applied_torque_abs_max_time",
    "applied_torque_abs_max_joint",
    "computed_torque_abs_max",
    "computed_torque_abs_max_time",
    "computed_torque_abs_max_joint",
    "com_x_minus_feet_center_start",
    "com_x_minus_feet_center_end",
    "com_y_minus_feet_center_start",
    "com_y_minus_feet_center_end",
    "com_xy_minus_feet_center_norm_max",
    "com_xy_minus_feet_center_norm_max_time",
    "com_x_minus_feet_center_tilt20",
    "com_y_minus_feet_center_tilt20",
    "com_x_minus_feet_center_drop",
    "com_y_minus_feet_center_drop",
    "com_x_minus_feet_center_termination",
    "com_y_minus_feet_center_termination",
    "duration",
    "trace_path",
)


def _variant_urdf_path(asset_root: Path, variant_name: str) -> Path:
    return asset_root / "urdf" / f"humanoid_publish.{variant_name}.urdf"


def _variant_usd_subdir(prefix: str, variant_name: str) -> str:
    safe_prefix = prefix.strip().strip("/\\")
    if not safe_prefix:
        raise ValueError("--usd-subdir-prefix must not be empty.")
    return f"{safe_prefix}_{variant_name}"


def _build_align_command(
    variant: AssetVariant,
    *,
    candidate_urdf: Path,
    output_urdf: Path,
    reference_asset_root: Path | None,
    reference_urdf: Path | None,
) -> list[str] | None:
    if variant.align_args is None:
        return None

    command = [
        sys.executable,
        str(ROOT / "tools" / "align_real_lite_urdf_to_reference.py"),
        "--candidate-urdf",
        str(candidate_urdf),
        "--output-urdf",
        str(output_urdf),
    ]
    if reference_asset_root is not None:
        command.extend(["--reference-asset-root", str(reference_asset_root)])
    if reference_urdf is not None:
        command.extend(["--reference-urdf", str(reference_urdf)])
    command.extend(variant.align_args)
    return command


def _build_export_command(
    *,
    urdf_path: Path,
    usd_subdir: str,
    usd_file_name: str,
    headless: bool,
    force: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "tools" / "reexport_real_lite_usd.py"),
        "--urdf_path",
        str(urdf_path),
        "--usd_subdir",
        usd_subdir,
        "--usd_file_name",
        usd_file_name,
    ]
    if headless:
        command.append("--headless")
    if force:
        command.append("--force")
    return command


def _build_diagnostic_command(
    *,
    task: str,
    duration: float,
    settle_time: float,
    trace_path: Path,
    root_z: float | None,
    hip_pitch_target: float,
    knee_pitch_target: float,
    ankle_pitch_target: float,
    hip_pitch_kp_scale: float,
    hip_pitch_kd_scale: float,
    knee_pitch_kp_scale: float,
    knee_pitch_kd_scale: float,
    ankle_pitch_kp_scale: float,
    ankle_pitch_kd_scale: float,
    ankle_roll_kp_scale: float,
    ankle_roll_kd_scale: float,
    continue_after_termination: bool,
    headless: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "tools" / "isaac_standing_diagnostic.py"),
        "--task",
        task,
        "--duration",
        f"{duration:g}",
        "--settle_time",
        f"{settle_time:g}",
        "--trace_out",
        str(trace_path),
        "--hip_pitch_target",
        f"{hip_pitch_target:g}",
        "--knee_pitch_target",
        f"{knee_pitch_target:g}",
        "--ankle_pitch_target",
        f"{ankle_pitch_target:g}",
        "--hip_pitch_kp_scale",
        f"{hip_pitch_kp_scale:g}",
        "--hip_pitch_kd_scale",
        f"{hip_pitch_kd_scale:g}",
        "--knee_pitch_kp_scale",
        f"{knee_pitch_kp_scale:g}",
        "--knee_pitch_kd_scale",
        f"{knee_pitch_kd_scale:g}",
        "--ankle_pitch_kp_scale",
        f"{ankle_pitch_kp_scale:g}",
        "--ankle_pitch_kd_scale",
        f"{ankle_pitch_kd_scale:g}",
        "--ankle_roll_kp_scale",
        f"{ankle_roll_kp_scale:g}",
        "--ankle_roll_kd_scale",
        f"{ankle_roll_kd_scale:g}",
    ]
    if root_z is not None:
        command.extend(["--root_z", f"{root_z:g}"])
    if continue_after_termination:
        command.append("--continue_after_termination")
    if headless:
        command.append("--headless")
    return command


def _run_command(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Command: {' '.join(command)}")
    print(f"[INFO] Log: {log_path}")
    if dry_run:
        return

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}. See {log_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate multiple URDF/USD asset variants and rank them with the same Isaac free-base "
            "standing diagnostic pose."
        )
    )
    parser.add_argument("--run-dir", required=False, default=None, help="Directory for logs, traces, and summary CSV.")
    parser.add_argument("--task", default="walk_real_lite")
    parser.add_argument("--asset-root", default=None, help="Asset root containing meshes/ and urdf/.")
    parser.add_argument("--candidate-urdf", default=None, help="Defaults to <asset-root>/urdf/humanoid_publish.urdf.")
    parser.add_argument("--reference-asset-root", default=None, help="Optional reference TienKung-Lab asset root.")
    parser.add_argument("--reference-urdf", default=None, help="Optional explicit reference URDF path.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(ASSET_VARIANTS.keys()),
        choices=tuple(ASSET_VARIANTS.keys()),
        help="Asset variants to generate and test.",
    )
    parser.add_argument("--list-variants", action="store_true", help="Print available variant names and exit.")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--settle-time", type=float, default=0.0)
    parser.add_argument("--height-drop-threshold", type=float, default=0.05)
    parser.add_argument("--tilt-threshold-deg", type=float, default=20.0)
    parser.add_argument("--root-z", type=float, default=0.782)
    parser.add_argument("--hip-pitch-target", type=float, default=-0.55)
    parser.add_argument("--knee-pitch-target", type=float, default=1.0)
    parser.add_argument("--ankle-pitch-target", type=float, default=-0.50)
    parser.add_argument("--hip-pitch-kp-scale", type=float, default=1.0)
    parser.add_argument("--hip-pitch-kd-scale", type=float, default=1.0)
    parser.add_argument("--knee-pitch-kp-scale", type=float, default=1.0)
    parser.add_argument("--knee-pitch-kd-scale", type=float, default=1.0)
    parser.add_argument("--ankle-pitch-kp-scale", type=float, default=1.0)
    parser.add_argument("--ankle-pitch-kd-scale", type=float, default=3.0)
    parser.add_argument("--ankle-roll-kp-scale", type=float, default=1.0)
    parser.add_argument("--ankle-roll-kd-scale", type=float, default=1.0)
    parser.add_argument("--usd-subdir-prefix", default="humanoid_publish_asset_variant")
    parser.add_argument("--continue-after-termination", action="store_true")
    parser.add_argument("--no-headless", action="store_true", help="Do not pass --headless to Isaac tools.")
    parser.add_argument("--no-force-export", action="store_true", help="Do not pass --force to USD re-export.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running Isaac.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_variants:
        for variant in ASSET_VARIANTS.values():
            print(f"{variant.name}: {variant.description}")
        return

    run_dir = Path(args.run_dir or ROOT / "logs" / "standing" / "isaac_asset_variant_sweep").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    asset_root = Path(args.asset_root).expanduser().resolve() if args.asset_root else resolve_real_lite_asset_root()
    candidate_urdf = (
        Path(args.candidate_urdf).expanduser().resolve()
        if args.candidate_urdf
        else asset_root / "urdf" / "humanoid_publish.urdf"
    )
    reference_asset_root = Path(args.reference_asset_root).expanduser().resolve() if args.reference_asset_root else None
    reference_urdf = Path(args.reference_urdf).expanduser().resolve() if args.reference_urdf else None

    if not candidate_urdf.is_file():
        raise FileNotFoundError(f"Candidate URDF not found: {candidate_urdf}")

    env = os.environ.copy()
    env["TIENKUNG_LITE_ASSET_ROOT"] = str(asset_root)
    headless = not args.no_headless
    force_export = not args.no_force_export

    rows: list[dict[str, object]] = []
    for index, variant_name in enumerate(args.variants, start=1):
        variant = ASSET_VARIANTS[variant_name]
        print(f"[INFO] [{index}/{len(args.variants)}] asset variant: {variant.name}")
        print(f"[INFO] {variant.description}")

        variant_dir = run_dir / variant.name
        align_log_path = variant_dir / "align.log"
        export_log_path = variant_dir / "reexport_usd.log"
        diagnostic_log_path = variant_dir / "standing.log"
        trace_path = variant_dir / "standing_trace.npz"

        urdf_path = candidate_urdf if variant.align_args is None else _variant_urdf_path(asset_root, variant.name)
        align_command = _build_align_command(
            variant,
            candidate_urdf=candidate_urdf,
            output_urdf=urdf_path,
            reference_asset_root=reference_asset_root,
            reference_urdf=reference_urdf,
        )
        if align_command is not None:
            _run_command(align_command, log_path=align_log_path, env=env, dry_run=args.dry_run)

        usd_subdir = _variant_usd_subdir(args.usd_subdir_prefix, variant.name)
        usd_file_name = f"{usd_subdir}.usd"
        usd_rel_path = Path("urdf") / usd_subdir / usd_file_name
        export_command = _build_export_command(
            urdf_path=urdf_path,
            usd_subdir=usd_subdir,
            usd_file_name=usd_file_name,
            headless=headless,
            force=force_export,
        )
        _run_command(export_command, log_path=export_log_path, env=env, dry_run=args.dry_run)

        diagnostic_env = dict(env)
        diagnostic_env["TIENKUNG_LITE_USD_REL_PATH"] = usd_rel_path.as_posix()
        diagnostic_command = _build_diagnostic_command(
            task=args.task,
            duration=args.duration,
            settle_time=args.settle_time,
            trace_path=trace_path,
            root_z=args.root_z,
            hip_pitch_target=args.hip_pitch_target,
            knee_pitch_target=args.knee_pitch_target,
            ankle_pitch_target=args.ankle_pitch_target,
            hip_pitch_kp_scale=args.hip_pitch_kp_scale,
            hip_pitch_kd_scale=args.hip_pitch_kd_scale,
            knee_pitch_kp_scale=args.knee_pitch_kp_scale,
            knee_pitch_kd_scale=args.knee_pitch_kd_scale,
            ankle_pitch_kp_scale=args.ankle_pitch_kp_scale,
            ankle_pitch_kd_scale=args.ankle_pitch_kd_scale,
            ankle_roll_kp_scale=args.ankle_roll_kp_scale,
            ankle_roll_kd_scale=args.ankle_roll_kd_scale,
            continue_after_termination=args.continue_after_termination,
            headless=headless,
        )
        _run_command(diagnostic_command, log_path=diagnostic_log_path, env=diagnostic_env, dry_run=args.dry_run)
        if args.dry_run:
            continue

        metrics = _extract_metrics(
            trace_path,
            height_drop_threshold=args.height_drop_threshold,
            tilt_threshold_deg=args.tilt_threshold_deg,
        )
        row: dict[str, object] = {
            "variant": variant.name,
            "description": variant.description,
            "urdf_path": str(urdf_path),
            "usd_rel_path": usd_rel_path.as_posix(),
            "align_log_path": "" if align_command is None else str(align_log_path),
            "export_log_path": str(export_log_path),
            "diagnostic_log_path": str(diagnostic_log_path),
            "trace_path": str(trace_path),
        }
        row.update(metrics)
        rows.append(row)

    if args.dry_run:
        return

    ranked_rows = _rank_rows(rows)
    summary_path = run_dir / "isaac_asset_variant_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow({field: _stringify_metric(row.get(field)) for field in SUMMARY_FIELDS})

    print(f"[INFO] Wrote asset variant summary: {summary_path}")
    print("[INFO] Ranked asset variants:")
    for row in ranked_rows:
        print(
            "[INFO]   "
            f"{row['variant']}: "
            f"termination={_stringify_metric(row.get('termination_contact_time')) or 'not_reached'}s, "
            f"tilt20={_stringify_metric(row.get('tilt_20_time')) or 'not_reached'}s, "
            f"foot_start={_stringify_metric(row.get('foot_force_total_start')) or 'n/a'}, "
            f"start_speed={_stringify_metric(row.get('start_joint_speed_abs_max')) or 'n/a'}"
            f"@{row.get('start_joint_speed_abs_max_joint') or 'n/a'}, "
            f"start_tau={_stringify_metric(row.get('start_applied_torque_abs_max')) or 'n/a'}"
            f"@{row.get('start_applied_torque_abs_max_joint') or 'n/a'}, "
            f"com_tilt20_x={_stringify_metric(row.get('com_x_minus_feet_center_tilt20')) or 'n/a'}, "
            f"usd={row['usd_rel_path']}"
        )


if __name__ == "__main__":
    main()
