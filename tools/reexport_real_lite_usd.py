from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_lite_lab.assets import resolve_real_lite_asset_root
from real_lite_lab.runtime_paths import ensure_writable_isaaclab_tmp


def _log(message: str) -> None:
    print(message, flush=True)


def _detect_fixed_root_markers(physics_usd_path: Path) -> dict[str, bool]:
    if not physics_usd_path.is_file():
        return {"physics_usd_exists": False, "has_root_joint": False, "has_fixed_token": False}

    usd_bytes = physics_usd_path.read_bytes()
    return {
        "physics_usd_exists": True,
        "has_root_joint": b"root_joint" in usd_bytes,
        "has_fixed_token": b"Fixed" in usd_bytes,
    }


def _parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Re-export the Real Lite URDF as a free-base USD asset.")
    parser.add_argument("--asset_root", type=str, default=None, help="Root directory that contains meshes/ and urdf/.")
    parser.add_argument(
        "--urdf_path",
        type=str,
        default=None,
        help="Explicit URDF path. Defaults to <asset_root>/urdf/humanoid_publish.urdf.",
    )
    parser.add_argument(
        "--usd_subdir",
        type=str,
        default="humanoid_publish_free_base",
        help="Output directory name under <asset_root>/urdf/.",
    )
    parser.add_argument(
        "--usd_file_name",
        type=str,
        default=None,
        help="Output USD file name. Defaults to <usd_subdir>.usd.",
    )
    parser.add_argument(
        "--fix_base",
        action="store_true",
        help="Export a fixed-base asset. Omit this flag for the intended free-base export.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force USD re-conversion even if output files already exist.",
    )
    parser.add_argument(
        "--make_instanceable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to export the USD as an instanceable asset.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    asset_root = Path(args.asset_root).expanduser().resolve() if args.asset_root else resolve_real_lite_asset_root()
    urdf_path = (
        Path(args.urdf_path).expanduser().resolve() if args.urdf_path else asset_root / "urdf" / "humanoid_publish.urdf"
    )
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    usd_subdir = args.usd_subdir.strip().strip("/\\")
    if not usd_subdir:
        raise ValueError("--usd_subdir must not be empty.")

    usd_dir = (asset_root / "urdf" / usd_subdir).resolve()
    usd_file_name = args.usd_file_name or f"{usd_subdir}.usd"
    if "/" in usd_file_name or "\\" in usd_file_name:
        raise ValueError("--usd_file_name must be a file name, not a path.")
    usd_path = usd_dir / usd_file_name

    _log(f"[INFO] asset_root: {asset_root}")
    _log(f"[INFO] urdf_path: {urdf_path}")
    _log(f"[INFO] usd_dir: {usd_dir}")
    _log(f"[INFO] usd_path: {usd_path}")
    _log(f"[INFO] export_mode: {'fixed_base' if args.fix_base else 'free_base'}")

    ensure_writable_isaaclab_tmp(ROOT / "logs" / "_isaaclab_tmp")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        try:
            from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        except ImportError:
            from isaaclab.sim.converters.urdf_converter import UrdfConverter, UrdfConverterCfg

        joint_drive = UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        )
        cfg = UrdfConverterCfg(
            asset_path=str(urdf_path),
            usd_dir=str(usd_dir),
            usd_file_name=usd_file_name,
            force_usd_conversion=args.force,
            make_instanceable=args.make_instanceable,
            fix_base=args.fix_base,
            root_link_name=None,
            link_density=0.0,
            merge_fixed_joints=True,
            convert_mimic_joints_to_normal_joints=False,
            joint_drive=joint_drive,
            collider_type="convex_hull",
            self_collision=False,
            replace_cylinders_with_capsules=False,
            collision_from_visuals=False,
        )
        _log("[INFO] Starting URDF -> USD conversion...")
        UrdfConverter(cfg)
        _log("[INFO] URDF -> USD conversion returned.")

        physics_usd_candidates = [
            usd_path.parent / "configuration" / f"{usd_path.stem}_physics.usd",
            usd_path.parent / "configuration" / "humanoid_publish_physics.usd",
        ]
        physics_usd_path = next((candidate for candidate in physics_usd_candidates if candidate.is_file()), physics_usd_candidates[0])
        fixed_root_markers = _detect_fixed_root_markers(physics_usd_path)

        generated_files = []
        if usd_dir.is_dir():
            generated_files = sorted(str(path.relative_to(usd_dir)) for path in usd_dir.rglob("*") if path.is_file())

        rel_usd_path = usd_path.relative_to(asset_root)
        _log(f"[INFO] physics_usd_path: {physics_usd_path}")
        _log(f"[INFO] usd_exists: {usd_path.is_file()}")
        _log(f"[INFO] test_with_env: export TIENKUNG_LITE_USD_REL_PATH={rel_usd_path.as_posix()}")
        _log(
            "[INFO] physics_usd_markers: "
            f"root_joint={fixed_root_markers['has_root_joint']}, "
            f"Fixed={fixed_root_markers['has_fixed_token']}"
        )
        _log(f"[INFO] generated_files_count: {len(generated_files)}")
        for rel_path in generated_files[:20]:
            _log(f"[INFO] generated_file: {rel_path}")
        if len(generated_files) > 20:
            _log(f"[INFO] generated_file: ... ({len(generated_files) - 20} more)")
        if not args.fix_base and fixed_root_markers["has_root_joint"] and fixed_root_markers["has_fixed_token"]:
            _log("[WARN] Generated USD still contains fixed-root markers; inspect the converter input/config before using it.")
        if not usd_path.is_file():
            _log("[WARN] Expected USD file was not created at the requested path.")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    sys.exit(exit_code)
