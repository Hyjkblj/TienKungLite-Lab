from __future__ import annotations

from pathlib import Path


FIXED_BASE_USD_REL_PATH = Path("urdf") / "humanoid_publish" / "humanoid_publish.usd"
FREE_BASE_USD_REL_PATH = Path("urdf") / "humanoid_publish_free_base" / "humanoid_publish_free_base.usd"


def resolve_physics_usd_path(usd_path: Path) -> Path:
    candidates = (
        usd_path.parent / "configuration" / f"{usd_path.stem}_physics.usd",
        usd_path.parent / "configuration" / "humanoid_publish_physics.usd",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def detect_fixed_root_markers(physics_usd_path: Path) -> dict[str, bool]:
    if not physics_usd_path.is_file():
        return {"exists": False, "has_root_joint": False, "has_fixed_token": False}

    usd_bytes = physics_usd_path.read_bytes()
    return {
        "exists": True,
        "has_root_joint": b"root_joint" in usd_bytes,
        "has_fixed_token": b"Fixed" in usd_bytes,
    }


def usd_has_fixed_root(usd_path: Path) -> bool:
    markers = detect_fixed_root_markers(resolve_physics_usd_path(usd_path))
    return markers["exists"] and markers["has_root_joint"] and markers["has_fixed_token"]


def validate_free_base_usd(usd_path: Path, *, allow_fixed_base: bool = False) -> None:
    if allow_fixed_base:
        return

    physics_usd_path = resolve_physics_usd_path(usd_path)
    markers = detect_fixed_root_markers(physics_usd_path)
    if markers["exists"] and markers["has_root_joint"] and markers["has_fixed_token"]:
        raise RuntimeError(
            "Real Lite USD appears to be fixed-base, but free-base standing/walking requires a floating root.\n"
            f"USD: {usd_path}\n"
            f"Physics USD: {physics_usd_path}\n"
            "Re-export a free-base USD with:\n"
            "  python tools/reexport_real_lite_usd.py --headless --force\n"
            "Then use it with:\n"
            f"  export TIENKUNG_LITE_USD_REL_PATH={FREE_BASE_USD_REL_PATH.as_posix()}\n"
            "If you intentionally need a fixed-base diagnostic, set TIENKUNG_LITE_ALLOW_FIXED_BASE_USD=1."
        )


def resolve_real_lite_usd_path(
    asset_dir: Path,
    *,
    configured_rel_path: str | None = None,
    allow_fixed_base: bool = False,
) -> Path:
    asset_dir = asset_dir.resolve()
    if configured_rel_path:
        usd_path = (asset_dir / Path(configured_rel_path)).resolve()
        if not usd_path.is_file():
            raise FileNotFoundError(
                f"Real Lite USD asset not found: {usd_path}\n"
                f"Resolved from TIENKUNG_LITE_USD_REL_PATH={configured_rel_path} under asset root: {asset_dir}"
            )
        validate_free_base_usd(usd_path, allow_fixed_base=allow_fixed_base)
        return usd_path

    candidates = (
        asset_dir / FREE_BASE_USD_REL_PATH,
        asset_dir / FIXED_BASE_USD_REL_PATH,
    )
    existing_candidates = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    for usd_path in existing_candidates:
        if allow_fixed_base or not usd_has_fixed_root(usd_path):
            return usd_path

    if existing_candidates:
        validate_free_base_usd(existing_candidates[0], allow_fixed_base=allow_fixed_base)

    searched = "\n".join(f"  - {candidate.resolve()}" for candidate in candidates)
    raise FileNotFoundError(
        "Real Lite free-base USD asset not found.\n"
        f"Searched:\n{searched}\n"
        "Generate it with:\n"
        "  python tools/reexport_real_lite_usd.py --headless --force\n"
        f"or set TIENKUNG_LITE_USD_REL_PATH={FREE_BASE_USD_REL_PATH.as_posix()}"
    )
