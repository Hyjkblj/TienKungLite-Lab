from __future__ import annotations

import os
from pathlib import Path


ASSET_ROOT_ENV_VAR = "TIENKUNG_LITE_ASSET_ROOT"
ASSET_ROOT_DIRNAME = "x_humanoid_0430_newfeet_newbody_publish"
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REAL_LITE_ASSET_ROOT = WORKSPACE_ROOT / "lite_urdf_publish" / ASSET_ROOT_DIRNAME
FALLBACK_REAL_LITE_ASSET_ROOT = WORKSPACE_ROOT / ASSET_ROOT_DIRNAME


def _missing_required_entries(asset_root: Path) -> list[str]:
    missing_entries = []
    for required_dir in ("meshes", "urdf"):
        if not (asset_root / required_dir).is_dir():
            missing_entries.append(required_dir)
    return missing_entries


def _default_asset_roots() -> tuple[Path, ...]:
    candidates: list[Path] = []
    for candidate in (DEFAULT_REAL_LITE_ASSET_ROOT, FALLBACK_REAL_LITE_ASSET_ROOT):
        resolved_candidate = candidate.resolve()
        if resolved_candidate not in candidates:
            candidates.append(resolved_candidate)
    return tuple(candidates)


AUTO_DISCOVERED_ASSET_ROOTS = _default_asset_roots()


def resolve_real_lite_asset_root() -> Path:
    configured_path = os.getenv(ASSET_ROOT_ENV_VAR)
    if configured_path:
        asset_root = Path(configured_path).expanduser().resolve()
        if not asset_root.exists():
            raise FileNotFoundError(
                f"Real Lite assets not found at: {asset_root}\n"
                f"Set {ASSET_ROOT_ENV_VAR} to the external asset directory that contains 'meshes' and 'urdf'."
            )

        missing_entries = _missing_required_entries(asset_root)
        if missing_entries:
            raise FileNotFoundError(
                f"Real Lite asset root is missing required directories {missing_entries!r}: {asset_root}\n"
                f"Set {ASSET_ROOT_ENV_VAR} to the external asset directory that contains 'meshes' and 'urdf'."
            )
        return asset_root

    searched_roots = []
    for asset_root in AUTO_DISCOVERED_ASSET_ROOTS:
        searched_roots.append(asset_root)
        if asset_root.exists() and not _missing_required_entries(asset_root):
            return asset_root

    searched_root_lines = "\n".join(f"  - {candidate}" for candidate in searched_roots)
    raise FileNotFoundError(
        "Real Lite assets were not found in any default location.\n"
        f"Searched:\n{searched_root_lines}\n"
        f"Set {ASSET_ROOT_ENV_VAR} to the external asset directory that contains 'meshes' and 'urdf'."
    )


REAL_LITE_ASSET_DIR = str(DEFAULT_REAL_LITE_ASSET_ROOT)
