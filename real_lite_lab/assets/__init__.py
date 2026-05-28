from __future__ import annotations

import os
from pathlib import Path


ASSET_ROOT_ENV_VAR = "TIENKUNG_LITE_ASSET_ROOT"
DEFAULT_REAL_LITE_ASSET_ROOT = (
    Path(__file__).resolve().parents[3] / "lite_urdf_publish" / "x_humanoid_0430_newfeet_newbody_publish"
)


def resolve_real_lite_asset_root() -> Path:
    configured_path = os.getenv(ASSET_ROOT_ENV_VAR)
    asset_root = Path(configured_path).expanduser().resolve() if configured_path else DEFAULT_REAL_LITE_ASSET_ROOT
    if not asset_root.exists():
        raise FileNotFoundError(
            f"Real Lite assets not found at: {asset_root}\n"
            f"Set {ASSET_ROOT_ENV_VAR} to the external asset directory that contains 'meshes' and 'urdf'."
        )
    return asset_root


REAL_LITE_ASSET_DIR = str(DEFAULT_REAL_LITE_ASSET_ROOT)
