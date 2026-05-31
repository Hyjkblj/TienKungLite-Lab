from __future__ import annotations

import os
import tempfile
from pathlib import Path


RUNTIME_ROOT_ENV_VAR = "TIENKUNG_LITE_RUNTIME_ROOT"
TMP_ENV_VARS = ("TMPDIR", "TMP", "TEMP")


def _default_runtime_root() -> Path:
    configured_root = os.getenv(RUNTIME_ROOT_ENV_VAR)
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    if os.name == "nt":
        local_app_data = os.getenv("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data).expanduser().resolve() / "TienKungLiteLab"
        return (Path.home() / "AppData" / "Local" / "TienKungLiteLab").resolve()

    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser().resolve() / "tienkunglite_lab"
    return (Path.home() / ".cache" / "tienkunglite_lab").resolve()


def _is_directory_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_path = path / ".write_probe"
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
        return True
    except OSError:
        return False


def _apply_tmp_root(tmp_root: Path) -> Path:
    resolved_tmp_root = tmp_root.expanduser().resolve()
    (resolved_tmp_root / "isaaclab" / "logs").mkdir(parents=True, exist_ok=True)

    for env_var in TMP_ENV_VARS:
        os.environ[env_var] = str(resolved_tmp_root)

    # Update tempfile's cached value so later callers use the repo-local directory immediately.
    tempfile.tempdir = str(resolved_tmp_root)
    return resolved_tmp_root


def ensure_writable_isaaclab_tmp(tmp_root: str | os.PathLike[str] | None = None) -> Path:
    """Ensure Isaac Lab writes temporary logs into a writable user-scoped directory."""

    if tmp_root is not None:
        return _apply_tmp_root(Path(tmp_root))

    current_tmp_root = Path(tempfile.gettempdir()).resolve()
    current_log_root = current_tmp_root / "isaaclab" / "logs"
    if _is_directory_writable(current_log_root):
        return current_tmp_root

    fallback_tmp_root = _default_runtime_root() / "tmp"
    return _apply_tmp_root(fallback_tmp_root)
