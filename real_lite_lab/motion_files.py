from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Iterable


MotionPath = str | PathLike[str]


def _iter_paths(motion_files: Iterable[MotionPath] | None) -> list[Path]:
    if motion_files is None:
        return []
    return [Path(motion_file) for motion_file in motion_files]


def validate_motion_files(
    *,
    task_name: str,
    display_motion_files: Iterable[MotionPath] | None = None,
    amp_motion_files: Iterable[MotionPath] | None = None,
) -> None:
    missing_entries: list[str] = []

    for label, motion_files in (
        ("display motion", display_motion_files),
        ("AMP expert motion", amp_motion_files),
    ):
        for motion_path in _iter_paths(motion_files):
            if not motion_path.is_file():
                missing_entries.append(f"{label}: {motion_path}")

    if not missing_entries:
        return

    formatted_entries = "\n".join(f"  - {entry}" for entry in missing_entries)
    raise FileNotFoundError(
        f"Task '{task_name}' is missing required motion dataset file(s):\n{formatted_entries}"
    )
