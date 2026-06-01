from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MOTION_FILES_PATH = REPO_ROOT / "real_lite_lab" / "motion_files.py"


def load_motion_file_helpers():
    spec = importlib.util.spec_from_file_location("test_motion_files_module", MOTION_FILES_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MOTION_FILES_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_motion_files


validate_motion_files = load_motion_file_helpers()


class MotionFileValidationTests(unittest.TestCase):
    def test_validate_motion_files_accepts_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            display_file = tmp_path / "display.json"
            amp_file = tmp_path / "amp.json"
            display_file.write_text("{}", encoding="utf-8")
            amp_file.write_text("{}", encoding="utf-8")

            validate_motion_files(
                task_name="walk_real_lite",
                display_motion_files=[display_file],
                amp_motion_files=[amp_file],
            )

    def test_validate_motion_files_reports_all_missing_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            missing_display = tmp_path / "upper_body_display.json"
            missing_amp = tmp_path / "upper_body_amp.json"

            with self.assertRaises(FileNotFoundError) as exc_info:
                validate_motion_files(
                    task_name="upper_body_real_lite",
                    display_motion_files=[missing_display],
                    amp_motion_files=[missing_amp],
                )

        message = str(exc_info.exception)
        self.assertIn("upper_body_real_lite", message)
        self.assertIn(str(missing_display), message)
        self.assertIn(str(missing_amp), message)
        self.assertIn("display motion", message)
        self.assertIn("AMP expert motion", message)


if __name__ == "__main__":
    unittest.main()
