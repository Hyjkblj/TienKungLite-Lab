from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from real_lite_lab.usd_asset_validation import (
    FIXED_BASE_USD_REL_PATH,
    FREE_BASE_USD_REL_PATH,
    resolve_real_lite_usd_path,
)


def _write_usd_pair(asset_root: Path, rel_path: Path, *, fixed_root: bool) -> Path:
    usd_path = asset_root / rel_path
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    physics_path = usd_path.parent / "configuration" / f"{usd_path.stem}_physics.usd"
    physics_path.parent.mkdir(parents=True, exist_ok=True)
    physics_text = "root_joint Fixed" if fixed_root else "floating_root"
    physics_path.write_text(physics_text, encoding="utf-8")
    return usd_path


class UsdAssetValidationTests(unittest.TestCase):
    def test_prefers_free_base_usd_over_legacy_fixed_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            asset_root = Path(tmp_dir)
            free_usd = _write_usd_pair(asset_root, FREE_BASE_USD_REL_PATH, fixed_root=False)
            _write_usd_pair(asset_root, FIXED_BASE_USD_REL_PATH, fixed_root=True)

            self.assertEqual(resolve_real_lite_usd_path(asset_root), free_usd.resolve())

    def test_rejects_fixed_base_usd_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            asset_root = Path(tmp_dir)
            _write_usd_pair(asset_root, FIXED_BASE_USD_REL_PATH, fixed_root=True)

            with self.assertRaises(RuntimeError) as exc_info:
                resolve_real_lite_usd_path(asset_root)

            self.assertIn("fixed-base", str(exc_info.exception))
            self.assertIn("reexport_real_lite_usd.py", str(exc_info.exception))

    def test_allows_fixed_base_when_explicitly_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            asset_root = Path(tmp_dir)
            fixed_usd = _write_usd_pair(asset_root, FIXED_BASE_USD_REL_PATH, fixed_root=True)

            resolved = resolve_real_lite_usd_path(asset_root, allow_fixed_base=True)

            self.assertEqual(resolved, fixed_usd.resolve())


if __name__ == "__main__":
    unittest.main()
