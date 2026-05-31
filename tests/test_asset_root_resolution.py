from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_MODULE_PATHS = (
    REPO_ROOT / "real_lite_lab" / "assets" / "__init__.py",
    REPO_ROOT / "tools" / "generate_real_lite_mjcf.py",
)


def load_module_from_path(module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"test_module_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AssetRootResolutionTests(unittest.TestCase):
    def test_env_override_requires_meshes_and_urdf(self) -> None:
        for module_path in ASSET_MODULE_PATHS:
            with self.subTest(module=module_path.name):
                module = load_module_from_path(module_path)

                with tempfile.TemporaryDirectory() as temp_dir:
                    asset_root = Path(temp_dir)
                    (asset_root / "meshes").mkdir()
                    (asset_root / "urdf").mkdir()

                    with mock.patch.dict(os.environ, {module.ASSET_ROOT_ENV_VAR: str(asset_root)}, clear=False):
                        self.assertEqual(module.resolve_real_lite_asset_root(), asset_root.resolve())

                    broken_root = asset_root / "broken"
                    broken_root.mkdir()
                    with mock.patch.dict(os.environ, {module.ASSET_ROOT_ENV_VAR: str(broken_root)}, clear=False):
                        with self.assertRaises(FileNotFoundError) as exc_info:
                            module.resolve_real_lite_asset_root()

                    error_message = str(exc_info.exception)
                    self.assertIn("meshes", error_message)
                    self.assertIn("urdf", error_message)
                    self.assertIn(module.ASSET_ROOT_ENV_VAR, error_message)

    def test_auto_discovery_uses_first_valid_root(self) -> None:
        for module_path in ASSET_MODULE_PATHS:
            with self.subTest(module=module_path.name):
                module = load_module_from_path(module_path)

                with tempfile.TemporaryDirectory() as invalid_dir, tempfile.TemporaryDirectory() as valid_dir:
                    invalid_root = Path(invalid_dir)
                    valid_root = Path(valid_dir)
                    (valid_root / "meshes").mkdir()
                    (valid_root / "urdf").mkdir()

                    with mock.patch.dict(os.environ, {}, clear=False):
                        with mock.patch.object(
                            module,
                            "AUTO_DISCOVERED_ASSET_ROOTS",
                            (invalid_root.resolve(), valid_root.resolve()),
                        ):
                            self.assertEqual(module.resolve_real_lite_asset_root(), valid_root.resolve())

    def test_auto_discovery_error_lists_searched_roots(self) -> None:
        for module_path in ASSET_MODULE_PATHS:
            with self.subTest(module=module_path.name):
                module = load_module_from_path(module_path)

                with tempfile.TemporaryDirectory() as missing_dir:
                    missing_root = Path(missing_dir).resolve()
                    searched_root = missing_root / "candidate"

                    with mock.patch.dict(os.environ, {}, clear=False):
                        with mock.patch.object(module, "AUTO_DISCOVERED_ASSET_ROOTS", (searched_root,)):
                            with self.assertRaises(FileNotFoundError) as exc_info:
                                module.resolve_real_lite_asset_root()

                    error_message = str(exc_info.exception)
                    self.assertIn("default location", error_message)
                    self.assertIn(str(searched_root), error_message)
                    self.assertIn(module.ASSET_ROOT_ENV_VAR, error_message)


if __name__ == "__main__":
    unittest.main()
