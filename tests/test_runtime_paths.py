from __future__ import annotations

import os
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATHS_MODULE_PATH = REPO_ROOT / "real_lite_lab" / "runtime_paths.py"


def load_runtime_paths_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_runtime_paths_module", RUNTIME_PATHS_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {RUNTIME_PATHS_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimePathTests(unittest.TestCase):
    def test_explicit_tmp_root_stays_inside_repo(self) -> None:
        module = load_runtime_paths_module()
        original_tempdir = tempfile.tempdir
        original_env = {env_var: os.environ.get(env_var) for env_var in module.TMP_ENV_VARS}

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_tmp_root = Path(temp_dir) / "logs" / "_isaaclab_tmp"

                with mock.patch.dict(os.environ, {}, clear=False):
                    resolved_tmp_root = module.ensure_writable_isaaclab_tmp(repo_tmp_root)

                    self.assertEqual(resolved_tmp_root, repo_tmp_root.resolve())
                    self.assertTrue((resolved_tmp_root / "isaaclab" / "logs").is_dir())
                    for env_var in module.TMP_ENV_VARS:
                        self.assertEqual(os.environ[env_var], str(resolved_tmp_root))
        finally:
            tempfile.tempdir = original_tempdir
            for env_var, value in original_env.items():
                if value is None:
                    os.environ.pop(env_var, None)
                else:
                    os.environ[env_var] = value


if __name__ == "__main__":
    unittest.main()
