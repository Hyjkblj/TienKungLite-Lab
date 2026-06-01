from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_ARGS_PATH = REPO_ROOT / "real_lite_lab" / "cli_args.py"


def load_apply_headless_env_cfg_overrides():
    spec = importlib.util.spec_from_file_location("test_cli_args_module", CLI_ARGS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {CLI_ARGS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_headless_env_cfg_overrides


apply_headless_env_cfg_overrides = load_apply_headless_env_cfg_overrides()


class HeadlessOverrideTests(unittest.TestCase):
    def test_disables_command_debug_vis_in_headless_mode(self) -> None:
        env_cfg = SimpleNamespace(commands=SimpleNamespace(debug_vis=True))

        result = apply_headless_env_cfg_overrides(env_cfg, headless=True)

        self.assertIs(result, env_cfg)
        self.assertFalse(env_cfg.commands.debug_vis)

    def test_keeps_command_debug_vis_when_not_headless(self) -> None:
        env_cfg = SimpleNamespace(commands=SimpleNamespace(debug_vis=True))

        apply_headless_env_cfg_overrides(env_cfg, headless=False)

        self.assertTrue(env_cfg.commands.debug_vis)

    def test_handles_configs_without_commands(self) -> None:
        env_cfg = SimpleNamespace()

        result = apply_headless_env_cfg_overrides(env_cfg, headless=True)

        self.assertIs(result, env_cfg)


if __name__ == "__main__":
    unittest.main()
