from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
DEBUG_SIGNALS_PATH = REPO_ROOT / "debug_signals.py"


def load_debug_signals_module():
    spec = importlib.util.spec_from_file_location("test_debug_signals_module", DEBUG_SIGNALS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {DEBUG_SIGNALS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DebugSignalsTests(unittest.TestCase):
    def test_install_stack_dump_signal_registers_usr1_handler(self) -> None:
        module = load_debug_signals_module()
        signal_value = 10

        with mock.patch.object(module.faulthandler, "enable") as enable_mock, mock.patch.object(
            module.faulthandler, "register", create=True
        ) as register_mock, mock.patch.object(module.signal, "SIGUSR1", signal_value, create=True):
            result = module.install_stack_dump_signal()

        self.assertEqual(result, signal_value)
        enable_mock.assert_called_once()
        register_mock.assert_called_once_with(signal_value, file=mock.ANY, all_threads=True, chain=False)

    def test_install_stack_dump_signal_handles_missing_usr1(self) -> None:
        module = load_debug_signals_module()

        with mock.patch.object(module.faulthandler, "enable") as enable_mock, mock.patch.object(
            module.faulthandler, "register", create=True
        ) as register_mock, mock.patch.object(module.signal, "SIGUSR1", None, create=True):
            result = module.install_stack_dump_signal()

        self.assertIsNone(result)
        enable_mock.assert_called_once()
        register_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
