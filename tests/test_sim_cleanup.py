from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_CLEANUP_PATH = REPO_ROOT / "real_lite_lab" / "sim_cleanup.py"


def load_close_simulation_context():
    spec = importlib.util.spec_from_file_location("test_sim_cleanup_module", SIM_CLEANUP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SIM_CLEANUP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.close_simulation_context


close_simulation_context = load_close_simulation_context()


class DummySim:
    def __init__(self, has_gui: bool = False, fail_on: tuple[str, ...] = ()) -> None:
        self._has_gui = has_gui
        self.fail_on = set(fail_on)
        self.calls: list[str] = []

    def _record(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail_on:
            raise RuntimeError(f"{name} failed")

    def has_gui(self) -> bool:
        self.calls.append("has_gui")
        return self._has_gui

    def stop(self) -> None:
        self._record("stop")

    def clear_all_callbacks(self) -> None:
        self._record("clear_all_callbacks")

    def clear_instance(self) -> None:
        self._record("clear_instance")


class SimCleanupTests(unittest.TestCase):
    def test_headless_cleanup_stops_and_clears(self) -> None:
        sim = DummySim(has_gui=False)

        close_simulation_context(sim)

        self.assertEqual(sim.calls, ["has_gui", "stop", "clear_all_callbacks", "clear_instance"])

    def test_gui_cleanup_skips_stop(self) -> None:
        sim = DummySim(has_gui=True)

        close_simulation_context(sim)

        self.assertEqual(sim.calls, ["has_gui", "clear_all_callbacks", "clear_instance"])

    def test_cleanup_continues_when_hooks_fail(self) -> None:
        sim = DummySim(has_gui=False, fail_on=("stop", "clear_all_callbacks", "clear_instance"))

        close_simulation_context(sim)

        self.assertEqual(sim.calls, ["has_gui", "stop", "clear_all_callbacks", "clear_instance"])


if __name__ == "__main__":
    unittest.main()
