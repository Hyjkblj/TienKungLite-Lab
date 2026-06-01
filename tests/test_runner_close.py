from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RSL_RL_PACKAGE_ROOT = REPO_ROOT / "rsl_rl" / "rsl_rl"


def load_runner_classes():
    package_init = RSL_RL_PACKAGE_ROOT / "__init__.py"
    existing = sys.modules.get("rsl_rl")
    if getattr(existing, "__file__", None) != str(package_init):
        sys.modules.setdefault("git", types.ModuleType("git"))
        spec = importlib.util.spec_from_file_location(
            "rsl_rl",
            package_init,
            submodule_search_locations=[str(RSL_RL_PACKAGE_ROOT)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load package from {package_init}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["rsl_rl"] = module
        spec.loader.exec_module(module)

    from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

    return (OnPolicyRunner, AmpOnPolicyRunner)


class TrackingWriter:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def _record(self, name):
        self.calls.append(name)
        if name in self.fail_on:
            raise RuntimeError(f"{name} failed")

    def flush(self):
        self._record("flush")

    def close(self):
        self._record("close")

    def stop(self):
        self._record("stop")


class PartialWriter:
    def __init__(self):
        self.calls = []

    def close(self):
        self.calls.append("close")


class RunnerCloseTests(unittest.TestCase):
    def test_close_runs_all_writer_hooks_and_is_idempotent(self) -> None:
        for runner_class in load_runner_classes():
            with self.subTest(runner_class=runner_class.__name__):
                runner = runner_class.__new__(runner_class)
                writer = TrackingWriter()
                runner.writer = writer

                runner.close()
                runner.close()

                self.assertEqual(writer.calls, ["flush", "close", "stop"])
                self.assertIsNone(runner.writer)

    def test_close_continues_after_writer_hook_errors(self) -> None:
        for runner_class in load_runner_classes():
            with self.subTest(runner_class=runner_class.__name__):
                runner = runner_class.__new__(runner_class)
                writer = TrackingWriter(fail_on={"flush", "close", "stop"})
                runner.writer = writer

                runner.close()

                self.assertEqual(writer.calls, ["flush", "close", "stop"])
                self.assertIsNone(runner.writer)

    def test_close_handles_writers_with_partial_hooks(self) -> None:
        for runner_class in load_runner_classes():
            with self.subTest(runner_class=runner_class.__name__):
                runner = runner_class.__new__(runner_class)
                writer = PartialWriter()
                runner.writer = writer

                runner.close()

                self.assertEqual(writer.calls, ["close"])
                self.assertIsNone(runner.writer)


if __name__ == "__main__":
    unittest.main()
