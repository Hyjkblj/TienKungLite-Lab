from __future__ import annotations

from typing import Any


def _print_shutdown(message: str) -> None:
    print(f"[SHUTDOWN] {message}", flush=True)


def close_simulation_context(sim: Any) -> None:
    """Best-effort teardown for Isaac Lab simulation contexts."""

    if sim is None:
        _print_shutdown("Simulation context cleanup skipped: sim is None.")
        return

    _print_shutdown("Skipping SimulationContext.stop() to avoid shutdown deadlock.")

    for method_name in ("clear_all_callbacks", "clear_instance"):
        method = getattr(sim, method_name, None)
        if callable(method):
            _print_shutdown(f"Calling SimulationContext.{method_name}()")
            try:
                method()
                _print_shutdown(f"SimulationContext.{method_name}() completed.")
            except Exception:
                _print_shutdown(f"SimulationContext.{method_name}() raised and was ignored.")
                pass
