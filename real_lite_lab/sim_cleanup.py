from __future__ import annotations

from typing import Any


def _print_shutdown(message: str) -> None:
    print(f"[SHUTDOWN] {message}", flush=True)


def close_simulation_context(sim: Any) -> None:
    """Best-effort teardown for Isaac Lab simulation contexts."""

    if sim is None:
        _print_shutdown("Simulation context cleanup skipped: sim is None.")
        return

    has_gui = getattr(sim, "has_gui", None)
    stop = getattr(sim, "stop", None)
    if callable(stop):
        _print_shutdown("Inspecting SimulationContext.has_gui() before stop().")
        try:
            if not callable(has_gui) or not has_gui():
                _print_shutdown("Calling SimulationContext.stop()")
                stop()
                _print_shutdown("SimulationContext.stop() completed.")
            else:
                _print_shutdown("Skipping SimulationContext.stop() because GUI is active.")
        except Exception:
            _print_shutdown("SimulationContext.stop() raised and was ignored.")
            pass

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
