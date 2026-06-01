from __future__ import annotations

from typing import Any


def close_simulation_context(sim: Any) -> None:
    """Best-effort teardown for Isaac Lab simulation contexts."""

    if sim is None:
        return

    has_gui = getattr(sim, "has_gui", None)
    stop = getattr(sim, "stop", None)
    if callable(stop):
        try:
            if not callable(has_gui) or not has_gui():
                stop()
        except Exception:
            pass

    for method_name in ("clear_all_callbacks", "clear_instance"):
        method = getattr(sim, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
