from __future__ import annotations


def close_writer(writer):
    """Best-effort shutdown for training writers with mixed backends."""

    if writer is None:
        return None

    for method_name in ("flush", "close", "stop"):
        method = getattr(writer, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass

    return None
