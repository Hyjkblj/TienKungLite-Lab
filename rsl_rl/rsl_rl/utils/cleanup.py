from __future__ import annotations


def _print_shutdown(message: str) -> None:
    print(f"[SHUTDOWN] {message}", flush=True)


def close_writer(writer):
    """Best-effort shutdown for training writers with mixed backends."""

    if writer is None:
        _print_shutdown("Writer cleanup skipped: writer is None.")
        return None

    for method_name in ("flush", "close", "stop"):
        method = getattr(writer, method_name, None)
        if callable(method):
            _print_shutdown(f"Calling writer.{method_name}()")
            try:
                method()
                _print_shutdown(f"writer.{method_name}() completed.")
            except Exception:
                _print_shutdown(f"writer.{method_name}() raised and was ignored.")
                pass

    return None
