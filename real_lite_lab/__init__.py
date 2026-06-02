from pathlib import Path
import sys

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


def register_tasks():
    from .registry import register_tasks as _register_tasks

    return _register_tasks()


def __getattr__(name: str):
    if name == "task_registry":
        from .task_registry import task_registry as _task_registry

        return _task_registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["register_tasks", "task_registry"]
