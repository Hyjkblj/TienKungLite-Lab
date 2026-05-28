from pathlib import Path
import sys

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from .task_registry import task_registry


def register_tasks():
    from .registry import register_tasks as _register_tasks

    return _register_tasks()

__all__ = ["register_tasks", "task_registry"]
