from pathlib import Path
import sys

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from .registry import register_tasks, task_registry

__all__ = ["register_tasks", "task_registry"]
