from __future__ import annotations

import faulthandler
import signal
import sys
from typing import TextIO


def install_stack_dump_signal(file: TextIO | None = None) -> int | None:
    """Enable faulthandler and register SIGUSR1 to dump Python stacks."""

    target = file if file is not None else sys.stderr
    faulthandler.enable(file=target, all_threads=True)

    dump_signal = getattr(signal, "SIGUSR1", None)
    register = getattr(faulthandler, "register", None)
    if dump_signal is None or register is None:
        return None

    register(dump_signal, file=target, all_threads=True, chain=False)
    return int(dump_signal)
