"""Independent stop escalation, including while the main thread is doing I/O."""

from __future__ import annotations

import _thread
import os
import signal
import threading
import time
from pathlib import Path
from typing import Callable

DEFAULT_STOP_INTERRUPT_SECONDS = 2.0
DEFAULT_STOP_TERMINATE_SECONDS = 5.0


def stop_signals() -> tuple[signal.Signals, ...]:
    """Console stop signals available on this host, including Windows Ctrl-Break."""
    return (signal.SIGINT, signal.SIGTERM) + (
        (signal.SIGBREAK,) if hasattr(signal, "SIGBREAK") else ()
    )


class StopMonitor:
    """Cancel commands, interrupt Python, then terminate a stuck runner.

    The daemon never reads the workspace or waits for database writes. A forced
    termination preserves working files and committed records, but can lose
    in-flight results and final exports.
    """

    def __init__(
        self,
        output: Path,
        requested: Callable[[], bool],
        cancel: Callable[[], None],
        *,
        interrupt_after: float = DEFAULT_STOP_INTERRUPT_SECONDS,
        terminate_after: float = DEFAULT_STOP_TERMINATE_SECONDS,
    ) -> None:
        self.output = output
        self.requested = requested
        self.cancel = cancel
        self.interrupt_after = interrupt_after
        self.terminate_after = terminate_after
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._watch, name="swarm-stop", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.closed.set()

    def _watch(self) -> None:
        started = None
        interrupted = False
        while not self.closed.wait(0.1):
            if started is None:
                if not self.requested():
                    continue
                started = time.monotonic()
            try:
                self.cancel()
            except Exception:
                # A cleanup error must not disable independent termination.
                pass
            elapsed = time.monotonic() - started
            if elapsed >= self.terminate_after:
                # Even a tiny status write can block on a stuck filesystem.
                os._exit(130)
            if elapsed >= self.interrupt_after and not interrupted:
                interrupted = True
                _thread.interrupt_main()
