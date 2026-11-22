"""FIFO command admission driven by Docker resources and controller-host file pressure."""

import ctypes
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future
from pathlib import Path

ADMISSION_INTERVAL_SECONDS = 0.5


class PressureUnavailable(RuntimeError):
    """Observation failed safely; keep commands queued until pressure can be read."""


def host_file_table_source():
    return (
        "sysctlbyname"
        if sys.platform == "darwin"
        else "procfs"
        if sys.platform == "linux"
        else None
    )


def _darwin_file_table():
    # No subprocess or new file descriptor is needed while the host is under
    # pressure. Resolve the sysctl symbol from the already-loaded process image.
    library = ctypes.CDLL(None, use_errno=True)
    sysctl = library.sysctlbyname
    sysctl.argtypes = (
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    )
    sysctl.restype = ctypes.c_int

    def number(name):
        value = ctypes.c_uint64()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if sysctl(name, ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        if size.value not in (4, 8):
            raise ValueError("unexpected sysctl integer size")
        return int.from_bytes(ctypes.string_at(ctypes.byref(value), size.value), sys.byteorder)

    return number(b"kern.num_files"), number(b"kern.maxfiles")


def _linux_file_table():
    allocated, unused, maximum = map(int, Path("/proc/sys/fs/file-nr").read_text().split())
    if unused < 0 or unused > allocated:
        raise ValueError("invalid Linux file-table counters")
    return allocated - unused, maximum


def host_file_table_pressure():
    """Controller-host allocated file handles as a percentage of its global limit."""
    source = host_file_table_source()
    if source is None:
        return None
    try:
        used, maximum = _darwin_file_table() if source == "sysctlbyname" else _linux_file_table()
        if used < 0 or maximum <= 0:
            raise ValueError("invalid host file-table counters")
        return {"host_file_table_percent": 100 * used / maximum}
    except (OSError, ValueError, OverflowError) as error:
        raise PressureUnavailable(
            f"Host file-table observation unavailable ({source}): {type(error).__name__}: {error}"
        ) from error


# Runs only in a bounded, read-only probe container, never on the controller host.
# /proc describes Docker's Linux host, including workloads outside this run.
PRESSURE_SCRIPT = (
    "set -e; head -n 1 /proc/stat; sleep 0.2; head -n 1 /proc/stat; "
    "grep -E '^(MemTotal|MemAvailable):' /proc/meminfo"
)


def parse_pressure(text):
    """Normalize aggregate CPU counters and reclaimable-memory-aware RAM usage."""
    try:
        before, after, *memory_rows = [line.split() for line in text.splitlines()]
        if before[0] != "cpu" or after[0] != "cpu" or len(before) < 9 or len(after) < 9:
            raise ValueError()
        # Guest time is already in user/nice. Some kernel counters (iowait) can decrease.
        delta = [max(0, int(end) - int(start)) for start, end in zip(before[1:9], after[1:9])]
        total, idle = sum(delta), delta[3] + delta[4]
        memory = {row[0].rstrip(":"): int(row[1]) for row in memory_rows}
        if (
            total <= 0
            or not 0 <= memory["MemAvailable"] <= memory["MemTotal"]
            or memory["MemTotal"] <= 0
        ):
            raise ValueError()
        return {
            "cpu_percent": 100 * (1 - idle / total),
            "memory_percent": 100 * (1 - memory["MemAvailable"] / memory["MemTotal"]),
        }
    except (ValueError, IndexError, KeyError) as error:
        raise ValueError("Docker returned invalid resource pressure") from error


class CommandQueue:
    """Waiting commands consume no worker threads; running commands share resources."""

    def __init__(self, execute, sample, target):
        self.execute, self.sample, self.target = execute, sample, target
        self.pending = deque()
        self.condition = threading.Condition()
        self.dispatcher = None
        self.running = 0
        self.error = None

    def submit(self, prepared):
        if not prepared._execution_lock.acquire(blocking=False):
            raise RuntimeError("prepared command execution may only run once")
        future = Future()
        with self.condition:
            self.pending.append((prepared, future))
            if self.dispatcher is None:
                self.dispatcher = threading.Thread(
                    target=self._dispatch, name="docker-admission", daemon=True
                )
                self.dispatcher.start()
            self.condition.notify_all()
        return future

    def _launch(self, prepared, future):
        if future.set_running_or_notify_cancel():
            self.running += 1
            worker = threading.Thread(
                target=self._run, args=(prepared, future), name="docker-command", daemon=True
            )
            try:
                worker.start()
            except Exception as error:
                self.running -= 1
                self.error = error
                future.set_exception(error)

    def _run(self, prepared, future):
        try:
            result = self.execute(prepared)
            with self.condition:
                if result.get("termination_reason") in {
                    "command_cleanup_failed",
                    "workspace_validation_failed",
                }:
                    self.error = RuntimeError("command cleanup unconfirmed; admission stopped")
            future.set_result(result)
        except Exception as error:
            with self.condition:
                self.error = error
            future.set_exception(error)
        finally:
            with self.condition:
                self.running -= 1
                self.condition.notify_all()

    def _dispatch(self):
        sampled_at = -float("inf")
        while True:
            with self.condition:
                for prepared, future in list(self.pending):
                    expired = (
                        prepared.timeout_seconds is not None
                        and time.monotonic() >= prepared.queued_at + prepared.timeout_seconds
                    )
                    if (
                        future.cancelled()
                        or self.error
                        or prepared.cancel_event.is_set()
                        or expired
                    ):
                        self.pending.remove((prepared, future))
                        if future.cancelled():
                            continue
                        if self.error:
                            future.set_exception(self.error)
                        else:
                            self._launch(prepared, future)
                if not self.pending:
                    if not self.running:
                        self.dispatcher = None
                        return
                    self.condition.wait(0.05)
                    continue
                if time.monotonic() - sampled_at < ADMISSION_INTERVAL_SECONDS:
                    self.condition.wait(0.05)
                    continue
                prepared, future = self.pending[0]
            try:
                pressure = self.sample(prepared)
                sampled_at = time.monotonic()
                with self.condition:
                    if self.error is None and max(pressure.values()) < self.target:
                        self.pending.popleft()
                        self._launch(prepared, future)
            except PressureUnavailable:
                sampled_at = time.monotonic()
            except Exception as error:
                with self.condition:
                    self.error = error
