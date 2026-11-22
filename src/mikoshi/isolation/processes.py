"""Host process lifecycle for Docker clients and isolated run controllers.

This module never chooses which programs agents can execute. Agent commands
still run exclusively through Docker; these helpers supervise the host clients.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import threading
import time


class ProcessCleanupError(RuntimeError):
    """A host process tree could not be confirmed terminated."""


def spawn_process(argv, **options):
    """Create an owned process tree, preventing Windows descendants before assignment."""
    if os.name == "posix":
        return subprocess.Popen(argv, **{**options, "start_new_session": True})
    if os.name != "nt":
        raise RuntimeError(f"process supervision is unsupported on {os.name}")
    from mikoshi.isolation.windows_processes import WindowsJob

    job = WindowsJob()
    process = None
    try:
        # A separate console group prevents a terminal Ctrl-C from bypassing the
        # controller's graceful STOP protocol. The Job Object still owns children.
        flags = options.pop("creationflags", 0) | 0x00000004 | 0x00000200
        process = subprocess.Popen(argv, **{**options, "creationflags": flags})
        process._mikoshi_job = job
        job.assign_and_resume(process)
        return process
    except BaseException:
        cleanup_error = None
        try:
            job.close()  # Kill-on-close also covers failure after primary-thread resume.
        except Exception as error:
            cleanup_error = error
        if process is not None:
            try:
                process.kill()  # Also covers job assignment failure while still suspended.
                process.wait(timeout=2)
            except Exception as error:
                cleanup_error = cleanup_error or error
            finally:
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if pipe is not None:
                        pipe.close()
        if cleanup_error is not None:
            raise ProcessCleanupError(f"failed launch cleanup unconfirmed: {cleanup_error}")
        raise


class ProcessPipe:
    """A single-reader pipe with bounded reads and no blocking reader thread.

    ``read`` returns None when no bytes are available, b"" at EOF, or one bounded
    chunk. Closing never waits for inherited write handles in another process.
    """

    def __init__(self, pipe):
        self.pipe = pipe
        self.eof = False
        self._windows = None
        if os.name == "nt":
            from mikoshi.isolation.windows_processes import WindowsAPI

            self._windows = WindowsAPI()
        else:
            os.set_blocking(pipe.fileno(), False)

    def read(self, max_bytes=65536):
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        if self.eof:
            return b""
        if self._windows is not None:
            available = self._windows.available_pipe_bytes(self.pipe.fileno())
            if available is None:
                self.eof = True
                return b""
            if available == 0:
                return None
            max_bytes = min(max_bytes, available)
        try:
            chunk = os.read(self.pipe.fileno(), max_bytes)
        except BlockingIOError:
            return None
        if not chunk:
            self.eof = True
        return chunk

    def close(self):
        self.eof = True
        self.pipe.close()


class _ProcessInspector:
    """Serialize POSIX scans and share only observations begun after cleanup."""

    def __init__(self):
        self._lock = threading.Lock()
        self._started_at = -math.inf
        self._rows = ()

    def rows(self, not_before):
        with self._lock:
            if self._started_at >= not_before:
                return self._rows
            started_at = time.monotonic()
            inspection = subprocess.run(
                ["/bin/ps", "-axo", "pid=,pgid=,stat="],
                capture_output=True,
                text=True,
                timeout=1,
                check=True,
            )
            rows = tuple(
                tuple(line.split()) for line in inspection.stdout.splitlines() if line.strip()
            )
            if not rows or any(
                len(row) != 3 or not row[0].isdigit() or not row[1].isdigit() for row in rows
            ):
                raise ProcessCleanupError("process inspection returned empty or malformed output")
            self._started_at, self._rows = started_at, rows
            return rows


_PROCESS_INSPECTOR = _ProcessInspector()


def _terminate_posix_group(process, timeout):
    signal_error = None
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError as error:
        signal_error = str(error)
    process.wait(timeout=timeout)
    deadline = time.monotonic() + timeout
    while True:
        rows = _PROCESS_INSPECTOR.rows(time.monotonic())
        live = [
            int(pid)
            for pid, group, state in rows
            if group == str(process.pid) and not state.startswith("Z")
        ]
        if not live or time.monotonic() >= deadline:
            result = {"confirmed": not live, "scope": "process_group", "live_pids": live[:32]}
            if signal_error is not None:
                result["signal_error"] = signal_error
            return result
        time.sleep(0.05)


def terminate_process_tree(process, timeout=2.0):
    """Terminate descendants, verify cleanup, and release native lifecycle handles.

    POSIX process groups cover ordinary children, which can detach; this is
    lifecycle management rather than a containment guarantee. Windows jobs keep
    ordinary CreateProcess descendants, including nested jobs, with breakaway
    disabled. Docker container destruction remains a separate obligation.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("cleanup timeout must be positive and finite")
    if hasattr(process, "_mikoshi_cleanup"):
        return dict(process._mikoshi_cleanup)
    job = getattr(process, "_mikoshi_job", None)
    if job is not None:
        try:
            result = job.terminate(process, timeout)
        finally:
            job.close()
    elif os.name == "posix":
        result = _terminate_posix_group(process, timeout)
    else:
        raise ProcessCleanupError("process lacks its required Windows Job Object")
    process._mikoshi_cleanup = dict(result)
    return result


def run_captured(argv, *, timeout, max_output_bytes=4 * 1024 * 1024, **options):
    """Capture a short host utility with bounded pipes and verified tree cleanup.

    Docker create/inspect/remove use this path too: subprocess.run's timeout only
    reaps its immediate child, which can leave a client helper running.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("capture timeout must be positive and finite")
    process = spawn_process(
        argv,
        **{
            **options,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        },
    )
    pipes = []
    output = [bytearray(), bytearray()]
    total = 0
    deadline = time.monotonic() + timeout

    def read_available():
        nonlocal total
        progress = False
        for pipe, target in zip(pipes, output):
            if pipe.eof:
                continue
            chunk = pipe.read()
            if chunk is not None:
                progress = True
                total += len(chunk)
                if total > max_output_bytes:
                    raise subprocess.SubprocessError("host utility output exceeded capture limit")
                target.extend(chunk)
        return progress

    try:
        for source in (process.stdout, process.stderr):
            pipes.append(ProcessPipe(source))
        while process.poll() is None:
            progress = read_available()
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(argv, timeout, bytes(output[0]), bytes(output[1]))
            if not progress:
                time.sleep(0.01)
        cleanup = terminate_process_tree(process)
        if not cleanup["confirmed"]:
            raise ProcessCleanupError(str(cleanup))
        drain_deadline = time.monotonic() + 2
        while not all(pipe.eof for pipe in pipes):
            if not read_available():
                if time.monotonic() >= drain_deadline:
                    raise ProcessCleanupError(
                        "host utility output pipes remained open after cleanup"
                    )
                time.sleep(0.01)
        return subprocess.CompletedProcess(
            argv, process.returncode, bytes(output[0]), bytes(output[1])
        )
    finally:
        try:
            cleanup = terminate_process_tree(process)
            if not cleanup["confirmed"]:
                raise ProcessCleanupError(str(cleanup))
        finally:
            for pipe in pipes:
                pipe.close()
            # Construction may fail before both wrappers have been installed.
            for source in (process.stdout, process.stderr):
                if not source.closed:
                    source.close()
