"""Supervise isolated host controllers without sharing their signal handlers."""

from __future__ import annotations

import asyncio
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mikoshi.isolation.processes import (
    ProcessCleanupError,
    ProcessPipe,
    spawn_process,
    terminate_process_tree,
)

_EVENT_BYTES = 8192
_ESCALATION_SECONDS = 2.0
_PIPE_DRAIN_SECONDS = 2.0


async def _spawn(config_path: Path):
    """Own nonblocking pipe readers on both POSIX and native Windows."""
    process = spawn_process(
        _command(config_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
    )
    pipes = []
    try:
        pipes = [ProcessPipe(process.stdout), ProcessPipe(process.stderr)]
        return process, pipes
    except BaseException:
        terminate_process_tree(process)
        for pipe in pipes:
            pipe.close()
        for source in (process.stdout, process.stderr):
            if source is not None:
                source.close()
        raise


async def _wait(process: subprocess.Popen) -> int:
    while (returncode := process.poll()) is None:
        await asyncio.sleep(0.01)
    return returncode


async def _cleanup(process: subprocess.Popen) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(terminate_process_tree, process)
    except Exception as error:
        return {"confirmed": False, "error": f"{type(error).__name__}: {error}"}


def _command(config_path: Path) -> list[str]:
    return [sys.executable, "-m", "mikoshi", "--config", str(config_path)]


def _emit(emit: Callable[[Mapping[str, Any]], None], event: Mapping[str, Any]) -> None:
    try:
        emit(event)
    except Exception:
        # An unavailable observer must not change the trial's behavior.
        pass


async def _drain(
    stream: ProcessPipe,
    path: Path,
    name: str,
    emit: Callable[[Mapping[str, Any]], None],
) -> None:
    pending = bytearray()
    truncated = False

    def publish() -> None:
        nonlocal truncated
        text = pending.decode("utf-8", errors="replace")
        try:
            event = json.loads(text) if not truncated else None
        except (ValueError, RecursionError):
            event = None
        if isinstance(event, dict):
            _emit(emit, event)
        elif text or truncated:
            _emit(
                emit,
                {"kind": "process_output", "stream": name, "text": text, "truncated": truncated},
            )
        pending.clear()
        truncated = False

    with path.open("wb") as target:
        while True:
            chunk = stream.read(65536)
            if chunk is None:
                await asyncio.sleep(0.01)
                continue
            if not chunk:
                break
            target.write(chunk)
            target.flush()
            parts = chunk.split(b"\n")
            for index, part in enumerate(parts):
                remaining = _EVENT_BYTES - len(pending)
                pending.extend(part[:remaining])
                truncated = truncated or len(part) > remaining
                if index < len(parts) - 1:
                    publish()
        if pending or truncated:
            publish()


async def _stop(
    process: subprocess.Popen,
    waiter: asyncio.Task[int],
    run_dir: Path,
    timeout: float,
    report: dict[str, Any],
) -> None:
    deadline = time.monotonic() + timeout
    while not waiter.done() and time.monotonic() < deadline:
        # Creating STOP too early would violate the CLI's empty-output check.
        if (run_dir / "run.json").is_file() and not report["stop_requested"]:
            (run_dir / "STOP").touch()
            report["stop_requested"] = True
        await asyncio.wait({waiter}, timeout=min(0.1, max(0, deadline - time.monotonic())))
    if waiter.done():
        return
    report["escalation"] = "terminate"
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    await asyncio.wait({waiter}, timeout=_ESCALATION_SECONDS)
    if not waiter.done():
        report["escalation"] = "kill"
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await waiter


async def _supervise(
    config_path: Path,
    run_dir: Path,
    emit: Callable[[Mapping[str, Any]], None],
    stop_timeout_seconds: float,
    stop_requested: Callable[[], bool] | None,
    cancelled: asyncio.Event,
) -> int:
    parent = run_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "stop_requested": False,
        "cancellation_requested": False,
        "escalation": None,
        "returncode": None,
    }
    process = None
    readers: list[asyncio.Task[None]] = []
    pipes = []
    cleanup = None
    try:
        process, pipes = await _spawn(config_path)
        readers = [
            asyncio.create_task(_drain(pipes[0], parent / "supervisor.stdout.log", "stdout", emit)),
            asyncio.create_task(_drain(pipes[1], parent / "supervisor.stderr.log", "stderr", emit)),
        ]
        waiter = asyncio.create_task(_wait(process))
        try:
            while not waiter.done():
                for reader in readers:
                    if reader.done() and (error := reader.exception()) is not None:
                        raise error
                if cancelled.is_set() or (stop_requested is not None and stop_requested()):
                    report["cancellation_requested"] = cancelled.is_set()
                    await _stop(process, waiter, run_dir, stop_timeout_seconds, report)
                    break
                await asyncio.wait({waiter}, timeout=0.1)
            report["returncode"] = await waiter
        except BaseException:
            await _stop(process, waiter, run_dir, stop_timeout_seconds, report)
            raise
        finally:
            cleanup = await _cleanup(process)
            report["controller_cleanup"] = cleanup
            _, pending = await asyncio.wait(readers, timeout=_PIPE_DRAIN_SECONDS)
            report["output_capture_complete"] = not pending
            if pending:
                report["output_capture_warning"] = (
                    "inherited output pipes remained open after child exit"
                )
                for pipe in pipes:
                    pipe.close()
                for reader in pending:
                    reader.cancel()
            results = await asyncio.gather(*readers, return_exceptions=True)
            for reader, result in zip(readers, results):
                if isinstance(result, BaseException) and reader not in pending:
                    report["output_capture_complete"] = False
                    raise result
            if not cleanup["confirmed"]:
                raise ProcessCleanupError(f"controller cleanup unconfirmed: {cleanup}")
        return report["returncode"]
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if process is not None and cleanup is None:
            report["controller_cleanup"] = await _cleanup(process)
        for pipe in pipes:
            pipe.close()
        if process is not None:
            report["returncode"] = process.returncode
        report["exports_present"] = (run_dir / "records.jsonl").is_file()
        try:
            status = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = None
        report["native_status"] = status
        # Process exit cannot establish whether Docker commands were cleaned up.
        report["cleanup_evidence"] = "consult native status and manifest"
        (parent / "supervisor.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )


async def run_process(
    config_path: Path,
    run_dir: Path,
    emit: Callable[[Mapping[str, Any]], None],
    *,
    stop_timeout_seconds: float = 10.0,
    stop_requested: Callable[[], bool] | None = None,
) -> int:
    """Run one native trial, retaining raw output and reaping it before cancellation.

    ``config_path`` must name a single-trial configuration whose output is
    ``run_dir``. Each trial needs its own parent directory for supervisor logs.
    Cancellation requests STOP, waits for native exports, then escalates if needed.
    Native status and manifests remain the authority on outcome and cleanup.
    """
    if not math.isfinite(stop_timeout_seconds) or stop_timeout_seconds <= 0:
        raise ValueError("stop_timeout_seconds must be positive and finite")
    cancelled = asyncio.Event()
    worker = asyncio.create_task(
        _supervise(config_path, run_dir, emit, stop_timeout_seconds, stop_requested, cancelled)
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancelled.set()
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Repeated cancellation must not orphan the host controller.
                continue
            except Exception:
                break
        if worker.done() and not worker.cancelled():
            worker.exception()
        raise
