"""Durable run status, stop escalation and terminal finalization."""

from __future__ import annotations

import atexit
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mikoshi.runtime.budgets import BudgetExhausted
from mikoshi.runtime.control import (
    DEFAULT_STOP_INTERRUPT_SECONDS,
    DEFAULT_STOP_TERMINATE_SECONDS,
    StopMonitor,
    stop_signals,
)
from mikoshi.schema import RUN_FILE, RecordWriter, create_run, read_document, update_run


def resolve_execution_status(stop_reason):
    if stop_reason in {
        "provider_unavailable",
        "security_boundary_failure",
        "command_cleanup_unconfirmed",
    } or bool(stop_reason and stop_reason.startswith("runtime_error:")):
        return "failed"
    if stop_reason in {
        "stop_requested",
        "budget_exhausted",
        "wall_time_exhausted",
        "context_window_exhausted",
    }:
        return "stopped"
    return "completed"


class RunLifecycle:
    """Persist a small terminal record even when a full manifest is never produced."""

    def __init__(
        self,
        output_root: Path,
        started_at: str,
        *,
        progress_callback=None,
        stop_interrupt_after: float = DEFAULT_STOP_INTERRUPT_SECONDS,
        stop_terminate_after: float = DEFAULT_STOP_TERMINATE_SECONDS,
    ) -> None:
        self._progress_callback = progress_callback
        self.output_root = output_root
        self.started_at = started_at
        self.current_phase = "preparing"
        self.last_progress: Mapping[str, Any] | None = None
        self.agents: dict[str, Any] = {}
        self.execution_stop_reason = None
        self.finished = False
        self.budget = None
        self._stop_reason = None
        self.cancel_commands = lambda: None
        self._previous_signals = {}
        if (output_root / RUN_FILE).exists():
            raise ValueError(
                f"{output_root} already contains a run; choose a new --output directory"
            )
        document = create_run(output_root, started_at=started_at)
        self.records = RecordWriter(output_root, document["id"])
        self._write("running")
        atexit.register(self._record_interruption)
        for signum in stop_signals():
            self._previous_signals[signum] = signal.signal(signum, self._request_stop)
        self.stop_monitor = StopMonitor(
            output_root,
            self._operator_stop_requested,
            lambda: self.cancel_commands(),
            interrupt_after=stop_interrupt_after,
            terminate_after=stop_terminate_after,
        )

    def _request_stop(self, signum, frame):
        if self._stop_reason is not None:
            raise KeyboardInterrupt("forced interruption after graceful stop request")
        self._stop_reason = "stop_requested"

    def _operator_stop_requested(self) -> bool:
        if self._stop_reason or (self.output_root / "STOP").exists():
            self._stop_reason = "stop_requested"
            return True
        return False

    def stop_requested(self) -> str | None:
        if self._stop_reason or (self.output_root / "STOP").exists():
            return "stop_requested"
        if self.budget is not None:
            try:
                self.budget.check()
            except BudgetExhausted as error:
                return str(error)
        return None

    def _restore_signals(self) -> None:
        self.stop_monitor.close()
        for signum, previous in self._previous_signals.items():
            signal.signal(signum, previous)
        self._previous_signals.clear()

    def phase(self, phase: str) -> None:
        self.current_phase = phase
        self.progress({"kind": "phase_started", "phase": phase})

    def progress(self, event: Mapping[str, Any]) -> None:
        """Persist and stream bounded progress without affecting run behavior."""
        if self.finished:
            return
        if event.get("kind") == "decision_prepared":
            self.records.append(
                "decision",
                event,
                source_id=f"decision:{event['decision_id']}",
                agent_id=event.get("agent_id"),
                activation_id=event.get("activation_id"),
            )
            event = {
                key: event[key]
                for key in ("kind", "decision_id", "agent_id", "activation_id", "activation_step")
            }
        event = self.records.prepare_progress(event)
        self.last_progress = {
            **dict(event),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        if agent_id := event.get("agent_id"):
            self.agents[str(agent_id)] = self.last_progress
        self._write("running")
        self.records.append(
            "progress",
            {"phase": self.current_phase, **self.last_progress},
            agent_id=event.get("agent_id"),
            activation_id=event.get("activation_id"),
        )
        if self._progress_callback is not None:
            try:
                self._progress_callback(
                    {"phase": self.current_phase, "progress": self.last_progress}
                )
            except Exception:
                # Terminal presentation cannot change the outcome of paid work.
                pass

    def complete(self, stop_reason: str | None = None, evaluation_status=None) -> bool:
        self.finished = True
        execution_status = resolve_execution_status(stop_reason)
        failed = execution_status == "failed" or evaluation_status == "failed"
        self.current_phase = "finished"
        self.records.append(
            "progress",
            {
                "kind": "run_finished",
                "status": "failed" if failed else execution_status,
                "stop_reason": stop_reason,
            },
        )
        evidence = read_document(self.output_root / RUN_FILE)["evidence"]
        self._write(
            "failed" if failed else execution_status,
            evidence={**evidence, "record_count": self.records.sequence},
            execution_status=execution_status,
            stop_reason=stop_reason,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        self._restore_signals()
        atexit.unregister(self._record_interruption)
        return failed

    def fail(self, error: BaseException) -> None:
        self.finished = True
        interrupted = isinstance(error, KeyboardInterrupt)
        try:
            self._write(
                "interrupted" if interrupted else "failed",
                execution_status="interrupted"
                if interrupted
                else (
                    resolve_execution_status(self.execution_stop_reason)
                    if self.execution_stop_reason
                    else "failed"
                ),
                error=f"{type(error).__name__}: {error}",
                stop_reason=self.execution_stop_reason,
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            if self.budget is not None:
                self.budget.save()
        finally:
            self._restore_signals()
            atexit.unregister(self._record_interruption)

    def _record_interruption(self) -> None:
        if not self.finished:
            self._write("interrupted", execution_status="interrupted")
        self.stop_monitor.close()

    def _write(self, status: str, **extra: Any) -> None:
        update_run(
            self.output_root,
            status=status,
            phase=self.current_phase,
            progress=self.last_progress,
            agents=self.agents,
            budget=self.budget.snapshot() if self.budget is not None else None,
            **extra,
        )
