"""In-memory wave admission and retry policy, independent of durable side effects.

The lifecycle thread owns this state. Provider and command workers only return
results; none of these scheduling decisions write to the world store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from mikoshi.contracts import ActivationSummary, Trigger
from mikoshi.decisions import DecisionRequest, ProviderTransportError
from mikoshi.team_protocol import TeamTool


def retryable_provider_error(error: Exception) -> bool:
    if isinstance(error, ProviderTransportError):
        status = error.status_code
        return status is None or status in {408, 409, 425, 429} or 500 <= status < 600
    # Provider adapters may expose a transient failure without HTTP metadata.
    return isinstance(error, RuntimeError)


@dataclass(frozen=True)
class ProviderRetryPolicy:
    delays: Sequence[float]
    rate_limit_delay: float

    def delay_for(self, error: Exception, failures: int) -> float | None:
        """Return the delay after a failure, or None when it should not be retried."""
        if not retryable_provider_error(error) or not 1 <= failures <= len(self.delays):
            return None
        delay = self.delays[failures - 1]
        requested = getattr(error, "retry_after_seconds", None)
        if requested is not None:
            return max(delay, requested)
        if getattr(error, "status_code", None) == 429:
            return max(delay, self.rate_limit_delay)
        return delay


@dataclass
class WorkspaceActivation:
    activation_id: int
    trigger: Trigger
    steps: int = 0
    pending_request: DecisionRequest | None = None
    pending_decision: Mapping[str, Any] | Exception | None = None
    pending_command_id: int | None = None
    pending_decision_round: int | None = None
    pending_handle: int | None = None
    provider_failures: int = 0
    repository_guidance: dict[str, Any] = field(default_factory=dict)
    batch_results: list[Any] = field(default_factory=list)
    tools: dict[str, TeamTool] = field(default_factory=dict)

    def clear_completed_turn(self) -> None:
        """Forget a recorded turn, preserving the activation's conversation identity."""
        self.batch_results = []
        self.pending_decision = None
        self.pending_request = None
        self.pending_decision_round = None
        self.pending_command_id = None


@dataclass
class WaveSchedule:
    active: list[WorkspaceActivation]
    decision_round: int = 0
    summaries: list[ActivationSummary] = field(default_factory=list)
    in_flight: dict[int, WorkspaceActivation] = field(default_factory=dict)
    unavailable: set[int] = field(default_factory=set)
    context_limited: set[int] = field(default_factory=set)
    retry_ready_at: dict[int, float] = field(default_factory=dict)
    admission_stop: str | None = None
    stop_reason: str | None = None

    def block_admission(self, reason: str) -> None:
        """Settle already admitted work without issuing additional requests."""
        self.admission_stop = self.admission_stop or reason

    def request_candidates(
        self, concurrency: int, command_ids: Mapping[int, Any], now: float
    ) -> Iterator[WorkspaceActivation]:
        """Yield runnable sessions in fair order, observing slots filled by the caller."""
        for state in self.active:
            if self.admission_stop or len(self.in_flight) >= concurrency:
                break
            if (
                state.activation_id in self.unavailable
                or now < self.retry_ready_at.get(state.activation_id, 0)
                or state.activation_id in command_ids
                or any(pending is state for pending in self.in_flight.values())
                or state.pending_decision is not None
            ):
                continue
            yield state

    def track_request(self, handle: int, state: WorkspaceActivation) -> None:
        self.retry_ready_at.pop(state.activation_id, None)
        state.pending_handle = handle
        self.in_flight[handle] = state

    def complete_turn(self, state: WorkspaceActivation, *, terminal: bool) -> None:
        state.clear_completed_turn()
        self.active.remove(state)
        if not terminal:
            self.active.append(state)

    def idle_stop_reason(self) -> str | None:
        """Explain a stalled wave after all immediate work has been considered."""
        if self.admission_stop:
            return self.admission_stop
        if self.active and all(s.activation_id in self.unavailable for s in self.active):
            return (
                "context_window_exhausted"
                if all(s.activation_id in self.context_limited for s in self.active)
                else "provider_unavailable"
            )
        return None

    def retry_wait(self, now: float) -> float:
        """Bound the sleep so stop requests remain responsive during long backoffs."""
        return min(0.05, max(0, min(self.retry_ready_at.values()) - now))
