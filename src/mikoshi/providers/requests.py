"""Owner-thread request states and the transport attempt shared with the I/O loop."""

from __future__ import annotations

from collections.abc import MutableMapping
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from mikoshi.decisions import BatchDecisionResult, DecisionRequest


class RequestState(Enum):
    PREPARED = "prepared"
    INFLIGHT = "inflight"
    READY = "ready"
    SUSPENDED = "suspended"


@dataclass(frozen=True)
class RequestIdentity:
    request_id: str
    decision_id: str
    previous_request_id: str | None
    endpoint: str
    agent_id: str | None = None
    activation_id: int | None = None
    activation_step: int | None = None

    def evidence(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TransportStats(MutableMapping[str, Any]):
    """Written by one I/O task, read after its future completes on the owner thread.

    Mapping access remains available to installed transport adapters that extend
    ``_request(payload, stats)``. Core code uses the explicit fields.
    """

    evidence: dict[str, Any]
    timeout_seconds: float
    transport_attempts: int = 0
    transport_seconds: float = 0.0
    queue_seconds: float = 0.0
    status_codes: list[int] = field(default_factory=list)
    submitted: bool = False
    response_received: bool = False
    finished: float | None = None
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __getitem__(self, key):
        if key in self.__dataclass_fields__ and key != "_extra":
            return getattr(self, key)
        return self._extra[key]

    def __setitem__(self, key, value):
        if key in self.__dataclass_fields__ and key != "_extra":
            setattr(self, key, value)
        else:
            self._extra[key] = value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self):
        yield from (key for key in self.__dataclass_fields__ if key != "_extra")
        yield from self._extra

    def __len__(self):
        return len(self.__dataclass_fields__) - 1 + len(self._extra)

    def __delitem__(self, key):
        if key in self.__dataclass_fields__ and key != "_extra":
            raise TypeError(f"cannot delete transport statistic {key!r}")
        del self._extra[key]

    def snapshot(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(result.pop("_extra"))
        return result


@dataclass
class RequestAttempt:
    identity: RequestIdentity
    budget_request_id: int | None
    started: float
    stats: TransportStats
    future: Future | None = None


@dataclass
class CompactionCheckpoint:
    request: DecisionRequest
    payload: dict[str, Any]
    checkpoints: list[dict[str, Any]]


@dataclass
class RequestRecord:
    """One caller handle; each continuation owns a fresh transport attempt.

    Transitions and results belong to the scheduler thread. Transport may mutate
    only its attempt's stats, never this record's lifecycle state.
    """

    handle: int
    request: DecisionRequest
    decision_id: str
    state: RequestState = field(default=RequestState.PREPARED, init=False)
    prepared: DecisionRequest | None = None
    payload: dict[str, Any] | None = None
    result: BatchDecisionResult | None = None
    attempt: RequestAttempt | None = None
    compaction_checkpoints: list[dict[str, Any]] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.state in (RequestState.READY, RequestState.SUSPENDED)

    @property
    def active_attempt(self) -> RequestAttempt:
        if self.attempt is None:
            raise RuntimeError("request has no transport attempt")
        return self.attempt

    def _require(self, *states: RequestState) -> None:
        if self.state not in states:
            expected = ", ".join(state.value for state in states)
            raise RuntimeError(f"cannot transition {self.state.value} request; expected {expected}")

    def reject(self, error: Exception) -> None:
        self._require(RequestState.PREPARED)
        self.result = error
        self.state = RequestState.READY

    def suspend(self, error: Exception) -> None:
        self._require(RequestState.PREPARED, RequestState.SUSPENDED, RequestState.INFLIGHT)
        self.result = error
        self.state = RequestState.SUSPENDED

    def dispatch(self, attempt: RequestAttempt) -> None:
        self._require(RequestState.PREPARED, RequestState.SUSPENDED)
        if self.prepared is None or self.payload is None:
            raise RuntimeError("cannot dispatch an unprepared request")
        self.attempt = attempt
        self.result = None
        self.state = RequestState.INFLIGHT

    def complete(self, result: BatchDecisionResult) -> None:
        self._require(RequestState.INFLIGHT)
        self.result = result
        self.state = RequestState.READY
        self.active_attempt.future = None

    def continue_after_compaction(
        self,
        payload: dict[str, Any],
        checkpoints: list[dict[str, Any]],
        interruption: Exception | None,
    ) -> None:
        self._require(RequestState.INFLIGHT)
        self.payload = payload
        self.compaction_checkpoints = checkpoints
        self.active_attempt.future = None
        self.result = interruption
        self.state = RequestState.SUSPENDED if interruption else RequestState.PREPARED


def snapshot_request(request: DecisionRequest) -> DecisionRequest:
    """Normalize caller-owned mappings to the same JSON values sent to adapters."""
    import json

    return DecisionRequest(**json.loads(json.dumps(asdict(request))))
