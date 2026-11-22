"""Provider-independent turn contract shared by team protocols, adapters and archives."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class DecisionRequest:
    name: str
    instructions: str
    input_text: str
    schema: Mapping[str, Any]
    max_tokens: int | None = None
    stage: str = "actor"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Last committed exchange only, for continuation and new tool results.
    history: Sequence[Mapping[str, Any]] = ()
    # Stable within-run work conversation; standalone control requests omit it.
    conversation_id: str | None = None
    # Ordinary turns use a delta; compaction continuations restore input_text in full.
    model_input_text: str | None = None
    # Optional baseline restored after compaction, excluding transient context.
    compaction_input_text: str | None = None


class DecisionProvider(Protocol):
    def generate_batch(
        self, requests: Sequence[DecisionRequest]
    ) -> Sequence[BatchDecisionResult]: ...


BatchDecisionResult = Mapping[str, Any] | Exception


class DecisionResult(dict):
    """Parsed native action with provider conversation data kept outside its schema."""

    def __init__(self, value, assistant_message):
        super().__init__(value)
        self.assistant_message = copy.deepcopy(assistant_message)


class ModelDecisionError(ValueError):
    """An invalid model action, returned to its existing session for correction."""

    def __init__(
        self, message: str, raw_output: str = "", assistant_message=None, *, feedback=None
    ) -> None:
        super().__init__(message)
        self.message = message
        # What the model is told; the full message names the provider and step for logs.
        self.feedback = feedback or message
        self.raw_output = raw_output
        self.assistant_message = copy.deepcopy(assistant_message)


class ProviderTransportError(RuntimeError):
    """A request failure without credential-bearing HTTP request/response objects."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class _InlineDecisionDispatcher:
    """Adapt immediate, synchronous scripted providers; network providers dispatch natively."""

    def __init__(self, provider):
        self.provider = provider
        self.results = {}
        self.counter = 0

    def submit_request(self, request):
        self.counter += 1
        handle = self.counter
        try:
            results = tuple(self.provider.generate_batch((request,)))
            self.results[handle] = (
                results[0]
                if len(results) == 1
                else ValueError("provider returned the wrong number of decisions")
            )
        except Exception as error:
            self.results[handle] = error
        return handle

    def collect_ready(self, handles=None, timeout_seconds=0):
        return {
            key: value for key, value in self.results.items() if handles is None or key in handles
        }

    def acknowledge_result(self, handle):
        self.results.pop(handle, None)

    def cancel_pending(self):
        pass


def dispatcher_for(provider):
    if callable(getattr(provider, "submit_request", None)):
        return provider
    return _InlineDecisionDispatcher(provider)
