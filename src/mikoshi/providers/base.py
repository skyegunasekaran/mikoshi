"""Concurrent remote inference, request contracts, and provider configuration."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping, Sequence

from mikoshi.contracts import (
    BATCH_ACTION,
    MAX_TOOL_CALLS,
    action_schemas,
)
from mikoshi.decisions import (
    BatchDecisionResult,
    DecisionRequest,
    ModelDecisionError,
    ProviderTransportError,
)
from mikoshi.decisions import (
    DecisionProvider as DecisionProvider,
)
from mikoshi.decisions import (
    DecisionResult as DecisionResult,
)
from mikoshi.providers.requests import (
    CompactionCheckpoint,
    RequestAttempt,
    RequestIdentity,
    RequestRecord,
    RequestState,
    TransportStats,
    snapshot_request,
)
from mikoshi.providers.settings import ProviderSettings
from mikoshi.providers.telemetry import TIMING_HISTORY_LIMIT, RequestTelemetry
from mikoshi.providers.usage import TokenMeter
from mikoshi.runtime.budgets import (
    DEFAULT_AGENT_LIMITS,
    DEFAULT_CONTEXT_WINDOW,
    BudgetExhausted,
    ContextWindowExhausted,
    ExecutionBudget,
    with_token_budget,
)

DEFAULT_PROVIDER_CONCURRENCY = 4
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 900

DEFAULT_MAX_TOKENS = DEFAULT_AGENT_LIMITS.output_tokens
DEFAULT_TEMPERATURE = None


@dataclass
class CompactionResult:
    """Internal checkpoint, never an executable decision or a runtime turn."""

    payload: dict[str, Any]
    message: Mapping[str, Any]


def parse_retry_after(value: str | None, *, now: float) -> float | None:
    """Parse a server delay without imposing a local retry policy."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                return None
            seconds = parsed.timestamp() - now
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


class RemoteDecisionProvider(ProviderSettings):
    """Completion-driven HTTP inference with caller-owned durable accounting.

    Only transport runs on the event-loop thread. Admission, settlement, output
    validation and callbacks run in the submitting/collecting scheduler thread.
    Completed decisions remain available until explicitly acknowledged.
    """

    provider_name: str
    api_key_env: str
    base_url: str
    endpoint: str
    transport_error: type[ProviderTransportError]
    reasoning_efforts: tuple[str, ...]
    output_limit_key: str
    input_keys: tuple[str, ...]
    native_fields: tuple[str, ...] = ()
    native_blocks_key: str | None = None

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        concurrency: int = DEFAULT_PROVIDER_CONCURRENCY,
        timeout_seconds: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
        temperature: float | None = DEFAULT_TEMPERATURE,
        top_p: float | None = None,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        http_transport: Any = None,
        reasoning_effort: str | None = None,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW,
        tokenizer_path: str | None = None,
        budget: ExecutionBudget | None = None,
    ) -> None:
        if not model_id or not model_id.strip():
            raise ValueError(f"model_id is required for {self.provider_name}")
        resolved_key = self.environment_api_key() if api_key is None else api_key
        if not resolved_key.strip():
            raise ValueError(f"{self.api_key_env} is required for {self.provider_name}")
        for name, value in (
            ("max_tokens", max_tokens),
            ("concurrency", concurrency),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and greater than zero")
        if temperature is not None and (not math.isfinite(temperature) or temperature < 0):
            raise ValueError("temperature must be finite and nonnegative")
        if top_p is not None and (not math.isfinite(top_p) or not 0 <= top_p <= 1):
            raise ValueError("top_p must be between 0 and 1")
        if reasoning_effort is not None and reasoning_effort not in self.reasoning_efforts:
            raise ValueError(f"reasoning_effort must be one of {self.reasoning_efforts}")
        if (
            not isinstance(context_window_tokens, int)
            or isinstance(context_window_tokens, bool)
            or context_window_tokens <= max_tokens
        ):
            raise ValueError("context_window_tokens must exceed max_tokens")
        self.context_window_tokens = context_window_tokens
        self._token_meter = TokenMeter(tokenizer_path)
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.timeout_seconds = timeout_seconds
        # Each transport attempt is recorded separately, including ambiguous failures.
        self.budget = budget
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self._api_key = resolved_key.strip()
        self._http_transport = http_transport
        self._progress_callback = progress_callback
        self._evidence_callback = None
        self._evidence_errors: list[str] = []
        self._evidence_error_count = 0
        self._decision_request_ids: dict[str, list[str]] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._io_tasks: dict[int, asyncio.Task] = {}
        self._loop_ready = threading.Event()
        self._client: Any = None
        self._semaphore: asyncio.Semaphore | None = None
        self._owner_thread = threading.get_ident()
        self._records: dict[int, RequestRecord] = {}
        self._telemetry = RequestTelemetry()
        self._timing_epoch = uuid.uuid4().hex
        self._closed = False
        self._cancelling = False
        self._compaction_pending: dict[tuple[str, str], CompactionCheckpoint] = {}
        self._generation_index = 0
        self._metrics: dict[str, Any] = {
            "requests": 0,
            "batch_calls": 0,
            "batch_sizes": [],
            "generation_attempts": 0,
            "transport_attempts": 0,
            "transport_failed_requests": 0,
            "rate_limit_responses": 0,
            "failed_requests": 0,
            "compactions": 0,
            "generation_seconds": 0.0,
            "transport_seconds": 0.0,
            "queue_seconds": 0.0,
            "validation_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "uncached_input_tokens_measured": 0,
            "cache_read_tokens_measured": 0,
            "cache_creation_tokens_measured": 0,
            "cache_measurement_counts": {},
            "reasoning_tokens_measured": 0,
            "final_output_tokens_measured": 0,
            "reasoning_measurement_counts": {},
            "final_output_measurement_counts": {},
        }

    @property
    def metrics(self) -> Mapping[str, Any]:
        metrics = copy.deepcopy(self._metrics)
        metrics.update(self._telemetry.snapshot())
        metrics["batch_sizes_truncated"] = metrics["batch_calls"] > len(metrics["batch_sizes"])
        seconds = metrics["generation_seconds"]
        metrics["output_tokens_per_second"] = (
            round(metrics["output_tokens"] / seconds, 3) if seconds > 0 else None
        )
        cache_measurements = metrics["cache_measurement_counts"]
        for category in ("uncached_input", "cache_read", "cache_creation"):
            metrics[f"{category}_tokens"] = (
                metrics[f"{category}_tokens_measured"]
                if cache_measurements and not cache_measurements.get("unavailable")
                else None
            )
        attributed_input = (
            metrics["uncached_input_tokens"]
            + metrics["cache_read_tokens"]
            + metrics["cache_creation_tokens"]
            if metrics["uncached_input_tokens"] is not None
            else 0
        )
        metrics["cache_hit_rate"] = (
            metrics["cache_read_tokens"] / attributed_input if attributed_input else None
        )
        measurements = metrics["reasoning_measurement_counts"]
        metrics["reasoning_tokens"] = (
            metrics["reasoning_tokens_measured"]
            if measurements and not measurements.get("unavailable")
            else None
        )
        measurements = metrics["final_output_measurement_counts"]
        metrics["final_output_tokens"] = (
            metrics["final_output_tokens_measured"]
            if measurements and not measurements.get("unavailable")
            else None
        )
        metrics["evidence_errors"] = list(self._evidence_errors)
        metrics["evidence_error_count"] = self._evidence_error_count
        metrics["evidence_errors_truncated"] = self._evidence_error_count > len(
            self._evidence_errors
        )
        return metrics

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("provider admission and collection require the owner thread")

    def _loop_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.run_until_complete(self._loop.shutdown_default_executor())
            self._loop.close()

    def _start_loop(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self.provider_name} provider is closed")
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=self.provider_name.lower()
            )
            self._executor.submit(self._loop_main)
            self._loop_ready.wait()

    def _run(self, factory: Callable[[], Any]) -> Any:
        self._start_loop()
        return asyncio.run_coroutine_threadsafe(factory(), self._loop).result()

    def billing_request(self, payload):
        return {"max_tokens": self.output_token_limit(payload), "model": self.model_id}

    def output_token_limit(self, payload):
        return payload.get(self.output_limit_key)

    def response_body_for_request(self, body, payload):
        """Decode a response, allowing adapters to distinguish summary requests."""
        return self.response_body(body)

    def billing_response(self, response):
        """Return a response (or {"usage": ...}) with reconcilable token categories."""
        return response

    def submitted_tool_results(self, payload):
        """Exact JSON pointers and call IDs for tool replies present on the wire."""
        return []

    def count_tokens(self, text):
        return self._token_meter.count(text)

    def request_headers(self):
        return {"Authorization": f"Bearer {self._api_key}"}

    def remember_response(self, request, payload, message):
        """Retain native continuation state, including invalid authored decisions."""

    def compaction_continuation(self, request, payload, message):
        """Restore authoritative context before inference can produce an action."""
        raise ValueError("provider does not support a compaction checkpoint")

    def request_endpoint(self, payload):
        return self.endpoint

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import httpx
        except ImportError as error:
            raise RuntimeError(
                f"{self.provider_name} requires httpx. Install the project dependencies."
            ) from error
        headers = self.request_headers()
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            trust_env=False,
            timeout=self.timeout_seconds,
            limits=httpx.Limits(
                max_connections=self.concurrency,
                max_keepalive_connections=self.concurrency,
            ),
            transport=self._http_transport,
        )
        self._semaphore = asyncio.Semaphore(self.concurrency)

    async def _cancel_on_loop(self) -> None:
        tasks = tuple(self._io_tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def cancel_pending(self) -> None:
        """Quiesce transport and retain completed actions and account for interrupted requests.

        Cancelling the asyncio task rather than its concurrent Future preserves a
        response that won the race with cancellation. Accounting is finalized on
        this caller thread, including unknown usage on interruption.
        """
        self._assert_owner()
        self._cancelling = True
        try:
            self.collect_ready()
            if not self._closed and self._executor is not None:
                self._run(self._cancel_on_loop)
            self.collect_ready()
        finally:
            self._cancelling = False

    def close(self) -> None:
        self._assert_owner()
        if self._closed:
            return
        try:
            self.cancel_pending()
            if self._client is not None:
                self._run(self._client.aclose)
        finally:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            self._closed = True
            self._compaction_pending.clear()
            if self.budget is not None:
                self.budget.save()

    def __enter__(self) -> RemoteDecisionProvider:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _redact(self, value: Any) -> str:
        return str(value).replace(self._api_key, "[REDACTED]")

    def _emit_progress(self, kind: str, **details: Any) -> None:
        if self._progress_callback is not None:
            try:
                self._progress_callback(
                    {"kind": kind, "provider": self.provider_name.lower(), **details}
                )
            except Exception:
                pass

    def set_evidence_callback(self, callback) -> None:
        """Set a best-effort observer that may be called from the I/O thread.

        The observer must be thread-safe and must not mutate lifecycle state.
        Request headers and client objects are never supplied to it.
        """
        self._assert_owner()
        if callback is not None and not callable(callback):
            raise TypeError("evidence callback must be callable or None")
        self._evidence_callback = callback

    def _evidence_value(self, value):
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, Mapping):
            return {
                self._redact(key) if isinstance(key, str) else key: self._evidence_value(child)
                for key, child in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._evidence_value(child) for child in value]
        return copy.deepcopy(value)

    def _emit_evidence(self, kind, **details):
        callback = self._evidence_callback
        if callback is None:
            return
        try:
            if kind == "provider_request_submitted":
                details["tool_result_paths"] = self.submitted_tool_results(details["payload"])
            preserved = self._evidence_value(details)
            callback(
                {
                    "kind": kind,
                    "provider": self.provider_name.lower(),
                    "model": self.model_id,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "redacted": preserved != details,
                    **preserved,
                }
            )
        except Exception as error:
            self._evidence_error_count += 1
            self._evidence_errors.append(
                self._redact(f"{kind}: {type(error).__name__}: {error}")[:1000]
            )
            del self._evidence_errors[:-TIMING_HISTORY_LIMIT]

    def _response_evidence(self, response, stats, *, kind="provider_response_received"):
        try:
            body = response.json()
            # Some JSON decoders accept NaN/Infinity even though they are not
            # JSON values. Preserve those HTTP bodies as text evidence.
            json.dumps(body, allow_nan=False)
        except (ValueError, TypeError):
            content = {"raw_text": response.text}
            body = None
        else:
            content = {"body": body}
        self._emit_evidence(
            kind,
            **stats.evidence,
            submitted=stats.submitted,
            status_code=response.status_code,
            content_type=response.headers.get("content-type"),
            provider_response_id=body.get("id")
            if isinstance(body, Mapping) and isinstance(body.get("id"), str)
            else None,
            **content,
        )

    async def _request(self, payload: Mapping[str, Any], stats: TransportStats) -> Any:
        """Send exactly one HTTP request, retaining ambiguous usage for accounting."""
        import httpx

        timeout_seconds = stats.timeout_seconds
        queued = time.perf_counter()
        started = None
        try:
            async with self._semaphore:
                started = time.perf_counter()
                stats.queue_seconds += started - queued
                if self.budget is not None:
                    remaining = self.budget.remaining_seconds
                    if remaining is not None:
                        if remaining <= 0:
                            raise BudgetExhausted("wall_time_exhausted")
                        timeout_seconds = min(timeout_seconds, remaining)
                stats.transport_attempts += 1
                stats.submitted = True
                self._emit_evidence(
                    "provider_request_submitted",
                    **stats.evidence,
                    submitted=True,
                    payload=payload,
                )
                response = await asyncio.wait_for(
                    self._client.post(
                        self.request_endpoint(payload), json=payload, timeout=timeout_seconds
                    ),
                    timeout=timeout_seconds,
                )
                stats.response_received = True
                self._response_evidence(response, stats)
            stats.status_codes.append(response.status_code)
            if response.is_success:
                return response
            raise self.transport_error(
                f"{self.provider_name} request failed: HTTP {response.status_code}: {self._redact(response.text)[:500]}",
                status_code=response.status_code,
                retry_after_seconds=parse_retry_after(
                    response.headers.get("Retry-After"), now=time.time()
                ),
            ) from None
        except (httpx.TransportError, asyncio.TimeoutError) as error:
            if self.budget is not None and self.budget.remaining_seconds == 0:
                raise BudgetExhausted("wall_time_exhausted") from error
            raise self.transport_error(
                f"{self.provider_name} request failed: {type(error).__name__}: {self._redact(error)[:500]}"
            ) from None
        finally:
            if started is not None:
                stats.transport_seconds += time.perf_counter() - started

    def input_payload(self, request):
        raise NotImplementedError

    def request_payload(self, prepared, tokens, handle):
        raise NotImplementedError

    def response_body(self, body):
        return body

    def raw_decision_text(self, message):
        if message.get("tool_calls") is not None:
            return json.dumps(message["tool_calls"], ensure_ascii=False)
        return message.get("content") or ""

    def parse_message(self, request, message):
        return _parse_tool_calls(request, message)

    def _prepare_entry(self, request: DecisionRequest, handle: int) -> RequestRecord:
        entry = RequestRecord(
            handle,
            snapshot_request(request),
            request.metadata.get("decision_id") or uuid.uuid4().hex,
        )
        try:
            if request.conversation_id and any(
                record.request.conversation_id == request.conversation_id
                and record.state in {RequestState.PREPARED, RequestState.INFLIGHT}
                for record in self._records.values()
            ):
                raise ValueError("A conversation can have only one in-flight request")
            tokens = request.max_tokens if request.max_tokens is not None else self.max_tokens
            if type(tokens) is not int or tokens < 1:
                raise ValueError("request max_tokens must be an integer of at least 1")
            tokens = min(self.max_tokens, tokens)
            prepared = with_token_budget(request, tokens, self.context_window_tokens)
            entry.prepared = prepared
            key = _compaction_key(prepared)
            pending = self._compaction_pending.get(key)
            if pending:
                if pending.request != snapshot_request(prepared):
                    raise ValueError(
                        "Compaction continuation requires its original pending request"
                    )
                entry.payload = copy.deepcopy(pending.payload)
                entry.compaction_checkpoints = copy.deepcopy(pending.checkpoints)
            else:
                entry.payload = self.request_payload(prepared, tokens, handle)
        except ContextWindowExhausted as error:
            entry.reject(error)
        except (TypeError, ValueError) as error:
            entry.reject(ValueError(self._redact(error)))
        return entry

    def _new_entry(self, request: DecisionRequest) -> RequestRecord:
        handle = self._generation_index
        entry = self._prepare_entry(request, handle)
        self._generation_index += 1
        self._metrics["requests"] += 1
        if entry.state is RequestState.READY:
            self._metrics["failed_requests"] += 1
        self._records[handle] = entry
        return entry

    async def _perform_request(self, entry: RequestRecord) -> Any:
        handle = entry.handle
        self._io_tasks[handle] = asyncio.current_task()
        try:
            await self._ensure_client()
            return await self._request(entry.payload, entry.active_attempt.stats)
        except asyncio.CancelledError:
            self._request_failure_evidence(entry, BudgetExhausted("request_interrupted"))
            return BudgetExhausted("request_interrupted")
        except BaseException as error:
            self._request_failure_evidence(entry, error)
            return error
        finally:
            entry.active_attempt.stats.finished = time.perf_counter()
            self._io_tasks.pop(handle, None)

    def _request_failure_evidence(self, entry, error):
        self._emit_evidence(
            "provider_request_failed",
            **entry.active_attempt.stats.evidence,
            submitted=entry.active_attempt.stats.submitted,
            response_received=entry.active_attempt.stats.response_received,
            error_type=type(error).__name__,
            error=self._redact(error),
        )

    def _dispatch_entries(self, entries: Sequence[RequestRecord]) -> None:
        entries = [
            entry
            for entry in entries
            if entry.state in (RequestState.PREPARED, RequestState.SUSPENDED)
        ]
        if not entries:
            return
        identities = []
        for entry in entries:
            decision_id = entry.decision_id
            previous = self._decision_request_ids.setdefault(decision_id, [])
            identity = RequestIdentity(
                uuid.uuid4().hex,
                decision_id,
                previous[-1] if previous else None,
                self.request_endpoint(entry.payload),
                **{
                    key: entry.request.metadata.get(key)
                    for key in ("agent_id", "activation_id", "activation_step")
                },
            )
            identities.append(identity)
            previous.append(identity.request_id)
            self._emit_evidence(
                "provider_request_prepared",
                **identity.evidence(),
                submitted=False,
                payload=entry.payload,
            )
        try:
            budget_request_ids = (
                self.budget.begin_requests(
                    [self.billing_request(entry.payload) for entry in entries],
                    [
                        {**entry.request.metadata, **identity.evidence()}
                        for entry, identity in zip(entries, identities)
                    ],
                )
                if self.budget is not None
                else [None] * len(entries)
            )
        except BudgetExhausted as error:
            for entry, identity in zip(entries, identities):
                entry.suspend(error)
                self._emit_evidence(
                    "provider_request_failed",
                    **identity.evidence(),
                    submitted=False,
                    response_received=False,
                    error_type=type(error).__name__,
                    error=self._redact(error),
                )
            return
        self._start_loop()
        self._metrics["generation_attempts"] += len(entries)
        self._emit_progress(
            "inference_started",
            model_id=self.model_id,
            batch_size=len(entries),
            stages=sorted({entry.request.stage for entry in entries}),
        )
        for entry, identity, budget_request_id in zip(entries, identities, budget_request_ids):
            timeout = self.timeout_seconds
            if self.budget is not None and self.budget.remaining_seconds is not None:
                timeout = min(timeout, max(0.001, self.budget.remaining_seconds))
            entry.dispatch(
                RequestAttempt(
                    identity,
                    budget_request_id,
                    time.perf_counter(),
                    TransportStats(identity.evidence(), timeout),
                )
            )
            entry.active_attempt.future = asyncio.run_coroutine_threadsafe(
                self._perform_request(entry), self._loop
            )

    def submit_request(self, request: DecisionRequest) -> int:
        """Admit one request and return its process-local result handle."""
        self._assert_owner()
        if self._closed:
            raise RuntimeError(f"{self.provider_name} provider is closed")
        entry = self._new_entry(request)
        self._dispatch_entries([entry])
        return entry.handle

    def collect_ready(
        self,
        handles: Sequence[int] | None = None,
        timeout_seconds: float = 0,
    ) -> dict[int, BatchDecisionResult]:
        """Return completed decisions without waiting for an unfinished sibling.

        Ready results remain available until acknowledged. All settlement,
        validation and progress callbacks happen here on the owner thread.
        """
        self._assert_owner()
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("timeout_seconds must be finite and nonnegative")
        selected = list(self._records) if handles is None else list(handles)
        entries = [self._records[handle] for handle in selected]
        futures = [
            entry.active_attempt.future for entry in entries if entry.state is RequestState.INFLIGHT
        ]
        if futures and not any(entry.available for entry in entries):
            wait(futures, timeout=timeout_seconds, return_when=FIRST_COMPLETED)
        completed = [
            entry
            for entry in entries
            if entry.state is RequestState.INFLIGHT and entry.active_attempt.future.done()
        ]
        completed.sort(key=lambda entry: entry.active_attempt.stats.finished or float("inf"))
        self._collect_entries(completed)
        available = [entry for entry in entries if entry.available]
        available.sort(
            key=lambda entry: (
                (entry.attempt.stats.finished or float("-inf")) if entry.attempt else float("-inf")
            )
        )
        return {entry.handle: entry.result for entry in available}

    def _collect_entries(self, entries: Sequence[RequestRecord]) -> None:
        if not entries:
            return
        responses = [entry.active_attempt.future.result() for entry in entries]
        if self.budget is not None:
            billing = []
            for response in responses:
                try:
                    billing.append(
                        response
                        if isinstance(response, BaseException)
                        else self.billing_response(response)
                    )
                except (AttributeError, KeyError, TypeError, ValueError):
                    billing.append(None)  # Usage cannot be reconciled for this response.
            self.budget.settle(
                [entry.active_attempt.budget_request_id for entry in entries], billing
            )
            for entry in entries:
                self._emit_evidence(
                    "provider_usage_recorded",
                    **entry.active_attempt.identity.evidence(),
                    budget_entry_id=entry.active_attempt.budget_request_id,
                    entry=dict(self.budget.entries[entry.active_attempt.budget_request_id]),
                )
        results = self._finalize_responses(entries, responses)
        continuations = []
        for entry, result in zip(entries, results):
            request = entry.prepared
            key = _compaction_key(request)
            if isinstance(result, CompactionResult):
                checkpoint = {
                    "request_id": entry.active_attempt.identity.request_id,
                    "decision_id": entry.decision_id,
                    "assistant_message": copy.deepcopy(result.message),
                    "restored_state": request.compaction_input_text or request.input_text,
                    "restored_instructions": request.instructions,
                }
                checkpoints = [*entry.compaction_checkpoints, checkpoint]
                self._compaction_pending[key] = CompactionCheckpoint(
                    snapshot_request(request), copy.deepcopy(result.payload), checkpoints
                )
                entry.continue_after_compaction(
                    result.payload,
                    checkpoints,
                    BudgetExhausted("request_interrupted") if self._cancelling else None,
                )
                self._emit_progress(
                    "compaction_completed",
                    request_name=request.name,
                    metadata=dict(request.metadata),
                    checkpoint=checkpoint,
                )
                if not self._cancelling:
                    continuations.append(entry)
                continue
            result.request_id = entry.active_attempt.identity.request_id
            result.decision_id = entry.decision_id
            result.request_ids = tuple(self._decision_request_ids[entry.decision_id])
            if getattr(result, "assistant_message", None) is not None:
                if entry.compaction_checkpoints:
                    result.assistant_message["compaction_checkpoints"] = copy.deepcopy(
                        entry.compaction_checkpoints
                    )
                self._compaction_pending.pop(key, None)
            if isinstance(result, BudgetExhausted):
                entry.suspend(result)
                entry.active_attempt.future = None
            else:
                entry.complete(result)
        self._dispatch_entries(continuations)

    def acknowledge_result(self, handle: int) -> None:
        self._assert_owner()
        entry = self._records[handle]
        if entry.state is not RequestState.READY:
            raise ValueError("only completed decisions can be acknowledged")
        del self._records[handle]

    def generate_batch(self, requests: Sequence[DecisionRequest]) -> Sequence[BatchDecisionResult]:
        """Ordered synchronous convenience API over the same request records.

        Batch admission remains atomic; completion-driven callers use submit and
        collect to refill each available execution slot independently.
        """
        self._assert_owner()
        if self._closed:
            raise RuntimeError(f"{self.provider_name} provider is closed")
        if not requests:
            return ()
        self._metrics["batch_calls"] += 1
        self._metrics["batch_sizes"].append(len(requests))
        del self._metrics["batch_sizes"][:-TIMING_HISTORY_LIMIT]
        entries = [self._new_entry(request) for request in requests]
        self._dispatch_entries(entries)
        while any(entry.state is RequestState.INFLIGHT for entry in entries):
            try:
                self.collect_ready(
                    [entry.handle for entry in entries if entry.state is RequestState.INFLIGHT],
                    timeout_seconds=1,
                )
            except BaseException:
                self.cancel_pending()
                raise
        suspended = [entry for entry in entries if entry.state is RequestState.SUSPENDED]
        if suspended:
            raise suspended[0].result
        results = tuple(entry.result for entry in entries)
        for entry in entries:
            self.acknowledge_result(entry.handle)
        return results

    def _finalize_responses(self, entries, responses):
        requests = [entry.request for entry in entries]
        results = [None] * len(entries)
        payloads = [entry.payload for entry in entries]
        prepared_requests = [entry.prepared for entry in entries]
        stats = [entry.active_attempt.stats for entry in entries]
        seconds = max(
            (entry.active_attempt.stats.finished or time.perf_counter()) for entry in entries
        ) - min(entry.active_attempt.started for entry in entries)
        input_counts, output_counts = [], []
        output_errors = []
        usage_measurements = []
        finish_reasons = []
        invalid = 0
        validation_started = time.perf_counter()
        for index, (response, request_stats) in enumerate(zip(responses, stats)):
            output_error = None
            measurement = {
                "input_tokens": None,
                "uncached_input_tokens": None,
                "cache_read_tokens": None,
                "cache_creation_tokens": None,
                "cache_measurement": "unavailable",
                "completion_tokens": None,
                "reasoning_tokens": None,
                "reasoning_measurement": "unavailable",
                "final_output_tokens": None,
                "final_output_measurement": "unavailable",
            }
            finish_reason = None
            for metric in (
                "transport_attempts",
                "transport_seconds",
                "queue_seconds",
            ):
                self._metrics[metric] += getattr(request_stats, metric)
            self._metrics["rate_limit_responses"] += request_stats.status_codes.count(429)
            input_count = output_count = 0
            if isinstance(response, BudgetExhausted):
                results[index] = response
            elif isinstance(response, BaseException):
                invalid += 1
                self._metrics["transport_failed_requests"] += 1
                results[index] = self.transport_error(
                    self._redact(response),
                    getattr(response, "status_code", None),
                    getattr(response, "retry_after_seconds", None),
                )
            else:
                content = ""
                message = None
                try:
                    body = self.response_body_for_request(response.json(), payloads[index])
                    # Usage is telemetry, not an action or continuation boundary.
                    # Preserve valid output even when accounting is unavailable.
                    measurement["raw_usage"] = body.get("native_usage", body.get("usage"))
                    try:
                        measured = self._token_meter.measure(body)
                        usage = body.get("usage") or {}
                        counts = (usage.get("prompt_tokens"), usage.get("completion_tokens"))
                        if any(type(n) is not int or n < 0 for n in counts):
                            raise ValueError("missing or invalid token totals")
                        for key, value in measured.items():
                            if (
                                key.endswith("_tokens")
                                and value is not None
                                and (type(value) is not int or value < 0)
                            ):
                                raise ValueError("invalid token measurement")
                        measurement = measured
                        input_count, output_count = counts
                    except (
                        ValueError,
                        TypeError,
                        KeyError,
                        IndexError,
                        AttributeError,
                        OverflowError,
                    ) as error:
                        measurement["error"] = self._redact(error)[:500]
                    choice = body["choices"][0]
                    finish_reason = choice.get("finish_reason")
                    message = choice.get("message")
                    if not isinstance(message, Mapping):
                        raise ValueError("response has no assistant message")
                    if (
                        self.request_endpoint(payloads[index]) != self.endpoint
                        and finish_reason != "compaction"
                    ):
                        message = None
                        raise ValueError(
                            "compaction endpoint did not return a compaction checkpoint"
                        )
                    content = self.raw_decision_text(message)
                    if finish_reason == "compaction":
                        checkpoint_message = message
                        # Malformed checkpoints are transport/protocol failures, not
                        # authored actions that could receive a tool-result reply.
                        message = None
                        if entries[index].compaction_checkpoints:
                            raise ContextWindowExhausted(
                                "Compaction repeated before an action; restored context does not fit"
                            )
                        continuation = self.compaction_continuation(
                            prepared_requests[index], payloads[index], checkpoint_message
                        )
                        results[index] = CompactionResult(continuation, checkpoint_message)
                        self._metrics["compactions"] += 1
                    else:
                        if any(
                            item.get("type") == "compaction"
                            for item in message.get(self.native_blocks_key, [])
                        ):
                            self._emit_progress(
                                "compaction_rejected",
                                request_name=requests[index].name,
                                metadata=dict(requests[index].metadata),
                                assistant_message=copy.deepcopy(message),
                            )
                            message = None
                            raise ValueError(
                                "compaction must complete and restore context before action generation"
                            )
                        if self.native_blocks_key is not None:
                            call_ids = [call.get("id") for call in message.get("tool_calls", [])]
                            if any(
                                not isinstance(value, str) or not value for value in call_ids
                            ) or len(set(call_ids)) != len(call_ids):
                                # Native calls cannot fall back to text error feedback.
                                # Keep the prior session and any pending checkpoint retryable.
                                message = None
                                raise ValueError("tool calls require unique nonempty string IDs")
                        self.remember_response(prepared_requests[index], payloads[index], message)
                        if finish_reason != "tool_calls":
                            raise ValueError(_finish_feedback(finish_reason))
                        results[index] = DecisionResult(
                            self.parse_message(requests[index], message),
                            _conversation_message(message, self.native_fields),
                        )
                except ContextWindowExhausted as error:
                    results[index] = error
                except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
                    invalid += 1
                    output_error = self._redact(error)[:2000]
                    self._emit_progress(
                        "output_validation_failed",
                        request_name=requests[index].name,
                        stage=requests[index].stage,
                        metadata=dict(requests[index].metadata),
                        error=output_error,
                    )
                    if isinstance(message, Mapping):
                        results[index] = ModelDecisionError(
                            f"{self.provider_name} returned an invalid {requests[index].name} decision: {output_error}",
                            self._redact(content) if isinstance(content, str) else "",
                            _conversation_message(message, self.native_fields),
                            feedback=output_error,
                        )
                    else:
                        self._metrics["transport_failed_requests"] += 1
                        results[index] = self.transport_error(
                            f"{self.provider_name} returned an invalid response envelope: {output_error}"
                        )
            usage_measurements.append(measurement)
            finish_reasons.append(finish_reason)
            if measurement["uncached_input_tokens"] is not None:
                self._metrics["uncached_input_tokens_measured"] += measurement[
                    "uncached_input_tokens"
                ]
                self._metrics["cache_read_tokens_measured"] += measurement["cache_read_tokens"]
                self._metrics["cache_creation_tokens_measured"] += measurement[
                    "cache_creation_tokens"
                ]
            cache_counts = self._metrics["cache_measurement_counts"]
            cache_source = measurement["cache_measurement"]
            cache_counts[cache_source] = cache_counts.get(cache_source, 0) + 1
            for measurement_kind in ("reasoning", "final_output"):
                value = measurement[f"{measurement_kind}_tokens"]
                if value is not None:
                    self._metrics[f"{measurement_kind}_tokens_measured"] += value
                counts = self._metrics[f"{measurement_kind}_measurement_counts"]
                source = measurement[f"{measurement_kind}_measurement"]
                counts[source] = counts.get(source, 0) + 1
            input_counts.append(input_count)
            output_counts.append(output_count)
            output_errors.append(output_error)
        validation_seconds = time.perf_counter() - validation_started
        self._metrics["validation_seconds"] += validation_seconds
        self._metrics["input_tokens"] += sum(input_counts)
        self._metrics["output_tokens"] += sum(output_counts)
        event = {
            "call_type": f"{self.provider_name.lower()}_batch",
            "timing_epoch": self._timing_epoch,
            "request_intervals": [
                [entry.active_attempt.started, entry.active_attempt.stats.finished]
                for entry in entries
            ],
            "batch_size": len(entries),
            "concurrency": self.concurrency,
            "request_names": [request.name for request in requests],
            "request_ids": [entry.active_attempt.identity.request_id for entry in entries],
            "decision_ids": [entry.decision_id for entry in entries],
            "request_stages": [request.stage for request in requests],
            "request_metadata": [dict(request.metadata) for request in prepared_requests],
            "max_tokens": [self.output_token_limit(payload) for payload in payloads],
            "request_endpoints": [self.request_endpoint(payload) for payload in payloads],
            "advertised_output_tokens": [
                request.metadata["output_token_budget"] for request in prepared_requests
            ],
            "advertised_input_tokens": [
                request.metadata["input_token_budget"] for request in prepared_requests
            ],
            "input_payload_token_estimates": [
                self._token_meter.count(
                    json.dumps(
                        {key: payload[key] for key in self.input_keys if key in payload},
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                )
                for payload in payloads
            ],
            "input_payload_measurement": f"retokenized_{'_and_'.join(self.input_keys)}_json",
            "usage_measurements": usage_measurements,
            "generation_seconds": seconds,
            "validation_seconds": validation_seconds,
            "input_tokens": sum(input_counts),
            "input_token_counts": input_counts,
            "output_tokens": sum(output_counts),
            "output_token_counts": output_counts,
            "output_errors": output_errors,
            "finish_reasons": finish_reasons,
            "chosen_actions": [
                result.get("action") if isinstance(result, Mapping) else None for result in results
            ],
            "valid_requests": sum(isinstance(result, Mapping) for result in results),
            "invalid_requests": invalid,
            "transport": [item.snapshot() for item in stats],
        }
        completed = {entry.handle for entry in entries}
        pending = {}
        for record in self._records.values():
            if record.state is RequestState.INFLIGHT and record.handle not in completed:
                pending.setdefault(str(record.request.stage), []).append(
                    record.active_attempt.started
                )
        self._telemetry.record(event, pending=pending)
        self._metrics["generation_seconds"] = self._telemetry.generation.seconds
        for index, entry in enumerate(entries):
            self._emit_evidence(
                "provider_request_completed",
                **entry.active_attempt.identity.evidence(),
                submitted=entry.active_attempt.stats.submitted,
                timing=event,
                timing_index=index,
            )
        self._emit_progress(
            "inference_completed",
            model_id=self.model_id,
            batch_size=len(entries),
            generation_seconds=round(seconds, 3),
            usage_measurements=usage_measurements,
            output_token_limits=[self.output_token_limit(payload) for payload in payloads],
            output_tokens=sum(output_counts),
            valid_requests=sum(isinstance(result, Mapping) for result in results),
            invalid_requests=invalid,
        )
        self._metrics["failed_requests"] += sum(
            isinstance(result, Exception) and not isinstance(result, BudgetExhausted)
            for result in results
        )
        return results


def _conversation_message(message, native_fields):
    """Retain supported assistant fields, including reasoning only when returned."""
    return {
        "role": "assistant",
        **{
            key: copy.deepcopy(message[key])
            for key in (
                "content",
                "tool_calls",
                "reasoning_content",
                *native_fields,
                "context_tokens",
                "compaction_checkpoints",
            )
            if key in message
        },
    }


def _finish_feedback(finish_reason):
    """Model-facing explanation of a response that ended the wrong way."""
    if finish_reason == "stop":
        return "The response ended without a tool call. Each turn needs at least one tool call."
    if finish_reason == "length":
        return (
            "The response reached the output limit before it finished. "
            "No tool calls from that response were executed. "
            "Retry with fewer calls and smaller tool arguments. Split large file writes "
            "into smaller writes or targeted edits, leaving room for reasoning."
        )
    return f"The response ended with finish_reason={finish_reason!r} instead of a tool call."


def action_tools(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Translate canonical runtime contracts into native function declarations."""
    tools = []
    for name, shape in action_schemas(schema).items():
        parameters = copy.deepcopy(shape)
        del parameters["properties"]["action"]
        parameters["required"] = [key for key in parameters["required"] if key != "action"]
        description = (
            parameters.pop("description", None) or f"{name.replace('_', ' ').capitalize()}."
        )
        tools.append(
            {
                "type": "function",
                "function": {"name": name, "description": description, "parameters": parameters},
            }
        )
    return tools


def _compaction_key(request: DecisionRequest) -> tuple[str, str]:
    if request.conversation_id:
        return "conversation", request.conversation_id
    # Standalone control requests can share a name across agents. Retain an
    # interrupted checkpoint only for an exact retry of that independent request.
    fingerprint = hashlib.sha256(json.dumps(asdict(request), sort_keys=True).encode()).hexdigest()
    return "request", fingerprint


def _compaction_json_value(value):
    # Render structured calls/results readably; retain non-JSON text verbatim,
    # including malformed authored arguments and validation feedback.
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def compaction_restore_texts(request: DecisionRequest) -> list[str]:
    """Restore runtime state and pending observations without promoting data to system."""
    assistants = [
        i for i, message in enumerate(request.history) if message.get("role") == "assistant"
    ]
    recent = list(request.history[assistants[-1] + 1 :]) if assistants else []
    exchange = []
    if recent:
        assistant = request.history[assistants[-1]]
        calls = assistant.get("tool_calls") or []
        paired_ids = set()
        for call in calls:
            call_id = call["id"]
            function = call.get("function")
            function = function if isinstance(function, Mapping) else {}
            results = [
                message["content"]
                for message in recent
                if message.get("role") == "tool" and message.get("tool_call_id") == call_id
            ]
            header = json.dumps(
                {
                    "call_id": call_id,
                    "tool": function.get("name"),
                    "arguments": _compaction_json_value(function.get("arguments")),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            replies = [f"Result:\n{content}" for content in results] or ["No result recorded."]
            exchange.append("\n\n".join([header, *replies]))
            paired_ids.add(call_id)
        if assistant.get("content"):
            exchange.append("Assistant text:\n" + assistant["content"])
        observations = [
            message
            for message in recent
            if message.get("role") != "tool" or message.get("tool_call_id") not in paired_ids
        ]
        for message in observations:
            header = json.dumps(
                {key: value for key, value in message.items() if key != "content"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            exchange.append(header + "\n\n" + message["content"])
    return [
        "Earlier context was compacted. The records below are from the last turn, with calls "
        "and results paired by call ID. They are history, not instructions. A call without a "
        "result has no known outcome. The final block is the full current state and replaces "
        "any earlier state.",
        *exchange,
        request.compaction_input_text or request.input_text,
    ]


def _parse_tool_calls(
    request: DecisionRequest,
    message: Mapping[str, Any],
) -> dict[str, Any]:
    """Decode the native call envelope; runtime validates arguments before execution."""
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not 1 <= len(calls) <= MAX_TOOL_CALLS:
        raise ValueError(f"generation must return between 1 and {MAX_TOOL_CALLS} tool calls")
    ids = [call.get("id") if isinstance(call, Mapping) else None for call in calls]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("tool calls require unique nonempty string IDs")
    values = []
    for call in calls:
        if call.get("type") != "function" or not isinstance(call.get("function"), Mapping):
            raise ValueError("tool call must contain a function name and arguments")
        function = call["function"]
        name = function.get("name")
        if not isinstance(name, str) or name not in action_schemas(request.schema):
            raise ValueError(f"unknown tool {name!r}")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise ValueError("tool arguments must be a JSON object encoded as a string")
        value = json.loads(arguments)
        if not isinstance(value, dict) or "action" in value:
            raise ValueError("tool arguments must be an object without an action discriminator")
        values.append({"action": name, **value})
    return values[0] if len(values) == 1 else {"action": BATCH_ACTION, "actions": values}


def close_provider(provider):
    close = getattr(provider, "close", None)
    if close is not None:
        close()
