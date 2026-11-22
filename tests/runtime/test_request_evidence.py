"""Request evidence preserves observed transport and accounting boundaries."""

import asyncio
import json
import threading

import httpx
import pytest
from support.provider import MockDecisionProvider, tool_calls

from mikoshi.contracts import action_schema
from mikoshi.providers.base import DecisionRequest, ProviderTransportError
from mikoshi.providers.openai import OpenAIDecisionProvider
from mikoshi.runtime.budgets import BudgetExhausted, ExecutionBudget


def request(decision="decision-1", agent="peer"):
    return DecisionRequest(
        "test",
        "Work",
        "state",
        {"oneOf": [action_schema("list_activity", {}, required=[])]},
        conversation_id=agent,
        metadata={
            "decision_id": decision,
            "agent_id": agent,
            "activation_id": 1,
            "activation_step": 1,
        },
    )


def success():
    return httpx.Response(
        200,
        json={
            "id": "response-1",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {"tool_calls": tool_calls({"action": "list_activity"})},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


def provider(handler, *, budget=None, concurrency=1):
    return MockDecisionProvider(
        "fixture",
        api_key="host-secret-value",
        max_tokens=100,
        context_window_tokens=1000,
        budget=budget,
        concurrency=concurrency,
        http_transport=httpx.MockTransport(handler),
    )


def ledger(tmp_path, limit=1):
    return ExecutionBudget(
        tmp_path / "budget.json",
        limit,
        input_rate=1,
        cache_read_rate=1,
        output_rate=1,
        thinking_rate=1,
    )


def test_retry_keeps_decision_identity_and_distinguishes_http_attempts(tmp_path):
    actual = []

    def respond(wire_request):
        actual.append(json.loads(wire_request.content))
        if len(actual) == 1:
            return httpx.Response(503, json={"error": "temporary"})
        return success()

    budget = ledger(tmp_path)
    remote = provider(respond, budget=budget)
    events = []
    remote.set_evidence_callback(events.append)
    try:
        first = remote.generate_batch([request()])[0]
        assert isinstance(first, ProviderTransportError)
        second = remote.generate_batch([request()])[0]
        assert second == {"action": "list_activity"}
        prepared = [e for e in events if e["kind"] == "provider_request_prepared"]
        submitted = [e for e in events if e["kind"] == "provider_request_submitted"]
        received = [e for e in events if e["kind"] == "provider_response_received"]
        assert [e["payload"] for e in submitted] == actual
        ids = [e["request_id"] for e in prepared]
        assert len(set(ids)) == 2
        assert prepared[1]["previous_request_id"] == ids[0]
        assert {e["decision_id"] for e in events} == {"decision-1"}
        assert second.request_id == ids[1] and second.request_ids == tuple(ids)
        assert [entry["request_id"] for entry in budget.entries] == ids
        assert [entry["status"] for entry in budget.entries] == ["unknown_usage", "reported"]
        recorded = [e for e in events if e["kind"] == "provider_usage_recorded"]
        assert [e["entry"] for e in recorded] == budget.entries
        assert [e["status_code"] for e in received] == [503, 200]
        assert received[1]["provider_response_id"] == "response-1"
        assert "host-secret-value" not in json.dumps(events)
    finally:
        remote.close()


def test_exhausted_admission_is_not_recorded_as_submitted(tmp_path):
    budget = ledger(tmp_path, limit=0.000001)
    remote = provider(lambda _: success(), budget=budget)
    events = []
    remote.set_evidence_callback(events.append)
    try:
        remote.generate_batch([request()])
        with pytest.raises(BudgetExhausted):
            remote.generate_batch([request("decision-2")])
        denied = [e for e in events if e["decision_id"] == "decision-2"]
        assert [e["kind"] for e in denied] == [
            "provider_request_prepared",
            "provider_request_failed",
        ]
        assert all(e["submitted"] is False for e in denied)
        assert len(budget.entries) == 1
    finally:
        remote.close()


def test_cancelled_queued_request_has_no_submission_record(tmp_path):
    started = threading.Event()

    async def respond(_):
        started.set()
        await asyncio.Event().wait()

    remote = provider(respond, budget=ledger(tmp_path))
    events = []
    remote.set_evidence_callback(events.append)
    try:
        remote.submit_request(request("first", "peer-a"))
        assert started.wait(2)
        remote.submit_request(request("queued", "peer-b"))
        remote.cancel_pending()
        queued = [e for e in events if e["decision_id"] == "queued"]
        assert not any(e["kind"] == "provider_request_submitted" for e in queued)
        failure = next(e for e in queued if e["kind"] == "provider_request_failed")
        assert failure["submitted"] is False and failure["response_received"] is False
        assert failure["error"] == "request_interrupted"
    finally:
        remote.close()


def test_observer_failures_and_credential_echo_do_not_change_transport():
    def respond(wire_request):
        assert json.loads(wire_request.content)["max_tokens"] == 100
        return httpx.Response(503, json={"host-secret-value": "rejected host-secret-value"})

    remote = provider(respond)
    observed = []

    def observer(event):
        observed.append(event)
        if "payload" in event:
            event["payload"]["max_tokens"] = 1
        raise RuntimeError("observer failed host-secret-value")

    remote.set_evidence_callback(observer)
    try:
        result = remote.generate_batch([request()])[0]
        assert isinstance(result, ProviderTransportError)
        receipt = next(e for e in observed if e["kind"] == "provider_response_received")
        assert receipt["redacted"] is True
        assert receipt["body"] == {"[REDACTED]": "rejected [REDACTED]"}
        assert "host-secret-value" not in json.dumps(observed)
        assert remote.metrics["evidence_errors"]
        assert "host-secret-value" not in json.dumps(remote.metrics["evidence_errors"])
    finally:
        remote.close()


@pytest.mark.parametrize("wire_body", ['{"id": 7}', '{"id": NaN}'])
def test_malformed_native_response_remains_recordable(wire_body):
    remote = OpenAIDecisionProvider(
        "fixture",
        api_key="host-secret-value",
        max_tokens=4096,
        context_window_tokens=128000,
        http_transport=httpx.MockTransport(lambda _: httpx.Response(200, text=wire_body)),
    )
    observed = []
    remote.set_evidence_callback(observed.append)
    try:
        result = remote.generate_batch([request()])[0]
        assert isinstance(result, ProviderTransportError)
        receipt = next(e for e in observed if e["kind"] == "provider_response_received")
        assert receipt["provider_response_id"] is None
        if "NaN" in wire_body:
            assert receipt["raw_text"] == wire_body
            assert "body" not in receipt
        else:
            assert receipt["body"] == {"id": 7}
        assert remote.metrics["evidence_errors"] == []
    finally:
        remote.close()
