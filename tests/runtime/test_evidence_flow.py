"""Public evidence follows real mocked wire requests through native batched tools."""

import json
import sys

import httpx
import pytest

from mikoshi import application, cli
from mikoshi.schema import (
    RecordWriter,
    create_run,
    iter_records,
    read_content,
    read_document,
    validate_bundle,
)


def _launch_native_turns(tmp_path, monkeypatch, capsys, *, first_failure=False):
    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "run"
    configuration = {
        "kind": "input",
        "problem": {
            "objective": {
                "id": "evidence",
                "title": "Record evidence",
                "description": "Record evidence",
            },
            "limits": {"max_agents": 1},
            "scheduled_notifications": [
                {
                    "id": "dispatch-marker",
                    "at_minutes": 0,
                    "recipient": "agent-1",
                    "message": "Operator evidence marker",
                }
            ],
        },
        "run": {
            "repository": str(repository),
            "output": str(output),
            "model": "fixture-model",
            "max_output_tokens": 4096,
            "context_window_tokens": 128000,
            "budget_usd": 1,
            "wall_time_seconds": 60,
        },
        "pricing": {
            "input_rate": 1,
            "cache_read_rate": 0.1,
            "output_rate": 2,
            "thinking_rate": 2,
        },
        "deployment": {"enable_commands": False},
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(configuration), encoding="utf-8")
    secret = "unit-test-secret-do-not-export"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    payloads, responses = [], []

    def respond(request):
        assert request.headers["authorization"] == f"Bearer {secret}"
        payloads.append(json.loads(request.content))
        if first_failure and len(payloads) == 1:
            response = {"error": "Temporary provider failure"}
            responses.append(response)
            return httpx.Response(503, json=response)
        if len(payloads) == 1 + int(first_failure):
            calls = [
                (
                    "call-write",
                    "write_file",
                    {"path": "evidence.txt", "content": "Recorded evidence"},
                ),
                ("call-read", "read_file", {"path": "evidence.txt"}),
            ]
        else:
            calls = [("call-yield", "peer_yield", {})]
        response = {
            "id": f"response-{len(payloads)}",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
                for call_id, name, arguments in calls
            ],
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 20,
            },
        }
        responses.append(response)
        return httpx.Response(200, json=response)

    original = application.build_decision_provider

    def mocked(args, progress_callback=None):
        provider = original(args, progress_callback)
        provider._http_transport = httpx.MockTransport(respond)
        return provider

    monkeypatch.setattr(application, "build_decision_provider", mocked)
    if first_failure:
        monkeypatch.setattr("mikoshi.runtime.lifecycle.PROVIDER_RETRY_DELAYS", (0,))
    monkeypatch.setattr(sys, "argv", ["mikoshi", "--config", str(path)])
    cli.main()
    capsys.readouterr()
    assert (repository / "evidence.txt").read_text(encoding="utf-8") == "Recorded evidence"
    assert len(payloads) == 2 + int(first_failure)
    assert payloads[-1]["previous_response_id"] == f"response-{1 + int(first_failure)}"
    assert {
        item["call_id"]
        for item in payloads[-1]["input"]
        if item.get("type") == "function_call_output"
    } == {"call-write", "call-read"}
    return output, payloads, responses, secret


def test_native_batched_tools_and_followup_input_are_retained(tmp_path, monkeypatch, capsys):
    output, payloads, responses, secret = _launch_native_turns(tmp_path, monkeypatch, capsys)
    run = read_document(output / "run.json", "run")
    assert run["status"] == "completed"
    records = list(iter_records(output))
    requests = [record["data"] for record in records if record["type"] == "request"]
    prepared = [record for record in requests if record["kind"] == "provider_request_prepared"]
    submitted = [record for record in requests if record["kind"] == "provider_request_submitted"]
    received = [
        record["data"]
        for record in records
        if record["type"] == "response" and record["data"]["kind"] == "provider_response_received"
    ]
    exchanges = [record["data"] for record in records if record["type"] == "exchange"]
    assert len(prepared) == len(submitted) == len(received) == len(exchanges) == 2
    assert len({record["request_id"] for record in submitted}) == 2
    assert len({record["decision_id"] for record in submitted}) == 2
    for wire, request in zip(payloads, submitted):
        assert read_content(output, request["payload_ref"]) == wire
    for wire, response in zip(responses, received):
        assert read_content(output, response["body_ref"]) == wire
    for request, response, exchange, usage in zip(
        submitted, received, exchanges, run["budget"]["entries"]
    ):
        assert (
            request["request_id"]
            == response["request_id"]
            == usage["request_id"]
            == exchange["request_id"]
        )
        assert (
            request["decision_id"]
            == response["decision_id"]
            == usage["decision_id"]
            == exchange["decision_id"]
        )

    first_actions = {
        record["data"]["action_type"]: record["data"]["payload"]
        for record in records
        if record["type"] == "action"
        and record["data"]["action_type"] in {"workspace_write", "workspace_read"}
    }
    assert first_actions["workspace_write"]["tool_call_id"] == "call-write"
    assert first_actions["workspace_read"]["tool_call_id"] == "call-read"
    assert first_actions["workspace_write"]["tool_call_index"] == 0
    assert first_actions["workspace_read"]["tool_call_index"] == 1
    assert {action["decision_id"] for action in first_actions.values()} == {
        submitted[0]["decision_id"]
    }
    first_messages = read_content(output, exchanges[0]["messages_ref"])
    produced = {
        message["tool_call_id"]: message["content"]
        for message in first_messages
        if message["role"] == "tool"
    }
    submitted_outputs = {
        item["call_id"]: item["output"]
        for item in read_content(output, submitted[1]["payload_ref"])["input"]
        if item.get("type") == "function_call_output"
    }
    assert submitted_outputs == produced
    included_results = {
        item["tool_call_id"]: read_content(output, item["content_ref"])
        for item in submitted[1]["included_content"]
        if item["kind"] == "tool_result"
    }
    assert included_results == produced
    notification = next(record["data"] for record in records if record["type"] == "notification")
    assert notification["event"] == "queued"
    assert read_content(output, notification["message_ref"]) == "Operator evidence marker"
    first_observation = next(
        item for item in submitted[0]["included_content"] if item["kind"] == "observation"
    )
    assert first_observation["notification_sources"] == [
        {
            "id": notification["id"],
            "source_ref": notification["source_ref"],
            "truncated": False,
        }
    ]
    model_observation = json.loads(read_content(output, first_observation["content_ref"]))
    assert model_observation["inbox"]["updates"][0]["message"] == "Operator evidence marker"
    assert validate_bundle(output)["status"] == "valid"
    # Persisted public records must never include transport credentials or headers.
    public_files = [output / "run.json", output / "records.jsonl"]
    public_files.extend((output / "content").rglob("*") if (output / "content").exists() else ())
    for path in public_files:
        if path.is_file():
            assert secret.encode() not in path.read_bytes()


def test_transport_retry_preserves_decision_identity_without_inventing_an_exchange(
    tmp_path, monkeypatch, capsys
):
    output, _, _, _ = _launch_native_turns(tmp_path, monkeypatch, capsys, first_failure=True)
    records = list(iter_records(output))
    submitted = [
        record["data"]
        for record in records
        if record["type"] == "request" and record["data"]["kind"] == "provider_request_submitted"
    ]
    exchanges = [record["data"] for record in records if record["type"] == "exchange"]
    assert len(submitted) == 3 and len(exchanges) == 2
    assert len({request["request_id"] for request in submitted}) == 3
    assert submitted[0]["decision_id"] == submitted[1]["decision_id"]
    assert submitted[2]["decision_id"] != submitted[1]["decision_id"]
    assert submitted[1]["previous_request_id"] == submitted[0]["request_id"]
    assert exchanges[0]["decision_id"] == submitted[1]["decision_id"]
    assert exchanges[0]["request_id"] == submitted[1]["request_id"]
    budget = read_document(output / "run.json", "run")["budget"]
    assert [entry["request_id"] for entry in budget["entries"]] == [
        request["request_id"] for request in submitted
    ]
    assert budget["unknown_usage_requests"] == 1
    assert budget["entries"][0]["status"] == "unknown_usage"
    assert validate_bundle(output)["status"] == "valid"


def test_tool_argument_shaped_like_a_result_is_not_classified_as_protocol_delivery(tmp_path):
    run = create_run(tmp_path)
    writer = RecordWriter(tmp_path, run["id"])
    record = writer.provider_event(
        {
            "kind": "provider_request_submitted",
            "request_id": "request",
            "decision_id": "decision",
            "payload": {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call",
                                "name": "example",
                                "input": {
                                    "nested": {
                                        "type": "tool_result",
                                        "tool_use_id": "fake",
                                        "content": "Authored argument",
                                    }
                                },
                            }
                        ],
                    }
                ],
            },
        }
    )
    assert record["data"]["included_content"] == []


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("content", "content fingerprint mismatch"),
        ("request_link", "unknown or mismatched request"),
        ("included_content", "submitted content differs"),
        ("native_call", "native tool|tool call"),
        ("request_predecessor", "predecessor|previous request"),
    ],
)
def test_bundle_validation_rejects_broken_evidence_links(
    tmp_path, monkeypatch, capsys, corruption, error
):
    output, _, _, _ = _launch_native_turns(tmp_path, monkeypatch, capsys)
    records = list(iter_records(output))
    assert validate_bundle(output)["status"] == "valid"
    if corruption == "content":
        submitted = next(
            record["data"]
            for record in records
            if record["type"] == "request"
            and record["data"]["kind"] == "provider_request_submitted"
        )
        (output / submitted["payload_ref"]["path"]).write_text("{}\n", encoding="utf-8")
    elif corruption in {"request_link", "native_call"}:
        action = next(
            record["data"]
            for record in records
            if record["type"] == "action" and record["data"]["action_type"] == "workspace_write"
        )
        if corruption == "request_link":
            action["payload"]["request_id"] = "missing-request"
        else:
            action["payload"]["tool_call_id"] = "missing-native-call"
    elif corruption == "request_predecessor":
        identity = next(
            record["data"]["request_id"] for record in records if record["type"] == "request"
        )
        for record in records:
            if (
                record["type"] in {"request", "response", "usage"}
                and record["data"]["request_id"] == identity
            ):
                record["data"]["previous_request_id"] = identity
    else:
        submitted = next(
            record["data"]
            for record in records
            if record["type"] == "request"
            and record["data"]["kind"] == "provider_request_submitted"
            and any(item["kind"] == "tool_result" for item in record["data"]["included_content"])
        )
        notification = next(
            record["data"] for record in records if record["type"] == "notification"
        )
        submitted["included_content"][0]["content_ref"] = notification["message_ref"]
    if corruption != "content":
        (output / "records.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match=error):
        validate_bundle(output)
