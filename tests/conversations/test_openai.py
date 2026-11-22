"""Native Responses recovery and compaction over the complete runtime path."""

import json
import re
from dataclasses import replace

import httpx
import pytest
from support.boundaries import run_for
from support.organization import problem, world

from mikoshi.problem import ScheduledNotificationSpec
from mikoshi.providers.openai import OpenAIDecisionProvider
from mikoshi.runtime.budgets import ExecutionBudget
from mikoshi.schema import RecordWriter, create_run, iter_records, read_content
from mikoshi.tools.output import read_output


def test_native_responses_preserves_failed_calls_and_compacts_before_recovery(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("mikoshi.runtime.lifecycle.PROVIDER_RETRY_DELAYS", (0,))
    requests = []
    generations = 0

    def respond(request):
        nonlocal generations
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.headers["authorization"] == "Bearer synthetic"
        if len(requests) == 1:
            return httpx.Response(503, json={"error": {"message": "Temporary fixture failure"}})
        if request.url.path.endswith("/compact"):
            assert "previous_response_id" in payload
            assert "path is required" in json.dumps(payload)
            output = [{"type": "compaction", "encrypted_content": "opaque-checkpoint"}]
        else:
            generations += 1
            assert payload["max_output_tokens"] == 4096
            if generations == 1:
                calls = [
                    ("write_file", {"path": "must-not-run.txt", "content": "invalid batch"}),
                    ("read_file", {}),
                ]
            elif generations == 2:
                assert "opaque-checkpoint" in json.dumps(payload["input"])
                assert "path is required" in json.dumps(payload["input"])
                assert not payload.get("previous_response_id")
                calls = [("write_file", {"path": "corrected.txt", "content": "recovered"})]
            else:
                calls = [("peer_yield", {})]
            output = [
                {
                    "type": "function_call",
                    "call_id": f"call-{generations}-{index}",
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
                for index, (name, arguments) in enumerate(calls)
            ]
        return httpx.Response(
            200,
            json={
                "id": f"response-{len(requests)}",
                "object": "response.compaction"
                if request.url.path.endswith("/compact")
                else "response",
                "status": "completed",
                "output": output,
                "usage": {
                    "input_tokens": 60000 if generations == 1 else 1000,
                    "output_tokens": 20,
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    provider = OpenAIDecisionProvider(
        "test-model",
        api_key="synthetic",
        max_tokens=4096,
        context_window_tokens=128000,
        compaction_input_tokens=50000,
        input_token_limit=123904,
        http_transport=httpx.MockTransport(respond),
    )
    store, runtime = world(tmp_path, provider, problem(max_agents=1), enable_commands=False)
    progress = []
    runtime._progress_callback = progress.append
    try:
        run_for(runtime)
        assert runtime.stop_reason == "all_agents_yielded", [
            event for event in progress if "error" in event
        ]
        target = runtime.workspace.canonical_repository
        assert not (target / "must-not-run.txt").exists()
        assert (target / "corrected.txt").read_text(encoding="utf-8") == "recovered"
        assert generations == 3 and len(requests) == 5
        rows = store._connection.execute("SELECT messages FROM agent_memory_exchanges").fetchall()
        archive = [message for row in rows for message in json.loads(row[0])]
        errors = [m for m in archive if m["role"] == "tool" and "path is required" in m["content"]]
        assert {m["tool_call_id"] for m in errors} == {"call-1-0", "call-1-1"}
        assert any(m.get("compaction_checkpoints") for m in archive)
    finally:
        provider.close()
        store.close()


@pytest.mark.parametrize("force_compaction", [False, True])
def test_large_read_batch_has_private_pageable_results_and_can_continue(tmp_path, force_compaction):
    wires = []
    compactions = []
    budget = ExecutionBudget(tmp_path / "budget.json", seconds=120)
    budget.pause()

    def respond(request):
        payload = json.loads(request.content)
        if request.url.path.endswith("/compact"):
            compactions.append(payload)
            return httpx.Response(
                200,
                json={
                    "id": "checkpoint",
                    "object": "response.compaction",
                    "status": "completed",
                    "output": [{"type": "compaction", "encrypted_content": "opaque"}],
                    "usage": {"input_tokens": 105000, "output_tokens": 50},
                },
            )
        wires.append(payload)
        if len(wires) == 1:
            budget.elapsed_before = 60  # Release evidence before the compacted second turn.
            calls = [("read_file", {"path": f"part-{i}.txt"}) for i in range(8)]
        elif len(wires) == 2:
            outputs = [
                item
                for item in (compactions or wires)[-1]["input"]
                if item.get("type") == "function_call_output"
            ]
            assert [item["call_id"] for item in outputs] == [f"1-{i}" for i in range(8)]
            match = re.search(r'reference="([^"]+)", offset=(\d+)', outputs[0]["output"])
            if force_compaction:
                assert outputs[0]["output"] in "\n".join(
                    part["text"] for part in payload["input"][-1]["content"]
                )
            calls = [
                ("read_output", {"reference": match[1], "offset": int(match[2]), "limit": 250})
            ]
        else:
            calls = [("peer_yield", {})]
        return httpx.Response(
            200,
            json={
                "id": f"response-{len(wires)}",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": f"{len(wires)}-{i}",
                        "name": name,
                        "arguments": json.dumps(args),
                    }
                    for i, (name, args) in enumerate(calls)
                ],
                "usage": {
                    "input_tokens": 120000 if force_compaction and len(wires) == 1 else 3000,
                    "output_tokens": 100,
                },
            },
        )

    provider = OpenAIDecisionProvider(
        "test-model",
        api_key="synthetic",
        max_tokens=4096,
        context_window_tokens=128000,
        input_token_limit=123904,
        http_transport=httpx.MockTransport(respond),
        budget=budget,
    )
    spec = replace(
        problem(max_agents=1),
        scheduled_notifications=(
            ScheduledNotificationSpec("evidence", 1, "participant", "Compaction inbox evidence"),
        ),
    )
    store, runtime = world(tmp_path, provider, spec, enable_commands=False)
    output = tmp_path / "evidence"
    document = create_run(output)
    records = RecordWriter(output, document["id"])
    records.attach(store)
    provider.set_evidence_callback(records.provider_event)

    def progress(event):
        if event["kind"] == "decision_prepared":
            records.append("decision", event)

    runtime._progress_callback = progress
    try:
        for i in range(8):
            (runtime.workspace.canonical_repository / f"part-{i}.txt").write_text(
                ("a" * 160 + "\n") * 100, encoding="utf-8"
            )
        run_for(runtime)
        assert runtime.stop_reason == "all_agents_yielded" and len(wires) == 3
        assert len(compactions) == int(force_compaction)
        requests = store._connection.execute(
            "SELECT messages FROM agent_memory_exchanges ORDER BY sequence"
        )
        first, second, _ = [json.loads(row[0]) for row in requests]
        original = next(m["content"] for m in first if m.get("role") == "tool")
        reader = next(m for m in second if m["role"] == "assistant")
        args = json.loads(reader["tool_calls"][0]["function"]["arguments"])
        start = args["offset"]
        continuation = compactions[0] if force_compaction else wires[1]
        assert continuation["input"][0]["output"].startswith(original[:start])
        assert wires[2]["input"][0]["output"].endswith(original[start : start + 250])
        retained = list(iter_records(output))
        notification = next(
            record["data"] for record in retained if record["type"] == "notification"
        )
        submitted = [
            record["data"]
            for record in retained
            if record["type"] == "request"
            and record["data"]["kind"] == "provider_request_submitted"
        ]
        # Check the generation after restoration, rather than the compaction request.
        continuation = submitted[2 if force_compaction else 1]
        observation = next(
            item
            for item in continuation["included_content"]
            if item["kind"] == "observation"
            and "Compaction inbox evidence" in read_content(output, item["content_ref"])
        )
        assert observation["notification_sources"] == [
            {
                "id": notification["id"],
                "source_ref": notification["source_ref"],
                "truncated": False,
            }
        ]
        if force_compaction:
            assert "compaction_state" in observation["source_fields"]
            assert "model_input" not in observation["source_fields"]
        with pytest.raises(ValueError, match="Unknown tool result"):
            read_output(store, runtime.workspace, runtime._conversations, "another-peer", args)
    finally:
        provider.close()
        store.close()


def test_locally_unfit_prompt_has_a_context_stop_reason(tmp_path):
    provider = OpenAIDecisionProvider(
        "test-model",
        api_key="synthetic",
        max_tokens=4096,
        context_window_tokens=128000,
        input_token_limit=100,
        http_transport=httpx.MockTransport(lambda request: pytest.fail("unexpected HTTP call")),
    )
    store, runtime = world(tmp_path, provider, problem(max_agents=1), enable_commands=False)
    try:
        run_for(runtime)
        assert runtime.stop_reason == "context_window_exhausted"
    finally:
        provider.close()
        store.close()
