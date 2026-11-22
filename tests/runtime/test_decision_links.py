"""Observed decision/call identities survive retries, batching and async commands."""

import json

from support.boundaries import run_for
from support.organization import swarm_problem, world

from mikoshi.contracts import BATCH_ACTION
from mikoshi.providers.base import DecisionResult


def response(calls, request_id):
    decision = calls[0] if len(calls) == 1 else {"action": BATCH_ACTION, "actions": calls}
    result = DecisionResult(
        decision,
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"{request_id}-call-{index}",
                    "type": "function",
                    "function": {
                        "name": call["action"],
                        "arguments": json.dumps({k: v for k, v in call.items() if k != "action"}),
                    },
                }
                for index, call in enumerate(calls)
            ],
        },
    )
    result.request_id = request_id
    result.request_ids = [request_id]
    return result


def test_async_command_and_skipped_calls_retain_original_native_identity(tmp_path, monkeypatch):
    calls = [
        {"action": "write_file", "path": "first.txt", "content": "retained"},
        {"action": "run_command", "command": "printf command"},
        {"action": "read_file", "path": "missing.txt"},
        {"action": "write_file", "path": "skipped.txt", "content": "never written"},
    ]

    class Provider:
        def __init__(self):
            self.requests = []

        def generate_batch(self, requests):
            self.requests.extend(requests)
            return [
                response(calls, "transport-first")
                if len(self.requests) == 1
                else response([{"action": "peer_yield"}], "transport-yield")
            ]

    provider = Provider()
    progress = []
    store, runtime = world(
        tmp_path,
        provider,
        swarm_problem(population=1),
        enable_commands=True,
        progress_callback=progress.append,
    )
    monkeypatch.setattr(
        runtime._command_runner,
        "_execute_container",
        lambda prepared: {
            "_command_id": prepared.command_id,
            "status": "succeeded",
            "exit_code": 0,
            "termination_reason": "completed",
            "output": "command",
            "isolation": {"backend": "docker", "cleanup": {"confirmed": True}},
        },
    )
    try:
        run_for(runtime)
        first, second = provider.requests
        decision_id = first.metadata["decision_id"]
        assert first.metadata["activation_id"] == second.metadata["activation_id"]
        assert first.metadata["activation_step"] == 1
        assert second.metadata["activation_step"] == 2
        assert second.metadata["decision_id"] != decision_id
        effects = [
            event
            for event in store.list_events()
            if event["payload"].get("decision_id") == decision_id
        ]
        assert {event["payload"]["request_id"] for event in effects} == {"transport-first"}
        for event in effects:
            index = event["payload"]["tool_call_index"]
            assert event["payload"]["tool_call_id"] == f"transport-first-call-{index}"
        commands = [
            event
            for event in effects
            if event["action_type"] in {"request_command", "execute_command"}
        ]
        assert len(commands) == 2
        assert {event["payload"]["tool_call_index"] for event in commands} == {1}
        assert not (runtime.workspace.canonical_repository / "skipped.txt").exists()
        replies = [message for message in second.history if message["role"] == "tool"]
        assert len(replies) == 4
        assert "Skipped" in replies[-1]["content"]
        assert replies[-1]["tool_call_id"] == "transport-first-call-3"
        archived = store._connection.execute(
            "SELECT * FROM agent_memory_exchanges ORDER BY sequence"
        ).fetchall()
        assert archived[0]["decision_id"] == decision_id
        assert archived[0]["request_id"] == "transport-first"
        assert json.loads(archived[0]["request_ids"]) == ["transport-first"]
        prepared = [event for event in progress if event["kind"] == "decision_prepared"]
        assert len(prepared) == 2
        assert prepared[0]["model_input"] == first.model_input_text
        assert prepared[1]["history"] == list(second.history)
    finally:
        store.close()


def test_runtime_retry_reuses_prepared_decision_without_claiming_native_calls(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("mikoshi.runtime.lifecycle.PROVIDER_RETRY_DELAYS", (0,))

    class Provider:
        def __init__(self):
            self.requests = []

        def generate_batch(self, requests):
            self.requests.extend(requests)
            if len(self.requests) == 1:
                return [RuntimeError("transient")]
            return [{"action": "peer_yield"}]

    provider = Provider()
    progress = []
    store, runtime = world(
        tmp_path,
        provider,
        swarm_problem(population=1),
        enable_commands=False,
        progress_callback=progress.append,
    )
    try:
        run_for(runtime)
        assert provider.requests[0] is provider.requests[1]
        decision_id = provider.requests[0].metadata["decision_id"]
        assert len([event for event in progress if event["kind"] == "decision_prepared"]) == 1
        event = next(event for event in store.list_events() if event["action_type"] == "peer_yield")
        assert event["payload"]["decision_id"] == decision_id
        assert event["payload"]["tool_call_id"] is None
        assert event["payload"]["request_id"] is None
        row = store._connection.execute(
            "SELECT decision_id, tool_calls FROM agent_memory_exchanges"
        ).fetchone()
        assert row["decision_id"] == decision_id
        assert json.loads(row["tool_calls"]) == []
    finally:
        store.close()


def test_prepared_evidence_failure_is_recorded_without_disrupting_work(tmp_path):
    def failed_observer(_):
        raise RuntimeError("private callback details")

    store, runtime = world(
        tmp_path,
        spec=swarm_problem(population=1),
        enable_commands=False,
        progress_callback=failed_observer,
    )
    try:
        run_for(runtime)
        assert runtime.stop_reason == "all_agents_yielded"
        assert "progress decision_prepared: RuntimeError" in store.event_observer_errors
        assert "private callback details" not in json.dumps(store.event_observer_errors)
    finally:
        store.close()
