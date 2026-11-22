"""Native multi-call turns preserve per-call results and completed effects."""

import json

from support.boundaries import run_for
from support.organization import swarm_problem, world

from mikoshi.contracts import BATCH_ACTION
from mikoshi.providers.base import DecisionResult


def batch(*calls):
    return {"action": BATCH_ACTION, "actions": list(calls)}


def write(path, content="written once"):
    return {"action": "write_file", "path": path, "content": content}


def setup(tmp_path, **kwargs):
    return world(tmp_path, spec=swarm_problem(population=1), **kwargs)


def test_async_command_in_batch_keeps_individual_results_and_executes_once(tmp_path, monkeypatch):
    source = 'def example():\n\treturn "quoted text"\\n\n'
    calls = [
        write("first.txt", source),
        {"action": "run_command", "command": "printf 'command result\\n'"},
        {"action": "list_activity"},
        {"action": "read_file", "path": "first.txt"},
    ]

    class Provider:
        def __init__(self):
            self.requests = []

        def generate_batch(self, requests):
            self.requests.extend(requests)
            if len(self.requests) == 1:
                return [
                    DecisionResult(
                        batch(*calls),
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": f"call_{index}",
                                    "type": "function",
                                    "function": {
                                        "name": call["action"],
                                        "arguments": json.dumps(
                                            {k: v for k, v in call.items() if k != "action"}
                                        ),
                                    },
                                }
                                for index, call in enumerate(calls)
                            ],
                        },
                    )
                ]
            return [{"action": "peer_yield"}]

    provider = Provider()
    store, runtime = setup(tmp_path, provider=provider, enable_commands=True)
    executions = []

    def execute(prepared):
        executions.append(prepared.command_id)
        return {
            "_command_id": prepared.command_id,
            "status": "succeeded",
            "exit_code": 0,
            "termination_reason": "completed",
            "output": "command result\n",
            "isolation": {"backend": "docker", "cleanup": {"confirmed": True}},
        }

    monkeypatch.setattr(runtime._command_runner, "_execute_container", execute)
    try:
        run_for(runtime, 1)
        # Batch, then an immediate yield.
        assert len(provider.requests) == 2, runtime.stop_reason
        assert len(executions) == 1
        results = [m for m in provider.requests[1].history if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in results] == ["call_0", "call_1", "call_2", "call_3"]
        assert "Output:\ncommand result\n" in results[1]["content"]
        activity = json.loads(results[2]["content"])["actions"]
        assert any(item.get("path") == "first.txt" for item in activity)
        assert any(item["action"] == "command_finished" for item in activity)
        assert (
            'Content:\n     1\tdef example():\n     2\t\treturn "quoted text"\\n'
            in results[3]["content"]
        )
        assert store._connection.execute("SELECT count(*) FROM command_requests").fetchone()[0] == 1
        (command,) = store.list_command_requests()
        assert command["status"] == "succeeded"
    finally:
        store.close()
