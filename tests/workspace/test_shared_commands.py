"""Command failure and active-command boundaries apply to the whole live repository."""

import json

from support.boundaries import run_for
from support.organization import ScriptedProvider, swarm_problem, world

from mikoshi.tools.commands import DEFAULT_COMMAND_TIMEOUT_SECONDS


def test_stop_cancels_all_commands_and_preserves_receipts(tmp_path, monkeypatch):
    import threading

    from support.organization import swarm_problem, world

    peers = ("agent-1", "agent-2")
    entered = {actor: threading.Event() for actor in peers}
    counts = {}
    executions = []

    class Provider:
        def generate_batch(self, requests):
            values = []
            for request in requests:
                actor = request.metadata["agent_id"]
                count = counts.get(actor, 0)
                counts[actor] = count + 1
                values.append(
                    {"action": "run_command", "command": actor}
                    if count == 0
                    else {
                        "action": "peer_yield",
                    }
                )
            return values

    store, runtime = world(
        tmp_path,
        provider=Provider(),
        spec=swarm_problem(population=2),
        enable_commands=True,
    )
    try:

        def command(prepared):
            # No run deadline here, so the harness default command timeout applies.
            assert prepared.timeout_seconds == DEFAULT_COMMAND_TIMEOUT_SECONDS
            actor = prepared.argv[2]
            executions.append(actor)
            entered[actor].set()
            assert prepared.cancel_event.wait(5)
            return {
                "_command_id": prepared.command_id,
                "status": "failed",
                "termination_reason": "cancelled",
                "exit_code": None,
            }

        monkeypatch.setattr(runtime._command_runner, "_execute_container", command)
        runtime.run(
            stop_requested=lambda: (
                "operator_stop" if all(event.is_set() for event in entered.values()) else None
            )
        )
        assert runtime.stop_reason == "operator_stop"
        assert len(store.list_command_requests()) == 2
        assert all(
            command["termination_reason"] == "cancelled"
            for command in store.list_command_requests()
        )
        assert not runtime._command_jobs
        assert sorted(executions) == sorted(peers)
        assert all(count == 1 for count in counts.values())
    finally:
        store.close()


def test_shell_script_loads_scoped_guidance_before_execution(tmp_path, monkeypatch):
    script = """cat <<'TEXT' | sed 's/"//g' > src/output.txt
"a b"
TEXT
apply_patch <<'PATCH'
*** Begin Patch
*** Update File: src/example.py
@@
-old
+new
*** End Patch
PATCH
"""
    decision = {"action": "run_command", "command": script}
    provider = ScriptedProvider([decision, decision])
    store, runtime = world(
        tmp_path,
        provider=provider,
        spec=swarm_problem(population=1),
        enable_commands=True,
    )
    source = runtime.workspace.canonical_repository / "src"
    source.mkdir()
    (source / "AGENTS.md").write_text(
        "Preserve the source directory's public interfaces.\n", encoding="utf-8"
    )
    skill = runtime.workspace.canonical_repository / ".agents/skills/review/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: review\ndescription: Review public interfaces.\n---\nRead the source.\n",
        encoding="utf-8",
    )
    executions = []

    def execute(prepared):
        executions.append(prepared.argv)
        return {
            "_command_id": prepared.command_id,
            "status": "succeeded",
            "exit_code": 0,
            "termination_reason": "completed",
            "output": "done\n",
        }

    monkeypatch.setattr(runtime._command_runner, "_execute_container", execute)
    try:
        run_for(runtime)
        assert runtime.stop_reason == "all_agents_yielded"
        assert executions == [("sh", "-c", script)]
        (command,) = store.list_command_requests()
        assert command["argv"] == ("sh", "-c", script)
        startup = json.loads(provider.requests[0].input_text)["repository_guidance"]["startup"]
        assert [entry["path"] for entry in startup["catalog"]["skills"]] == [
            ".agents/skills/review/SKILL.md"
        ]
        guidance = json.loads(provider.requests[1].input_text)["repository_guidance"]["activity"]
        assert [document["path"] for document in guidance["documents"]] == ["src/AGENTS.md"]
        assert any(
            event["action_type"] == "run_command" and event["outcome"] == "deferred"
            for event in store.list_events()
        )
    finally:
        store.close()
