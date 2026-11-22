"""Peers list live activity without exposing private or unrelated records."""

import json
from dataclasses import replace

import pytest
from support.organization import ScriptedProvider, problem, world

from mikoshi.contracts import Trigger
from mikoshi.problem import ScheduledNotificationSpec
from mikoshi.runtime.budgets import ExecutionBudget


def test_activity_covers_pages_files_commands_discussion_and_scope(tmp_path, monkeypatch):
    provider = ScriptedProvider()
    provider.budget = ExecutionBudget(tmp_path / "budget.json", seconds=120)
    provider.budget.pause()
    spec = problem(max_agents=3)
    spec = replace(
        spec,
        team=replace(
            spec.team,
            members=(
                spec.team.members[0],
                replace(spec.team.members[1], instructions="PRIVATE ROLE INSTRUCTIONS"),
                spec.team.members[2],
            ),
        ),
        scheduled_notifications=(
            ScheduledNotificationSpec(
                "private-evidence", 1, "worker_a", "PRIVATE SCHEDULED MESSAGE"
            ),
        ),
    )
    store, runtime = world(tmp_path, provider, spec, enable_commands=False)
    try:
        reader, writer = runtime._claim_wave(2, 1)
        project = reader.trigger.project_id

        def apply(state, action, **arguments):
            return runtime._apply_workspace_decision(state, {"action": action, **arguments})[1]

        def activity_of(**arguments):
            return apply(reader, "list_activity", **arguments)

        assert "PRIVATE ROLE INSTRUCTIONS" in runtime._workspace_request(writer).instructions
        first_page = apply(
            reader,
            "add_page",
            path="/parser",
            title="Review parser",
            markdown="Review src/parser.py before editing.",
        )
        apply(
            reader,
            "add_page",
            path="/lexer",
            title="Review lexer",
            markdown="Review src/lexer.py.",
        )
        writer_page = apply(
            writer,
            "add_page",
            path="/generator",
            title="Repair generator",
            markdown="Working on src/generator.py.",
        )
        apply(
            writer,
            "edit_page",
            path="/generator",
            edits=[
                {
                    "old_text": "Working on src/generator.py.",
                    "new_text": "Waiting for reproduction evidence before editing src/generator.py.",
                }
            ],
        )
        apply(writer, "write_file", path="src/shared.py", content="original\n")
        apply(writer, "write_file", path="src/shared.py.other", content="separate\n")
        apply(reader, "read_file", path="src/shared.py")
        apply(
            writer,
            "edit_file",
            path="src/shared.py",
            edits=[{"old_text": "original", "new_text": "reviewed"}],
        )
        with pytest.raises(ValueError):
            apply(
                reader,
                "edit_file",
                path="src/shared.py",
                edits=[{"old_text": "original", "new_text": "stale"}],
            )
        notes = apply(
            reader, "add_page", path="/notes", title="Findings", markdown="Shared evidence"
        )
        comment = apply(
            writer,
            "comment_page",
            path="/notes",
            markdown="Please avoid editing the generator until I reproduce the issue.",
            notify=["participant"],
        )
        apply(reader, "view_page", path="/notes")
        apply(reader, "check_budget")
        commands = [
            store.request_command(
                state.activation_id,
                state.trigger.agent_id,
                ("sh", "-c", command),
                "repository",
                project_id=project,
            )
            for state, command in [
                (reader, "python check_parser.py # " + "long command context " * 20),
                (writer, "python check_generator.py"),
                (writer, "python failing_check.py"),
            ]
        ]
        store.complete_command_request(
            writer.activation_id,
            writer.trigger.agent_id,
            commands[2]["id"],
            {
                "status": "failed",
                "exit_code": 2,
                "termination_reason": "completed",
                "output": "Check failed",
            },
        )
        monkeypatch.setattr(
            runtime,
            "command_statuses",
            lambda: {commands[0]["id"]: "execution_in_progress", commands[1]["id"]: "queued"},
        )
        provider.budget.elapsed_before = 60
        runtime._scheduled.release_due()

        # A member may also act in another project; actor identity must not widen scope.
        db = store._connection
        with db:
            db.execute(
                "INSERT INTO projects VALUES ('unrelated','Unrelated','Private work','active',0,0)"
            )
            foreign_trigger = db.execute(
                "INSERT INTO triggers(agent_id,reason,priority,created_tick,status,kind,project_id) VALUES (?,'other project',1,0,'running','peer_session','unrelated')",
                (writer.trigger.agent_id,),
            ).lastrowid
            activation = store.start_activation(
                Trigger(
                    foreign_trigger,
                    writer.trigger.agent_id,
                    "other project",
                    1,
                    None,
                    0,
                    "peer_session",
                    "unrelated",
                ),
                1,
            )
            store.record_workspace_action(
                activation,
                writer.trigger.agent_id,
                "workspace_write",
                "succeeded",
                "UNRELATED FILE",
                {"project_id": "unrelated", "content": "UNRELATED CONTENT"},
            )

        activity = activity_of(limit=20)
        assert set(activity) == {"current_tick", "active_commands", "actions", "next_offset"}
        assert {item["status"] for item in activity["active_commands"]} == {"queued", "running"}
        long_command = next(
            c for c in activity["active_commands"] if c["command_id"] == commands[0]["id"]
        )
        assert len(long_command["command"]) <= 160 and long_command["command_truncated"]
        events = activity["actions"]
        # Actions carry the tool names peers call, never internal event names.
        assert {event["action"] for event in events} >= {
            "read_file",
            "edit_file",
            "comment_page",
            "add_page",
            "edit_page",
            "run_command",
            "command_finished",
            "activation_started",
        }
        assert not {"workspace_read", "execute_command", "check_budget"} & {
            event["action"] for event in events
        }
        assert all(set(event) >= {"tick", "peer_id", "action"} for event in events)
        assert not any(event["action"] == "scheduled_message" for event in events)
        scheduled = apply(writer, "list_activity", category="messages")["actions"]
        assert len(scheduled) == 1
        assert scheduled[0]["action"] == "scheduled_message"
        assert "comment_id" not in scheduled[0]
        assert any(event.get("outcome") == "failed" for event in events)
        assert any(
            event.get("comment_id") == comment["comment_id"] and event["path"] == notes["path"]
            for event in events
        )
        assert any(event.get("output_ref") == f"command:{commands[2]['id']}" for event in events)
        edited = next(event for event in events if event["action"] == "edit_page")
        assert edited["path"] == writer_page["path"] == "/generator"
        page_changes = [
            item
            for item in activity_of(category="pages", limit=20)["actions"]
            if item["action"] in {"add_page", "edit_page"}
        ]
        assert [(item["action"], item["path"]) for item in page_changes] == [
            ("add_page", "/notes"),
            ("edit_page", "/generator"),
            ("add_page", "/generator"),
            ("add_page", "/lexer"),
            ("add_page", "/parser"),
        ]
        assert not any("work_id" in item or "status" in item for item in page_changes)
        for event in events:
            if event.get("command_id") in {c["command_id"] for c in activity["active_commands"]}:
                assert "command" not in event
        assert not any(
            secret in json.dumps(activity)
            for secret in [
                "PRIVATE ROLE INSTRUCTIONS",
                "PRIVATE SCHEDULED MESSAGE",
                "UNRELATED FILE",
                "UNRELATED CONTENT",
            ]
        )

        # Paging walks back from the first page; newer actions don't enter older pages.
        first = activity_of(limit=1)
        apply(reader, "write_file", path="new.txt", content="Newer than the first page")
        pages = [first]
        while pages[-1]["next_offset"] is not None:
            assert len(pages) < 40
            pages.append(activity_of(limit=1, offset=pages[-1]["next_offset"]))
        ticks = [item["tick"] for page in pages for item in page["actions"]]
        assert ticks == sorted(ticks, reverse=True) == [event["tick"] for event in events]
        fresh = activity_of(category="files", since_tick=first["current_tick"])
        assert [item["path"] for item in fresh["actions"]] == ["new.txt"]
        filtered = activity_of(path="src/shared.py", peer_id=writer.trigger.agent_id)
        assert {item["action"] for item in filtered["actions"]} == {"write_file", "edit_file"}
        assert all(item["path"] == "src/shared.py" for item in filtered["actions"])
        assert len(filtered["active_commands"]) == 1
        with pytest.raises(ValueError, match="path only applies to file actions"):
            activity_of(path="src/shared.py", category="pages")
        with pytest.raises(ValueError, match="'unknown-peer' is not a peer"):
            activity_of(peer_id="unknown-peer")
        with pytest.raises(ValueError, match="relative to the repository root"):
            activity_of(path="../outside")

        # Retained command output is separate from peer-authored pages.
        exports = runtime.workspace.root / "pages"
        viewed = apply(reader, "read_output", reference=f"command:{commands[2]['id']}")
        assert viewed["content"] == "Check failed"
        assert apply(reader, "list_pages", path="/coordination")["pages"] == []
        assert apply(reader, "search_pages", query="Check failed")["total"] == 0
        assert first_page["path"] == "/parser"
        listed = apply(reader, "list_pages", path="/")["pages"]
        assert any(
            page["path"] == "/parser" and page["title"] == "Review parser" for page in listed
        )
        assert (exports / "objective.md").is_file()
        assert (exports / "coordination.md").is_file()
        assert not (exports / "organization.md").exists()
        assert not list(exports.glob("project-*.md"))
    finally:
        store.close()


def test_retained_command_output_is_paged_shared_and_project_scoped(tmp_path):
    from mikoshi.runtime.lifecycle import _command_result

    store, runtime = world(tmp_path, spec=problem(max_agents=2), enable_commands=False)
    try:
        author, reader = runtime._claim_wave(2, 1)
        command = store.request_command(
            author.activation_id,
            author.trigger.agent_id,
            ("sh", "-c", "echo evidence"),
            "repository",
            project_id=author.trigger.project_id,
        )

        def apply(**arguments):
            return runtime._apply_workspace_decision(
                reader, {"action": "read_output", **arguments}
            )[1]

        reference = f"command:{command['id']}"
        with pytest.raises(ValueError, match="after the command finishes"):
            apply(reference=reference)
        text = "Complete output: 🧪 café\r\nline\rprogress\n" * 80
        logs = runtime.workspace.root / "command-logs"
        logs.mkdir(exist_ok=True)
        path = logs / f"command-{command['id']}.log"
        path.write_text(text, encoding="utf-8")
        store.complete_command_request(
            author.activation_id,
            author.trigger.agent_id,
            command["id"],
            {
                "status": "succeeded",
                "exit_code": 0,
                "termination_reason": "completed",
                "output": "A short tail preview",
                "output_truncated": True,
                "log_path": f"/logs/{path.name}",
            },
        )
        completed = _command_result(store.get_command_request(command["id"]))
        assert completed["output_ref"] == reference
        assert "receipt_page" not in completed and "log_path" not in completed
        first = apply(reference=reference, limit=23)
        assert first["content"] == text[:23]
        assert (
            apply(reference=reference, offset=first["next_offset"], limit=23)["content"]
            == text[23:46]
        )
        assert apply(reference=reference, offset=len(text))["next_offset"] is None
        assert apply(reference=reference, offset=len(text))["content"] == ""
        with pytest.raises(ValueError, match="exceeds"):
            apply(reference=reference, offset=len(text) + 1)

        with store._connection:
            store._connection.execute(
                "DELETE FROM project_members WHERE agent_id=?", (reader.trigger.agent_id,)
            )
        with pytest.raises(ValueError, match="not a member"):
            apply(reference=reference)
        assert not store.event_observer_errors
    finally:
        store.close()


def test_changes_only_activity_leaves_out_reads_and_failed_attempts(tmp_path):
    store, runtime = world(tmp_path, spec=problem(max_agents=2), enable_commands=False)
    try:
        reader, writer = runtime._claim_wave(2, 1)

        def apply(state, action, **arguments):
            return runtime._apply_workspace_decision(state, {"action": action, **arguments})[1]

        apply(writer, "add_page", path="/plan", title="Plan", markdown="Draft")
        apply(writer, "view_page", path="/plan")
        apply(writer, "write_file", path="notes.txt", content="one\n")
        apply(writer, "read_file", path="notes.txt")
        with pytest.raises(ValueError):
            apply(writer, "edit_page", path="/plan", edits=[{"old_text": "Gone", "new_text": "x"}])
        apply(writer, "comment_page", path="/plan", markdown="Ready for review")

        everything = apply(reader, "list_activity", limit=20)["actions"]
        assert {"view_page", "read_file", "edit_page"} <= {a["action"] for a in everything}
        changes = apply(reader, "list_activity", changes_only=True, limit=20)["actions"]
        assert [a["action"] for a in changes] == ["comment_page", "write_file", "add_page"]
        pages = apply(reader, "list_activity", changes_only=True, category="pages")["actions"]
        assert [a["action"] for a in pages] == ["comment_page", "add_page"]
    finally:
        store.close()
