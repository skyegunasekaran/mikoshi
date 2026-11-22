"""Default peer names and the paths peers use to address the shared checkout."""

from dataclasses import replace

import pytest
from support.organization import problem, swarm_problem, world

from mikoshi.problem import TeamSpec, fixed_members, validate_problem
from mikoshi.runtime.lifecycle import model_error
from mikoshi.runtime.repository_guidance import activity_guidance


def test_default_roster_uses_names_that_pass_identifier_validation():
    spec = swarm_problem(population=40)
    roster = [member.id for member in fixed_members(spec)]
    assert roster == [f"agent-{index}" for index in range(1, 41)]
    assert [member.id for member in fixed_members(validate_problem(spec))] == roster


def test_comments_only_notify_explicit_peer_ids_and_replies_remain_public(tmp_path):
    store, runtime = world(tmp_path, spec=swarm_problem(population=3), enable_commands=False)
    try:
        states = {state.trigger.agent_id: state for state in runtime._claim_wave(3, 1)}

        def comment(peer, **arguments):
            return runtime._apply_workspace_decision(
                states[peer], {"action": "comment_page", "path": "/coordination", **arguments}
            )[1]

        explicit = comment(
            "agent-1", markdown="@agent-3 please look", notify=["agent-2", "agent-1"]
        )
        assert explicit["notified"] == ["agent-2"] and explicit["woken"] == []
        assert runtime._notifications.digest("agent-3")[0]["updates"] == []
        for unknown in ("agent2", "a2", "@agent-2", "unknown"):
            with pytest.raises(ValueError, match="unknown peer IDs"):
                comment("agent-1", markdown="Rejected", notify=[unknown])
        first = states["agent-1"]
        store.finish_activation(first.activation_id, first.trigger.id, "yielded", first.steps)
        quiet = comment("agent-3", markdown="Looking now", reply_to=explicit["comment_id"])
        assert quiet["notified"] == [] and quiet["woken"] == []
        assert store.queued_trigger_count() == 0
        addressed = comment(
            "agent-3", markdown="Please verify", reply_to=explicit["comment_id"], notify=["agent-1"]
        )
        assert addressed["notified"] == addressed["woken"] == ["agent-1"]
        assert [state.trigger.agent_id for state in runtime._claim_wave(1, 2)] == ["agent-1"]
        (delivered,) = runtime._notifications.digest("agent-1")[0]["updates"]
        assert delivered["reply_to"] == explicit["comment_id"]
        assert delivered["comment_id"] == addressed["comment_id"]
        assert runtime._notifications.digest("agent-3")[0]["updates"] == []
    finally:
        store.close()


def test_file_tools_and_guidance_accept_container_workspace_paths(tmp_path):
    store, runtime = world(tmp_path, spec=swarm_problem(population=1), enable_commands=False)
    try:
        (state,) = runtime._claim_wave(1, 1)
        repository = runtime.workspace.canonical_repository
        (repository / "notes.txt").write_text("draft\n", encoding="utf-8")

        def apply(**decision):
            return runtime._apply_workspace_decision(state, decision)[1]

        assert "draft" in apply(action="read_file", path="/workspace/notes.txt")["content"]
        (repository / "long.txt").write_text(
            "".join(f"line {n}\n" for n in range(1, 6)), encoding="utf-8"
        )
        part = apply(action="read_file", path="long.txt", limit=2)
        assert part["next_offset"] == 3 and "Continue with offset=3" in part["content"]
        rest = apply(action="read_file", path="long.txt", offset=part["next_offset"])
        assert rest["next_offset"] is None and "line 5" in rest["content"]
        with pytest.raises(FileNotFoundError) as missing:
            apply(
                action="edit_file", path="missing.txt", edits=[{"old_text": "a", "new_text": "b"}]
            )
        assert model_error(missing.value) == "no file at missing.txt"
        apply(
            action="edit_file",
            path="/workspace/notes.txt",
            edits=[{"old_text": "draft", "new_text": "final"}],
        )
        assert (repository / "notes.txt").read_text(encoding="utf-8") == "final\n"
        assert "final" in apply(action="read_file", path=f"{repository}/notes.txt")["content"]
        with pytest.raises(Exception, match="outside the repository"):
            apply(action="read_file", path="/etc/hostname")

        (repository / "src").mkdir()
        (repository / "src/AGENTS.md").write_text("Keep interfaces stable.\n", encoding="utf-8")
        guidance = activity_guidance(
            runtime.workspace,
            state.trigger.agent_id,
            {"action": "run_command", "argv": ("sh", "-c", "cat /workspace/src/example.py")},
        )
        assert [document["path"] for document in guidance["documents"]] == ["src/AGENTS.md"]
    finally:
        store.close()


def test_read_only_listing_and_long_line_continuation(tmp_path):
    spec = replace(
        problem(max_agents=1),
        team=TeamSpec(capabilities=("read_workspace",)),
    )
    store, runtime = world(tmp_path, spec=spec, enable_commands=False)
    try:
        (state,) = runtime._claim_wave(1, 1)
        names = {
            s["properties"]["action"]["enum"][0]
            for s in runtime._workspace_request(state).schema["oneOf"]
        }
        assert {"list_files", "read_file"} <= names and not {"run_command", "write_file"} & names
        repo = runtime.workspace.canonical_repository
        subdir = repo / "src"
        subdir.mkdir()
        text = "a" * 16383 + "🙂" * 5000
        original = (text + "\r\nend\r\n").encode("utf-8")
        (subdir / "generated.json").write_bytes(original)
        for index in range(100):
            (repo / f"file{index:03}").touch()

        def call(tool, **args):
            return runtime._apply_workspace_decision(state, {"action": tool, **args})[1]

        page = call("list_files")
        assert len(page["entries"]) == page["next_offset"] == 100
        assert call("list_files", path="/workspace", offset=100)["entries"] == [
            {"name": "src", "type": "directory"}
        ]
        assert call("list_files", path="src")["entries"] == [
            {"name": "generated.json", "type": "file"}
        ]
        output = ""
        offset, column = 1, 0
        while offset == 1:
            part = call("read_file", path="src/generated.json", offset=offset, column=column)
            chunk = part["content"].split("\t", 1)[1]
            assert 0 < len(chunk.encode()) <= 16 * 1024
            output += chunk
            offset, column = part["next_offset"], part.get("next_column", 0)
        assert output == text and offset == 2
        assert call("read_file", path="src/generated.json", offset=offset)["content"].endswith(
            "\tend"
        )
        assert (subdir / "generated.json").read_bytes() == original
    finally:
        store.close()
