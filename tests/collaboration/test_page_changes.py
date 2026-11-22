"""One delivery window covers peer edits and unaddressed public discussion."""

import json

from support.organization import problem, world

from mikoshi.decisions import DecisionResult


def _commit(runtime, reader, request, *, compacted=False):
    result = (
        DecisionResult(
            {"action": "check_budget"},
            {"role": "assistant", "compaction_checkpoints": [{"id": "checkpoint"}]},
        )
        if compacted
        else {"action": "check_budget"}
    )
    reader.steps += 1
    runtime._conversations.append_exchange(
        reader.trigger.agent_id, f"pages:{reader.steps}", request, result, {}
    )
    runtime._notifications.acknowledge(
        reader.trigger.agent_id, request.metadata["notification_cursor"]
    )


def test_observations_are_self_contained_and_changes_survive_retries_compaction_and_inflight_edits(
    tmp_path,
):
    store, runtime = world(tmp_path, spec=problem(max_agents=3), enable_commands=False)
    try:
        reader, writer, other = runtime._claim_wave(3, 1)
        repository = runtime.workspace.canonical_repository
        (repository / "AGENTS.md").write_text("Shared startup guidance", encoding="utf-8")
        initial = runtime._workspace_request(reader)
        assert json.loads(initial.model_input_text)["changes"]["pages"] == []
        _commit(runtime, reader, initial)

        def apply(state, action, **arguments):
            return runtime._apply_workspace_decision(state, {"action": action, **arguments})[1]

        apply(writer, "add_page", path="/notes", title="Findings", markdown="First evidence")
        apply(
            other, "edit_page", path="/notes", edits=[{"old_text": "First", "new_text": "Second"}]
        )
        apply(reader, "add_page", path="/own", title="Own notes", markdown="Personal progress")
        comment = apply(writer, "comment_page", path="/notes", markdown="Public discussion")
        addressed = apply(
            writer,
            "comment_page",
            path="/objective",
            markdown="Please review",
            notify=[reader.trigger.agent_id],
        )
        request = runtime._workspace_request(reader)
        observed = json.loads(request.model_input_text)
        changes = observed["changes"]
        assert changes["since_tick"] == json.loads(initial.input_text)["changes"]["through_tick"]
        assert changes["through_tick"] == store.current_tick and changes["has_more"] is False
        assert changes["pages"] == [
            {
                "path": "/notes",
                "title": "Findings",
                "revision": 2,
                "author_id": other.trigger.agent_id,
                "tick": changes["pages"][0]["tick"],
                "comments": 1,
                "latest_comment_id": comment["comment_id"],
            }
        ]
        assert observed["inbox"]["updates"][0]["comment_id"] == addressed["comment_id"]
        assert observed["agent_id"] == reader.trigger.agent_id and "team" in observed
        assert (
            not {
                "observation",
                "set",
                "remove",
                "notifications",
                "recent_discussion",
                "since_snapshot",
                "page_changes",
            }
            & observed.keys()
        )
        assert "First evidence" not in request.model_input_text
        assert "Public discussion" not in request.model_input_text
        assert "Shared startup guidance" not in request.model_input_text
        assert "Shared startup guidance" in request.compaction_input_text
        assert json.loads(runtime._workspace_request(reader).model_input_text)["changes"] == changes

        apply(
            writer, "edit_page", path="/notes", edits=[{"old_text": "Second", "new_text": "Third"}]
        )
        assert json.loads(request.compaction_input_text)["changes"] == changes
        _commit(runtime, reader, request, compacted=True)
        after = runtime._workspace_request(reader)
        (page,) = json.loads(after.model_input_text)["changes"]["pages"]
        assert page["revision"] == 3 and "comments" not in page
        assert json.loads(after.model_input_text)["inbox"]["updates"] == []
        _commit(runtime, reader, after)
        assert (
            json.loads(runtime._workspace_request(reader).model_input_text)["changes"]["pages"]
            == []
        )
    finally:
        store.close()


def test_changes_carry_overflow_forward_and_respect_project_access(tmp_path, monkeypatch):
    monkeypatch.setattr("mikoshi.runtime.notifications.CHANGED_PAGE_LIMIT", 2)
    store, runtime = world(tmp_path, spec=problem(max_agents=3), enable_commands=False)
    try:
        reader, writer, outsider = runtime._claim_wave(3, 1)
        since = store.current_tick
        for index in range(3):
            runtime._apply_workspace_decision(
                writer,
                {
                    "action": "add_page",
                    "path": f"/notes-{index}",
                    "title": f"Notes {index}",
                    "markdown": "Evidence",
                },
            )
        changes = runtime._notifications.changes(reader.trigger.agent_id, since, store.current_tick)
        # The oldest pages arrive first and the window stops after them.
        assert [page["path"] for page in changes["pages"]] == ["/notes-1", "/notes-0"]
        assert changes["has_more"] is True
        assert changes["through_tick"] == changes["pages"][0]["tick"] < store.current_tick
        rest = runtime._notifications.changes(
            reader.trigger.agent_id, changes["through_tick"], store.current_tick
        )
        assert [page["path"] for page in rest["pages"]] == ["/notes-2"]
        assert rest["has_more"] is False and rest["through_tick"] == store.current_tick

        with store._connection:
            store._connection.execute(
                "DELETE FROM project_members WHERE agent_id=?", (outsider.trigger.agent_id,)
            )
        assert (
            runtime._notifications.changes(outsider.trigger.agent_id, since, store.current_tick)[
                "pages"
            ]
            == []
        )
        latest = runtime._apply_workspace_decision(
            writer,
            {
                "action": "comment_page",
                "path": "/coordination",
                "markdown": "Shared plan",
            },
        )[1]
        visible = runtime._notifications.changes(
            outsider.trigger.agent_id, since, store.current_tick
        )
        assert [page["path"] for page in visible["pages"]] == ["/coordination"]
        assert visible["pages"][0]["latest_comment_id"] == latest["comment_id"]
        assert "revision" not in visible["pages"][0]
        assert visible["has_more"] is False
        assert runtime._notifications.digest(reader.trigger.agent_id)[0]["updates"] == []
        assert store.queued_trigger_count() == 0
    finally:
        store.close()
