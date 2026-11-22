"""Peers coordinate through shared, editable pages with attributed history."""

import pytest
from support.organization import problem, world


def test_every_peer_starts_at_the_same_editable_coordination_page(tmp_path):
    store, runtime = world(tmp_path, spec=problem(max_agents=2), enable_commands=False)
    try:
        first, second = runtime._claim_wave(2, 1)

        def apply(state, action, **arguments):
            return runtime._apply_workspace_decision(state, {"action": action, **arguments})[1]

        for peer in (first, second):
            context = runtime._pages.context(peer)
            assert {page["path"] for page in context["entry_points"]} == {
                "/objective",
                "/coordination",
            }
            assert {page["path"] for page in apply(peer, "list_pages")["pages"]} == {
                "/objective",
                "/coordination",
            }
        (scope,) = store._connection.execute(
            "SELECT project_id FROM shared_pages WHERE id='coordination'"
        ).fetchone()
        assert scope is None

        initial = apply(first, "view_page", path="/coordination")["markdown"]
        apply(
            first,
            "edit_page",
            path="/coordination",
            edits=[{"old_text": initial, "new_text": "# Plan\n\nInvestigate the parser."}],
        )
        apply(
            second,
            "edit_page",
            path="/coordination",
            edits=[{"old_text": "Investigate the parser.", "new_text": "Parser reviewed."}],
        )
        shared = apply(first, "view_page", path="/coordination")
        assert shared["markdown"] == "# Plan\n\nParser reviewed."
        assert shared["revision"] == 3
        assert apply(second, "search_pages", query="Parser reviewed")["total"] == 1
        assert (runtime.workspace.root / "pages" / "coordination.md").read_text(
            encoding="utf-8"
        ) == shared["markdown"]
        assert {path.name for path in (runtime.workspace.root / "pages").glob("*.md")} == {
            "objective.md",
            "coordination.md",
        }
    finally:
        store.close()


def test_peers_can_edit_each_others_pages_with_exact_edits_and_attributed_history(tmp_path):
    store, runtime = world(tmp_path, spec=problem(max_agents=2), enable_commands=False)
    try:
        author, other = runtime._claim_wave(2, 1)

        def apply(state, action, **arguments):
            return runtime._apply_workspace_decision(state, {"action": action, **arguments})[1]

        created = apply(author, "add_page", path="/plan", title="Plan", markdown="Areas: tbd")
        updated = apply(
            other,
            "edit_page",
            path="/plan",
            edits=[{"old_text": "tbd", "new_text": "parser and tests"}],
        )
        assert updated["revision"] == created["revision"] + 1
        assert not {"work_id", "status", "author_id"}.intersection(updated)
        with pytest.raises(ValueError, match="Could not find"):
            apply(
                author,
                "edit_page",
                path="/plan",
                edits=[{"old_text": "tbd", "new_text": "stale replacement"}],
            )
        assert apply(author, "view_page", path="/plan")["markdown"] == "Areas: parser and tests"
        history = store._connection.execute(
            "SELECT r.revision, r.markdown, r.author_id FROM page_revisions r "
            "JOIN shared_pages p ON p.id=r.page_id WHERE p.name='plan' ORDER BY r.revision"
        ).fetchall()
        assert [tuple(row) for row in history] == [
            (1, "Areas: tbd", author.trigger.agent_id),
            (2, "Areas: parser and tests", other.trigger.agent_id),
        ]
        edit = next(
            event
            for event in store.list_events()
            if event["action_type"] == "edit_page" and event["outcome"] == "succeeded"
        )
        assert edit["payload"]["previous_resource_version"] != edit["payload"]["resource_version"]

        with pytest.raises(ValueError, match="read-only"):
            apply(
                other,
                "edit_page",
                path="/objective",
                edits=[{"old_text": "Constraints", "new_text": "Notes"}],
            )
        comment = apply(other, "comment_page", path="/objective", markdown="Objective reviewed.")
        assert (
            str(comment["comment_id"]) in apply(author, "view_page", path="/objective")["comments"]
        )
        assert not store.event_observer_errors
    finally:
        store.close()
