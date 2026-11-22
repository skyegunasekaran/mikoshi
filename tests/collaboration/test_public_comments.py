"""Public page discussions use explicit, exact peer notifications."""

import pytest
from support.organization import problem, world


def test_comments_are_public_and_only_explicit_notify_delivers_and_wakes(tmp_path):
    store, runtime = world(tmp_path, spec=problem(max_agents=3), enable_commands=False)
    try:
        author, recipient, other = runtime._claim_wave(3, 1)
        peer_id = recipient.trigger.agent_id

        def apply(state, action, **arguments):
            return runtime._apply_workspace_decision(
                state, {"action": action, "path": "/coordination", **arguments}
            )[1]

        comment = apply(author, "comment_page", markdown=f"Finding for @{peer_id}")
        reply = apply(other, "comment_page", markdown="Reviewed", reply_to=comment["comment_id"])
        assert comment["notified"] == reply["notified"] == []
        assert runtime._notifications.digest(peer_id)[0]["updates"] == []
        assert runtime._notifications.digest(author.trigger.agent_id)[0]["updates"] == []

        runtime._apply_workspace_decision(recipient, {"action": "peer_yield"})
        delivered = apply(author, "comment_page", markdown="Please verify", notify=[peer_id])
        assert delivered["notified"] == delivered["woken"] == [peer_id]
        inbox = runtime._notifications.digest(peer_id)[0]["updates"]
        assert [item["comment_id"] for item in inbox] == [delivered["comment_id"]]
        assert runtime._notifications.digest(other.trigger.agent_id)[0]["updates"] == []

        for reader in (author, other):
            view = apply(reader, "view_page")
            assert "Please verify" in view["comments"] and "Reviewed" in view["comments"]
            assert apply(reader, "search_pages", query="Please verify")["total"] == 1
            assert (
                runtime._apply_workspace_decision(
                    reader, {"action": "read_comment", "comment_id": reply["comment_id"]}
                )[1]["reply_to"]
                == comment["comment_id"]
            )
        schema = runtime._tools_for(author)["comment_page"].schema
        assert set(schema["properties"]) == {"action", "path", "markdown", "reply_to", "notify"}
        assert not store.event_observer_errors
    finally:
        store.close()


def test_comment_validation_is_atomic_and_preserves_page_access(tmp_path):
    store, runtime = world(tmp_path, spec=problem(max_agents=3), enable_commands=False)
    try:
        author, reader, outsider = runtime._claim_wave(3, 1)

        def apply(state, action, **arguments):
            return runtime._apply_workspace_decision(
                state, {"action": action, "path": "/notes", **arguments}
            )[1]

        apply(author, "add_page", title="Notes", markdown="Project notes")
        public = apply(author, "comment_page", markdown="Shared evidence")
        before = store._connection.execute("SELECT COUNT(*) FROM page_comments").fetchone()[0]
        for arguments, error in (
            ({"notify": ["unknown-peer"]}, "unknown peer IDs"),
            ({"notify": None}, "must be a list"),
            ({"path": "/coordination", "reply_to": public["comment_id"]}, "not a comment on"),
        ):
            with pytest.raises(ValueError, match=error):
                apply(author, "comment_page", markdown="Rejected", **arguments)
        assert (
            store._connection.execute("SELECT COUNT(*) FROM page_comments").fetchone()[0] == before
        )
        assert runtime._notifications.digest(reader.trigger.agent_id)[0]["updates"] == []

        with store._connection:
            store._connection.execute(
                "DELETE FROM project_members WHERE agent_id=?", (outsider.trigger.agent_id,)
            )
        with pytest.raises(ValueError, match="not a member"):
            apply(outsider, "view_page")
        with pytest.raises(ValueError, match="not a member"):
            runtime._apply_workspace_decision(
                outsider, {"action": "read_comment", "comment_id": public["comment_id"]}
            )
        with pytest.raises(ValueError, match="unknown peer IDs"):
            apply(author, "comment_page", markdown="Rejected", notify=[outsider.trigger.agent_id])
        assert apply(reader, "view_page")["comments"].count("Shared evidence") == 1
        assert (
            store._connection.execute("SELECT COUNT(*) FROM page_comments").fetchone()[0] == before
        )
    finally:
        store.close()
