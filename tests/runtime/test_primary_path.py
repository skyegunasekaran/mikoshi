"""Peers edit shared pages and files, preserve results, and yield."""

import json
import os

import pytest
from support.boundaries import run_for
from support.filesystem import symlink_or_skip
from support.organization import problem, world
from support.runtime import create_repository, git

from mikoshi.runtime.context_audit import audit_database
from mikoshi.tools.workspace import WorkspaceAccessError


class WorkingPeer:
    def __init__(self):
        self.requests = []

    def generate_batch(self, requests):
        results = []
        for request in requests:
            self.requests.append(request)
            actions = [
                {
                    "action": "add_page",
                    "path": "/inspect",
                    "title": "Inspect",
                    "markdown": "Record findings",
                },
                {"action": "write_file", "path": "findings.md", "content": "Initial findings"},
                {"action": "read_file", "path": "findings.md"},
                {
                    "action": "edit_file",
                    "path": "findings.md",
                    "edits": [{"old_text": "stale", "new_text": "discard"}],
                },
                {
                    "action": "edit_file",
                    "path": "findings.md",
                    "edits": [{"old_text": "Initial", "new_text": "Reviewed"}],
                },
                {
                    "action": "edit_page",
                    "path": "/inspect",
                    "edits": [{"old_text": "Record findings", "new_text": "Findings recorded"}],
                },
                {"action": "write_file", "path": "followup.md", "content": "Further evidence"},
            ]
            index = len(self.requests) - 1
            results.append(actions[index] if index < len(actions) else {"action": "peer_yield"})
        return results


def test_peer_lifecycle_preserves_results_and_recovers_from_rejected_edit(tmp_path):
    repository = tmp_path / "checkout"
    source = create_repository(tmp_path)
    git(source, "worktree", "add", "-b", "fixture", str(repository))
    (repository / "existing.txt").write_text("Committed content", encoding="utf-8")
    git(repository, "add", "existing.txt")
    git(repository, "commit", "-m", "Existing work")
    (repository / "existing.txt").write_text("Uncommitted user content", encoding="utf-8")
    (repository / "untracked.txt").write_text("Untracked user content", encoding="utf-8")
    provider = WorkingPeer()
    store, runtime = world(
        tmp_path,
        provider,
        problem(max_agents=1),
        repository=repository,
    )
    try:
        assert runtime.workspace.canonical_repository == repository
        assert (
            runtime.workspace.read_text("participant", "existing.txt")["content"]
            == "Uncommitted user content"
        )
        assert runtime._command_runner is not None
        run_for(runtime)
        assert runtime.stop_reason == "all_agents_yielded"
        root = runtime.workspace.canonical_repository
        assert (root / "findings.md").read_text(encoding="utf-8") == "Reviewed findings"
        assert (root / "followup.md").read_text(encoding="utf-8") == "Further evidence"
        assert (repository / "existing.txt").read_text(
            encoding="utf-8"
        ) == "Uncommitted user content"
        assert (repository / "untracked.txt").read_text(
            encoding="utf-8"
        ) == "Untracked user content"
        assert not (runtime.workspace.root / "repository").exists()
        assert "Initial findings" in json.dumps(provider.requests[3].history)
        assert provider.requests[4].history[-1]["content"].startswith("Could not find the text")
        page = store._connection.execute(
            "SELECT r.markdown, p.revision FROM shared_pages p "
            "JOIN page_revisions r ON r.page_id=p.id AND r.revision=p.revision "
            "WHERE p.name='inspect'"
        ).fetchone()
        assert tuple(page) == ("Findings recorded", 2)
        assert [
            event["action_type"]
            for event in store.list_events()
            if event["action_type"] in {"add_page", "edit_page"}
        ] == ["add_page", "edit_page"]
        assert len(store.list_activations()) == 1
        assert not store.queued_trigger_count()
        assert sum(
            len(json.loads(row[0]))
            for row in store._connection.execute(
                "SELECT messages FROM agent_memory_exchanges WHERE agent_id = 'participant'"
            )
        ) > len(provider.requests[-1].history)
        private = tmp_path / "private.txt"
        private.write_text("Outside the target", encoding="utf-8")
        os.link(private, root / "hardlinked")
        with pytest.raises(WorkspaceAccessError):
            runtime.workspace.read_text("participant", "../private.txt")
        with pytest.raises(WorkspaceAccessError):
            runtime.workspace.read_text("participant", "hardlinked")
        archive = tmp_path / "archive.sqlite"
        store.backup(archive)
        audit = audit_database(archive)
        assert audit["text_bytes"]["state_observations"] > 0
        assert "Initial findings" not in json.dumps(audit)
    finally:
        store.close()


def test_peer_lifecycle_shares_page_edits_and_keeps_each_authors_history(tmp_path):
    class ReviewingPeers:
        concurrency = 1

        def __init__(self):
            self.requests = []
            self.steps = {}

        def generate_batch(self, requests):
            actions = {
                "participant": [
                    {
                        "action": "add_page",
                        "path": "/inspect",
                        "title": "Inspection findings",
                        "markdown": "Initial evidence",
                    },
                    {
                        "action": "edit_page",
                        "path": "/inspect",
                        "edits": [{"old_text": "Initial", "new_text": "Reviewed"}],
                    },
                ],
                "worker_a": [
                    {"action": "view_page", "path": "/inspect"},
                    {
                        "action": "edit_page",
                        "path": "/inspect",
                        "edits": [{"old_text": "Reviewed", "new_text": "Verified"}],
                    },
                ],
            }
            decisions = []
            for request in requests:
                self.requests.append(request)
                actor = request.metadata["agent_id"]
                step = self.steps.get(actor, 0)
                self.steps[actor] = step + 1
                decisions.append(
                    actions[actor][step] if step < len(actions[actor]) else {"action": "peer_yield"}
                )
            return decisions

    provider = ReviewingPeers()
    store, runtime = world(
        tmp_path,
        provider,
        problem(max_agents=2),
        enable_commands=False,
    )
    try:
        run_for(runtime)
        assert runtime.stop_reason == "all_agents_yielded"
        revisions = store._connection.execute(
            "SELECT r.author_id, r.markdown FROM page_revisions r "
            "JOIN shared_pages p ON p.id=r.page_id WHERE p.name='inspect' "
            "ORDER BY r.revision"
        ).fetchall()
        assert [tuple(row) for row in revisions] == [
            ("participant", "Initial evidence"),
            ("participant", "Reviewed evidence"),
            ("worker_a", "Verified evidence"),
        ]
        assert {activation["agent_id"] for activation in store.list_activations()} == {
            "participant",
            "worker_a",
        }
        assert not store.queued_trigger_count()
        assert all(
            "work_state" not in json.loads(request.input_text) for request in provider.requests
        )
        runtime._pages.export()
        history = json.loads(
            (runtime.workspace.root / "pages/history.json").read_text(encoding="utf-8")
        )
        page_id = next(page["id"] for page in history["shared_pages"] if page["name"] == "inspect")
        assert [
            row["markdown"] for row in history["page_revisions"] if row["page_id"] == page_id
        ] == ["Initial evidence", "Reviewed evidence", "Verified evidence"]
        assert [
            event["agent_id"]
            for event in store.list_events()
            if event["action_type"] == "edit_page"
        ] == ["participant", "worker_a"]
    finally:
        store.close()


def test_workspace_file_tools_reject_symbolic_links(tmp_path):
    store, runtime = world(tmp_path, spec=problem(max_agents=1), enable_commands=False)
    try:
        private = tmp_path / "private"
        private.mkdir()
        (private / "secret.txt").write_text("Outside the target", encoding="utf-8")
        root = runtime.workspace.canonical_repository
        symlink_or_skip(private / "secret.txt", root / "linked")
        symlink_or_skip(private, root / "alias", target_is_directory=True)
        assert runtime.workspace.list_files("participant")["entries"] == []
        with pytest.raises(WorkspaceAccessError):
            runtime.workspace.read_text("participant", "linked")
        with pytest.raises(WorkspaceAccessError):
            runtime.workspace.list_files("participant", "alias")
    finally:
        store.close()
