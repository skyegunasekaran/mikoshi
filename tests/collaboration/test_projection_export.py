"""Derived page exports avoid rewriting unchanged files and repair external edits."""

import os

from support.organization import world


def test_page_exports_skip_unchanged_files_and_repair_external_changes(tmp_path, monkeypatch):
    store, runtime = world(tmp_path, enable_commands=False)
    try:
        state = runtime._claim_wave(1, 1)[0]

        def apply(action, path, **arguments):
            return runtime._apply_workspace_decision(
                state, {"action": action, "path": path, **arguments}
            )[1]

        apply("add_page", "/alpha", title="Alpha", markdown="Initial alpha")
        apply("add_page", "/beta", title="Beta", markdown="Initial beta")
        root = runtime.workspace.root / "pages"

        def export_path(name):
            (page_id,) = store._connection.execute(
                "SELECT id FROM shared_pages WHERE name=?", (name,)
            ).fetchone()
            return root / f"{page_id}.md"

        alpha_path, beta_path = export_path("alpha"), export_path("beta")
        replaced = []
        path_type = type(root)
        replace = path_type.replace

        def track_replace(path, target):
            replaced.append(target)
            return replace(path, target)

        monkeypatch.setattr(path_type, "replace", track_replace)
        runtime._pages.export()
        assert replaced == [root / "history.json"]
        replaced.clear()
        runtime._pages.export()
        assert replaced == []
        with monkeypatch.context() as checks:

            def unexpected(*args):
                raise AssertionError("Budget checks cannot render pages or serialize their history")

            checks.setattr(runtime._pages, "render", unexpected)
            checks.setattr("mikoshi.tools.shared_pages.records", unexpected)
            runtime._apply_workspace_decision(state, {"action": "check_budget"})
            assert not store.event_observer_errors

        apply("edit_page", "/alpha", edits=[{"old_text": "Initial", "new_text": "Updated"}])
        assert "Updated alpha" in alpha_path.read_text(encoding="utf-8")
        assert alpha_path in replaced
        assert beta_path not in replaced

        # Replacing a projection while preserving its size and mtime must still
        # invalidate the cache; the inode is part of the recorded identity.
        expected = beta_path.read_text(encoding="utf-8")
        original = beta_path.stat()
        external = beta_path.with_suffix(".external")
        external.write_text("X" * len(expected), encoding="utf-8")
        os.utime(external, ns=(original.st_atime_ns, original.st_mtime_ns))
        external.replace(beta_path)
        alpha_path.unlink()
        replaced.clear()
        runtime._pages.export()
        assert beta_path.read_text(encoding="utf-8") == expected
        assert "Updated alpha" in alpha_path.read_text(encoding="utf-8")
        assert set(replaced) == {alpha_path, beta_path, root / "history.json"}
    finally:
        store.close()
