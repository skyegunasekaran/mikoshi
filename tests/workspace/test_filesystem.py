"""Real filesystem operations retain the same confinement on POSIX and Windows."""

import os
import subprocess
from contextlib import closing
from types import SimpleNamespace

import pytest

from mikoshi.filesystem import UnsafePathError, open_directory, regular_file, validate_component
from mikoshi.persistence.artifacts import snapshot_repository
from mikoshi.persistence.store import WorldStore
from mikoshi.problem import ObjectiveSpec, ProblemSpec, ResourceLimits
from mikoshi.runtime.organization import bootstrap
from mikoshi.tools.workspace import LiveWorkspace, RepositoryIdentity, WorkspaceAccessError


@pytest.fixture
def workspace(tmp_path):
    source = tmp_path / "target"
    source.mkdir()
    with closing(WorldStore()) as store:
        bootstrap(
            store,
            ProblemSpec(ObjectiveSpec("test", "Test", "Inspect files"), ResourceLimits(1)),
        )
        workspace = LiveWorkspace(tmp_path / "controls", store, source)
        workspace.initialize(RepositoryIdentity(str(source.resolve()), None))
        yield workspace


def test_native_create_read_edit_list_and_snapshot(workspace, tmp_path):
    first = workspace.write_text("agent-1", "notes/ünicode.txt", "Evidence α\n")
    assert first["material_change"] is True
    assert workspace.read_text("agent-1", "notes/ünicode.txt")["content"] == "Evidence α\n"
    changed = workspace.edit_text(
        "agent-1", "notes/ünicode.txt", [{"old_text": "α", "new_text": "β"}]
    )
    assert changed["previous_resource_version"]["sha256"] == first["sha256"]
    assert workspace.list_files("agent-1")["entries"] == [{"name": "notes", "type": "directory"}]
    assert workspace.list_files("agent-1", "notes")["entries"] == [
        {"name": "ünicode.txt", "type": "file"}
    ]
    snapshot_repository(workspace.canonical_repository, tmp_path / "retained")
    assert (tmp_path / "retained" / "notes" / "ünicode.txt").read_text(
        encoding="utf-8"
    ) == "Evidence β\n"


def test_hardlinks_cannot_be_read_replaced_or_snapshotted(workspace, tmp_path):
    outside = tmp_path / "private.txt"
    outside.write_text("Private evidence", encoding="utf-8")
    os.link(outside, workspace.canonical_repository / "alias.txt")
    with pytest.raises(WorkspaceAccessError, match="hard links"):
        workspace.read_text("agent-1", "alias.txt")
    with pytest.raises(WorkspaceAccessError, match="hard links"):
        workspace.write_text("agent-1", "alias.txt", "Overwrite")
    assert workspace.list_files("agent-1")["entries"] == []
    with pytest.raises(ValueError, match="hard links"):
        snapshot_repository(workspace.canonical_repository, tmp_path / "copy")
    assert outside.read_text(encoding="utf-8") == "Private evidence"


def test_atomic_creation_does_not_replace_a_concurrent_new_file(tmp_path):
    target = tmp_path / "new.txt"
    with open_directory(tmp_path) as directory:
        with pytest.raises(FileExistsError):
            directory.write_file(
                target.name,
                b"Staged bytes",
                0o644,
                replace=False,
                before_publish=lambda: target.write_bytes(b"Concurrent bytes"),
            )
    assert target.read_bytes() == b"Concurrent bytes"
    assert [path.name for path in tmp_path.iterdir()] == [target.name]


def test_staging_is_removed_when_publication_check_fails(tmp_path):
    def interrupted():
        raise ValueError("changed target")

    with open_directory(tmp_path) as directory:
        with pytest.raises(ValueError, match="changed target"):
            directory.write_file(
                "new.txt", b"Staged bytes", 0o444, replace=False, before_publish=interrupted
            )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "name",
    [
        "C:relative",
        "C:",
        "stream:secret",
        "trailing.",
        "trailing ",
        "NUL",
        "con.txt",
        "COM¹.log",
        "lpt9",
        "aux .txt",
        "a?b",
        "a\nb",
    ],
)
def test_windows_component_validation_rejects_ambiguous_names_on_every_host(name):
    with pytest.raises(UnsafePathError):
        validate_component(name, windows=True)


def test_git_environment_keeps_windows_system_variables_without_provider_credentials(monkeypatch):
    from mikoshi.tools import workspace as module

    monkeypatch.setattr(
        module,
        "os",
        SimpleNamespace(
            name="nt",
            devnull="nul",
            environ={
                "sYsTeMrOoT": "C:\\Windows",
                "WINDIR": "C:\\Windows",
                "PATH": "bin",
                "OPENAI_API_KEY": "private",
                "GIT_CONFIG_COUNT": "99",
            },
        ),
    )
    _, environment = module._git_invocation(("rev-parse", "HEAD"))
    assert environment["sYsTeMrOoT"] == "C:\\Windows"
    assert environment["WINDIR"] == "C:\\Windows"
    assert "OPENAI_API_KEY" not in environment and "GIT_CONFIG_COUNT" not in environment
    assert environment["GIT_CONFIG_SYSTEM"] == "nul"


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse points require Windows")
def test_native_windows_junctions_are_not_traversed_or_snapshotted(workspace, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("Private", encoding="utf-8")
    junction = workspace.canonical_repository / "junction"
    subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
        check=True,
        capture_output=True,
    )
    with pytest.raises(WorkspaceAccessError, match="reparse"):
        workspace.read_text("agent-1", "junction/private.txt")
    with pytest.raises(WorkspaceAccessError, match="reparse"):
        workspace.write_text("agent-1", "junction/private.txt", "Changed")
    assert workspace.list_files("agent-1")["entries"] == []
    with pytest.raises(ValueError, match="reparse"):
        snapshot_repository(workspace.canonical_repository, tmp_path / "copy")
    assert (outside / "private.txt").read_text(encoding="utf-8") == "Private"


@pytest.mark.skipif(os.name != "nt", reason="Windows directory handle sharing requires Windows")
def test_native_windows_pinned_directory_cannot_be_renamed(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "evidence.txt").write_text("Evidence", encoding="utf-8")
    with open_directory(tmp_path, ("nested",)) as directory:
        with pytest.raises(OSError):
            nested.rename(tmp_path / "moved")
        with regular_file(directory, "evidence.txt") as descriptor:
            assert not os.get_inheritable(descriptor)
            assert os.read(descriptor, 100) == b"Evidence"
    nested.rename(tmp_path / "moved")
    assert (tmp_path / "moved" / "evidence.txt").read_text(encoding="utf-8") == "Evidence"


@pytest.mark.skipif(os.name != "nt", reason="Windows path semantics require Windows")
@pytest.mark.parametrize("path", ["NUL", "file.txt:stream", "C:relative", "trailing.", "dir /file"])
def test_native_windows_workspace_rejects_device_stream_and_alias_paths(workspace, path):
    with pytest.raises(WorkspaceAccessError):
        workspace.write_text("agent-1", path, "Rejected")


def test_windows_backend_rejects_network_and_device_roots_without_opening_them():
    from mikoshi.windows_filesystem import WindowsDirectory

    for anchor in ("\\\\server\\share\\", "\\\\?\\C:\\", "\\\\.\\C:\\"):
        with pytest.raises(UnsafePathError, match="local fixed drive"):
            WindowsDirectory.anchor(anchor)
