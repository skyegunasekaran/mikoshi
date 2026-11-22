"""Retained candidates are independent, verified copies of working-tree contents."""

import json
import os
from pathlib import Path

import pytest
from support.filesystem import symlink_or_skip

from mikoshi.persistence.artifacts import (
    freeze_candidate,
    load_verified_artifacts,
    snapshot_digest,
    snapshot_repository,
)
from mikoshi.schema import create_run, update_run


def test_freeze_retains_uncommitted_contents_modes_and_empty_directories(tmp_path):
    source = tmp_path / "repository"
    source.mkdir()
    (source / "empty").mkdir()
    script = source / "run.sh"
    script.write_text("original\n")
    script.chmod(0o755)
    source_mode = script.stat().st_mode & 0o777
    (source / ".git").write_text("gitdir: /outside/private/metadata")
    output = tmp_path / "output"
    output.mkdir()
    create_run(output, input={"model": "example"})
    result = freeze_candidate(source, output)
    script.write_text("changed later\n")

    verified = load_verified_artifacts(output)
    candidate = Path(verified["candidate_path"])
    assert verified["candidate_sha256"] == result["candidate_sha256"]
    assert (candidate / "run.sh").read_text() == "original\n"
    assert (candidate / "run.sh").stat().st_mode & 0o777 == source_mode
    assert (candidate / "empty").is_dir()
    assert not (candidate / ".git").exists()
    assert result["input_sha256"]


@pytest.mark.parametrize("change", ["contents", "git_metadata", "evidence", "provenance"])
def test_load_rejects_changed_candidate_or_evidence(tmp_path, change):
    source = tmp_path / "repository"
    source.mkdir()
    (source / "answer").write_text("retained")
    output = tmp_path / "output"
    output.mkdir()
    create_run(output, input={})
    freeze_candidate(source, output)
    if change == "contents":
        (output / "candidate" / "answer").write_text("tampered")
    elif change == "git_metadata":
        (output / "candidate" / ".git").write_text("gitdir: /outside")
    elif change == "evidence":
        update_run(output, input={"different": True})
    else:
        update_run(output, provenance={"changed": True})
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_verified_artifacts(output)


@pytest.mark.parametrize("special", ["symlink", "fifo"])
def test_snapshot_refuses_links_and_special_files(tmp_path, special):
    source = tmp_path / "repository"
    source.mkdir()
    if special == "symlink":
        symlink_or_skip(tmp_path / "outside", source / "escape")
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("this host has no POSIX FIFO support")
        os.mkfifo(source / "pipe")
    with pytest.raises(ValueError, match="symlinks and special files"):
        snapshot_repository(source, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


def test_empty_snapshot_and_unsafe_destinations(tmp_path):
    source = tmp_path / "repository"
    source.mkdir()
    assert snapshot_repository(source, tmp_path / "copy") == snapshot_digest(source)
    with pytest.raises(ValueError, match="separate directories"):
        snapshot_repository(source, source / "nested")
    output = tmp_path / "output"
    output.mkdir()
    create_run(output, input={})
    freeze_candidate(source, output)
    manifest_path = output / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["candidate"]["path"] = "../repository"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid candidate artifact manifest"):
        load_verified_artifacts(output)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_freeze_cleans_readonly_staging_without_masking_failure(
    tmp_path, monkeypatch, cleanup_fails
):
    from mikoshi.persistence import artifacts

    source = tmp_path / "repository"
    source.mkdir()
    readonly = source / "readonly.txt"
    readonly.write_text("Retain original", encoding="utf-8")
    readonly.chmod(0o444)
    output = tmp_path / "output"
    create_run(output, input={})
    copy = artifacts.snapshot_repository

    def failed_snapshot(*args):
        copy(*args)
        raise RuntimeError("injected failure after copying")

    monkeypatch.setattr(artifacts, "snapshot_repository", failed_snapshot)
    if cleanup_fails:

        def failed_cleanup(path):
            raise PermissionError("injected cleanup failure")

        monkeypatch.setattr(artifacts, "_remove_partial_candidate", failed_cleanup)
    with pytest.raises(RuntimeError, match="injected failure after copying") as raised:
        freeze_candidate(source, output)
    assert readonly.read_text(encoding="utf-8") == "Retain original"
    assert not (output / "candidate").exists()
    assert json.loads((output / "run.json").read_text(encoding="utf-8"))["candidate"] is None
    if cleanup_fails:
        assert list(output.glob(".candidate-*"))
        assert "cleanup could not be confirmed" in raised.value.__notes__[0]
    else:
        assert list(output.glob(".candidate-*")) == []
