"""Durable candidate snapshots and verifiable evidence for independent evaluation."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

from mikoshi.filesystem import (
    FileChangedError,
    UnsafePathError,
    is_link_or_reparse,
    is_reparse,
    open_directory,
    regular_file,
)
from mikoshi.schema import RUN_FILE, read_document, update_run

EVIDENCE_FIELDS = ("resolved", "provenance", "summary", "provider_metrics", "budget")


def _input_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _directory(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    try:
        with open_directory(path):
            pass
    except (OSError, UnsafePathError) as error:
        raise ValueError(
            f"snapshot requires a real directory without symlinks or reparse points: {path}"
        ) from error
    return path


def _stable_file(info):
    return info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode, info.st_nlink


def _hash_file(directory, name):
    digest = hashlib.sha256()
    with regular_file(directory, name) as descriptor:
        before = os.fstat(descriptor)
        size = 0
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        if _stable_file(before) != _stable_file(os.fstat(descriptor)):
            raise FileChangedError("repository changed while hashing its snapshot")
    return digest.hexdigest(), size, stat.S_IMODE(before.st_mode)


def _excluded(name, exclude_git):
    return exclude_git and (name == ".git" or (os.name == "nt" and name.casefold() == ".git"))


def _inventory(repository: Path, *, exclude_git: bool = True) -> list[dict[str, Any]]:
    repository = _directory(repository)
    entries = []

    def visit(directory, relative):
        entries.append(
            {"path": relative, "mode": stat.S_IMODE(directory.info().st_mode), "type": "directory"}
        )
        for child in sorted(directory.entries(), key=lambda child: child.name):
            if _excluded(child.name, exclude_git):
                continue
            child_path = child.name if relative == "." else f"{relative}/{child.name}"
            if is_reparse(child.info):
                raise UnsafePathError("snapshot refuses reparse points")
            if stat.S_ISDIR(child.info.st_mode):
                with closing(directory.child(child.name)) as nested:
                    visit(nested, child_path)
            elif stat.S_ISREG(child.info.st_mode):
                digest, size, mode = _hash_file(directory, child.name)
                entries.append(
                    {
                        "path": child_path,
                        "mode": mode,
                        "type": "file",
                        "sha256": digest,
                        "size": size,
                    }
                )
            else:
                raise UnsafePathError("snapshot refuses special files")

    try:
        with open_directory(repository) as directory:
            visit(directory, ".")
    except UnsafePathError as error:
        raise ValueError(
            f"snapshot refuses symlinks and special files, hard links, and reparse points: {repository}"
        ) from error
    return entries


def _copy_tree(source, destination):
    for child in sorted(source.entries(), key=lambda child: child.name):
        if _excluded(child.name, True):
            continue
        if is_reparse(child.info):
            raise UnsafePathError("snapshot refuses reparse points")
        if stat.S_ISDIR(child.info.st_mode):
            with (
                closing(source.child(child.name)) as nested_source,
                closing(
                    destination.child(child.name, create=True, exclusive=True)
                ) as nested_destination,
            ):
                _copy_tree(nested_source, nested_destination)
        elif stat.S_ISREG(child.info.st_mode):
            with regular_file(source, child.name) as source_descriptor:
                before = os.fstat(source_descriptor)
                target = destination.open_file(child.name, create=True)
                try:
                    with (
                        os.fdopen(source_descriptor, "rb", closefd=False) as input_stream,
                        os.fdopen(target, "wb", closefd=False) as output_stream,
                    ):
                        shutil.copyfileobj(input_stream, output_stream)
                        output_stream.flush()
                        os.fsync(target)
                        destination.file_mode(target, stat.S_IMODE(before.st_mode))
                    if _stable_file(before) != _stable_file(os.fstat(source_descriptor)):
                        raise FileChangedError("repository changed while copying its snapshot")
                finally:
                    os.close(target)
        else:
            raise UnsafePathError("snapshot refuses symlinks and special files")
    destination.set_mode(stat.S_IMODE(source.info().st_mode))


def _digest(entries: list[dict[str, Any]]) -> str:
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def snapshot_digest(repository: Path) -> str:
    """Hash paths, contents, modes and empty directories, excluding Git metadata."""
    return _digest(_inventory(repository))


def snapshot_repository(source: Path, destination: Path) -> str:
    """Copy an entire working tree, including uncommitted files, into a new directory."""
    source = _directory(source)
    destination = Path(os.path.abspath(destination))
    _directory(destination.parent)
    if destination.exists() or is_link_or_reparse(destination):
        raise ValueError(f"snapshot destination already exists: {destination}")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("snapshot source and destination must be separate directories")
    initial = snapshot_digest(source)
    with (
        open_directory(source) as source_directory,
        open_directory(destination.parent) as parent,
        closing(
            parent.child(destination.name, create=True, exclusive=True)
        ) as destination_directory,
    ):
        _copy_tree(source_directory, destination_directory)
    copied = snapshot_digest(destination)
    if copied != initial or snapshot_digest(source) != initial:
        raise ValueError("repository changed while creating its snapshot")
    return copied


def _remove_partial_candidate(temporary):
    """Remove only the owned staging tree, including Windows readonly files."""

    def readonly_cleanup(function, filename, error):
        if os.name != "nt" or function not in {os.unlink, os.rmdir} or error.errno != errno.EACCES:
            raise error
        relative = Path(filename).relative_to(temporary.parent)
        if not relative.parts or relative.parts[0] != temporary.name:
            raise UnsafePathError("cleanup path escapes the partial candidate")
        # All parents remain pinned while the final object is opened without
        # following reparses, made writable, and deleted through that handle.
        with open_directory(temporary.parent, relative.parts[:-1]) as parent:
            parent.remove_readonly(relative.name)

    if is_link_or_reparse(temporary):
        raise UnsafePathError("partial candidate was replaced by a link or reparse point")
    if temporary.is_dir():
        shutil.rmtree(temporary, onexc=readonly_cleanup)


def freeze_candidate(repository: Path, output_root: Path) -> dict[str, Any]:
    """Retain a candidate independent of future changes to the live repository.

    Call only after peers and command containers have stopped. The manifest is
    published last in run.json, so its presence signals a completed snapshot. It authenticates
    retained bytes against accidental changes, not against a hostile host user.
    """
    repository = _directory(repository)
    output_root = _directory(output_root)
    candidate = output_root / "candidate"
    run = read_document(output_root / RUN_FILE, "run")
    if run.get("candidate") or candidate.exists() or is_link_or_reparse(candidate):
        raise ValueError("candidate artifacts already exist")
    temporary = output_root / f".candidate-{uuid.uuid4().hex}"
    failure = None
    try:
        candidate_digest = snapshot_repository(repository, temporary)
        temporary.rename(candidate)
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            _remove_partial_candidate(temporary)
        except Exception as cleanup_error:
            if failure is None:
                raise
            failure.add_note(
                f"Partial candidate cleanup could not be confirmed: {temporary} "
                f"({type(cleanup_error).__name__})"
            )
    manifest = {
        "source_repository": str(repository),
        "path": "candidate",
        "sha256": candidate_digest,
        "input_sha256": _input_digest(run["input"]),
        "evidence_sha256": {field: _input_digest(run[field]) for field in EVIDENCE_FIELDS},
    }
    update_run(output_root, candidate=manifest)
    return {**manifest, "candidate_path": str(candidate), "candidate_sha256": candidate_digest}


def load_verified_artifacts(output_root: Path) -> dict[str, Any]:
    """Load retained evaluation inputs after checking all declared fingerprints."""
    output_root = _directory(output_root)
    run = read_document(output_root / RUN_FILE, "run")
    manifest = run.get("candidate")
    if not isinstance(manifest, dict) or manifest.get("path") != "candidate":
        raise ValueError("invalid candidate artifact manifest")
    candidate = output_root / "candidate"
    if _digest(_inventory(candidate, exclude_git=False)) != manifest.get("sha256"):
        raise ValueError("candidate artifact fingerprint mismatch")
    if _input_digest(run["input"]) != manifest.get("input_sha256"):
        raise ValueError("input evidence fingerprint mismatch")
    if {field: _input_digest(run[field]) for field in EVIDENCE_FIELDS} != manifest.get(
        "evidence_sha256"
    ):
        raise ValueError("run evidence fingerprint mismatch")
    return {**manifest, "candidate_path": str(candidate), "candidate_sha256": manifest["sha256"]}
