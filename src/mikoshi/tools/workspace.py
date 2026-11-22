"""Shared live repository, guarded edits, and immutable observation evidence."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import stat
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from mikoshi.filesystem import (
    FileChangedError,
    UnsafePathError,
    is_link_or_reparse,
    is_reparse,
    open_directory,
    validate_component,
)
from mikoshi.persistence.exports import write_json
from mikoshi.tools.text_tools import (
    MAX_BYTES,
    apply_edits,
    format_size,
    split_lines,
    truncate_head,
)

OBSERVATION_LIMIT = 256
CONTROL_NAMES = {".git", ".control", ".cas"}


class WorkspaceAccessError(ValueError):
    """Raised when a requested path is outside the activation's authorized view."""


class WorkspaceChangedError(ValueError):
    """A concurrent command invalidated a file observation; the caller can retry."""


@dataclass(frozen=True)
class WorkspaceLimits:
    max_file_bytes: int = 32_000_000
    max_read_chars: int = 12_000

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True)
class RepositoryIdentity:
    source: str
    commit: str | None


def resolve_repository(source: Path) -> RepositoryIdentity:
    """Describe an existing working directory; HEAD is optional provenance, not a snapshot.

    No status, index refresh, filters, hooks or repository inventory is needed.
    Ordinary Git configuration and worktree administrative files are supported.
    Uncommitted and untracked files remain in the working directory.
    """
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(f"repository is not a directory: {source}")
    command, environment = _git_invocation(
        (
            "-c",
            f"safe.directory={source}",
            "-C",
            str(source),
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
        ),
        environment={"GIT_OPTIONAL_LOCKS": "0"},
    )
    commit = None
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
            timeout=5,
        )
        value = result.stdout.decode("ascii", errors="replace").strip()
        if (
            result.returncode == 0
            and len(value) in {40, 64}
            and all(character in "0123456789abcdef" for character in value)
        ):
            commit = value
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Git is optional: plain directories and environments without Git still work.
        pass
    return RepositoryIdentity(str(source), commit)


class LiveWorkspace:
    """One existing project directory shared by all peers."""

    def __init__(self, root, store, source_repository, limits=None):
        self.root = Path(root).resolve()
        self.store = store
        self.limits = limits or WorkspaceLimits()
        self.canonical_repository = Path(source_repository).resolve(strict=True)
        if (
            self.root == self.canonical_repository
            or self.root in self.canonical_repository.parents
            or self.canonical_repository in self.root.parents
        ):
            raise WorkspaceAccessError("repository and runtime controls must not overlap")
        self.control_root = self.root / ".control"
        self.identity_path = self.root / "repository.json"
        self.identity = None
        self._observations = {}
        self._lock = threading.RLock()
        self._archive_cache = {}

    def initialize(self, repository_identity=None):
        self._assert_available()
        if not self.canonical_repository.is_dir() or is_link_or_reparse(self.canonical_repository):
            raise WorkspaceAccessError("repository must be an existing real directory")
        if self.identity_path.is_file():
            identity = RepositoryIdentity(**json.loads(self.identity_path.read_text()))
        else:
            identity = repository_identity or resolve_repository(self.canonical_repository)
        if Path(identity.source) != self.canonical_repository or (
            repository_identity is not None and repository_identity != identity
        ):
            raise ValueError("workspace identity does not match source")
        self.root.mkdir(parents=True, exist_ok=True)
        self.control_root.mkdir(exist_ok=True)
        (self.root / ".cas").mkdir(exist_ok=True)
        self.identity = identity
        if not self.identity_path.exists():
            write_json(self.identity_path, asdict(identity))

    def _assert_available(self):
        if (self.control_root / "quarantined.json").exists():
            raise WorkspaceAccessError(
                "shared workspace quarantined after command isolation failure"
            )

    def quarantine_repository(self, reason):
        self.control_root.mkdir(exist_ok=True)
        write_json(
            self.control_root / "quarantined.json",
            {"reason": str(reason)},
        )

    def _require_initialized(self):
        self._assert_available()
        if is_link_or_reparse(self.canonical_repository):
            raise WorkspaceAccessError("repository was replaced by a symbolic link")
        if self.identity is None or not self.canonical_repository.is_dir():
            raise RuntimeError("live workspace is not initialized")

    def _check_actor(self, agent_id):
        self.store.get_agent(agent_id)

    @staticmethod
    def _clean_relative(value):
        if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
            raise WorkspaceAccessError("workspace path must be a normalized relative string")
        path = PurePosixPath(value)
        if path.is_absolute() or any(
            part in {"", ".", ".."} or part.casefold() in CONTROL_NAMES for part in value.split("/")
        ):
            raise WorkspaceAccessError(
                "workspace path must be normalized, relative, and outside control files"
            )
        try:
            for part in value.split("/"):
                validate_component(part)
        except UnsafePathError as error:
            raise WorkspaceAccessError(str(error)) from error
        return path

    def _reject_symlink_components(self, path):
        try:
            parts = path.relative_to(self.root).parts
        except ValueError as error:
            raise WorkspaceAccessError("path escapes workspace") from error
        current = self.root
        for part in parts:
            current = current / part
            if is_link_or_reparse(current):
                raise WorkspaceAccessError("symbolic links are not allowed in workspace paths")
            if current.exists() and current.is_file() and current.stat().st_nlink != 1:
                raise WorkspaceAccessError("hard links are not allowed in workspace paths")

    @contextmanager
    def _directory(self, parts=(), *, create=False):
        """Pin each directory by its native handle; never follow command-created links."""
        self._require_initialized()
        try:
            with open_directory(self.canonical_repository, parts, create=create) as directory:
                yield directory
        except UnsafePathError as error:
            raise WorkspaceAccessError(str(error)) from error
        except FileChangedError as error:
            raise WorkspaceChangedError(str(error)) from error

    def _read_at(self, directory, name):
        descriptor = directory.open_file(name)
        try:
            before = os.fstat(descriptor)
            if before.st_nlink == 0:
                raise WorkspaceChangedError("file was removed while reading; retry")
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise WorkspaceAccessError(
                    "workspace files must be regular files without hard links"
                )
            if before.st_size > self.limits.max_file_bytes:
                raise WorkspaceAccessError("workspace file exceeds byte limit")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                data = stream.read(min(before.st_size + 1, self.limits.max_file_bytes + 1))
            after = os.fstat(descriptor)
            if len(data) > self.limits.max_file_bytes:
                raise WorkspaceAccessError("workspace file exceeds byte limit")
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise WorkspaceChangedError("file changed while reading; retry the operation")
            return data, stat.S_IMODE(after.st_mode)
        finally:
            os.close(descriptor)

    @staticmethod
    def _file_identity(info):
        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_mode,
            info.st_nlink,
        )

    @staticmethod
    def _manifest_entry(path, data):
        return {
            "path": path,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "media_type": mimetypes.guess_type(path)[0] or "application/octet-stream",
        }

    def file_state(self):
        with self._lock:
            self._require_initialized()
            return {
                "tree_sha256": None,
                "file_count": None,
                "total_bytes": None,
                "files": [dict(entry) for entry in self._observations.values()],
                "observation_scope": "recent_file_tool_operations_only",
                "observation_limit": OBSERVATION_LIMIT,
                "whole_tree_observed": False,
                "aggregate_limits_enforced": False,
            }

    def _archived_version(self, path, digest, size):
        blob = self.root / ".cas" / digest[:2] / digest
        self._reject_symlink_components(blob)
        if not blob.exists():
            return None
        info = blob.stat()
        identity = self._file_identity(info)
        if self._archive_cache.get(digest) != identity:
            if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                raise WorkspaceAccessError("content archive digest verification failed")
            with blob.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != digest or self._file_identity(blob.stat()) != identity:
                raise WorkspaceAccessError("content archive digest verification failed")
            self._archive_cache[digest] = identity
        return {
            "path": path,
            "sha256": digest,
            "bytes": size,
            "blob_path": blob.relative_to(self.root).as_posix(),
        }

    def _remember_observation(self, path, data):
        self._observations.pop(path, None)
        self._observations[path] = self._manifest_entry(path, data)
        while len(self._observations) > OBSERVATION_LIMIT:
            del self._observations[next(iter(self._observations))]

    def _capture_bytes(self, path, data):
        digest = hashlib.sha256(data).hexdigest()
        version = self._archived_version(path, digest, len(data))
        if version is None:
            blob = self.root / ".cas" / digest[:2] / digest
            blob.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write(blob, data, 0o444)
            version = self._archived_version(path, digest, len(data))
        return version

    @staticmethod
    def _atomic_write(target, data, mode):
        descriptor, name = tempfile.mkstemp(prefix=".write-", dir=target.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(mode)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def _authorized_text(self, agent_id, path):
        self._check_actor(agent_id)
        relative = self._clean_relative(path)
        try:
            with self._directory(relative.parts[:-1]) as directory:
                data, mode = self._read_at(directory, relative.name)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"no file at {path}") from error
        try:
            return mode, data, data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise WorkspaceAccessError("file is not UTF-8 text") from error

    def read_text(self, agent_id, path, max_chars=None, *, offset=0):
        with self._lock:
            _, data, content = self._authorized_text(agent_id, path)
            if offset < 0 or offset > len(content):
                raise ValueError("offset exceeds file length")
            limit = min(max_chars or self.limits.max_read_chars, self.limits.max_read_chars)
            version = self._capture_bytes(path, data)
            self._remember_observation(path, data)
            return {
                "path": path,
                "content": content[offset : offset + limit],
                "offset": offset,
                "next_offset": offset + limit if offset + limit < len(content) else None,
                "truncated": offset + limit < len(content),
                "characters": len(content),
                "sha256": hashlib.sha256(data).hexdigest(),
                "resource_version": version,
            }

    def list_files(self, agent_id, path=".", offset=0):
        """Page through the immediate, file-tool-accessible children of one directory."""
        self._check_actor(agent_id)
        parts = () if path == "." else self._clean_relative(path).parts
        entries = []
        with self._directory(parts) as directory:
            for child in directory.entries():
                if child.name.casefold() in CONTROL_NAMES or "\\" in child.name:
                    continue
                try:
                    child.name.encode("utf-8")
                    validate_component(child.name)
                    info = child.info
                except (FileNotFoundError, UnicodeEncodeError, UnsafePathError):
                    continue
                if is_reparse(info):
                    continue
                if stat.S_ISDIR(info.st_mode):
                    kind = "directory"
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    kind = "file"
                else:
                    continue
                entries.append({"name": child.name, "type": kind})
        entries.sort(key=lambda entry: entry["name"])
        end = offset + 100
        return {
            "path": path,
            "entries": entries[offset:end],
            "next_offset": end if end < len(entries) else None,
        }

    def read_lines(self, agent_id, path, offset=None, limit=None, column=0):
        """Read numbered lines from a 1-based offset, with a bounded preview.

        The content ends with a notice saying how to continue when lines remain.
        """
        with self._lock:
            _, data, text = self._authorized_text(agent_id, path)
            version = self._capture_bytes(path, data)
            self._remember_observation(path, data)
        lines = split_lines(text)
        start = max(0, offset - 1) if offset else 0
        if (lines and start >= len(lines)) or (not lines and start > 0):
            raise ValueError(f"Offset {offset} is beyond end of file ({len(lines)} lines total)")
        stop = min(start + limit, len(lines)) if limit is not None else len(lines)
        if column and (not lines or column > len(lines[start])):
            raise ValueError("column exceeds the selected line length")
        content, kept, truncated_by = (
            truncate_head("\n".join(lines[start:stop])) if not column else ("", 0, "first_line")
        )
        first, total = start + 1, len(lines)
        if kept:
            # cat -n style, as in Claude Code's Read: right-aligned number, a tab, the line.
            content = "\n".join(
                f"{first + index:>6}\t{line}" for index, line in enumerate(content.split("\n"))
            )
        next_offset = None
        continuation = {}
        if truncated_by == "first_line":
            line = lines[start]
            chunk = line[column : column + MAX_BYTES].encode()[:MAX_BYTES].decode(errors="ignore")
            content = f"{first:>6}\t{chunk}"
            if column + len(chunk) < len(line):
                next_offset = first
                continuation["next_column"] = column + len(chunk)
            elif first < total:
                next_offset = first + 1
        elif truncated_by is not None:
            last = first + kept - 1
            limit_note = f" ({format_size(MAX_BYTES)} limit)" if truncated_by == "bytes" else ""
            next_offset = last + 1
            content += (
                f"\n\n[Showing lines {first}-{last} of {total}{limit_note}. "
                f"Continue with offset={next_offset}.]"
            )
        elif stop < total:
            next_offset = stop + 1
            content += (
                f"\n\n[{total - stop} more lines in the file. Continue with offset={next_offset}.]"
            )
        return {
            "path": path,
            "next_offset": next_offset,
            **continuation,
            "content": content,
            "sha256": hashlib.sha256(data).hexdigest(),
            "resource_version": version,
        }

    def write_text(self, agent_id, path, content, *, base=None):
        """Create or replace a file. With base, fail if the file no longer has those bytes."""
        with self._lock:
            self._check_actor(agent_id)
            relative = self._clean_relative(path)
            if not isinstance(content, str):
                raise TypeError("workspace content must be text")
            data = content.encode()
            with self._directory(relative.parts[:-1], create=True) as directory:
                try:
                    previous, mode = self._read_at(directory, relative.name)
                except FileNotFoundError:
                    previous, mode = None, 0o644
                if base is not None and previous != base:
                    raise WorkspaceChangedError("file changed while editing; retry the edit")
                if len(data) > self.limits.max_file_bytes:
                    raise WorkspaceAccessError("write exceeds workspace limits")
                previous_version = (
                    self._capture_bytes(path, previous) if previous is not None else None
                )
                version = self._capture_bytes(path, data)

                def unchanged():
                    try:
                        current, _ = self._read_at(directory, relative.name)
                    except FileNotFoundError:
                        current = None
                    if current != previous:
                        raise WorkspaceChangedError(
                            "file changed before writing; reread before editing"
                        )

                # Stage and publish through the pinned directory on the target mount.
                directory.write_file(
                    relative.name,
                    data,
                    mode,
                    replace=previous is not None,
                    before_publish=unchanged,
                )
            self._remember_observation(path, data)
            return {
                "path": path,
                "bytes": len(data),
                "sha256": version["sha256"],
                "resource_version": version,
                "previous_resource_version": previous_version,
                "material_change": previous != data,
            }

    def edit_text(self, agent_id, path, edits):
        """Pi's edit: replace unique, non-overlapping exact-text regions of the file."""
        with self._lock:
            _, data, content = self._authorized_text(agent_id, path)
            result = self.write_text(agent_id, path, apply_edits(content, edits, path), base=data)
            return {**result, "replaced_blocks": len(edits)}

    def command_working_directory(self, agent_id):
        self._check_actor(agent_id)
        self._require_initialized()
        return self.canonical_repository


def _git_invocation(
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    # Only read-only rev-parse/status calls use this helper. Disable executable
    # hooks and filesystem monitors without configuring unused transport/write paths.
    command = [
        "git",
        "-c",
        f"core.hooksPath={os.devnull}",
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"core.attributesFile={os.devnull}",
    ]
    command.extend(str(item) for item in arguments)
    process_environment = {
        **{
            name: value
            for name, value in os.environ.items()
            if os.name == "nt"
            and name.casefold()
            in {"systemroot", "windir", "systemdrive", "comspec", "pathext", "temp", "tmp"}
        },
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        **dict(environment or {}),
    }
    return command, process_environment
