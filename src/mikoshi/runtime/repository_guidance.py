"""Bounded startup guidance, skill metadata, and pre-action repository context."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path

from mikoshi.runtime.skill_catalog import build_skill_catalog
from mikoshi.tools.commands import CONTAINER_WORKSPACE
from mikoshi.tools.workspace import WorkspaceAccessError, WorkspaceChangedError

MAX_DOCUMENT_CHARS = 32_768
MAX_SCOPE_BYTES = 65_536


def _next_line(loaded):
    """read_file offset (a 1-based line) at which a cut-off document continues."""
    return loaded.count("\n") + 1


def _read(workspace, actor, path, warnings, limit=MAX_DOCUMENT_CHARS):
    try:
        result = workspace.read_text(actor, path, min(limit, 12_000))
        while result["next_offset"] is not None and len(result["content"]) < limit:
            chunk = workspace.read_text(
                actor,
                path,
                min(limit - len(result["content"]), 12_000),
                offset=result["next_offset"],
            )
            if chunk["sha256"] != result["sha256"]:
                raise WorkspaceChangedError("guidance changed while loading")
            result["content"] += chunk["content"]
            result["next_offset"] = chunk["next_offset"]
        if result["next_offset"] is not None:
            result["next_offset"] = _next_line(result["content"])
        return {key: result[key] for key in ("path", "content", "sha256", "next_offset")}
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, WorkspaceAccessError, WorkspaceChangedError) as error:
        warnings.append({"path": path, "error": str(error)})
        return None


def _instructions(workspace, actor, directory, warnings):
    for filename in ("AGENTS.override.md", "AGENTS.md"):
        path = f"{directory}/{filename}" if directory else filename
        document = _read(workspace, actor, path, warnings, limit=MAX_SCOPE_BYTES)
        if document and document["content"].strip():
            return {**document, "scope": directory or "."}
    return None


def root_guidance(workspace, actor):
    warnings = []
    document = _instructions(workspace, actor, "", warnings)
    if document:
        # Codex's native project instruction budget is measured in UTF-8 bytes.
        content = document["content"]
        document["content"] = content.encode()[:MAX_DOCUMENT_CHARS].decode("utf-8", errors="ignore")
        if document["content"] != content:
            document["next_offset"] = _next_line(document["content"])
    return {"documents": [document] if document else [], "warnings": warnings}


def load_repository_guidance(workspace, actor, context_window_tokens=None, count_tokens=None):
    root = root_guidance(workspace, actor)
    catalog = build_skill_catalog(workspace, actor, context_window_tokens, count_tokens)
    if (
        not root["documents"]
        and not root["warnings"]
        and not catalog["skills"]
        and not catalog["warnings"]
    ):
        return None
    return {**root, "catalog": catalog}


def warning_documents(warnings):
    return [
        {"path": "warning:" + item.get("path", ""), "content": json.dumps(item, sort_keys=True)}
        for item in warnings
    ]


class GuidanceLedger:
    """Private per-conversation fingerprints, durable across peer activations."""

    def __init__(self, store):
        self._records = store.repository_guidance

    def _read(self, actor):
        return self._records.load(actor)

    def _write(self, actor, value):
        self._records.save(actor, value)

    def pending(self, actor, documents):
        seen = self._read(actor)["seen"]
        selected, size = [], 0
        for document in documents:
            if seen.get(document["path"]) == _fingerprint(document):
                continue
            length = len(document["content"].encode())
            if selected and size + length > MAX_SCOPE_BYTES:
                continue  # Still eligible on the next matching action.
            selected.append(document)
            size += length
        return selected

    def acknowledge(self, actor, documents, checkpoints=()):
        value = self._read(actor)
        if checkpoints:
            checkpoint = hashlib.sha256(
                json.dumps(checkpoints, sort_keys=True).encode()
            ).hexdigest()
            if checkpoint != value["checkpoint"]:
                value = {"seen": {}, "checkpoint": checkpoint}
            # Only native startup context is restored after compaction.
        else:
            for document in documents:
                value["seen"][document["path"]] = _fingerprint(document)
        self._write(actor, value)


def _fingerprint(document):
    return hashlib.sha256(
        json.dumps(
            {key: document.get(key) for key in ("content", "scope", "next_offset")},
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _is_directory(path):
    try:
        return path.is_dir()
    except (OSError, ValueError):
        return False


def activity_guidance(workspace, actor, decision):
    """Select instructions for an action, without performing that action."""
    paths = _activity(workspace.canonical_repository, decision)
    warnings, documents = [], {}
    visited = set()
    for path in sorted(paths):
        directory = (
            path if _is_directory(workspace.canonical_repository / path) else str(Path(path).parent)
        )
        parts = Path(directory).parts
        for length in range(1, len(parts) + 1):
            directory = Path(*parts[:length]).as_posix()
            if directory == "." or directory in visited:
                continue
            visited.add(directory)
            # Command arguments such as sed expressions can look like paths; only real
            # repository directories can hold scoped instructions.
            if not _is_directory(workspace.canonical_repository / directory):
                break
            if document := _instructions(workspace, actor, directory, warnings):
                documents[document["path"]] = document
    if paths:
        for document in _invariants(workspace, actor, paths, warnings):
            documents[document["path"]] = document
    complete = []
    for document in documents.values():
        if (
            document["next_offset"] is not None
            or len(document["content"].encode()) > MAX_SCOPE_BYTES
        ):
            warnings.append(
                {
                    "path": document["path"],
                    "error": "Scoped instructions exceed 64 KiB and were not loaded; the file "
                    "can be read with read_file.",
                }
            )
        else:
            complete.append(document)
    return {"documents": complete, "warnings": warnings}


def _invariants(workspace, actor, paths, warnings):
    matched = []
    for file in sorted((workspace.canonical_repository / ".codex/invariants").glob("INV-*.json")):
        document = _read(
            workspace,
            actor,
            file.relative_to(workspace.canonical_repository).as_posix(),
            warnings,
            limit=MAX_SCOPE_BYTES,
        )
        if not document:
            continue
        try:
            rule = json.loads(document["content"])
            if not isinstance(rule, dict) or not isinstance(rule.get("body"), str):
                raise ValueError("invalid invariant body")
            for key in ("patterns", "supersedes", "globs"):
                if not isinstance(rule.get(key), list) or not all(
                    isinstance(v, str) for v in rule[key]
                ):
                    raise ValueError(f"invalid invariant {key}")
            if not isinstance(rule.get("id"), str):
                raise ValueError("invalid invariant id")
            if any(re.search(pattern, path) for pattern in rule["patterns"] for path in paths):
                matched.append((document, rule))
        except (ValueError, re.error) as error:
            warnings.append({"path": document["path"], "error": str(error)})
    superseded = {identifier for _, rule in matched for identifier in rule["supersedes"]}
    return [
        {**document, "content": rule["body"], "scope": rule["globs"], "format": "invariant_body"}
        for document, rule in matched
        if rule["id"] not in superseded
    ]


def _relative(root, cwd, value):
    if not value or value.startswith("-") or any(c in value for c in "\n\x00$`*"):
        return None
    if value == CONTAINER_WORKSPACE or value.startswith(CONTAINER_WORKSPACE + "/"):
        # Commands see the checkout at /workspace; map it back to the host repository.
        value = str(root) + value[len(CONTAINER_WORKSPACE) :]
    try:
        path = Path(value) if Path(value).is_absolute() else cwd / value
        relative = path.resolve().relative_to(root).as_posix()
        return relative
    except (ValueError, OSError, RuntimeError):
        return None


def _activity(root, decision):
    paths = set()
    if decision["action"] in {"list_files", "read_file", "write_file", "edit_file"}:
        path = decision.get("path", "")
        if path := _relative(root, root, path):
            paths.add(path)
    elif decision["action"] == "run_command":
        argv = list(decision["argv"])
        shell = Path(argv[0]).name in {"sh", "bash", "zsh"}
        # Patch headers apply to direct arguments and shell heredocs alike.
        patch = argv[1] if Path(argv[0]).name == "apply_patch" and len(argv) == 2 else ""
        if shell and len(argv) >= 3 and "c" in argv[1].lstrip("-"):
            patch = script = argv[2]
            try:
                lexer = shlex.shlex(script, posix=True, punctuation_chars=";&|")
                lexer.whitespace_split = True
                tokens = list(lexer)
            except ValueError:
                tokens = []
        else:
            tokens = argv
        for value in re.findall(
            r"^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+)$", patch, re.M
        ):
            if path := _relative(root, root, value):
                paths.add(path)
        cwd, words = root, []
        for token in [*tokens, ";"]:
            if token not in {";", "&&", "||", "|", "&"}:
                words.append(token)
                continue
            if words[:1] == ["cd"] and len(words) == 2 and token == "&&":
                candidate = _relative(root, cwd, words[1])
                if candidate is None:
                    break  # Outside the checkout or computed by the shell.
                cwd = root / candidate
            else:
                _command_activity(root, cwd, words, paths)
            words = []
    return paths


def _command_activity(root, cwd, words, paths):
    while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
        words = words[1:]
    if not words:
        return
    command, args = Path(words[0]).name, words[1:]
    if command == "env":
        while args and ("=" in args[0] or args[0].startswith("-")):
            args = args[1:]
        return _command_activity(root, cwd, args, paths)
    if command == "git":
        while args:
            if args[0] in {"-C", "-c"} and len(args) > 1:
                if args[0] == "-C":
                    candidate = _relative(root, cwd, args[1])
                    if candidate is None:
                        return
                    cwd = root / candidate
                args = args[2:]
            elif args[0] in {"--no-pager", "--no-optional-locks"} or args[0].startswith("-c"):
                args = args[1:]
            else:
                break
    if command not in {
        "cat",
        "sed",
        "head",
        "tail",
        "rg",
        "grep",
        "ls",
        "git",
        "pytest",
        "jest",
        "vitest",
        "npm",
        "pnpm",
        "yarn",
        "make",
        "python",
        "python3",
        "python3.12",
        "npx",
        "uv",
    }:
        return
    if cwd != root:
        paths.add(cwd.relative_to(root).as_posix())
    for value in args:
        if value.startswith("--") and "=" in value:
            value = value.split("=", 1)[1]
        if path := _relative(root, cwd, value):
            if "/" in value or Path(value).suffix or (root / path).exists():
                paths.add(path)
