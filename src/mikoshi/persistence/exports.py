"""Read models and filesystem exports, separate from world-state transactions."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from mikoshi.persistence.store import WorldStore


MAX_STATUS_BYTES = 65_536
MAX_CHANGED_PATHS = 256


def write_json(path: Path, value: Any, *, sort_keys: bool = True) -> None:
    """Replace a JSON file atomically, creating its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=sort_keys, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def export_state(store: WorldStore) -> Mapping[str, Any]:
    """One decoded operator view; readers never need to join private SQLite tables."""
    from mikoshi.tools.shared_pages import records as shared_page_records

    shared = shared_page_records(store)
    pages = []
    for page in shared["shared_pages"]:
        pages.append(
            {
                **page,
                "revisions": [r for r in shared["page_revisions"] if r["page_id"] == page["id"]],
                "comments": [c for c in shared["page_comments"] if c["page_id"] == page["id"]],
                "snapshots": [
                    {**s, "sections": json.loads(s["sections"])}
                    for s in shared["page_snapshots"]
                    if s["page_id"] == page["id"]
                ],
            }
        )
    commands = []
    for command in store.list_command_requests():
        commands.append(
            {
                **command,
                "log_path": f"workspace/command-logs/command-{command['id']}.log"
                if command["log_path"]
                else None,
            }
        )
    return {
        "agents": [asdict(agent) for agent in store.list_agents()],
        "pages": pages,
        "projects": list(store.list_projects()),
        "commands": commands,
        "activations": list(store.list_activations()),
        "triggers": list(store.list_triggers()),
        "inbox": store.list_inbox_messages(),
        "recording_errors": list(store.event_observer_errors),
    }


def queued_notifications(store: WorldStore, event_id: int):
    """Messages inserted in the same committed transaction as their source action."""
    return store.list_inbox_messages(event_id=event_id)


def checkout_metadata(repository):
    """Read bounded launch context without snapshotting uncommitted contents."""
    from mikoshi.tools.workspace import _git_invocation

    result = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "status": "unavailable",
        "branch": None,
        "commit": None,
        "upstream": None,
        "changes": [],
        "truncated": False,
    }
    try:
        # Bound memory and execution time even for large or slow checkouts. Git's
        # optional index refresh is disabled; no fetch, merge or hooks are invoked.
        with tempfile.TemporaryFile() as output:
            command, environment = _git_invocation(
                (
                    "--no-optional-locks",
                    "-C",
                    str(repository),
                    "status",
                    "--porcelain=v2",
                    "--branch",
                    "-z",
                    "--untracked-files=normal",
                ),
                environment={"GIT_OPTIONAL_LOCKS": "0"},
            )
            completed = subprocess.run(
                command,
                stdout=output,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                timeout=3,
                check=False,
                env=environment,
            )
            if completed.returncode:
                return result
            output.seek(0)
            data = output.read(MAX_STATUS_BYTES + 1)
    except (OSError, subprocess.TimeoutExpired):
        return result
    result["status"] = "available"
    result["truncated"] = len(data) > MAX_STATUS_BYTES
    # Discard a possibly incomplete final record, including partial UTF-8.
    records = data[:MAX_STATUS_BYTES].split(b"\0")[:-1]
    index = 0
    while index < len(records):
        record = records[index].decode("utf-8", errors="replace")
        index += 1
        if record.startswith("# branch."):
            key, value = record.split(" ", 2)[1:]
            field = {
                "branch.head": "branch",
                "branch.oid": "commit",
                "branch.upstream": "upstream",
            }.get(key)
            if field:
                result[field] = None if value in {"(initial)", "(detached)"} else value
            continue
        kind = record[:1]
        if kind not in {"1", "2", "u", "?"}:
            continue
        if len(result["changes"]) >= MAX_CHANGED_PATHS:
            result["truncated"] = True
            break
        parts = record.split(" ", {"1": 8, "2": 9, "u": 10, "?": 1}[kind])
        change = {"status": "??" if kind == "?" else parts[1], "path": parts[-1]}
        if kind == "2":
            if index >= len(records):
                result["truncated"] = True
                break
            change["original_path"] = records[index].decode("utf-8", errors="replace")
            index += 1
        result["changes"].append(change)
    return result
