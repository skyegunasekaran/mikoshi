"""Recent project activity, described with the tool names peers use."""

import json
from pathlib import PurePosixPath

from mikoshi.contracts import action_schema
from mikoshi.persistence.activity import CATEGORIES

PREVIEW_CHARS = 160
ACTIVITY_DESCRIPTION = (
    "List recent actions by peers in this project, newest first, together with the "
    "commands that are queued or running. Actions show the tool that was used; outcome "
    "appears only when an action failed, was denied or was deferred. File changes made by "
    "commands are not recorded."
)
TOOL_NAMES = {event: tool for names in CATEGORIES.values() for event, tool in names.items()}
ACTIVITY_SCHEMA = action_schema(
    "list_activity",
    {
        "peer_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 100,
            "description": "Only this peer's actions and commands.",
        },
        "category": {
            "type": "string",
            "enum": ["files", "commands", "pages", "activations", "messages"],
            "description": "Only actions of this kind.",
        },
        "path": {
            "type": "string",
            "minLength": 1,
            "maxLength": 400,
            "description": "Only file actions on this file or directory. Implies the files "
            "category.",
        },
        "since_tick": {
            "type": "integer",
            "minimum": 0,
            "description": "Only actions after this tick.",
        },
        "changes_only": {
            "type": "boolean",
            "description": "Only file writes and edits, page edits and comments, and finished "
            "commands; leaves out reads, listings and actions that didn't run.",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
            "description": "Actions per call; default 10.",
        },
        "offset": {
            "type": "integer",
            "minimum": 0,
            "description": "Only actions before this tick. Pass next_offset from the previous "
            "result to see older actions.",
        },
    },
    required=[],
)


def _path(value):
    if (
        PurePosixPath(value).is_absolute()
        or ".." in PurePosixPath(value).parts
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError("path must be relative to the repository root, without '..'")
    path = str(PurePosixPath(value))
    return "" if path == "." else path


def _command_preview(row):
    argv = json.loads(row["argv_json"])
    preview = (
        argv[2]
        if len(argv) == 3 and argv[:2] == ["sh", "-c"]
        else json.dumps(argv, ensure_ascii=False)
    )
    result = {"command": preview[:PREVIEW_CHARS]}
    if len(preview) > PREVIEW_CHARS:
        result["command_truncated"] = True
    return result


def _event(store, row, commands_seen, reader_id):
    """A short public summary; details stay on referenced pages and retained output."""
    event = row["action_type"]
    result = {
        "tick": row["tick"],
        "peer_id": row["agent_id"],
        "action": TOOL_NAMES.get(event, event),
    }
    if row["outcome"] not in {"succeeded", "authorized"}:
        result["outcome"] = row["outcome"]
    payload = json.loads(row["payload"])
    resource = row["resource_id"] or ""
    if row["category"] == "files":
        result["path"] = resource
    elif row["category"] == "commands" and resource.startswith("command:"):
        command = store.activity_queries.command(int(resource.split(":", 1)[1]))
        if command:
            result["command_id"] = command["id"]
            if command["id"] not in commands_seen:
                result.update(_command_preview(command))
                commands_seen.add(command["id"])
            if event == "execute_command":
                if command["exit_code"] is not None:
                    result["exit_code"] = command["exit_code"]
                result["output_ref"] = f"command:{command['id']}"
    elif row["category"] in {"pages", "messages"}:
        if payload.get("page_id"):
            result["path"] = store.page_path(payload["page_id"])
        elif resource:
            # Failed page actions record the path the peer asked for.
            result["path"] = resource
        if "comment_id" in payload:
            result["comment_id"] = payload["comment_id"]
    # Error text can name private audiences, so only the peer who acted sees it.
    if isinstance(payload.get("error"), str) and row["agent_id"] == reader_id:
        result["error"] = payload["error"][:400]
    return result


def list_activity(store, project_id, decision, *, command_statuses, reader_id):
    """Read recent project activity and active commands, newest first."""
    queries = store.activity_queries
    peer = decision.get("peer_id")
    if peer is not None and not store.is_project_member(peer, project_id):
        members = ", ".join(queries.project_peer_ids(project_id))
        raise ValueError(f"peer_id {peer!r} is not a peer in this project; peers are {members}")
    path = _path(decision["path"]) if "path" in decision else None
    category = decision.get("category")
    if path is not None:
        if category not in (None, "files"):
            raise ValueError("path only applies to file actions; drop category or use files")
        category = "files"
    limit = decision.get("limit", 10)
    before = decision.get("offset") or None

    command_rows = queries.approved_commands(project_id, peer)
    active_commands = [
        {
            "command_id": row["id"],
            "peer_id": row["requested_by"],
            **_command_preview(row),
            "status": "queued"
            if command_statuses.get(row["id"], "queued") == "queued"
            else "running",
        }
        for row in command_rows
    ]

    rows = queries.recent_events(
        project_id,
        peer_id=peer,
        reader_id=reader_id,
        since_tick=decision.get("since_tick", 0),
        before=before,
        category=category,
        path=path or None,
        limit=limit + 1,
        changes_only=decision.get("changes_only", False),
    )
    commands_seen = {command["command_id"] for command in active_commands}
    actions = [_event(store, row, commands_seen, reader_id) for row in rows[:limit]]
    return {
        "current_tick": store.current_tick,
        "active_commands": active_commands,
        "actions": actions,
        "next_offset": rows[limit - 1]["tick"] if len(rows) > limit else None,
    }
