"""One paged reader for retained command output and a peer's archived tool replies."""

from __future__ import annotations

import os
import re

from mikoshi.contracts import action_schema
from mikoshi.filesystem import open_directory, regular_file
from mikoshi.tools.commands import COMPLETED_STATUSES
from mikoshi.tools.workspace import WorkspaceAccessError

REFERENCE = re.compile(r"(?:command:[1-9][0-9]{0,17}|tool:[1-9][0-9]{0,17}/[0-9]{1,2})")
READ_OUTPUT_SCHEMA = action_schema(
    "read_output",
    {
        "reference": {
            "type": "string",
            "pattern": "^" + REFERENCE.pattern + "$",
            "description": "The output_ref returned by a command or a shortened tool reply.",
        },
        "offset": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 4096},
    },
    required=["reference"],
) | {
    "description": "Read retained output. Command output is shared with project peers; "
    "archived tool replies belong to your conversation. Offsets and limits count characters. "
    "Pass next_offset to continue. Capture limits may have shortened the retained output."
}


def read_output(store, workspace, conversations, actor, decision, *, log_root=None):
    reference = decision["reference"]
    if not isinstance(reference, str) or not REFERENCE.fullmatch(reference):
        raise ValueError("invalid output reference")
    offset, limit = decision.get("offset", 0), decision.get("limit", 2048)
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 4096:
        raise ValueError("offset must be nonnegative and limit must be between 1 and 4096")
    if reference.startswith("tool:"):
        result = conversations.read_tool_result(actor, reference[5:], offset, limit)
        return {
            "reference": reference,
            "offset": offset,
            "content": result["content"],
            "next_offset": result["next_offset"],
        }

    command = store.get_command_request(int(reference[8:]))
    if not store.is_project_member(actor, command["project_id"]):
        raise WorkspaceAccessError("you are not a member of this output's project")
    if command["status"] not in COMPLETED_STATUSES:
        raise ValueError("command output is available after the command finishes")
    if command.get("log_path"):
        root = log_root if log_root is not None else workspace.root / "command-logs"
        with (
            open_directory(root) as directory,
            regular_file(directory, f"command-{command['id']}.log") as descriptor,
        ):
            with os.fdopen(
                descriptor, "r", encoding="utf-8", errors="replace", newline="", closefd=False
            ) as stream:
                remaining = offset
                while remaining:
                    skipped = stream.read(min(remaining, 65536))
                    if not skipped:
                        raise ValueError("offset exceeds the retained output length")
                    remaining -= len(skipped)
                content = stream.read(limit + 1)
    else:
        text = str(command.get("output") or "")
        if offset > len(text):
            raise ValueError("offset exceeds the retained output length")
        content = text[offset : offset + limit + 1]
    return {
        "reference": reference,
        "offset": offset,
        "content": content[:limit],
        "next_offset": offset + limit if len(content) > limit else None,
    }
