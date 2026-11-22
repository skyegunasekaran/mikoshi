"""Shared repository access and activity inspection for project participants."""

from mikoshi.contracts import action_schema
from mikoshi.tools.commands import CONTAINER_WORKSPACE
from mikoshi.tools.text_tools import MAX_BYTES, MAX_LINES
from mikoshi.tools.workspace import WorkspaceAccessError
from mikoshi.tools.workspace_inspection import (
    ACTIVITY_DESCRIPTION,
    ACTIVITY_SCHEMA,
    list_activity,
)


class SharedWorkTools:
    def __init__(self, runtime, *, prepare_action=None):
        self.runtime = runtime
        self.store = runtime.store
        self.workspace = runtime.workspace
        self.prepare_action = prepare_action

    def _repository_relative(self, path):
        """Accept absolute repository paths, as commands or the host see it, as relative."""
        for root in (CONTAINER_WORKSPACE, str(self.workspace.canonical_repository).rstrip("/")):
            if path == root:
                return ""
            if path.startswith(root + "/"):
                return path[len(root) + 1 :]
        return path

    def _capabilities(self, state):
        return self.runtime.organization.capabilities(state.trigger.agent_id)

    def schemas(self, state=None):
        caps = (
            self._capabilities(state) if state is not None else {"read_workspace", "write_artifact"}
        )
        path = {
            "path": {
                "type": "string",
                "minLength": 1,
                "maxLength": 400,
                "description": "File path relative to the repository root, or under /workspace.",
            }
        }
        content = {
            "type": "string",
            "maxLength": min(
                getattr(getattr(self.runtime, "_limits", None), "file_chars", 12000),
                self.workspace.limits.max_file_bytes,
            ),
        }
        schemas = []
        if "read_workspace" in caps:
            schemas += [
                dict(ACTIVITY_SCHEMA),
                action_schema(
                    "list_files",
                    {**path, "offset": {"type": "integer", "minimum": 0}},
                    required=[],
                ),
                action_schema(
                    "read_file",
                    {
                        **path,
                        "offset": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Line to start from; the first line is 1. Pass "
                            "next_offset from the previous result to continue.",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Maximum number of lines to read.",
                        },
                        "column": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Character offset within the starting line. Pass next_column with next_offset when continuing a long line.",
                        },
                    },
                    required=["path"],
                ),
            ]
        if "write_artifact" in caps:
            schemas += [
                action_schema("write_file", {**path, "content": content}),
                action_schema(
                    "edit_file",
                    {
                        **path,
                        "edits": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 50,
                            "description": "Replacements, each matched against the file as it "
                            "was before this edit.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old_text": {
                                        **content,
                                        "minLength": 1,
                                        "description": "Exact text to replace; must occur once.",
                                    },
                                    "new_text": {
                                        **content,
                                        "description": "Replacement text.",
                                    },
                                },
                                "required": ["old_text", "new_text"],
                                "additionalProperties": False,
                            },
                        },
                    },
                ),
            ]
        descriptions = {
            "list_files": "List files and directories directly under path (default repository root). Pass next_offset to continue.",
            "list_activity": ACTIVITY_DESCRIPTION,
            "read_file": (
                "Read a UTF-8 text file, with a line number before each line, up to "
                f"{MAX_LINES} lines or {MAX_BYTES // 1024} KiB at a time."
            ),
            "write_file": "Create or replace a whole UTF-8 text file, creating parent "
            "directories as needed.",
            "edit_file": (
                "Replace exact text in a file. Edits must not overlap, and old_text leaves out "
                "the line numbers that read_file adds."
            ),
        }
        for schema in schemas:
            schema["description"] = descriptions[schema["properties"]["action"]["enum"][0]]
        return schemas

    def apply(self, state, decision):
        actor, action = state.trigger.agent_id, decision["action"]
        cap = "write_artifact" if action in {"write_file", "edit_file"} else "read_workspace"
        if cap not in self._capabilities(state):
            access = "write" if cap == "write_artifact" else "read"
            raise WorkspaceAccessError(
                f"{action} needs repository {access} access, which this run doesn't grant"
            )
        self.workspace._check_actor(actor)
        self.workspace._require_initialized()
        if isinstance(decision.get("path"), str):
            decision = {**decision, "path": self._repository_relative(decision["path"])}
        path = decision.get("path")
        if isinstance(path, str) and path.startswith("/"):
            raise WorkspaceAccessError(
                f"{path} is outside the repository; file tools only reach paths under /workspace"
            )
        if action != "list_activity" and self.prepare_action is not None:
            if guidance := self.prepare_action(state, decision):
                return False, guidance
        if action == "list_files":
            path = path or "."
            result = self.workspace.list_files(actor, path, decision.get("offset", 0))
            event_action = action
        elif action == "list_activity":
            self.runtime.organization.authorize(state)
            if not self.store.is_project_member(actor, state.trigger.project_id):
                raise WorkspaceAccessError(f"{actor} is not a member of this project")
            result = list_activity(
                self.store,
                state.trigger.project_id,
                decision,
                command_statuses=self.runtime.command_statuses(),
                reader_id=actor,
            )
            event_action = action
        elif action == "read_file":
            result = self.workspace.read_lines(
                actor,
                path,
                offset=decision.get("offset"),
                limit=decision.get("limit"),
                column=decision.get("column", 0),
            )
            event_action = "workspace_read"
        elif action == "write_file":
            result = self.workspace.write_text(actor, path, decision["content"])
            event_action = "workspace_write"
        else:
            result = self.workspace.edit_text(actor, path, decision["edits"])
            event_action = "workspace_edit"
        record = {
            **result,
            "project_id": state.trigger.project_id,
        }
        self.store.record_workspace_action(
            state.activation_id, actor, event_action, "succeeded", path, record
        )
        return False, _model_result(action, path, result)


def _model_result(action, path, result):
    """What the model sees; versions and digests stay in the recorded event."""
    if action == "read_file":
        return {
            key: result[key]
            for key in ("path", "content", "next_offset", "next_column")
            if key in result
        }
    if action == "write_file":
        return {"path": path, "message": f"Wrote {result['bytes']} bytes to {path}."}
    if action == "edit_file":
        blocks = result["replaced_blocks"]
        return {"path": path, "message": f"Replaced {blocks} block(s) in {path}."}
    return result
