"""Project-scoped activity queries; operator messages stay visible to recipients."""

from __future__ import annotations

# Recorded event types, grouped by category and shown under the tool name a peer used.
CATEGORIES = {
    "files": {
        "list_files": "list_files",
        "workspace_read": "read_file",
        "workspace_write": "write_file",
        "workspace_edit": "edit_file",
        "read_file": "read_file",
        "write_file": "write_file",
        "edit_file": "edit_file",
    },
    "commands": {
        "request_command": "run_command",
        "run_command": "run_command",
        "execute_command": "command_finished",
    },
    "pages": {
        name: name
        for name in (
            "list_pages",
            "search_pages",
            "view_page",
            "add_page",
            "edit_page",
            "comment_page",
            "read_comment",
        )
    },
    "activations": {"begin_peer_session": "activation_started", "peer_yield": "peer_yield"},
    "messages": {"scheduled_notification": "scheduled_message"},
}
HIDDEN_EVENTS = ("list_activity", "check_budget", "read_output")
# Recorded actions that change shared state; reads and listings are left out.
CHANGE_EVENTS = (
    "workspace_write",
    "workspace_edit",
    "write_file",
    "edit_file",
    "execute_command",
    "add_page",
    "edit_page",
    "comment_page",
)


def _category_sql():
    cases = " ".join(
        f"WHEN e.action_type IN ({','.join(repr(event) for event in events)}) THEN '{category}'"
        for category, events in CATEGORIES.items()
    )
    return f"CASE {cases} ELSE 'other' END"


def _event_scope(project, agent, reader_id):
    # Session events inherit trigger scope. Operator messages additionally require
    # the reader to be one of the explicit recipients, even on the shared objective.
    hidden = ",".join(repr(event) for event in HIDDEN_EVENTS)
    return (
        f"""FROM events e JOIN activations a ON a.id=e.activation_id
        JOIN triggers t ON t.id=a.trigger_id
        WHERE (t.project_id=? OR (t.project_id IS NULL AND (
            json_extract(e.payload,'$.project_id')=? OR
            (e.action_type='scheduled_notification' AND EXISTS
             (SELECT 1 FROM organization_config WHERE scope_id=?)))))
        AND (? IS NULL OR e.agent_id=?) AND e.action_type NOT IN ({hidden})
        AND (e.action_type!='scheduled_notification' OR EXISTS
             (SELECT 1 FROM json_each(e.payload,'$.recipients') WHERE value=?))""",
        [project, project, project, agent, agent, reader_id],
    )


class ActivityQueries:
    def __init__(self, connection):
        self._connection = connection

    def project_peer_ids(self, project_id):
        return [
            row[0]
            for row in self._connection.execute(
                "SELECT agent_id FROM project_members WHERE project_id=? ORDER BY agent_id",
                (project_id,),
            )
        ]

    def command(self, command_id):
        row = self._connection.execute(
            "SELECT * FROM command_requests WHERE id=?", (command_id,)
        ).fetchone()
        return dict(row) if row else None

    def approved_commands(self, project_id, peer_id):
        return [
            dict(row)
            for row in self._connection.execute(
                "SELECT * FROM command_requests WHERE project_id=? AND status='approved' "
                "AND (? IS NULL OR requested_by=?) ORDER BY id",
                (project_id, peer_id, peer_id),
            )
        ]

    def recent_events(
        self,
        project_id,
        *,
        peer_id,
        reader_id,
        since_tick,
        before,
        category,
        path,
        limit,
        changes_only=False,
    ):
        scope, args = _event_scope(project_id, peer_id, reader_id)
        # A finished command may have changed files whatever its exit status.
        changes = (
            " AND action_type IN ("
            + ",".join(repr(event) for event in CHANGE_EVENTS)
            + ") AND (outcome='succeeded' OR action_type='execute_command')"
            if changes_only
            else ""
        )
        return [
            dict(row)
            for row in self._connection.execute(
                "WITH activity AS (SELECT e.*, "
                + _category_sql()
                + " AS category "
                + scope
                + """) SELECT * FROM activity
                WHERE tick>? AND (? IS NULL OR tick<?) AND (? IS NULL OR category=?)
                AND (? IS NULL OR resource_id=? OR substr(resource_id,1,length(?)+1)=? || '/')"""
                + changes
                + " ORDER BY tick DESC LIMIT ?",
                [*args, since_tick, before, before, category, category, *([path] * 4), limit],
            )
        ]
