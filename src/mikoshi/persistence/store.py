from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mikoshi.contracts import Agent, Trigger
from mikoshi.persistence.activity import ActivityQueries
from mikoshi.persistence.conversations import ConversationRepository, GuidanceRepository
from mikoshi.persistence.notifications import InboxRepository


def require_database_schema(connection, expected, label):
    """Reject missing or changed runtime tables and indexes without modifying them."""

    def definitions(database):
        return {
            (row[0], row[1]): " ".join(row[2].split())
            for row in database.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL"
            )
        }

    actual = definitions(connection)
    mismatched = [key[1] for key, sql in definitions(expected).items() if actual.get(key) != sql]
    if mismatched:
        raise ValueError(f"{label} database schema does not match runtime: {', '.join(mismatched)}")


class WorldStore:
    def __init__(self, database: str | Path = ":memory:") -> None:
        self._connection = sqlite3.connect(str(database))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._exchange_observers = []
        self._committed_event_observers: list[Callable[[Mapping[str, object]], None]] = []
        self._event_observer_errors: list[str] = []
        try:
            self._create_schema()
        except Exception:
            self._connection.close()
            raise
        self.conversations = ConversationRepository(self._connection)
        self.inbox = InboxRepository(self._connection)
        self.repository_guidance = GuidanceRepository(self._connection)
        self.activity_queries = ActivityQueries(self._connection)

    @property
    def execution_quarantined(self):
        return bool(
            self._connection.execute(
                "SELECT 1 FROM metadata WHERE key='execution_quarantined'"
            ).fetchone()
        )

    def quarantine_execution(self):
        """Persist the execution barrier after an unresolved integrity failure."""
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES ('execution_quarantined',1)"
            )

    def latest_causal_wave_id(self):
        row = self._connection.execute(
            "SELECT COALESCE(MAX(causal_wave_id),1) FROM activations"
        ).fetchone()
        return max(1, row[0])

    def organization_configuration(self):
        if not self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='organization_config'"
        ).fetchone():
            return None
        row = self._connection.execute("SELECT * FROM organization_config").fetchone()
        return dict(row) if row else None

    def list_inbox_messages(self, *, event_id=None):
        if event_id is None:
            rows = self._connection.execute("SELECT * FROM inbox_messages ORDER BY id")
        else:
            rows = self._connection.execute(
                "SELECT * FROM inbox_messages WHERE event_id=? ORDER BY id", (event_id,)
            )
        return [dict(row) for row in rows]

    def page_path(self, page_id):
        """Resolve a durable page identity to its current peer-visible path."""
        from mikoshi.tools.shared_pages import page_path

        return page_path(self._connection, page_id)

    def observe_exchanges(self, observer):
        self._exchange_observers.append(observer)

    def publish_exchange(self, exchange):
        for observer in tuple(self._exchange_observers):
            try:
                observer(exchange)
            except Exception as error:
                self._event_observer_errors.append(f"exchange: {type(error).__name__}: {error}")

    def observe_committed_events(
        self,
        observer: Callable[[Mapping[str, object]], None],
    ) -> None:
        """Notify an observer after each domain transaction containing an event commits.

        Observers are deliberately best-effort: projection or telemetry failures must not
        roll back an already committed lifecycle action. A later event (or an explicit
        rebuild) lets an idempotent observer catch up from the ledger.
        """
        if observer not in self._committed_event_observers:
            self._committed_event_observers.append(observer)

    @property
    def event_observer_errors(self) -> Sequence[str]:
        return tuple(self._event_observer_errors)

    def record_observer_error(self, source: str, error: Exception) -> None:
        """Retain a failure without copying potentially sensitive exception text."""
        self._event_observer_errors.append(f"{source}: {type(error).__name__}"[:1000])

    def close(self) -> None:
        self._connection.close()

    def backup(self, database: Path) -> None:
        """Atomically write a consistent SQLite backup of the current world."""
        database.parent.mkdir(parents=True, exist_ok=True)
        temporary = database.with_name(f".{database.name}.tmp")
        if temporary.exists():
            temporary.unlink()
        target = sqlite3.connect(str(temporary))
        try:
            self._connection.backup(target)
        finally:
            target.close()
        temporary.replace(database)

    def _create_schema(self) -> None:
        existing = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        if existing:
            expected = WorldStore()
            try:
                if any(row[0] == "organization_config" for row in existing):
                    from mikoshi.runtime.organization import SCHEMA as ORGANIZATION_SCHEMA

                    expected._connection.executescript(ORGANIZATION_SCHEMA)
                require_database_schema(self._connection, expected._connection, "run")
            finally:
                expected.close()
            tick = self._connection.execute(
                "SELECT value FROM metadata WHERE key='current_tick'"
            ).fetchone()
            if tick is None or type(tick[0]) is not int or tick[0] < 0:
                raise ValueError("run database has an invalid current tick")
            return
        self._connection.executescript("""
			CREATE TABLE IF NOT EXISTS metadata (
				key TEXT PRIMARY KEY,
				value INTEGER NOT NULL
			);
			INSERT OR IGNORE INTO metadata(key, value) VALUES ('current_tick', 0);

			CREATE TABLE IF NOT EXISTS agents (
				id TEXT PRIMARY KEY,
				name TEXT NOT NULL,
				instructions TEXT NOT NULL,
				role TEXT NOT NULL DEFAULT 'member'
			);

			CREATE TABLE IF NOT EXISTS projects (
				id TEXT PRIMARY KEY,
				name TEXT NOT NULL,
				goal TEXT NOT NULL,
				state TEXT NOT NULL CHECK(state IN ('pending', 'active', 'completed', 'blocked')),
				created_tick INTEGER NOT NULL,
				updated_tick INTEGER NOT NULL
			);

			CREATE TABLE IF NOT EXISTS project_members (
				project_id TEXT NOT NULL REFERENCES projects(id),
				agent_id TEXT NOT NULL REFERENCES agents(id),
				PRIMARY KEY(project_id, agent_id)
			);

			CREATE TABLE IF NOT EXISTS triggers (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				agent_id TEXT NOT NULL REFERENCES agents(id),
				reason TEXT NOT NULL,
				priority INTEGER NOT NULL,
				cause_event_id INTEGER REFERENCES events(id),
				created_tick INTEGER NOT NULL,
				status TEXT NOT NULL CHECK(status IN ('queued', 'running', 'completed')),
				kind TEXT NOT NULL CHECK(kind IN (
					'environment', 'peer_session'
				)),
				project_id TEXT
			);

			CREATE TABLE IF NOT EXISTS activations (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				agent_id TEXT NOT NULL REFERENCES agents(id),
				trigger_id INTEGER NOT NULL REFERENCES triggers(id),
				started_tick INTEGER NOT NULL,
				causal_wave_id INTEGER NOT NULL DEFAULT 0,
				current_decision_round INTEGER NOT NULL DEFAULT 0,
				current_activation_step INTEGER NOT NULL DEFAULT 0,
                current_decision_id TEXT,
                current_request_id TEXT,
                current_tool_call_id TEXT,
                current_tool_call_index INTEGER,
				ended_tick INTEGER,
				status TEXT NOT NULL,
				action_count INTEGER NOT NULL DEFAULT 0,
				error TEXT
			);

            CREATE TABLE IF NOT EXISTS events (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				tick INTEGER NOT NULL UNIQUE,
				commit_ordinal INTEGER NOT NULL DEFAULT 0,
				activation_id INTEGER NOT NULL REFERENCES activations(id),
				causal_wave_id INTEGER NOT NULL DEFAULT 0,
				decision_round INTEGER NOT NULL DEFAULT 0,
				activation_step INTEGER NOT NULL DEFAULT 0,
				agent_id TEXT NOT NULL REFERENCES agents(id),
				action_type TEXT NOT NULL,
				outcome TEXT NOT NULL,
				resource_id TEXT,
				payload TEXT NOT NULL
			);

			CREATE TABLE IF NOT EXISTS command_requests (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				project_id TEXT NOT NULL REFERENCES projects(id),
				activation_id INTEGER NOT NULL REFERENCES activations(id),
				requested_by TEXT NOT NULL REFERENCES agents(id),
				argv_json TEXT NOT NULL,
				working_directory TEXT NOT NULL,
                decision_id TEXT,
                request_id TEXT,
                tool_call_id TEXT,
                tool_call_index INTEGER,
				status TEXT NOT NULL CHECK(status IN (
					'approved',
					'succeeded', 'failed', 'timed_out'
				)),
				exit_code INTEGER,
				termination_reason TEXT,
				output TEXT,
				output_truncated INTEGER NOT NULL DEFAULT 0
					CHECK(output_truncated IN (0, 1)),
				log_path TEXT,
				duration_ms INTEGER,
				workspace_tree_sha256 TEXT,
				requested_tick INTEGER NOT NULL,
				completed_tick INTEGER,
                isolation_json TEXT NOT NULL DEFAULT '{}',
                timeout_seconds REAL
			);

			CREATE INDEX IF NOT EXISTS idx_project_members_agent
				ON project_members(agent_id, project_id);
			CREATE INDEX IF NOT EXISTS idx_triggers_queue
				ON triggers(status, priority DESC, id);
			CREATE INDEX IF NOT EXISTS idx_triggers_coalescing
				ON triggers(agent_id, reason, status);
			CREATE UNIQUE INDEX IF NOT EXISTS idx_events_commit_ordinal
                ON events(commit_ordinal);
			CREATE INDEX IF NOT EXISTS idx_events_activation
				ON events(activation_id, tick);
		""")
        from mikoshi.runtime.conversations import SCHEMA as CONVERSATION_SCHEMA
        from mikoshi.runtime.notifications import SCHEMA as INBOX_SCHEMA
        from mikoshi.tools.shared_pages import SCHEMA

        self._connection.executescript(SCHEMA)
        self._connection.executescript(CONVERSATION_SCHEMA)
        self._connection.executescript(INBOX_SCHEMA)
        self._connection.commit()

    @property
    def current_tick(self) -> int:
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'current_tick'"
        ).fetchone()
        return int(row["value"])

    def get_agent(self, agent_id: str) -> Agent:
        row = self._connection.execute(
            "SELECT id, name, instructions, role FROM agents WHERE id = ?", (agent_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown agent {agent_id!r}")
        return Agent(**dict(row))

    def list_agents(self) -> Sequence[Agent]:
        rows = self._connection.execute(
            "SELECT id, name, instructions, role FROM agents ORDER BY id"
        ).fetchall()
        return tuple(Agent(**dict(row)) for row in rows)

    def is_project_member(self, agent_id: str, project_id: str) -> bool:
        return (
            self._connection.execute(
                "SELECT 1 FROM project_members WHERE agent_id=? AND project_id=?",
                (agent_id, project_id),
            ).fetchone()
            is not None
        )

    def list_projects(self) -> Sequence[Mapping[str, Any]]:
        return tuple(
            dict(row) for row in self._connection.execute("SELECT * FROM projects ORDER BY id")
        )

    def claim_next_trigger(
        self,
        excluded_agent_ids: Sequence[str] = (),
    ) -> Trigger | None:
        with self._connection:
            from mikoshi.runtime.organization import admit_activation

            # Swarm roles carry no scheduling privilege. Prefer the identity least
            # recently activated; activation IDs preserve fairness across session wakeups.
            order = (
                "COALESCE((SELECT MAX(a.id) FROM activations a "
                "WHERE a.agent_id=triggers.agent_id),0), triggers.priority DESC, triggers.id ASC"
            )

            rows = self._connection.execute(
                """
				SELECT id, agent_id, reason, priority, cause_event_id, created_tick,
					kind, project_id
				FROM triggers
				WHERE triggers.status = 'queued'
				ORDER BY """
                + order
            ).fetchall()
            excluded = set(excluded_agent_ids)
            row = next(
                (
                    item
                    for item in rows
                    if item["agent_id"] not in excluded
                    and admit_activation(self, Trigger(**dict(item)))
                ),
                None,
            )
            if row is None:
                return None
            self._connection.execute(
                "UPDATE triggers SET status = 'running' WHERE id = ?", (row["id"],)
            )
        return Trigger(**dict(row))

    def start_activation(self, trigger: Trigger, causal_wave_id: int) -> int:
        if causal_wave_id < 1:
            raise ValueError("causal_wave_id must be at least 1")
        with self._connection:
            cursor = self._connection.execute(
                """
				INSERT INTO activations(
					agent_id, trigger_id, started_tick, causal_wave_id, status
				) VALUES (?, ?, ?, ?, 'running')
				""",
                (trigger.agent_id, trigger.id, self.current_tick, causal_wave_id),
            )
        return int(cursor.lastrowid)

    def set_activation_context(
        self,
        activation_id: int,
        decision_round: int,
        activation_step: int,
    ) -> None:
        """Set the immutable labels copied onto subsequent events in this action."""
        if decision_round < 1 or activation_step < 1:
            raise ValueError("decision round and activation step must be at least 1")
        with self._connection:
            cursor = self._connection.execute(
                """
				UPDATE activations
				SET current_decision_round = ?, current_activation_step = ?
				WHERE id = ? AND status = 'running'
				""",
                (decision_round, activation_step, activation_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("temporal context requires a running activation")

    def set_action_context(
        self,
        activation_id,
        *,
        decision_id=None,
        request_id=None,
        tool_call_id=None,
        tool_call_index=None,
    ):
        """Retain observed request/call identities for events emitted by this action."""
        with self._connection:
            cursor = self._connection.execute(
                "UPDATE activations SET current_decision_id=?, current_request_id=?, current_tool_call_id=?, current_tool_call_index=? WHERE id=?",
                (decision_id, request_id, tool_call_id, tool_call_index, activation_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("action context requires an existing activation")

    def finish_activation(
        self,
        activation_id: int,
        trigger_id: int,
        status: str,
        action_count: int,
        error: str | None = None,
    ) -> None:
        with self._connection:
            self.finish_activation_in_transaction(
                activation_id, trigger_id, status, action_count, error
            )

    def finish_activation_in_transaction(
        self, activation_id, trigger_id, status, action_count, error=None
    ):
        """Settle the trigger and any unread wake inside the caller's action transaction."""
        self._connection.execute(
            "UPDATE activations SET ended_tick=?, status=?, action_count=?, error=? WHERE id=?",
            (self.current_tick, status, action_count, error, activation_id),
        )
        self._connection.execute(
            "UPDATE triggers SET status = 'completed' WHERE id = ?", (trigger_id,)
        )
        if status == "yielded":
            from mikoshi.runtime.organization import wake_inbox

            actor = self._connection.execute(
                "SELECT agent_id FROM activations WHERE id=?", (activation_id,)
            ).fetchone()[0]
            wake_inbox(self, actor)

    def record_workspace_action(
        self,
        activation_id: int,
        agent_id: str,
        action_type: str,
        outcome: str,
        resource_id: str | None,
        payload: Mapping[str, object],
    ) -> int:
        """Append a brokered file-tool action to the authoritative event ledger."""
        with self._connection:
            event_id, _ = self._record_event(
                activation_id,
                agent_id,
                action_type,
                outcome,
                resource_id,
                dict(payload),
            )
        self._publish_committed_event(event_id)
        return event_id

    def request_command(
        self,
        activation_id: int,
        agent_id: str,
        argv: Sequence[str],
        working_directory: str,
        *,
        project_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Mapping[str, Any]:
        """Persist exact argv before container execution; no peer review is involved."""
        if project_id is None:
            raise ValueError("commands require a project scope")
        if not self._connection.execute(
            "SELECT 1 FROM project_members WHERE project_id=? AND agent_id=?",
            (project_id, agent_id),
        ).fetchone():
            raise ValueError("command requester must belong to its project")
        activation = self._connection.execute(
            "SELECT * FROM activations WHERE id=? AND agent_id=?", (activation_id, agent_id)
        ).fetchone()
        if activation is None:
            raise ValueError("command requester must own its activation")
        with self._connection:
            requested_tick = self.current_tick + 1
            cursor = self._connection.execute(
                """
				INSERT INTO command_requests(
					project_id, activation_id, requested_by,
					argv_json, working_directory, decision_id, request_id, tool_call_id, tool_call_index,
					status, requested_tick, timeout_seconds
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'approved', ?, ?)
				""",
                (
                    project_id,
                    activation_id,
                    agent_id,
                    json.dumps(list(argv)),
                    working_directory,
                    activation["current_decision_id"],
                    activation["current_request_id"],
                    activation["current_tool_call_id"],
                    activation["current_tool_call_index"],
                    requested_tick,
                    timeout_seconds,
                ),
            )
            request_id = int(cursor.lastrowid)
            event_id, event_tick = self._record_event(
                activation_id,
                agent_id,
                "request_command",
                "authorized",
                f"command:{request_id}",
                {
                    "project_id": project_id,
                    "argv": list(argv),
                    "working_directory": working_directory,
                },
            )
            if event_tick != requested_tick:
                raise RuntimeError("command request tick diverged from its event")
        self._publish_committed_event(event_id)
        return self.get_command_request(request_id)

    def complete_command_request(
        self,
        activation_id: int,
        agent_id: str,
        request_id: int,
        result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        request = self.get_command_request(request_id)
        if request["status"] != "approved":
            raise ValueError("a command must be approved before execution is recorded")
        if request["requested_by"] != agent_id:
            raise ValueError("command result has the wrong requester")
        status = str(result["status"])
        if status not in {"succeeded", "failed", "timed_out"}:
            raise ValueError(f"invalid command result status {status!r}")
        output = str(result.get("output", ""))
        workspace_state = result.get("workspace_state")
        workspace_tree_sha256 = (
            workspace_state.get("tree_sha256") if isinstance(workspace_state, Mapping) else None
        )
        self.set_action_context(
            activation_id,
            decision_id=request["decision_id"],
            request_id=request["request_id"],
            tool_call_id=request["tool_call_id"],
            tool_call_index=request["tool_call_index"],
        )
        with self._connection:
            event_id, tick = self._record_event(
                activation_id,
                agent_id,
                "execute_command",
                status,
                f"command:{request_id}",
                {
                    "project_id": request["project_id"],
                    "exit_code": result.get("exit_code"),
                    "termination_reason": result.get("termination_reason"),
                    "isolation": result.get("isolation", {}),
                    "duration_ms": result.get("duration_ms"),
                    "output_truncated": bool(result.get("output_truncated", False)),
                    "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                    "log_path": result.get("log_path"),
                    "workspace_tree_sha256": workspace_tree_sha256,
                    "workspace_observation": "non_atomic; not execution attribution",
                    "workspace_error": result.get("workspace_error"),
                    "resource_version_error": result.get("resource_version_error"),
                    "resource_versions_before": list(result.get("resource_versions_before", ())),
                    "resource_versions": list(result.get("resource_versions", ())),
                },
            )
            self._connection.execute(
                """
				UPDATE command_requests
				SET status = ?, exit_code = ?, termination_reason = ?, output = ?,
					output_truncated = ?, log_path = ?, duration_ms = ?,
					workspace_tree_sha256 = ?, completed_tick = ?, isolation_json = ?
				WHERE id = ?
				""",
                (
                    status,
                    result.get("exit_code"),
                    result.get("termination_reason"),
                    output,
                    int(bool(result.get("output_truncated", False))),
                    result.get("log_path"),
                    result.get("duration_ms"),
                    workspace_tree_sha256,
                    tick,
                    json.dumps(result.get("isolation", {}), sort_keys=True),
                    request_id,
                ),
            )
        self._publish_committed_event(event_id)
        return self.get_command_request(request_id)

    def get_command_request(self, request_id: int) -> Mapping[str, Any]:
        row = self._connection.execute(
            "SELECT * FROM command_requests WHERE id = ?", (request_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown command request {request_id}")
        return _command_request_from_row(row)

    def list_command_requests(self) -> Sequence[Mapping[str, Any]]:
        rows = self._connection.execute("SELECT * FROM command_requests ORDER BY id").fetchall()
        return tuple(_command_request_from_row(row) for row in rows)

    def _record_event(
        self,
        activation_id: int,
        agent_id: str,
        action_type: str,
        outcome: str,
        resource_id: str | None,
        payload: dict[str, object],
    ) -> tuple[int, int]:
        tick = self.current_tick + 1
        activation = self._connection.execute(
            """
			SELECT causal_wave_id, current_decision_round, current_activation_step,
                current_decision_id, current_request_id, current_tool_call_id, current_tool_call_index
			FROM activations WHERE id = ?
			""",
            (activation_id,),
        ).fetchone()
        if activation is None:
            raise ValueError(f"unknown activation {activation_id}")
        if activation["current_decision_id"] is not None:
            payload.update(
                decision_id=activation["current_decision_id"],
                request_id=activation["current_request_id"],
                tool_call_id=activation["current_tool_call_id"],
                tool_call_index=activation["current_tool_call_index"],
            )
        cursor = self._connection.execute(
            """
			INSERT INTO events(
				tick, commit_ordinal, activation_id, causal_wave_id, decision_round,
				activation_step, agent_id, action_type, outcome, resource_id, payload
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			""",
            (
                tick,
                tick,
                activation_id,
                int(activation["causal_wave_id"]),
                int(activation["current_decision_round"]),
                int(activation["current_activation_step"]),
                agent_id,
                action_type,
                outcome,
                resource_id,
                json.dumps(payload, sort_keys=True),
            ),
        )
        self._connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'current_tick'", (tick,)
        )
        return int(cursor.lastrowid), tick

    def _publish_committed_event(self, event_id: int) -> None:
        """Publish exactly one durable event after its enclosing transaction commits."""
        if not self._committed_event_observers:
            return
        row = self._connection.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"committed event {event_id} is unavailable")
        event = _event_from_row(row)
        for observer in tuple(self._committed_event_observers):
            try:
                observer(event)
            except Exception as error:
                self._event_observer_errors.append(
                    f"event {event_id}: {type(error).__name__}: {error}"
                )

    def queued_trigger_count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS count FROM triggers WHERE status = 'queued'"
        ).fetchone()
        return int(row["count"])

    def list_events(self) -> Sequence[dict[str, object]]:
        rows = self._connection.execute("SELECT * FROM events ORDER BY tick").fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def list_activations(self) -> Sequence[dict[str, object]]:
        rows = self._connection.execute("SELECT * FROM activations ORDER BY id").fetchall()
        return tuple(dict(row) for row in rows)

    def list_triggers(self) -> Sequence[dict[str, object]]:
        rows = self._connection.execute("SELECT * FROM triggers ORDER BY id").fetchall()
        return tuple(dict(row) for row in rows)


def _event_from_row(row: sqlite3.Row) -> dict[str, object]:
    value = dict(row)
    value["payload"] = json.loads(value["payload"])
    return value


def _command_request_from_row(row: sqlite3.Row) -> Mapping[str, Any]:
    value = dict(row)
    value["argv"] = tuple(json.loads(value.pop("argv_json")))
    value["output_truncated"] = bool(value["output_truncated"])
    value["isolation"] = json.loads(value.pop("isolation_json"))
    return value
