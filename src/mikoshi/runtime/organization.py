"""Fixed peer identities and activation scheduling over the runtime ledger."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict

from mikoshi.contracts import action_schema
from mikoshi.persistence.store import require_database_schema

SCHEMA = """
CREATE TABLE IF NOT EXISTS organization_config (
 singleton INTEGER PRIMARY KEY CHECK(singleton=1), problem_json TEXT NOT NULL,
 scope_id TEXT NOT NULL REFERENCES projects(id)
);
"""


def config(store):
    return store.organization_configuration()


def require_current_organization(store):
    """Validate a stored organization before attaching execution or exporting files."""
    from mikoshi.problem import fixed_members, parse_problem

    with closing(sqlite3.connect(":memory:")) as expected:
        expected.executescript(SCHEMA)
        require_database_schema(store._connection, expected, "organization")
    configuration = config(store)
    if configuration is None:
        raise ValueError("run database must contain a problem organization")
    try:
        raw = json.loads(configuration["problem_json"])
        if not isinstance(raw, dict):
            raise ValueError("stored problem must be an object")
        if "team" not in raw:
            raise ValueError("stored problem requires an explicit team")
        problem = parse_problem(raw)
        expected = {member.id for member in fixed_members(problem)}
        roster = {agent.id for agent in store.list_agents() if agent.role == "peer"}
        if roster != expected:
            raise ValueError("fixed team roster differs from the configured identities")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid stored problem: {error}") from error
    return configuration


def bootstrap(store, problem):
    """Initialize the configured roster and empty objective scope before workspace seeding."""
    from mikoshi.problem import fixed_members, validate_problem

    problem = validate_problem(problem)
    db = store._connection
    spec = asdict(problem)
    encoded = json.dumps(spec, sort_keys=True)
    existing = config(store)
    if existing:
        require_current_organization(store)
        prior_spec = json.loads(existing["problem_json"])
        if json.dumps(prior_spec, sort_keys=True) != encoded:
            raise ValueError("problem configuration differs from run database")
        return
    if store.list_agents() or store.list_projects():
        raise ValueError("problem initialization requires an empty world")
    db.executescript(SCHEMA)
    objective = spec["objective"]
    roster = fixed_members(problem)
    with db:
        for member in roster:
            db.execute(
                "INSERT INTO agents VALUES (?,?,?,?)",
                (member.id, member.id, member.instructions, "peer"),
            )
        db.execute(
            "INSERT INTO projects VALUES (?,?,?,'active',?,?)",
            (
                objective["id"],
                objective["title"],
                objective["description"],
                store.current_tick,
                store.current_tick,
            ),
        )
        db.executemany(
            "INSERT INTO project_members VALUES (?,?)",
            [(objective["id"], member.id) for member in roster],
        )
        db.execute(
            "INSERT INTO organization_config VALUES (1,?,?)",
            (encoded, objective["id"]),
        )
        _enqueue(store, "initialize organization from the problem")


def peer_count(store):
    """Every peer may hold a live activation; model request slots are limited separately."""
    (count,) = store._connection.execute("SELECT COUNT(*) FROM agents WHERE role='peer'").fetchone()
    return count


def _enqueue(store, reason, cause_event_id=None, recipients=None):
    """Coalesce wakeups independently per identity; new evidence can wake waiting peers."""
    db, configuration = store._connection, config(store)
    if not configuration:
        return None
    recipients = (
        recipients
        if recipients is not None
        else [row[0] for row in db.execute("SELECT id FROM agents WHERE role='peer' ORDER BY id")]
    )
    trigger_id = None
    for recipient in recipients:
        queued = db.execute(
            "SELECT id FROM triggers WHERE agent_id=? AND kind='peer_session' AND status='queued'",
            (recipient,),
        ).fetchone()
        trigger_id = (
            queued[0]
            if queued
            else db.execute(
                """INSERT INTO triggers(agent_id,reason,priority,cause_event_id,created_tick,status,kind,project_id)
            VALUES (?, ?, ?, ?, ?, 'queued', 'peer_session', ?)""",
                (
                    recipient,
                    reason,
                    50,
                    cause_event_id,
                    store.current_tick,
                    configuration["scope_id"],
                ),
            ).lastrowid
        )
    return trigger_id


def resolve_peers(store, author, project_id, requested, *, field):
    """Validate exact, authorized peer IDs for explicit comment notifications."""
    if not isinstance(requested, list) or not all(isinstance(name, str) for name in requested):
        raise ValueError(f"{field} must be a list of peer IDs")
    peers = {
        row[0]
        for row in store._connection.execute(
            "SELECT id FROM agents WHERE role='peer' AND "
            "(? IS NULL OR id IN (SELECT agent_id FROM project_members WHERE project_id=?))",
            (project_id, project_id),
        )
    }
    unknown = set(requested) - peers
    if unknown:
        raise ValueError(
            f"{field}: unknown peer IDs {', '.join(sorted(unknown))}; "
            f"peers are {', '.join(sorted(peers))}"
        )
    return sorted(set(requested) - {author})


def wake_inbox(store, agent_id):
    """Wake an idle peer for an undelivered wake-requesting inbox entry.

    Called in the writer's transaction and again after an activation yields.
    Busy peers need no queued follow-up if they consume the message before yielding.
    """
    from mikoshi.runtime.notifications import Notifications

    configuration = config(store)
    if not configuration:
        return
    db = store._connection
    if db.execute(
        "SELECT 1 FROM triggers WHERE agent_id=? AND status IN ('queued','running')",
        (agent_id,),
    ).fetchone():
        return
    pending = db.execute(
        "SELECT id,event_id FROM inbox_messages WHERE recipient=? AND id>? "
        "AND wake_requested=1 ORDER BY id LIMIT 1",
        (agent_id, Notifications(store).cursor(agent_id)),
    ).fetchone()
    if pending:
        _enqueue(
            store,
            f"inbox message {pending['id']}",
            cause_event_id=pending["event_id"],
            recipients=[agent_id],
        )


def admit_activation(store, trigger):
    """Admit one preserved session per identity; inference concurrency is scheduled separately."""
    configuration = config(store)
    if not configuration:
        raise ValueError("activation admission requires a problem-first organization")
    running = store._connection.execute(
        "SELECT DISTINCT agent_id FROM triggers WHERE status='running'"
    ).fetchall()
    # A previously claimed trigger owns its existing slot.
    existing = store._connection.execute(
        "SELECT status FROM triggers WHERE id=?", (trigger.id,)
    ).fetchone()
    if existing and existing["status"] == "running":
        return True
    return trigger.agent_id not in {r[0] for r in running}


class Organization:
    def __init__(self, store, pages, budget=None):
        self.budget = budget
        self.store, self.pages, self.db = store, pages, store._connection
        self.config = require_current_organization(store)
        self.problem = json.loads(self.config["problem_json"])

    def authorize(self, state):
        self.pages._actor(state)
        self._agent(state.trigger.agent_id)
        return config(self.store)

    def begin(self, activation_id, trigger):
        if trigger.kind != "peer_session":
            raise ValueError("unauthorized peer planning activation")
        self._agent(trigger.agent_id)
        self.store.record_workspace_action(
            activation_id,
            trigger.agent_id,
            "begin_peer_session",
            "succeeded",
            self.config["scope_id"],
            {"reason": trigger.reason},
        )
        return None

    def context(self, state):
        return {
            "agent_id": state.trigger.agent_id,
            "team": self.team_context(),
            "objective": dict(self.problem["objective"]),
            "shared_workspace": self.pages.context(state),
            "activation_origin": {"reason": state.trigger.reason},
        }

    def schemas(self):
        return [
            action_schema("peer_yield")
            | {
                "description": (
                    "End this activation. Yield only when you cannot contribute anything more "
                    "to the objective. Finishing the area you chose isn't "
                    "enough, and neither is a passing test suite. Don't yield to wait for a "
                    "reply. If every peer yields, the run ends and the remaining time and "
                    "budget go unused. Must be the only call in its turn."
                )
            },
        ]

    def team_context(self):
        ids = [
            row[0] for row in self.db.execute("SELECT id FROM agents WHERE role='peer' ORDER BY id")
        ]
        active = [
            row[0]
            for row in self.db.execute(
                "SELECT DISTINCT agent_id FROM triggers WHERE status IN ('queued','running') "
                "AND agent_id IN (SELECT id FROM agents WHERE role='peer') ORDER BY agent_id"
            )
        ]
        return {
            "peer_ids": ids,
            # Peers working now or queued to work, including you; the rest have yielded.
            "active_peer_ids": active,
        }

    def capabilities(self, agent_id):
        """Return one participant's static capabilities without building live snapshots."""
        self._agent(agent_id)
        return self.problem["team"]["capabilities"]

    def apply(self, state, decision):
        """End one activation while retaining its scheduling and audit evidence."""
        configuration = self.authorize(state)
        action = decision["action"]
        if action != "peer_yield":
            raise ValueError(f"unknown tool {action!r}")
        actor = state.trigger.agent_id
        payload = dict(decision)
        with self.db:
            event, _ = self.store._record_event(
                state.activation_id, actor, action, "succeeded", configuration["scope_id"], payload
            )
            self.store.finish_activation_in_transaction(
                state.activation_id, state.trigger.id, "yielded", state.steps
            )
        self.store._publish_committed_event(event)
        return True, {"status": "yielded"}

    def _agent(self, agent_id):
        row = self.db.execute(
            "SELECT * FROM agents WHERE id=? AND role='peer'", (agent_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"{agent_id} is not a peer")
        return row
