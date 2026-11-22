"""Reopening stored runs requires intact structure and organization identities."""

import sqlite3
from contextlib import closing

import pytest

from mikoshi.persistence.store import WorldStore
from mikoshi.problem import ObjectiveSpec, ProblemSpec, ResourceLimits
from mikoshi.runtime.organization import bootstrap, require_current_organization


def test_backup_reopens_with_organization_and_recorded_state(tmp_path):
    problem = ProblemSpec(
        ObjectiveSpec("objective", "Retain evidence", "Keep the records."), ResourceLimits(2)
    )
    with closing(WorldStore()) as store:
        bootstrap(store, problem)
        expected = require_current_organization(store)
        agents = store.list_agents()
        trigger = store.claim_next_trigger()
        activation = store.start_activation(trigger, 1)
        store.record_workspace_action(
            activation, trigger.agent_id, "inspect", "succeeded", None, {}
        )
        store.finish_activation(activation, trigger.id, "yielded", 1)
        events = store.list_events()
        activations = store.list_activations()
        triggers = store.list_triggers()
        archive = tmp_path / "run.sqlite"
        store.backup(archive)
    with closing(WorldStore(archive)) as restored:
        assert restored.list_agents() == agents
        assert require_current_organization(restored) == expected
        assert restored.list_events() == events
        assert restored.list_activations() == activations
        assert restored.list_triggers() == triggers


@pytest.mark.parametrize(
    "corruption",
    [
        "DROP TABLE command_requests",
        "ALTER TABLE agent_memory_exchanges DROP COLUMN request_id",
        "DROP INDEX idx_events_commit_ordinal",
        "DELETE FROM metadata WHERE key='current_tick'",
    ],
)
def test_reopening_rejects_incomplete_database_without_changing_it(tmp_path, corruption):
    database = tmp_path / "run.sqlite"
    WorldStore(database).close()
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(corruption)
        connection.commit()
        before = list(connection.iterdump())
    with pytest.raises(ValueError, match="database"):
        WorldStore(database)
    with closing(sqlite3.connect(database)) as connection:
        assert list(connection.iterdump()) == before


@pytest.mark.parametrize("corruption", ["table", "roster"])
def test_organization_attachment_requires_structure_and_matching_roster(corruption):
    problem = ProblemSpec(
        ObjectiveSpec("objective", "Retain evidence", "Keep the records."), ResourceLimits(2)
    )
    with closing(WorldStore()) as store:
        bootstrap(store, problem)
        if corruption == "table":
            store._connection.execute("DROP TABLE organization_config")
        else:
            store._connection.execute("UPDATE agents SET role='environment' WHERE id='agent-2'")
        with pytest.raises(ValueError, match="schema|roster"):
            require_current_organization(store)


def test_organization_attachment_ignores_operator_environment_identity():
    problem = ProblemSpec(
        ObjectiveSpec("objective", "Retain evidence", "Keep the records."), ResourceLimits(2)
    )
    with closing(WorldStore()) as store:
        bootstrap(store, problem)
        expected = require_current_organization(store)
        store._connection.execute(
            "INSERT INTO agents VALUES ('environment','Operator','Private instructions','environment')"
        )
        assert require_current_organization(store) == expected
