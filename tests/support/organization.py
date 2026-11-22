"""Current organization scenarios over real SQLite and Git state."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from support.runtime import create_repository

from mikoshi.application import build_from_problem
from mikoshi.persistence.store import WorldStore
from mikoshi.problem import (
    ObjectiveSpec,
    ProblemSpec,
    ResourceLimits,
    TeamMemberSpec,
    TeamSpec,
)
from mikoshi.tools.workspace import resolve_repository


def problem(**limits):
    return ProblemSpec(
        ObjectiveSpec(
            "catalog",
            "Maintain the repository index",
            "Index the repository documentation.",
            ("Preserve source files and identify evidence for index entries.",),
        ),
        replace(ResourceLimits(), **limits),
        team=TeamSpec(
            members=(TeamMemberSpec("participant"),)
            + tuple(
                TeamMemberSpec(f"worker_{chr(97 + index)}")
                for index in range(limits.get("max_agents", 8) - 1)
            ),
        ),
    )


def world(
    tmp_path,
    provider=None,
    spec=None,
    *,
    repository=None,
    initial_agent_only=False,
    **runtime_options,
):
    store = WorldStore()
    repository = repository or create_repository(tmp_path)
    # Lifecycle tests mock Docker inspection and pressure; execution is replaced per test.
    with (
        patch("mikoshi.tools.commands.shutil.which", return_value="/mock/docker"),
        patch(
            "mikoshi.tools.commands.CommandRunner._inspect_docker",
            _mock_docker_admission,
        ),
        patch(
            "mikoshi.tools.commands.CommandRunner._pressure_sample",
            return_value={"cpu_percent": 0, "memory_percent": 0},
        ),
    ):
        runtime = build_from_problem(
            store,
            provider or ScriptedProvider(),
            spec or problem(),
            tmp_path / "workspace",
            resolve_repository(repository),
            **runtime_options,
        )
    if initial_agent_only:
        # Lifecycle-focused fixtures can isolate one activation; ordinary startup
        # always wakes the complete peer roster.
        with store._connection:
            store._connection.execute(
                "UPDATE triggers SET status='completed' WHERE id != (SELECT MIN(id) FROM triggers)"
            )
    return store, runtime


class ScriptedProvider:
    """Return authored actions in order, then yield until the runtime settles."""

    def __init__(self, decisions=()):
        self.requests = []
        self.decisions = iter(decisions)

    def generate_batch(self, requests):
        self.requests.extend(requests)
        return [next(self.decisions, {"action": "peer_yield"}) for request in requests]


def swarm_problem(*, population=3):
    return replace(
        problem(max_agents=population),
        team=TeamSpec(),
    )


def _mock_docker_admission(runner):
    runner.image_id = "sha256:" + "a" * 64
    runner.image_platform = "linux/amd64"
