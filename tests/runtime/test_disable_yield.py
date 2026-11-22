"""Disabled voluntary completion keeps working until ordinary resource controls stop it."""

import json

import pytest
from support.organization import swarm_problem, world

from mikoshi.configuration import configuration_document, parse_configuration, run_parser
from mikoshi.contracts import action_schemas
from mikoshi.runtime.budgets import ExecutionBudget


def test_disabled_yield_rejects_attempt_and_reaches_wall_limit(tmp_path):
    budget = ExecutionBudget(tmp_path / "budget.json", seconds=60)

    class Provider:
        def __init__(self):
            self.budget = budget
            self.requests = []

        def generate_batch(self, requests):
            self.requests.extend(requests)
            if len(self.requests) == 1:
                return [{"action": "peer_yield"}]
            if len(self.requests) == 2:
                return [
                    {"action": "write_file", "path": "continued.txt", "content": "Still working"}
                ]
            budget.elapsed_before = 61
            return [{"action": "check_budget"}]

    provider = Provider()
    store, runtime = world(
        tmp_path,
        provider,
        swarm_problem(population=1),
        disable_yield=True,
        enable_commands=False,
    )
    try:
        runtime.run()
        assert runtime.stop_reason == "wall_time_exhausted"
        assert (runtime.workspace.canonical_repository / "continued.txt").read_text(
            encoding="utf-8"
        ) == "Still working"
        assert len(provider.requests) == 3
        for request in provider.requests:
            assert "peer_yield" not in action_schemas(request.schema)
            assert "Voluntary yielding is disabled" in request.instructions
            assert "Yield when you" not in request.instructions
        assert "unknown tool 'peer_yield'" in str(provider.requests[1].history)
        assert not store._connection.execute(
            "SELECT 1 FROM events WHERE action_type='peer_yield'"
        ).fetchone()
        with pytest.raises(ValueError, match="disabled"):
            runtime._apply_workspace_decision(None, {"action": "peer_yield"})
    finally:
        store.close()


@pytest.mark.parametrize("setting", [False, True])
def test_setting_roundtrips_config_and_cli(tmp_path, setting):
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "kind": "input",
                "problem": {
                    "objective": {"id": "test", "title": "Finish", "description": "Finish"}
                },
                "run": {
                    "repository": str(tmp_path),
                    "model": "synthetic",
                    "disable_yield": setting,
                },
                "pricing": {
                    "input_rate": 1,
                    "cache_read_rate": 1,
                    "output_rate": 1,
                    "thinking_rate": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    args = parse_configuration(run_parser(), ["--config", str(path)])
    assert args.disable_yield is setting
    assert configuration_document(args)["run"]["disable_yield"] is setting
    override = parse_configuration(run_parser(), ["--config", str(path), "--disable-yield"])
    assert override.disable_yield is True
    value = json.loads(path.read_text(encoding="utf-8"))
    value["run"]["disable_yield"] = "false"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_configuration(run_parser(), ["--config", str(path)])
