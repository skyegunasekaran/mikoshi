"""Native repetitions preserve inputs, isolate work and retain every trial outcome."""

import asyncio
import json

from mikoshi.configuration import parse_configuration, run_parser
from mikoshi.experiments import run_experiment
from mikoshi.persistence.exports import write_json


def experiment(tmp_path, *, trials=3, concurrency=2):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "uncommitted.txt").write_text("same starting state", encoding="utf-8")
    config = tmp_path / "config.json"
    write_json(
        config,
        {
            "kind": "input",
            "problem": {
                "objective": {
                    "id": "test",
                    "title": "Objective",
                    "description": "an arbitrary objective",
                }
            },
            "run": {
                "repository": str(repository),
                "output": str(tmp_path / "experiment"),
                "provider": "openai",
                "model": "synthetic-model",
                "budget_usd": 2,
                "wall_time_seconds": 10,
                "trials": trials,
                "trial_concurrency": concurrency,
            },
            "pricing": {
                "input_rate": 1,
                "cache_read_rate": 1,
                "output_rate": 1,
                "thinking_rate": 1,
            },
            "deployment": {"enable_commands": False},
        },
    )
    return parse_configuration(run_parser(), ["--config", str(config)])


def test_repeated_trials_isolate_inputs_and_preserve_failures(tmp_path, monkeypatch):
    args = experiment(tmp_path)
    active = peak = 0

    async def child(config_path, run_dir, emit, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        repository = config_path.parent / "repository"
        assert (repository / "uncommitted.txt").read_text(encoding="utf-8") == "same starting state"
        assert config["run"]["trials"] == config["run"]["trial_concurrency"] == 1
        assert config["run"]["budget_usd"] == 2
        (repository / "uncommitted.txt").write_text(run_dir.parent.name, encoding="utf-8")
        await asyncio.sleep(0.01)
        active -= 1
        # One trial reaches its time limit; another exits without terminal status.
        status = "completed" if run_dir.parent.name == "trial-001" else "stopped"
        if run_dir.parent.name == "trial-003":
            status = "running"
        budget = {"accounted_usd": 0.5, "pending_requests": 0, "unknown_usage_requests": 0}
        write_json(
            run_dir / "run.json",
            {
                "status": status,
                "stop_reason": "wall_time_exhausted",
                "budget": budget,
                "finished_at": "now",
                "evaluation": {"status": "completed", "scores": {"quality": {"value": 0}}},
            },
        )
        emit({"progress": {"kind": "trial_finished"}})
        return 0

    monkeypatch.setattr("mikoshi.experiments.run_process", child)
    assert run_experiment(args)
    results = json.loads((args.output / "run.json").read_text(encoding="utf-8"))
    assert peak == args.trial_concurrency
    assert results["status"] == "failed"
    assert [row["status"] for row in results["trials"]] == ["completed", "stopped", "failed"]
    assert [row["run_dir"] for row in results["trials"]] == [
        "trials/trial-001/run",
        "trials/trial-002/run",
        "trials/trial-003/run",
    ]
    assert results["trials"][0]["supervisor"]["path"] == "trials/trial-001/supervisor.json"
    assert results["budget"]["accounted_usd"] == 1.5
    assert results["budget"]["limit_usd"] == 6
    assert results["score_summary"] == {"quality": {"count": 2, "mean": 0, "missing": 1}}
    assert (args.repository / "uncommitted.txt").read_text(
        encoding="utf-8"
    ) == "same starting state"
    assert (args.output / "inputs/repository/uncommitted.txt").read_text(
        encoding="utf-8"
    ) == "same starting state"


def test_stop_prevents_queued_trials_from_launching(tmp_path, monkeypatch):
    args = experiment(tmp_path, concurrency=1)
    launches = []

    async def child(config_path, run_dir, emit, **kwargs):
        launches.append(run_dir)
        (args.output / "STOP").touch()
        assert kwargs["stop_requested"]()
        write_json(
            run_dir / "run.json",
            {"status": "stopped", "stop_reason": "stop_requested", "finished_at": "now"},
        )
        return 0

    monkeypatch.setattr("mikoshi.experiments.run_process", child)
    assert not run_experiment(args)
    result = json.loads((args.output / "run.json").read_text(encoding="utf-8"))
    assert len(launches) == 1
    assert result["status"] == "stopped"
    assert [row["status"] for row in result["trials"]] == ["stopped", "not_started", "not_started"]
