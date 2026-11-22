"""One experiment configuration flows through setup, launch, exports and status."""

import json
import shutil
import sys

import pytest
from support.organization import ScriptedProvider

from mikoshi import application, cli
from mikoshi.persistence.artifacts import load_verified_artifacts
from mikoshi.schema import read_content, validate_bundle


def test_experiment_configuration_checks_launches_and_reports_resolved_settings(
    tmp_path, monkeypatch, capsys
):
    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "run"
    path = tmp_path / "experiment.json"
    (tmp_path / "grader.py").write_text(
        "from pathlib import Path\n"
        "def grade(context):\n"
        "    assert context['run']['stop_reason'] == 'all_agents_yielded'\n"
        "    return {'quality': (Path(context['candidate']) / 'evidence.md').read_text() == 'Recorded'}\n",
        encoding="utf-8",
    )
    path.write_text(
        json.dumps(
            {
                "kind": "input",
                "problem": {
                    "objective": {
                        "id": "test",
                        "title": "Record",
                        "description": "Record evidence",
                    },
                    "limits": {"max_agents": 2},
                },
                "run": {
                    "repository": "repository",
                    "output": "run",
                    "provider": "openai",
                    "model": "test-model",
                    "max_output_tokens": 8192,
                    "context_window_tokens": 128000,
                    "compaction_input_tokens": 96000,
                    "provider_concurrency": 2,
                    "wall_time_seconds": 600,
                    "budget_usd": 1,
                    "temperature": 0.2,
                    "top_p": 0.95,
                },
                "pricing": {
                    "input_rate": 1,
                    "cache_read_rate": 0.1,
                    "output_rate": 2,
                    "thinking_rate": 2,
                },
                "deployment": {
                    "enable_commands": False,
                    "provider": {"concurrency": 4},
                    "command_limits": {
                        "default_timeout_seconds": 3600,
                        "max_timeout_seconds": 7200,
                    },
                },
                "evaluation": {"scorer": "grader.py:grade"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-host-key")
    monkeypatch.setattr(sys, "argv", ["mikoshi", "check", "--config", str(path)])
    cli.main()
    checked = json.loads(capsys.readouterr().out)
    assert checked["model_limits"]["actor_output_ceiling"] == 8192
    assert checked["run"]["repository"] == str(repository)
    assert checked["run"]["provider_concurrency"] == 2
    assert not output.exists()
    monkeypatch.setattr(sys, "argv", ["mikoshi", "check", "--config", str(path), "--no-budget-cap"])
    cli.main()
    checked = json.loads(capsys.readouterr().out)
    assert checked["run"].get("budget_usd") is None and checked["run"]["no_budget_cap"]
    monkeypatch.setattr(
        sys, "argv", ["mikoshi", "check", "--config", str(path), "--max-output-tokens", "128000"]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and not output.exists()
    capsys.readouterr()
    construct = application.build_decision_provider
    script = ScriptedProvider(
        [{"action": "write_file", "path": "evidence.md", "content": "Recorded"}]
    )
    payloads = []

    def scripted(args, progress_callback):
        provider = construct(args, progress_callback)
        assert provider.concurrency == 1
        assert (
            provider.context_window_tokens == 128000 and provider.compaction_input_tokens == 96000
        )
        provider.submit_request = None

        def generate(requests):
            if not payloads:
                request = requests[0]
                assert request.max_tokens == 4096
                payloads.append(provider.request_payload(request, request.max_tokens, 0))
            return script.generate_batch(requests)

        provider.generate_batch = generate
        return provider

    monkeypatch.setattr(application, "build_decision_provider", scripted)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mikoshi",
            "--config",
            str(path),
            "--task",
            "Record evidence",
            "--agents",
            "1",
            "--provider-concurrency",
            "1",
            "--max-output-tokens",
            "4096",
            "--wall-time-seconds",
            "60",
        ],
    )
    cli.main()
    assert payloads[0]["max_output_tokens"] == 4096
    assert payloads[0]["temperature"] == 0.2 and payloads[0]["top_p"] == 0.95
    assert (repository / "evidence.md").read_text(encoding="utf-8") == "Recorded"
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    resolved = manifest["input"]
    assert manifest["stop_reason"] == "all_agents_yielded"
    assert manifest["resolved"]["model_limits"]["actor_output_ceiling"] == 4096
    assert manifest["resolved"]["command_limits"]["default_timeout_seconds"] == 3600
    assert manifest["resolved"]["command_limits"]["max_timeout_seconds"] == 7200
    assert resolved["deployment"]["command_limits"]["default_timeout_seconds"] == 3600
    assert manifest["budget"]["wall_limit_seconds"] == 60
    assert (
        resolved["run"]["max_output_tokens"] == 4096
        and resolved["problem"]["limits"]["max_agents"] == 1
    )
    result = manifest
    assert result["status"] == "completed"
    assert result["evaluation"]["status"] == "completed"
    assert result["evaluation"]["scores"] == {"quality": {"value": True}}
    assert load_verified_artifacts(output)["candidate_sha256"]
    assert "synthetic-host-key" not in (output / "run.json").read_text(encoding="utf-8")
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert any(
        event.get("progress", {}).get("kind") == "run_started"
        and event["progress"]["output"] == str(output)
        for event in events
    )
    before = (output / "run.json").read_bytes()
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setattr(sys, "argv", ["mikoshi", "status", str(output), "--json"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    sys.argv.pop()
    cli.main()
    assert "all_agents_yielded" in capsys.readouterr().out
    assert (output / "run.json").read_bytes() == before
    monkeypatch.setattr(sys, "argv", ["mikoshi", "stop", str(output)])
    cli.main()
    assert (output / "STOP").is_file()
    capsys.readouterr()
    (repository / "evidence.md").write_text("Changed after the run", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["mikoshi", "score", str(output)])
    cli.main()
    assert json.loads(capsys.readouterr().out)["scores"] == result["evaluation"]["scores"]
    assert len(list((output / "evaluations").glob("*.json"))) == 2
    monkeypatch.setattr(sys, "argv", ["mikoshi", "watch", str(output), "--json"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "completed"

    assert validate_bundle(output)["records"] > 0
    records = [
        json.loads(line)
        for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {r["type"] for r in records} == {"progress", "action", "exchange", "state", "decision"}
    exchanges = [r for r in records if r["type"] == "exchange"]
    assert any(
        m["role"] == "assistant"
        for r in exchanges
        for m in read_content(output, r["data"]["messages_ref"])
    )
    portable = tmp_path / "portable"
    shutil.copytree(output, portable)
    shutil.rmtree(portable / ".internal")
    assert validate_bundle(portable)["records"] == len(records)
