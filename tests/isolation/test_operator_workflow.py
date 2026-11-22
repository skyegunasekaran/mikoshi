"""An operator can prepare, review, execute and inspect an experiment without an agent."""

import json
from types import SimpleNamespace

import pytest
from support.organization import ScriptedProvider

from mikoshi import application, cli, presentation
from mikoshi.cli import main
from mikoshi.schema import validate_bundle


def initialize(tmp_path, *, interactive=False, monkeypatch=None):
    repository = tmp_path / "target checkout's $evidence 🧪"
    repository.mkdir()
    config = tmp_path / "experiment's $settings 🧪.json"
    flags = [
        "--repository",
        str(repository),
        "--objective",
        "Record café evidence 🧪",
        "--model",
        "fixture-model",
        "--agents",
        "1",
        "--budget-usd",
        "2",
        "--wall-time-seconds",
        "60",
        "--no-commands",
        "--input-rate",
        "1",
        "--cache-read-rate",
        "0.1",
        "--output-rate",
        "2",
        "--thinking-rate",
        "2",
    ]
    if interactive:
        answers = iter(
            [
                str(repository),
                "Record café evidence 🧪",
                "",
                "fixture-model",
                "1",
                "2",
                "60",
                "1",
                "0.1",
                "2",
                "2",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _: next(answers))
        flags = ["--interactive", "--no-commands"]
    main(["init", str(config), *flags])
    return repository, config


def test_init_plan_run_and_maintain_without_live_dependencies(tmp_path, monkeypatch, capsys):
    # Exercise Windows output encoding and PowerShell suggestions with pytest's
    # captured streams while keeping the host filesystem implementation intact.
    monkeypatch.setattr(cli, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(presentation, "os", SimpleNamespace(name="nt"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    main(["runs"])
    assert "No saved runs found" in capsys.readouterr().out
    assert not (tmp_path / "state").exists()
    repository, config = initialize(tmp_path)
    saved = capsys.readouterr().out
    quoted = "'" + str(config).replace("'", "''") + "'"
    assert f"Start: mikoshi run --config {quoted}" in saved
    original = config.read_bytes()
    main(["plan", "--config", str(config)])
    plan = capsys.readouterr().out
    assert "Nominal total spending limit: $2" in plan
    assert "Objective: Record café evidence 🧪" in plan
    assert "Commands: disabled" in plan
    assert "No credentials" in plan
    assert not (tmp_path / "state").exists()
    assert list(repository.iterdir()) == []
    assert validate_bundle(config)["status"] == "valid"
    guided_directory = tmp_path / "guided"
    guided_directory.mkdir()
    _, guided_config = initialize(guided_directory, interactive=True, monkeypatch=monkeypatch)
    guided, supplied = json.loads(guided_config.read_text(encoding="utf-8")), json.loads(original)
    guided["run"]["repository"] = supplied["run"]["repository"]
    assert guided == supplied
    capsys.readouterr()
    main(["plan", "--config", str(config), "--json", "--trials", "3", "--trial-concurrency", "2"])
    planned = json.loads(capsys.readouterr().out)
    assert planned["input"]["run"]["trials"] == 3
    assert planned["input"]["run"]["trial_concurrency"] == 2
    assert config.read_bytes() == original
    with pytest.raises(SystemExit) as missing_key:
        main(["run", "--config", str(config)])
    assert missing_key.value.code == 2
    assert not (tmp_path / "state").exists()
    capsys.readouterr()
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-host-key")
    monkeypatch.setattr(
        application,
        "build_decision_provider",
        lambda *args: ScriptedProvider(
            [{"action": "write_file", "path": "evidence.txt", "content": "Recorded"}]
        ),
    )
    main(["run", "--config", str(config)])
    terminal = capsys.readouterr()
    assert "Output:" in terminal.err and "Agents are working" in terminal.err
    assert "all_agents_yielded" in terminal.out
    assert "fixture-host-key" not in terminal.out + terminal.err
    assert (repository / "evidence.txt").read_text(encoding="utf-8") == "Recorded"
    main(["runs", "--json"])
    listing = json.loads(capsys.readouterr().out)
    assert listing["total"] == 1 and listing["runs"][0]["status"] == "completed"
    before = config.read_bytes()
    with pytest.raises(SystemExit) as overwrite:
        main(["init", str(config)])
    assert overwrite.value.code == 2 and config.read_bytes() == before


def test_noninteractive_setup_requires_explicit_prices_and_never_writes_partial_config(
    tmp_path, capsys
):
    repository = tmp_path / "target"
    repository.mkdir()
    config = tmp_path / "experiment.json"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "init",
                str(config),
                "--repository",
                str(repository),
                "--objective",
                "Review",
                "--model",
                "fixture",
                "--budget-usd",
                "1",
            ]
        )
    assert error.value.code == 2
    assert "--input-rate" in capsys.readouterr().err
    assert not config.exists()


def test_readable_repeated_trial_failures_keep_child_diagnostics(tmp_path, monkeypatch, capsys):
    from mikoshi.configuration import parse_configuration, run_parser
    from mikoshi.experiments import run_experiment
    from mikoshi.presentation import TerminalReporter

    _, config = initialize(tmp_path)
    capsys.readouterr()
    args = parse_configuration(
        run_parser(), ["--config", str(config), "--trials", "2", "--output", str(tmp_path / "runs")]
    )

    async def child(config_path, run_dir, emit, **kwargs):
        emit({"phase": "working", "progress": {"kind": "phase_started", "phase": "working"}})
        emit({"kind": "command_output", "command_id": 1, "output": "partial output\n"})
        emit(
            {
                "kind": "process_output",
                "stream": "stderr",
                "text": "startup diagnostic",
                "truncated": False,
            }
        )
        return 1

    monkeypatch.setattr("mikoshi.experiments.run_process", child)
    reporter = TerminalReporter()
    assert run_experiment(args, progress_callback=reporter.progress)
    output = capsys.readouterr().err
    assert "Stop: mikoshi stop" in output and str(args.output) in output
    assert "[trial-001] Agents are working" in output
    assert "[trial-001] [command 1] partial output" in output
    assert "[trial-001] stderr: startup diagnostic" in output
    assert "[trial-001] Trial failed:" in output
