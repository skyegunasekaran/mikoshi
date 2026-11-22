"""Provider shutdown must settle late results before terminal run records are frozen."""

import json
import sys
from types import SimpleNamespace

import pytest
from support.organization import ScriptedProvider

from mikoshi import application, cli
from mikoshi.runtime.run_lifecycle import RunLifecycle
from mikoshi.schema import read_document, validate_bundle


def test_windows_saved_run_directory_uses_local_app_data_and_honors_override(tmp_path, monkeypatch):
    monkeypatch.setattr(application, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    local = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    assert application.state_directory() == local / "mikoshi"
    assert not local.exists()
    override = tmp_path / "custom-state"
    monkeypatch.setenv("XDG_STATE_HOME", str(override))
    assert application.state_directory() == override / "mikoshi"
    assert not override.exists()


class LateSettlingProvider(ScriptedProvider):
    """A scripted controller whose shutdown collects already-paid result telemetry."""

    def __init__(self, args, progress, *, fail_close=False):
        super().__init__()
        self.budget = args.execution_budget
        self.concurrency = 1
        self.max_tokens = 4096
        self.context_window_tokens = 128000
        self.progress = progress
        self.output = args.output
        self.close_count = 0
        self.input_tokens = 640
        self.output_tokens = 0
        self.fail_close = fail_close
        self.close_observations = []
        # Pending entries model results which are only collected by close().
        self.ledger_ids = self.budget.begin_requests(
            [{"model": "scripted", "max_tokens": 4096}] * 4
        )
        self.budget.settle([self.ledger_ids[0]], [self.report(640, 0)])

    @staticmethod
    def report(input_tokens, output_tokens):
        return SimpleNamespace(
            json=lambda: {
                "usage": {
                    "input_tokens": input_tokens,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": output_tokens,
                    "output_tokens_details": {"reasoning_tokens": 0},
                }
            }
        )

    @property
    def metrics(self):
        return {
            "requests": 4,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    def close(self):
        self.close_count += 1
        current = read_document(self.output / "run.json", "run")
        self.close_observations.append(
            {
                "status": current["status"],
                "phase": current["phase"],
                "candidate_exists": (self.output / "candidate").exists(),
                "evaluations_exist": bool(list((self.output / "evaluations").glob("*.json"))),
            }
        )
        self.budget.settle(
            self.ledger_ids[1:],
            [self.report(134267, 182), RuntimeError("interrupted"), RuntimeError("interrupted")],
        )
        self.input_tokens += 134267
        self.output_tokens += 182
        self.progress(
            {
                "kind": "inference_completed",
                "input_tokens": 134267,
                "output_tokens": 182,
            }
        )
        if self.fail_close:
            raise RuntimeError("provider close failed")


def launch(tmp_path, monkeypatch, *, stop_reason=None, failure=None):
    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "run"
    config = tmp_path / "experiment.json"
    config.write_text(
        json.dumps(
            {
                "kind": "input",
                "problem": {
                    "objective": {"id": "finish", "title": "Finish", "description": "Finish"},
                    "limits": {"max_agents": 1},
                },
                "run": {
                    "repository": str(repository),
                    "output": str(output),
                    "provider": "openai",
                    "model": "scripted",
                    "max_output_tokens": 4096,
                    "context_window_tokens": 128000,
                    "wall_time_seconds": 60,
                    "budget_usd": 1,
                },
                "pricing": {
                    "input_rate": 1,
                    "cache_read_rate": 1,
                    "output_rate": 1,
                    "thinking_rate": 1,
                },
                "deployment": {"enable_commands": False},
            }
        ),
        encoding="utf-8",
    )
    providers = []

    def build_provider(args, progress_callback):
        provider = LateSettlingProvider(args, progress_callback, fail_close=failure == "close")
        providers.append(provider)
        return provider

    original_build = application.build_from_problem

    def build_runtime(*args, **kwargs):
        if failure == "runtime_setup":
            raise RuntimeError("runtime setup failed")
        runtime = original_build(*args, **kwargs)
        if stop_reason is not None:
            original_run = runtime.run

            def run(**options):
                summary = original_run(**options)
                runtime.stop_reason = stop_reason
                return summary

            runtime.run = run
        return runtime

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    monkeypatch.setattr(application, "build_decision_provider", build_provider)
    monkeypatch.setattr(application, "build_from_problem", build_runtime)
    monkeypatch.setattr(sys, "argv", ["mikoshi", "--config", str(config)])
    if failure == "export":
        monkeypatch.setattr(
            application,
            "export_state",
            lambda _: (_ for _ in ()).throw(RuntimeError("export failed")),
        )
    return output, providers


@pytest.mark.parametrize("stop_reason", [None, "command_cleanup_unconfirmed"])
def test_close_settles_late_usage_before_final_index_and_candidate(
    tmp_path, monkeypatch, capsys, stop_reason
):
    output, providers = launch(tmp_path, monkeypatch, stop_reason=stop_reason)
    if stop_reason is None:
        cli.main()
    else:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 1
    capsys.readouterr()
    provider = providers[0]
    run = read_document(output / "run.json", "run")
    page_history = json.loads((output / "workspace/pages/history.json").read_text(encoding="utf-8"))
    assert "coordination" in {row["id"] for row in page_history["shared_pages"]}
    assert "organization" not in {row["id"] for row in page_history["shared_pages"]}
    records = [
        json.loads(line)
        for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        record["type"] == "action" and record["data"]["action_type"] == "peer_yield"
        for record in records
    )
    assert provider.close_count == 1
    assert run["status"] == ("completed" if stop_reason is None else "failed")
    assert run["stop_reason"] == (stop_reason or "all_agents_yielded")
    assert run["finished_at"] is not None
    assert run["provider_metrics"] == provider.metrics
    assert run["budget"]["pending_requests"] == 0
    assert run["budget"]["unknown_usage_requests"] == 2
    assert run["budget"]["accounted_usd"] == pytest.approx((640 + 134267 + 182) / 1e6)
    assert run["evidence"]["record_count"] == len(records)
    assert records[-1]["data"]["kind"] == "run_finished"
    assert len(provider.close_observations) == 1
    assert provider.close_observations[0]["status"] == "running"
    assert provider.close_observations[0]["phase"] not in {
        "exporting",
        "freezing",
        "scoring",
        "finished",
    }
    assert provider.close_observations[0]["candidate_exists"] is False
    assert provider.close_observations[0]["evaluations_exist"] is False
    assert validate_bundle(output)["status"] == "valid"


@pytest.mark.parametrize("failure", ["runtime_setup", "export", "close"])
def test_failure_paths_close_once_and_keep_late_accounting_terminal(
    tmp_path, monkeypatch, capsys, failure
):
    output, providers = launch(tmp_path, monkeypatch, failure=failure)
    with pytest.raises((RuntimeError, SystemExit)):
        cli.main()
    capsys.readouterr()
    provider = providers[0]
    run = read_document(output / "run.json", "run")
    assert provider.close_count == 1
    assert run["status"] == "failed"
    assert run["finished_at"] is not None
    assert run["budget"]["pending_requests"] == 0
    assert run["budget"]["unknown_usage_requests"] == 2
    assert not (output / "candidate").exists()


@pytest.mark.parametrize("failed", [False, True])
def test_terminal_lifecycle_ignores_late_progress(tmp_path, failed):
    lifecycle = RunLifecycle(tmp_path, "2026-10-30T00:00:00+00:00")
    if failed:
        lifecycle.fail(RuntimeError("earlier failure"))
    else:
        lifecycle.complete("all_agents_yielded", "not_configured")
    index = (tmp_path / "run.json").read_bytes()
    records = (tmp_path / "records.jsonl").read_bytes()
    lifecycle.progress({"kind": "inference_completed", "output_tokens": 182})
    assert (tmp_path / "run.json").read_bytes() == index
    assert (tmp_path / "records.jsonl").read_bytes() == records


def test_forced_interrupt_restores_signal_handlers_and_settles_usage(tmp_path, monkeypatch, capsys):
    import signal

    from mikoshi.runtime.control import stop_signals

    output, providers = launch(tmp_path, monkeypatch)
    previous = {signum: signal.getsignal(signum) for signum in stop_signals()}

    def interrupt(*args):
        raise KeyboardInterrupt("forced interruption during export")

    monkeypatch.setattr(application, "freeze_candidate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cli.main()
    capsys.readouterr()
    run = read_document(output / "run.json", "run")
    assert run["status"] == run["execution_status"] == "interrupted"
    assert run["budget"]["pending_requests"] == 0
    assert providers[0].close_count == 1
    assert {signum: signal.getsignal(signum) for signum in previous} == previous
