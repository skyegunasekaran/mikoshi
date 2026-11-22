"""Heterogeneous peers retain independent continuations and share one cost ledger."""

import json
import sys
from dataclasses import asdict

import httpx
import pytest
from support.boundaries import run_for
from support.organization import world

from mikoshi import application, cli
from mikoshi.problem import ObjectiveSpec, ProblemSpec, ResourceLimits, TeamMemberSpec, TeamSpec
from mikoshi.providers.openai import OpenAIDecisionProvider
from mikoshi.providers.pricing import TokenPrices
from mikoshi.providers.routing import RoutedDecisionProvider
from mikoshi.runtime.budgets import ExecutionBudget


def test_routed_peers_keep_continuations_and_billing_separate(tmp_path):
    models = ("model-1", "model-1.1")
    rates = {
        model: TokenPrices(price, price / 10, price * 5, price * 5)
        for model, price in zip(models, (1, 3))
    }
    budget = ExecutionBudget(
        tmp_path / "budget.json",
        allow_unlimited=True,
        model_prices=rates,
        **asdict(rates[models[0]]),
    )
    calls = {model: [] for model in models}

    def respond(request):
        payload = json.loads(request.content)
        model = payload["model"]
        calls[model].append(payload)
        number = len(calls[model])
        assert request.headers["authorization"] == "Bearer synthetic"
        if number == 1:
            action, arguments = "write_file", {"path": f"{model}.txt", "content": model}
            assert "previous_response_id" not in payload
        else:
            assert payload["previous_response_id"] == f"{model}-response-1"
            action, arguments = "peer_yield", {}
        return httpx.Response(
            200,
            json={
                "id": f"{model}-response-{number}",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": f"{model}-call-{number}",
                        "name": action,
                        "arguments": json.dumps(arguments),
                    }
                ],
                "usage": {
                    "input_tokens": 1000,
                    "input_tokens_details": {"cached_tokens": 200},
                    "output_tokens": 100,
                    "output_tokens_details": {"reasoning_tokens": 30},
                },
            },
        )

    providers = {
        model: OpenAIDecisionProvider(
            model,
            api_key="synthetic",
            max_tokens=4096,
            context_window_tokens=128000,
            budget=budget,
            http_transport=httpx.MockTransport(respond),
        )
        for model in models
    }
    routed = RoutedDecisionProvider(
        providers[models[0]], providers, dict(zip(models, models)), concurrency=1
    )
    spec = ProblemSpec(
        ObjectiveSpec("fixture", "Record evidence", "Record your model name"),
        ResourceLimits(2),
        team=TeamSpec(members=tuple(TeamMemberSpec(model) for model in models)),
    )
    store, runtime = world(tmp_path, routed, spec, enable_commands=False)
    try:
        assert runtime._concurrency_limit() == 1
        run_for(runtime)
        assert runtime.stop_reason == "all_agents_yielded"
        for model in models:
            assert (runtime.workspace.canonical_repository / f"{model}.txt").read_text(
                encoding="utf-8"
            ) == model
            assert len(calls[model]) == 2
            expected = (
                2
                * (
                    800 * rates[model].input_rate
                    + 200 * rates[model].cache_read_rate
                    + 100 * rates[model].output_rate
                )
                / 1e6
            )
            assert budget.summary()["by_model"][model]["reported_usd"] == pytest.approx(expected)
            assert budget.summary()["by_agent"][model]["reported_usd"] == pytest.approx(expected)
        assert budget.summary()["pending_requests"] == 0
        assert routed.metrics["requests"] == 4
        assert budget.snapshot()["wall_limit_seconds"] is None
        assert budget.snapshot()["limit_usd"] is None
        budget.check()
    finally:
        routed.close()
        store.close()


def test_unlimited_accounting_requires_explicit_opt_in(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        ExecutionBudget(tmp_path / "budget.json")


def test_routed_configuration_launches_and_exports_without_caps(tmp_path, monkeypatch, capsys):
    (tmp_path / "repository").mkdir()
    spec = ProblemSpec(
        ObjectiveSpec("fixture", "Finish", "Finish"),
        ResourceLimits(2),
        team=TeamSpec(members=(TeamMemberSpec("model-1"), TeamMemberSpec("model-1.1"))),
    )
    prices = {"input_rate": 1, "cache_read_rate": 0.1, "output_rate": 5, "thinking_rate": 5}
    config = {
        "kind": "input",
        "problem": asdict(spec),
        "run": {
            "repository": "repository",
            "model": "model-1",
            "output": "run",
            "no_budget_cap": True,
            "no_time_limit": True,
            "max_output_tokens": 4096,
            "context_window_tokens": 128000,
        },
        "deployment": {
            "enable_commands": False,
            "provider": {"agent_models": {m.id: m.id for m in spec.team.members}},
        },
        "pricing": {**prices, "models": {m.id: prices for m in spec.team.members}},
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    construct = application.build_decision_provider

    def respond(request):
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": payload["model"],
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "done",
                        "name": "peer_yield",
                        "arguments": json.dumps({}),
                    }
                ],
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 10,
                },
            },
        )

    def mocked(args, progress_callback=None):
        provider = construct(args, progress_callback)
        if isinstance(provider, RoutedDecisionProvider):
            for child in provider.providers.values():
                child._http_transport = httpx.MockTransport(respond)
        else:
            provider._http_transport = httpx.MockTransport(respond)
        return provider

    monkeypatch.setattr(application, "build_decision_provider", mocked)
    monkeypatch.setattr(sys, "argv", ["mikoshi", "--config", str(path)])
    cli.main()
    manifest = json.loads((tmp_path / "run/run.json").read_text(encoding="utf-8"))
    resolved = manifest["input"]
    assert manifest["stop_reason"] == "all_agents_yielded"
    assert (
        manifest["input"]["deployment"]["provider"]["agent_models"]
        == config["deployment"]["provider"]["agent_models"]
    )
    assert resolved["pricing"]["models"]["model-1.1"]["output_rate"] == 5
    assert manifest["budget"]["wall_limit_seconds"] is None
    assert manifest["budget"]["limit_usd"] is None
    assert manifest["budget"]["accounted_usd"] == pytest.approx(0.0003)
    assert "synthetic" not in (tmp_path / "run/run.json").read_text(encoding="utf-8")
    capsys.readouterr()
