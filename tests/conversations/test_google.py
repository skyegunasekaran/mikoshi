"""Native Gemini replay, accounting and compaction without paid API calls."""

import copy
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from support.boundaries import run_for
from support.organization import problem, world

from mikoshi import cli
from mikoshi.contracts import action_schema
from mikoshi.decisions import DecisionRequest, ModelDecisionError, ProviderTransportError
from mikoshi.providers.google import GoogleDecisionProvider
from mikoshi.providers.limits import resolve_model_limits
from mikoshi.providers.registry import provider_adapter
from mikoshi.runtime.budgets import BudgetExhausted, ContextWindowExhausted, ExecutionBudget
from mikoshi.runtime.conversations import conversation_exchange
from mikoshi.schema import iter_records, validate_bundle


def request(**kwargs):
    return DecisionRequest(
        "test",
        "Current instructions",
        "Full current state",
        {"oneOf": [action_schema("list_activity", {}, required=[])]},
        conversation_id="peer",
        **kwargs,
    )


def response(parts=None, *, usage=None, reason="STOP"):
    return {
        "responseId": "response-1",
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": parts
                    if parts is not None
                    else [{"functionCall": {"name": "list_activity", "args": {}}}],
                },
                "finishReason": reason,
            }
        ],
        "usageMetadata": usage
        if usage is not None
        else {
            "promptTokenCount": 100,
            "cachedContentTokenCount": 40,
            "candidatesTokenCount": 10,
            "thoughtsTokenCount": 20,
            "totalTokenCount": 130,
        },
    }


def ledger(tmp_path):
    return ExecutionBudget(
        tmp_path / "budget.json",
        1,
        input_rate=1,
        cache_read_rate=0.1,
        output_rate=2,
        thinking_rate=3,
    )


def provider(handler, **kwargs):
    return GoogleDecisionProvider(
        "models/gemini-test",
        api_key="synthetic-google-secret",
        http_transport=httpx.MockTransport(handler),
        **kwargs,
    )


def test_offline_google_configuration_checks_model_ids_without_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    options = SimpleNamespace(
        model="models/gemini-test", provider_reasoning_effort=None, temperature=None, top_p=None
    )
    GoogleDecisionProvider.validate_configuration(options)
    options.model = "models/invalid/model"
    with pytest.raises(ValueError, match="Google model must be a model ID"):
        GoogleDecisionProvider.validate_configuration(options)
    options.model = "gemini-test"
    with pytest.raises(ValueError, match="GOOGLE_API_KEY or GEMINI_API_KEY"):
        GoogleDecisionProvider.validate_settings(options)


@pytest.mark.parametrize("native_ids", [False, True])
def test_native_replay_preserves_signatures_parallel_calls_and_validation_feedback(
    tmp_path, native_ids
):
    wires, native_contents, events = [], [], []

    def respond(wire):
        payload = json.loads(wire.content)
        wires.append(payload)
        assert wire.url.path == "/v1beta/models/gemini-test:generateContent"
        assert not wire.url.query
        assert wire.headers["x-goog-api-key"] == "synthetic-google-secret"
        assert "authorization" not in wire.headers
        assert payload["generationConfig"]["maxOutputTokens"] == 4096
        declarations = payload["tools"][0]["functionDeclarations"]
        assert all("parametersJsonSchema" in declaration for declaration in declarations)
        assert payload["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
        turn = len(wires)
        if turn == 1:
            calls = [
                ("write_file", {"path": "must-not-run.txt", "content": "invalid batch"}),
                ("read_file", {}),
            ]
        elif turn == 2:
            assert payload["contents"][1] == native_contents[0]
            replies = [
                p["functionResponse"]
                for p in payload["contents"][-1]["parts"]
                if "functionResponse" in p
            ]
            assert [reply["name"] for reply in replies] == ["write_file", "read_file"]
            assert all("path is required" in reply["response"]["output"] for reply in replies)
            assert all(("id" in reply) == native_ids for reply in replies)
            if native_ids:
                assert [reply["id"] for reply in replies] == ["call-1-0", "call-1-1"]
            calls = [("write_file", {"path": "corrected.txt", "content": "recovered"})]
        else:
            assert turn == 3
            assert payload["contents"][1] == native_contents[0]
            assert payload["contents"][3] == native_contents[1]
            calls = [("peer_yield", {})]
        parts = [
            {"text": "A thought summary", "thought": True, "thoughtSignature": "text-signature"},
            *[
                {
                    "functionCall": {
                        "name": name,
                        "args": arguments,
                        **({"id": f"call-{turn}-{index}"} if native_ids else {}),
                    },
                    **({"thoughtSignature": "opaque-function-signature"} if index == 0 else {}),
                }
                for index, (name, arguments) in enumerate(calls)
            ],
        ]
        body = response(parts)
        native_contents.append(copy.deepcopy(body["candidates"][0]["content"]))
        return httpx.Response(200, json=body)

    remote = provider(respond, max_tokens=4096)
    remote.set_evidence_callback(events.append)
    store, runtime = world(tmp_path, remote, problem(max_agents=1), enable_commands=False)
    try:
        run_for(runtime)
        assert runtime.stop_reason == "all_agents_yielded"
        target = runtime.workspace.canonical_repository
        assert not (target / "must-not-run.txt").exists()
        assert (target / "corrected.txt").read_text(encoding="utf-8") == "recovered"
        assert len(wires) == 3
        assert not remote.metrics["evidence_errors"]
        submitted = [e for e in events if e["kind"] == "provider_request_submitted"]
        assert [e["payload"] for e in submitted] == wires
        paths = submitted[1]["tool_result_paths"]
        assert len(paths) == 2
        for link in paths:
            value = wires[1]
            for key in link["path"].split("/")[1:]:
                value = value[int(key)] if isinstance(value, list) else value[key]
            assert "path is required" in value
        assert "synthetic-google-secret" not in json.dumps(events)
    finally:
        remote.close()
        store.close()


def test_usage_and_generation_settings_reach_the_shared_ledger(tmp_path):
    wires = []

    def respond(wire):
        wires.append(json.loads(wire.content))
        return httpx.Response(200, json=response())

    budget = ledger(tmp_path)
    with provider(
        respond, budget=budget, reasoning_effort="low", temperature=0.4, top_p=0.9
    ) as remote:
        result = remote.generate_batch([request(max_tokens=123)])[0]
        assert result == {"action": "list_activity"}
        assert wires[0]["generationConfig"] == {
            "candidateCount": 1,
            "maxOutputTokens": 123,
            "thinkingConfig": {"thinkingLevel": "LOW"},
            "temperature": 0.4,
            "topP": 0.9,
        }
        entry = budget.entries[0]
        assert entry["status"] == "reported"
        assert entry["uncached_input_tokens"] == 60
        assert entry["cache_read_tokens"] == 40
        assert entry["output_tokens"] == 10
        assert entry["thinking_tokens"] == 20
        assert entry["charged_usd"] == pytest.approx((60 + 4 + 20 + 60) / 1e6)
        metrics = remote.metrics
        assert metrics["output_tokens"] == 30
        assert metrics["reasoning_tokens"] == 20
        assert metrics["final_output_tokens"] == 10
        assert metrics["timing_events"][0]["max_tokens"] == [123]


@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"promptTokenCount": 10},
        {"promptTokenCount": 10, "candidatesTokenCount": -1},
        {"promptTokenCount": 10, "candidatesTokenCount": 1, "thoughtsTokenCount": True},
        {"promptTokenCount": 10, "candidatesTokenCount": 1, "cachedContentTokenCount": 11},
        {"promptTokenCount": 10, "candidatesTokenCount": 1, "totalTokenCount": 99},
    ],
)
def test_missing_or_invalid_usage_remains_unknown_without_discarding_actions(tmp_path, usage):
    budget = ledger(tmp_path)
    with provider(
        lambda _: httpx.Response(200, json=response(usage=usage)), budget=budget
    ) as remote:
        assert remote.generate_batch([request()])[0] == {"action": "list_activity"}
    assert budget.entries[0]["status"] == "unknown_usage"
    assert budget.charged == 0


def test_blocked_response_still_accounts_for_prompt_tokens(tmp_path):
    body = {
        "promptFeedback": {"blockReason": "SAFETY"},
        "usageMetadata": {"promptTokenCount": 100, "totalTokenCount": 100},
    }
    budget = ledger(tmp_path)
    with provider(lambda _: httpx.Response(200, json=body), budget=budget) as remote:
        result = remote.generate_batch([request()])[0]
        assert isinstance(result, ProviderTransportError)
        assert "SAFETY" in str(result)
        assert not remote._sessions
    assert budget.entries[0]["status"] == "reported"
    assert budget.entries[0]["completion_tokens"] == 0
    assert budget.charged == pytest.approx(100 / 1e6)


@pytest.mark.parametrize("reason", ["MAX_TOKENS", "SAFETY", "MALFORMED_FUNCTION_CALL"])
def test_unfinished_or_blocked_calls_are_not_executable(reason):
    with provider(lambda _: httpx.Response(200, json=response(reason=reason))) as remote:
        result = remote.generate_batch([request()])[0]
        assert isinstance(result, ModelDecisionError)
        assert result.assistant_message["google_content"]


def test_auto_mode_text_response_gets_feedback_and_can_recover():
    wires = []

    def respond(wire):
        payload = json.loads(wire.content)
        wires.append(payload)
        assert payload["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
        if len(wires) == 1:
            return httpx.Response(200, json=response([{"text": "I will inspect the activity."}]))
        assert "Each turn needs at least one tool call" in json.dumps(payload["contents"])
        return httpx.Response(200, json=response())

    with provider(respond) as remote:
        first_request = request()
        first = remote.generate_batch([first_request])[0]
        assert isinstance(first, ModelDecisionError)
        history = conversation_exchange(first_request, first, {"error": first.feedback})
        recovered = remote.generate_batch([request(history=history)])[0]
        assert recovered == {"action": "list_activity"}


def test_compaction_restores_state_and_results_and_accounts_for_both_requests(tmp_path):
    wires = []
    budget = ledger(tmp_path)

    def respond(wire):
        payload = json.loads(wire.content)
        wires.append(payload)
        if len(wires) == 1:
            return httpx.Response(
                200,
                json=response(
                    [
                        {"text": "Old context. " * 300},
                        {"functionCall": {"name": "list_activity", "args": {}}},
                    ]
                ),
            )
        if len(wires) == 2:
            assert "tools" not in payload and "toolConfig" not in payload
            return httpx.Response(200, json=response([{"text": "Compacted work summary"}]))
        assert len(wires) == 3
        assert "tools" in payload
        assert len(payload["contents"]) == 1
        assert payload["systemInstruction"]["parts"][0]["text"].startswith("Updated instructions")
        restored = json.dumps(payload["contents"])
        assert "Compacted work summary" in restored
        assert "pending result receipt" in restored
        assert "Full authoritative state" in restored
        return httpx.Response(200, json=response())

    with provider(respond, budget=budget, compaction_input_tokens=2000) as remote:
        first_request = request()
        first = remote.generate_batch([first_request])[0]
        history = conversation_exchange(first_request, first, {"message": "pending result receipt"})
        second_request = replace(
            request(
                history=history,
                model_input_text="state delta",
                compaction_input_text="Full authoritative state",
            ),
            instructions="Updated instructions",
        )
        result = remote.generate_batch([second_request])[0]
        assert result == {"action": "list_activity"}
        assert len(result.request_ids) == 2
        checkpoint = result.assistant_message["compaction_checkpoints"][0]
        assert checkpoint["assistant_message"]["content"] == "Compacted work summary"
        assert remote.metrics["compactions"] == 1
        assert len(budget.entries) == 3
        assert all(entry["status"] == "reported" for entry in budget.entries)


def test_budget_can_stop_between_summary_and_action_request(tmp_path):
    wires = []
    budget = ledger(tmp_path)

    def respond(wire):
        payload = json.loads(wire.content)
        wires.append(payload)
        return httpx.Response(
            200, json=response([{"text": "Summary"}] if "tools" not in payload else None)
        )

    with provider(respond, budget=budget, compaction_input_tokens=1) as remote:
        first_request = request()
        first = remote.generate_batch([first_request])[0]
        budget.limit = budget.charged * 1.5
        history = conversation_exchange(first_request, first, "receipt")
        with pytest.raises(BudgetExhausted, match="budget_exhausted"):
            remote.generate_batch([request(history=history)])
        assert len(wires) == 2
        assert remote.metrics["compactions"] == 1
        assert remote._compaction_pending


def test_bad_compaction_summary_does_not_advance_conversation():
    wires = []

    def respond(wire):
        payload = json.loads(wire.content)
        wires.append(payload)
        return httpx.Response(
            200,
            json=response(
                [{"text": "Truncated summary"}] if "tools" not in payload else None,
                reason="MAX_TOKENS" if "tools" not in payload else "STOP",
            ),
        )

    with provider(respond, compaction_input_tokens=1) as remote:
        first_request = request()
        first = remote.generate_batch([first_request])[0]
        before = copy.deepcopy(remote._sessions)
        history = conversation_exchange(first_request, first, "receipt")
        result = remote.generate_batch([request(history=history)])[0]
        assert isinstance(result, ProviderTransportError)
        assert remote._sessions == before
        assert remote.metrics["compactions"] == 0


def test_http_errors_redact_credentials_and_retain_retry_after():
    events = []
    with provider(
        lambda _: httpx.Response(
            429, json={"error": "synthetic-google-secret"}, headers={"Retry-After": "3"}
        )
    ) as remote:
        remote.set_evidence_callback(events.append)
        result = remote.generate_batch([request()])[0]
        assert isinstance(result, ProviderTransportError)
        assert result.status_code == 429 and result.retry_after_seconds == 3
        assert "synthetic-google-secret" not in str(result)
        assert "synthetic-google-secret" not in json.dumps(events)


def test_input_limit_and_stale_continuation_prevent_dispatch():
    wires = []
    with provider(lambda wire: wires.append(wire), input_token_limit=10) as remote:
        result = remote.generate_batch([request()])[0]
        assert isinstance(result, ContextWindowExhausted)
        assert not wires
    with provider(lambda _: httpx.Response(200, json=response())) as remote:
        first = remote.generate_batch([request()])[0]
        history = conversation_exchange(request(), first, "receipt")
        history[1]["google_turn"] = "stale"
        result = remote.generate_batch([request(history=history)])[0]
        assert isinstance(result, ValueError) and "last native turn" in str(result)


@pytest.mark.parametrize(
    "model", ["../secret", "https://example.org/model", "a?key=secret", "models/", "a/b"]
)
def test_model_name_cannot_change_endpoint(model):
    with pytest.raises(ValueError, match="model ID"):
        GoogleDecisionProvider(model, api_key="synthetic")


@pytest.mark.parametrize("compact", [False, True])
def test_google_configuration_flows_through_preflight_and_mocked_cli_run(
    tmp_path, monkeypatch, capsys, compact
):
    assert provider_adapter("google") is GoogleDecisionProvider
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    options = SimpleNamespace(
        model="gemini-test", provider_reasoning_effort=None, temperature=None, top_p=None
    )
    with pytest.raises(ValueError, match="GOOGLE_API_KEY or GEMINI_API_KEY"):
        GoogleDecisionProvider.validate_settings(options)
    monkeypatch.setenv("GEMINI_API_KEY", "alias-secret")
    GoogleDecisionProvider.validate_settings(options)
    with GoogleDecisionProvider("gemini-test") as remote:
        assert remote.request_headers() == {"x-goog-api-key": "alias-secret"}
    monkeypatch.setenv("GOOGLE_API_KEY", "primary-secret")
    with GoogleDecisionProvider("gemini-test") as remote:
        assert remote.request_headers() == {"x-goog-api-key": "primary-secret"}
    with GoogleDecisionProvider("gemini-test", api_key="explicit-secret") as remote:
        assert remote.request_headers() == {"x-goog-api-key": "explicit-secret"}
    limits = resolve_model_limits("google", "gemini-test")
    assert limits.configured_context_window == 1_000_000
    assert limits.actor_output_ceiling == 16384
    repository = tmp_path / "repository"
    repository.mkdir()
    output_directory = tmp_path / "run"
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "kind": "input",
                "run": {
                    "repository": str(repository),
                    "output": str(output_directory),
                    "provider": "google",
                    "model": "gemini-test",
                    "budget_usd": 1,
                    **({"compaction_input_tokens": 1} if compact else {}),
                },
                "problem": {
                    "objective": {"id": "test", "title": "Test", "description": "Test"},
                    "limits": {"max_agents": 1},
                },
                "deployment": {"enable_commands": False},
                "pricing": {
                    "input_rate": 1,
                    "cache_read_rate": 0.1,
                    "output_rate": 2,
                    "thinking_rate": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["mikoshi", "check", "--config", str(path)])
    cli.main()
    output = capsys.readouterr().out
    checked = json.loads(output)
    assert checked["status"] == "checked" and checked["run"]["provider"] == "google"
    assert "secret" not in output

    generations = 0

    def respond(wire):
        nonlocal generations
        payload = json.loads(wire.content)
        if "tools" not in payload:
            return httpx.Response(200, json=response([{"text": "Summary"}]))
        generations += 1
        action = "list_activity" if generations < 3 else "peer_yield"
        assert generations <= 3
        return httpx.Response(200, json=response([{"functionCall": {"name": action, "args": {}}}]))

    original = GoogleDecisionProvider.__init__

    def construct(self, *args, **kwargs):
        original(self, *args, **kwargs, http_transport=httpx.MockTransport(respond))

    monkeypatch.setattr(GoogleDecisionProvider, "__init__", construct)
    monkeypatch.setattr(sys, "argv", ["mikoshi", "--config", str(path)])
    cli.main()
    run = json.loads(capsys.readouterr().out)
    assert run["stop_reason"] == "all_agents_yielded"
    assert run["provenance"]["provider_endpoint"].endswith("models/gemini-test:generateContent")
    assert run["provider_metrics"]["compactions"] == (2 if compact else 0)
    assert run["budget"]["accounted_usd"] == pytest.approx((5 if compact else 3) * 124 / 1e6)
    assert validate_bundle(output_directory)["status"] == "valid"
    assert "secret" not in json.dumps(run)
    submitted = [
        record["data"]
        for record in iter_records(output_directory, record_type="request")
        if record["data"]["kind"] == "provider_request_submitted"
    ]
    assert any(item["kind"] == "tool_result" for item in submitted[1]["included_content"])
