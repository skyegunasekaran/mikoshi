"""Installed provider and protocol extensions compose through the normal CLI."""

import json
import sys

import httpx
import pytest
from support.organization import world
from support.provider import MockDecisionProvider, tool_calls

from mikoshi import cli
from mikoshi.contracts import BATCH_GUIDANCE, action_schema
from mikoshi.persistence.content import read_content
from mikoshi.schema import iter_records, validate_bundle
from mikoshi.team_protocol import TeamProtocol, TeamTool


class FixtureProtocol(TeamProtocol):
    def tools(self, available, *, peer):
        def identify(state, decision):
            return False, {"message": f"received by {state.trigger.agent_id}"}

        return [
            TeamTool(action_schema("identify"), identify),
            *(tool for tool in available if tool.name != "write_file"),
        ]

    def environment(self, *, disable_yield):
        return "Use the identify tool to identify yourself."

    def peer_context(self, peer):
        return {"recipient": peer}


class FixtureAdapter(MockDecisionProvider):
    default_output_tokens = 1024
    default_context_window = 65536

    @classmethod
    def adapter_options(cls, options):
        if set(options) != {"tag"} or not isinstance(options["tag"], str):
            raise ValueError("fixture requires a tag")
        return {"tag": options["tag"]}

    def __init__(self, *, tag, compaction_input_tokens, **kwargs):
        self.turn = 0
        self.tag = tag
        super().__init__(http_transport=httpx.MockTransport(self.respond), **kwargs)

    def submitted_tool_results(self, payload):
        return [
            {"path": f"/messages/{i}/content", "tool_call_id": m["tool_call_id"]}
            for i, m in enumerate(payload["messages"])
            if m["role"] == "tool"
        ]

    def respond(self, request):
        self.turn += 1
        action = ("identify", "write_file", "peer_yield")[self.turn - 1]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": tool_calls({"action": action}),
                            "content": self.tag,
                        },
                    }
                ],
                "usage": {"prompt_tokens": 101, "completion_tokens": 2},
            },
        )


def test_cli_composes_named_extensions_and_retains_evidence(tmp_path, monkeypatch, capsys):
    # Real entry point discovery from installed distribution metadata, without pip or a network.
    dist = tmp_path / "fixture-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: fixture\nVersion: 1.0\n", encoding="utf-8")
    (dist / "entry_points.txt").write_text(
        f"[mikoshi.providers]\nfixture = {__name__}:FixtureAdapter\n"
        f"[mikoshi.team_protocols]\ncustom = {__name__}:FixtureProtocol\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("MOCK_API_KEY", "fixture-host-key")
    repository = tmp_path / "repo"
    repository.mkdir()
    output = tmp_path / "out"
    config = tmp_path / "run-input.json"
    config.write_text(
        json.dumps(
            {
                "kind": "input",
                "run": {
                    "repository": "repo",
                    "output": "out",
                    "model": "fixture",
                    "provider": "fixture",
                    "team_protocol": "custom",
                    "budget_usd": 1,
                    "wall_time_seconds": 20,
                },
                "problem": {
                    "objective": {"id": "x", "title": "x", "description": "Identify"},
                    "limits": {"max_agents": 1},
                },
                "pricing": dict.fromkeys(
                    ("input_rate", "cache_read_rate", "output_rate", "thinking_rate"), 1
                ),
                "deployment": {
                    "enable_commands": False,
                    "provider": {"options": {"tag": "custom wire"}},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["mikoshi", "--config", str(config)])
    cli.main()
    capsys.readouterr()
    assert validate_bundle(output)["status"] == "valid"
    run = json.loads((output / "run.json").read_bytes())
    assert run["stop_reason"] == "all_agents_yielded"
    assert run["resolved"]["team_protocol"] == "custom"
    assert run["provenance"]["extensions"]["mikoshi.providers"]["version"] == "1.0"
    assert run["provenance"]["extensions"]["mikoshi.team_protocols"]["name"] == "custom"
    assert run["input"]["run"]["provider"] == "fixture"
    assert run["resolved"]["model_limits"]["actor_output_ceiling"] == 1024
    assert run["budget"]["accounted_usd"] == pytest.approx(3 * 103 / 1e6)
    submitted = [
        r["data"]
        for r in iter_records(output, record_type="request")
        if r["data"]["kind"] == "provider_request_submitted"
    ]
    first, second, third = [read_content(output, r["payload_ref"]) for r in submitted]
    actor = submitted[0]["agent_id"]
    assert json.loads(first["messages"][-1]["content"])["protocol_context"] == {"recipient": actor}
    assert first["messages"][0]["content"].startswith("Use the identify tool")
    assert first["messages"][0]["content"].count(BATCH_GUIDANCE) == 1
    names = [tool["function"]["name"] for tool in first["tools"]]
    assert names[0] == "identify" and "write_file" not in names
    assert f"received by {actor}" in str(second)
    assert "unknown tool 'write_file'" in str(third)
    assert any(
        item["kind"] == "tool_result" and item["path"].startswith("/messages/")
        for item in submitted[1]["included_content"]
    )
    assert list(repository.iterdir()) == []


@pytest.mark.parametrize("provider_name", ["openai", "anthropic"])
def test_builtins_translate_the_same_custom_peer_turn(tmp_path, provider_name):
    from mikoshi.providers.registry import provider_adapter

    provider = provider_adapter(provider_name)(
        "fixture", api_key="fixture-key", max_tokens=1024, context_window_tokens=128000
    )
    store, runtime = world(
        tmp_path, provider, team_protocol=FixtureProtocol(), enable_commands=False
    )
    try:
        (state,) = runtime._claim_wave(1, 1)
        request = runtime._workspace_request(state)
        payload = provider.input_payload(request)
        names = [tool["name"] for tool in payload["tools"]]
        assert names[0] == "identify" and "write_file" not in names
        contract = (
            payload["instructions"]
            if provider_name == "openai"
            else payload["messages"][-1]["content"]
        )
        assert contract == request.instructions
        assert contract.count(BATCH_GUIDANCE) == 1
        assert json.loads(request.compaction_input_text)["protocol_context"] == {
            "recipient": state.trigger.agent_id
        }
    finally:
        provider.close()
        store.close()
