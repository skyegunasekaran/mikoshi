"""Anthropic requests require complete streamed messages, including compaction."""

import copy
import json
from dataclasses import replace

import httpx
import pytest
from support.boundaries import run_for
from support.organization import problem, world

from mikoshi.contracts import action_schema
from mikoshi.decisions import ModelDecisionError
from mikoshi.providers.anthropic import AnthropicDecisionProvider, _stream_message
from mikoshi.providers.base import DecisionRequest
from mikoshi.runtime.budgets import ExecutionBudget
from mikoshi.runtime.conversations import conversation_exchange


def events(blocks, reason):
    result = [
        {
            "type": "message_start",
            "message": {
                "id": "message-1",
                "role": "assistant",
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 20, "output_tokens": 1, "cache_read_input_tokens": 100},
            },
        }
    ]
    for index, (block, deltas) in enumerate(blocks):
        result.append({"type": "content_block_start", "index": index, "content_block": block})
        result.extend(
            {"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas
        )
        result.append({"type": "content_block_stop", "index": index})
    result.extend(
        [
            {"type": "ping"},
            {
                "type": "message_delta",
                "delta": {"stop_reason": reason},
                "usage": {"output_tokens": 30},
            },
            {"type": "message_stop"},
        ]
    )
    return result


def wire(values):
    return "".join(
        f"event: {value['type']}\ndata: {json.dumps(value, ensure_ascii=False)}\n\n"
        for value in values
    )


def action_events():
    return events(
        [
            (
                {"type": "thinking", "thinking": "", "signature": ""},
                [
                    {"type": "thinking_delta", "thinking": ""},
                    {"type": "signature_delta", "signature": "signed-thinking"},
                ],
            ),
            (
                {"type": "text", "text": ""},
                [{"type": "text_delta", "text": "Inspecting\u2028activity."}],
            ),
            (
                {"type": "tool_use", "id": "call-1", "name": "list_activity", "input": {}},
                [
                    {"type": "input_json_delta", "partial_json": "{"},
                    {"type": "input_json_delta", "partial_json": "}"},
                ],
            ),
        ],
        "tool_use",
    )


def test_streamed_compaction_restores_state_before_complete_action_and_preserves_usage():
    requests = []
    checkpoint = events(
        [
            (
                {"type": "compaction", "content": None},
                [
                    {"type": "compaction_delta", "content": "Earlier "},
                    {"type": "compaction_delta", "content": "work"},
                ],
            ),
        ],
        "compaction",
    )

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["stream"] is True
        if len(requests) == 2:
            assert payload["messages"][-3]["content"] == [
                {"type": "compaction", "content": "Earlier work"}
            ]
            assert payload["messages"][-2]["role"] == "user"
            assert payload["messages"][-2]["content"][-1]["text"] == "restored state"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=wire(checkpoint if len(requests) == 1 else action_events()),
        )

    provider = AnthropicDecisionProvider(
        "claude-opus-5-5",
        api_key="synthetic",
        max_tokens=128_000,
        context_window_tokens=1_000_000,
        compaction_input_tokens=784_800,
        reasoning_effort="high",
        http_transport=httpx.MockTransport(handler),
    )
    evidence = []
    provider.set_evidence_callback(evidence.append)
    try:
        result = provider.generate_batch(
            [
                DecisionRequest(
                    name="worker",
                    instructions="Work",
                    input_text="state",
                    compaction_input_text="restored state",
                    conversation_id="agent-1",
                    schema={"oneOf": [action_schema("list_activity", {}, required=[])]},
                    metadata={"decision_id": "compacting-decision", "agent_id": "agent-1"},
                )
            ]
        )[0]
        assert result == {"action": "list_activity"}
        assert len(requests) == 2
        native = result.assistant_message["anthropic_content"]
        assert native[0]["signature"] == "signed-thinking"
        assert native[1]["text"] == "Inspecting\u2028activity."
        assert native[2] == {
            "type": "tool_use",
            "id": "call-1",
            "name": "list_activity",
            "input": {},
        }
        assert result.assistant_message["compaction_checkpoints"]
        submitted = [e for e in evidence if e["kind"] == "provider_request_submitted"]
        assert len(submitted) == 2 and len({e["request_id"] for e in submitted}) == 2
        assert result.request_ids == tuple(e["request_id"] for e in submitted)
        assert {e["decision_id"] for e in evidence} == {"compacting-decision"}
        received = [e for e in evidence if e["kind"] == "provider_response_received"]
        decoded = [e for e in evidence if e["kind"] == "provider_response_decoded"]
        assert received[0]["raw_text"] == wire(checkpoint)
        assert received[0]["content_type"] == "text/event-stream"
        assert decoded[0]["body"]["content"] == [{"type": "compaction", "content": "Earlier work"}]
        assert decoded[0]["provider_response_id"] == "message-1"
        usage = _stream_message(wire(action_events()))["usage"]
        assert usage == {"input_tokens": 20, "output_tokens": 30, "cache_read_input_tokens": 100}
    finally:
        provider.close()


@pytest.mark.parametrize("close_block", [False, True])
@pytest.mark.parametrize("reason", ["max_tokens", "model_context_window_exceeded"])
def test_truncated_tool_batch_is_billed_and_gets_feedback_without_executing(
    tmp_path, close_block, reason
):
    truncated = events(
        [
            (
                {"type": "thinking", "thinking": "", "signature": ""},
                [{"type": "signature_delta", "signature": "signed-thinking"}],
            ),
            (
                {"type": "tool_use", "id": "complete", "name": "write_file", "input": {}},
                [
                    {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(
                            {"path": "must-not-run.txt", "content": "rejected batch"}
                        ),
                    }
                ],
            ),
            (
                {"type": "tool_use", "id": "partial", "name": "write_file", "input": {}},
                [
                    {
                        "type": "input_json_delta",
                        "partial_json": '{"path":"partial.txt","content":"unfinished',
                    }
                ],
            ),
        ],
        reason,
    )
    if not close_block:
        truncated = [
            e for e in truncated if not (e["type"] == "content_block_stop" and e["index"] == 2)
        ]
    next(e for e in truncated if e["type"] == "message_delta")["usage"] = {
        "output_tokens": 16384,
        "output_tokens_details": {"thinking_tokens": 3000},
    }
    requests = []
    budget = ExecutionBudget(
        tmp_path / "budget.json",
        dollars=50,
        seconds=7200,
        input_rate=4,
        cache_read_rate=0.2,
        output_rate=20,
        thinking_rate=20,
    )

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["max_tokens"] == 16384
        assert payload["output_config"]["effort"] == "xhigh"
        if len(requests) <= 3:
            if len(requests) > 1:
                assistant = next(
                    m for m in reversed(payload["messages"]) if m["role"] == "assistant"
                )
                assert assistant["content"][0]["signature"] == "signed-thinking"
                assert assistant["content"][-1]["id"] == f"partial-{len(requests) - 1}"
                assert assistant["content"][-1]["input"] == {}
                replies = [
                    b for b in payload["messages"][-1]["content"] if b["type"] == "tool_result"
                ]
                assert {r["tool_use_id"] for r in replies} == {
                    f"complete-{len(requests) - 1}",
                    f"partial-{len(requests) - 1}",
                }
                assert all("output limit" in r["content"] for r in replies)
                assert all(
                    "No tool calls from that response were executed" in r["content"]
                    for r in replies
                )
                assert all("smaller tool arguments" in r["content"] for r in replies)
        if len(requests) <= 2:
            result = copy.deepcopy(truncated)
            for event in result:
                if (
                    event["type"] == "content_block_start"
                    and event["content_block"]["type"] == "tool_use"
                ):
                    event["content_block"]["id"] += f"-{len(requests)}"
        else:
            if len(requests) == 3:
                name, arguments = "write_file", {"path": "corrected.txt", "content": "recovered"}
            else:
                name, arguments = "peer_yield", {}
            result = events(
                [
                    (
                        {
                            "type": "tool_use",
                            "id": f"call-{len(requests)}",
                            "name": name,
                            "input": arguments,
                        },
                        [],
                    )
                ],
                "tool_use",
            )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=wire(result))

    provider = AnthropicDecisionProvider(
        "test-model",
        api_key="synthetic",
        max_tokens=16384,
        context_window_tokens=272000,
        reasoning_effort="xhigh",
        budget=budget,
        http_transport=httpx.MockTransport(respond),
    )
    store, runtime = world(tmp_path, provider, problem(max_agents=1), enable_commands=False)
    try:
        run_for(runtime)
        assert runtime.stop_reason == "all_agents_yielded"
        repository = runtime.workspace.canonical_repository
        assert not (repository / "must-not-run.txt").exists()
        assert not (repository / "partial.txt").exists()
        assert (repository / "corrected.txt").read_text() == "recovered"
        assert len(requests) == 4
        assert [e["completion_tokens"] for e in budget.entries[:2]] == [16384, 16384]
        assert all(e["status"] == "reported" for e in budget.entries)
        assert budget.snapshot()["unknown_usage_requests"] == 0
        assert budget.charged > 0.64
    finally:
        provider.close()
        store.close()


def test_open_tool_block_without_token_limit_remains_a_protocol_error():
    invalid = action_events()
    invalid = [e for e in invalid if not (e["type"] == "content_block_stop" and e["index"] == 2)]
    with pytest.raises(ValueError, match="message stopped before content"):
        _stream_message(wire(invalid))


def test_token_limit_without_message_stop_remains_incomplete():
    incomplete = events(
        [
            (
                {"type": "tool_use", "id": "partial", "name": "list_activity", "input": {}},
                [{"type": "input_json_delta", "partial_json": "{"}],
            )
        ],
        "max_tokens",
    )
    incomplete = [e for e in incomplete if e["type"] not in ("content_block_stop", "message_stop")]
    with pytest.raises(ValueError, match="incomplete Anthropic stream"):
        _stream_message(wire(incomplete))


@pytest.mark.parametrize(
    "block,deltas,close_block",
    [
        ({"type": "text", "text": ""}, [{"type": "text_delta", "text": "unfinished"}], False),
        ({"type": "text", "text": ""}, [], False),
        ({"type": "thinking", "thinking": "", "signature": ""}, [], False),
        (
            {"type": "thinking", "thinking": "", "signature": ""},
            [{"type": "signature_delta", "signature": "unfinished-signature"}],
            False,
        ),
        ({"type": "thinking", "thinking": "", "signature": ""}, [], True),
        (
            {"type": "compaction", "content": None},
            [{"type": "compaction_delta", "content": "unfinished summary"}],
            False,
        ),
        (
            {"type": "compaction", "content": None},
            [{"type": "compaction_delta", "content": "unfinished summary"}],
            True,
        ),
    ],
)
def test_token_limit_recovery_never_replays_unsigned_thinking_or_partial_compaction(
    tmp_path, block, deltas, close_block
):
    limited = events(
        [
            ({"type": "thinking", "thinking": "", "signature": "signed-prefix"}, []),
            (block, deltas),
        ],
        "max_tokens",
    )
    if not close_block:
        limited = [
            e for e in limited if not (e["type"] == "content_block_stop" and e["index"] == 1)
        ]
    next(e for e in limited if e["type"] == "message_delta")["usage"]["output_tokens"] = 16384
    requests = []
    budget = ExecutionBudget(
        tmp_path / "budget.json",
        dollars=1,
        input_rate=1,
        cache_read_rate=0.1,
        output_rate=2,
        thinking_rate=2,
    )

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["max_tokens"] == 16384
        if len(requests) == 2:
            assistant = next(m for m in reversed(payload["messages"]) if m["role"] == "assistant")
            native = assistant["content"]
            assert native[0] == {"type": "thinking", "thinking": "", "signature": "signed-prefix"}
            assert all(b["type"] in {"thinking", "text"} for b in native)
            assert all(
                b.get("signature") == "signed-prefix" for b in native if b["type"] == "thinking"
            )
            assert any(b["type"] == "text" and b["text"] for b in native)
            assert "smaller tool arguments" in json.dumps(payload["messages"][-1])
        result = limited if len(requests) == 1 else action_events()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=wire(result))

    provider = AnthropicDecisionProvider(
        "test-model",
        api_key="synthetic",
        max_tokens=16384,
        context_window_tokens=272000,
        budget=budget,
        http_transport=httpx.MockTransport(respond),
    )
    request = DecisionRequest(
        "worker",
        "Work",
        "state",
        {"oneOf": [action_schema("list_activity")]},
        conversation_id="worker",
    )
    try:
        first = provider.generate_batch([request])[0]
        assert isinstance(first, ModelDecisionError)
        history = conversation_exchange(request, first, {"error": first.feedback})
        second = provider.generate_batch([replace(request, history=history)])[0]
        assert second == {"action": "list_activity"}
        assert budget.entries[0]["completion_tokens"] == 16384
        assert all(e["status"] == "reported" for e in budget.entries)
        assert provider.metrics["compactions"] == 0
    finally:
        provider.close()


@pytest.mark.parametrize(
    "partial", ["", "{", '{"path":"file"', '{"content":"unfinished\\', "[]", "null"]
)
def test_token_limit_keeps_incomplete_tool_inputs_replyable_but_never_executable(partial):
    values = events(
        [
            (
                {"type": "tool_use", "id": "partial", "name": "write_file", "input": {}},
                [{"type": "input_json_delta", "partial_json": partial}],
            )
        ],
        "max_tokens",
    )
    values = [e for e in values if e["type"] != "content_block_stop"]
    message = _stream_message(wire(values))
    assert message["stop_reason"] == "max_tokens"
    assert message["usage"]["output_tokens"] == 30
    assert message["content"] == [
        {"type": "tool_use", "id": "partial", "name": "write_file", "input": {}}
    ]
