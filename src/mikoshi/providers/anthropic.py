"""Streaming Anthropic Messages adapter over the shared durable HTTP transport."""

from __future__ import annotations

import copy
import json
import uuid
from typing import Any

from mikoshi.providers.base import (
    ProviderTransportError,
    RemoteDecisionProvider,
    action_tools,
    compaction_restore_texts,
)
from mikoshi.providers.usage import aggregate_message_usage

# Match Opus 5's documented limits. Other deployments can override both.
ANTHROPIC_DEFAULT_CONTEXT_WINDOW = 1_000_000
ANTHROPIC_DEFAULT_MAX_TOKENS = 128_000
ANTHROPIC_REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_TOKEN_LIMIT_REASONS = {"max_tokens", "model_context_window_exceeded"}


_SYSTEM = (
    "The latest system message in the conversation holds the current instructions and "
    "replaces earlier ones."
)


def _tools(request):
    """One native Anthropic tool per action, with its description and argument schema."""
    tools = action_tools(request.schema)
    return [
        {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "input_schema": {"type": "object", **tool["function"]["parameters"]},
        }
        for tool in tools
    ]


class AnthropicTransportError(ProviderTransportError):
    """Credential-free Anthropic transport failure."""


def _stream_message(text):
    """Accumulate a complete SSE message before exposing actions or settling usage."""
    message = None
    active = None
    fragments = {}
    stopped = False
    data = []
    inputs = {}
    incomplete = set()

    def finish_block():
        nonlocal active
        block = message["content"][active]
        for field, chunks in fragments.items():
            value = "".join(chunks)
            if field == "partial_json":
                # Wait for the stop reason before judging incomplete JSON.
                inputs[active] = value
            else:
                block[field] = (block.get(field) or "") + value
        active = None

    # SSE uses CR/LF delimiters; Unicode line separators can occur inside JSON strings.
    for line in [*text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), ""]:
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line and data:
            event = json.loads("\n".join(data))
            data = []
            kind = event["type"]
            if kind == "error":
                error_type = event.get("error", {}).get("type", "unknown")
                detail = event.get("error", {}).get("message", "")
                raise AnthropicTransportError(
                    f"Anthropic stream failed: {error_type}: {detail}",
                    status_code=529 if error_type == "overloaded_error" else None,
                )
            if kind == "ping":
                continue
            if stopped:
                raise ValueError("event after message_stop")
            if kind == "message_start":
                if message is not None:
                    raise ValueError("duplicate message_start")
                message = copy.deepcopy(event["message"])
                if message.get("content") != []:
                    raise ValueError("message_start must have empty content")
            elif kind in {
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            }:
                if message is None:
                    raise ValueError("stream event before message_start")
                if kind == "content_block_start":
                    if active is not None or event["index"] != len(message["content"]):
                        raise ValueError("invalid content block order")
                    active = event["index"]
                    fragments = {}
                    message["content"].append(copy.deepcopy(event["content_block"]))
                elif kind in {"content_block_delta", "content_block_stop"}:
                    if active is None or event["index"] != active:
                        raise ValueError("content event does not match open block")
                    if kind == "content_block_delta":
                        delta = event["delta"]
                        fields = {
                            "text_delta": "text",
                            "thinking_delta": "thinking",
                            "signature_delta": "signature",
                            "input_json_delta": "partial_json",
                            "compaction_delta": "content",
                        }
                        field = fields.get(delta["type"])
                        if field is None:
                            raise ValueError("unsupported content delta")
                        fragments.setdefault(field, []).append(delta[field] or "")
                    else:
                        finish_block()
                elif kind == "message_delta":
                    message.update(event["delta"])
                    # Usage deltas are cumulative, not additive. Preserve input
                    # and cache fields from message_start when omitted here.
                    message.setdefault("usage", {}).update(event.get("usage", {}))
                else:
                    truncated = message.get("stop_reason") in _TOKEN_LIMIT_REASONS
                    if not message.get("stop_reason"):
                        raise ValueError("message stopped before content or stop reason")
                    if active is not None:
                        # Token limits can interrupt text, thinking or tool input.
                        # Keep terminal billing independent of content completion;
                        # response_body removes blocks that cannot be replayed.
                        if not truncated:
                            raise ValueError("message stopped before content or stop reason")
                        incomplete.add(active)
                        finish_block()
                    for index, value in inputs.items():
                        if not value:
                            continue  # No-argument tools can emit an empty delta.
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            if not truncated:
                                raise
                            # Keep the authored call ID replyable with length
                            # feedback. The length finish reason rejects the whole
                            # batch before execution; these are not repaired args.
                            value = {}
                            incomplete.add(index)
                        if truncated and not isinstance(value, dict):
                            value = {}
                            incomplete.add(index)
                        message["content"][index]["input"] = value
                    if incomplete:
                        message["_incomplete_block_indices"] = sorted(incomplete)
                    stopped = True
            # New top-level event types can be ignored per the provider protocol.
    if not stopped:
        raise ValueError("incomplete Anthropic stream")
    return message


class AnthropicDecisionProvider(RemoteDecisionProvider):
    provider_name = "Anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    base_url = "https://api.anthropic.com/v1/"
    endpoint = "messages"
    transport_error = AnthropicTransportError
    reasoning_efforts = ANTHROPIC_REASONING_EFFORTS
    output_limit_key = "max_tokens"
    input_keys = ("system", "messages", "tools")

    default_context_window = ANTHROPIC_DEFAULT_CONTEXT_WINDOW
    default_output_tokens = ANTHROPIC_DEFAULT_MAX_TOKENS
    minimum_compaction_tokens = 50_000
    native_fields = ("anthropic_content", "anthropic_turn")
    native_blocks_key = "anthropic_content"

    @classmethod
    def validate_prices(cls, prices):
        if prices.output_rate != prices.thinking_rate:
            raise ValueError("Anthropic requires equal output and thinking token prices")

    @classmethod
    def provenance(cls, provider):
        return {
            **super().provenance(provider),
            "anthropic_prompt_caching": getattr(provider, "prompt_caching", None),
            "anthropic_cache_ttl": getattr(provider, "cache_ttl", None),
        }

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        *,
        max_tokens: int = ANTHROPIC_DEFAULT_MAX_TOKENS,
        prompt_caching: bool = True,
        cache_ttl: str = "5m",
        compaction_input_tokens: int | None = None,
        context_window_tokens: int = ANTHROPIC_DEFAULT_CONTEXT_WINDOW,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> None:
        budget = kwargs.get("budget")
        if budget is not None and budget.output_rate != budget.thinking_rate:
            raise ValueError("Anthropic requires equal output and thinking token prices")
        if not isinstance(prompt_caching, bool):
            raise ValueError("prompt_caching must be boolean")
        if cache_ttl not in ("5m", "1h"):
            raise ValueError("cache_ttl must be 5m or 1h")
        self.cache_ttl = cache_ttl
        if compaction_input_tokens is not None and (
            type(compaction_input_tokens) is not int
            or not 50000 <= compaction_input_tokens <= context_window_tokens - max_tokens
        ):
            raise ValueError(
                "compaction_input_tokens must be between 50000 and the input allowance"
            )
        self.compaction_input_tokens = compaction_input_tokens
        self.prompt_caching = prompt_caching
        self._sessions = {}
        if temperature is not None and temperature > 1:
            raise ValueError("temperature must be between 0 and 1")
        super().__init__(
            model_id,
            api_key,
            max_tokens=max_tokens,
            context_window_tokens=context_window_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    def request_headers(self):
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "compact-2026-01-12,thinking-binding-controls-2026-08-01",
        }

    async def _request(self, payload, stats):
        """Normalize the required event stream before shared accounting and validation."""
        import httpx

        response = await super()._request(payload, stats)
        try:
            if "text/event-stream" not in response.headers.get("content-type", ""):
                raise ValueError("expected an Anthropic event stream")
            # HTTPX buffers the wire stream while reading it under the shared
            # deadline. The runtime consumes only the completed native message.
            message = _stream_message(response.text)
        except AnthropicTransportError as error:
            # SSE errors arrive in successful HTTP responses. Preserve their
            # diagnostic message, but apply the same redaction and size bound
            # as HTTP errors before it reaches logs or persisted run evidence.
            raise self.transport_error(
                self._redact(error)[:500], status_code=error.status_code
            ) from None
        except (ValueError, TypeError, KeyError, IndexError) as error:
            raise self.transport_error(
                f"Invalid Anthropic stream: {self._redact(error)[:500]}"
            ) from None
        normalized = httpx.Response(200, json=message, request=response.request)
        self._response_evidence(normalized, stats, kind="provider_response_decoded")
        return normalized

    def input_payload(self, request):
        session = self._sessions.get(request.conversation_id) if request.conversation_id else None
        tools = _tools(request)
        messages = copy.deepcopy(session["messages"]) if session else []
        content = []
        if request.history:
            if not session:
                raise ValueError("Anthropic continuation requires its live native conversation")
            assistants = [m for m in request.history if m.get("role") == "assistant"]
            if len(assistants) != 1 or assistants[0].get("anthropic_turn") != session["turn"]:
                raise ValueError("Anthropic continuation does not match the last native turn")
            for message in request.history[list(request.history).index(assistants[0]) + 1 :]:
                if message.get("role") == "tool":
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": message["tool_call_id"],
                            "content": message["content"],
                        }
                    )
                else:
                    content.append({"type": "text", "text": message["content"]})
        elif session:
            raise ValueError("Anthropic continuation is missing the last committed exchange")
        content.append(
            {
                "type": "text",
                "text": request.model_input_text
                if request.model_input_text is not None
                else request.input_text,
            }
        )
        messages.append({"role": "user", "content": content})
        contract = request.instructions
        if not session or contract != session["contract"]:
            messages.append({"role": "system", "content": contract})
        return {
            "system": _SYSTEM,
            "messages": messages,
            "tools": tools,
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": False},
        }

    def billing_request(self, payload):
        return {**super().billing_request(payload), "cache_control": payload.get("cache_control")}

    def submitted_tool_results(self, payload):
        return [
            {
                "path": f"/messages/{index}/content/{position}/content",
                "tool_call_id": block["tool_use_id"],
            }
            for index, message in enumerate(payload.get("messages", ()))
            if message.get("role") == "user" and isinstance(message.get("content"), list)
            for position, block in enumerate(message["content"])
            if block.get("type") == "tool_result"
        ]

    def remember_response(self, request, payload, message):
        if request.conversation_id:
            turn = uuid.uuid4().hex
            message["anthropic_turn"] = turn
            self._sessions[request.conversation_id] = {
                "turn": turn,
                "messages": copy.deepcopy(payload["messages"])
                + [{"role": "assistant", "content": copy.deepcopy(message["anthropic_content"])}],
                "contract": request.instructions,
            }

    def close(self):
        try:
            super().close()
        finally:
            self._sessions.clear()

    def request_payload(self, prepared, tokens, handle):
        margin = min(16384, max(1, self.context_window_tokens // 64))
        threshold = self.compaction_input_tokens or (self.context_window_tokens - tokens - margin)
        if threshold < 50000:
            raise ValueError(
                "Anthropic compaction requires at least 50000 input tokens of headroom"
            )
        return {
            "context_management": {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {"type": "input_tokens", "value": threshold},
                        "pause_after_compaction": True,
                    }
                ]
            },
            "model": self.model_id,
            "stream": True,
            **self.input_payload(prepared),
            "max_tokens": tokens,
            **(
                {"cache_control": {"type": "ephemeral", "ttl": self.cache_ttl}}
                if self.prompt_caching
                else {}
            ),
            **({"temperature": self.temperature} if self.temperature is not None else {}),
            **({"top_p": self.top_p} if self.top_p is not None else {}),
            **(
                {
                    "thinking": {
                        "type": "adaptive",
                        "block_binding": {"prefix_mismatch_behavior": "error"},
                    },
                    "output_config": {"effort": self.reasoning_effort},
                }
                if self.reasoning_effort is not None
                else {}
            ),
        }

    def compaction_continuation(self, request, payload, message):
        blocks = message["anthropic_content"]
        if (
            len(blocks) != 1
            or blocks[0].get("type") != "compaction"
            or not isinstance(blocks[0].get("content"), str)
            or not blocks[0]["content"].strip()
        ):
            raise ValueError("compaction pause requires exactly one nonempty compaction block")
        restored = copy.deepcopy(payload)
        restored["messages"].extend(
            [
                {"role": "assistant", "content": copy.deepcopy(blocks)},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text} for text in compaction_restore_texts(request)
                    ],
                },
                # Mid-conversation system content must follow a user turn and
                # precede the next assistant turn (or end the request).
                {"role": "system", "content": request.instructions},
            ]
        )
        return restored

    def parse_message(self, request, message):
        for block in message["anthropic_content"]:
            if block.get("type") not in (
                "text",
                "thinking",
                "redacted_thinking",
                "tool_use",
            ):
                raise ValueError("unexpected Anthropic content block type")
        return super().parse_message(request, message)

    def response_body(self, body):
        if not isinstance(body, dict) or not isinstance(body.get("content"), list):
            raise ValueError("response has no content block list")
        content = body["content"]
        if not all(isinstance(block, dict) for block in content):
            raise ValueError("response contains malformed content blocks")
        reason = body.get("stop_reason")
        if reason in _TOKEN_LIMIT_REASONS:
            incomplete = set(body.get("_incomplete_block_indices", ()))
            content = [
                block
                for index, block in enumerate(content)
                if block.get("type") != "compaction"
                and not (
                    block.get("type") in {"thinking", "redacted_thinking"}
                    and (
                        index in incomplete
                        or (
                            block.get("type") == "thinking"
                            and not (isinstance(block.get("signature"), str) and block["signature"])
                        )
                    )
                )
                and not (block.get("type") == "text" and not block.get("text"))
            ]
            # Unsigned thinking and unfinished summaries cannot be sent back as
            # native conversation content. Preserve signed, completed blocks and
            # use an explicit runtime marker when no replyable content remains.
            # The original wire response stays in the durable evidence.
            if not any(block.get("type") in {"text", "tool_use"} for block in content):
                content = [
                    *content,
                    {
                        "type": "text",
                        "text": "[Runtime: the response reached its token limit before completing a tool call.]",
                    },
                ]
        message = {
            "role": "assistant",
            "anthropic_content": copy.deepcopy(content),
            "content": "".join(b.get("text", "") for b in content if b.get("type") == "text"),
        }
        calls = [b for b in content if b.get("type") == "tool_use"]
        if calls:
            message["tool_calls"] = [
                {
                    "id": b.get("id"),
                    "type": "function",
                    "function": {
                        "name": b.get("name"),
                        "arguments": json.dumps(b["input"])
                        if isinstance(b.get("input"), dict)
                        else "null",
                    },
                }
                for b in calls
            ]
        native_usage = body.get("usage")
        try:
            usage = aggregate_message_usage(native_usage)
        except (TypeError, ValueError):
            usage = None
        if not isinstance(usage, dict):
            usage = {}
        # Ordinary input, cache creation and cache reads are disjoint categories.
        created = usage.get("cache_creation_input_tokens", 0)
        cached = usage.get("cache_read_input_tokens", 0)
        uncached = usage.get("input_tokens")
        valid = all(type(n) is int and n >= 0 for n in (created, cached, uncached))
        normalized = {
            "prompt_tokens": uncached + cached + created if valid else None,
            "completion_tokens": usage.get("output_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "output_tokens_details": usage.get("output_tokens_details", {}),
            "prompt_tokens_details": {"cached_tokens": cached, "cache_creation_tokens": created},
        }
        # Thinking is included in output_tokens; preserve a reported breakdown
        # when available, without inventing one for older responses.
        return {
            "native_usage": native_usage,
            "usage": normalized,
            "choices": [
                {
                    "message": message,
                    "finish_reason": {
                        "tool_use": "tool_calls",
                        "end_turn": "stop",
                        "max_tokens": "length",
                        "model_context_window_exceeded": "length",
                    }.get(reason, reason),
                }
            ],
        }
