"""Stored OpenAI Responses chaining over shared durable transport."""

from __future__ import annotations

import copy
import json
from typing import Any

from mikoshi.providers.base import (
    ProviderTransportError,
    RemoteDecisionProvider,
    action_tools,
    compaction_restore_texts,
)
from mikoshi.runtime.budgets import ContextWindowExhausted

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1/"
_OPENAI_HOSTS = ("api.openai.com",) + tuple(
    f"{region}.api.openai.com"
    for region in ("us", "eu", "au", "ca", "jp", "in", "sg", "kr", "gb", "ae")
)
OPENAI_DEFAULT_CONTEXT_WINDOW = 128000
OPENAI_GPT6_CONTEXT_WINDOW = 272000
OPENAI_DEFAULT_MAX_TOKENS = 16384
OPENAI_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")


def _strict_compatible(schema: dict) -> bool:
    """Recognize simple strict schemas without rewriting argument semantics.

    Optional fields, unions and unfamiliar keywords keep best-effort generation.
    Local validation remains authoritative for every tool.
    """
    supported = {
        "type",
        "description",
        "enum",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
    }
    if set(schema) - supported:
        return False
    kind = schema.get("type")
    if kind == "object":
        properties = schema.get("properties", {})
        return (
            schema.get("additionalProperties") is False
            and set(schema.get("required", ())) == set(properties)
            and all(_strict_compatible(child) for child in properties.values())
        )
    if kind == "array":
        return isinstance(schema.get("items"), dict) and _strict_compatible(schema["items"])
    return kind in ("string", "integer", "number", "boolean", "null")


def openai_context_window(
    model_id: str, configured: int | None = None, input_token_limit: int | None = None
) -> int:
    """Use the lower operating context bound for GPT-6 models."""
    if model_id.startswith("gpt-6"):
        if (
            configured is not None
            and configured > OPENAI_GPT6_CONTEXT_WINDOW
            and (
                input_token_limit is None or not 0 < input_token_limit <= OPENAI_GPT6_CONTEXT_WINDOW
            )
        ):
            raise ValueError(
                "GPT-6 context window must not exceed the 272000-token lower operating bound"
            )
        return OPENAI_GPT6_CONTEXT_WINDOW if configured is None else configured
    return OPENAI_DEFAULT_CONTEXT_WINDOW if configured is None else configured


def normalize_openai_base_url(value: str) -> str:
    """Accept only official standard/regional HTTPS endpoints, with no URL extras.

    Regional domains are listed in https://developers.openai.com/api/docs/guides/your-data.
    Restricting the complete URL avoids transmitting API keys to arbitrary hosts.
    """
    allowed = {f"https://{host}/v1" for host in _OPENAI_HOSTS}
    if not isinstance(value, str) or value.removesuffix("/") not in allowed:
        raise ValueError(
            "OpenAI base_url must be an official HTTPS api.openai.com or regional "
            "endpoint with path /v1/ and no credentials, port, query, or fragment"
        )
    return value.removesuffix("/") + "/"


class OpenAITransportError(ProviderTransportError):
    """OpenAI transport failure without credential-bearing HTTP objects."""


class OpenAIDecisionProvider(RemoteDecisionProvider):
    """Use Responses function items and explicitly retained encrypted reasoning."""

    provider_name = "OpenAI"
    api_key_env = "OPENAI_API_KEY"
    base_url = "https://api.openai.com/v1/"
    endpoint = "responses"
    transport_error = OpenAITransportError
    reasoning_efforts = OPENAI_REASONING_EFFORTS
    output_limit_key = "max_output_tokens"
    input_keys = ("instructions", "input", "tools")

    default_output_tokens = OPENAI_DEFAULT_MAX_TOKENS
    context_window = staticmethod(openai_context_window)
    max_temperature = 2
    native_fields = ("response_id", "responses_output")
    native_blocks_key = "responses_output"

    @classmethod
    def deployment_options(cls, deployment, limits):
        return {
            "base_url": deployment.openai_base_url or OPENAI_DEFAULT_BASE_URL,
            "input_token_limit": limits.input_ceiling,
            **cls.adapter_options(deployment.options),
        }

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        *,
        base_url: str = OPENAI_DEFAULT_BASE_URL,
        max_tokens: int = OPENAI_DEFAULT_MAX_TOKENS,
        context_window_tokens: int | None = None,
        input_token_limit: int | None = None,
        compaction_input_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> None:
        context_window_tokens = openai_context_window(
            model_id, context_window_tokens, input_token_limit
        )
        allowance = context_window_tokens - max_tokens
        if input_token_limit is not None and (
            type(input_token_limit) is not int or not 1 <= input_token_limit <= allowance
        ):
            raise ValueError("input_token_limit must fit the context minus output allowance")
        if compaction_input_tokens is not None and (
            type(compaction_input_tokens) is not int
            or not 1 <= compaction_input_tokens <= (input_token_limit or allowance)
        ):
            raise ValueError("compaction_input_tokens must fit the input allowance")
        self.input_token_limit = input_token_limit
        self.compaction_input_tokens = compaction_input_tokens
        self.base_url = normalize_openai_base_url(base_url)
        if temperature is not None and temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        super().__init__(
            model_id,
            api_key,
            max_tokens=max_tokens,
            context_window_tokens=context_window_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    def input_payload(self, request):
        # Reuse the runtime's action schema and writing guidance, then translate
        # only the wire representation. Acceptance still uses the original schema.
        items = []
        previous = None
        if request.history:
            assistants = [m for m in request.history if m.get("role") == "assistant"]
            if len(assistants) != 1 or not assistants[0].get("response_id"):
                raise ValueError("OpenAI continuation requires the last exchange's response ID")
            previous = assistants[0]["response_id"]
            after = request.history[list(request.history).index(assistants[0]) + 1 :]
            for message in after:
                if message.get("role") == "tool":
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": message["tool_call_id"],
                            "output": message["content"],
                        }
                    )
                else:
                    items.append(copy.deepcopy(message))
        items.append(
            {
                "role": "user",
                "content": request.model_input_text
                if request.model_input_text is not None
                else request.input_text,
            }
        )
        payload = {"instructions": request.instructions, "input": items}
        if previous:
            payload["previous_response_id"] = previous
        payload.update(
            tools=[
                {
                    "type": "function",
                    **tool["function"],
                    "strict": not self.model_id.startswith("ft:")
                    and _strict_compatible(tool["function"]["parameters"]),
                }
                for tool in action_tools(request.schema)
            ],
            tool_choice="required",
            parallel_tool_calls=True,
        )
        return payload

    def _new_input_estimate(self, payload):
        rendered = json.dumps(
            {key: payload[key] for key in self.input_keys if key in payload}, ensure_ascii=False
        )
        measured = self._token_meter.count(rendered)
        return len(rendered.encode("utf-8")) if measured is None else measured

    def _check_input_limit(self, estimated):
        if self.input_token_limit is not None and estimated > self.input_token_limit:
            raise ContextWindowExhausted(
                "estimated OpenAI input exceeds input_token_limit; refusing dispatch"
            )

    def request_payload(self, prepared, tokens, handle):
        margin = min(16384, max(1, self.context_window_tokens // 64))
        threshold = self.compaction_input_tokens or max(
            1, self.context_window_tokens - tokens - margin
        )
        if self.input_token_limit is not None:
            threshold = min(threshold, self.input_token_limit)
        payload = self._generation_payload(prepared, tokens)
        new_tokens = self._new_input_estimate(payload)
        previous_tokens = 0
        if prepared.history:
            assistant = next(m for m in prepared.history if m.get("role") == "assistant")
            previous_tokens = assistant.get("context_tokens")
        combined_tokens = (previous_tokens or 0) + new_tokens
        if prepared.history and (previous_tokens is None or combined_tokens >= threshold):
            # A large new observation can cross the ordinary input ceiling in one
            # turn. Send it to the compaction endpoint while it still fits by
            # itself instead of converting a recoverable context boundary into a
            # transport failure. The restored compacted chain is checked again
            # below before action generation.
            compact = {
                key: payload[key]
                for key in ("model", "instructions", "input", "previous_response_id")
            }
            self._check_input_limit(self._new_input_estimate(compact))
            return compact
        self._check_input_limit(combined_tokens)
        return payload

    def _generation_payload(self, prepared, tokens):
        return {
            "model": self.model_id,
            **self.input_payload(prepared),
            "max_output_tokens": tokens,
            "store": True,
            "truncation": "disabled",
            "stream": False,
            **({"temperature": self.temperature} if self.temperature is not None else {}),
            **({"top_p": self.top_p} if self.top_p is not None else {}),
            **(
                {"reasoning": {"effort": self.reasoning_effort}}
                if self.reasoning_effort is not None
                else {}
            ),
        }

    def raw_decision_text(self, message):
        return json.dumps(message["responses_output"], ensure_ascii=False)

    def submitted_tool_results(self, payload):
        return [
            {"path": f"/input/{index}/output", "tool_call_id": item["call_id"]}
            for index, item in enumerate(payload.get("input", ()))
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]

    def request_endpoint(self, payload):
        return "responses" if "max_output_tokens" in payload else "responses/compact"

    def compaction_continuation(self, request, payload, message):
        if self.request_endpoint(payload) != "responses/compact":
            raise ValueError("unexpected compaction during action generation")
        output = message["responses_output"]
        compactions = [item for item in output if item.get("type") == "compaction"]
        if not compactions or any(
            not isinstance(item.get("encrypted_content"), str) or not item["encrypted_content"]
            for item in compactions
        ):
            raise ValueError("compaction requires nonempty encrypted_content")
        restored = self._generation_payload(request, request.max_tokens)
        restored.pop("previous_response_id", None)
        # Preserve the complete canonical window, including retained items.
        restored["input"] = copy.deepcopy(output) + [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text}
                    for text in compaction_restore_texts(request)
                ],
            }
        ]
        # encrypted_content is an opaque provider checkpoint. Its serialized byte
        # length is not its model-context size; using that length caused successful
        # compactions to be rejected before their continuation could run. Use the
        # provider-reported checkpoint output tokens plus the ordinary restored
        # runtime state, retaining the conservative byte estimate as a fallback
        # when usage is unavailable.
        checkpoint_tokens = message.get("context_tokens")
        if type(checkpoint_tokens) is int and checkpoint_tokens >= 0:
            without_checkpoint = copy.deepcopy(restored)
            without_checkpoint["input"] = without_checkpoint["input"][len(output) :]
            restored_tokens = checkpoint_tokens + self._new_input_estimate(without_checkpoint)
        else:
            restored_tokens = self._new_input_estimate(restored)
        self._check_input_limit(restored_tokens)
        return restored

    def parse_message(self, request, message):
        for item in message["responses_output"]:
            if item.get("type") not in ("function_call", "message", "reasoning"):
                raise ValueError("unexpected Responses output item type")
            if item.get("type") == "function_call" and (
                not isinstance(item.get("call_id"), str) or not item["call_id"]
            ):
                raise ValueError("function call requires a nonempty call_id")
        return super().parse_message(request, message)

    def response_body(self, body):
        """Adapt response fields for shared validation and accounting, losslessly.

        The complete native output is retained separately from its validation view;
        subsequent requests refer to the stored response ID.
        """
        if not isinstance(body, dict) or not isinstance(body.get("output"), list):
            raise ValueError("response has no output item list")
        output = body["output"]
        if not all(isinstance(item, dict) for item in output):
            raise ValueError("response contains malformed output items")
        response_id = body.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("response has no stored response ID")
        message = {
            "role": "assistant",
            "response_id": response_id,
            "responses_output": copy.deepcopy(output),
        }
        calls = [item for item in output if item.get("type") == "function_call"]
        text = []
        for item in output:
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text.append(content["text"])
        message["content"] = "".join(text)
        if calls:
            message["tool_calls"] = [
                {
                    "id": item.get("call_id"),
                    "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments")},
                }
                for item in calls
            ]
        native_usage = body.get("usage")
        usage = native_usage if isinstance(native_usage, dict) else {}
        normalized_usage = {
            **usage,
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "prompt_tokens_details": usage.get("input_tokens_details") or {},
            "completion_tokens_details": usage.get("output_tokens_details") or {},
        }
        counts = (usage.get("input_tokens"), usage.get("output_tokens"))
        compacted = body.get("object") == "response.compaction"
        message["context_tokens"] = (
            usage.get("output_tokens")
            if compacted and type(usage.get("output_tokens")) is int
            else sum(counts)
            if all(type(n) is int and n >= 0 for n in counts)
            else None
        )
        complete = body.get("status") == "completed"
        return {
            "native_usage": native_usage,
            "usage": normalized_usage,
            "choices": [
                {
                    "message": message,
                    "finish_reason": "compaction"
                    if compacted
                    else (
                        ("tool_calls" if calls else "stop")
                        if complete
                        else body.get("status", "missing_status")
                    ),
                }
            ],
        }
