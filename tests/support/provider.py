"""Provider transport fixtures with a deliberately small decision schema."""

from __future__ import annotations

import json

from mikoshi.providers.base import (
    ProviderTransportError,
    RemoteDecisionProvider,
    action_tools,
)


def tool_calls(action):
    """Render an explicitly authored action as a native tool response."""
    name = action["action"]
    arguments = json.dumps({key: value for key, value in action.items() if key != "action"})
    return [
        {
            "id": "call_mock",
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }
    ]


class MockDecisionProvider(RemoteDecisionProvider):
    """Mock-only transport for shared scheduler/validation tests; never contacts a service."""

    provider_name = "Mock"
    api_key_env = "MOCK_API_KEY"
    base_url = "https://mock.invalid/"
    endpoint = "decisions"
    transport_error = ProviderTransportError
    reasoning_efforts = ("low", "medium", "high")
    output_limit_key = "max_tokens"
    input_keys = ("messages", "tools")

    def __init__(self, *args, **kwargs):
        if kwargs.get("http_transport") is None:
            raise ValueError("mock provider requires an explicit test transport")
        super().__init__(*args, **kwargs)

    def input_payload(self, request):
        return {
            "messages": [
                {"role": "system", "content": request.instructions},
                *request.history,
                {"role": "user", "content": request.model_input_text or request.input_text},
            ],
            "tools": action_tools(request.schema),
        }

    def request_payload(self, prepared, tokens, handle):
        return {
            "model": self.model_id,
            **self.input_payload(prepared),
            "max_tokens": tokens,
            "request_index": handle,
        }
