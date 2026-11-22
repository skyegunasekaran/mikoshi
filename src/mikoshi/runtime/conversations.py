"""Agent conversations and observation baselines, retained in the run database.

Providers manage model-visible compaction; original exchanges stay available for audit.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from mikoshi.contracts import BATCH_ACTION
from mikoshi.decisions import DecisionRequest

SCHEMA = """
            CREATE TABLE IF NOT EXISTS agent_memory_exchanges (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                conversation_id TEXT,
                exchange_key TEXT NOT NULL,
                decision_id TEXT,
                request_id TEXT,
                request_ids TEXT NOT NULL DEFAULT '[]',
                tool_calls TEXT NOT NULL DEFAULT '[]',
                messages TEXT NOT NULL,
                full_state_json TEXT NOT NULL,
                UNIQUE(agent_id, exchange_key)
            );
"""


_TEXT_SECTIONS = {
    "content": "Content",
    "output": "Output",
    "markdown": "Markdown",
    "comments": "Comments",
    "message": "Message",
    "error": "Error",
}


def render_tool_result(result) -> str:
    """Keep code, logs and prose literal instead of embedding them in JSON strings."""
    if isinstance(result, str):
        return result
    if not isinstance(result, Mapping):
        return json.dumps(result, ensure_ascii=False, default=str)
    text = {key: result[key] for key in _TEXT_SECTIONS if isinstance(result.get(key), str)}
    metadata = {
        key: value for key, value in result.items() if key not in text and value is not None
    }
    parts = (
        [json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), default=str)]
        if metadata
        else []
    )
    for key, value in text.items():
        if not metadata and len(text) == 1 and key in {"message", "error"}:
            parts.append(value)
        else:
            parts.append(f"{_TEXT_SECTIONS[key]}:\n{value}")
    return "\n\n".join(parts) if parts else "{}"


def native_tool_calls(result) -> list[dict]:
    """Return only authored native call identities, preserving provider order."""
    assistant = getattr(result, "assistant_message", None) or {}

    calls = assistant.get("tool_calls")
    if not isinstance(calls, list):
        return []
    raw = [
        (
            call.get("id"),
            call["function"].get("name") if isinstance(call.get("function"), Mapping) else None,
        )
        for call in calls
        if isinstance(call, Mapping)
    ]
    return [
        {
            "index": index,
            "id": call_id if isinstance(call_id, str) and call_id else None,
            "name": name if isinstance(name, str) and name else None,
        }
        for index, (call_id, name) in enumerate(raw)
    ]


def conversation_exchange(request: DecisionRequest, result, observation: Any) -> list[dict]:
    """Build a complete exchange, including tool replies before the next request."""
    messages = [
        {
            "role": "user",
            "content": request.model_input_text
            if request.model_input_text is not None
            else request.input_text,
        }
    ]
    assistant = copy.deepcopy(getattr(result, "assistant_message", None))
    if not assistant:
        # Scripted providers have authored actions but no native assistant message.
        content = (
            getattr(result, "raw_output", "")
            if isinstance(result, Exception)
            else json.dumps(dict(result), ensure_ascii=False)
        )
        assistant = {"role": "assistant", "content": content or str(result)}
    calls = assistant.get("tool_calls")
    calls = [] if calls is None else calls
    # Native call identities remain replyable even when authored arguments or
    # the normalized action name fail validation. Never execute them here.
    if (
        not isinstance(calls, list)
        or not all(
            isinstance(call, Mapping) and isinstance(call.get("id"), str) and call["id"]
            for call in calls
        )
        or len({call["id"] for call in calls}) != len(calls)
    ):
        # Malformed provider output cannot form a valid native tool exchange. Keep
        # the exact returned object as text rather than manufacturing call IDs.
        assistant = {
            "role": "assistant",
            "content": json.dumps(assistant, ensure_ascii=False),
            **{k: v for k, v in assistant.items() if k not in {"role", "content", "tool_calls"}},
        }
        calls = []
    messages.append(assistant)
    content = render_tool_result(observation)
    if calls:
        # Validation can reject the entire batch before any calls run. That shared
        # error still needs a reply for every native call so the next turn can recover.
        batched = (
            not isinstance(result, Exception)
            and result.get("action") == BATCH_ACTION
            and isinstance(observation, Mapping)
            and "results" in observation
        )
        observations = observation["results"] if batched else []
        if batched and (not isinstance(observations, list) or len(observations) != len(calls)):
            raise ValueError("native tool batch requires one result per call")
        messages.extend(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": render_tool_result(observations[index]) if batched else content,
            }
            for index, call in enumerate(calls)
        )
    else:
        messages.append({"role": "user", "content": content})
    return messages


class ConversationHistory:
    """Per-agent conversation continuity and archived exchanges."""

    def __init__(self, store):
        self._store = store
        self._records = store.conversations
        self._records.require_supported_format()

    def append_exchange(
        self, agent_id: str, exchange_key: str, request: DecisionRequest, result, observation: Any
    ) -> None:
        messages = conversation_exchange(request, result, observation)
        compacted = bool(
            (getattr(result, "assistant_message", None) or {}).get("compaction_checkpoints")
        )
        baseline = (
            request.compaction_input_text
            if compacted and request.compaction_input_text is not None
            else request.input_text
        )
        decision_id = request.metadata.get("decision_id")
        request_id = getattr(result, "request_id", None)
        request_ids = list(getattr(result, "request_ids", ()) or ())
        tool_calls = native_tool_calls(result)
        sequence = self._records.append(
            agent_id=agent_id,
            conversation_id=request.conversation_id or agent_id,
            exchange_key=exchange_key,
            decision_id=decision_id,
            request_id=request_id,
            request_ids=request_ids,
            tool_calls=tool_calls,
            messages=messages,
            full_state_json=baseline,
        )

        if sequence is not None:
            try:
                delivered_state = json.loads(baseline)
            except (TypeError, ValueError):
                delivered_state = {"raw_text": baseline}
            if not isinstance(delivered_state, dict):
                delivered_state = {"raw_text": baseline}
            self._store.publish_exchange(
                {
                    "sequence": sequence,
                    "agent_id": agent_id,
                    "conversation_id": request.conversation_id or agent_id,
                    "exchange_key": exchange_key,
                    "decision_id": decision_id,
                    "request_id": request_id,
                    "request_ids": request_ids,
                    "tool_calls": tool_calls,
                    "messages": messages,
                    "state": delivered_state,
                }
            )

    def _previous_state_json(self, agent_id: str) -> str | None:
        return self._records.latest_state_json(agent_id)

    def last_state(self, agent_id: str) -> dict | None:
        """The full snapshot from the last successfully delivered observation, if any."""
        text = self._previous_state_json(agent_id)
        try:
            return json.loads(text) if text is not None else None
        except (ValueError, TypeError):
            return None

    def read_tool_result(self, agent_id, result_id, offset=0, limit=2048):
        sequence, index = map(int, result_id.split("/"))
        replies = [
            message
            for message in self._records.messages(agent_id, sequence)
            if message["role"] == "tool"
        ]
        if not 0 <= index < len(replies):
            raise ValueError("Unknown tool result in this peer's conversation")
        text = replies[index]["content"]
        if offset > len(text):
            raise ValueError("Offset exceeds the tool result length")
        end = offset + limit
        return {
            "result_id": result_id,
            "content": text[offset:end],
            "next_offset": end if end < len(text) else None,
        }

    def last_exchange(self, agent_id: str, *, result_budget=None, pageable=True) -> list[dict]:
        """Read only the continuation of the last committed action."""
        row = self._records.latest_exchange(agent_id)
        history = json.loads(row["messages"]) if row else []
        replies = [m for m in history if m.get("role") == "tool"]
        if result_budget is None or not replies:
            return history
        sizes = [len(json.dumps(m["content"], ensure_ascii=False).encode()) for m in replies]
        if sum(sizes) > result_budget:
            share = result_budget // len(replies)
            for index, (reply, size) in enumerate(zip(replies, sizes)):
                # Keep short receipts intact, even when the handle would be larger.
                if size <= max(share, 128):
                    continue
                body_budget = max(0, share - 128)
                prefix = reply["content"][:body_budget]
                while prefix and len(json.dumps(prefix, ensure_ascii=False).encode()) > body_budget:
                    prefix = prefix[: len(prefix) // 2]
                if not pageable:
                    reply["content"] = f"{prefix}\n[Shortened tool result.]"
                    continue
                result_id = f"{row['sequence']}/{index}"
                reply["content"] = (
                    f"{prefix}\n[Shortened tool result. Continue with "
                    f'read_output(reference="tool:{result_id}", offset={len(prefix)}).]'
                )
        return history
