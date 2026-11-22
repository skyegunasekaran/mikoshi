"""Offline size attribution for archived conversations; never calls a provider.

Usage: python -m mikoshi.runtime.context_audit path/to/run.sqlite
Measures UTF-8 text bytes, not model tokens or billed cost. Prints sizes and
categories only, never file contents, command output or reasoning text.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _size(text):
    return len(text.encode("utf-8"))


def _category(pointer):
    parts = pointer.split("/")
    return "/".join(parts[:2])


def _action(messages):
    for message in messages:
        for item in message.get("responses_output", ()):
            if item.get("type") == "function_call":
                return item.get("name", "unknown")
        for item in message.get("tool_calls") or ():
            return item.get("function", {}).get("name", "unknown")
        if message.get("role") == "assistant":
            try:
                result = json.loads(message.get("content") or "null")
            except (ValueError, TypeError):
                continue
            if isinstance(result, dict):
                return result.get("action", "unknown")
    return "unknown"


def _call_actions(messages):
    return {
        call.get("id"): call.get("function", {}).get("name", "unknown")
        for message in messages
        for call in message.get("tool_calls") or ()
        if isinstance(call, dict) and isinstance(call.get("function"), dict)
    }


def audit_database(path: Path) -> dict:
    """Attribute stored text once per exchange, not once per subsequent replay."""
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    agents = {}
    try:
        for row in connection.execute(
            "SELECT sequence,agent_id,messages,full_state_json FROM agent_memory_exchanges "
            "ORDER BY sequence"
        ):
            agent = agents.setdefault(
                row["agent_id"],
                {
                    "exchanges": 0,
                    "text_bytes": Counter(),
                    "state_fields_bytes": Counter(),
                    "tool_result_bytes_by_action": Counter(),
                    "largest_tool_results": [],
                    "opaque_provider_bytes": 0,
                },
            )
            agent["exchanges"] += 1
            messages = json.loads(row["messages"])
            action = _action(messages)
            call_actions = _call_actions(messages)
            for index, message in enumerate(messages):
                content = message.get("content", "")
                content = content if isinstance(content, str) else _json(content)
                if index == 0 and message.get("role") == "user":
                    agent["text_bytes"]["state_observations"] += _size(content)
                    observation = json.loads(content)
                    for pointer, value in (
                        ("/" + key, value) for key, value in observation.items()
                    ):
                        agent["state_fields_bytes"][_category(pointer)] += (
                            _size(_json({pointer: value})) - 2
                        )
                elif "responses_output" in message:
                    # The adapter sends native items, not its duplicate chat view.
                    for item in message["responses_output"]:
                        kind = item.get("type")
                        if kind == "function_call":
                            agent["text_bytes"]["assistant_actions"] += _size(
                                item.get("arguments", "")
                            )
                        elif kind == "message":
                            for part in item.get("content", ()):
                                agent["text_bytes"]["assistant_text"] += _size(part.get("text", ""))
                        elif kind in {"reasoning", "compaction"}:
                            agent["opaque_provider_bytes"] += _size(
                                item.get("encrypted_content", "")
                            )
                elif message.get("role") == "assistant":
                    agent["text_bytes"]["assistant_actions"] += _size(content)
                    for call in message.get("tool_calls") or ():
                        agent["text_bytes"]["assistant_actions"] += _size(
                            call.get("function", {}).get("arguments", "")
                        )
                    agent["text_bytes"]["provider_reasoning"] += _size(
                        message.get("reasoning_content") or ""
                    )
                else:
                    count = _size(content)
                    result_action = call_actions.get(message.get("tool_call_id"), action)
                    agent["text_bytes"]["tool_results"] += count
                    agent["tool_result_bytes_by_action"][result_action] += count
                    agent["largest_tool_results"].append(
                        {"sequence": row["sequence"], "action": result_action, "bytes": count}
                    )
            agent["largest_tool_results"] = sorted(
                agent["largest_tool_results"], key=lambda item: item["bytes"], reverse=True
            )[:10]
    finally:
        connection.close()
    totals = Counter()
    fields = Counter()
    tools = Counter()
    for agent in agents.values():
        totals.update(agent["text_bytes"])
        fields.update(agent["state_fields_bytes"])
        tools.update(agent["tool_result_bytes_by_action"])
    return {
        "database": str(path.resolve()),
        "units": "UTF-8 text bytes",
        "text_bytes": dict(totals.most_common()),
        "state_fields_bytes": dict(fields.most_common()),
        "tool_result_bytes_by_action": dict(tools.most_common()),
        "agents": agents,
        "interpretation": (
            "Each archived exchange is counted once, including history later compacted. "
            "These are text sizes, not input tokens, replay-weighted costs or a billing estimate. "
            "System instructions, tool schemas, failed transport attempts and message framing "
            "are excluded. Opaque provider content is measured separately and is not decoded. "
            "Compaction restoration text is excluded."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_database(args.database), indent=2))


if __name__ == "__main__":
    main()
