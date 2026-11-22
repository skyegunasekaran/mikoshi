"""Shared runtime records, tool schemas, and native tool-turn validation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Agent:
    id: str
    name: str
    instructions: str
    role: str = "member"


@dataclass(frozen=True)
class Trigger:
    id: int
    agent_id: str
    reason: str
    priority: int
    cause_event_id: int | None
    created_tick: int
    kind: str
    project_id: str | None = None


@dataclass(frozen=True)
class ActivationSummary:
    id: int
    agent_id: str
    trigger_id: int
    status: str
    action_count: int
    error: str | None = None


@dataclass(frozen=True)
class RunSummary:
    activations: Sequence[ActivationSummary]
    remaining_triggers: int
    current_tick: int
    causal_waves: int


TEXT_FIELD_CHARS = 65_536
# Even four-byte UTF-8 characters fit below a typical 128 KiB per-argument limit.
COMMAND_ARGUMENT_CHARS = 32_000


def action_schema(
    action: str,
    properties: Mapping[str, Any] | None = None,
    *,
    required: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    """Build a closed discriminated action object."""
    if properties is not None and "action" in properties:
        raise ValueError("action_schema properties cannot replace the action discriminator")
    fields = {"action": {"type": "string", "enum": [action]}}
    fields.update(properties or {})
    required_fields = (
        tuple(fields)
        if required is None
        else ("action", *(field for field in required if field != "action"))
    )
    return {
        "type": "object",
        "properties": fields,
        "required": list(required_fields),
        "additionalProperties": False,
    }


def action_schemas(schema):
    """Index the canonical tool contracts by their unique action names."""
    actions = {}
    for shape in schema.get("oneOf", [schema]):
        names = shape.get("properties", {}).get("action", {}).get("enum", [])
        if (
            len(names) != 1
            or not isinstance(names[0], str)
            or not names[0]
            or "action" not in shape.get("required", [])
            or names[0] in actions
        ):
            raise ValueError("each tool requires a unique, fixed action name")
        actions[names[0]] = shape
    if not actions:
        raise ValueError("at least one tool is required")
    return actions


def validate_json_schema(value: Any, schema: Mapping[str, Any], path: str = "") -> None:
    """Validate tool arguments; messages name fields the way the model wrote them."""
    label = path or "arguments"
    if "oneOf" in schema:
        actions = action_schemas(schema)
        action = value.get("action") if isinstance(value, dict) else None
        if not isinstance(action, str) or action not in actions:
            raise ValueError(f"unknown tool {action!r}; available tools: {sorted(actions)}")
        validate_json_schema(value, actions[action], path)
        return

    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            raise TypeError(f"{label} must be an object")
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{_field(path, key)} is required")
        if schema.get("additionalProperties") is False:
            extra = set(value).difference(properties)
            if extra:
                allowed = sorted(key for key in properties if key != "action")
                raise ValueError(
                    f"{label} has unexpected fields {sorted(extra)}; allowed fields: {allowed}"
                )
        for key, child in value.items():
            if key in properties:
                validate_json_schema(child, properties[key], _field(path, key))
    elif expected == "array":
        if not isinstance(value, list):
            raise TypeError(f"{label} must be an array")
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValueError(f"{label} must contain at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{label} must contain at most {schema['maxItems']} items")
        for index, item in enumerate(value):
            validate_json_schema(item, schema.get("items", {}), f"{label}[{index}]")
    elif expected == "string" and not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    elif expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise TypeError(f"{label} must be an integer")
    elif expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise TypeError(f"{label} must be a number")
    elif expected == "boolean" and not isinstance(value, bool):
        raise TypeError(f"{label} must be true or false")
    if expected == "number" and isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if "minLength" in schema and isinstance(value, str) and len(value) < schema["minLength"]:
        raise ValueError(f"{label} must be at least {schema['minLength']} characters")
    if "maxLength" in schema and isinstance(value, str) and len(value) > schema["maxLength"]:
        raise ValueError(f"{label} must be at most {schema['maxLength']} characters")
    if "pattern" in schema and isinstance(value, str):
        if re.search(str(schema["pattern"]), value) is None:
            raise ValueError(f"{label} must match the pattern {schema['pattern']}")
    if "minimum" in schema and value < schema["minimum"]:
        raise ValueError(f"{label} must be at least {schema['minimum']}")
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        raise ValueError(f"{label} must be greater than {schema['exclusiveMinimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        raise ValueError(f"{label} must be at most {schema['maximum']}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{label} must be one of {schema['enum']}")


def _field(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


BATCH_ACTION = "__tool_batch__"
MAX_TOOL_CALLS = 16
SOLO_ACTIONS = frozenset({"peer_yield"})
BATCH_GUIDANCE = (
    f"Each turn makes 1 to {MAX_TOOL_CALLS} tool calls, run in order; results arrive after "
    "the turn. If a call fails, is blocked or is deferred, the remaining calls are skipped "
    "and earlier effects remain. peer_yield must be the only call in its turn."
)


def validate_tool_turn(value, schema):
    if not isinstance(value, Mapping) or value.get("action") != BATCH_ACTION:
        validate_json_schema(value, schema)
        return
    calls = value.get("actions")
    if (
        set(value) != {"action", "actions"}
        or not isinstance(calls, list)
        or not 2 <= len(calls) <= MAX_TOOL_CALLS
    ):
        raise ValueError(f"a turn can make 1 to {MAX_TOOL_CALLS} tool calls")
    for call in calls:
        try:
            validate_json_schema(call, schema)
        except (TypeError, ValueError) as error:
            name = call.get("action") if isinstance(call, dict) else "tool call"
            raise type(error)(f"{name}: {error}") from error
        if call["action"] in SOLO_ACTIONS:
            raise ValueError(f"{call['action']} must be the only call in its turn")
