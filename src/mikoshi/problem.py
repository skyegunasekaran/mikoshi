"""Problem-first run inputs, independent of an initial organization or task graph."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

IMPLEMENTED_CAPABILITIES = ("read_workspace", "write_artifact")


def validate_supported_capabilities(capabilities: Sequence[str]) -> None:
    duplicates = {capability for capability in capabilities if capabilities.count(capability) > 1}
    if duplicates:
        raise ValueError(f"supported capabilities are duplicated: {sorted(duplicates)}")
    unknown = set(capabilities).difference(IMPLEMENTED_CAPABILITIES)
    if unknown:
        raise ValueError(
            f"supported capabilities have no executable runtime action: {sorted(unknown)}"
        )


@dataclass(frozen=True)
class ObjectiveSpec:
    id: str
    title: str
    description: str
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResourceLimits:
    max_agents: int = 8


@dataclass(frozen=True)
class TeamMemberSpec:
    id: str
    instructions: str = ""


@dataclass(frozen=True)
class TeamSpec:
    capabilities: tuple[str, ...] = IMPLEMENTED_CAPABILITIES
    members: tuple[TeamMemberSpec, ...] = ()


@dataclass(frozen=True)
class ScheduledNotificationSpec:
    id: str
    at_minutes: float
    recipient: str
    message: str


@dataclass(frozen=True)
class ProblemSpec:
    objective: ObjectiveSpec
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    team: TeamSpec = field(default_factory=TeamSpec)
    scheduled_notifications: tuple[ScheduledNotificationSpec, ...] = ()


def _object(value: Any, label: str, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - fields
    if unknown:
        raise ValueError(f"unknown {label} fields: {sorted(unknown)}")
    return value


def _text(value: Any, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{label} must be a {'string' if empty else 'nonempty string'}")
    return value


def _identifier(value: Any, label: str) -> str:
    value = _text(value, label)
    if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", value):
        raise ValueError(
            f"{label} must be a lowercase identifier (letters, digits, _, ., -; at most 64 characters)"
        )
    return value


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return tuple(_text(item, label) for item in value)


def parse_problem(value: Any) -> ProblemSpec:
    """Validate the objective, fixed roster, capabilities and execution policy."""
    root = _object(
        value,
        "problem",
        {
            "objective",
            "limits",
            "team",
            "scheduled_notifications",
        },
    )
    obj = _object(
        root.get("objective"),
        "objective",
        {
            "id",
            "title",
            "description",
            "constraints",
        },
    )
    objective = ObjectiveSpec(
        id=_identifier(obj.get("id"), "objective.id"),
        title=_text(obj.get("title"), "objective.title"),
        description=_text(obj.get("description"), "objective.description"),
        constraints=_strings(obj.get("constraints", []), "objective.constraints"),
    )
    defaults = ResourceLimits()
    raw_limits = _object(root.get("limits", {}), "limits", set(defaults.__dataclass_fields__))
    limits = {}
    for name in defaults.__dataclass_fields__:
        amount = raw_limits.get(name, getattr(defaults, name))
        if type(amount) is not int or amount < 1:
            raise ValueError(f"limits.{name} must be a positive integer")
        limits[name] = amount
    raw_team = _object(root.get("team", {}), "team", {"capabilities", "members"})
    capabilities = _strings(
        raw_team.get("capabilities", list(IMPLEMENTED_CAPABILITIES)), "team capabilities"
    )
    validate_supported_capabilities(capabilities)
    raw_members = raw_team.get("members", [])
    if not isinstance(raw_members, list):
        raise ValueError("team.members must be an array")
    members = []
    for item in raw_members:
        member = _object(item, "team member", {"id", "instructions"})
        member_id = _identifier(member.get("id"), "team member id")
        if member_id in {"system_events", "all"}:
            raise ValueError("team member id is reserved")
        instructions = _text(member.get("instructions", ""), "team member instructions", empty=True)
        members.append(TeamMemberSpec(member_id, instructions))
    if len({member.id for member in members}) != len(members):
        raise ValueError("team member ids must be unique")
    if members and len(members) != limits["max_agents"]:
        raise ValueError("team.members must match the exact configured population")
    problem = ProblemSpec(
        objective=objective,
        limits=ResourceLimits(**limits),
        team=TeamSpec(capabilities=capabilities, members=tuple(members)),
    )
    raw_notifications = root.get("scheduled_notifications", [])
    if not isinstance(raw_notifications, list):
        raise ValueError("scheduled_notifications must be an array")
    recipients = {member.id for member in fixed_members(problem)}
    notifications = []
    notification_ids = set()
    for item in raw_notifications:
        notification = _object(
            item, "scheduled notification", {"id", "at_minutes", "recipient", "message"}
        )
        notification_id = _identifier(notification.get("id"), "scheduled notification id")
        if notification_id in notification_ids:
            raise ValueError("scheduled notification ids must be unique")
        notification_ids.add(notification_id)
        at_minutes = notification.get("at_minutes")
        try:
            valid_time = (
                type(at_minutes) in (int, float) and math.isfinite(at_minutes) and at_minutes >= 0
            )
        except OverflowError:
            valid_time = False
        if not valid_time:
            raise ValueError("scheduled notification at_minutes must be finite and nonnegative")
        recipient = _text(notification.get("recipient"), "scheduled notification recipient")
        if recipient != "all" and recipient not in recipients:
            raise ValueError(
                "scheduled notification recipient must be 'all' or a fixed roster agent id"
            )
        message = _text(notification.get("message"), "scheduled notification message")
        if len(message) > 8000:
            raise ValueError("scheduled notification message must be at most 8000 characters")
        notifications.append(
            ScheduledNotificationSpec(notification_id, float(at_minutes), recipient, message)
        )
    return ProblemSpec(
        objective=problem.objective,
        limits=problem.limits,
        team=problem.team,
        scheduled_notifications=tuple(notifications),
    )


def validate_problem(problem: ProblemSpec) -> ProblemSpec:
    """Apply the same contract to direct Python callers before creating runtime state."""
    if not isinstance(problem, ProblemSpec):
        raise ValueError("problem must be a ProblemSpec")
    try:
        raw = json.loads(json.dumps(asdict(problem)))
    except (TypeError, ValueError) as error:
        raise ValueError("problem fields must be JSON-serializable") from error
    return parse_problem(raw)


def fixed_members(problem: ProblemSpec) -> tuple[TeamMemberSpec, ...]:
    """Resolve the complete immutable roster from validated run inputs."""
    if problem.team.members:
        return problem.team.members
    return tuple(
        TeamMemberSpec(f"agent-{index}") for index in range(1, problem.limits.max_agents + 1)
    )
