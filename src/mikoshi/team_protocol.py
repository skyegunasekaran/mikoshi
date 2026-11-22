"""The team-facing contract. Protocols select tools and frame observations."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from mikoshi.contracts import action_schemas
from mikoshi.extensions import extension_names, load_extension
from mikoshi.prompts import worker_environment


@dataclass(frozen=True)
class TeamTool:
    """A canonical action schema paired with its (activation, decision) handler.

    The handler returns (activation_finished, observation).
    """

    schema: Mapping[str, Any]
    execute: Callable

    @property
    def name(self):
        return self.schema["properties"]["action"]["enum"][0]


class TeamProtocol:
    """Default fixed peers. A fresh protocol instance belongs to each team."""

    def tools(self, available: tuple[TeamTool, ...], *, peer: str):
        """Select, reorder or extend the capability-checked tools for this turn."""
        return available

    def environment(self, *, disable_yield: bool) -> str:
        return worker_environment(disable_yield=disable_yield)

    def peer_context(self, peer: str) -> Mapping[str, Any]:
        """Additional peer-visible state, shared by ordinary and restored observations."""
        return {}


def bind_tools(tools):
    by_name = {}
    for tool in tools:
        if not isinstance(tool, TeamTool) or not callable(tool.execute) or "oneOf" in tool.schema:
            raise ValueError("each team tool requires one action schema and a callable handler")
        (name,) = action_schemas(tool.schema)
        if name in by_name:
            raise ValueError(f"duplicate team tool {name!r}")
        by_name[name] = tool
    if not by_name:
        raise ValueError("a team protocol must offer at least one tool")
    return by_name


def protocol_names():
    return extension_names("mikoshi.team_protocols", {"peers": TeamProtocol})


def load_protocol(name):
    factory = load_extension("mikoshi.team_protocols", name, {"peers": TeamProtocol})
    protocol = factory()
    if not isinstance(protocol, TeamProtocol):
        raise ValueError(f"team protocol {name!r} must produce a TeamProtocol")
    return protocol
