"""Prepare peer-visible observations and decision evidence before provider adaptation."""

import json
import uuid

from mikoshi.decisions import DecisionRequest
from mikoshi.runtime.budgets import budget_context, budget_warnings


def peer_request(
    *,
    state,
    agent,
    tick,
    packet,
    actions,
    notifications,
    conversations,
    protocol,
    disable_yield,
    budget,
):
    actor = agent.id
    if budget is not None:
        # Stable limits return with the full context after compaction.
        packet["run_resources"] = budget_context(budget)
    warnings = budget_warnings(budget)
    if warnings:
        packet["budget_warnings"] = warnings
    digest, cursor, digest_sources = notifications.digest(actor)
    packet["inbox"] = digest
    previous = conversations.last_state(actor) or {}
    changes_since = previous.get("changes", {}).get("through_tick", 0)
    packet["changes"] = notifications.changes(actor, changes_since, tick)
    instructions = (
        protocol.environment(disable_yield=disable_yield) + "\n\nYou are " + agent.name + "."
    )
    if agent.instructions:
        # Operator-authored private instructions belong to the stable system
        # contract, including every provider compaction continuation.
        instructions += "\n\n" + agent.instructions
    # Audit history stays in SQLite; only the last exchange crosses the provider boundary.
    # Scoped instructions live in conversation history. After compaction the
    # startup chain/catalog return, and matching scopes become eligible again.
    if extra := protocol.peer_context(actor):
        packet["protocol_context"] = dict(extra)
    compaction_packet = {
        **packet,
        **(
            {
                "repository_guidance": {
                    key: value
                    for key, value in packet["repository_guidance"].items()
                    if key != "activity"
                }
            }
            if "repository_guidance" in packet
            else {}
        ),
    }
    # Every turn carries a complete dynamic observation. Stable startup guidance
    # already lives in the conversation and returns in full after compaction.
    observed = dict(packet)
    for key in ("run_resources", "commands", "execution_access", "protocol_context"):
        if key in observed and previous.get(key) == observed[key]:
            observed.pop(key)
    guidance = packet.get("repository_guidance")
    if guidance and previous.get("repository_guidance"):
        held = previous["repository_guidance"]
        current = {
            key: value
            for key, value in guidance.items()
            if key == "activity" or held.get(key) != value
        }
        if current:
            observed["repository_guidance"] = current
        else:
            observed.pop("repository_guidance")
    full_state = json.dumps(packet, sort_keys=True)
    observation = json.dumps(observed, sort_keys=True)
    return DecisionRequest(
        name=f"peer session step {state.steps + 1}",
        instructions=instructions,
        input_text=full_state,
        compaction_input_text=json.dumps(compaction_packet, sort_keys=True),
        model_input_text=observation,
        conversation_id=actor,
        schema={"oneOf": actions},
        stage="actor",
        metadata={
            "decision_id": uuid.uuid4().hex,
            "agent_id": actor,
            "activation_id": state.activation_id,
            "activation_step": state.steps + 1,
            "lifecycle_kind": "peer_session",
            "notification_cursor": cursor,
            "observation_notification_sources": digest_sources,
            "context_characters": {
                "internal_state": len(full_state),
                "current_observation": len(observation),
                "instructions": len(instructions),
                "action_schemas": len(json.dumps(actions)),
            },
        },
    )
