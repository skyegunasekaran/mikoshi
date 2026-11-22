"""Terminal presentation kept outside the experiment application service."""

from __future__ import annotations

import json
import os
import shlex
import sys
from dataclasses import asdict

from mikoshi.operations import format_status
from mikoshi.problem import fixed_members


def shell_quote(value):
    """Quote a suggested command argument for PowerShell or a POSIX shell."""
    value = str(value)
    return "'" + value.replace("'", "''") + "'" if os.name == "nt" else shlex.quote(value)


def format_plan(prepared):
    args = prepared.settings
    total = args.budget_usd * args.trials if args.budget_usd is not None else None
    minutes = args.wall_time_seconds / 60 if args.wall_time_seconds is not None else None
    rates = asdict(args.pricing)
    lines = [
        f"Experiment: {args.problem.objective.title}",
        f"Objective: {args.problem.objective.description}",
        f"Target: {args.repository}",
        f"Provider / model: {args.provider} / {args.model}",
        f"Team: {len(fixed_members(args.problem))} agents; {args.provider_concurrency} model request slots",
        f"Trials: {args.trials}; up to {args.trial_concurrency} at once",
        f"Spending limit per trial: {f'${args.budget_usd:g}' if args.budget_usd is not None else 'uncapped'}",
        f"Nominal total spending limit: {f'${total:g}' if total is not None else 'uncapped'}",
        f"Time limit per trial: {f'{minutes:g} minutes' if minutes is not None else 'uncapped'}",
        "Prices (USD per million tokens): "
        + ", ".join(
            f"{label} ${rates[key]:g}"
            for key, label in (
                ("input_rate", "uncached input"),
                ("cache_read_rate", "cached input"),
                ("output_rate", "output"),
                ("thinking_rate", "thinking"),
            )
        ),
        f"Token limits: {args.model_limits.actor_output_ceiling:,} output; "
        f"{args.model_limits.configured_context_window:,} context; "
        f"compaction at {args.model_limits.compaction_trigger:,} input",
        f"Commands: {'Docker (' + args.docker.image + ')' if args.enable_commands else 'disabled'}",
        f"Output: {args.output}",
        f"Scorer: {args.scorer or 'none; inspect and evaluate the retained candidate yourself'}",
    ]
    if args.provider_deployment.agent_models:
        lines.append(
            "Per-agent models: "
            + ", ".join(
                f"{peer}={model}" for peer, model in args.provider_deployment.agent_models.items()
            )
        )
    if args.model_pricing:
        for model, prices in sorted(args.model_pricing.items()):
            lines.append(
                f"Prices for {model} (USD per million tokens): "
                f"uncached input ${prices.input_rate:g}, cached input ${prices.cache_read_rate:g}, "
                f"output ${prices.output_rate:g}, thinking ${prices.thinking_rate:g}"
            )
    lines.append(
        "Each single trial edits its target in place, including uncommitted files."
        if args.trials == 1
        else "Repeated trials edit separate copies of the same starting target."
    )
    lines.append(
        "In-flight requests can exceed the spending limit; supplied prices drive accounting."
    )
    return "\n".join(lines)


class TerminalReporter:
    def __init__(self, *, json_output=False):
        self.json_output = json_output
        self._last_phase = {}

    def progress(self, packet):
        if self.json_output:
            print(json.dumps(packet, sort_keys=True), file=sys.stderr, flush=True)
            return
        trial = packet.get("trial_id")
        event = packet
        while isinstance(event.get("progress"), dict):
            event = event["progress"]
        prefix = f"[{trial}] " if trial else ""
        kind = event.get("kind")
        if kind == "command_output":
            self.command_output(event.get("command_id"), event.get("output", ""), prefix=prefix)
            return
        if kind == "process_output":
            text = event.get("text", "")
            if text:
                print(
                    f"{prefix}{event.get('stream', 'child')}: {text[:8192]}",
                    file=sys.stderr,
                    flush=True,
                )
            return
        if kind == "run_started":
            output = event["output"]
            quoted = shell_quote(output)
            message = (
                f"Output: {output}\nFollow: mikoshi watch {quoted}\nStop: mikoshi stop {quoted}"
            )
        elif kind == "phase_started":
            phase = event.get("phase")
            if self._last_phase.get(trial) == phase:
                return
            self._last_phase[trial] = phase
            message = {
                "initializing": "Preparing the team and workspace…",
                "working": "Agents are working. Ctrl-C requests a graceful stop.",
                "quiescing": "Settling in-flight requests and usage…",
                "exporting": "Saving conversations, actions, and command results…",
                "freezing": "Retaining the candidate for inspection…",
                "scoring": "Evaluating the retained candidate…",
            }.get(phase, str(phase))
        elif kind == "action_completed":
            message = f"{event.get('agent_id', 'agent')}: {event.get('action', 'action completed')}"
        elif kind == "model_request_retry_scheduled":
            message = f"Provider request delayed; retry in {event['delay_seconds']:g}s."
        elif kind in {"model_request_unavailable", "model_context_exhausted"}:
            message = f"{event.get('agent_id', 'agent')}: {event.get('error', kind)}"
        elif kind == "trial_finished":
            message = f"Trial {event['status']}: {event.get('error') or event.get('stop_reason') or 'finished'}"
        else:
            return
        print(prefix + message, file=sys.stderr, flush=True)

    def command_output(self, command_id, output, *, prefix=""):
        if self.json_output:
            print(
                json.dumps({"kind": "command_output", "command_id": command_id, "output": output}),
                file=sys.stderr,
                flush=True,
            )
        else:
            sys.stderr.write(f"{prefix}[command {command_id}] {output}")
            if output and not output.endswith("\n"):
                sys.stderr.write("\n")
            sys.stderr.flush()

    def result(self, output, value):
        if self.json_output:
            print(json.dumps(value, indent=2))
            return
        print(format_status(output, value))
        quoted = shell_quote(output)
        print(f"\nInspect: {output / 'run.json'}\nValidate evidence: mikoshi validate {quoted}")
        if value.get("candidate"):
            print(f"Candidate: {output / value['candidate']['path']}")
