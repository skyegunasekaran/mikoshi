"""Local setup checks and recorded run status, without model requests."""

from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from mikoshi.configuration import configuration_document
from mikoshi.persistence.store import WorldStore
from mikoshi.providers.usage import TokenMeter
from mikoshi.schema import RUN_FILE, read_document
from mikoshi.tools.commands import CommandRunner
from mikoshi.tools.workspace import LiveWorkspace, resolve_repository


def check_run(args):
    """Inspect startup dependencies without creating a run or executing commands."""
    identity = resolve_repository(args.repository)
    TokenMeter(args.provider_deployment.tokenizer_path)
    store = WorldStore()
    try:
        execution = None
        if args.enable_commands:
            with tempfile.TemporaryDirectory(prefix="mikoshi-check-") as temporary:
                workspace = LiveWorkspace(Path(temporary) / "workspace", store, args.repository)
                execution = CommandRunner(
                    workspace, options=args.docker, limits=args.command_limits
                ).execution_context()
    finally:
        store.close()
    return {
        "status": "checked",
        "run": configuration_document(args)["run"],
        "repository_commit": identity.commit,
        "model_limits": asdict(args.model_limits),
        "agent_models": args.provider_deployment.agent_models,
        "agent_model_limits": {
            model: asdict(limits)
            for model, limits in getattr(args, "agent_model_limits", {}).items()
        },
        "execution": execution,
        "note": (
            "Local configuration, credential presence, tokenizer, and SQLite checked. "
            "Docker daemon and image inspected when commands are enabled. "
            "No model requests or containers started; provider access, task dependencies, "
            "and container execution remain untested."
        ),
    }


def read_status(output: Path):
    """Read the atomic status snapshot without opening or modifying the database."""
    value = read_document(output / RUN_FILE, "run")
    if not {"status", "phase", "updated_at"} <= value.keys():
        raise ValueError("run.json is missing status fields")
    return value


def format_status(output: Path, value):
    lines = [
        f"Run: {output}",
        f"Recorded status: {value['status']} ({value['phase']})",
        f"Updated: {value['updated_at']}",
    ]
    if value.get("execution_status") != value["status"] and value.get("execution_status"):
        lines.append(f"Execution: {value['execution_status']}")
    if value["status"] == "running":
        lines.append("Recorded snapshot; watch the launcher to confirm it is still advancing.")
    budget = value.get("budget") or {}
    if budget:
        spent = budget.get("accounted_usd")
        limit = budget.get("limit_usd")
        lines.append(
            f"Accounted spend: {f'${spent:.6f}' if spent is not None else 'unknown'} / "
            f"{f'${limit:g}' if limit is not None else 'uncapped'}"
        )
        if budget.get("unreported_trials"):
            lines.append(f"Usage reports missing for {budget['unreported_trials']} trial(s).")
        if "elapsed_seconds" in budget:
            elapsed = budget["elapsed_seconds"]
            seconds = budget.get("wall_limit_seconds")
            lines.append(
                f"Elapsed at update: {elapsed:.1f}s / "
                f"{f'{seconds:g}s' if seconds is not None else 'uncapped'}"
            )
        lines.append(
            f"Requests: {budget.get('pending_requests', 0)} pending, "
            f"{budget.get('unknown_usage_requests', 0)} with unknown usage"
        )
    if value.get("progress"):
        progress = value["progress"]
        summary = {
            key: progress[key]
            for key in ("kind", "agent_id", "request_name", "action", "command_id", "step", "error")
            if key in progress
        }
        lines.append("Latest progress: " + json.dumps(summary, sort_keys=True)[:2000])
    for key in ("stop_reason", "error", "manifest"):
        if value.get(key) is not None:
            lines.append(f"{key}: {value[key]}")
    explanation = {
        "all_agents_yielded": "All peers yielded with nothing queued; inspect the candidate for task quality.",
        "budget_exhausted": "The spending limit stopped admission; in-flight usage is included when reported.",
        "wall_time_exhausted": "The trial reached its time limit.",
        "context_window_exhausted": "The configured model context could not fit the next request.",
        "stop_requested": "An operator requested a graceful stop.",
        "command_cleanup_unconfirmed": "Command cleanup could not be confirmed; inspect execution evidence.",
        "security_boundary_failure": "Execution stopped at an isolation boundary; inspect the recorded error.",
    }.get(value.get("stop_reason"))
    if explanation:
        lines.append(explanation)
    evaluation = value.get("evaluation") or {}
    if evaluation:
        lines.append(f"Evaluation: {evaluation['status']}")
        for name, score in evaluation.get("scores", {}).items():
            lines.append(f"  {name}: {score['value']}")
        if evaluation.get("error"):
            lines.append(f"Evaluation error: {evaluation['error']}")
    for trial in value.get("trials", []):
        detail = trial.get("stop_reason") or trial.get("phase") or trial.get("error") or ""
        lines.append(f"{trial['id']}: {trial['status']} {detail}".rstrip())
    for peer, event in sorted(value.get("agents", {}).items()):
        lines.append(f"{peer}: {event.get('kind', 'unknown')} ({event.get('recorded_at', '')})")
    return "\n".join(lines)


def watch_status(output: Path, *, interval_seconds=2.0, json_output=False):
    """Follow recorded snapshots until terminal; observing never starts or stops work."""
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive and finite")
    previous = None
    while True:
        value = read_status(output)
        if value != previous:
            print(json.dumps(value) if json_output else format_status(output, value), flush=True)
            previous = value
        if value["status"] != "running":
            return
        time.sleep(interval_seconds)
