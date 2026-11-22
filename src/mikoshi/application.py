"""Prepare and execute experiments independently of terminal argument handling."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import platform
import sys
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from mikoshi import RUNTIME_ASSET_NAMES, RUNTIME_ASSET_SUFFIXES
from mikoshi.configuration import configuration_document
from mikoshi.decisions import DecisionProvider
from mikoshi.evaluation import evaluate_run, validate_scorer
from mikoshi.extensions import installed_extension
from mikoshi.isolation.docker import DockerSettings
from mikoshi.persistence.artifacts import freeze_candidate
from mikoshi.persistence.exports import (
    checkout_metadata,
    export_state,
)
from mikoshi.persistence.store import WorldStore
from mikoshi.problem import (
    ProblemSpec,
    fixed_members,
    validate_problem,
)
from mikoshi.providers.base import DEFAULT_PROVIDER_TIMEOUT_SECONDS
from mikoshi.providers.base import close_provider as _close_provider
from mikoshi.providers.limits import resolve_model_limits
from mikoshi.providers.registry import build_decision_provider, provider_adapter
from mikoshi.runtime.budgets import (
    DEFAULT_AGENT_LIMITS,
    AgentLimits,
    ExecutionBudget,
    limits_for,
)
from mikoshi.runtime.lifecycle import LifecycleOrchestrator
from mikoshi.runtime.notifications import with_time_reminders
from mikoshi.runtime.run_lifecycle import RunLifecycle, resolve_execution_status
from mikoshi.schema import (
    RUN_FILE,
    read_document,
    update_run,
)
from mikoshi.team_protocol import TeamProtocol, load_protocol
from mikoshi.tools.commands import (
    CommandLimits,
    CommandRunner,
)
from mikoshi.tools.workspace import (
    LiveWorkspace,
    RepositoryIdentity,
    WorkspaceLimits,
    resolve_repository,
)

PACKAGE_ROOT = Path(__file__).parent


def state_directory() -> Path:
    """Default home for saved experiments; resolving it does not create it."""
    configured = os.environ.get("XDG_STATE_HOME")
    if configured:
        root = Path(configured)
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        root = Path.home() / ".local" / "state"
    return root.expanduser() / "mikoshi"


def build_from_problem(
    store: WorldStore,
    provider: DecisionProvider,
    problem: ProblemSpec,
    workspace_root: Path,
    repository_identity: RepositoryIdentity,
    enable_commands: bool = True,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    docker: DockerSettings | None = None,
    disable_yield: bool = False,
    command_limits: CommandLimits | None = None,
    team_protocol: TeamProtocol | None = None,
    budget=None,
    agent_limits: AgentLimits | None = None,
    command_output_callback=None,
) -> LifecycleOrchestrator:
    """Start from an objective and configured team."""
    from mikoshi.runtime.organization import bootstrap

    problem = validate_problem(problem)
    if budget is None:
        budget = getattr(provider, "budget", None)
    problem = with_time_reminders(problem, getattr(budget, "seconds", None))
    if problem.scheduled_notifications and budget is None:
        raise ValueError("scheduled notifications require an execution budget clock")
    bootstrap(store, problem)
    agent_limits = agent_limits or limits_for(provider)
    workspace = LiveWorkspace(
        workspace_root,
        store,
        Path(repository_identity.source),
        limits=replace(WorkspaceLimits(), max_read_chars=agent_limits.read_chars),
    )
    workspace.initialize(repository_identity)
    command_runner = (
        CommandRunner(
            workspace,
            limits=command_limits,
            options=docker,
            output_callback=command_output_callback,
        )
        if enable_commands
        else None
    )
    return LifecycleOrchestrator(
        store,
        provider,
        workspace,
        command_runner=command_runner,
        progress_callback=progress_callback,
        disable_yield=disable_yield,
        team_protocol=team_protocol,
        budget=budget,
        agent_limits=agent_limits,
    )


@dataclass(frozen=True)
class PreparedRun:
    settings: Any
    repository_identity: RepositoryIdentity
    team_protocol: TeamProtocol


def prepare_run(settings, *, require_credentials=True) -> PreparedRun:
    """Resolve intent without starting a model request, container, or run directory."""
    args = copy.copy(settings)
    if args.wall_time_seconds is None and not args.no_time_limit:
        args.wall_time_seconds = 1800.0
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = state_directory() / "runs" / f"{stamp}-{uuid.uuid4().hex[:8]}"
    args.output = args.output.expanduser().resolve()
    args.repository = args.repository.expanduser().resolve()
    if (
        args.output == args.repository
        or args.repository in args.output.parents
        or args.output in args.repository.parents
    ):
        raise ValueError("run output must be separate from the repository")
    try:
        adapter = provider_adapter(args.provider)
        if require_credentials:
            adapter.validate_settings(args)
        else:
            adapter.validate_configuration(args)
        team_protocol = load_protocol(args.team_protocol)
        args.model_limits = resolve_model_limits(
            args.provider,
            args.model,
            configured_context_window=args.context_window_tokens,
            actor_output_ceiling=args.max_output_tokens,
            compaction_trigger=args.compaction_input_tokens,
        )
        adapter.deployment_options(args.provider_deployment, args.model_limits)
        problem = args.problem
        peer_ids = {member.id for member in fixed_members(problem)}
        agent_count = len(peer_ids)
        agent_models = args.provider_deployment.agent_models
        if agent_models and set(agent_models) != peer_ids:
            raise ValueError("deployment.provider.agent_models must match the complete peer roster")
        if missing := set(agent_models.values()) - args.model_pricing.keys():
            raise ValueError(
                f"pricing.models requires explicit rates for: {', '.join(sorted(missing))}"
            )
        args.agent_model_limits = {
            model: resolve_model_limits(
                args.provider,
                model,
                configured_context_window=args.context_window_tokens,
                actor_output_ceiling=args.max_output_tokens,
                compaction_trigger=args.compaction_input_tokens,
            )
            for model in dict.fromkeys(agent_models.values())
        }
        repository_identity = resolve_repository(args.repository)
        if args.scorer is not None:
            args.scorer = validate_scorer(args.scorer, args.repository, args.output)
    except (OSError, ValueError):
        raise
    args.provider_concurrency = (
        args.provider_concurrency or args.provider_deployment.concurrency or agent_count
    )
    for name in ("budget_usd", "wall_time_seconds"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if args.no_budget_cap and args.wall_time_seconds is None and not args.no_time_limit:
        raise ValueError("--no-budget-cap requires --wall-time-seconds or --no-time-limit")
    if args.budget_usd is None and not args.no_budget_cap:
        raise ValueError("--budget-usd or --no-budget-cap is required")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("run output must be a new or empty directory")
    return PreparedRun(args, repository_identity, team_protocol)


def execute_run(prepared: PreparedRun, *, progress_callback=None, command_output_callback=None):
    """Execute already-resolved intent; return the terminal record without printing it."""
    args = prepared.settings
    repository_identity = prepared.repository_identity
    team_protocol = prepared.team_protocol
    if args.trials > 1:
        from mikoshi.experiments import run_experiment

        run_experiment(args, progress_callback=progress_callback)
        return read_document(args.output / RUN_FILE)
    started_at = datetime.now(timezone.utc).isoformat()
    lifecycle = RunLifecycle(args.output, started_at, progress_callback=progress_callback)
    try:
        update_run(args.output, input=configuration_document(args))
        lifecycle.progress({"kind": "run_started", "output": str(args.output)})
        args.execution_budget = ExecutionBudget(
            args.output / ".internal/budget.json",
            args.budget_usd,
            args.wall_time_seconds,
            **asdict(args.pricing),
            start_paused=True,
            allow_unlimited=args.no_budget_cap and args.no_time_limit,
            model_prices=args.model_pricing,
        )
        lifecycle.budget = args.execution_budget
        problem = with_time_reminders(args.problem, args.wall_time_seconds)
        update_run(
            args.output,
            input=configuration_document(args),
            resolved={
                "problem": asdict(
                    replace(problem, team=replace(problem.team, members=fixed_members(problem)))
                ),
                "model_limits": asdict(args.model_limits),
                "agent_model_limits": {
                    model: asdict(limits) for model, limits in args.agent_model_limits.items()
                },
                "provider_concurrency": args.provider_concurrency,
                "provider_timeout_seconds": DEFAULT_PROVIDER_TIMEOUT_SECONDS,
                "agent_limits": asdict(DEFAULT_AGENT_LIMITS),
                "team_protocol": args.team_protocol,
                "command_limits": asdict(args.command_limits),
                "workspace_limits": asdict(
                    WorkspaceLimits(max_read_chars=DEFAULT_AGENT_LIMITS.read_chars)
                ),
            },
        )
    except BaseException as error:
        lifecycle.fail(error)
        raise
    store = None
    provider = None
    provider_closed = False

    def close_provider_once():
        nonlocal provider_closed
        if provider is not None and not provider_closed:
            # Close can settle results and emit evidence/progress. It must finish
            # before terminal exports, including on failure paths.
            provider_closed = True
            _close_provider(provider)

    try:
        extensions = {
            group: source
            for group, name in (
                ("mikoshi.providers", args.provider),
                ("mikoshi.team_protocols", args.team_protocol),
            )
            if (source := installed_extension(group, name)) is not None
        }
        update_run(
            args.output,
            provenance={"checkout": checkout_metadata(args.repository), "extensions": extensions},
        )
        provider = build_decision_provider(
            args,
            lifecycle.progress,
        )
        if callable(getattr(provider, "set_evidence_callback", None)):
            provider.set_evidence_callback(lifecycle.records.provider_event)
        (args.output / ".internal").mkdir(exist_ok=True)
        store = WorldStore(args.output / ".internal/run.sqlite")
        lifecycle.records.attach(store)
    except BaseException as error:
        try:
            close_provider_once()
        finally:
            if store is not None:
                store.close()
            lifecycle.fail(error)
        raise
    try:
        lifecycle.phase("initializing")
        runtime = build_from_problem(
            store,
            provider,
            problem,
            args.output / "workspace",
            repository_identity,
            args.enable_commands,
            progress_callback=lifecycle.progress,
            docker=args.docker,
            disable_yield=args.disable_yield,
            command_limits=args.command_limits,
            team_protocol=team_protocol,
            budget=args.execution_budget,
            command_output_callback=command_output_callback,
        )
        if runtime._command_runner is not None:
            update_run(
                args.output,
                provenance={
                    **read_document(args.output / RUN_FILE)["provenance"],
                    "execution": runtime._command_runner.execution_context(),
                },
            )
        lifecycle.cancel_commands = runtime.cancel_commands
        args.execution_budget.start()
        lifecycle.phase("working")
        summary = runtime.run(stop_requested=lifecycle.stop_requested)
        # Work has stopped, so any stop request has been honored. Disarm the forced
        # interrupt so it cannot abort the final exports (a second signal still can).
        lifecycle.stop_monitor.close()
        args.execution_budget.pause()
        args.execution_budget.save()
        lifecycle.execution_stop_reason = runtime.stop_reason
        lifecycle.phase("quiescing")
        close_provider_once()
        args.execution_budget.save()
        lifecycle.phase("exporting")
        runtime.organization.pages.export()
        lifecycle.records.append("state", export_state(store))
        _export_run(args, summary, provider, repository_identity, stop_reason=runtime.stop_reason)
        if runtime.stop_reason in {"security_boundary_failure", "command_cleanup_unconfirmed"}:
            from mikoshi.evaluation import record_evaluation_failure

            evaluation = record_evaluation_failure(
                args.output, "UnsafeExecution", runtime.stop_reason
            )
        else:
            lifecycle.phase("freezing")
            freeze_candidate(args.repository, args.output)
            lifecycle.phase("scoring")
            evaluation = evaluate_run(args.output, args.scorer)
        lifecycle.complete(runtime.stop_reason, evaluation["status"])
        return read_document(args.output / RUN_FILE)
    except BaseException as error:
        args.execution_budget.pause()
        try:
            close_provider_once()
        finally:
            lifecycle.fail(error)
        raise
    finally:
        try:
            args.execution_budget.pause()
            close_provider_once()
        finally:
            store.close()


def _export_run(args, summary, provider, repository_identity, *, stop_reason):
    current = read_document(args.output / RUN_FILE, "run")
    update_run(
        args.output,
        summary=asdict(summary),
        stop_reason=stop_reason,
        budget=args.execution_budget.snapshot(),
        provider_metrics=getattr(provider, "metrics", None),
        execution_status=resolve_execution_status(stop_reason),
        provenance={
            **current["provenance"],
            "repository_identity": asdict(repository_identity),
            **provider_adapter(args.provider).provenance(provider),
            "system": _system_metadata(),
            "source_tree_sha256": _source_tree_sha256(),
            "quality_policy": "posthoc_only",
        },
    )


def _system_metadata() -> Mapping[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def _source_tree_sha256() -> str:
    """Fingerprint runtime code and prompts, excluding run inputs."""
    digest = hashlib.sha256()
    paths = sorted(
        (
            path
            for path in PACKAGE_ROOT.rglob("*")
            if path.is_file()
            and (path.suffix in RUNTIME_ASSET_SUFFIXES or path.name in RUNTIME_ASSET_NAMES)
        ),
        key=lambda path: path.relative_to(PACKAGE_ROOT).as_posix(),
    )
    for path in paths:
        digest.update(path.relative_to(PACKAGE_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
