from __future__ import annotations

import json
import threading
import time
from concurrent.futures import wait
from dataclasses import replace
from typing import (
    Any,
    Callable,
    Mapping,
    Sequence,
)

from mikoshi.contracts import (
    BATCH_ACTION,
    BATCH_GUIDANCE,
    ActivationSummary,
    RunSummary,
    action_schema,
    validate_tool_turn,
)
from mikoshi.decisions import (
    DecisionProvider,
    DecisionRequest,
    ModelDecisionError,
    dispatcher_for,
)
from mikoshi.persistence.exports import write_json
from mikoshi.persistence.store import WorldStore
from mikoshi.runtime.budgets import (
    CHECK_BUDGET_SCHEMA,
    AgentLimits,
    BudgetExhausted,
    ContextWindowExhausted,
    budget_request,
    check_budget,
    limits_for,
)
from mikoshi.runtime.conversations import ConversationHistory, native_tool_calls
from mikoshi.runtime.notifications import (
    Notifications,
    ScheduledNotifications,
)
from mikoshi.runtime.peer_turn import peer_request
from mikoshi.runtime.repository_guidance import (
    GuidanceLedger,
    activity_guidance,
    load_repository_guidance,
    root_guidance,
    warning_documents,
)
from mikoshi.runtime.scheduling import ProviderRetryPolicy, WaveSchedule, WorkspaceActivation
from mikoshi.team_protocol import TeamProtocol, TeamTool, bind_tools
from mikoshi.tools.commands import COMPLETED_STATUSES, CommandLimits, CommandRunner
from mikoshi.tools.output import READ_OUTPUT_SCHEMA, read_output
from mikoshi.tools.shared_pages import SharedPages
from mikoshi.tools.shared_work_tools import SharedWorkTools
from mikoshi.tools.workspace import LiveWorkspace, WorkspaceAccessError

PROVIDER_RETRY_DELAYS = (1.0, 2.0, 4.0)
PROVIDER_RATE_LIMIT_DELAY = 60.0


class SecurityBoundaryFailure(RuntimeError):
    """Execution integrity is uncertain; do not continue model work."""

    stop_reason = "security_boundary_failure"


class CommandCleanupUnconfirmed(SecurityBoundaryFailure):
    """Command children could not be confirmed terminated."""

    stop_reason = "command_cleanup_unconfirmed"


class _AwaitTool(Exception):
    """This activation is waiting for its own background tool operation."""


class LifecycleOrchestrator:
    """Runs problem-defined AI agents as durable, cooperatively scheduled sessions."""

    def __init__(
        self,
        store: WorldStore,
        provider: DecisionProvider,
        workspace: LiveWorkspace,
        command_runner: CommandRunner | None = None,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        disable_yield: bool = False,
        team_protocol: TeamProtocol | None = None,
        budget=None,
        agent_limits: AgentLimits | None = None,
    ) -> None:
        from mikoshi.runtime.organization import Organization, require_current_organization

        require_current_organization(store)
        if store.execution_quarantined:
            raise SecurityBoundaryFailure(
                "execution is quarantined after an integrity failure; explicit reconciliation is required"
            )
        if list((workspace.control_root / "command_execution").glob("*.json")):
            raise SecurityBoundaryFailure("command execution has an unresolved durable outcome")
        self._store = store
        if type(disable_yield) is not bool:
            raise ValueError("disable_yield must be boolean")
        self._disable_yield = disable_yield
        self._protocol = TeamProtocol() if team_protocol is None else team_protocol
        self._budget = budget if budget is not None else getattr(provider, "budget", None)
        self._provider = provider
        self._dispatcher = dispatcher_for(provider)
        self._command_jobs = {}
        self._limits = agent_limits or limits_for(provider)
        self._workspace = workspace
        self._guidance_ledger = GuidanceLedger(store)
        self._stop_requested = None
        if command_runner is not None and command_runner.workspace is not workspace:
            raise ValueError("command runner must use the lifecycle workspace")
        self._command_runner = command_runner
        self._progress_callback = progress_callback
        self._pages = SharedPages(store, workspace)
        self._pages.seed()
        self._notifications = Notifications(self._store)
        self._organization = Organization(
            store,
            self._pages,
            budget=self._budget,
        )
        self._scheduled = ScheduledNotifications(store, self._organization, self._budget)
        self._conversations = ConversationHistory(store)
        self._shared_work = SharedWorkTools(self, prepare_action=self._prepare_repository_action)
        self._stopped = False
        self.stop_reason = None

    @property
    def store(self):
        return self._store

    def command_statuses(self):
        """Snapshot executor states without exposing command jobs to inspection tools."""
        return {
            command["id"]: "awaiting_recording"
            if future.done()
            else "execution_in_progress"
            if future.running()
            else "queued"
            for future, _, _, command in self._command_jobs.values()
        }

    @property
    def workspace(self) -> LiveWorkspace:
        return self._workspace

    @property
    def organization(self):
        """Host interface for inspecting control state."""
        return self._organization

    def _emit_progress(self, kind: str, **details: Any) -> None:
        if self._progress_callback is None:
            return
        try:
            self._progress_callback({"kind": kind, **details})
        except Exception as error:
            # Observability must never change lifecycle behavior.
            self._store.record_observer_error(f"progress {kind}", error)
            return

    def cancel_commands(self):
        """Signal active command jobs without consulting durable state."""
        for _, cancel, _, _ in list(self._command_jobs.values()):
            cancel.set()

    def run(
        self,
        *,
        stop_requested: Callable[[], str | None] | None = None,
    ) -> RunSummary:
        if self._store.execution_quarantined:
            raise SecurityBoundaryFailure(
                "execution is quarantined; explicit reconciliation is required before continuing"
            )
        self._stop_requested = stop_requested
        self.stop_reason = None
        summaries = []
        causal_waves = 0
        while True:
            reason = stop_requested() if stop_requested is not None else None
            if reason:
                self.stop_reason = reason
                break
            budget = self._budget
            if budget is not None:
                try:
                    budget.check()
                except BudgetExhausted as error:
                    self.stop_reason = str(error)
                    break
            if self._scheduled.schedule:
                self._scheduled.release_due()
            wave_id = causal_waves + 1
            wave = self._claim_wave(self._concurrency_limit(), wave_id)
            if not wave:
                self.stop_reason = self.stop_reason or self._idle_stop_reason()
                break
            self._emit_progress(
                "wave_started",
                wave=wave_id,
                activations=len(wave),
                agents=[state.trigger.agent_id for state in wave],
            )
            causal_waves += 1
            wave_summaries = self._run_wave(wave)
            summaries.extend(wave_summaries)
            if self._stopped:
                break
            self._emit_progress(
                "wave_completed",
                wave=causal_waves,
                activations=len(wave_summaries),
                remaining_triggers=self._store.queued_trigger_count(),
            )
        return self._summary(summaries, causal_waves)

    def _idle_stop_reason(self):
        if self._store.queued_trigger_count():
            return "organization_resource_blocked"
        return "all_agents_yielded"

    def _claim_wave(
        self,
        limit: int,
        causal_wave_id: int,
        excluded_agent_ids: Sequence[str] = (),
    ) -> Sequence[WorkspaceActivation]:
        claimed: list[WorkspaceActivation] = []
        active_agents: set[str] = set(excluded_agent_ids)
        for _ in range(limit):
            trigger = self._store.claim_next_trigger(active_agents)
            if trigger is None:
                break
            activation_id = self._store.start_activation(trigger, causal_wave_id)
            try:
                self._organization.begin(activation_id, trigger)
                state = WorkspaceActivation(activation_id, trigger)
            except Exception as error:
                self._store.finish_activation(
                    activation_id,
                    trigger.id,
                    "invalid_trigger",
                    0,
                    f"{type(error).__name__}: {error}",
                )
                continue
            claimed.append(state)
            active_agents.add(trigger.agent_id)
        return tuple(claimed)

    def _run_wave(
        self,
        wave: Sequence[WorkspaceActivation],
    ) -> Sequence[ActivationSummary]:
        """Advance sessions in phases; only this thread mutates durable state."""
        schedule = WaveSchedule(list(wave))
        try:
            while schedule.active:
                # Harvest before filling slots: finished siblings never wait for a slow call.
                self._receive_wave_results(schedule, timeout_seconds=0)
                self._collect_command_jobs()
                if schedule.stop_reason:
                    break
                self._check_wave_budget(schedule)
                if schedule.stop_reason:
                    break
                if not schedule.admission_stop:
                    self._scheduled.release_due()
                self._apply_ready_decisions(schedule)
                if schedule.stop_reason:
                    break
                schedule.stop_reason = self._stop_requested() if self._stop_requested else None
                if schedule.stop_reason:
                    break
                self._admit_wave_sessions(schedule)
                self._dispatch_wave_requests(schedule)
                self._await_wave_progress(schedule)
                if schedule.stop_reason:
                    break
        except SecurityBoundaryFailure as error:
            schedule.stop_reason = error.stop_reason
            self._quarantine()
        except BudgetExhausted as error:
            schedule.stop_reason = str(error)
        except Exception as error:
            schedule.stop_reason = f"runtime_error: {type(error).__name__}: {error}"
        finally:
            self._settle_wave(schedule)
        if schedule.active:
            self._stop(schedule.stop_reason or "provider_unavailable")
        elif schedule.stop_reason:
            self.stop_reason = schedule.stop_reason
        return tuple(sorted(schedule.summaries, key=lambda item: item.id))

    def _check_wave_budget(self, schedule: WaveSchedule) -> None:
        if self._budget is None:
            return
        try:
            self._budget.check()
        except BudgetExhausted as error:
            if str(error) == "budget_exhausted":
                schedule.block_admission(str(error))
            unsettled_work = (
                schedule.in_flight
                or self._command_jobs
                or any(state.pending_decision is not None for state in schedule.active)
            )
            if str(error) != "budget_exhausted" or not unsettled_work:
                schedule.stop_reason = str(error)

    def _receive_wave_results(
        self, schedule: WaveSchedule, *, timeout_seconds: float, suspending: bool = False
    ) -> None:
        if not schedule.in_flight:
            return
        ready = self._dispatcher.collect_ready(
            tuple(schedule.in_flight), timeout_seconds=timeout_seconds
        )
        retry_policy = ProviderRetryPolicy(PROVIDER_RETRY_DELAYS, PROVIDER_RATE_LIMIT_DELAY)
        for handle, result in ready.items():
            state = schedule.in_flight.pop(handle)
            if isinstance(result, BudgetExhausted):
                # Context exhaustion arrives as ContextWindowExhausted, a provider failure.
                if str(result) == "budget_exhausted" and not suspending:
                    schedule.block_admission(str(result))
                else:
                    schedule.stop_reason = schedule.stop_reason or str(result)
                continue
            self._dispatcher.acknowledge_result(handle)
            state.pending_handle = None
            if isinstance(result, Exception) and not isinstance(result, ModelDecisionError):
                self._record_provider_failure(schedule, state, result, retry_policy, suspending)
            else:
                state.provider_failures = 0
                self._acknowledge_repository_guidance(state, state.pending_request, result)
                schedule.retry_ready_at.pop(state.activation_id, None)
                self._notifications.acknowledge(
                    state.trigger.agent_id,
                    state.pending_request.metadata["notification_cursor"],
                )
                state.pending_decision = result
            self._emit_progress(
                "decision_returned",
                decision_id=state.pending_request.metadata.get("decision_id"),
                request_id=getattr(result, "request_id", None),
                activation_id=state.activation_id,
                agent_id=state.trigger.agent_id,
                handle=handle,
                failed=isinstance(result, Exception),
            )

    def _record_provider_failure(
        self,
        schedule: WaveSchedule,
        state: WorkspaceActivation,
        error: Exception,
        retry_policy: ProviderRetryPolicy,
        suspending: bool,
    ) -> None:
        state.provider_failures += 1
        delay = None if suspending else retry_policy.delay_for(error, state.provider_failures)
        details = {
            "decision_id": state.pending_request.metadata.get("decision_id"),
            "activation_id": state.activation_id,
            "agent_id": state.trigger.agent_id,
            "error": f"{type(error).__name__}: {error}",
        }
        if delay is not None:
            schedule.retry_ready_at[state.activation_id] = time.monotonic() + delay
            self._emit_progress(
                "model_request_retry_scheduled",
                **details,
                attempt=state.provider_failures + 1,
                delay_seconds=delay,
            )
            return
        schedule.unavailable.add(state.activation_id)
        if isinstance(error, ContextWindowExhausted):
            schedule.context_limited.add(state.activation_id)
        self._emit_progress(
            "model_context_exhausted"
            if isinstance(error, ContextWindowExhausted)
            else "model_request_unavailable",
            **details,
            attempt=state.provider_failures,
            status_code=getattr(error, "status_code", None),
        )

    def _apply_ready_decisions(self, schedule: WaveSchedule) -> None:
        for state in list(schedule.active):
            if (
                state.pending_decision is None
                or state.pending_handle is not None
                or state.activation_id in self._command_jobs
            ):
                continue
            request = state.pending_request
            decision = state.pending_decision
            state.steps += 1
            self._store.set_activation_context(
                state.activation_id,
                state.pending_decision_round or schedule.decision_round,
                state.steps,
            )
            observation = None
            try:
                if isinstance(decision, Exception):
                    raise decision
                terminal, observation = self._apply_tool_turn(state, decision, request.schema)
            except _AwaitTool:
                state.steps -= 1
                continue
            except SecurityBoundaryFailure as error:
                self._quarantine()
                self._record_workspace_failure(state, error.stop_reason, "runtime", error)
                state.steps -= 1
                schedule.stop_reason = error.stop_reason
                return
            except BudgetExhausted as error:
                state.steps -= 1
                schedule.stop_reason = str(error)
                return
            except Exception as error:
                observation = model_error(error)
                terminal = False
            else:
                self._emit_progress(
                    "action_completed",
                    activation_id=state.activation_id,
                    agent_id=state.trigger.agent_id,
                    action=str(decision.get("action", "unknown")),
                    step=state.steps,
                    terminal=terminal,
                )
                if terminal:
                    schedule.summaries.append(
                        ActivationSummary(
                            state.activation_id,
                            state.trigger.agent_id,
                            state.trigger.id,
                            "yielded",
                            state.steps,
                            None,
                        )
                    )
            self._conversations.append_exchange(
                state.trigger.agent_id,
                f"{state.activation_id}:{state.steps}",
                request,
                decision,
                observation,
            )
            self._store.set_action_context(state.activation_id)
            schedule.complete_turn(state, terminal=terminal)

    def _admit_wave_sessions(self, schedule: WaveSchedule) -> None:
        from mikoshi.runtime.organization import peer_count

        capacity = peer_count(self._store) - len(schedule.active)
        if capacity > 0 and not schedule.admission_stop:
            schedule.active.extend(
                self._claim_wave(
                    capacity,
                    self._store.latest_causal_wave_id(),
                    [state.trigger.agent_id for state in schedule.active],
                )
            )

    def _dispatch_wave_requests(self, schedule: WaveSchedule) -> None:
        for state in schedule.request_candidates(
            self._concurrency_limit(), self._command_jobs, time.monotonic()
        ):
            schedule.decision_round += 1
            schedule.retry_ready_at.pop(state.activation_id, None)
            self._store.set_activation_context(
                state.activation_id, schedule.decision_round, state.steps + 1
            )
            request = state.pending_request or self._workspace_request(state)
            state.pending_request = request
            state.pending_decision_round = state.pending_decision_round or schedule.decision_round
            handle = self._dispatcher.submit_request(request)
            schedule.track_request(handle, state)
            self._emit_progress(
                "decision_dispatched",
                decision_id=request.metadata.get("decision_id"),
                activation_id=state.activation_id,
                agent_id=state.trigger.agent_id,
                stage=request.stage,
                handle=handle,
                in_flight=len(schedule.in_flight),
                batch_size=len(schedule.in_flight),
            )

    def _await_wave_progress(self, schedule: WaveSchedule) -> None:
        if schedule.in_flight:
            self._receive_wave_results(schedule, timeout_seconds=0.05)
        elif self._command_jobs:
            # Processes supervise their own deadline; this wait never holds model results.
            threading.Event().wait(0.01)
        elif schedule.admission_stop:
            schedule.stop_reason = schedule.admission_stop
        elif schedule.retry_ready_at:
            # Backoff belongs to the affected agent; no model slot is held.
            threading.Event().wait(schedule.retry_wait(time.monotonic()))
        else:
            schedule.stop_reason = schedule.idle_stop_reason()

    def _settle_wave(self, schedule: WaveSchedule) -> None:
        """Settle provider accounting, then always confirm command cleanup."""
        try:
            if schedule.in_flight:
                self._dispatcher.cancel_pending()
                self._receive_wave_results(schedule, timeout_seconds=0, suspending=True)
        except Exception as error:
            schedule.stop_reason = (
                f"runtime_error: provider cleanup failed: {type(error).__name__}: {error}"
            )
        finally:
            # Process cleanup must run even if provider accounting or cancellation fails.
            for future, cancel, _, _ in self._command_jobs.values():
                cancel.set()
            wait([future for future, _, _, _ in self._command_jobs.values()])
            try:
                self._collect_command_jobs()
            except SecurityBoundaryFailure as error:
                if schedule.stop_reason != "command_cleanup_unconfirmed":
                    schedule.stop_reason = error.stop_reason
                self._quarantine()

    def _command_journal(self, command):
        return self._workspace.control_root / "command_execution" / f"{command['id']}.json"

    def _quarantine(self):
        self._store.quarantine_execution()

    def _collect_command_jobs(self):
        failures = []
        for activation_id, (future, cancel, state, command) in list(self._command_jobs.items()):
            if not future.done():
                continue
            del self._command_jobs[activation_id]
            self._store.set_activation_context(
                state.activation_id,
                state.pending_decision_round or 0,
                state.steps + 1,
            )
            try:
                raw = future.result()
                write_json(
                    self._command_journal(command),
                    {
                        "phase": "executed",
                        "command_id": command["id"],
                        "raw_result": raw,
                    },
                )
                result = self._command_runner.finalize_execution(raw, command_id=int(command["id"]))
                self._finish_command(state, command, result)
            except Exception as error:
                # Preserve the durable intent/raw result; collect every finished sibling before pausing.
                self._emit_progress(
                    "command_finalization_failed",
                    command_id=command["id"],
                    error=f"{type(error).__name__}: {error}",
                )
                failures.append(error)
        if failures:
            cleanup_failure = next(
                (error for error in failures if isinstance(error, CommandCleanupUnconfirmed)), None
            )
            if cleanup_failure is not None:
                raise cleanup_failure
            raise SecurityBoundaryFailure(
                "command execution outcome could not be finalized"
            ) from failures[0]

    def _concurrency_limit(self) -> int:
        from mikoshi.runtime.organization import peer_count

        peers = peer_count(self._store)
        return min(peers, getattr(self._provider, "concurrency", peers))

    def _stop(self, reason):
        self.stop_reason = reason
        self._stopped = True

    def _workspace_request(self, state: WorkspaceActivation) -> DecisionRequest:
        actor = state.trigger.agent_id
        packet = self._organization.context(state)
        capabilities = self._organization.capabilities(actor)
        if "read_workspace" in capabilities:
            self._initialize_repository_guidance(state)
            if any(state.repository_guidance.values()):
                packet["repository_guidance"] = state.repository_guidance
        state.tools = self._tools_for(state)
        actions = [tool.schema for tool in state.tools.values()]
        command_enabled = "run_command" in state.tools and self._command_runner is not None
        if command_enabled:
            packet["commands"] = self._command_context()
        packet["execution_access"] = {
            "enabled": command_enabled,
            "reason": "run_command is available."
            if command_enabled
            else "Commands are disabled for this run."
            if self._command_runner is None
            else "Commands are unavailable under this team protocol."
            if {"read_workspace", "write_artifact"}.issubset(capabilities)
            else "Commands require the read_workspace and write_artifact capabilities; "
            "the team is missing one.",
        }
        request = peer_request(
            state=state,
            agent=self._store.get_agent(actor),
            tick=self._store.current_tick,
            packet=packet,
            actions=actions,
            notifications=self._notifications,
            conversations=self._conversations,
            protocol=self._protocol,
            disable_yield=self._disable_yield,
            budget=self._budget,
        )
        request = budget_request(self._provider, request)
        request = replace(request, instructions=request.instructions + "\n" + BATCH_GUIDANCE)
        # Reserve space for state, the tool contract and reply envelopes. Bound
        # only the next model input; the exchange archive keeps the original text.
        provider = (
            self._provider.for_agent(actor)
            if callable(getattr(self._provider, "for_agent", None))
            else self._provider
        )
        ceiling = (
            getattr(provider, "input_token_limit", None) or request.metadata["input_token_ceiling"]
        )
        fixed = json.dumps(
            [request.instructions, request.model_input_text, request.schema], ensure_ascii=False
        )
        counter = getattr(provider, "count_tokens", None)
        measured = counter(fixed) if counter else None
        overhead = measured if measured is not None else len(fixed.encode())
        envelope_reserve = min(2048, ceiling // 16)
        allowance = max(0, min(ceiling // 2, ceiling - overhead - envelope_reserve))
        history = self._conversations.last_exchange(
            actor, result_budget=allowance, pageable="read_output" in state.tools
        )
        request = replace(request, history=history)
        request.metadata["context_characters"]["history"] = len(
            json.dumps(history, ensure_ascii=False)
        )

        self._emit_progress(
            "decision_prepared",
            decision_id=request.metadata["decision_id"],
            agent_id=actor,
            activation_id=state.activation_id,
            activation_step=state.steps + 1,
            metadata=dict(request.metadata),
            instructions=request.instructions,
            model_input=request.model_input_text
            if request.model_input_text is not None
            else request.input_text,
            history=list(request.history),
            compaction_state=request.compaction_input_text,
            schema=request.schema,
            state=request.input_text,
            conversation_id=request.conversation_id,
            stage=request.stage,
            name=request.name,
        )
        return request

    def _set_tool_context(self, state, index):
        calls = native_tool_calls(state.pending_decision)
        call_id = calls[index]["id"] if index < len(calls) else None
        self._store.set_action_context(
            state.activation_id,
            decision_id=state.pending_request.metadata.get("decision_id")
            if state.pending_request
            else None,
            request_id=getattr(state.pending_decision, "request_id", None),
            tool_call_id=call_id,
            tool_call_index=index,
        )
        return call_id

    def _command_context(self):
        execution_context = self._command_runner.execution_context()
        return {
            "default_timeout_seconds": self._command_runner.limits.default_timeout_seconds,
            "max_timeout_seconds": self._command_runner.limits.max_timeout_seconds,
            "execution_context": {
                key: execution_context[key]
                for key in (
                    "working_directory",
                    "backend",
                    "network",
                    "pids_per_command",
                    "resource_target_percent",
                    "admission",
                )
                if key in execution_context
            }
            | {
                "note": "Only changes under /workspace persist between commands; shell state "
                "and services do not."
                + (
                    " There is no network access, so dependencies must already be installed "
                    "in the image."
                    if execution_context.get("network", "none") == "none"
                    else ""
                )
            },
        }

    def _apply_tool_turn(self, state, decision, schema):
        validate_tool_turn(decision, schema)
        if decision["action"] != BATCH_ACTION:
            self._set_tool_context(state, 0)
            return self._apply_workspace_decision(state, decision)
        calls = decision["actions"]
        while len(state.batch_results) < len(calls):
            budget = self._budget
            if (
                budget is not None
                and budget.remaining_seconds is not None
                and budget.remaining_seconds <= 0
            ):
                raise BudgetExhausted("wall_time_exhausted")
            if self._stop_requested and self._stop_requested():
                raise _AwaitTool()
            call = calls[len(state.batch_results)]
            self._set_tool_context(state, len(state.batch_results))
            try:
                terminal, observation = self._apply_workspace_decision(state, call)
            except (_AwaitTool, SecurityBoundaryFailure, BudgetExhausted):
                raise
            except Exception as error:
                observation = {"error": model_error(error)}
                terminal = False
            state.batch_results.append(observation)
            state.pending_command_id = None
            self._emit_progress(
                "action_completed",
                activation_id=state.activation_id,
                agent_id=state.trigger.agent_id,
                action=call["action"],
                step=state.steps,
                batch_index=len(state.batch_results) - 1,
                terminal=terminal,
            )
            failed = isinstance(observation, Mapping) and (
                "error" in observation
                or observation.get("action_executed") is False
                or observation.get("status") in {"failed", "deferred"}
                or (
                    observation.get("status") == "blocked"
                    and observation.get("action_executed") is not True
                )
                or ("exit_code" in observation and observation["exit_code"] != 0)
            )
            if terminal or failed:
                state.batch_results.extend(
                    {
                        "status": "skipped",
                        "action_executed": False,
                        "reason": "Skipped because an earlier call in this turn did not succeed.",
                    }
                    for _ in range(len(calls) - len(state.batch_results))
                )
                return terminal, {"results": list(state.batch_results)}
        return False, {"results": list(state.batch_results)}

    def _tools_for(self, state):
        peer = state.trigger.agent_id
        available = []
        for group, schemas in (
            (self._organization, self._organization.schemas()),
            (self._pages, self._pages.schemas(self._limits.file_chars)),
            (self._shared_work, self._shared_work.schemas(state)),
        ):
            available.extend(TeamTool(schema, group.apply) for schema in schemas)
        if self._command_runner is not None and {"read_workspace", "write_artifact"}.issubset(
            self._organization.capabilities(peer)
        ):
            available.append(
                TeamTool(_request_command_schema(self._command_runner.limits), self._command_tool)
            )
        available.extend(
            (
                TeamTool(CHECK_BUDGET_SCHEMA, self._budget_tool),
                TeamTool(READ_OUTPUT_SCHEMA, self._read_output_tool),
            )
        )
        if self._disable_yield:
            available = [tool for tool in available if tool.name != "peer_yield"]
        tools = bind_tools(self._protocol.tools(tuple(available), peer=peer))
        if self._disable_yield and "peer_yield" in tools:
            raise ValueError("peer_yield is disabled for this run")
        return tools

    def _command_tool(self, state, decision):
        if not {"read_workspace", "write_artifact"}.issubset(
            self._organization.capabilities(state.trigger.agent_id)
        ):
            raise WorkspaceAccessError("commands require read_workspace and write_artifact")
        return False, self._run_command(state, decision)

    def _budget_tool(self, state, decision):
        self._organization.authorize(state)
        result = check_budget(self._budget)
        self._store.record_workspace_action(
            state.activation_id,
            state.trigger.agent_id,
            decision["action"],
            "succeeded",
            "run",
            result,
        )
        return False, result

    def _read_output_tool(self, state, decision):
        self._organization.authorize(state)
        result = read_output(
            self._store,
            self._workspace,
            self._conversations,
            state.trigger.agent_id,
            decision,
            log_root=self._command_runner.log_root if self._command_runner is not None else None,
        )
        version = self._workspace._capture_bytes(
            "outputs/" + decision["reference"].replace(":", "-").replace("/", "-") + ".txt",
            result["content"].encode(),
        )
        self._store.record_workspace_action(
            state.activation_id,
            state.trigger.agent_id,
            decision["action"],
            "succeeded",
            decision["reference"],
            {
                "offset": result["offset"],
                "next_offset": result["next_offset"],
                "resource_version": version,
            },
        )
        return False, result

    def _apply_workspace_decision(self, state, decision):
        action = decision["action"]
        if action == "peer_yield" and self._disable_yield:
            raise ValueError("peer_yield is disabled for this run")
        tools = state.tools or self._tools_for(state)
        if action not in tools:
            raise ValueError(f"unknown tool {action!r}")
        try:
            return tools[action].execute(state, decision)
        except (_AwaitTool, BudgetExhausted, SecurityBoundaryFailure):
            raise
        except Exception as error:
            self._record_workspace_failure(state, action, str(decision.get("path", "")), error)
            raise

    def _initialize_repository_guidance(self, state):
        if "startup" not in state.repository_guidance:
            actor = state.trigger.agent_id
            provider = (
                self._provider.for_agent(actor)
                if hasattr(self._provider, "for_agent")
                else self._provider
            )
            state.repository_guidance["startup"] = load_repository_guidance(
                self._workspace,
                actor,
                getattr(provider, "context_window_tokens", None),
                getattr(provider, "count_tokens", None),
            )

    def _acknowledge_repository_guidance(self, state, request, result):
        guide = json.loads(request.input_text).get("repository_guidance", {})
        activity = guide.get("activity", {})
        checkpoints = (getattr(result, "assistant_message", None) or {}).get(
            "compaction_checkpoints", ()
        )
        self._guidance_ledger.acknowledge(
            state.trigger.agent_id,
            [*activity.get("documents", []), *warning_documents(activity.get("warnings", []))],
            checkpoints,
        )
        state.repository_guidance.pop("activity", None)

    def _prepare_repository_action(self, state, decision):
        actor = state.trigger.agent_id
        if "read_workspace" not in self._organization.capabilities(actor):
            return None
        self._initialize_repository_guidance(state)
        root = root_guidance(self._workspace, actor)
        previous_startup = state.repository_guidance["startup"] or {}
        startup = {**previous_startup, **root}
        if not any(startup.values()):
            startup = None
        activity = activity_guidance(self._workspace, actor, decision)
        activity["documents"] = self._guidance_ledger.pending(actor, activity["documents"])
        activity["warnings"] = [
            json.loads(item["content"])
            for item in self._guidance_ledger.pending(
                actor, warning_documents(activity["warnings"])
            )
        ]
        changed = startup != state.repository_guidance["startup"] or bool(
            activity["documents"] or activity["warnings"]
        )
        state.repository_guidance["startup"] = startup
        if activity["documents"] or activity["warnings"]:
            state.repository_guidance["activity"] = activity
        else:
            state.repository_guidance.pop("activity", None)
        if not changed:
            return None
        result = {
            "status": "guidance_loaded",
            "action_executed": False,
            "message": "The action did not run. Repository instructions that apply to it "
            "arrive in the next observation; repeat the action after reading them. A document "
            "with a next_offset continues in read_file from that line.",
            "loaded_paths": [item["path"] for item in activity["documents"]],
        }
        self._store.record_workspace_action(
            state.activation_id, actor, decision["action"], "deferred", decision.get("path"), result
        )
        return result

    def _run_command(
        self,
        state: WorkspaceActivation,
        decision: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self._command_runner is None:
            raise ValueError("command execution is disabled")
        if state.pending_command_id is None:
            normalized = self._command_runner.normalize_request(
                ["sh", "-c", decision["command"]],
            )
            if guidance := self._prepare_repository_action(
                state, {"action": "run_command", "argv": normalized["argv"]}
            ):
                return guidance
            command = self._store.request_command(
                state.activation_id,
                state.trigger.agent_id,
                normalized["argv"],
                "repository",
                project_id=state.trigger.project_id,
                timeout_seconds=decision.get(
                    "timeout_seconds", self._command_runner.limits.default_timeout_seconds
                ),
            )
            state.pending_command_id = int(command["id"])
        else:
            command = self._store.get_command_request(state.pending_command_id)
        if command["status"] != "approved":
            # Completed process receipts survive stopping between execution and observation.
            state.pending_command_id = None
            return _command_result(command)
        budget = self._budget
        timeout = None
        if budget is not None:
            # Command execution does not incur inference charges. Already-returned
            # command decisions can execute while new model requests are cost-blocked.
            timeout = budget.remaining_seconds
            if timeout is not None and timeout <= 0:
                raise BudgetExhausted("wall_time_exhausted")
        limit = command.get("timeout_seconds")
        if limit is not None:
            timeout = limit if timeout is None else min(limit, timeout)
        prepared = self._command_runner.prepare_execution(
            state.trigger.agent_id,
            command,
            timeout_seconds=timeout,
        )
        write_json(
            self._command_journal(command),
            {
                "phase": "executing",
                "command_id": command["id"],
            },
        )
        self._command_jobs[state.activation_id] = (
            self._command_runner.submit(prepared),
            prepared.cancel_event,
            state,
            command,
        )
        raise _AwaitTool()

    def _finish_command(self, state, command, result):
        result = dict(result)
        try:
            write_json(
                self._command_journal(command),
                {
                    "phase": "executed",
                    "command_id": command["id"],
                    "result": result,
                },
            )
            completed = self._store.complete_command_request(
                state.activation_id,
                state.trigger.agent_id,
                int(command["id"]),
                result,
            )
        except Exception as error:
            # Execution already happened; preserve its receipt even if finalization fails.
            raise SecurityBoundaryFailure(
                "executed command receipt could not be persisted"
            ) from error
        try:
            self._command_journal(command).unlink()
        except Exception as error:
            raise SecurityBoundaryFailure(
                "command receipt persisted but journal acknowledgement failed"
            ) from error
        if result.get("termination_reason") == "command_cleanup_failed":
            raise CommandCleanupUnconfirmed("command cleanup unconfirmed; receipt preserved")
        if result.get("termination_reason") == "workspace_validation_failed":
            raise SecurityBoundaryFailure(
                "command left an invalid shared workspace or unconfirmed process cleanup; receipt preserved"
            )
        return completed

    def _record_workspace_failure(
        self,
        state: WorkspaceActivation,
        action: str,
        path: str,
        error: Exception,
    ) -> None:
        self._store.record_workspace_action(
            state.activation_id,
            state.trigger.agent_id,
            action,
            "denied" if isinstance(error, WorkspaceAccessError) else "failed",
            path,
            {
                "project_id": state.trigger.project_id,
                "error": model_error(error),
                "error_type": type(error).__name__,
            },
        )

    def _summary(
        self,
        summaries: Sequence[ActivationSummary],
        causal_waves: int,
    ) -> RunSummary:
        return RunSummary(
            activations=tuple(sorted(summaries, key=lambda item: item.id)),
            remaining_triggers=self._store.queued_trigger_count(),
            current_tick=self._store.current_tick,
            causal_waves=causal_waves,
        )


def model_error(error: Exception) -> str:
    """Tool errors as the model sees them: the message, without Python exception names."""
    if isinstance(error, ModelDecisionError):
        return error.feedback
    if isinstance(error, (ValueError, TypeError, OSError)) and str(error):
        return str(error)
    return f"{type(error).__name__}: {error}"


def _request_command_schema(limits: CommandLimits) -> Mapping[str, Any]:
    return action_schema(
        "run_command",
        {
            "command": {
                "type": "string",
                "minLength": 1,
                "maxLength": limits.max_arg_chars,
                "description": "Shell command; supports pipes, redirects, and multiline scripts.",
            },
            "timeout_seconds": {
                "type": "number",
                "minimum": 1,
                "maximum": limits.max_timeout_seconds,
                "description": (
                    f"Seconds before the command is stopped; default "
                    f"{limits.default_timeout_seconds}."
                ),
            },
        },
        required=("command",),
    ) | {
        "description": (
            "Run a shell command with sh in a fresh container at /workspace, and wait for it "
            "to finish while peers keep working. Background processes end with the command. "
            "Returns the exit code, the end of the combined output, and an output_ref "
            "for read_output. commands.execution_context describes "
            "the container."
        )
    }


def _command_result(command: Mapping[str, Any]) -> Mapping[str, Any]:
    """The model-facing result: the runner's tail preview, unmodified, plus references."""
    return {
        "exit_code": command.get("exit_code"),
        "output": str(command.get("output") or "") or "(no output)",
        "output_truncated": bool(command.get("output_truncated", False)),
        "output_ref": f"command:{command['id']}"
        if command["status"] in COMPLETED_STATUSES
        else None,
    }
