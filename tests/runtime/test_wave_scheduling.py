"""Admission, backoff, and cleanup remain independent across peer sessions."""

import threading
import time
from collections import Counter

from support.boundaries import run_for
from support.organization import swarm_problem, world

from mikoshi.decisions import ProviderTransportError


def test_peer_retries_release_slots_and_stop_at_permanent_errors_or_attempt_limits(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("mikoshi.runtime.lifecycle.PROVIDER_RETRY_DELAYS", (0.001,))

    class Provider:
        def __init__(self):
            self.requests = []
            self.attempts = Counter()

        def generate_batch(self, requests):
            (request,) = requests
            self.requests.extend(requests)
            agent = request.metadata["agent_id"]
            self.attempts[agent] += 1
            if agent == "agent-2":
                return [ProviderTransportError("unauthorized", 401)]
            if agent == "agent-3":
                return [ProviderTransportError("unavailable", 503)]
            if self.attempts[agent] == 1:
                return [ProviderTransportError("rate limited", 429, retry_after_seconds=0.1)]
            return [{"action": "peer_yield"}]

    provider = Provider()
    store, runtime = world(
        tmp_path,
        provider,
        swarm_problem(population=3),
        enable_commands=False,
    )
    retries = []
    runtime._progress_callback = lambda event: (
        retries.append(event) if event["kind"] == "model_request_retry_scheduled" else None
    )
    try:
        summary = run_for(runtime)
        assert runtime.stop_reason == "provider_unavailable"
        assert provider.attempts == {"agent-1": 2, "agent-2": 1, "agent-3": 2}
        agents = [request.metadata["agent_id"] for request in provider.requests]
        assert agents.index("agent-2") < agents.index("agent-1", 1)
        peer_requests = [
            request for request in provider.requests if request.metadata["agent_id"] == "agent-1"
        ]
        assert peer_requests[0] is peer_requests[1]
        first_retry = next(event for event in retries if event["agent_id"] == "agent-1")
        assert first_retry["delay_seconds"] == 0.1
        assert sum(activation.action_count for activation in summary.activations) == 1
    finally:
        store.close()


def test_provider_cleanup_failure_still_cancels_and_records_command(tmp_path, monkeypatch):
    entered = threading.Event()
    owner_thread = threading.get_ident()

    class Provider:
        def __init__(self):
            self.handles = 0
            self.ready = {}

        def submit_request(self, request):
            self.handles += 1
            if self.handles == 1:
                self.ready[1] = {"action": "run_command", "command": "wait for cancellation"}
            return self.handles

        def collect_ready(self, handles, timeout_seconds=0):
            return {handle: result for handle, result in self.ready.items() if handle in handles}

        def acknowledge_result(self, handle):
            self.ready.pop(handle)

        def cancel_pending(self):
            raise RuntimeError("accounting unavailable")

    provider = Provider()
    store, runtime = world(
        tmp_path,
        provider,
        swarm_problem(population=2),
        enable_commands=True,
    )

    def execute(prepared):
        entered.set()
        assert prepared.cancel_event.wait(5)
        return {
            "_command_id": prepared.command_id,
            "status": "failed",
            "termination_reason": "cancelled",
            "exit_code": None,
        }

    recorded_threads = []
    complete_command = store.complete_command_request

    def record_command(*args, **kwargs):
        recorded_threads.append(threading.get_ident())
        return complete_command(*args, **kwargs)

    monkeypatch.setattr(runtime._command_runner, "_execute_container", execute)
    monkeypatch.setattr(store, "complete_command_request", record_command)
    deadline = time.monotonic() + 5
    try:
        runtime.run(
            stop_requested=lambda: (
                "operator_stop"
                if entered.is_set()
                else "test_watchdog"
                if time.monotonic() >= deadline
                else None
            )
        )
        assert runtime.stop_reason == (
            "runtime_error: provider cleanup failed: RuntimeError: accounting unavailable"
        )
        assert entered.is_set()
        assert not runtime._command_jobs
        assert recorded_threads == [owner_thread]
        (command,) = store.list_command_requests()
        assert command["termination_reason"] == "cancelled"
        assert not list((runtime.workspace.control_root / "command_execution").glob("*.json"))
    finally:
        store.close()


def test_every_peer_can_hold_an_activation_while_requests_share_provider_slots(tmp_path):
    class Provider:
        concurrency = 2

        def __init__(self):
            self.store = None
            self.turns = Counter()
            self.most_running = 0

        def generate_batch(self, requests):
            (running,) = self.store._connection.execute(
                "SELECT COUNT(*) FROM activations WHERE status='running' "
                "AND agent_id IN (SELECT id FROM agents WHERE role='peer')"
            ).fetchone()
            self.most_running = max(self.most_running, running)
            decisions = []
            for request in requests:
                agent = request.metadata["agent_id"]
                self.turns[agent] += 1
                done = self.turns[agent] > 1
                decisions.append({"action": "peer_yield" if done else "check_budget"})
            return decisions

    provider = Provider()
    store, runtime = world(tmp_path, provider, swarm_problem(population=4), enable_commands=False)
    provider.store = store
    try:
        run_for(runtime)
        assert runtime.stop_reason == "all_agents_yielded"
        assert set(provider.turns) == {"agent-1", "agent-2", "agent-3", "agent-4"}
        assert provider.most_running == 4
        assert runtime._concurrency_limit() == 2
    finally:
        store.close()
