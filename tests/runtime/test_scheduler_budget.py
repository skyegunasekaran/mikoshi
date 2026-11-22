"""Actual spending stops new work without cancelling in-flight requests."""

import asyncio
import json
import time

import httpx
import pytest
from support.organization import world
from support.provider import MockDecisionProvider, tool_calls

from mikoshi.runtime.budgets import BudgetExhausted, ExecutionBudget
from mikoshi.runtime.organization import _enqueue


def setup(tmp_path, handler, limit, *, input_rate=1, output_rate=2):
    budget = ExecutionBudget(
        tmp_path / "budget.json",
        limit,
        input_rate=input_rate,
        cache_read_rate=input_rate,
        output_rate=output_rate,
        thinking_rate=output_rate,
    )
    provider = MockDecisionProvider(
        "mock",
        api_key="mock",
        max_tokens=100,
        context_window_tokens=1000,
        concurrency=2,
        timeout_seconds=5,
        budget=budget,
        http_transport=httpx.MockTransport(handler),
    )
    store, runtime = world(tmp_path, provider, initial_agent_only=True)
    runtime._claim_wave(1, 1)
    _enqueue(store, "test peer session", recipients=["worker_a", "worker_b"])
    workers = runtime._claim_wave(2, 1, ["participant"])
    return provider, store, runtime, budget, workers


def bounded_stop(predicate=lambda: False):
    deadline = time.monotonic() + 5

    def stop():
        if predicate():
            return "test_pause"
        if time.monotonic() >= deadline:
            return "test_watchdog"
        return None

    return stop


def test_budget_settles_admitted_work_and_blocks_further_spending(tmp_path):
    limit = 0.00001
    calls, cancelled, completed, applied = [], [], [], set()
    both_started = asyncio.Barrier(2)

    async def handler(request):
        packet = json.loads(json.loads(request.content)["messages"][-1]["content"])
        actor = packet["agent_id"]
        calls.append(actor)
        # Starting a request records no speculative charge.
        assert budget.charged == 0
        try:
            await both_started.wait()
        except asyncio.CancelledError:
            cancelled.append(actor)
            raise
        completed.append(actor)
        # A terminal action prevents a fast worker starting another request
        # while the other worker's first response is still being collected.
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": tool_calls(
                                {
                                    "action": "peer_yield",
                                }
                            )
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    provider, store, runtime, budget, workers = setup(tmp_path, handler, limit)
    runtime._progress_callback = lambda event: (
        applied.add(event["agent_id"]) if event["kind"] == "action_completed" else None
    )
    runtime._stop_requested = bounded_stop(lambda: applied >= {"worker_a", "worker_b"})
    try:
        runtime._run_wave(workers)
        assert runtime.stop_reason in {"test_pause", "budget_exhausted"}
        assert applied == {"worker_a", "worker_b"}
        assert cancelled == []
        assert sorted(calls) == sorted(completed) == ["worker_a", "worker_b"]
        # The stop boundary may cancel a just-admitted follow-up before transport
        # starts. Only the two completed requests should contribute charges.
        assert sum(entry["status"] == "reported" for entry in budget.entries) == 2
        assert budget.charged == pytest.approx(0.00006)
        assert all("reserved_usd" not in entry for entry in budget.entries)
        assert all(state.steps == 1 for state in workers)
        if budget.charged >= limit:
            with pytest.raises(BudgetExhausted):
                budget.check()
    finally:
        provider.close()
        store.close()
