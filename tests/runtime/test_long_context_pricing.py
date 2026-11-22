"""Billing includes cached input tiers and exposes unresolved provider usage."""

import json
from dataclasses import asdict

import httpx
import pytest

from mikoshi.providers.pricing import TokenPrices
from mikoshi.runtime.budgets import BudgetExhausted, ExecutionBudget


def test_billing_tiers_cache_ttls_unknown_usage_and_spending_admission(tmp_path):
    rates = TokenPrices(
        input_rate=1,
        cache_read_rate=0.1,
        output_rate=5,
        thinking_rate=5,
        long_context_threshold=100000,
        long_context_multiplier=5,
    )
    budget = ExecutionBudget(tmp_path / "budget.json", dollars=1, **asdict(rates))
    ids = budget.begin_requests([{"model": "fixture"}] * 2)
    budget.settle(
        ids,
        [
            httpx.Response(
                200,
                json={
                    "usage": {
                        "input_tokens": 25000,
                        "cache_read_input_tokens": 50000,
                        "cache_creation_input_tokens": created,
                        "cache_creation": {
                            "ephemeral_5m_input_tokens": 15000,
                            "ephemeral_1h_input_tokens": created - 15000,
                        },
                        "output_tokens": 100,
                        "output_tokens_details": {"thinking_tokens": 40},
                    }
                },
            )
            for created in (25000, 25001)
        ],
    )
    first, second = budget.entries
    assert first["request_cost_multiplier"] == 1 and second["request_cost_multiplier"] == 5
    assert second["input_tokens"] == 100001
    assert second["cache_creation_5m_usd"] == pytest.approx(15000 * 1.25 * 5 / 1e6)
    assert second["cache_creation_1h_usd"] == pytest.approx(10001 * 2 * 5 / 1e6)
    assert second["output_usd"] == pytest.approx(60 * 5 * 5 / 1e6)
    assert second["thinking_usd"] == pytest.approx(40 * 5 * 5 / 1e6)
    unknown = budget.begin_requests([{"model": "fixture"}])
    budget.settle(unknown, [httpx.Response(200, json={})])
    assert budget.entries[-1]["status"] == "unknown_usage"
    assert budget.summary()["unknown_usage_requests"] == 1
    charged = sum(entry["charged_usd"] for entry in (first, second))
    assert budget.charged == pytest.approx(charged)
    assert (
        json.loads(budget.path.read_text(encoding="utf-8"))["entries"][-1]["status"]
        == "unknown_usage"
    )
    budget.limit = charged
    with pytest.raises(BudgetExhausted, match="budget_exhausted"):
        budget.begin_requests([{"model": "fixture"}])
