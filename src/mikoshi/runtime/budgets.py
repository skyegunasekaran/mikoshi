"""Agent context limits, token measurement, and durable execution budgets.

Character limits bound stored evidence; token limits bound model requests. Cost
accounting uses reported usage and caller-supplied prices per million tokens.
Pending requests and unavailable usage are tracked without estimated charges.
Reported prompt and completion totals are partitioned into cache writes/reads and
visible output/thinking so discounted tokens are never charged twice.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from mikoshi.contracts import action_schema
from mikoshi.providers.usage import usage_breakdown

if TYPE_CHECKING:
    from mikoshi.decisions import DecisionRequest


DEFAULT_CONTEXT_WINDOW = 1048576
TIME_REMINDER_PERCENTAGES = (5, 1)
_TOKEN_BUDGET_SECTION = re.compile(r"\n\[Token budgets\]\n.*?\n\[/Token budgets\]\n", re.S)
_INPUT_CATEGORIES = (
    "uncached_input",
    "cache_read",
    "cache_creation",
    "cache_creation_5m",
    "cache_creation_1h",
)


def rate_name(category):
    """Price field for a billing category; uncached input uses the base input price."""
    return "input_rate" if category == "uncached_input" else f"{category}_rate"


_BILLING_CATEGORIES = (
    *_INPUT_CATEGORIES,
    "output",
    "thinking",
    "unattributed_input",
    "unattributed_completion",
)


CHECK_BUDGET_SCHEMA = action_schema("check_budget") | {
    "description": (
        "Report the run's working time, shared by all peers: minutes elapsed, minutes "
        "remaining and the limit. Also reports spending when the run has a spending limit."
    )
}


def working_time(budget):
    """Whole minutes elapsed and remaining, or None when the run has no time limit."""
    if budget is None or budget.seconds is None:
        return None
    limit = round(budget.seconds / 60)
    elapsed = min(limit, int(budget.elapsed // 60))
    return {
        "elapsed_minutes": elapsed,
        "remaining_minutes": limit - elapsed,
        "limit_minutes": limit,
    }


def check_budget(budget):
    result = {"working_time": working_time(budget)}
    if budget is not None and budget.limit is not None and budget.charged is not None:
        result["spending"] = {
            "spent_usd": round(budget.charged, 2),
            "remaining_usd": round(max(0.0, budget.limit - budget.charged), 2),
            "limit_usd": budget.limit,
        }
    return result


def budget_context(budget):
    """Stable run allowances. The live clock is available through check_budget."""
    if budget is None:
        return None
    return {"wall_time_limit_seconds": budget.seconds, "spending_limit_usd": budget.limit}


def budget_warnings(budget):
    """Stable near-limit bands for model context; exact accounting stays in the ledger."""
    if budget is None:
        return []
    warnings = []
    charged = budget.charged
    for resource, limit, remaining in (
        ("time", budget.seconds, budget.remaining_seconds),
        (
            "spending",
            budget.limit,
            None if budget.limit is None or charged is None else budget.limit - charged,
        ),
    ):
        if limit is None or remaining is None:
            continue
        thresholds = reversed(TIME_REMINDER_PERCENTAGES) if resource == "time" else (1, 5, 10)
        for percent in thresholds:
            if remaining <= limit * percent / 100:
                warnings.append({"resource": resource, "remaining_percent_at_most": percent})
                break
    return warnings


@dataclass(frozen=True)
class AgentLimits:
    input_tokens: int = 131072
    output_tokens: int = 131072
    file_chars: int = 65536
    read_chars: int = 16000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_AGENT_LIMITS = AgentLimits()


def limits_for(provider):
    return getattr(provider, "agent_limits", DEFAULT_AGENT_LIMITS)


def with_token_budget(
    request: DecisionRequest,
    output_limit: int,
    context_window: int,
) -> DecisionRequest:
    """Set the request ceiling and replace its model-facing budget guidance."""
    if output_limit < 1 or output_limit >= context_window:
        raise ValueError("output token limit must leave room for input in the context window")
    input_limit = context_window - output_limit
    section = (
        "\n[Token budgets]\n"
        f"Context window: {context_window:,} tokens. Output limit per response: "
        f"{output_limit:,} tokens, shared by reasoning and the response.\n"
        "Keep each tool-call batch small enough to finish within that shared limit. "
        "Split large file writes and edits across turns, leaving room for reasoning "
        "and complete tool arguments.\n[/Token budgets]\n"
    )
    instructions, replacements = _TOKEN_BUDGET_SECTION.subn(section, request.instructions)
    return replace(
        request,
        max_tokens=output_limit,
        instructions=instructions if replacements else request.instructions + section,
        metadata={
            **request.metadata,
            "input_token_ceiling": input_limit,
            "output_token_ceiling": output_limit,
            "input_token_budget": input_limit,
            "output_token_budget": output_limit,
            "context_window_tokens": context_window,
            "input_budget_enforcement": "derived_from_unified_context",
        },
    )


def budget_request(provider, request: DecisionRequest) -> DecisionRequest:
    """Apply configured actor limits when constructing a request."""
    if callable(getattr(provider, "for_agent", None)):
        provider = provider.for_agent(request.metadata.get("agent_id"))
    limits = limits_for(provider)
    default_tokens = getattr(provider, "max_tokens", limits.output_tokens)
    tokens = request.max_tokens if request.max_tokens is not None else default_tokens
    tokens = min(tokens, getattr(provider, "max_tokens", tokens))
    return with_token_budget(
        request,
        tokens,
        getattr(provider, "context_window_tokens", DEFAULT_CONTEXT_WINDOW),
    )


class BudgetExhausted(Exception):
    """Stop new model requests when the run reaches a resource limit."""


class ContextWindowExhausted(ValueError):
    """The provider's local admission check cannot fit the next model input."""


class ExecutionBudget:
    """Track requests durably and charge only provider-reported usage.

    Rates are dollars per million tokens. Charge uncached input, cache reads,
    cache creation, visible output and separately reported thinking according to
    the provider's usage breakdown. Missing usage remains unresolved rather than
    receiving an estimated charge. Persist pending requests and charges for audit.
    Requests above an optional inclusive-input threshold multiply every cost
    category. A time-only budget may omit token prices.
    """

    def __init__(
        self,
        path,
        dollars=None,
        seconds=None,
        *,
        input_rate=None,
        cache_read_rate=None,
        cache_creation_rate=None,
        cache_creation_5m_rate=None,
        cache_creation_1h_rate=None,
        output_rate=None,
        thinking_rate=None,
        long_context_threshold=None,
        long_context_multiplier=1.0,
        start_paused=False,
        allow_unlimited=False,
        model_prices=None,
    ):
        if dollars is None and seconds is None and not allow_unlimited:
            raise ValueError("at least one execution budget limit is required")
        if long_context_threshold is not None and (
            type(long_context_threshold) is not int or long_context_threshold <= 0
        ):
            raise ValueError("long_context_threshold must be a positive integer or null")
        if (
            type(long_context_multiplier) not in (int, float)
            or not math.isfinite(long_context_multiplier)
            or long_context_multiplier <= 0
        ):
            raise ValueError("long_context_multiplier must be positive and finite")
        rates = (input_rate, cache_read_rate, output_rate, thinking_rate)
        for value in (dollars, seconds):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError("budget limits must be positive and finite")
        for value in (*rates, cache_creation_rate, cache_creation_5m_rate, cache_creation_1h_rate):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value < 0
            ):
                raise ValueError("token prices must be nonnegative and finite")
        if any(rate is not None for rate in rates) and any(rate is None for rate in rates):
            raise ValueError("provide cache write, cache read, output, and thinking rates, or none")
        if dollars is not None and input_rate is None:
            raise ValueError("a dollar limit requires explicit token prices")
        self.path = Path(path)
        self.limit = dollars
        self.seconds = seconds
        self.input_rate = input_rate
        self.cache_read_rate = cache_read_rate
        if input_rate is None and any(
            r is not None
            for r in (cache_creation_rate, cache_creation_5m_rate, cache_creation_1h_rate)
        ):
            raise ValueError("cache creation prices require base token prices")
        for name, rate, multiplier in (
            ("cache_creation_rate", cache_creation_rate, 1.25),
            ("cache_creation_5m_rate", cache_creation_5m_rate, 1.25),
            ("cache_creation_1h_rate", cache_creation_1h_rate, 2),
        ):
            if rate is None and input_rate is not None:
                rate = multiplier * input_rate
            if rate is not None and not math.isfinite(rate):
                raise ValueError("cache creation prices must be finite")
            setattr(self, name, rate)
        self.output_rate = output_rate
        self.thinking_rate = thinking_rate
        self.long_context_threshold = long_context_threshold
        self.long_context_multiplier = long_context_multiplier
        self.model_prices = dict(model_prices or {})
        from mikoshi.providers.pricing import TokenPrices

        if any(not isinstance(rates, TokenPrices) for rates in self.model_prices.values()):
            raise ValueError("model_prices must contain validated TokenPrices")
        self.entries = []
        self.elapsed_before = 0.0
        if self.path.exists():
            raise ValueError("budget ledger already exists; choose a fresh run output directory")
        self.started = None if start_paused else time.monotonic()

    @property
    def elapsed(self):
        return self.elapsed_before + (
            time.monotonic() - self.started if self.started is not None else 0.0
        )

    def start(self):
        """Begin accounting working time after launch preparation finishes."""
        if self.started is None:
            self.started = time.monotonic()

    def pause(self):
        """Stop the working-time clock during setup or finalization within this run."""
        self.elapsed_before = self.elapsed
        self.started = None

    @property
    def remaining_seconds(self):
        return None if self.seconds is None else max(0.0, self.seconds - self.elapsed)

    @property
    def charged(self):
        if self.input_rate is None:
            return None
        return sum(entry["charged_usd"] for entry in self.entries if entry["status"] == "reported")

    def snapshot(self):
        """Return cumulative accounting and elapsed time for this run."""
        return {
            "limit_usd": self.limit,
            "wall_limit_seconds": self.seconds,
            **{
                rate_name(category): getattr(self, rate_name(category))
                for category in (*_INPUT_CATEGORIES, "output", "thinking")
            },
            "long_context_threshold": self.long_context_threshold,
            "long_context_multiplier": self.long_context_multiplier,
            "model_prices": {model: asdict(rates) for model, rates in self.model_prices.items()},
            "elapsed_seconds": self.elapsed,
            "accounted_usd": self.charged,
            **self._unresolved_counts(),
            "entries": [dict(entry) for entry in self.entries],
            "cache_cost_comparison": self.cache_cost_comparison(),
        }

    def cache_cost_comparison(self):
        """Counterfactual input pricing for reported tokens, not a task-level benchmark."""
        if self.input_rate is None:
            return None
        reported = [entry for entry in self.entries if entry["status"] == "reported"]
        ordinary = sum(
            entry["input_tokens"]
            * self._prices_for(entry.get("model")).input_rate
            * self._request_cost_multiplier(entry["input_tokens"], entry.get("model"))
            / 1e6
            for entry in reported
        )
        actual = sum(
            entry.get(f"{category}_usd", 0.0)
            for entry in reported
            for category in (*_INPUT_CATEGORIES, "unattributed_input")
        )
        return {
            "basis": "Same reported input tokens at the uncached input price; excludes unreported requests and changes in model behavior.",
            "reported_requests": len(reported),
            "excluded_requests": len(self.entries) - len(reported),
            "input_usd_without_caching": ordinary,
            "input_usd_with_caching": actual,
            "estimated_input_savings_usd": ordinary - actual,
        }

    def save(self):
        data = self.snapshot()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        return data

    def check(self):
        if self.remaining_seconds is not None and self.remaining_seconds <= 0:
            raise BudgetExhausted("wall_time_exhausted")
        if self.limit is not None and self.charged >= self.limit:
            raise BudgetExhausted("budget_exhausted")

    def summary(self):
        """Compact spending view, attributed to model and agent rather than actions."""
        groups = {"by_model": {}, "by_agent": {}}
        for entry in self.entries:
            for name, key in (("by_model", "model"), ("by_agent", "agent_id")):
                label = entry.get(key)
                if label is None:
                    continue
                group = groups[name].setdefault(
                    label,
                    {
                        "reported_usd": 0.0,
                        "pending_requests": 0,
                        "unknown_usage_requests": 0,
                        "input_tokens": 0,
                        "completion_tokens": 0,
                        **{f"{category}_tokens": 0 for category in _BILLING_CATEGORIES},
                        **{f"{category}_usd": 0.0 for category in _BILLING_CATEGORIES},
                    },
                )
                if entry.get("status") == "reported":
                    group["reported_usd"] += entry.get("charged_usd") or 0.0
                    group["input_tokens"] += entry.get("input_tokens", 0)
                    group["completion_tokens"] += entry.get("completion_tokens", 0)
                    for category in _BILLING_CATEGORIES:
                        group[f"{category}_tokens"] += entry.get(f"{category}_tokens") or 0
                        group[f"{category}_usd"] += entry.get(f"{category}_usd") or 0.0
                elif entry["status"] == "pending":
                    group["pending_requests"] += 1
                elif entry["status"] == "unknown_usage":
                    group["unknown_usage_requests"] += 1
        for dimension in groups.values():
            for group in dimension.values():
                attributed_input = sum(
                    group[f"{category}_tokens"] for category in _INPUT_CATEGORIES
                )
                group["cache_hit_rate"] = (
                    group["cache_read_tokens"] / attributed_input
                    if attributed_input and not group["unattributed_input_tokens"]
                    else None
                )
        charged = self.charged
        return {
            "limit_usd": self.limit,
            "accounted_usd": charged,
            **self._unresolved_counts(),
            "available_usd": None if self.limit is None else max(0.0, self.limit - charged),
            **{
                f"{category}_price_per_million": getattr(self, rate_name(category))
                for category in (*_INPUT_CATEGORIES, "output", "thinking")
            },
            "long_context_threshold": self.long_context_threshold,
            "long_context_multiplier": self.long_context_multiplier,
            "model_prices": {model: asdict(rates) for model, rates in self.model_prices.items()},
            "remaining_seconds": self.remaining_seconds,
            "cache_cost_comparison": self.cache_cost_comparison(),
            **groups,
        }

    def _unresolved_counts(self):
        return {
            "pending_requests": sum(entry["status"] == "pending" for entry in self.entries),
            "unknown_usage_requests": sum(
                entry["status"] == "unknown_usage" for entry in self.entries
            ),
        }

    def begin_requests(self, payloads, metadata=()):
        """Persist pending requests before transport, without estimating their cost."""
        self.check()
        request_ids = []
        for position, payload in enumerate(payloads):
            request_ids.append(len(self.entries))
            self.entries.append(
                {
                    "max_tokens": payload.get("max_tokens", payload.get("max_output_tokens")),
                    "status": "pending",
                    "model": payload.get("model"),
                    "cache_ttl": (
                        payload["cache_control"].get("ttl", "5m")
                        if payload.get("cache_control")
                        else None
                    ),
                    "agent_id": metadata[position].get("agent_id") if metadata else None,
                    **{
                        key: metadata[position].get(key) if metadata else None
                        for key in ("request_id", "decision_id", "previous_request_id")
                    },
                }
            )
        self.save()
        return request_ids

    def settle(self, ids, responses):
        for index, response in zip(ids, responses):
            entry = self.entries[index]
            try:
                usage = usage_breakdown(
                    response["usage"] if isinstance(response, dict) else response.json()["usage"],
                    cache_ttl=entry.get("cache_ttl"),
                )
                model = entry.get("model")
                costs = self._cost_breakdown(usage, model)
                entry.update(
                    status="reported",
                    **usage,
                    **costs,
                    charged_usd=sum(costs.values()) if costs else None,
                    request_cost_multiplier=self._request_cost_multiplier(
                        usage["input_tokens"], model
                    ),
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                entry["status"] = "unknown_usage"
        self.save()

    def _prices_for(self, model):
        return self.model_prices.get(model, self)

    def _request_cost_multiplier(self, input_tokens, model=None):
        """Use the inclusive reported input total to select this request's rates."""
        prices = self._prices_for(model)
        if (
            prices.long_context_threshold is not None
            and input_tokens > prices.long_context_threshold
        ):
            return prices.long_context_multiplier
        return 1.0

    def _cost_breakdown(self, usage, model=None):
        prices = self._prices_for(model)
        if prices.input_rate is None:
            return {}
        if usage["cache_read_tokens"] is None and prices.input_rate != prices.cache_read_rate:
            raise ValueError("cached token usage is required for distinct cache prices")
        if usage["thinking_tokens"] is None and prices.output_rate != prices.thinking_rate:
            raise ValueError("reasoning token usage is required for distinct generation prices")
        rates = {
            category: getattr(prices, rate_name(category))
            for category in (*_INPUT_CATEGORIES, "output", "thinking")
        }
        rates.update(
            unattributed_input=prices.input_rate,
            unattributed_completion=prices.output_rate,
        )
        multiplier = self._request_cost_multiplier(usage["input_tokens"], model)
        return {
            f"{category}_usd": (usage[f"{category}_tokens"] or 0) * rate * multiplier / 1e6
            for category, rate in rates.items()
        }
