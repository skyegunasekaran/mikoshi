"""Bounded request detail with lifetime aggregates independent of the retained tail."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mikoshi.providers.diagnostics import summarize_request_diagnostics

TIMING_HISTORY_LIMIT = 256


def merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return disjoint spans so concurrent requests are counted once."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted((start, end) for start, end in intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


@dataclass
class IntervalTotals:
    """Compact settled prefixes; keep only spans an older open request may overlap."""

    settled_seconds: float = 0.0
    spans: list[tuple[float, float]] = field(default_factory=list)

    def record(self, intervals, *, pending_since: float = float("inf")) -> None:
        merged = merge_intervals([*self.spans, *intervals])
        self.spans = []
        for start, end in merged:
            if end <= pending_since:
                self.settled_seconds += end - start
            else:
                self.spans.append((start, end))

    @property
    def seconds(self) -> float:
        return self.settled_seconds + sum(end - start for start, end in self.spans)


class RequestTelemetry:
    """Accumulate lifetime totals and expose only a bounded tail of raw details.

    Exact medians require retaining all observations. Once the tail drops data,
    ``median`` is null and ``sample_median`` explicitly describes the tail.
    Original events are separately emitted as durable request evidence.
    """

    def __init__(self, history_limit: int = TIMING_HISTORY_LIMIT):
        self.history_limit = history_limit
        self.events = deque(maxlen=history_limit)
        self.event_count = 0
        self.request_count = 0
        self.slow_request_count = 0
        self.distributions: dict[str, dict[str, Any]] = {}
        self.generation = IntervalTotals()
        self._stages: dict[str, dict[str, Any]] = {}
        self._stage_intervals: dict[str, IntervalTotals] = {}

    def record(self, event: Mapping[str, Any], *, pending: Mapping[str, Sequence[float]]) -> None:
        self.events.append(event)
        self.event_count += 1
        starts = [start for values in pending.values() for start in values]
        self.generation.record(
            event["request_intervals"], pending_since=min(starts, default=float("inf"))
        )
        by_stage = {}
        for stage, input_count, output_count, interval in zip(
            event["request_stages"],
            event["input_token_counts"],
            event["output_token_counts"],
            event["request_intervals"],
        ):
            stage = str(stage)
            bucket = self._stages.setdefault(
                stage, {"attempted_requests": 0, "input_tokens": 0, "output_tokens": 0}
            )
            bucket["attempted_requests"] += 1
            bucket["input_tokens"] += input_count
            bucket["output_tokens"] += output_count
            by_stage.setdefault(stage, []).append(tuple(interval))
        for stage, intervals in by_stage.items():
            self._stage_intervals.setdefault(stage, IntervalTotals()).record(
                intervals, pending_since=min(pending.get(stage, ()), default=float("inf"))
            )
        diagnostics = summarize_request_diagnostics({"timing_events": [event]})
        self.request_count += diagnostics["request_count"]
        self.slow_request_count += len(diagnostics["slow_requests"])
        for name in ("input_tokens", "elapsed_seconds", "reasoning_tokens"):
            source = diagnostics[name]
            total = self.distributions.setdefault(
                name, {"measured_requests": 0, "missing_requests": 0, "maximum": None, "sum": None}
            )
            for count in ("measured_requests", "missing_requests"):
                total[count] += source[count]
            if source["sum"] is not None:
                total["sum"] = (total["sum"] or 0) + source["sum"]
                total["maximum"] = max(
                    source["maximum"], total["maximum"] if total["maximum"] is not None else 0
                )

    def stages(self) -> dict[str, Any]:
        result = copy.deepcopy(self._stages)
        for stage, bucket in result.items():
            seconds = self._stage_intervals[stage].seconds
            bucket["wall_generation_seconds"] = seconds
            bucket["output_tokens_per_second"] = (
                round(bucket["output_tokens"] / seconds, 3) if seconds > 0 else None
            )
        return result

    def diagnostics(self) -> dict[str, Any]:
        result = summarize_request_diagnostics({"timing_events": list(self.events)})
        retained_count = result["request_count"]
        truncated = retained_count != self.request_count
        result["retained_request_count"] = retained_count
        result["request_count"] = self.request_count
        result["slow_request_count"] = self.slow_request_count
        result["slow_requests_truncated"] = len(result["slow_requests"]) != self.slow_request_count
        for name, total in self.distributions.items():
            sample = result[name]
            result[name] = {
                **total,
                "mean": total["sum"] / total["measured_requests"]
                if total["measured_requests"]
                else None,
                "median": None if truncated else sample["median"],
                "median_scope": "retained_tail" if truncated else "all_requests",
                "sample_median": sample["median"],
                "sample_measured_requests": sample["measured_requests"],
            }
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "timing_events": copy.deepcopy(list(self.events)),
            "timing_event_count": self.event_count,
            "timing_history_limit": self.history_limit,
            "timing_events_truncated": self.event_count > len(self.events),
            "stages": self.stages(),
            "request_diagnostics": self.diagnostics(),
        }
