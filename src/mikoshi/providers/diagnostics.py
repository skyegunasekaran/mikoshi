"""Read-only model request diagnostics; thresholds never alter execution settings."""

from __future__ import annotations

import math
import statistics


def _number(value):
    return (
        value
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        else None
    )


def _at(values, index):
    return values[index] if isinstance(values, (list, tuple)) and index < len(values) else None


def summarize_request_diagnostics(metrics, *, slow_request_seconds=60.0):
    """Distinguish elapsed requests, measured reasoning and missing usage evidence.

    Provider intervals include queue and transport time. Neither token ratios nor
    a slow interval establish how much wall time was spent reasoning.
    """
    requests = []
    for event in metrics.get("timing_events", ()):
        names = event.get("request_names", ())
        for index, name in enumerate(names):
            interval = _at(event.get("request_intervals"), index)
            duration = None
            if isinstance(interval, (list, tuple)) and len(interval) == 2:
                start, end = _number(interval[0]), _number(interval[1])
                if start is not None and end is not None and end >= start:
                    duration = end - start
            usage = _at(event.get("usage_measurements"), index) or {}
            transport = _at(event.get("transport"), index) or {}
            metadata = _at(event.get("request_metadata"), index) or {}
            completion = _number(usage.get("completion_tokens"))
            reasoning = _number(usage.get("reasoning_tokens"))
            requests.append(
                {
                    "name": name,
                    "agent_id": metadata.get("agent_id"),
                    "stage": _at(event.get("request_stages"), index),
                    "action": _at(event.get("chosen_actions"), index),
                    "elapsed_seconds": duration,
                    "queue_seconds": _number(transport.get("queue_seconds")),
                    "input_tokens": _number(usage.get("input_tokens")),
                    "completion_tokens": completion,
                    "reasoning_tokens": reasoning,
                    "reasoning_measurement": usage.get("reasoning_measurement", "unavailable"),
                    "reasoning_fraction_of_completion": reasoning / completion
                    if reasoning is not None and completion and reasoning <= completion
                    else None,
                    "finish_reason": _at(event.get("finish_reasons"), index),
                    "context_characters": metadata.get("context_characters"),
                }
            )

    def distribution(key):
        values = [r[key] for r in requests if r[key] is not None]
        return {
            "measured_requests": len(values),
            "missing_requests": len(requests) - len(values),
            "median": statistics.median(values) if values else None,
            "maximum": max(values) if values else None,
            "sum": sum(values) if values else None,
        }

    return {
        "request_count": len(requests),
        "input_tokens": distribution("input_tokens"),
        "elapsed_seconds": distribution("elapsed_seconds"),
        "reasoning_tokens": distribution("reasoning_tokens"),
        "slow_request_threshold_seconds": slow_request_seconds,
        "slow_requests": [
            r
            for r in requests
            if r["elapsed_seconds"] is not None and r["elapsed_seconds"] >= slow_request_seconds
        ],
        "interpretation": "Request elapsed time includes queue and transport. Reasoning shares describe measured output tokens, not wall-time attribution. Missing measurements remain null. Diagnostics do not change model settings or cancel requests.",
    }
