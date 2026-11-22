"""Provider history bounds preserve accounting and portable request evidence."""

import json
from dataclasses import replace

import httpx
import pytest
from support.provider import MockDecisionProvider, tool_calls

from mikoshi.contracts import action_schema
from mikoshi.decisions import DecisionRequest
from mikoshi.providers.telemetry import TIMING_HISTORY_LIMIT, RequestTelemetry
from mikoshi.schema import RecordWriter, create_run, iter_records, update_run, validate_bundle


def test_long_request_history_keeps_lifetime_totals_and_complete_archived_timings(tmp_path):
    run = create_run(tmp_path)
    writer = RecordWriter(tmp_path, run["id"])
    request = DecisionRequest(
        "history",
        "Work",
        "Current state",
        {"oneOf": [action_schema("list_activity", {}, required=[])]},
        conversation_id="peer",
    )
    count = TIMING_HISTORY_LIMIT + 3
    provider = MockDecisionProvider(
        "fixture",
        api_key="host-secret",
        max_tokens=100,
        context_window_tokens=1000,
        http_transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {"tool_calls": tool_calls({"action": "list_activity"})},
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                },
            )
        ),
    )
    provider.set_evidence_callback(writer.provider_event)
    try:
        for index in range(count):
            metadata = {
                "decision_id": f"decision-{index}",
                "agent_id": "peer",
                "activation_id": 1,
                "activation_step": index + 1,
            }
            writer.append(
                "decision",
                {
                    "kind": "decision_prepared",
                    **metadata,
                    "metadata": metadata,
                    "instructions": request.instructions,
                    "model_input": request.input_text,
                    "history": [],
                    "schema": request.schema,
                    "state": {},
                },
                agent_id="peer",
                activation_id=1,
            )
            result = provider.generate_batch([replace(request, metadata=metadata)])[0]
            assert result == {"action": "list_activity"}
        metrics = provider.metrics
    finally:
        provider.close()
    update_run(
        tmp_path,
        provider_metrics=metrics,
        evidence={
            **run["evidence"],
            "record_count": writer.sequence,
        },
    )
    assert validate_bundle(tmp_path)["records"] == writer.sequence
    completed = [
        record["data"]
        for record in iter_records(tmp_path, resolve_content=True)
        if record["data"]["kind"] == "provider_request_completed"
    ]
    assert len(completed) == count
    assert len(metrics["timing_events"]) == TIMING_HISTORY_LIMIT
    assert len(metrics["batch_sizes"]) == TIMING_HISTORY_LIMIT
    assert metrics["timing_event_count"] == metrics["requests"] == count
    assert metrics["timing_events_truncated"] is metrics["batch_sizes_truncated"] is True
    assert metrics["input_tokens"] == metrics["stages"]["actor"]["input_tokens"] == count * 2
    assert metrics["output_tokens"] == metrics["stages"]["actor"]["output_tokens"] == count * 3
    assert metrics["stages"]["actor"]["attempted_requests"] == count
    diagnostics = metrics["request_diagnostics"]
    assert diagnostics["request_count"] == count
    assert diagnostics["retained_request_count"] == TIMING_HISTORY_LIMIT
    assert diagnostics["input_tokens"] == {
        "measured_requests": count,
        "missing_requests": 0,
        "maximum": 2,
        "sum": count * 2,
        "mean": 2,
        "median": None,
        "median_scope": "retained_tail",
        "sample_median": 2,
        "sample_measured_requests": TIMING_HISTORY_LIMIT,
    }
    elapsed = []
    for item in completed:
        event, index = item["timing"], item["timing_index"]
        assert event["request_ids"][index] == item["request_id"]
        assert event["decision_ids"][index] == item["decision_id"]
        start, end = event["request_intervals"][index]
        elapsed.append(end - start)
    assert metrics["generation_seconds"] == pytest.approx(sum(elapsed))
    assert diagnostics["elapsed_seconds"]["sum"] == pytest.approx(sum(elapsed))
    assert diagnostics["elapsed_seconds"]["maximum"] == max(elapsed)
    assert diagnostics["elapsed_seconds"]["mean"] == pytest.approx(sum(elapsed) / count)
    assert metrics["evidence_errors"] == []
    records = list(iter_records(tmp_path))
    first_completion = next(
        record for record in records if record["data"]["kind"] == "provider_request_completed"
    )
    first_completion["data"]["timing_index"] = count
    (tmp_path / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="request timing belongs to another request"):
        validate_bundle(tmp_path)


def timing(start, end, *, stage="actor"):
    return {
        "request_names": ["timing"],
        "request_intervals": [[start, end]],
        "request_stages": [stage],
        "input_token_counts": [2],
        "output_token_counts": [3],
        "usage_measurements": [{"input_tokens": 2, "completion_tokens": 3}],
    }


def test_out_of_order_overlapping_requests_keep_exact_lifetime_stage_wall_times():
    telemetry = RequestTelemetry(history_limit=2)
    telemetry.record(timing(10, 14), pending={"actor": [0]})
    telemetry.record(timing(20, 30), pending={"actor": [0]})
    telemetry.record(timing(0, 35), pending={})
    assert telemetry.generation.seconds == 35
    assert telemetry.generation.spans == []
    assert telemetry.stages()["actor"] == {
        "attempted_requests": 3,
        "input_tokens": 6,
        "output_tokens": 9,
        "wall_generation_seconds": 35,
        "output_tokens_per_second": round(9 / 35, 3),
    }
    assert telemetry.diagnostics()["elapsed_seconds"]["sum"] == 49
    assert telemetry.diagnostics()["elapsed_seconds"]["median"] is None


def test_global_wall_time_and_per_stage_wall_time_have_different_overlap_scopes():
    telemetry = RequestTelemetry()
    telemetry.record(timing(0, 10), pending={"review": [5]})
    telemetry.record(timing(5, 15, stage="review"), pending={})
    assert telemetry.generation.seconds == 15
    assert telemetry.stages()["actor"]["wall_generation_seconds"] == 10
    assert telemetry.stages()["review"]["wall_generation_seconds"] == 10
    assert telemetry.diagnostics()["elapsed_seconds"]["median"] == 10
