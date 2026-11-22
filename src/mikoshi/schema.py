"""The public input/output contract. SQLite and runtime objects are implementation details."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from mikoshi.persistence.content import (
    ContentStore,
    content_bytes,
    content_reference,
    read_content,
)

RUN_FILE = "run.json"
RECORDS_FILE = "records.jsonl"


def now():
    return datetime.now(timezone.utc).isoformat()


def envelope(kind, **fields):
    return {"kind": kind, **fields}


def read_document(path, kind=None):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    document_kind = value.get("kind") if isinstance(value, dict) else None
    if document_kind not in ("input", "run", "record", "evaluation"):
        raise ValueError(f"expected a Mikoshi document with a recognized kind: {path}")
    if kind is not None and document_kind != kind:
        raise ValueError(f"expected document kind {kind!r}: {path}")
    return value


def update_run(root, **changes):
    from mikoshi.persistence.exports import write_json

    path = Path(root) / RUN_FILE
    value = read_document(path, "run")
    value.update(changes, updated_at=now())
    write_json(path, value)
    return value


def create_run(root, *, scope="trial", input=None, started_at=None):
    from mikoshi.persistence.exports import write_json

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / RUN_FILE).exists():
        raise ValueError(f"{root} already contains a run")
    value = envelope(
        "run",
        id=uuid.uuid4().hex,
        scope=scope,
        status="running",
        execution_status="running",
        phase="preparing",
        started_at=started_at or now(),
        updated_at=now(),
        finished_at=None,
        stop_reason=None,
        input=input,
        resolved={},
        provenance={},
        summary={},
        budget=None,
        provider_metrics=None,
        progress=None,
        agents={},
        candidate=None,
        evaluation=None,
        evaluations=[],
        trials=[],
        score_summary={},
        evidence={
            "schema": "schema.json",
            "records": RECORDS_FILE,
            "database": ".internal/run.sqlite" if scope == "trial" else None,
            "workspace": "workspace" if scope == "trial" else None,
            "content": "content" if scope == "trial" else None,
            "record_count": None,
        },
        error=None,
    )
    write_json(root / RUN_FILE, value)
    write_json(root / "schema.json", json_schema())
    (root / RECORDS_FILE).touch()
    return value


class RecordWriter:
    """One append-only operator stream; sequence is recording order, not causal order."""

    def __init__(self, root, run_id):
        self.path = Path(root) / RECORDS_FILE
        self.run_id = run_id
        self.sequence = 0
        self.content = ContentStore(root)
        self._lock = threading.RLock()
        self._observations = {}

    def append(self, record_type, data, *, source_id=None, agent_id=None, activation_id=None):
        with self._lock:
            data = self._archive(record_type, dict(data))
            record = envelope(
                "record",
                id=f"{self.run_id}:{self.sequence + 1}",
                run_id=self.run_id,
                sequence=self.sequence + 1,
                recorded_at=now(),
                type=record_type,
                source_id=source_id,
                agent_id=agent_id,
                activation_id=activation_id,
                data=data,
            )
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self.sequence += 1
            return record

    def _archive(self, record_type, data):
        fields = {
            "exchange": ("messages", "state"),
            "decision": (
                "instructions",
                "model_input",
                "history",
                "compaction_state",
                "schema",
                "state",
            ),
            "notification": ("message",),
            "progress": ("checkpoint",),
        }.get(record_type, ())
        for field in fields:
            if field in data:
                data[field + "_ref"] = self.content.put(data.pop(field))
        if record_type == "decision":
            self._observations[data["decision_id"]] = data
        if "payload" in data and record_type == "request":
            payload = data.pop("payload")
            if data["kind"] == "provider_request_submitted":
                data["included_content"] = self._included_content(
                    payload, data.get("decision_id"), data.pop("tool_result_paths", ())
                )
            data["payload_ref"] = self.content.put(payload)
        if "timing" in data and record_type == "request":
            data["timing_ref"] = self.content.put(data.pop("timing"))
        if record_type == "response":
            for field in ("body", "raw_text"):
                if field in data:
                    data["body_ref"] = self.content.put(data.pop(field))
                    data["body_format"] = "json" if field == "body" else "text"
        return data

    def prepare_progress(self, event):
        with self._lock:
            return self._archive("progress", dict(event))

    def _included_content(self, payload, decision_id, tool_result_paths):
        """Describe literal submitted fields, without inferring receipt or comprehension."""
        included = []
        observation = self._observations.get(decision_id)

        from mikoshi.persistence.evidence import pointer_value

        for item in tool_result_paths:
            included.append(
                {
                    "kind": "tool_result",
                    "path": item["path"],
                    "tool_call_id": item["tool_call_id"],
                    "content_ref": self.content.put(pointer_value(payload, item["path"])),
                }
            )

        def walk(value, pointer):
            if isinstance(value, dict):
                for key, child in value.items():
                    walk(child, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, pointer + "/" + str(index))
            elif isinstance(value, str) and observation:
                reference = content_reference(content_bytes(value))
                source_fields = [
                    field
                    for field in ("model_input", "compaction_state", "state")
                    if reference["content_id"]
                    == observation.get(field + "_ref", {}).get("content_id")
                ]
                if source_fields:
                    entry = {
                        "kind": "observation",
                        "path": pointer,
                        "content_ref": reference,
                        "source_fields": source_fields,
                    }
                    # Ordinary input and compaction restoration carry the same
                    # prepared inbox, even when stable context makes their text differ.
                    entry["notification_sources"] = observation.get("metadata", {}).get(
                        "observation_notification_sources", []
                    )
                    included.append(entry)

        walk(payload, "")
        return included

    def provider_event(self, event):
        kind = event["kind"]
        record_type = (
            "usage"
            if kind == "provider_usage_recorded"
            else "response"
            if kind.startswith("provider_response_")
            else "request"
        )
        return self.append(
            record_type,
            event,
            source_id=f"request:{event['request_id']}:{kind}",
            agent_id=event.get("agent_id"),
            activation_id=event.get("activation_id"),
        )

    def attach(self, store):
        def committed(event):
            from mikoshi.persistence.exports import queued_notifications

            self.append(
                "action",
                event,
                source_id=f"event:{event['id']}",
                agent_id=event["agent_id"],
                activation_id=event["activation_id"],
            )
            for notification in queued_notifications(store, event["id"]):
                self.append(
                    "notification",
                    {"event": "queued", **notification},
                    source_id=f"notification:{notification['id']}",
                    agent_id=notification["recipient"],
                )

        store.observe_committed_events(committed)
        store.observe_exchanges(
            lambda exchange: self.append(
                "exchange",
                exchange,
                source_id=f"exchange:{exchange['sequence']}",
                agent_id=exchange["agent_id"],
                activation_id=int(exchange["exchange_key"].split(":")[0]),
            )
        )


@lru_cache(maxsize=1)
def json_schema():
    """Return the bundled contract as JSON Schema Draft 2020-12."""
    from dataclasses import fields

    from mikoshi.configuration import RUN_OPTIONS
    from mikoshi.isolation.docker import DockerSettings
    from mikoshi.providers.limits import ProviderDeployment
    from mikoshi.providers.pricing import TokenPrices

    def obj(properties, required=()):
        return {
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        }

    def document(kind, properties, required):
        return obj(
            {"kind": {"const": kind}, **properties},
            ("kind", *required),
        )

    text = {"type": "string"}
    nullable_text = {"type": ["string", "null"]}
    arbitrary = {"type": "object"}
    options = {}
    for name, (kind, _) in RUN_OPTIONS.items():
        if name in {"task", "agents", "problem"}:
            continue
        scalar = (
            "boolean"
            if kind is bool
            else "number"
            if kind is float
            else "string"
            if kind in (str, Path)
            else "integer"
        )
        options[name] = {"type": [scalar, "null"]}
        if scalar == "integer":
            options[name]["minimum"] = 1
        elif name in {"budget_usd", "wall_time_seconds"}:
            options[name]["exclusiveMinimum"] = 0
    for name in ("repository", "model"):
        options[name] = {"type": "string", "minLength": 1}
    # Archived experiment inputs remain readable without their installed extensions.
    for name in ("provider", "team_protocol"):
        options[name] = {"type": "string", "minLength": 1}
    problem = obj(
        {
            "objective": obj(
                {
                    "id": text,
                    "title": text,
                    "description": text,
                    "constraints": {"type": "array", "items": text},
                },
                ("id", "title", "description"),
            ),
            "limits": obj(
                {
                    "max_agents": {"type": "integer", "minimum": 1},
                }
            ),
            "team": obj(
                {
                    "capabilities": {
                        "type": "array",
                        "items": {"enum": ["read_workspace", "write_artifact"]},
                    },
                    "members": {
                        "type": "array",
                        "items": obj({"id": text, "instructions": text}, ("id",)),
                    },
                }
            ),
            "scheduled_notifications": {
                "type": "array",
                "items": obj(
                    {
                        "id": text,
                        "at_minutes": {"type": "number", "minimum": 0},
                        "recipient": text,
                        "message": text,
                    },
                    ("id", "at_minutes", "recipient", "message"),
                ),
            },
        },
        ("objective",),
    )
    rates = {f.name: {"type": ["number", "null"]} for f in fields(TokenPrices)}
    for name in ("input_rate", "cache_read_rate", "output_rate", "thinking_rate"):
        rates[name] = {"type": "number", "minimum": 0}
    rates["long_context_threshold"] = {"type": ["integer", "null"], "minimum": 1}
    rates["long_context_multiplier"] = {"type": "number", "exclusiveMinimum": 0}
    pricing = obj(
        {
            **rates,
            "models": {
                "type": "object",
                "additionalProperties": obj(
                    rates, ("input_rate", "cache_read_rate", "output_rate", "thinking_rate")
                ),
            },
        },
        ("input_rate", "cache_read_rate", "output_rate", "thinking_rate"),
    )
    docker = {
        f.name: {
            "type": "number"
            if f.name == "resource_target_percent"
            else "integer"
            if f.name == "pids_limit"
            else ["string", "null"]
        }
        for f in fields(DockerSettings)
    }
    provider = {
        f.name: {
            "type": ["integer", "null"]
            if f.name == "concurrency"
            else ["object"]
            if f.name == "agent_models"
            else ["string", "null"]
        }
        for f in fields(ProviderDeployment)
    }
    provider["options"] = arbitrary
    provider["agent_models"] = {"type": "object", "additionalProperties": text}
    input_schema = document(
        "input",
        {
            "run": obj(options, ("repository", "model")),
            "problem": problem,
            "pricing": pricing,
            "deployment": obj(
                {
                    "docker": obj(docker),
                    "provider": obj(provider),
                    "enable_commands": {"type": "boolean"},
                    "command_limits": obj(
                        {
                            name: {"type": "integer", "minimum": 1}
                            for name in ("default_timeout_seconds", "max_timeout_seconds")
                        }
                    ),
                }
            ),
            "evaluation": obj({"scorer": text}),
        },
        ("run", "problem", "pricing"),
    )
    record = document(
        "record",
        {
            "id": text,
            "run_id": text,
            "sequence": {"type": "integer", "minimum": 1},
            "recorded_at": text,
            "type": {
                "enum": [
                    "progress",
                    "action",
                    "exchange",
                    "state",
                    "decision",
                    "request",
                    "response",
                    "usage",
                    "notification",
                ]
            },
            "source_id": nullable_text,
            "agent_id": nullable_text,
            "activation_id": {"type": ["integer", "null"]},
            "data": arbitrary,
        },
        (
            "id",
            "run_id",
            "sequence",
            "recorded_at",
            "type",
            "source_id",
            "agent_id",
            "activation_id",
            "data",
        ),
    )
    content_ref = obj(
        {
            "content_id": {"type": "string", "pattern": "^sha256:[a-f0-9]{64}$"},
            "path": {"type": "string", "pattern": "^content/[a-f0-9]{2}/[a-f0-9]{64}\\.json$"},
            "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "bytes": {"type": "integer", "minimum": 0},
            "media_type": {"const": "application/json"},
        },
        ("content_id", "path", "sha256", "bytes", "media_type"),
    )
    reference = {"$ref": "#/$defs/content_ref"}
    action = obj(
        {
            **{
                key: {"type": "integer"}
                for key in (
                    "id",
                    "tick",
                    "commit_ordinal",
                    "activation_id",
                    "causal_wave_id",
                    "decision_round",
                    "activation_step",
                )
            },
            "agent_id": text,
            "action_type": text,
            "outcome": text,
            "resource_id": nullable_text,
            "payload": arbitrary,
        },
        ("id", "tick", "activation_id", "agent_id", "action_type", "outcome", "payload"),
    )
    exchange = obj(
        {
            "sequence": {"type": "integer"},
            "agent_id": text,
            "conversation_id": text,
            "exchange_key": {"type": "string", "pattern": "^[0-9]+:[0-9]+$"},
            "decision_id": nullable_text,
            "request_id": nullable_text,
            "request_ids": {"type": "array", "items": text},
            "tool_calls": {
                "type": "array",
                "items": obj(
                    {
                        "index": {"type": "integer", "minimum": 0},
                        "id": nullable_text,
                        "name": nullable_text,
                    },
                    ("index", "id", "name"),
                ),
            },
            "messages_ref": reference,
            "state_ref": reference,
        },
        (
            "sequence",
            "agent_id",
            "conversation_id",
            "exchange_key",
            "decision_id",
            "request_id",
            "request_ids",
            "tool_calls",
            "messages_ref",
            "state_ref",
        ),
    )
    decision = obj(
        {
            "kind": {"const": "decision_prepared"},
            "decision_id": text,
            "agent_id": text,
            "activation_id": {"type": "integer"},
            "activation_step": {"type": "integer"},
            "conversation_id": nullable_text,
            "stage": text,
            "name": text,
            "metadata": arbitrary,
            **{
                field + "_ref": reference
                for field in (
                    "instructions",
                    "model_input",
                    "history",
                    "compaction_state",
                    "schema",
                    "state",
                )
            },
        },
        (
            "kind",
            "decision_id",
            "agent_id",
            "activation_id",
            "activation_step",
            "metadata",
            "instructions_ref",
            "model_input_ref",
            "history_ref",
            "schema_ref",
            "state_ref",
        ),
    )
    provider_fields = {
        "kind": text,
        "request_id": text,
        "decision_id": text,
        "previous_request_id": nullable_text,
        "provider": text,
        "model": text,
        "endpoint": text,
        "agent_id": nullable_text,
        "activation_id": {"type": ["integer", "null"]},
        "activation_step": {"type": ["integer", "null"]},
        "observed_at": text,
        "redacted": {"type": "boolean"},
        "submitted": {"type": "boolean"},
    }
    provider_required = (
        "kind",
        "request_id",
        "decision_id",
        "previous_request_id",
        "provider",
        "model",
        "endpoint",
        "observed_at",
        "redacted",
    )
    included_content = obj(
        {
            "kind": {"enum": ["observation", "tool_result"]},
            "path": text,
            "content_ref": reference,
            "tool_call_id": nullable_text,
            "notification_sources": {"type": "array", "items": arbitrary},
            "source_fields": {
                "type": "array",
                "items": {"enum": ["model_input", "compaction_state", "state"]},
            },
        },
        ("kind", "path", "content_ref"),
    )
    request = obj(
        {
            **provider_fields,
            "kind": {
                "enum": [
                    "provider_request_prepared",
                    "provider_request_submitted",
                    "provider_request_failed",
                    "provider_request_completed",
                ]
            },
            "payload_ref": reference,
            "included_content": {"type": "array", "items": included_content},
            "response_received": {"type": "boolean"},
            "error_type": text,
            "error": text,
            "timing_ref": reference,
            "timing_index": {"type": "integer", "minimum": 0},
        },
        provider_required,
    )
    request["allOf"] = [
        {
            "if": {
                "properties": {
                    "kind": {"enum": ["provider_request_prepared", "provider_request_submitted"]}
                }
            },
            "then": {"required": ["payload_ref", "submitted"]},
        },
        {
            "if": {"properties": {"kind": {"const": "provider_request_completed"}}},
            "then": {"required": ["timing_ref", "timing_index", "submitted"]},
        },
    ]
    response = obj(
        {
            **provider_fields,
            "kind": {"enum": ["provider_response_received", "provider_response_decoded"]},
            "status_code": {"type": "integer"},
            "content_type": nullable_text,
            "provider_response_id": nullable_text,
            "body_ref": reference,
            "body_format": {"enum": ["json", "text"]},
        },
        (*provider_required, "status_code", "body_ref", "body_format"),
    )
    usage = obj(
        {
            **provider_fields,
            "kind": {"const": "provider_usage_recorded"},
            "budget_entry_id": {"type": "integer"},
            "entry": arbitrary,
        },
        (*provider_required, "budget_entry_id", "entry"),
    )
    notification = obj(
        {
            "event": {"const": "queued"},
            "id": {"type": "integer"},
            "recipient": text,
            "sender": text,
            "kind": text,
            "source_ref": text,
            "page_ref": nullable_text,
            "message_ref": reference,
            "wake_requested": {"enum": [0, 1]},
            "event_id": {"type": "integer"},
        },
        (
            "event",
            "id",
            "recipient",
            "sender",
            "kind",
            "source_ref",
            "message_ref",
            "wake_requested",
            "event_id",
        ),
    )
    state = obj(
        {
            key: {"type": "array", "items": arbitrary}
            for key in (
                "agents",
                "pages",
                "projects",
                "commands",
                "activations",
                "triggers",
                "inbox",
            )
        },
        ("agents", "pages", "commands", "activations", "triggers", "inbox"),
    )
    state["properties"]["recording_errors"] = {"type": "array", "items": text}
    state["required"] = list(state["properties"])
    record["allOf"] = [
        {
            "if": {"properties": {"type": {"const": kind}}},
            "then": {"properties": {"data": {"$ref": f"#/$defs/{kind}"}}},
        }
        for kind in (
            "action",
            "exchange",
            "state",
            "decision",
            "request",
            "response",
            "usage",
            "notification",
        )
    ]
    score = obj(
        {
            "value": {"type": ["string", "boolean", "number"]},
            "explanation": text,
            "metadata": arbitrary,
        },
        ("value",),
    )
    evaluation = document(
        "evaluation",
        {
            "id": text,
            "run_id": text,
            "evaluated_at": text,
            "status": {"enum": ["completed", "failed", "not_configured"]},
            "scores": {"type": "object", "additionalProperties": score},
            "scorer": {"type": ["object", "null"]},
            "candidate_sha256": nullable_text,
            "evaluation_path": text,
            "error": arbitrary,
        },
        ("id", "run_id", "evaluated_at", "status", "scores", "candidate_sha256", "evaluation_path"),
    )
    run = document(
        "run",
        {
            "id": text,
            "scope": {"enum": ["trial", "experiment"]},
            "status": {"enum": ["running", "completed", "stopped", "failed", "interrupted"]},
            "execution_status": {
                "enum": ["running", "completed", "stopped", "failed", "interrupted"]
            },
            "phase": text,
            "started_at": text,
            "updated_at": text,
            "finished_at": nullable_text,
            "stop_reason": nullable_text,
            "input": {"anyOf": [{"$ref": "#/$defs/input"}, {"type": "null"}]},
            **{
                key: arbitrary
                for key in (
                    "resolved",
                    "provenance",
                    "summary",
                    "agents",
                    "score_summary",
                    "evidence",
                )
            },
            **{
                key: {"type": ["object", "null"]}
                for key in ("budget", "provider_metrics", "progress", "candidate", "evaluation")
            },
            "evaluations": {"type": "array", "items": text},
            "trials": {"type": "array", "items": arbitrary},
            "error": nullable_text,
        },
        ("id", "scope", "status", "phase", "started_at", "updated_at", "input", "evidence"),
    )
    run["properties"]["candidate"] = {
        "anyOf": [
            {"type": "null"},
            obj(
                {
                    "path": {"const": "candidate"},
                    "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                    "input_sha256": text,
                    "evidence_sha256": {
                        "type": "object",
                        "additionalProperties": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                    },
                    "source_repository": text,
                },
                ("path", "sha256", "input_sha256", "evidence_sha256", "source_repository"),
            ),
        ]
    }
    run["properties"]["evaluation"] = {"anyOf": [{"type": "null"}, {"$ref": "#/$defs/evaluation"}]}
    run["properties"]["evidence"] = obj(
        {
            "schema": {"const": "schema.json"},
            "records": {"const": RECORDS_FILE},
            "database": {"type": ["string", "null"]},
            "workspace": {"type": ["string", "null"]},
            "content": {"type": ["string", "null"]},
            "record_count": {"type": ["integer", "null"], "minimum": 0},
        },
        ("schema", "records", "database", "workspace"),
    )
    run["required"] = list(run["properties"])
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Mikoshi input and output",
        "$defs": {
            "input": input_schema,
            "run": run,
            "record": record,
            "evaluation": evaluation,
            "action": action,
            "exchange": exchange,
            "state": state,
            "content_ref": content_ref,
            "decision": decision,
            "request": request,
            "response": response,
            "usage": usage,
            "notification": notification,
        },
        "oneOf": [{"$ref": f"#/$defs/{kind}"} for kind in ("input", "run", "record", "evaluation")],
    }


def validate_document(value, *, schema=None):
    from jsonschema import Draft202012Validator

    json.dumps(value, allow_nan=False)
    if schema is None:
        schema = json_schema()
    kind = value.get("kind") if isinstance(value, dict) else None
    if kind in ("input", "run", "record", "evaluation"):
        schema = {"$defs": schema["$defs"], "$ref": f"#/$defs/{kind}"}
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: str(e.path))
    if errors:
        error = errors[0]
        raise ValueError(f"schema validation failed at {list(error.path)}: {error.message}")


def validate_bundle(path):
    """Offline structural validation, including stream identity/order and evidence references."""
    path = Path(path)
    if not path.is_dir():
        value = read_document(path)
        validate_document(value)
        return {"status": "valid", "kind": value["kind"], "path": str(path)}
    run = read_document(path / RUN_FILE, "run")
    validate_document(run)
    from jsonschema import Draft202012Validator, SchemaError

    retained_schema = json.loads(
        bundle_path(path, run["evidence"]["schema"]).read_text(encoding="utf-8")
    )
    try:
        Draft202012Validator.check_schema(retained_schema)
    except SchemaError as error:
        raise ValueError(f"invalid retained JSON Schema: {error.message}") from error
    count = 0
    from mikoshi.persistence.evidence import EvidenceLinks

    links = EvidenceLinks(path)
    records = bundle_path(path, run["evidence"]["records"])
    blobs = {}
    contents = {}

    def verify_references(value):
        if isinstance(value, dict):
            if "content_id" in value:
                identity = value["content_id"]
                if contents.get(identity) != value:
                    read_content(path, value)
                    contents[identity] = dict(value)
            if "blob_path" in value:
                blob = bundle_path(path, "workspace/" + value["blob_path"])
                if blob not in blobs:
                    data = blob.read_bytes()
                    blobs[blob] = (hashlib.sha256(data).hexdigest(), len(data))
                if blobs[blob] != (value.get("sha256"), value.get("bytes")):
                    raise ValueError(f"content blob fingerprint mismatch: {blob}")
            for child in value.values():
                verify_references(child)
        elif isinstance(value, list):
            for child in value:
                verify_references(child)

    with records.open(encoding="utf-8") as stream:
        for count, line in enumerate(stream, 1):
            record = json.loads(line)
            # A saved bundle carries the contract its records were written under.
            # Validate historical evidence without rewriting it to today's shape.
            validate_document(record, schema=retained_schema)
            if (
                record["run_id"] != run["id"]
                or record["sequence"] != count
                or record["id"] != f"{run['id']}:{count}"
            ):
                raise ValueError(f"record identity/order mismatch at line {count}")
            verify_references(record["data"])
            links.observe(record)
            if record["type"] in {"action", "exchange"}:
                data = record["data"]
                activation = (
                    data["activation_id"]
                    if record["type"] == "action"
                    else int(data["exchange_key"].split(":")[0])
                )
                if record["agent_id"] != data["agent_id"] or record["activation_id"] != activation:
                    raise ValueError(f"record actor/activation mismatch at line {count}")
            if record["type"] == "state":
                if record["data"]["recording_errors"]:
                    raise ValueError("run reports incomplete record export")
                for command in record["data"]["commands"]:
                    if (
                        command.get("log_path")
                        and not bundle_path(path, command["log_path"]).is_file()
                    ):
                        raise ValueError(f"missing command log: {command['log_path']}")
    expected_count = run["evidence"].get("record_count")
    if expected_count is not None and count != expected_count:
        raise ValueError(f"record count mismatch: expected {expected_count}, found {count}")
    links.finish(run)
    if run.get("candidate"):
        from mikoshi.persistence.artifacts import load_verified_artifacts

        load_verified_artifacts(path)
    latest = None
    for location in run["evaluations"]:
        evaluation_path = (
            Path(location) if Path(location).is_absolute() else bundle_path(path, location)
        )
        evaluation = read_document(evaluation_path, "evaluation")
        validate_document(evaluation)
        if evaluation["run_id"] != run["id"]:
            raise ValueError("evaluation belongs to another run")
        latest = evaluation
    if run["evaluation"] != latest:
        raise ValueError("latest evaluation does not match evaluation history")
    for trial in run["trials"]:
        if trial.get("run_dir"):
            child = bundle_path(path, trial["run_dir"])
            if (child / RUN_FILE).is_file():
                validate_bundle(child)
            elif trial["status"] in {"completed", "stopped"}:
                raise ValueError(f"missing trial output: {trial['id']}")
    return {"status": "valid", "kind": "run", "run_id": run["id"], "records": count}


def bundle_path(root, reference):
    root = Path(root).resolve()
    reference = Path(reference)
    destination = (root / reference).resolve()
    if reference.is_absolute() or destination == root or root not in destination.parents:
        raise ValueError(f"evidence reference must stay inside its bundle: {reference}")
    return destination


def iter_records(root, *, agent_id=None, record_type=None, resolve_content=False):
    """Stream portable public records without importing the database implementation."""
    root = Path(root)
    run = read_document(root / RUN_FILE, "run")
    with bundle_path(root, run["evidence"]["records"]).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if agent_id is not None and record["agent_id"] != agent_id:
                continue
            if record_type is not None and record["type"] != record_type:
                continue
            if resolve_content:
                record = resolve_record(root, record)
            yield record


def resolve_record(root, record):
    """Return a reader view with referenced top-level data fields expanded."""
    data = dict(record["data"])
    for key, value in tuple(data.items()):
        if key.endswith("_ref") and isinstance(value, dict) and "content_id" in value:
            data[key[:-4]] = read_content(root, value)
    return {**record, "data": data}
