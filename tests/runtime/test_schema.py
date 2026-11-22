"""The public contract is readable, portable and rejects incomplete evidence."""

import json
import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from mikoshi.configuration import configuration_document, parse_configuration, run_parser
from mikoshi.schema import (
    RecordWriter,
    create_run,
    iter_records,
    json_schema,
    read_document,
    update_run,
    validate_bundle,
    validate_document,
)


def test_documented_input_and_schema_roundtrip_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    guide = (Path(__file__).resolve().parents[2] / "EXPERIMENTS.md").read_text(encoding="utf-8")
    document = json.loads(re.search(r"```json\n(.*?)\n```", guide, re.S).group(1))
    document["problem"]["objective"]["description"] += " — 🧪"
    Draft202012Validator.check_schema(json_schema())
    validate_document(document)
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    args = parse_configuration(run_parser(), ["--config", str(path)])
    retained = configuration_document(args)
    validate_document(retained)
    assert retained["problem"]["objective"] == document["problem"]["objective"] | {
        "constraints": []
    }
    assert read_document(path, "input") == document
    assert retained["kind"] == "input"
    document["unexpected"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_configuration(run_parser(), ["--config", str(path)])
    with pytest.raises(ValueError, match="schema validation"):
        validate_document(document)


@pytest.mark.parametrize("document", [None, {"kind": "other"}, {"kind": []}])
def test_document_readers_reject_missing_or_invalid_kinds(tmp_path, document):
    path = tmp_path / "document.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="recognized kind"):
        read_document(path)
    with pytest.raises(ValueError, match="schema validation"):
        validate_document(document)
    with pytest.raises(SystemExit):
        parse_configuration(run_parser(), ["--config", str(path)])


def test_document_reader_checks_the_requested_kind(tmp_path):
    create_run(tmp_path)
    with pytest.raises(ValueError, match="expected document kind 'input'"):
        read_document(tmp_path / "run.json", "input")


@pytest.mark.parametrize("corruption", ["owner", "sequence", "tail", "schema"])
def test_validator_rejects_incomplete_or_misattributed_stream(tmp_path, corruption):
    run = create_run(tmp_path)
    writer = RecordWriter(tmp_path, run["id"])
    record = writer.append("progress", {"kind": "run_started"})
    update_run(tmp_path, evidence={**run["evidence"], "record_count": 1})
    assert validate_bundle(tmp_path)["records"] == 1
    assert list(iter_records(tmp_path, record_type="exchange")) == []
    if corruption == "owner":
        record["run_id"] = "another-run"
    elif corruption == "sequence":
        record["sequence"] = 2
    elif corruption == "schema":
        record["unexpected"] = True
    (tmp_path / "records.jsonl").write_text(
        "" if corruption == "tail" else json.dumps(record) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        validate_bundle(tmp_path)


def test_validator_requires_referenced_content_and_recorded_schema(tmp_path):
    run = create_run(tmp_path)
    writer = RecordWriter(tmp_path, run["id"])
    writer.append(
        "progress",
        {"resource_version": {"blob_path": ".cas/missing", "sha256": "0" * 64, "bytes": 1}},
    )
    with pytest.raises(FileNotFoundError):
        validate_bundle(tmp_path)
    (tmp_path / "schema.json").unlink()
    with pytest.raises(FileNotFoundError):
        validate_bundle(tmp_path)


def test_bundle_validator_preserves_historical_state_with_its_retained_schema(tmp_path):
    run = create_run(tmp_path)
    schema_path = tmp_path / "schema.json"
    retained = json.loads(schema_path.read_text(encoding="utf-8"))
    state = retained["$defs"]["state"]
    for field in ("work", "work_transitions"):
        state["properties"][field] = {"type": "array", "items": {"type": "object"}}
        state["required"].append(field)
    schema_path.write_text(json.dumps(retained), encoding="utf-8")
    data = {field: [] for field in state["properties"]}
    data["work"] = [{"id": "work-1", "state": "finished"}]
    record = RecordWriter(tmp_path, run["id"]).append("state", data)
    update_run(tmp_path, evidence={**run["evidence"], "record_count": 1})
    records_path = tmp_path / "records.jsonl"
    original = records_path.read_bytes()

    with pytest.raises(ValueError, match="schema validation"):
        validate_document(record)
    assert validate_bundle(tmp_path)["records"] == 1
    assert records_path.read_bytes() == original

    record["data"]["work"] = "invalid historical state"
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema validation"):
        validate_bundle(tmp_path)


def test_literal_observation_is_preserved_when_exporting_an_exchange():
    from mikoshi.persistence.store import WorldStore
    from mikoshi.providers.base import DecisionRequest
    from mikoshi.runtime.conversations import ConversationHistory

    store = WorldStore()
    observed = []
    store.observe_exchanges(observed.append)
    try:
        request = DecisionRequest("literal", "", "plain observation", {})
        ConversationHistory(store).append_exchange("peer", "1:1", request, {}, {})
        assert observed[0]["messages"][0]["content"] == "plain observation"
        assert observed[0]["state"] == {"raw_text": "plain observation"}
    finally:
        store.close()
