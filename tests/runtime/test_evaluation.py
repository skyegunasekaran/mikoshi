import json

import pytest

from mikoshi.evaluation import evaluate_run, validate_scorer
from mikoshi.persistence.artifacts import freeze_candidate
from mikoshi.schema import create_run, read_document, update_run


def retained_run(tmp_path, source, *, configured=True):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "answer.txt").write_text("retained answer", encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    grader = tmp_path / "grader.py"
    grader.write_text(source, encoding="utf-8")
    spec = f"{grader}:grade"
    config = {"run": {"repository": str(repository)}}
    if configured:
        config["evaluation"] = {"scorer": spec}
    create_run(output, input=config)
    update_run(output, stop_reason="peers_finished", status="completed")
    freeze_candidate(repository, output)
    return repository, output, spec


def test_scoring_and_rescoring_use_retained_candidate(tmp_path):
    repository, output, _ = retained_run(
        tmp_path,
        "from pathlib import Path\n"
        "def grade(context):\n"
        "    answer = (Path(context['candidate']) / 'answer.txt').read_text()\n"
        "    return {'quality': {'value': answer == 'retained answer',\n"
        "                        'explanation': answer,\n"
        "                        'metadata': {'stop_reason': context['run']['stop_reason']}}}\n",
    )
    first = evaluate_run(output)
    (repository / "answer.txt").write_text("live tree changed", encoding="utf-8")
    update_run(output, status="completed")
    second = evaluate_run(output)

    assert first["status"] == second["status"] == "completed"
    assert (
        first["scores"]
        == second["scores"]
        == {
            "quality": {
                "value": True,
                "explanation": "retained answer",
                "metadata": {"stop_reason": "peers_finished"},
            }
        }
    )
    assert first["candidate_sha256"] == second["candidate_sha256"]
    assert first["scorer"]["sha256"] == second["scorer"]["sha256"]
    assert first["evaluation_path"] != second["evaluation_path"]
    assert first["evaluation_path"].startswith("evaluations/")
    assert "\\" not in first["evaluation_path"]
    assert read_document(output / "run.json")["evaluation"] == second
    assert json.loads((output / first["evaluation_path"]).read_text(encoding="utf-8")) == first
    assert json.loads((output / second["evaluation_path"]).read_text(encoding="utf-8")) == second


@pytest.mark.parametrize(
    "body",
    [
        "return {'quality': float('nan')}",
        "raise RuntimeError('verifier unavailable')",
    ],
)
def test_grader_errors_are_failures_without_scores(tmp_path, body):
    _, output, _ = retained_run(tmp_path, f"def grade(context):\n    {body}\n")
    result = evaluate_run(output)
    assert result["status"] == "failed"
    assert result["scores"] == {}
    assert result["error"]["message"]
    assert result["scorer"]["sha256"]


def test_retained_candidate_changes_fail_evaluation(tmp_path):
    _, output, _ = retained_run(
        tmp_path,
        "from pathlib import Path\n"
        "def grade(context):\n"
        "    (Path(context['candidate']) / 'answer.txt').write_text('changed')\n"
        "    return {'quality': 1}\n",
    )
    result = evaluate_run(output)
    assert result["status"] == "failed"
    assert result["scores"] == {}
    assert "fingerprint mismatch" in result["error"]["message"]


def test_explicit_scorer_validation_does_not_execute_and_rejects_agent_paths(tmp_path):
    repository, output, spec = retained_run(
        tmp_path, "raise RuntimeError('must not execute during configuration')\n"
    )
    assert validate_scorer(spec, repository, output) == spec
    for directory in (repository, output / "candidate", output / "workspace"):
        directory.mkdir(exist_ok=True)
        grader = directory / "grader.py"
        grader.write_text("def grade(context): return {'quality': 1}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="outside"):
            validate_scorer(f"{grader}:grade", repository, output)


def test_unconfigured_evaluation_stays_distinct_from_failure(tmp_path):
    _, output, _ = retained_run(
        tmp_path, "def grade(context): return {'quality': 0}\n", configured=False
    )
    result = evaluate_run(output)
    assert result["status"] == "not_configured"
    assert result["scores"] == {}
    assert result["scorer"] is None
    assert result["candidate_sha256"]


def test_evaluation_destination_cannot_replace_inputs(tmp_path):
    repository, output, _ = retained_run(tmp_path, "def grade(context): return {'quality': 1}\n")
    for destination in (
        output / "run.json",
        output / "candidate" / "answer.txt",
        repository / "answer.txt",
    ):
        before = destination.read_bytes()
        with pytest.raises(ValueError, match="output"):
            evaluate_run(output, output=destination)
        assert destination.read_bytes() == before
