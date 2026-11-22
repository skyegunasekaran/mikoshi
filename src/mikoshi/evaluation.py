"""Independent, operator-supplied scoring of retained run artifacts.

Graders are trusted Python, run on the host, and must live outside agent-writable
trees. They may inspect candidate files, but must use Docker for candidate code.
The runtime supplies no task-specific grading rules.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
import types
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mikoshi.persistence.artifacts import load_verified_artifacts
from mikoshi.persistence.exports import write_json
from mikoshi.schema import RUN_FILE, envelope, read_document, update_run


def validate_scorer(spec: str, repository: Path, output_root: Path) -> str:
    """Validate an explicit trusted file:callable without executing its code."""
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError("scorer must be an absolute Python file path followed by :callable")
    filename, qualified_name = spec.rsplit(":", 1)
    original = Path(filename).expanduser()
    if not original.is_absolute() or not all(
        part.isidentifier() for part in qualified_name.split(".")
    ):
        raise ValueError("scorer must be an absolute Python file path followed by :callable")
    path = original.resolve()
    for forbidden in (repository.resolve(), output_root.resolve()):
        if any(
            location == forbidden or forbidden in location.parents for location in (original, path)
        ):
            raise ValueError("scorer must live outside the agent repository and run output")
    if not path.is_file():
        raise ValueError(f"scorer file does not exist: {path}")
    try:
        compile(path.read_bytes(), str(path), "exec")
    except SyntaxError as error:
        raise ValueError(f"invalid scorer syntax: {error}") from error
    return f"{path}:{qualified_name}"


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, bool, int, float)) and (
        not isinstance(value, float) or math.isfinite(value)
    )


def _scores(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("grader must return a nonempty mapping of named scores")
    scores = {}
    for name, score in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("score names must be nonempty strings")
        if _scalar(score):
            score = {"value": score}
        elif isinstance(score, Mapping):
            score = dict(score)
            if set(score) - {"value", "explanation", "metadata"} or not _scalar(score.get("value")):
                raise ValueError(f"invalid score: {name}")
            if "explanation" in score and not isinstance(score["explanation"], str):
                raise ValueError(f"score explanation must be a string: {name}")
            if "metadata" in score and not isinstance(score["metadata"], dict):
                raise ValueError(f"score metadata must be a JSON object: {name}")
        else:
            raise ValueError(f"score must be a finite scalar or score object: {name}")
        scores[name] = score
    # Reject non-JSON objects and nonfinite values anywhere in grader metadata.
    return json.loads(json.dumps(scores, allow_nan=False))


def _call_scorer(path: Path, qualified_name: str, source: bytes, context: dict[str, Any]) -> Any:
    name = f"_mikoshi_grader_{uuid.uuid4().hex}"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
        grader: Any = module
        for part in qualified_name.split("."):
            grader = getattr(grader, part)
        if not callable(grader) or inspect.iscoroutinefunction(grader):
            raise ValueError("grader must be a synchronous callable")
        result = grader(context)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise ValueError("grader must return scores synchronously")
        return result
    finally:
        sys.modules.pop(name, None)


def _destination(run_dir, output, now):
    if output is None:
        directory = run_dir / "evaluations"
        directory.mkdir(exist_ok=True)
        output = directory / f"{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}.json"
    output = Path(output).expanduser().resolve()
    # Within a bundle, only new evaluation files can be written; native evidence
    # and existing evaluations are immutable to the scorer interface.
    if run_dir in output.parents and (
        output.parent != run_dir / "evaluations" or output.suffix != ".json"
    ):
        raise ValueError(
            "evaluation output must be a new JSON file in evaluations/ or outside the run"
        )
    if output.exists():
        raise ValueError("evaluation output already exists")
    return output


def _save_evaluation(run_dir, run, result, output):
    write_json(output, result)
    location = output.relative_to(run_dir).as_posix() if run_dir in output.parents else str(output)
    update_run(run_dir, evaluation=result, evaluations=[*run["evaluations"], location])
    return result


def record_evaluation_failure(run_dir, error_type, message):
    return evaluate_run(run_dir, failure={"type": error_type, "message": message})


def evaluate_run(
    run_dir: Path, scorer: str | None = None, *, output: Path | None = None, failure=None
):
    """Score the retained candidate; append an evaluation and update run.json.

    grade(context) receives run, input, candidate, candidate_sha256,
    artifact_manifest and run_dir. Earlier evaluations remain immutable.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    run = read_document(run_dir / RUN_FILE, "run")
    now = datetime.now(timezone.utc)
    output = _destination(run_dir, output, now)
    candidate = run.get("candidate")
    if isinstance(candidate, dict):
        repository = Path(candidate["source_repository"]).resolve()
        if output == repository or repository in output.parents:
            raise ValueError("evaluation output must live outside the agent repository")
    result = envelope(
        "evaluation",
        id=uuid.uuid4().hex,
        run_id=run["id"],
        evaluated_at=now.isoformat(),
        status="not_configured",
        scores={},
        scorer=None,
        candidate_sha256=None,
        evaluation_path=output.relative_to(run_dir).as_posix()
        if run_dir in output.parents
        else str(output),
    )
    try:
        if failure:
            result.update(status="failed", error=failure)
            return _save_evaluation(run_dir, run, result, output)
        manifest = load_verified_artifacts(run_dir)
        result["candidate_sha256"] = manifest["candidate_sha256"]
        config = run["input"]
        if scorer is None:
            scorer = config.get("evaluation", {}).get("scorer")
        if scorer is not None:
            repository = Path(manifest["source_repository"])
            spec = validate_scorer(scorer, repository, run_dir)
            filename, qualified_name = spec.rsplit(":", 1)
            path = Path(filename)
            source = path.read_bytes()
            result["scorer"] = {
                "path": str(path),
                "callable": qualified_name,
                "sha256": hashlib.sha256(source).hexdigest(),
            }
            context = {
                "candidate": manifest["candidate_path"],
                "candidate_sha256": manifest["candidate_sha256"],
                "artifact_manifest": manifest,
                "run_dir": str(run_dir),
                "input": config,
                "run": run,
            }
            scores = _scores(_call_scorer(path, qualified_name, source, context))
            verified = load_verified_artifacts(run_dir)
            if verified["candidate_sha256"] != result["candidate_sha256"]:
                raise ValueError("candidate changed during scoring")
            result.update(status="completed", scores=scores)
    except Exception as error:
        result.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    return _save_evaluation(run_dir, run, result, output)
