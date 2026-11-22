#!/usr/bin/env python3
"""Run the pinned upstream evaluator without modifying its scoring or test code."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--cpus", required=True, type=int)
    parser.add_argument("--memory", required=True)
    parser.add_argument(
        "--branch", action="append", default=[], help="Explicit control diagnostic subset"
    )
    args = parser.parse_args()
    # Imports occur only after the caller has selected the pinned host environment.
    from programbench.constants import DOCKER_RUN_ARGS, TASKS_DIR
    from programbench.eval.eval import Evaluator
    from programbench.utils.load_data import (
        get_active_branches,
        get_expected_tests,
        get_ignored_branches,
        get_ignored_tests,
        load_all_instances,
    )

    recipe = Path(__file__).resolve().parent
    lock = json.loads((recipe / "environment.lock.json").read_text())
    iid = lock["instance_id"]
    instance = next(i for i in load_all_instances() if i["instance_id"] == iid)
    active_branches = get_active_branches(instance)
    branches = args.branch or active_branches
    if set(branches) - set(active_branches):
        raise ValueError("Unknown or inactive control branch")
    DOCKER_RUN_ARGS.extend(
        [
            "--platform",
            lock["platform"],
            "--network",
            lock["evaluation_network"],
            "--pull",
            "never",
            "--memory",
            args.memory,
            "--memory-swap",
            args.memory,
            "--pids-limit",
            str(lock["local_resources"]["pids_limit"]),
        ]
    )
    image_name, image_tag = args.image.rsplit(":", 1)
    result = Evaluator(
        image_name=image_name,
        image_tag=image_tag,
        instance_id=iid,
        solution_branch="submission",
        submission_archive=args.output / iid / "submission.tar.gz",
        blob_dir=args.state / "fixtures" / iid,
        tests_branches=branches,
        tests_by_branch={b: instance["branches"][b]["tests"] for b in branches},
        ignored_tests=get_ignored_tests(instance),
        ignored_branches=get_ignored_branches(instance),
        remove_hashes=instance.get("eval_clean_hashes", []),
        docker_cpus=args.cpus,
        branch_workers=1,
        branch_retries=1,
    ).run()
    result_path = args.output / iid / f"{iid}.eval.json"
    result_path.write_text(result.model_dump_json(indent=2) + "\n")
    expected = get_expected_tests(instance)
    ignored = get_ignored_tests(instance)
    filtered = result.for_branches(branches).only_expected(expected).without_ignored(ignored)
    active_expected = {t for t in expected if t.split("/", 1)[0] in branches} - ignored
    observed = {t.full_name for t in filtered}
    summary = {
        "instance_id": iid,
        "score": filtered.score,
        "passed": filtered.n_resolved,
        "tests": len(filtered),
        "expected_tests": len(active_expected),
        "statuses": dict(Counter(t.status for t in filtered)),
        "active_branches": branches,
        "scope": "full_instance" if set(branches) == set(active_branches) else "control_subset",
        "error_code": result.error_code,
        "test_branch_errors": {
            b: [e.model_dump() for e in errors] for b, errors in result.test_branch_errors.items()
        },
        "warnings": result.warnings,
        "missing_tests": sorted(active_expected - observed),
        "manifest_sha256": hashlib.sha256(
            (TASKS_DIR / iid / "tests.json").read_bytes()
        ).hexdigest(),
        "executable_sha256": result.executable_hash,
    }
    summary["infrastructure_ok"] = not (
        summary["error_code"]
        or summary["test_branch_errors"]
        or summary["warnings"]
        or summary["missing_tests"]
        or any(t.status in {"error", "system_error", "not_run"} for t in filtered)
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["infrastructure_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
