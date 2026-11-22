"""Native repeated trials with independent workspaces and a durable result index."""

from __future__ import annotations

import asyncio
import json
import signal
from datetime import datetime, timezone
from pathlib import Path

from mikoshi.configuration import configuration_document
from mikoshi.persistence.artifacts import snapshot_repository
from mikoshi.persistence.exports import checkout_metadata, write_json
from mikoshi.runtime.control import stop_signals
from mikoshi.runtime.process import run_process
from mikoshi.schema import RUN_FILE, RecordWriter, create_run, update_run


def _now():
    return datetime.now(timezone.utc).isoformat()


def _record(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _score_summary(rows):
    """Aggregate available numeric scores without treating missing grades as zero."""
    values = {}
    for row in rows:
        result = row.get("evaluation") or {}
        if row["status"] not in {"completed", "stopped"} or result.get("status") != "completed":
            continue
        for name, score in result.get("scores", {}).items():
            value = score.get("value")
            if isinstance(value, (int, float)):
                values.setdefault(name, []).append(value)
    return {
        name: {
            "count": len(items),
            "mean": sum(items) / len(items),
            "missing": len(rows) - len(items),
        }
        for name, items in sorted(values.items())
    }


def run_experiment(args, *, progress_callback=None) -> bool:
    """Run a finite set of independent trials; never retry a paid attempt implicitly."""
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("experiment output must be a new or empty directory")
    rows = [
        {"id": f"trial-{index:03d}", "status": "pending"} for index in range(1, args.trials + 1)
    ]
    started = _now()
    config = configuration_document(args)
    document = create_run(output, scope="experiment", input=config, started_at=started)
    records = RecordWriter(output, document["id"])
    requested = False

    def stop(signum=None, frame=None):
        nonlocal requested
        requested = True

    def stopping():
        return requested or (output / "STOP").exists()

    def emit(event):
        if progress_callback is not None:
            try:
                progress_callback(event)
            except Exception:
                pass  # Presentation must not change paid trial behavior.

    def update(status="running", **extra):
        budgets = [row.get("budget") or {} for row in rows]
        changes = {
            "status": status,
            "phase": "trials",
            "trials": rows,
            "budget": {
                # Snapshot fields can be present but null, e.g. before any request is priced.
                "accounted_usd": sum(b.get("accounted_usd") or 0 for b in budgets),
                "pending_requests": sum(b.get("pending_requests") or 0 for b in budgets),
                "unknown_usage_requests": sum(
                    b.get("unknown_usage_requests") or 0 for b in budgets
                ),
                "unreported_trials": sum(
                    row["status"] not in {"pending", "preparing", "not_started"}
                    and not row.get("budget")
                    for row in rows
                ),
                "limit_usd": args.trials * args.budget_usd if args.budget_usd else None,
            },
            **extra,
        }
        value = update_run(output, **changes)
        records.append("progress", {"status": status, "phase": value["phase"], **extra})
        return value

    previous = {signum: signal.signal(signum, stop) for signum in stop_signals()}
    try:
        update(phase="preparing")
        emit({"progress": {"kind": "run_started", "output": str(output)}})
        source_metadata = checkout_metadata(args.repository)
        seed = output / "inputs" / "repository"
        seed.parent.mkdir()
        seed_digest = snapshot_repository(args.repository, seed)
        tokenizer = args.provider_deployment.tokenizer_path
        if tokenizer is not None:
            source = Path(tokenizer)
            destination = output / "inputs" / ("tokenizer" + source.suffix)
            destination.write_bytes(source.read_bytes())
            config["deployment"]["provider"]["tokenizer_path"] = str(destination)
        update_run(
            output,
            input=config,
            provenance={
                "trial_count": args.trials,
                "trial_concurrency": args.trial_concurrency,
                "limits_apply": "per_trial",
                "source_checkout": source_metadata,
                "input_sha256": seed_digest,
                "automatic_retries": False,
            },
        )

        async def run_trials():
            slots = asyncio.Semaphore(args.trial_concurrency)

            async def trial(row):
                async with slots:
                    if stopping():
                        row["status"] = "not_started"
                        update()
                        return
                    root = output / "trials" / row["id"]
                    root.mkdir(parents=True)
                    repository, run_dir = root / "repository", root / "run"
                    row.update(status="preparing", run_dir=run_dir.relative_to(output).as_posix())
                    update()
                    try:
                        digest = await asyncio.to_thread(snapshot_repository, seed, repository)
                        if digest != seed_digest:
                            raise ValueError("frozen experiment inputs changed")
                        if stopping():
                            row["status"] = "not_started"
                            update()
                            return
                        child = json.loads(json.dumps(config))
                        child["run"].update(
                            repository=str(repository),
                            output=str(run_dir),
                            trials=1,
                            trial_concurrency=1,
                        )
                        config_path = root / "config.json"
                        write_json(config_path, child)
                        row["status"] = "running"
                        update()

                        def progress(event):
                            current = _record(run_dir / RUN_FILE)
                            row.update(
                                progress=event,
                                phase=current.get("phase"),
                                budget=current.get("budget"),
                                agents=current.get("agents", {}),
                            )
                            update()
                            emit({"trial_id": row["id"], "progress": event})

                        exit_code = await run_process(
                            config_path, run_dir, progress, stop_requested=stopping
                        )
                        terminal = _record(run_dir / RUN_FILE)
                        supervisor = _record(root / "supervisor.json")
                        row.update(
                            status=terminal.get("status", "failed"),
                            execution_status=terminal.get(
                                "execution_status", terminal.get("status", "failed")
                            ),
                            stop_reason=terminal.get("stop_reason"),
                            budget=terminal.get("budget"),
                            exit_code=exit_code,
                            evaluation=terminal.get("evaluation"),
                            finished_at=_now(),
                            supervisor={
                                "path": (root / "supervisor.json").relative_to(output).as_posix(),
                                "output_capture_complete": supervisor.get(
                                    "output_capture_complete"
                                ),
                                "escalation": supervisor.get("escalation"),
                            },
                        )
                        if (
                            exit_code
                            or terminal.get("status") not in {"completed", "stopped"}
                            or terminal.get("finished_at") is None
                        ):
                            row["status"] = "failed"
                            row["error"] = (
                                terminal.get("error") or "trial did not finalize run.json"
                            )
                    except Exception as error:
                        row.update(
                            status="failed",
                            execution_status="failed",
                            error=f"{type(error).__name__}: {error}",
                        )
                    update()
                    emit(
                        {
                            "trial_id": row["id"],
                            "progress": {
                                "kind": "trial_finished",
                                "status": row["status"],
                                "stop_reason": row.get("stop_reason"),
                                "error": row.get("error"),
                            },
                        }
                    )

            await asyncio.gather(*(trial(row) for row in rows))

        asyncio.run(run_trials())
        failed = any(row["status"] == "failed" for row in rows)
        stopped = stopping() or any(row["status"] in {"stopped", "not_started"} for row in rows)
        status = "failed" if failed else "stopped" if stopped else "completed"
        execution_failed = any(
            row.get("execution_status") not in {"completed", "stopped", None} for row in rows
        )
        execution_stopped = stopping() or any(
            row.get("execution_status") == "stopped" or row["status"] == "not_started"
            for row in rows
        )
        execution_status = (
            "failed" if execution_failed else "stopped" if execution_stopped else "completed"
        )
        update_run(output, score_summary=_score_summary(rows), execution_status=execution_status)
        final = update(status, finished_at=_now())
        final = update_run(output, evidence={**final["evidence"], "record_count": records.sequence})
        return failed
    except BaseException as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
