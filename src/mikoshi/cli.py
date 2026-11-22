"""Human and machine command-line interfaces for the experiment application."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from mikoshi import application
from mikoshi.configuration import (
    configuration_document,
    parse_configuration,
    positive_integer,
    run_parser,
)
from mikoshi.evaluation import evaluate_run
from mikoshi.experiment_setup import create_configuration
from mikoshi.operations import check_run, format_status, read_status, watch_status
from mikoshi.presentation import TerminalReporter, format_plan, shell_quote
from mikoshi.schema import RUN_FILE, json_schema, validate_bundle

OVERVIEW = """Mikoshi — run experiments with teams of AI agents.

Start here:
  mikoshi init experiment.json          Create settings with guided prompts
  mikoshi plan --config experiment.json Review the resolved plan offline
  mikoshi check --config experiment.json Check credentials and local setup
  mikoshi run --config experiment.json  Start the paid experiment

Monitor and maintain:
  mikoshi runs                         Find saved runs
  mikoshi status OUTPUT                Read the latest recorded status
  mikoshi watch OUTPUT                 Follow status updates
  mikoshi stop OUTPUT                  Request a graceful stop
  mikoshi validate OUTPUT              Verify retained evidence offline
  mikoshi score OUTPUT --scorer FILE:FUNCTION  Evaluate a retained candidate

Use COMMAND --help for details. `mikoshi --help` lists all run settings.
Without a subcommand, `mikoshi --config ...` runs with JSON output, like `run --json`.
"""


def _parser(command, description):
    return argparse.ArgumentParser(
        prog=f"mikoshi {command}", description=description, allow_abbrev=False
    )


def _saved_runs(argv):
    parser = _parser("runs", "Find saved run records; this never starts or stops work.")
    parser.add_argument(
        "--directory",
        type=Path,
        default=application.state_directory() / "runs",
        help="directory containing run folders",
    )
    parser.add_argument("--limit", type=positive_integer, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        directory = args.directory.expanduser().resolve()
        rows, errors = [], []
        if directory.exists():
            for path in directory.iterdir():
                if not (path / RUN_FILE).is_file():
                    continue
                try:
                    value = read_status(path)
                except (OSError, ValueError, TypeError) as error:
                    errors.append({"path": str(path), "error": str(error)})
                    continue
                rows.append(
                    {
                        "path": str(path),
                        **{
                            key: value.get(key)
                            for key in ("status", "scope", "updated_at", "stop_reason")
                        },
                    }
                )
        rows.sort(key=lambda row: row["updated_at"] or "", reverse=True)
        if args.json:
            print(
                json.dumps(
                    {
                        "directory": str(directory),
                        "runs": rows[: args.limit],
                        "total": len(rows),
                        "errors": errors,
                    },
                    indent=2,
                )
            )
        else:
            print(f"Saved runs in {directory}:")
            for row in rows[: args.limit]:
                print(f"{row['updated_at']}  {row['status']}  {row['path']}")
            if not rows:
                print("No saved runs found. Use --directory for a custom output location.")
            if len(rows) > args.limit:
                print(f"Showing {args.limit} of {len(rows)}; increase --limit to see more.")
            for error in errors:
                print(f"Could not read {error['path']}: {error['error']}", file=sys.stderr)
    except OSError as error:
        parser.error(str(error))


def main(argv=None) -> None:
    if os.name == "nt":
        # Keep redirected Unicode plans and JSON usable with Windows' legacy code pages.
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(OVERVIEW)
        return
    command, tail = argv[0], argv[1:]
    if command == "init":
        path = create_configuration(tail)
        quoted = shell_quote(path)
        print(
            f"Saved {path}\nReview: mikoshi plan --config {quoted}\nCheck setup: mikoshi check --config {quoted}\nStart: mikoshi run --config {quoted}"
        )
        return
    if command == "runs":
        _saved_runs(tail)
        return
    if command == "schema":
        parser = _parser(command, "Print the unified input/output JSON Schema.")
        parser.parse_args(tail)
        print(json.dumps(json_schema(), indent=2))
        return
    if command == "validate":
        parser = _parser(command, "Validate an input or retained run bundle offline.")
        parser.add_argument("path", type=Path)
        options = parser.parse_args(tail)
        try:
            print(json.dumps(validate_bundle(options.path.expanduser().resolve()), indent=2))
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return
    if command == "watch":
        parser = _parser(command, "Watch recorded snapshots until the run becomes terminal.")
        parser.add_argument("output", type=Path)
        parser.add_argument("--interval-seconds", type=float, default=2.0)
        parser.add_argument("--json", action="store_true")
        options = parser.parse_args(tail)
        try:
            watch_status(
                options.output.expanduser().resolve(),
                interval_seconds=options.interval_seconds,
                json_output=options.json,
            )
        except KeyboardInterrupt:
            pass
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"cannot watch status: {error}")
        return
    if command == "score":
        parser = _parser(command, "Score a frozen run without launching agents.")
        parser.add_argument("run", type=Path)
        parser.add_argument("--scorer", help="operator-owned Python FILE:FUNCTION")
        parser.add_argument("--output", type=Path, help="new evaluation file outside the candidate")
        options = parser.parse_args(tail)
        try:
            result = evaluate_run(
                options.run.expanduser().resolve(), options.scorer, output=options.output
            )
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"cannot score run: {error}")
        print(json.dumps(result, indent=2))
        if result["status"] == "failed":
            raise SystemExit(1)
        return
    if command == "status":
        parser = _parser(
            command, "Inspect the latest recorded run status, without probing its process."
        )
        parser.add_argument("output", type=Path)
        parser.add_argument("--json", action="store_true")
        options = parser.parse_args(tail)
        try:
            output = options.output.expanduser().resolve()
            value = read_status(output)
            print(
                json.dumps(value, indent=2, sort_keys=True)
                if options.json
                else format_status(output, value)
            )
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"cannot read run status: {error}")
        return
    if command == "stop":
        parser = _parser(command, "Request a graceful stop of a running Mikoshi session.")
        parser.add_argument("output", type=Path)
        options = parser.parse_args(tail)
        output = options.output.expanduser().resolve()
        try:
            read_status(output)
            (output / "STOP").touch()
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"cannot request stop: {error}")
        print("Stop requested. Active commands are cancelled; unresponsive runs are terminated.")
        return
    named_run = command == "run"
    planning = command == "plan"
    checking = command == "check"
    parser = run_parser(checking)
    if planning:
        parser.prog = "mikoshi plan"
        parser.description = "Review resolved experiment settings offline. No credential checks, model requests, or containers."
    elif named_run:
        parser.prog = "mikoshi run"
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--json", action="store_true", help="print structured JSON output and progress"
    )
    output_group.add_argument(
        "--human", action="store_true", help="print readable setup checks and progress"
    )
    args = parse_configuration(parser, tail if named_run or planning or checking else argv)
    try:
        prepared = application.prepare_run(args, require_credentials=not planning)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args = prepared.settings
    if planning:
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "planned",
                        "input": configuration_document(args),
                        "resolved": {
                            "model_limits": asdict(args.model_limits),
                            "agent_model_limits": {
                                model: asdict(limits)
                                for model, limits in args.agent_model_limits.items()
                            },
                        },
                    },
                    indent=2,
                )
            )
        else:
            print(format_plan(prepared))
            print(
                "\nPlan only. No credentials, provider access, Docker, or task dependencies checked."
            )
        return
    if checking:
        try:
            result = check_run(args)
        except Exception as error:
            parser.error(f"local setup check failed: {error}")
        if args.human or (sys.stdout.isatty() and not args.json):
            print(format_plan(prepared) + "\n\nLocal setup check passed.\n" + result["note"])
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        return
    reporter = TerminalReporter(json_output=args.json or (not named_run and not args.human))
    if not reporter.json_output:
        print(format_plan(prepared), file=sys.stderr, flush=True)
    try:
        result = application.execute_run(
            prepared,
            progress_callback=reporter.progress,
            command_output_callback=reporter.command_output,
        )
    except (Exception, KeyboardInterrupt) as error:
        if not named_run and not args.human:
            raise
        try:
            recorded = read_status(args.output)
        except (OSError, ValueError, TypeError):
            print(
                f"Run could not finish: {type(error).__name__}: {error}\nOutput: {args.output}",
                file=sys.stderr,
            )
        else:
            reporter.result(args.output, recorded)
        raise SystemExit(130 if isinstance(error, KeyboardInterrupt) else 1) from None
    reporter.result(args.output, result)
    if result["status"] == "failed":
        raise SystemExit(1)
