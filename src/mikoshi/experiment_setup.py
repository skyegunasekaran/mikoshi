"""Create an operator-owned experiment configuration without starting work."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from mikoshi.configuration import positive_integer
from mikoshi.providers.pricing import TokenPrices
from mikoshi.providers.registry import provider_names
from mikoshi.schema import validate_document


def _finite_number(value, *, positive=False):
    number = float(value)
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise ValueError(
            "enter a finite positive number" if positive else "enter a finite nonnegative number"
        )
    return number


def create_configuration(argv):
    parser = argparse.ArgumentParser(
        prog="mikoshi init",
        allow_abbrev=False,
        description="Create a JSON experiment configuration. No model requests or containers are started.",
    )
    parser.add_argument("path", type=Path, nargs="?", default=Path("experiment.json"))
    parser.add_argument("--interactive", action="store_true", help="prompt for missing settings")
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--objective")
    parser.add_argument("--provider", choices=provider_names())
    parser.add_argument("--model")
    parser.add_argument("--agents", type=positive_integer)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--wall-time-seconds", type=float)
    parser.add_argument(
        "--no-commands", action="store_true", help="allow file tools only; no agent commands"
    )
    for name in ("input_rate", "cache_read_rate", "output_rate", "thinking_rate"):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=float, help="USD per million tokens"
        )
    args = parser.parse_args(argv)
    args.path = args.path.expanduser()
    interactive = args.interactive or sys.stdin.isatty()
    if args.path.exists() or args.path.is_symlink():
        parser.error(f"{args.path} already exists; choose a new file or edit it directly")

    def ask(flag, prompt, convert=str, default=None):
        supplied = getattr(args, flag)
        while True:
            if supplied is not None:
                raw = supplied
            elif interactive:
                suffix = f" [{default}]" if default is not None else ""
                try:
                    raw = input(prompt + suffix + ": ").strip()
                except (EOFError, KeyboardInterrupt):
                    parser.error("setup cancelled; no configuration was written")
                raw = raw or default
            else:
                raw = default
            if raw is None:
                if interactive:
                    print("A value is required.", file=sys.stderr)
                    continue
                parser.error(
                    f"missing --{flag.replace('_', '-')}; run init in a terminal for guided setup"
                )
            try:
                value = convert(raw)
                if isinstance(value, str) and not value.strip():
                    raise ValueError("enter a nonempty value")
                return value
            except (ValueError, TypeError, argparse.ArgumentTypeError) as error:
                if supplied is not None or not interactive:
                    parser.error(f"--{flag.replace('_', '-')}: {error}")
                print(f"Try again: {error}", file=sys.stderr)

    def repository_path(raw):
        path = Path(raw).expanduser().resolve()
        if not path.is_dir():
            raise ValueError("target directory must already exist")
        return str(path)

    def provider_name(raw):
        if raw not in provider_names():
            raise ValueError("choose " + ", ".join(provider_names()))
        return raw

    if interactive:
        print("Create an experiment. This saves settings only; running is a separate command.")
        print("Use a target checkout prepared for agent edits, including uncommitted files.")
        print("Enter your provider's current token prices in USD per million tokens.")
    repository = ask("repository", "Target directory", repository_path)
    objective = ask("objective", "What should the team accomplish?")
    provider = ask(
        "provider", "Provider (" + ", ".join(provider_names()) + ")", provider_name, "openai"
    )
    model = ask("model", "Model ID")
    agents = ask("agents", "Number of agents", positive_integer, 4)
    budget = ask(
        "budget_usd", "Spending limit per trial (USD)", lambda v: _finite_number(v, positive=True)
    )
    seconds = ask(
        "wall_time_seconds",
        "Time limit per trial (seconds)",
        lambda v: _finite_number(v, positive=True),
        1800,
    )
    prices = {
        name: ask(name, label + " (USD per million tokens)", _finite_number)
        for name, label in (
            ("input_rate", "Uncached input price"),
            ("cache_read_rate", "Cached input price"),
            ("output_rate", "Output price"),
            ("thinking_rate", "Thinking price (match output for Anthropic)"),
        )
    }
    try:
        TokenPrices(**prices).validate_provider(provider)
        document = {
            "kind": "input",
            "problem": {
                "objective": {
                    "id": "experiment",
                    "title": objective.splitlines()[0][:160],
                    "description": objective,
                },
                "limits": {"max_agents": agents},
            },
            "run": {
                "repository": repository,
                "provider": provider,
                "model": model,
                "budget_usd": budget,
                "wall_time_seconds": seconds,
            },
            "pricing": prices,
        }
        if args.no_commands:
            document["deployment"] = {"enable_commands": False}
        validate_document(document)
        with args.path.expanduser().open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError) as error:
        parser.error(f"could not create configuration: {error}")
    return args.path.expanduser().resolve()
