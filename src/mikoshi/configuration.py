"""Run intent plus explicit prices and deployment settings, independent of argv transport."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from mikoshi.isolation.docker import DockerSettings
from mikoshi.problem import ObjectiveSpec, ProblemSpec, ResourceLimits, parse_problem
from mikoshi.providers.limits import ProviderDeployment
from mikoshi.providers.pricing import TokenPrices
from mikoshi.providers.registry import provider_names
from mikoshi.schema import envelope, validate_document
from mikoshi.team_protocol import protocol_names
from mikoshi.tools.commands import CommandLimits

PATH_FIELDS = {"repository", "output"}


def positive_integer(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


# One public setting catalog drives CLI construction, config validation and export.
RUN_OPTIONS = {
    "repository": (Path, "existing directory to edit in place, including uncommitted files"),
    "task": (str, "plain-language shared objective"),
    "agents": (positive_integer, "peer count for --task; default 4"),
    "team_protocol": (str, "team interaction protocol; default peers"),
    "provider": (str, "model provider; default openai"),
    "model": (str, "explicit provider model or deployment resource"),
    "provider_reasoning_effort": (str, "reasoning setting; omitted to use provider defaults"),
    "max_output_tokens": (positive_integer, "maximum response tokens, including reasoning"),
    "context_window_tokens": (
        positive_integer,
        "context window, including input and reserved output",
    ),
    "compaction_input_tokens": (positive_integer, "native compaction input threshold"),
    "provider_concurrency": (
        positive_integer,
        "model request slots; default deployment or peer count",
    ),
    "temperature": (float, "sampling temperature; omitted to use provider defaults"),
    "top_p": (float, "sampling top-p; omitted to use provider defaults"),
    "budget_usd": (float, "spending ceiling; requires explicit token prices"),
    "no_budget_cap": (bool, "uncapped spending; requires explicit prices"),
    "wall_time_seconds": (float, "execution time ceiling, excluding setup"),
    "no_time_limit": (bool, "no execution time ceiling; explicit opt-in"),
    "disable_yield": (
        bool,
        "disable voluntary peer yielding; resource limits or operator stop end work",
    ),
    "output": (Path, "run records outside target; default unique user-state directory"),
    "trials": (positive_integer, "independent repetitions; default 1; limits apply per trial"),
    "trial_concurrency": (positive_integer, "concurrent trials; default 1"),
}
RUN_FIELDS = tuple(RUN_OPTIONS)


def run_parser(checking=False):
    parser = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} check" if checking else None,
        description=(
            "Check local run setup without model calls or container execution."
            if checking
            else "Pursue a shared objective with a fixed team of peers."
        ),
        epilog=(
            "Commands: init [FILE], plan [run options], check [run options], run [run options], "
            "runs, schema, validate PATH, status OUTPUT [--json], watch OUTPUT, stop OUTPUT, "
            "score OUTPUT [--scorer FILE:FUNCTION]. Run mikoshi with no arguments for a workflow guide."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, help="JSON run settings, prices and deployment")
    objectives = parser.add_mutually_exclusive_group()
    spending = parser.add_mutually_exclusive_group()
    timing = parser.add_mutually_exclusive_group()
    for name, (kind, help_text) in RUN_OPTIONS.items():
        group = (
            objectives
            if name in {"task", "problem"}
            else spending
            if name in {"budget_usd", "no_budget_cap"}
            else timing
            if name in {"wall_time_seconds", "no_time_limit"}
            else parser
        )
        options = {"help": help_text}
        if kind is bool:
            options["action"] = "store_true"
        else:
            options["type"] = kind
        if name == "repository":
            options["required"] = True
        elif name == "provider":
            options.update(choices=provider_names(), default="openai")
        elif name == "team_protocol":
            options.update(choices=protocol_names(), default="peers")
        group.add_argument("--" + name.replace("_", "-"), **options)
    return parser


def _object(value, name, allowed):
    if not isinstance(value, dict) or value.keys() - set(allowed):
        raise ValueError(f"{name} must be an object containing only: {', '.join(allowed)}")
    return value


def parse_configuration(parser, argv):
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--config", type=Path)
    path = bootstrap.parse_known_args(argv)[0].config
    try:
        document = json.loads(path.read_text(encoding="utf-8")) if path else {}
        _object(
            document,
            "config",
            ("kind", "run", "problem", "pricing", "deployment", "evaluation"),
        )
        if path and document.get("kind") != "input":
            raise ValueError("config requires kind 'input'")
        run = _object(
            document.get("run", {}),
            "run",
            tuple(k for k in RUN_FIELDS if k not in {"task", "agents"}),
        )
        deployment = _object(
            document.get("deployment", {}),
            "deployment",
            ("docker", "provider", "enable_commands", "command_limits"),
        )
        tokens = []
        supplied = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
        for key, value in run.items():
            option = "--" + key.replace("_", "-")
            if option in supplied:
                continue
            if key in {"budget_usd", "no_budget_cap"} and supplied.intersection(
                {"--budget-usd", "--no-budget-cap"}
            ):
                continue
            if key in {"wall_time_seconds", "no_time_limit"} and supplied.intersection(
                {"--wall-time-seconds", "--no-time-limit"}
            ):
                continue
            if value is None:
                continue
            if key in {"no_budget_cap", "no_time_limit", "disable_yield"}:
                if type(value) is not bool:
                    raise ValueError(f"run.{key} must be boolean")
                if value:
                    tokens.append(option)
            else:
                if isinstance(value, (dict, list, bool)):
                    raise ValueError(f"run.{key} must be a scalar")
                if key in PATH_FIELDS:
                    value = str((path.parent / Path(str(value)).expanduser()).resolve())
                tokens.append(f"{option}={value}")
        args = parser.parse_args([*tokens, *argv])
        if args.task is not None:
            if not args.task.strip():
                raise ValueError("--task must not be empty")
            count = args.agents or 4
            args.problem = ProblemSpec(
                ObjectiveSpec("task", args.task.strip().splitlines()[0][:160], args.task),
                ResourceLimits(count),
            )
        else:
            if args.agents is not None:
                raise ValueError("--agents requires --task; otherwise edit problem.limits and team")
            args.problem = parse_problem(document.get("problem"))
        args.docker = DockerSettings(
            **_object(
                deployment.get("docker", {}),
                "deployment.docker",
                DockerSettings.__dataclass_fields__,
            )
        )
        settings = dict(
            _object(
                deployment.get("provider", {}),
                "deployment.provider",
                ProviderDeployment.__dataclass_fields__,
            )
        )
        tokenizer = settings.get("tokenizer_path")
        if isinstance(tokenizer, str) and tokenizer.strip():
            settings["tokenizer_path"] = str((path.parent / Path(tokenizer).expanduser()).resolve())
        args.provider_deployment = ProviderDeployment(**settings)
        args.enable_commands = deployment.get("enable_commands", True)
        args.command_limits = CommandLimits(
            **_object(
                deployment.get("command_limits", {}),
                "deployment.command_limits",
                ("default_timeout_seconds", "max_timeout_seconds"),
            )
        )
        if type(args.enable_commands) is not bool:
            raise ValueError("deployment.enable_commands must be boolean")
        prices = dict(
            _object(
                document.get("pricing", {}),
                "pricing",
                (*TokenPrices.__dataclass_fields__, "models"),
            )
        )
        model_prices = prices.pop("models", {})
        if not isinstance(model_prices, dict):
            raise ValueError("pricing.models must map model IDs to explicit token rates")
        args.model_pricing = {}
        for model, rates in model_prices.items():
            if not isinstance(model, str) or not model.strip():
                raise ValueError("pricing.models requires nonempty model IDs")
            rates = _object(rates, f"pricing.models.{model}", TokenPrices.__dataclass_fields__)
            required = {"input_rate", "cache_read_rate", "output_rate", "thinking_rate"}
            if required - rates.keys():
                raise ValueError(f"pricing.models.{model} requires all four base rates")
            args.model_pricing[model] = TokenPrices(**rates)
            args.model_pricing[model].validate_provider(args.provider)
        required_prices = {"input_rate", "cache_read_rate", "output_rate", "thinking_rate"}
        if missing := required_prices - prices.keys():
            raise ValueError(
                f"pricing requires explicit rates in --config: {', '.join(sorted(missing))}"
            )
        args.pricing = TokenPrices(**prices)
        args.pricing.validate_provider(args.provider)
        evaluation = _object(document.get("evaluation", {}), "evaluation", ("scorer",))
        args.scorer = evaluation.get("scorer")
        if args.scorer is not None:
            if not isinstance(args.scorer, str) or ":" not in args.scorer:
                raise ValueError(
                    "evaluation.scorer must be a Python file path followed by :function"
                )
            filename, function = args.scorer.rsplit(":", 1)
            if (
                not filename
                or not function
                or not all(part.isidentifier() for part in function.split("."))
            ):
                raise ValueError("evaluation.scorer requires FILE:FUNCTION")
            scorer_path = Path(filename).expanduser()
            if not scorer_path.is_absolute():
                scorer_path = (path.parent if path else Path.cwd()) / scorer_path
            args.scorer = f"{scorer_path.resolve()}:{function}"
        args.trials = args.trials or 1
        args.trial_concurrency = args.trial_concurrency or 1
        if args.trial_concurrency > args.trials:
            raise ValueError("trial_concurrency cannot exceed trials")
        validate_document(configuration_document(args))
        return args
    except (OSError, TypeError, ValueError) as error:
        parser.error(f"invalid run configuration: {error}")


def configuration_document(args):
    """Serializable validated input; never includes credentials or runtime objects."""
    value = {
        **envelope("input"),
        "problem": asdict(args.problem),
        "run": {
            key: str(value) if isinstance(value, Path) else value
            for key in RUN_FIELDS
            if key not in {"task", "agents"} and (value := getattr(args, key, None)) is not None
        },
        "pricing": {
            **asdict(args.pricing),
            **(
                {"models": {model: asdict(rates) for model, rates in args.model_pricing.items()}}
                if getattr(args, "model_pricing", None)
                else {}
            ),
        },
        "deployment": {
            "docker": asdict(args.docker),
            "provider": asdict(args.provider_deployment),
            "enable_commands": args.enable_commands,
            "command_limits": {
                name: getattr(args.command_limits, name)
                for name in ("default_timeout_seconds", "max_timeout_seconds")
            },
        },
        "evaluation": {"scorer": args.scorer} if getattr(args, "scorer", None) else {},
    }

    return json.loads(json.dumps(value, allow_nan=False))
