# Mikoshi

Mikoshi is a toolkit for experiments with teams of AI agents. Agents work in a
shared repository, coordinate through shared pages and comments, and run commands
in Docker, with limits on
time and spending. Conversations, changes, and costs are retained for evaluation.

It supports OpenAI, Anthropic, and Google Gemini, with extensions for other
providers and team protocols. The host runs natively on Windows, macOS, and Linux.

To get started, see the [experiment guide](EXPERIMENTS.md#getting-started).
It covers installation, configuration, monitoring, and evaluating results.
The [architecture](ARCHITECTURE.md) and [output schema](SCHEMA.md) describe the runtime.

## Quick start

After [installation](EXPERIMENTS.md#getting-started), create and review your settings:

```sh
mikoshi init experiment.json
mikoshi plan --config experiment.json
mikoshi check --config experiment.json --human
mikoshi run --config experiment.json
```

Only `run` starts paid work. A single trial edits its target in place; in-flight
requests can take spending past the limit. Review the resulting changes yourself.

---

## Contributing

With Python 3.12+ and an activated virtual environment:

```sh
python -m pip install -e ".[dev]"
python tests/support/development.py check
python tests/support/development.py build
```

`make check` and `make build` are also available. Tests mock providers and Docker;
they make no paid requests. See the [test guide](tests/README.md) for details.

Copyright 2026 Skye Gunasekaran. [MIT License](LICENSE) · [Third-party notices](THIRD_PARTY.md).
