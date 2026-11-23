![An abstract swarm of connected cyan, lavender, and coral nodes.](assets/readme/swarm.svg)

# Mikoshi

**Open infrastructure for agent swarm research**

Mikoshi lets you study how language-model agents pursue a shared objective. Vary the population, instructions, and resource limits, then inspect how their interactions shape collective behavior. Conversations, actions, changes, and costs are retained for evaluation.

[Start an experiment](EXPERIMENTS.md#getting-started) ·
[Explore the architecture](ARCHITECTURE.md) ·
[Read the evidence format](SCHEMA.md)

## Pipeline

| 🟢 Configure | 🟣 Coordinate | 🟠 Inspect |
| --- | --- | --- |
| Population, model, per-agent instructions, and resource limits. | Shared Markdown pages, addressed comments, repository edits, and Docker commands. | Conversations, actions, candidate files, token usage, and accounted cost. |

Peers work in a shared repository and organize their own coordination through
pages and comments. For comparisons, run repeated trials from copies of the same
starting files, with your graders and private evaluation data kept outside the
target. See [experiment design](EXPERIMENTS.md#design-an-experiment).

Built-in providers: **OpenAI · Anthropic · Google Gemini**. The host runs natively
on **Windows · macOS · Linux**, with extensions for providers and peer protocols.

## Quick start

After [installation](EXPERIMENTS.md#getting-started), create and review your settings:

```sh
mikoshi init experiment.json
mikoshi plan --config experiment.json
mikoshi check --config experiment.json --human
mikoshi run --config experiment.json
```

> [!IMPORTANT]
> Only `run` starts paid work. A single trial edits its target **in place,
> including uncommitted files**. In-flight requests can take spending past the
> limit. Review the resulting changes yourself.

## Inspect a run

Start with `run.json` for settings, stop reason, spending, and evaluation. Follow
`records.jsonl` for the interaction timeline, and inspect `candidate/` from safely
finished trials. A completed run still needs evaluation against your criteria.
See [monitoring and evaluation](EXPERIMENTS.md#inspect-a-run) and the
[output schema](SCHEMA.md) for details.

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
