# Working on Mikoshi

Most of the work here is preparing experiments, launching runs the user has
authorized, keeping an eye on them, and evaluating what comes out. Anything
specific to one experiment, like its tasks, data, and scoring, stays outside the
runtime.

For an operator working directly in a terminal, `mikoshi init experiment.json`
provides guided setup and `mikoshi plan --config experiment.json` previews the
effective settings offline. Neither command starts a paid run. The readable
launcher is `mikoshi run --config experiment.json`; use `--json` for structured
output. `mikoshi runs` finds saved runs in the default output directory.

## Start an experiment

1. Pin down the objective, target directory, agent count, provider and model, real
   token prices, spending limit, and time limit. If the user has already given you
   some of these, use them, and write the configuration yourself. Only ask about
   missing decisions that would actually block the launch, and leave any tuning
   they didn't ask for at the defaults. A request to change code or docs is not
   permission to start a paid run.
2. Use the existing Python 3.12+ environment and the local Docker daemon. If
   something isn't set up yet, follow [installation](EXPERIMENTS.md#getting-started).
   Use native Python and PowerShell on Windows with Docker Desktop Linux containers.
   Build the command image once, and rebuild it only when its Dockerfile or
   required dependencies change. Credentials stay in the host environment, never
   in files or task text.
3. Agents edit the target in place, uncommitted files included. For controlled
   comparisons, use separate checkouts from equivalent starting states and keep
   private evaluation data somewhere else. Never reset, clean, stash, or overwrite
   the user's unrelated work, and keep run output outside the target.
4. Put choices in a single JSON file with `kind: input`, together with
   `run`, `pricing` and inline `problem` (objective, limits, roster, private
   instructions and scheduled messages). See
   [experiment design](EXPERIMENTS.md#design-an-experiment).

The common variables live in `run`: `max_output_tokens`, `context_window_tokens`,
`compaction_input_tokens`, `wall_time_seconds`, `budget_usd`,
`provider_concurrency`, `provider_reasoning_effort`, `temperature`, and `top_p`.
What each one means, and how they constrain each other, is covered under
[settings and constraints](EXPERIMENTS.md#experimental-settings). CLI flags use hyphens
and override config values, and relative paths in the config resolve from the
file's directory. Set the population in `problem.limits.max_agents`; an explicit
roster must match it. For example:

```sh
.venv/bin/mikoshi run --config /absolute/path/to/experiment.json
.venv/bin/mikoshi run --config /absolute/path/to/experiment.json \
  --max-output-tokens 16384 --wall-time-seconds 1200
```

If you want to diagnose setup, you can optionally run `mikoshi check` with the
same options. It checks the configuration, whether credentials are present, the
tokenizer, SQLite, and the local Docker and image metadata, all without starting a
run, a model request, or a container. That means provider access, task
dependencies, and container execution go untested.

Once the user has authorized the run and its spending envelope, go ahead without
asking again. Don't insist on benchmark setup, a seed repository, a verifier, or a
full test suite before launching; let the agents choose their own checks. Keep in
mind that in-flight requests can push spending past the limit.

## Monitor and finish

On launch, Mikoshi prints the output directory and streams progress and command
output. Keep long runs attached to a managed terminal session and check in on
their progress. From another terminal:

```sh
.venv/bin/mikoshi status /absolute/path/to/run
.venv/bin/mikoshi status /absolute/path/to/run --json
tail -n 20 /absolute/path/to/run/records.jsonl
.venv/bin/mikoshi stop /absolute/path/to/run
```

`status` is the latest recorded snapshot, not proof that the process is still
alive, so check its update timestamp and any pending or unknown-usage requests.
Command output is kept in `workspace/command-logs/`. Stop the run when the user
asks, or when you need to in order to stay within the authorized envelope. Ctrl-C
also requests a stop. A forced shutdown can lose in-flight results and final
exports, and runs can't be resumed.

After the run exits, look at authoritative `run.json` and its linked `records.jsonl`. Report the output path, the stop reason, the accounted spending and any
gaps in usage data, the checks that were actually performed, and any open
questions. Use `run.json.input` and `run.json.resolved` to compare settings. Be clear
about whether the run ended because the peers finished, a resource limit was hit,
the operator stopped it, or something failed. Be accurate about host restrictions
and about any cleanup you couldn't confirm.

Inspect the changes yourself and evaluate them against the research question.
Completion and successful commands are evidence that things executed, not that the
work is acceptable. Mikoshi doesn't open a PR or merge changes automatically.

## Runtime boundaries

- Peers, file tools, collaboration, and provider calls run on the host, while
  agent commands should not run locally. Never add a local fallback, and never
  pass provider credentials into containers.
- `src/mikoshi/` must stay independent of tasks, repositories, datasets, benchmarks,
  and expected answers. Reusable runtime tests go in `tests/`, and shared fixtures
  go in `tests/support/`. Scenario acceptance tests and model-behavior repairs stay
  out. See the [architecture map](ARCHITECTURE.md).
- When you move code or assets, update imports, entry points, documentation, and
  package data in the same change. Wheels contain the runtime code and bundled
  prompts, and source distributions also include the contributor tests. The local
  `docs/`, `output/`, `scripts/`, `tmp/`, and `training/` directories stay local
  and aren't tracked or shipped as release artifacts. The `examples/` directory
  is tracked, but remains excluded from release artifacts.

Install the contributor dependencies in the active environment with
`python -m pip install -e ".[dev]"`. Run `make check` after runtime
changes, and `make build` after packaging, entry-point, or bundled-asset changes.
On Windows, use `python tests/support/development.py check` and
`python tests/support/development.py build` for the same checks without Make.
The tests mock providers and Docker calls, so they don't make paid model requests.
If you run live integration checks, say so explicitly. See the
[test guide](tests/README.md).
