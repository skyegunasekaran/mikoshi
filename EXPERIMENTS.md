# Experiments

## Getting started

Install Python 3.12+, Git, and Docker. On Windows and macOS, use [Docker Desktop](https://docs.docker.com/desktop/)
with **Linux containers**; on Linux, Docker Engine also works. Start Docker before
checking setup. The Mikoshi host runs directly in PowerShell on Windows; a WSL
shell is not required. Agent commands run inside Linux containers on every host.

Clone this repository, then create and activate a virtual environment.

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
docker build -f src/mikoshi/isolation/Dockerfile -t mikoshi-runtime:local .
$env:OPENAI_API_KEY = "YOUR_API_KEY"
```

If your PowerShell policy blocks activation, use `.venv\Scripts\python.exe` and
`.venv\Scripts\mikoshi.exe` in place of `python` and `mikoshi` below; activation
is optional. Use a local NTFS target directory. File tools reject reparse points
(including junctions), symbolic links, and files with multiple hard links.
Docker Desktop must be able to share the target drive. Windows uses the Linux
user configured in the image; macOS and Linux use the host's numeric user/group.
Linked Git worktree metadata is mounted at `/mikoshi/git-metadata` inside the
container rather than exposing Windows drive paths to Git.

**macOS / Linux:**

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
docker build -f src/mikoshi/isolation/Dockerfile -t mikoshi-runtime:local .
export OPENAI_API_KEY="YOUR_API_KEY"
```

For Anthropic, set `ANTHROPIC_API_KEY`; for Gemini, set `GOOGLE_API_KEY` instead.
Credentials stay in the host environment, outside configuration files and the
target repository. Build the image again only when its Dockerfile or dependencies
change.

Create a configuration with guided prompts, preview it, and check local setup:

```sh
mikoshi init experiment.json
mikoshi plan --config experiment.json
mikoshi check --config experiment.json --human
```

The prompts ask for your objective, target directory, team size, provider, model,
current token prices, and time and spending limits. Review the plan before
starting the paid experiment:

```sh
mikoshi run --config experiment.json
```

`mikoshi init --help` lists the flags for creating a config without prompts.
`plan` previews effective settings offline; `check` inspects local setup without
starting a model request or container. Neither verifies provider access or task
dependencies.

## Configuration

Save this as `experiment.json`. **Replace the target path, model, and prices**
before running. Prices are illustrative USD per million tokens; all four rates
are required. For Anthropic, output and thinking rates must match.
`input_rate` means ordinary uncached input pricing; see [pricing options](src/mikoshi/providers/pricing.py)
for separate cache-creation and long-context rates.

```json
{
  "kind": "input",
  "problem": {
    "objective": {
      "id": "bug-repair",
      "title": "Bug repair with evidence",
      "description": "Find and fix a bug. Record evidence and remaining uncertainty."
    },
    "limits": { "max_agents": 2 }
  },
  "run": {
    "repository": "/absolute/path/to/experiment-checkout",
    "provider": "openai",
    "model": "YOUR_MODEL_ID",
    "budget_usd": 5.0,
    "wall_time_seconds": 600
  },
  "pricing": {
    "input_rate": 1.0,
    "cache_read_rate": 0.1,
    "output_rate": 2.0,
    "thinking_rate": 2.0
  }
}
```

> [!IMPORTANT]
> A single trial edits the target **in place, including uncommitted files**.
> Use a checkout prepared for the experiment. In-flight requests can take spending
> past the budget; accounting uses the prices you supply.

For an existing config, validate it, review the plan, then start the paid run:

```sh
mikoshi validate experiment.json
mikoshi plan --config experiment.json
mikoshi run --config experiment.json
```

Peers start with two shared pages: read-only `/objective` and editable
`/coordination`. They can create additional Markdown pages and use addressed
comments to coordinate. Each peer sees one feed of page edits and discussions.
Responsibilities, plans and progress are ordinary page content that peers can
revise together. Page edits do not wake idle peers; address a comment to a peer
when their attention is needed, using exact peer IDs in `notify`. Peer comments
are public to page readers; reply threading and `@mentions` do not deliver
notifications. Scheduled operator messages go directly to their recipients'
inboxes. Commands return a preview and an `output_ref` for `read_output`; full
logs and command metadata remain in the run evidence.

Agent commands run in Docker without network access by default; model calls and
file tools run on the host. A run ends when all peers yield with nothing queued, a limit is reached,
the operator stops it, or execution fails.

## Experimental settings

Set model, token limits, and concurrency in `run`; set the population in
`problem.limits.max_agents`. Paths in the config resolve from its directory;
CLI flags override config values. CLI paths resolve from the current directory.
On Windows, JSON paths can use forward slashes, for example `C:/work/target`;
backslashes must be escaped as `C:\\work\\target`. Command text uses Linux shell
syntax inside Docker, even when the host launcher runs in PowerShell.
Most experiments only need the choices covered by `init`; tune the following
when the research question calls for it:

| Setting | Meaning |
| --- | --- |
| `problem.limits.max_agents` | Team size. Every peer can be active at once; an explicit `problem.team.members` roster must match it. |
| `run.provider_concurrency` | Model request slots; defaults to the deployment setting or team size. |
| `run.budget_usd` | Accounted spending limit per trial. In-flight requests can overrun it. |
| `run.wall_time_seconds` | Time per trial, excluding setup; defaults to 1,800 seconds. |
| `run.max_output_tokens` | Response allowance, including reasoning tokens. |
| `run.context_window_tokens` | Operating context window, including input and reserved output. |
| `run.compaction_input_tokens` | Input threshold for native conversation compaction. |
| `run.temperature`, `run.top_p`, `run.provider_reasoning_effort` | Optional provider-supported generation settings. |
| `run.output` | New or empty output directory, outside the target; otherwise a unique user-state directory. |
| `run.trials`, `run.trial_concurrency` | Number of independent attempts and how many can run at once. |
| `deployment.enable_commands` | Defaults to true. False leaves file and collaboration tools available without agent commands. |

Token limits use adapter defaults when omitted. `plan` shows the effective values;
confirm that the operating limits fit the actual model you selected. Unlimited
spending and time each require explicit opt-in (`--no-budget-cap`, `--no-time-limit`).
Use `mikoshi run --help` for overrides. Advanced cache-creation and long-context
prices are described in [pricing](src/mikoshi/providers/pricing.py).

## Design an experiment

Keep the objective, optional roster, private instructions, and scheduled messages
in `problem`; see the [problem format](src/mikoshi/problem.py). For repeated trials,
each attempt gets a fresh copy of the same seed, including uncommitted files:

```sh
mikoshi run --config experiment.json --trials 5 --trial-concurrency 1
```

**Limits apply per trial**: five $5 trials have a nominal $25 total, before any
in-flight overrun. Define success independently of agents' claims. An optional
`evaluation.scorer` runs trusted host code after a trial; keep graders and private
answers outside targets and run output. See [evaluation](src/mikoshi/evaluation.py).

## Inspect a run

The launcher prints the output directory and streams progress. Find recent runs
with `mikoshi runs`, or use `mikoshi runs --directory /path/to/output-parent`
for a custom location. In another terminal:

```sh
mikoshi status /absolute/path/to/run --json
mikoshi watch /absolute/path/to/run --interval-seconds 2
mikoshi stop /absolute/path/to/run
```

Default run storage is `%LOCALAPPDATA%\mikoshi\runs` on Windows and
`~/.local/state/mikoshi/runs` on macOS/Linux. `XDG_STATE_HOME` overrides the state
base on every host; `run.output` or `--output` selects an individual output path.
Quote paths containing spaces. Suggested Windows commands use PowerShell quoting.

`run.json` records settings, stop reason, usage, and evaluation; `records.jsonl`
holds the timeline. Safely finished trials retain `candidate/` for inspection.
See [the output schema](SCHEMA.md). Status is the latest recorded snapshot, not
proof the process is alive. Runs cannot be resumed, and forced shutdown can lose
in-flight results or final exports.

Ctrl-C in the running launcher requests a graceful stop; another interrupt can
force it. Ctrl-C in `watch` only stops watching. After `stop`, wait for the launcher
to finish and inspect the final record, including pending requests, unknown usage,
and any unconfirmed cleanup.

Windows also accepts Ctrl-Break. Host process cleanup uses Windows Job Objects;
Docker container cleanup is recorded separately. CPU and memory admission checks
read the Linux Docker host on every platform. The additional host-wide open-file
sensor is available on macOS and Linux; it is omitted on Windows.

| Recorded outcome | What to do next |
| --- | --- |
| `all_agents_yielded` | The peers finished with nothing queued. Inspect the candidate and checks; this does not establish task quality. |
| `budget_exhausted`, `wall_time_exhausted`, `context_window_exhausted` | A resource limit stopped work. Inspect partial results before choosing limits for a new attempt. |
| `stop_requested` | An operator requested a stop. Review settled spending and retained results. |
| `failed` or `interrupted` status | Read `error`, `execution_status`, and `evaluation` in `run.json` to distinguish execution, evaluation, and finalization failures. |
| `command_cleanup_unconfirmed`, `security_boundary_failure` | Review cleanup evidence before using the target or attempting another run. |

Keep the configuration file to repeat or maintain an experiment. Update prices
when your deployment's rates change. To start a new attempt, omit `run.output`
or pass a fresh `--output`; Mikoshi refuses to overwrite a populated run directory.
Use isolated repeated trials for comparisons that need equivalent starting files.

Useful retained files:

| Path within a run | Purpose |
| --- | --- |
| `run.json` | Effective settings, status, spending, and evaluation summary. |
| `records.jsonl` and `content/` | Complete action, conversation, request, and timing evidence. |
| `workspace/command-logs/` | Agent command output. |
| `candidate/` | Frozen files from safely finished trials. |
| `evaluations/` | Immutable scoring records. `mikoshi score OUTPUT` can rescore without launching agents. |

Use `mikoshi validate OUTPUT` to check bundle structure and referenced evidence
offline. Evaluation still needs your own criteria or a trusted external scorer;
the runtime does not decide whether the agents solved the objective.

Source code and visible instructions are sent to your model provider. Local
records retain conversations, operator messages, prompts, and command output.
Use authorized data and review exports before sharing them.
