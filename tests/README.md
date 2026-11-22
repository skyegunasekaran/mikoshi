# Infrastructure tests

Run `python tests/support/development.py check` (or `make check`) after the
[contributor setup](../README.md#contributing). The suite exercises real SQLite state and Git worktrees, with scripted or
mocked providers and Docker orchestration. It makes no paid model calls.

| Area | Retained coverage |
| --- | --- |
| `runtime/` | Peer lifecycle, durable conversations, spending admission and billing, async tool batches, pressure-gated FIFO admission |
| `collaboration/` | Shared page revisions, public comments, exact recipient wakeups, inbox delivery, one change feed, and project-scoped output/activity inspection |
| `conversations/` | Native OpenAI retries, invalid-call recovery and compaction; Anthropic streamed compaction and instruction restoration |
| `isolation/` | Experiment configuration, preflight, launch, status and exports; immutable Docker images, worktree mounts and fail-closed cleanup |
| `workspace/` | Active-command stop records, scoped guidance before execution, attach-client output, credential stripping, process cleanup, and durable candidate integrity |

The lifecycle scenario also checks traversal, symlink and hardlink rejection.
Native experiment tests cover isolated repeated trials, cancellation, external
scoring, and rescoring without model calls. Evaluation fixtures contain toy
operator callbacks; benchmark acceptance criteria stay outside the runtime.
Package boundaries and installed launch, watch, status, stop, and score entry
points are verified separately by `python tests/support/development.py build`
(or `make build`). The Python helper works in PowerShell without Make.

The operator workflow also covers guided configuration creation, offline planning
without credentials, readable mocked execution, and saved-run discovery.
Long-history tests verify provider timing tails,
complete archived timing evidence, and exact lifetime accounting. CI runs
the checks, release build, and dependency-lock verification on Windows, Linux, and macOS with
Python 3.12 and 3.14; `uv sync --locked --extra dev` reproduces its dependencies.

Configuration, launch, scoring, and portable-bundle checks share one run.
Guided and flag-based setup share one operator flow; retry behavior is exercised
through peer scheduling rather than a table of implementation values.

Prefer extending a representative infrastructure flow over adding argument-value
grids, exact prose snapshots or implementation-level tests. Keep shared fixtures
in `support/`; author model actions explicitly and do
not repair them in fixtures. Task acceptance tests and model-quality evaluations
belong outside this suite.

The attach-client supervisor test launches controlled local child processes and
requires permission to inspect process groups on POSIX or create Job Objects on
Windows. Symbolic-link cases skip only when Windows denies link-creation
privileges; junction, hardlink, traversal, and ordinary filesystem checks remain
active. This is not a live Docker isolation
test. Report host restrictions and unconfirmed cleanup accurately. Live provider
and Docker integration checks must be identified separately; mocked success does
not establish experiment quality or container isolation.
