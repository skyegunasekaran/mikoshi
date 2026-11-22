# Architecture

[CLI](src/mikoshi/cli.py) → [application service](src/mikoshi/application.py) →
[trial launcher](src/mikoshi/experiments.py) and
[peer runtime](src/mikoshi/runtime/) → [providers](src/mikoshi/providers/) and
[tools](src/mikoshi/tools/). [Persistence](src/mikoshi/persistence/) records evidence;
[isolation](src/mikoshi/isolation/) handles Docker commands. See
[extension examples](tests/runtime/test_extensions.py) for custom providers and team behavior.
Tasks, datasets, and graders stay outside the runtime.

The application service resolves settings and returns run records without printing
terminal output. [Run lifecycle](src/mikoshi/runtime/run_lifecycle.py) owns durable
status and stop escalation; [scheduling](src/mikoshi/runtime/scheduling.py) owns
ephemeral admission and retry state. Providers use
[typed request records](src/mikoshi/providers/requests.py) and
[bounded telemetry](src/mikoshi/providers/telemetry.py). Focused persistence
repositories own conversation, inbox, and observation queries; compound domain
actions retain their existing transaction boundaries.

[Filesystem access](src/mikoshi/filesystem.py) pins directory handles while
opening and replacing files: POSIX uses directory descriptors and
[Windows](src/mikoshi/windows_filesystem.py) uses native relative handles.
[Host process supervision](src/mikoshi/isolation/processes.py) uses POSIX groups
or Windows Job Objects and bounded pipe reads. These platform layers supervise
host resources; agent commands still execute only in Linux Docker containers.

Peers coordinate through shared Markdown pages and comments. The runtime seeds
two pages: read-only `/objective` and editable `/coordination`. It delivers bounded
references to peer page edits and discussions in one change feed. Explicitly
addressed comments and operator messages enter peer inboxes. Commands and archived
tool replies use one `read_output` interface, separate from authored pages.
Pages carry no ownership or workflow status; scheduling and audit records track
activations independently of the content peers write.

Each observation contains current dynamic fields, with stable startup context
restored after compaction. The configured roster, agents, activations, triggers,
inboxes and event ledger own execution state without duplicate organization
projections.
