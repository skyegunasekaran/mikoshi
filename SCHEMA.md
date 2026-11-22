# Input and output reference

`kind` distinguishes `input`, `run`, `record` and `evaluation` documents.
Print the JSON Schema with `mikoshi schema`. Every output bundle retains
that schema in `schema.json`.

## Input

One `kind: input` document contains `run`, `problem`, `pricing`, optional
`deployment` and optional `evaluation`. The objective, limits, roster, private
instructions and scheduled notifications are inline in `problem`.
The [configuration example](EXPERIMENTS.md#configuration) is a complete input.

`run.provider` and `run.team_protocol` select installed extensions independently.
The built-in team protocol is `peers`. Provider packages may accept a JSON
object under `deployment.provider.options`. Offline validation can read an
archived bundle without loading its extensions. Launch and preflight resolve
the selected names in the operator environment.

Paths resolve from the input file's directory. CLI flags override run settings.
`--task TEXT --agents N` is a convenience override normalized to the same problem
object; these are CLI options rather than config fields.

`run.json.input` retains the effective input, including CLI overrides. Resolved problem settings also record automatic time
reminders and the complete effective roster. `run.json.resolved` contains effective model/context/tool limits.
Do not infer defaults from omitted input values.

## Output index

Start with `run.json` for both individual trials and repeated experiments:

| Field | Meaning |
| --- | --- |
| `id`, `scope` | Unique identity; `trial` or `experiment` |
| `status`, `phase` | Overall controller status and latest phase |
| `execution_status`, `stop_reason` | Execution outcome independent of scoring |
| `started_at`, `updated_at`, `finished_at` | UTC timestamps; unfinished runs have null finish |
| `input`, `resolved` | Effective input and resolved operating settings |
| `provenance` | Checkout, runtime hash, system, endpoint and command environment |
| `summary` | Activation summaries and scheduler counters |
| `budget` | Accounted cost, rates, request entries, pending and unknown usage |
| `provider_metrics` | Timing, tokens, compaction and transport diagnostics |
| `progress`, `agents` | Latest overall and per-agent progress |
| `candidate` | Relative path, content/mode/tree digest, input/evidence digests and source path |
| `evaluation`, `evaluations` | Latest evaluation and paths to every retained evaluation |
| `trials`, `score_summary` | Attempt index and descriptive repeated-experiment summaries |
| `evidence` | Paths to schema, timeline, shared content, workspace and optional internal database |
| `error` | Controller error, or null |

`status: failed` can reflect an evaluation failure even when execution completed.
Resource-limited trials have `execution_status: stopped` and can still have valid
candidates and scores. A null evaluation means scoring was not performed;
`not_configured` means scoring was reached without a scorer.

Repeated-experiment indexes use the same schema. Their `trials` array retains
every attempt's ID, relative run directory, status, cost, exit code, evaluation
and supervisor report. These are recorded child summaries; each child's `run.json`
is its authority. Missing grades never become zero. Limits apply per trial.

## Provider metrics

Built-in remote providers retain the latest 256 collection events in
`provider_metrics.timing_events` and the latest 256 synchronous batch sizes in
`batch_sizes`. A collection event may contain several requests.
`timing_event_count` counts all collection events; `timing_history_limit`,
`timing_events_truncated` and `batch_sizes_truncated` describe the retained tails.
Routed providers retain each model's metrics under `by_model`; their combined
timing list also reports the lifetime event count and truncation flag.

Token totals, request counts, generation/transport/validation totals, and stage
totals cover the full provider lifetime. Generation wall time counts overlapping
requests once; stage wall times count overlap within each stage. Diagnostic
distribution counts, sums, means and maxima also cover the full lifetime.
`median` is exact while all observations remain retained. After eviction it is
null; `sample_median`, `sample_measured_requests` and `median_scope: retained_tail`
describe the retained observations. `slow_request_count` is a lifetime count;
`slow_requests` contains retained details, with `slow_requests_truncated` marking
omitted slow requests. Missing usage measurements remain explicit.

Every finalized dispatched request preserves its complete collection timing event
through a request completion record, described below. Exact lifetime medians and
all slow-request details can be recomputed from those archived events. Evidence
observer errors retain a bounded tail alongside `evidence_error_count` and
`evidence_errors_truncated`.

These bounds apply to telemetry history. The billing ledger and the registry of
request IDs for decision retry chains still grow across a run. Interval summaries
may retain spans that an older outstanding request can still overlap.

## Timeline

`records.jsonl` contains one JSON document per line:

| Field | Meaning |
| --- | --- |
| `kind` | `record` |
| `id`, `run_id`, `sequence` | Run-scoped identity and increasing recording ordinal |
| `recorded_at` | UTC time when this record was exported |
| `type` | `progress`, `decision`, `request`, `response`, `usage`, `notification`, `action`, `exchange`, `state` |
| `source_id` | Native identity such as `event:12` or `exchange:8`, when applicable |
| `agent_id`, `activation_id` | Actor/activation, or null for run-level records |
| `data` | Typed payload described below |

Sequence and recorded time describe recording, not action start or inferred
causality. Action ticks/commit ordinals describe domain commit order. Commands,
model calls and other peers can overlap.

- `progress`: phase, provider and scheduler progress. Payload `kind` identifies
  the progress event; final lifecycle progress records the stop reason.
  `decision_dispatched` means handoff to the dispatcher, and `decision_returned`
  means a result became available. These scheduler events do not establish HTTP submission.
- `decision`: a logical decision's UUID, actor, activation/step, metadata and
  references to prepared instructions, observation, history, full state,
  compaction state and tool schema. One decision identity survives transport retries.
- `request`: one provider request identity and its `provider_request_prepared`,
  `provider_request_submitted`, `provider_request_failed` or
  `provider_request_completed` event. `payload_ref`
  preserves the submitted/prepared JSON value. Preparation precedes budget
  admission; a prepared request can be rejected or cancelled without submission.
  Completion means the dispatched attempt has been collected and finalized,
  including failed or interrupted attempts; it does not establish a valid action.
  Its `timing_ref` preserves the full collection timing event, and `timing_index`
  identifies this request's entry in its parallel arrays. Requests collected
  together share one content-addressed timing value. Deduplicate `timing_ref`
  values by `content_id` when reconstructing lifetime collection history.
- `response`: `provider_response_received` preserves a returned JSON value or
  raw text in `body_ref`. Anthropic streams also produce a
  `provider_response_decoded` record for the parsed native message. Both share
  the same request ID. HTTP success, protocol validity and usage availability
  remain separate facts; malformed response IDs project to null but remain in
  the retained body. Nonfinite or non-JSON bodies are retained as text.
- `usage`: a `provider_usage_recorded` snapshot of the ledger entry after
  settlement, with request/decision IDs and a budget-entry index. Unknown usage
  remains explicit. Final ledger entries in `run.json.budget` carry the same IDs.
- `notification`: an inbox row queued in the source action's committed
  transaction, including recipient, sender, source reference, wake request,
  source event ID and full `message_ref`. Queueing does not establish submission.
- `action`: event ID/tick, commit ordinal, actor, activation, causal wave, decision
  round, activation step, action type, outcome, resource and payload. Resource
  versions reference exact bytes in `workspace/.cas/`. When known, payloads link
  `decision_id`, final originating `request_id`, `tool_call_id` and batch index.
  Asynchronous command results preserve the original authored call identity.
- `exchange`: archive sequence, actor, conversation ID, activation/step
  `exchange_key`, `decision_id`, final originating `request_id`, associated
  `request_ids`, and authored `tool_calls`. `messages_ref` preserves ordered
  observations, native assistant envelopes, tool replies and compaction
  checkpoints; `state_ref` preserves the observation baseline. Scripted
  providers have null transport/native call IDs. Skipped calls remain paired
  replies and do not become execution events.
- `state`: decoded final operator view of agents, pages with revisions/comments/
  snapshots, projects, commands, activations,
  triggers and inbox. Peer comments are public within their page scope. Command `log_path`
  values are bundle-relative. `recording_errors` reports failed export observers.

`request_id` identifies one prepared transport request, including compaction
continuations. `previous_request_id` chains requests within the same decision.
This differs from the provider's `previous_response_id`, retained in the payload,
which can connect conversation context across decisions. Native tool-call IDs
are scoped by their originating request/decision; they are not globally unique.
Associated `request_ids` may include attempts prepared but never submitted.
When Gemini omits a function-call ID, the Google adapter assigns a deterministic
local pairing ID scoped to that decision. The archived `google_content` and
submitted native parts remain unchanged; local IDs are not presented as IDs
returned by Google.

Submission events are recorded immediately before invoking the HTTP client.
They establish a submission attempt, not server receipt or model processing.
`included_content` identifies literal values by JSON Pointer in that payload:
native protocol tool results, and exact matches to the prepared model input,
full state or compaction state. `source_fields` names matching prepared fields;
notification source descriptors accompany matches to the prepared model input.
This is a partial list of mechanically known sources. Its absence does not
establish that a model lacked information: chained provider state and older
history can carry additional content.

Producing a tool result does not establish subsequent submission; the next
request's payload and included-content references establish its inclusion.
The archive does not contain every interrupted/in-flight response or hidden
provider state. Compaction changes visible memory without erasing the archive.
File-tool versions are retained; shell operations are not a complete
filesystem journal and concurrent writes prevent automatic attribution. Model
previews and retained command logs can be truncated separately.

## Content references

Large prepared inputs, payloads, bodies, timing events, messages and notification text are JSON
values stored under `content/`. References contain `content_id` (`sha256:...`),
bundle-relative `path`, `sha256`, `bytes` and `media_type: application/json`.
Repeated identical values reuse the same file. JSON formatting is canonicalized;
these are exact JSON values rather than byte-for-byte HTTP bodies. Raw SSE/text
is stored as a JSON string. The workspace CAS preserves file bytes separately.

Provider evidence excludes credential headers. Known host-key echoes in provider
evidence are replaced and `redacted: true` records that alteration. Operator
inputs, prompts and repository content otherwise retain their original values.
References always remain available in reader views even when content is expanded.

## Analysis without SQLite

```python
from mikoshi.schema import read_document, iter_records, read_content

run = read_document("/absolute/path/to/run/run.json", "run")
print(run["execution_status"], run["stop_reason"])
print(run["evaluation"])

for record in iter_records("/absolute/path/to/run", agent_id="agent-1", resolve_content=True):
    print(record["sequence"], record["type"], record["data"])

for record in iter_records("/absolute/path/to/run", record_type="exchange", resolve_content=True):
    for message in record["data"]["messages"]:
        print(record["agent_id"], message)

for record in iter_records("/absolute/path/to/run", record_type="request"):
    if record["data"]["kind"] == "provider_request_submitted":
        payload = read_content("/absolute/path/to/run", record["data"]["payload_ref"])
        print(record["data"]["request_id"], payload)
```

Other languages can read JSON/JSONL directly using `schema.json`. Analysis does
not require `.internal/`. Moving the public bundle preserves local evidence
references; source/scorer paths remain provenance. Explicit external evaluation
outputs remain external references and must be supplied separately.

## Evaluation and validation

Each evaluation records ID, run ID, evaluation time, status, named scores, scorer
provenance, candidate digest and output path. Scores contain `value` and optional
`explanation`/`metadata`. Rescoring appends an evaluation and updates the index.
Earlier evaluations cannot be overwritten by the scoring command.

Operator `grade(context)` receives `candidate`, `candidate_sha256`,
`artifact_manifest`, `run_dir`, `input` and `run`. Private scoring rules and
behavioral annotations stay outside the runtime.

```sh
.venv/bin/mikoshi validate experiment.json
.venv/bin/mikoshi validate /absolute/path/to/run
```

Validation makes no model requests and executes no candidate or grader code.
Bundle validation checks structure, record identity/order, referenced content/
blobs/logs, decision/request ownership, submitted-content pointers, usage and timing links,
known export errors, candidate/input fingerprints, evaluation ownership and
present child bundles.
It does not prove liveness, complete recording, correctness, provider access or
statistical validity. Validate after shutdown for a stable result. Fingerprints
do not protect against an operator who can rewrite host records.
