# Agent/session identity and orchestration model

## Purpose

Polylogue needs enough identity to answer four product questions without creating a separate agent ontology:

1. which provider session or imported run acted;
2. which child/subagent run it spawned;
3. what context was visible at session, subagent, continuation, or review-injection boundaries;
4. which review/check/tool events were merely observed versus actually delivered, acknowledged, or acted on.

The model stays run-centric. Provider/session ids remain native evidence. Polylogue refs provide stable addressability for work packets, recovery reports, query rows, API/MCP payloads, and later web drilldown.

## Identity layers

### Provider identity

A provider identity is the raw origin that produced the transcript or stream, for example `codex-session`, `claude-code-session`, `chatgpt-export`, or a future local capture origin. It is not semantic authority for task state, delivery state, or user intent.

Current DTO field: `ProjectedRun.provider_origin`.

### Session identity

A session identity is the provider-native session/thread id when available. Polylogue preserves it as `ProjectedRun.native_session_id` and also exposes `ObjectRef(kind="session", object_id=<session_id>)` where a session itself is addressable.

For subagents, `ProjectedRun.native_parent_session_id` preserves the parent provider session id when the parent is known. This is evidence, not an orchestration registry.

### Run identity

A run is one bounded execution attempt inferred from evidence. Runs are addressed as `run:<id>` object refs. A main session run usually uses the session id. A child run uses the resolved child session id, child session id, task id, or a bounded fallback if no stable provider id exists.

Run lineage is a tuple of run refs, root first. A child subagent run therefore carries `(run:<parent>, run:<child>)` and `parent_run_ref=run:<parent>`.

### Agent identity

An agent ref names the acting harness role inside a run projection, not a durable principal or mailbox. The current shape is `agent:<harness>/<role-or-subagent-type>`, for example `agent:codex/main` or `agent:codex/Explore`.

Agent refs are deliberately small: they let work packets say who acted without claiming stable long-term identity across resets, machines, accounts, or future orchestrators.

### Context snapshot identity

A context snapshot is what a run could see at a boundary. Snapshots are addressed as `context-snapshot:<run-or-session-id>:<boundary>[:ordinal]`.

Current boundaries are:

- `session_start` for the root transcript/session evidence;
- `subagent_start` for child task dispatch and inherited parent state;
- `review_injection` when review material is actually visible in context;
- `resume` and `unknown` as existing compatibility vocabulary.

Inheritance modes remain `clean`, `summary`, `prefix`, `snapshot`, `injected`, or `unknown`.

## Event and ref vocabulary

### ObjectRef kinds consumed by this lane

- `session`: archived session/transcript.
- `message`: archived message.
- `block`: message block/tool block.
- `run`: projected execution attempt.
- `agent`: acting harness role or subagent type inside a run projection.
- `context-snapshot`: visible context at a boundary.
- `observed-event`: evidence-backed event in a run projection.
- `tool-call`: provider tool call when a stable tool id exists.
- `subagent-report`: extracted child/subagent handoff or final report.
- `check-run`: named local or remote check/test run.
- `github-issue`, `github-pr`, `github-review`: external coordination objects.
- `branch`, `file`: work target context.

### ObservedEvent delivery states

Delivery remains event state over refs, not a mailbox model:

- `observed`: evidence exists.
- `seen_by_tool`: the run fetched or displayed it.
- `injected_context`: it appeared in a context snapshot.
- `acknowledged`: the agent or user explicitly acknowledged it.
- `acted_on`: a patch, test, reply, or follow-up addressed it.
- `unknown`: evidence is insufficient.

This keeps the critical boundary clear: a GitHub review existing is not evidence that the run saw it.

## Minimal code slices now landed

1. `ObjectRef` now includes `tool-call` and `subagent-report` so work-packet rows can cite subagent/task evidence without private renderer ids.
2. `ProjectedRun` carries `provider_origin`, `native_parent_session_id`, `agent_ref`, and `lineage_refs`.
3. Subagent runs now expose `agent:<harness>/<subagent_type>` and `subagent-report:<session>:<tool-or-task-id>` refs.
4. Review-injection recovery events now create a `ContextSnapshot(boundary="review_injection", inheritance_mode="injected")` and the matching observed event points at both the snapshot and review ref.
5. Work-packet execution/subagent sections render those refs, making the existing continuation bundle more useful without adding storage tables.

## What not to model yet

Do not add `AgentPrincipal`, `AgentMailbox`, `RoleSpec`, `CommunicationEvent`, delivery queues, durable context-envelope tables, spawn APIs, or orchestration-runtime state until real fixtures prove Run + ContextSnapshot + ObservedEvent cannot represent the behavior.

Do not treat OpenTelemetry spans, GitHub reviews, harness names, or source origins as authority. They are evidence sources or projections over the run/event/ref model.

Do not add a new table for work packets by default. A work packet remains an on-demand `Bundle(kind=work_packet)` over sessions, runs, observed events, assertions, refs, and raw evidence until measured query/reuse pressure justifies persistence.

## Next exact residuals

- Add `tool-call` object refs to tool-summary execution rows once the provider tool id can be surfaced without changing existing external refs.
- Add a missed-review fixture where `review_posted` exists before `pr_merged` but no `seen_by_tool`, `review_injected_context`, `acknowledged`, or `acted_on` event exists.
- Expose the same projected identity fields through API/MCP recovery-work-packet JSON snapshots after the work-packet JSON profile is next regenerated.
