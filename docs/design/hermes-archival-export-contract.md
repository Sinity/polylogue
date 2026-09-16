# Hermes archival export contract (fs1.7)

Status: v1 defined and implemented **on the Polylogue side only**. The
corresponding upstream Hermes-repo commit is external (open source, not
owned by this workspace) — it was not authored or merged from this
environment, and this document is the handoff proposal a Hermes maintainer
would review to implement the producer side. Until that lands, "checked-in
bytes match Hermes" is proven only against the fixture in this repository
(`tests/fixtures/hermes/archival_export/v1/example-session.json`), not
against a real Hermes build.

Implementation: `polylogue/schemas/hermes_export_contract.py` (the export
schema), `polylogue/sources/hooks.py` (the local event carriers, general
across Claude Code/Codex/Hermes), `polylogue/sources/parsers/hermes_lifecycle.py`
(the runtime lifecycle-event taxonomy and snapshot reconciliation). Tests:
`tests/unit/sources/test_hermes_export_contract.py`,
`tests/unit/sources/test_hook_carriers.py` (Hermes parametrizations),
`tests/unit/sources/parsers/test_hermes_lifecycle.py`.

## Why two channels, not one

Hermes lifecycle hooks are best-effort: a synchronous HTTP call from a hook
can be lost during a Polylogue outage. Two channels exist so a lost event
never means lost history:

1. **Session snapshot export** (this document, `hermes_export_contract.py`):
   a versioned, self-contained per-session document Hermes produces from one
   consistent read transaction. This is the recovery source of record — if
   every runtime event for a session were lost, re-exporting the snapshot
   still reconstructs the full session.
2. **Runtime lifecycle-event carriers** (`sources/hooks.py`,
   `sources/parsers/hermes_lifecycle.py`): low-latency, high-frequency
   evidence about *what happened during* a session — model attempts/retries/
   fallbacks, tool start/finish/failure/denial, approvals, subagent start/
   finish, compaction, rewind, and the durable-finalize/per-turn-end
   distinction. Event bodies carry ids, hashes, timings, and outcomes —
   **never** a second copy of message text (enforced by
   `_reject_duplicated_transcript`, tested end-to-end through both the
   `contrib/polylogue-hook` shell prototype and the `polylogue-hooks` pip
   package).

`sources.parsers.hermes_lifecycle.reconcile_lifecycle_events` renders the gap
between these two channels *visible* rather than assuming either is complete:
unpaired start/finish events and events referencing a message id the snapshot
does not (yet) retain both surface as explicit caveats on a
`HermesLifecycleReconciliation`, not a silently-accepted partial history.

## Session export schema (v1)

One `HermesArchivalExportV1` document per session revision
(`polylogue/schemas/hermes_export_contract.py`). Every field mentioned in the
2026-07-10 Nous follow-up refinement is present:

| Field | Purpose |
| --- | --- |
| `schema_version`, `producer` | Producer/schema version — a version bump this parser does not recognize fails loudly (`HermesExportSchemaError`), never silently coerces. |
| `profile_id` | Stable install/profile identity (hashed, no raw path — mirrors `hermes_state._profile_key`). |
| `session_id`, `session_revision_hash` | Identity plus the dedup key: identical hash ⇒ same revision (skip); changed hash ⇒ new retained history, never an in-place overwrite. |
| `messages[].state` | One of `active`/`inactive`/`rewound`/`compacted`/`observed` — every message state Hermes can hold. |
| `messages[].tool_calls`/`tool_results` | Stable `action_id`; `tool_results.output_preview` is explicitly *bounded* — the full output already lives in `messages[].text` for that turn, so a full second copy here would itself violate the no-duplicated-transcript rule this whole contract exists to uphold. |
| `usage`, `cost` | Token lanes and full cost provenance (billing provider/mode, estimated/actual cost, status, source, pricing version) — mirrors `hermes_state._COST_FIELDS`. |
| `parent_session_id`, `parent_relationship` | Explicit fork/resume/subagent/continuation relationship, not inferred from heuristics on the consumer side. |
| `archive_state`, `handoff_platform` | Archive/handoff/finalization state. |
| `repository_cwd`, `git_branch`, `git_repo_root` | Repository/cwd when available. |
| `finalized` | True only for a durable close (`on_session_finalize`), never a per-turn `on_session_end`. |

See the fixture for a worked example covering every message state, a
`resume` parent relationship, and a `handoff-complete` archive state.

## Runtime lifecycle-event taxonomy

`sources.parsers.hermes_lifecycle.HERMES_LIFECYCLE_EVENT_TYPES`:

```
model_attempt, model_failure, model_retry, model_fallback,
tool_start, tool_finish, tool_failure, tool_denial,
approval_request, approval_response,
subagent_start, subagent_finish,
compaction, rewind,
on_session_end,        # per assistant turn — NOT a durable-session signal
on_session_finalize,    # exactly once, when the session is durably closed
context_injected,       # a Polylogue-compiled context pack reached a live turn (fs1.11)
```

`on_session_end` vs. `on_session_finalize` is a hard requirement: a Hermes
turn ends many times per session, but a session finalizes once. Conflating
the two would make "session ended" ambiguous evidence for every consumer
downstream (forensics, cost reconciliation, recall auditing).

## Delivery: append-only local carriers, acquired like any other source

Producers call `polylogue.sources.hooks.append_hook_event(provider="hermes",
...)` — the same contract Claude Code/Codex hooks already use
(`SUPPORTED_PROVIDERS`, extended for fs1.7). The write is one
newline-terminated JSON line appended with a single `O_APPEND` write to the
producer process's own carrier under
`carriers/hermes/<UTC day>/<pid>.ndjson`. There is no temp file, no rename and
no fsync on the producer path: the archive acquires the carrier's bytes as an
ordinary raw-only artifact and the `hook_events` derivation materializes its
events out of those retained bytes. Killing Polylogue mid-delivery and
restarting materializes **exactly once**, because both identities involved are
content-derived: the producer's own `event_id`, and the event's carrier
coordinate (the file plus the byte offset of its line).

**Batch delivery** needs no separate mechanism. A Hermes integration that
cannot wire a synchronous hook call at all can write batched envelopes into a
carrier from a cron job or a post-session export step; the same watcher and
the same derivation pick it up, because a carrier is just an append-only
JSONL file in a watched directory.

## Two working local prototypes (not a verified Hermes integration)

Both are genuine, tested producers of this exact carrier format — not stubs —
but neither is confirmed against Hermes's own hook invocation contract (no
local checkout of the Hermes hook source was available while writing this):

- `contrib/polylogue-hook --provider hermes` (POSIX shell + embedded Python,
  zero Python-package dependency — the same script Claude Code/Codex hooks
  already use in production, extended with the Hermes event vocabulary and
  the duplicated-transcript guard).
- `polylogue-hooks` (the standalone pip package, `packaging/polylogue-hooks/`)
  — same extension, for installs that don't want a dependency on the main
  `polylogue` distribution.

Both are exercised end-to-end (subprocess, not mocked) in
`tests/unit/sources/test_hook_carriers.py::test_published_hook_adapters_append_then_materialize`
and `::test_published_hook_adapters_refuse_duplicated_transcript_payloads`.

## What is explicitly NOT done here

- **The upstream Hermes-repo commit.** This document, the schema module, and
  the fixture are the proposal; a Hermes maintainer with write access to that
  repository still needs to implement the producer side and wire real hook
  call sites to `contrib/polylogue-hook`/`polylogue-hooks`.
- **Physical session-tree merge of observer-layer spans into the
  conversational session.** See
  `polylogue/sources/parsers/hermes_spans.py` (fs1.2) — ATIF and ATOF spans
  each land as their own artifact-qualified observer-evidence session
  (`observer:atif:<id>` / `observer:atof:<id>`, fs1.14 fixed a prior
  collision where both shared one `observer:<id>` identity), correlated by
  shared Hermes session id at read time by the caller, not physically merged
  into the state-db-ingested message tree.
