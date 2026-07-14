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
schema), `polylogue/sources/hooks.py` (the durable local event spool, general
across Claude Code/Codex/Hermes), `polylogue/sources/parsers/hermes_lifecycle.py`
(the runtime lifecycle-event taxonomy and snapshot reconciliation). Tests:
`tests/unit/sources/test_hermes_export_contract.py`,
`tests/unit/sources/test_hook_spool.py` (Hermes parametrizations),
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
2. **Runtime lifecycle-event spool** (`sources/hooks.py`,
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

## Delivery: durable local spool, with file-watch fallback

Producers call `polylogue.sources.hooks.enqueue_hook_event(provider="hermes",
...)` — the same atomic-enqueue-then-idempotent-drain contract Claude
Code/Codex hooks already use (`_SUPPORTED_PROVIDERS`, extended for fs1.7).
The write is a durable, immutable JSON file placed in `pending/` via
temp-file-plus-`os.replace`-plus-`fsync` (`_atomic_json_write`); the daemon
only moves a file into `acknowledged/` after its `raw_hook_events` row has
committed. Killing Polylogue mid-delivery and restarting drains the spool
**exactly once** — proven by
`test_hermes_hook_spool_replay_is_idempotent_after_interrupted_acknowledgement`.

**File-watch fallback**: if the synchronous producer path is unavailable (the
hook binary can't run, or a batch of events needs to be replayed after an
outage), the daemon's `LiveWatcher` already watches the same
`pending_hook_spool_dir()` via `watchfiles` and drains anything it finds —
this is not a new mechanism, it is the existing live-watch path
(`polylogue/sources/live/watcher.py`) pointed at the Hermes spool directory
exactly like it already is for Claude Code/Codex
(`test_live_watcher_drains_the_configured_hook_spool_root`,
`test_live_watcher_observes_a_spool_created_after_startup`). A Hermes
integration that cannot wire a synchronous hook call at all can instead
periodically write batched envelopes into `pending/` from a cron job or
post-session export step and rely on this same watcher/drain path — no
separate "batch mode" needs to be built.

## Two working local prototypes (not a verified Hermes integration)

Both are genuine, tested producers of this exact spool format — not stubs —
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
`tests/unit/sources/test_hook_spool.py::test_published_hook_adapters_spool_then_materialize`
and `::test_published_hook_adapters_refuse_duplicated_transcript_payloads`.

## What is explicitly NOT done here

- **The upstream Hermes-repo commit.** This document, the schema module, and
  the fixture are the proposal; a Hermes maintainer with write access to that
  repository still needs to implement the producer side and wire real hook
  call sites to `contrib/polylogue-hook`/`polylogue-hooks`.
- **Physical session-tree merge of observer-layer spans into the
  conversational session.** See
  `polylogue/sources/parsers/hermes_spans.py` (fs1.2) — spans land as their
  own observer-evidence session, correlated by shared Hermes session id, not
  physically merged into the state-db-ingested message tree.
