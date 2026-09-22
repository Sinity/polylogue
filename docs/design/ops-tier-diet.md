# The ops.db diet: amended design

Status: **amendment** to polylogue-pnxl6, 2026-09-22, resolved against head
`9ef655cb8`. This is a design record only. It authorizes no schema change and
runs nothing.

The original plan named a four-table `ops.db` target and a nine-table deletion
list. A worker refused to dispatch it on 2026-09-21 with evidence, and the
refusal holds. This amendment supplies the three things that refusal said were
missing: an unknown-representation for facts that stop being rows, a ruling on
`convergence_debt`, and a reader-to-replacement map. It also corrects the table
count the plan was written against.

## Ground truth at head

`polylogue/storage/sqlite/archive_tiers/ops.py` declares **17** tables, not the
18 the design names and not the 4 the acceptance assumes:

`schema_drift_samples`, `ingest_attempts`, `embedding_catchup_runs`,
`ingest_cursor`, `convergence_debt`, `whole_archive_convergence_pledge`,
`cursor_lag_samples`, `daemon_stage_events`, `daemon_events`,
`judgment_scheduler_receipts`, `daemon_lifecycle`, `secret_scan_status`,
`mcp_call_log`, `mcp_call_session_refs`, `route_observations`,
`fts_drift_samples`, `context_injection_ledger`.

`slo_samples` is already gone. `schema_identity` is bootstrap metadata, not an
ops data table, and carries its own DDL-lifecycle waiver.

`ops.db` is the **disposable** tier. It is rebuildable, carries no version
chain, and its DDL feeds the derived schema-identity stamp — so every change
here is *additive-derived* or a derived deletion, never a durable migration.
The one exception is the `mcp_call_log` move below, whose destination is
`audit.db` and therefore *is* additive-durable.

## Amendment 1: the unknown-representation

Every table this plan deletes replaces a persisted row with a structured event
plus an in-process rollup. That trades a durable fact for a fact scoped to the
current process, and the whole plan turns on saying so honestly rather than
letting a fresh process's empty rollup read as a healthy zero.

**Do not invent a vocabulary. `polylogue/scenarios/workload.py` already declares
one**, and it is validated rather than conventional:

* `WorkloadPhaseObservation.unavailable` is an explicit tuple of measure names
  that were *not* measured. Its `__post_init__` rejects a name that is not a
  declared measure, and rejects a measure that is both observed and unavailable
  — so "unmeasured" cannot silently coexist with a value.
* `WorkloadRunStatus.MEASUREMENT_UNAVAILABLE` is the run-level terminal for the
  same fact.

The rule this amendment adopts, stated as a contract any replacement must meet:

1. **A rollup-backed field is three-state, never two.** It is *measured* with a
   value and an explicit sample window, or *unavailable* with a named reason.
   `0`, `null`, an absent key, and "ok" are all forbidden spellings of
   unavailable. This is the same operator ruling that forbids silent truncation:
   bound with a typed refusal or not at all.
2. **A restart begins unavailable.** Until the current process has observed a
   sample, every rollup-backed field on `polylogue status` and `/api/events`
   reports unavailable with the reason "no sample in this process". It never
   replays an older run's health or latency as current.
3. **The rollup carries its own provenance**: runtime/build identity, sample
   window start, last update, and an observation-loss count for events the
   rollup dropped under backpressure. A nonzero loss count is itself a
   degradation, not a footnote.
4. **An unavailable field degrades the envelope.** The terminal-outcome contract
   already ranks `degraded` above `empty`; a status envelope with any
   unavailable rollup field is `degraded`, so zero rows behind a named gap is
   never served as an empty scope.
5. **No diagnostic may gate authority.** A dropped event or an overflowing
   observer queue cannot advance an acquisition cursor, certify a source as
   retained, or cancel a committed write. This is why only tables in the
   "delete" column below may become rollups at all: everything that gates
   authority stays a row.

## Amendment 2: the `convergence_debt` ruling

**Ruling: `convergence_debt` stays. Remove the dependency on polylogue-foour.**

The plan deferred this to `foour`, and the 2026-09-21 refusal recorded that
`foour` is an open epic with no `convergence_debt` ruling in its acceptance.
Re-read at head: that is not a timing problem. `foour`'s five acceptance
criteria are about retiring stage/repair coupling around profile publication;
none of them asks for a disposition of `convergence_debt`, and its only
`convergence_debt`-shaped child is closed. Waiting for a ruling that criterion
will never produce is a dead gate, so this amendment makes the call directly.

The call is *keep*, on three grounds:

* **It is the declared retryable-backlog mechanism.** Hot-file deferral and
  `convergence_debt` are what retain work the converger could not finish. A
  rollup cannot hold a retry queue across a restart.
* **It has the widest reader set in the tier**: 22 production modules, including
  the daemon's convergence stages, debt alerting and debt status, the live
  ingest cursor, archive query source-freshness, the public API, archive
  verification, and the workload profile schema. Nothing in the plan names a
  replacement for any of them.
* **Deleting it would violate this amendment's own rule 5.** Debt decides
  whether a stage re-runs; that is authority, not diagnostics.

## Amendment 3: the reader-to-replacement map

Ten tables stay, two move, five are deleted. **AC1's four-table target is
refuted and should be restated as ten.** No table may be removed before its row
in this map names a live replacement and every reader listed has been
re-pointed.

### Stay as rows — original keeps

| Table | Why it cannot be a rollup |
| --- | --- |
| `ingest_cursor` | acquisition authority; a lost cursor re-ingests or skips |
| `ingest_attempts` | attempt history across restarts, and the ETA denominator |
| `daemon_lifecycle` | run identity, which the rollup's own provenance cites |
| `daemon_events` | backs the `/api/events` SSE stream and must survive a reader reconnect |

### Stay as rows — corrections to the deletion list

These four were listed for deletion as "telemetry samples". None is one.

| Table | Readers | Why deletion is wrong |
| --- | --- | --- |
| `convergence_debt` | 22 modules | Amendment 2 above |
| `secret_scan_status` | `security/secret_scan.py` | an **incremental coverage cursor**, not a sample: the sweep selects sessions not yet covered at the current `SECRET_SCAN_VERSION`, and commits coverage rows in the same transaction as the findings they cover. A per-process rollup makes every restart rescan the whole archive, and breaks the version-bump rescan design outright. |
| `whole_archive_convergence_pledge` | `sources/live/cursor.py` | **live lease state**: inserted, listed while OPEN, and released by id. An in-memory rollup cannot return open pledges after a restart, so pledges leak permanently. |
| `context_injection_ledger` | `context/scheduler.py`, and the public `list_context_injection_ledger` through `api/archive.py` / `api/parity.py` | **not superseded by `user.db context_deliveries`.** The ledger is per-item admission evidence — decision, token cost, source-local rank, budget before/after, disclosure and authority verdicts. `context_deliveries` is one row per delivered snapshot image. Different grain, different facts, and the public surface serves the former. |

### Stay as rows — folds that do not fit

| Table | Why the fold into `ingest_attempts` fails |
| --- | --- |
| `embedding_catchup_runs` | carries scanned / embedded / skipped / error_count / embedded_messages / estimated_cost_usd. Folding needs six sparse columns on a table that is not about embedding, or a JSON blob its readers would have to decode. |
| `judgment_scheduler_receipts` | carries a **counter-sum invariant** (`batch_limit`, considered, accepted, rejected, escalated, idempotent, failed) validated at the write boundary, and exists precisely so queue-health readers never decode a JSON payload. Its accessor `_read_latest_judgment_scheduler_receipt` feeds seven modules across the daemon, CLI status, judge command and payload surfaces. |

Re-decide both as **keep**. If a narrower shape is still wanted later, it is
their own attempt-shaped table, not a widened `ingest_attempts`.

### Move to `audit.db`

| Table | Destination | Regime |
| --- | --- | --- |
| `mcp_call_log` | `machine_requests` | **additive-durable** on audit |
| `mcp_call_session_refs` | `machine_request_parts` | same change |

These are machine-request evidence and audit already declares both
destinations. This half needs a numbered migration under
`storage/sqlite/migrations/audit/`, behind a verified backup, one
`PRAGMA user_version` step. Derive the slot number at implementation time from
the directory's contents, not from this page: audit currently sits at
`ARCHIVE_FORMAT_FLOOR_VERSION` and its migration directory holds only
`__init__.py`, while `source/` and `user/` each already carry a numbered train.
Readers to re-point: `daemon/http.py`, `operations/route_observation.py`,
`cli/commands/diagnostics.py`.

### Delete, becoming structured events plus a rollup

Each row names the replacement its readers move to. Every one of these carries
a *diagnostic* fact: none of them gates a cursor, a retention certificate or a
write.

| Table | Readers | Replacement |
| --- | --- | --- |
| `schema_drift_samples` | `analysis/schema_drift.py`, `cli/commands/diagnostics.py`, `daemon/health.py`, `operations/daemon_status.py`, `schemas/drift_sentinel_sampling.py` | drift event + rollup; **paired with `fts_drift_samples`** |
| `fts_drift_samples` | `cli/commands/diagnostics.py` | same drift event; the two are written and read together on one diagnostics route, so deleting either alone leaves a half-migrated command |
| `cursor_lag_samples` | `daemon/cursor_lag_baseline.py`, `operations/daemon_workload_probe.py`, `schemas/generation/archive_workload_profile.py` | lag event + rollup; the workload-profile schema needs an explicit unavailable term |
| `route_observations` | `cli/commands/diagnostics.py`, `operations/route_observation.py` | route-timing event + rollup |
| `daemon_stage_events` | `api/archive.py`, `daemon/convergence_stages.py`, `daemon/status.py`, `daemon/health.py`, `daemon/catchup_status.py`, `daemon/metrics.py`, `maintenance/archive_verification.py`, `operations/daemon_workload_probe.py`, `sources/live/cursor.py`, `storage/archive_readiness.py` | stage event + rollup — **the largest and last**: ten reader modules including archive readiness and archive verification. Verify for each whether it reads stage events as *diagnostics* or as *evidence of stage completion*; the second kind is authority and must not become a rollup. |

`schema_drift_samples`/`fts_drift_samples` move as a pair. `daemon_stage_events`
moves last, after its ten readers have each been classified.

## What this amendment does not decide

* The per-chunk connection budget (AC2/AC4). Separate measurement: the real
  remaining ops cost at head is connection *creation* outside the held
  `ops_write_scope`, not commit count, and that is a live-ingest change.
* Whether the four sibling beads that still record obligations against the
  removed tables have been amended. That gate is unchanged: publish this map,
  amend them, and only then remove a table.
* Any deletion order beyond the two constraints stated above.
