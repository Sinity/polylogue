# Maintenance

This guide is for operators choosing between
`polylogue ops maintenance preview`, `polylogue ops maintenance plan`,
`polylogue ops maintenance run`, `polylogue ops reset`, and "do nothing — the
daemon will catch up." It also collects runbook recipes for the most
common operational incidents.

## Applying a durable schema change train

Durable schema changes are an offline release operation. Before applying a
`source.db`, `user.db`, or `audit.db` migration above its adoption floor,
confirm that the release contains the matching
`migrations/{source,user,audit}/NNN.train.json`
sidecar. The sidecar reserves the exact slot and SQL hash and records the
runtime and restart evidence needed for the change.

Stop `polylogued`, create a fresh verified backup with the normal
`backup_archive(..., verify=True)` route, then invoke the existing maintenance
command with that manifest:

```bash
polylogue ops maintenance migrate-tier source \
  --backup-manifest /path/to/verified-source-backup/manifest.json \
  --output-format json
```

The command acquires the daemon startup exclusion and archive ownership before
opening SQLite. It refuses when a live daemon or another archive writer holds
either authority. The migration runner then validates the package sidecar,
revalidates the backup against the current database, and performs the numbered
SQL step in the existing transaction. It verifies row and schema parity,
SQLite integrity, foreign keys, and canonical DDL parity before commit. A
failed transaction is rolled back and may be retried after the cause is
repaired.

The JSON output reports the migration receipt and stopped-daemon authority.
Restart health and runtime-consumer convergence are the final lifecycle proof
and are recorded by the durable train lifecycle API, not inferred from this
command's migration result alone.

### Rebuild deployment-currency preflight

Before a managed `rebuild-index`, confirm that the package selected for the
operation owns the live durable schemas. The read-only preflight checks every
canonical durable migration tier: `source.db`, `user.db`, and `audit.db`.
`index.db` may be behind because the rebuild is the supported way to replace
that derived tier.

```bash
polylogue ops maintenance rebuild-index --preflight --output-format json
```

It emits `rebuild-schema-currency` JSON with each durable tier's observed and
package-expected `user_version`, and exits nonzero when a durable tier differs.
The execution route checks it before consuming the schema-inference receipt,
repeats it after archive ownership acquisition, and rejects daemon bulk
transaction creation before any bookkeeping or candidate generation.

For a safe deployment recovery, first choose the exact target package commit.
With the daemon stopped, create a fresh verified full-evidence backup. If the
preflight reports that a newly introduced durable tier is absent, initialize
only that absent file through the archive ownership gate:

```bash
polylogue ops maintenance migrate-tier audit --initialize-missing --output-format json
```

The flag stages the canonical database in the private archive directory and
publishes it with an atomic no-replace link. It refuses any existing path,
including one created concurrently, and never replaces durable data. Run
`migrate-tier source`, `migrate-tier user`, and `migrate-tier audit` for
existing tiers when the target package requires numbered migrations, then
deploy that exact package. Run the preflight again and require a ready result
before invoking `polylogue ops maintenance rebuild-index`; use that blue-green
command rather than `ops reset --index` for an active managed generation.
Restart the daemon only after the rebuilt generation is promoted and the
post-deploy status shows no durable-tier mismatch.

For the conceptual model behind derived insights and the FTS / blob
substrate, see [architecture.md](architecture.md) and
[internals.md](internals.md). For daemon ownership of the inline
maintenance loop, see [daemon.md](daemon.md).

## What a maintenance operation is

A *maintenance operation* is an explicit, resumable, idempotent pass
over already-ingested archive state. It does **not** acquire new
source data. It does **not** rewrite or delete imported sessions
beyond targeted cleanup. It rebuilds, repairs, or prunes the things
the archive depends on but does not own as primary data:

- derived read models (session insights, actions, work threads,
  day/week summaries, message-type classifications);
- search indexes (the FTS5 projections over messages and action
  events);
- backfilled columns (e.g. `message_type` for rows ingested before the
  classifier existed);
- archive-cleanup scopes (orphaned messages, orphaned content blocks,
  empty sessions, orphaned attachments, orphaned blobs);
A WAL checkpoint is not a maintenance operation. Ingest runs bounded passive
checkpoints after commits, the daemon runs periodic truncate checkpoints, and
status/metrics report WAL pressure. If WAL stays large after those automatic
paths have had a chance to run, treat that as a daemon/storage bug rather than a
user-facing repair verb.

A maintenance operation is distinguished from three adjacent things:

| Surface | What it does | When you reach for it |
| --- | --- | --- |
| **Import** (`polylogued`, `polylogue import PATH`) | Daemon acquires source payloads, parses provider records, writes archive rows, and advances derived models *for the new rows*. `polylogue import PATH` asks the running daemon to schedule an explicit file or directory. | You have new exports/sessions to import. |
| **Daemon convergence** (`polylogued` inline loops) | Performs the same operations as ingest plus automatic WAL checkpointing, FTS convergence, heartbeat, health checks, and embedding/profile catch-up. | The daemon is running. You do nothing. |
| **Maintenance** (`polylogue ops maintenance ...`) | Rebuilds derived state and prunes archive debt over already-ingested rows. Read-only by default; mutations are explicit. | A derived model is stale or missing for old rows that the daemon's small inline windows will not pick up. |
| **Reset** (`polylogue ops reset`) | Deletes data: the SQLite database, the blob store, attachments, cache, OAuth tokens, or named sessions (soft-delete via tombstones). | The data itself is wrong or unwanted, not just a derived projection of it. |

The order of preference is: **do nothing → daemon → maintenance →
reset**. Reset is the only one that destroys primary data.

## Typed scopes

Maintenance targets are grouped into four scopes:

| Scope | Mode | Destructive | Targets |
| --- | --- | --- | --- |
| `derived` (derived_repair) | repair | no | `session_insights`, `message_type_backfill` |
| `archive_cleanup` | cleanup | **yes** | `empty_sessions`, `orphaned_blobs` |
| `backfill` | repair | no | column/row backfills surfaced by the planner (currently subsumed by `derived`). Re-acquiring raw artifacts from source, WAL checkpointing, and repairing FTS coherence are daemon/ingest convergence responsibilities, not maintenance targets. |

The canonical target list is enforced by
`polylogue/maintenance/targets.py`. The CLI `--target` option's
`click.Choice` is built from
`MAINTENANCE_TARGET_NAMES`, so the source is the type system: an
unknown target is rejected at the CLI boundary.

## When to use which surface

```text
                          something looks off
                                 |
            +--------------------+--------------------+
            |                                         |
   one or a few sessions               wide swath of the archive
   look wrong / outdated                    looks stale (FTS misses
            |                                hits, session profiles
            |                                missing for old data, ...)
            |                                         |
   polylogue ops maintenance preview          polylogue ops maintenance preview
   (scoped to that session)          (no scope — full inventory)
            |                                         |
   nothing stale?                          nothing stale?
   - the data really is that way.          - the daemon already converged.
     stop. open an issue with a            stop. nothing to do.
     concrete acceptance criterion.
            |                                         |
   stale rows reported?                    stale rows reported?
   polylogue ops maintenance run               polylogue ops maintenance run
     --session <id>                     (no scope) or
     [--target ...]                          --target <subset>
            |                                         |
   still wrong? the data itself            failures reported?
   is wrong, not the projection.            inspect failure_samples,
   polylogue ops reset --session           re-run with --operation-id
   <id>  (tombstones it; preserves         to resume from cursor.
   identity ledger for re-import)
```

Heuristics:

- **Preview before plan, plan before run.** `preview` is read-only;
  `plan` is a dry-run summary; `run-preview` is a heavier, resumable
  dry run that exercises the real execution path; `run` is the only
  mutating verb.
- **Prefer the narrowest target.** `--target session_insights` is
  cheaper and safer than rebuilding everything.
- **Do not reach for `reset` to "fix a stale projection."** That is
  what `maintenance` is for. Reset destroys the primary data the
  projection is built from.
- **If the daemon is running and the issue is recent**, wait one
  convergence cycle (~10 minutes for FTS, ~5 minutes for WAL) and
  re-check before reaching for maintenance.
- **If maintenance reports zero stale rows but the archive still
  looks wrong**, the bug is upstream (ingest, parser, schema) — open
  an issue, do not loop on maintenance.

## Subcommands

### `polylogue ops maintenance blob-namespace-quarantine`: offline filesystem quarantine

Read-only by default. This is the actuator for entries that
`BlobStore.iter_namespace()` classifies outside the canonical
`blob/<2-hex>/<62-hex>` layout: historical SQLite `-wal`/`-shm` sidecars,
stranded `.blob.*` files, malformed shards, and non-regular entries. It does
not classify orphaned canonical blobs and never runs GC.

```bash
polylogue ops maintenance blob-namespace-quarantine --output-format json
polylogue ops maintenance blob-namespace-quarantine --plan \
  --backup-manifest /path/to/verified-source-backup/manifest.json \
  --output-format json
polylogue ops maintenance blob-namespace-quarantine --apply \
  --backup-manifest /path/to/verified-source-backup/manifest.json \
  --receipt-dir /path/to/new/namespace-quarantine-receipt \
  --output-format json
polylogue ops maintenance blob-namespace-quarantine --recover \
  --receipt-dir /path/to/existing/namespace-quarantine-receipt
```

`--plan` is the backup-gated operator audit. It authenticates the supplied
source-tier backup against an immutable read of `source.db`, then emits a
typed census of canonical blobs, SQLite `-wal`/`-shm` sidecars, `.blob.*`
temporary files, and other invalid entries. It does not create receipts,
checkpoint SQLite, move files, delete files, or change archive rows. Run this
plan before any later offline quarantine decision.

This plan is an offline safety prerequisite, not a production cleanup receipt
or a complete bead-closure claim. The full-hash pristine receipt required by
`r9xsj` remains a separate residual dependency, and production cleanup remains
a separate operator-authorized residual dependency. No receipt is claimed here.

Apply requires the daemon stopped, no archive writer lease, the archive-wide
exclusive maintenance lease, a successful attested source-tier backup manifest
whose live identity still matches `source.db`, and a clean WAL checkpoint. The
receipt directory must be new and explicit. Before the first move, the command
writes and fsyncs immutable `before.json`; afterwards it writes immutable
`after.json`. Invalid entries move with same-filesystem `os.replace()` into a
sibling `blob-namespace-quarantine/<operation-id>/` tree. Existing destination
paths, path escapes, symlinks in required directories, stat/read failures, and
canonical-shaped hash mismatches all refuse the operation.

No SQLite rows are changed, canonical blobs are never moved, and no deletion
or garbage collection occurs. The after receipt proves the canonical inventory
is byte-identical, every candidate is present in quarantine with its recorded
no-follow tree digest, no invalid namespace entries remain, and the full
canonical hash pass has no failures. Run the independent final operator gate
before treating a production pass as complete:

```bash
polylogue ops doctor --blob-integrity --full --format json
```

`--recover` is read-only and idempotent. It reports `rolled_back` when every
source still exists and every destination is absent, `committed` when every
destination matches the before receipt, and `indeterminate` for mixed or
conflicting state. It never attempts a repair move.

### `polylogue ops maintenance blob-reference-liveness` - historical blob-ref reconciliation

Read-only by default. It classifies source-tier `blob_refs` rows with the
actual referent join for each `ref_type`; source-tier `attachment` refs join
`raw_sessions.raw_id` because they are keyed by the parent raw acquisition,
not by index-tier `attachment_refs` or `raw_artifacts.artifact_id`.
`hook_payload` refs join `raw_hook_events.hook_event_id`. Unknown or unavailable
ref types are counted as explicit census dispositions and block an apply; blob
GC also retains their bytes until a typed disposition is available. The
command never deletes blob files. Use `--census-only` for the privacy-safe
production census. It returns counts and dispositions only, without reference
identifiers, source paths, hashes, or unknown reference-type names.

```bash
polylogue ops maintenance blob-reference-liveness --output-format json
polylogue ops maintenance blob-reference-liveness --census-only --output-format json
polylogue ops maintenance blob-reference-liveness --apply \
  --backup-manifest /path/to/verified-source-backup-manifest.json \
  --receipt-file /path/to/new/blob-ref-liveness.jsonl \
  --output-format json
```

The apply command requires the daemon and all archive writers to be stopped,
an existing verified backup manifest covering the current `source.db`, and a
new receipt path that does not already exist. It revalidates the backup and
reclassifies under `BEGIN IMMEDIATE` before fsyncing the prepared receipt and
deleting the exact candidate set. Review the receipt's final `committed` line
before treating the pass as complete.

### `polylogue ops maintenance blob-reference-closure` - acquired reference closure

Read-only by default. It checks that each `raw_sessions` row has exactly one
matching `raw_payload` ref and that each acquired index attachment is reachable
through `attachment_refs`. Raw gaps are repaired from the retained raw row's
exact hash, path, size, and acquisition timestamp. Attachment gaps are repaired
only when a complete reparse of authoritative `source.db` bytes reproduces the
attachment identity and its owning message still exists. Other rows are
reported as typed blockers and remain untouched.

```bash
polylogue ops maintenance blob-reference-closure --output-format json
polylogue ops maintenance blob-reference-closure --apply \
  --backup-manifest /path/to/verified-full-evidence-manifest.json \
  --receipt-file /path/to/new/blob-reference-closure.jsonl \
  --output-format json
```

Apply requires the daemon to be offline, a verified backup manifest covering
both `source.db` and `index.db`, and a new receipt path. It inserts exact refs
only, never deletes or replaces existing refs. The source and index commits are
recorded separately in the receipt so a retry can safely continue an additive
repair. Reindex acceptance runs the same closure check against the candidate
index before promotion.

### `polylogue ops maintenance hook-payload-ref-reconcile` - legacy hook-ref repair

Read-only by default. It classifies historical orphaned `raw_payload` refs and
only re-keys a row when its deterministic legacy id exactly matches one
unambiguous `raw_hook_events` candidate. Unmatched rows remain untouched.

```bash
polylogue ops maintenance hook-payload-ref-reconcile --output-format json
polylogue ops maintenance hook-payload-ref-reconcile --apply \
  --backup-manifest /path/to/verified-source-backup-manifest.json \
  --receipt-file /path/to/new/hook-payload-reconciliation.jsonl \
  --output-format json
```

Apply requires the daemon and all archive writers to be offline, a verified
backup manifest for the current `source.db`, and a new receipt destination.
The receipt records the tool version, backup-manifest identity, exact
pre/post classifications, reconciled hook ids, and a terminal committed or
recovered state. If an interrupted apply leaves a prepared receipt, rerun the
same command with that receipt path to record recovery; use a fresh path only
after reviewing the recovered terminal state.

### `polylogue ops maintenance preview` — staleness inventory

Read-only. Produces a per-model inventory of stale, missing, orphan,
or version-mismatched rows tagged with a typed `InvalidationReason`
(`missing`, `stale`, `orphan`, `missing_provenance`,
`version_mismatch`, `orphan_archive_row`). Models with nothing stale
produce explicit zero rows rather than being absent from the output.

```bash
polylogue ops maintenance preview
polylogue ops maintenance preview --scope derived
polylogue ops maintenance preview --scope archive_cleanup --output-format json
polylogue ops maintenance preview --shallow   # skip expensive full-verification path
```

Use this before triggering `run` so you know what would be touched
and why. A write-watching SQLite hook in the test suite confirms zero
writes during a preview pass.

### `polylogue ops maintenance plan` — dry-run summary

Read-only. Resolves targets, evaluates affected rows, and produces a
`BackfillOperation` envelope without executing any repair. Use it to
sanity-check what the next `run` will do.

```bash
polylogue ops maintenance plan
polylogue ops maintenance plan --target session_insights --target message_type_backfill
polylogue ops maintenance plan --output-format json | jq .
```

`--output-format json` emits the shared
`MaintenanceOperationEnvelope` so the CLI output is byte-for-byte
identical to the daemon HTTP and MCP responses.

### `polylogue ops maintenance run-preview` — resumable dry run

Read-only. Runs the exact same resumable replay path as `run` --
including per-target repair simulation and checkpoint tracking -- but
never mutates the archive. This is a heavier, more faithful dry run than
`plan`: it exercises the same code each target's real `execute` step would
take, not just an affected-row estimate. Use it to combine the safety of
`plan` with the full execution-path code before committing to `run`.

```bash
polylogue ops maintenance run-preview
polylogue ops maintenance run-preview --target session_insights --output-format json
```

### `polylogue ops maintenance run` — execute

Runs the resolved targets. Per-target failures are isolated as
`FailureSample` entries; one failing target does not abort the rest.
Pass `--operation-id <uuid>` together with `--resume` to pick up an
interrupted operation. This command always mutates; it carries no
`--dry-run` flag -- use `run-preview` for the read-only twin.

```bash
polylogue ops maintenance run --target session_insights --output-format json
```

### Blob-reference integrity — preview and apply pairs

`polylogue ops maintenance blob-reference-debt` and
`blob-reference-recovery-plan` are read-only classification/planning
commands. The two commands that can actually mutate the archive each
follow the same preview/apply split as `run`/`run-preview`: a dedicated
read-only `-preview` command with no `--yes`/`--apply` flag, and a lean
apply command that always mutates.

```bash
# Read-only: simulate what a replace-from-source pass would change.
polylogue ops maintenance blob-reference-replace-from-source-preview --output-format json

# Apply: always mutates; --manifest-file is required.
polylogue ops maintenance blob-reference-replace-from-source \
  --manifest-file /tmp/replace.jsonl --output-format json

# Read-only: simulate what an orphan prune would remove.
polylogue ops maintenance blob-reference-prune-orphans-preview --output-format json

# Apply: always mutates; writes a quarantine JSONL before deleting rows.
polylogue ops maintenance blob-reference-prune-orphans \
  --quarantine-file /tmp/quarantine.jsonl --output-format json
```

### `polylogue ops maintenance reindex-canary` — inactive-generation semantic diff

Read-only with respect to the active index. Before a full reindex, this command selects a bounded representative set of sessions, rebuilds those raws into an inactive generation, and diffs the resulting sessions, messages, blocks, links, and derived rows against the active generation. It requires `--no-promote`. A run with observed differences writes an unreviewed durable report and exits non-zero. Re-run with `--review-manifest` to persist one classification per difference, then use `--consume-report` to validate the reviewed report and approve its evidence. Approval never authorizes promotion. Treat every difference as either an expected effect of a named repair or a newly discovered defect. It is a preflight gate, not a replacement for the full managed rebuild.

```bash
polylogue ops maintenance reindex-canary \
  --archive-root /realm/tmp/polylogue-canary-archive \
  --input /realm/tmp/polylogue-canary-archive/index.db \
  --schema-inference-receipt /realm/tmp/schema-inference-gate-receipt.json \
  --sample 100 \
  --report /realm/tmp/polylogue-reindex-canary.json \
  --no-promote \
  --output-format json
```

After reviewing the observed identities printed by the failed run, persist the classifications and validate the report. Consumption acquires the same archive ownership and rebuild lease as the rebuild path, verifies referenced raw-payload bytes through `BlobStore`, and revalidates the source closure, candidate generation, receipt, and comparison immediately before approval. Membership rows and logical-source-key expansion are part of the receipt, so drift fails closed:

```bash
polylogue ops maintenance reindex-canary \
  --archive-root /realm/tmp/polylogue-canary-archive \
  --input /realm/tmp/polylogue-canary-archive/index.db \
  --sample 100 \
  --report /realm/tmp/polylogue-reindex-canary.json \
  --review-manifest /realm/tmp/polylogue-reindex-reviews.json \
  --no-promote

polylogue ops maintenance reindex-canary \
  --archive-root /realm/tmp/polylogue-canary-archive \
  --report /realm/tmp/polylogue-reindex-canary.json \
  --consume-report \
  --no-promote
```

### `polylogue ops maintenance verify-archive` — coherence gate

Read-only. Runs a fixed registry of independent checks over the whole
archive and reports each as `ok`/`warning`/`error`/`skip` plus evidence
numbers, never just a boolean. This is the repeatable substitute for the
manual checklist an operator used to run by hand after a blue-green index
rebuild or a full restore — "does the archive prove its own restore?"

```bash
polylogue ops maintenance verify-archive
polylogue ops maintenance verify-archive --output-format json | jq .
polylogue ops maintenance verify-archive --check tier-schema --check pointer-coherence
polylogue ops maintenance verify-archive --strict   # also fail on warnings, not only errors
```

Checks (see `polylogue/maintenance/archive_verification.py` for the
extensible registry):

| Check | Proves |
| --- | --- |
| `tier-schema` | Every tier file (source/index/embeddings/user/ops) exists at its current `PRAGMA user_version`. |
| `pointer-coherence` | The conventional `index.db` path and the active `.index-active-pointer` generation agree (an interrupted blue-green promotion leaves these diverged — polylogue-k8kj class). |
| `source-index-coverage` | Every raw session with a complete census has a materialized index session (missing work), and every index session's `raw_id` still resolves to a real raw row (orphans) — reported as counts and id samples, not booleans. |
| `fts-parity` | `messages_fts`/`blocks_command_trigram` exactly cover their source `blocks` rows, archive-wide, with the worst-offending sessions surfaced by name. |
| `lineage-sanity` | `session_links.resolved_dst_session_id` and `branch_point_message_id` resolve to real sessions/messages (the latter is deliberately not a foreign key — see the data-model docs). |
| `planner-stats` | `sqlite_stat1` covers `blocks`/`messages`/`action_pairs` (warn-level: a fresh generation without `ANALYZE` picks pathological query plans — polylogue-l3tk class). |
| `counts-summary` | Archive-wide session/message/block counts and an origin breakdown — the numbers-freeze starting point for an operator handoff. |

Exit code is non-zero when any check reports `error` (or, with `--strict`,
`warning`). A single check's failure — including a tier database being
temporarily busy under a concurrent rebuild — never aborts the rest; each
check independently reports its own outcome.

### `--operation-id` and `--resume`: worked example

Replay execution writes a small JSON state file under
`<archive_root>/.maintenance-state/<operation_id>.json` after each
target completes. The state file is removed when the operation
terminates successfully. The cursor is an opaque string
(`target:N`) encoding the index of the next target to run.

```bash
# Start an operation, capture its id.
op=$(polylogue ops maintenance run --output-format json \
       --target session_insights \
       --target message_type_backfill \
     | jq -r .operation_id)

# ... operation is killed mid-run (Ctrl-C, OOM, oncall reboot) ...

# Resume from the persisted cursor — same id, same target set, no flag needed.
polylogue ops maintenance run --operation-id "$op" \
       --target session_insights \
       --target message_type_backfill

# Explicit cursor override (rare — for surgical replays).
polylogue ops maintenance run --operation-id "$op" --resume target:2 \
       --target session_insights \
       --target message_type_backfill
```

Two correctness guarantees the executor provides:

1. **Convergence.** Running the same operation twice in a row produces
   no additional changes after the first pass converges. The
   underlying repair functions are idempotent by construction; the
   replay loop adds the multi-target convergence guarantee.
2. **Resume integrity.** Targets already marked done in the state
   file are skipped on resume, and no target is run twice.

If the state file is missing and `--operation-id` is supplied without
`--resume`, the executor treats the id as a fresh start.

## Scope filters

The current shipping surface accepts repeatable `--session-id`, `--origin`,
`--source-family`, `--source-root`, `--since`/`--until`, `--failure-kind`,
and `--parser-version` filters. Each target decides which dimensions it can
honestly narrow; unsupported dimensions are preserved in the envelope but do
not pretend to reduce the affected-row count.

```bash
polylogue ops maintenance run --session-id abc123 --target session_insights
polylogue ops maintenance run --origin claude          --target session_insights
polylogue ops maintenance run --since 2026-04-01 --until 2026-05-01 \
                          --target session_insights
polylogue ops maintenance run --failure-kind parse_error --target message_type_backfill
```

Until #1196 lands, the only way to narrow a run is through `--target`
and `--scope`. Do not script against flag names that are not yet on
`polylogue ops maintenance run --help`.

## Status surface

A long-running operation exposes its current cursor and in-flight
failure samples through three coherent surfaces:

| Surface | How to read |
| --- | --- |
| CLI (`polylogue ops maintenance run`) | Progress lines printed to stderr each checkpoint: `[processed/total] target cursor=target:N failures=K`. The final stdout block reports `operation_id`, target results, elapsed time, and `Failures:` listing. |
| Daemon HTTP | `POST /api/maintenance/plan` and `POST /api/maintenance/run` return the same `MaintenanceOperationEnvelope` as the CLI. A dedicated `GET /api/maintenance/status/<op_id>` endpoint is tracked in [#1197](https://github.com/Sinity/polylogue/issues/1197). |
| MCP | `maintenance_preview` and `maintenance_execute` return the same envelope as the CLI/HTTP. A `maintenance_status` tool is tracked in [#1197](https://github.com/Sinity/polylogue/issues/1197). |

All three surfaces share the same `MaintenanceOperationEnvelope`
contract from `polylogue/maintenance/envelope.py`, so a `jq` script
that parses the CLI JSON also parses HTTP and MCP responses byte for
byte. The envelope carries `operation_id`, `status`, `targets`,
`resume_cursor`, `affected_rows`, `started_at`, `completed_at`,
per-target `results`, and a bounded `failure_samples` envelope.

## Failure surface

Replay failures are bounded by `BoundedFailureSamples` (a small
fixed cap per operation) so a runaway target cannot fill the
operation envelope with samples. Failures appear in three places:

- **`polylogue ops maintenance run` stderr** — the final `Failures:`
  block lists `<kind> @ <locator>: <message>` for each captured
  sample. A truncation marker is printed if the cap was hit.
- **`polylogue ops doctor` / `polylogue ops doctor`** — readiness reports
  include maintenance-target readiness rows
  (see `MaintenanceTargetSpec.doctor_readiness_operation` and
  `doctor_repair_operation`).
- **Daemon raw-failure surface** — once
  [#1198](https://github.com/Sinity/polylogue/issues/1198) lands,
  maintenance failures will route into the same raw-failure surface
  that ingest uses, so they show up in `polylogued` status, the
  health checks added in
  [#844](https://github.com/Sinity/polylogue/issues/844), and any
  notification backend configured under `[notifications]`.

If a replay fails repeatedly with the same `FailureSample.kind` and
`locator`, that is the signal to escalate from "re-run with
`--operation-id`" to "open an issue against the underlying repair
function."

## Idempotency contract

Re-running the same operation against unchanged input is a no-op.
Concretely:

- `preview` is read-only; running it twice produces the same
  inventory minus timing jitter.
- `plan` is read-only; running it twice produces the same envelope
  modulo timestamps.
- `run` converges: the second `run` for the same target set against
  unchanged source rows reports zero affected rows and zero failure
  samples. This is enforced by repair functions being idempotent by
  construction (see `polylogue/storage/repair.py`) plus the replay
  loop's per-target convergence guarantee.

The convergence guarantee is what makes resume safe: an interrupted
operation that already advanced past target *N* will not redo target
*N* on resume, and the redo would have been a no-op anyway.

---

## Runbooks

The runbooks below assume:

- You have a recent local backup (`polylogue ops backup` — see
  [daemon.md § Operator-Owned Tasks](daemon.md#operator-owned-tasks)).
- You can stop the daemon if a runbook requires exclusive write
  access (`systemctl --user stop polylogued.service`).
- You ran `polylogue ops maintenance preview` first to confirm the
  symptom matches the runbook.

### Recovering from a stale FTS index

**Symptoms.** Search returns fewer hits than expected for known
strings. `polylogue ops doctor` reports a `messages_fts` discrepancy.
`polylogue ops diagnostics workload`
shows non-empty `fts_trigger_state.missing` or `regressed` triggers.

For a deployment-bound, read-only gate that checks schema versions, exact FTS
debt, raw frontier integrity, replay candidates, cursor failures, and
convergence debt, add `--preflight`:

```bash
polylogue ops diagnostics workload --preflight --json > preflight.json
jq '.preflight_ledger | {state, blocking_checks, warning_checks}' preflight.json
```

The preflight reports quarantined raw bytes and missing
`raw_membership_census` rows by origin. Quarantine is authority-pending
evidence, not an automatic failure. Missing census is `coverage_unknown` and
blocks the gate until a verdict exists. Only source rows with present,
non-terminal census evidence are classified as actionable parse/validation
debt.

**Root cause.** `messages_fts` is a contentless FTS5 table
(`content=''`, `contentless_delete=1`) indexing `blocks.search_text`,
kept in sync by three rowid-keyed triggers on `blocks`
(`messages_fts_ai`/`_ad`/`_au`) — there is no bulk-suspend/rebuild step to
interrupt (an earlier design that suspended these triggers during bulk
writes was removed; SQLite DDL is transactional, so that suspension
window never actually produced committed drift). A missing or regressed
trigger today means schema corruption or an incomplete/partial schema
application, not an interrupted suspension window. The daemon startup
check and FTS convergence loop restore a missing trigger by re-running
the canonical DDL, which is idempotent (`CREATE TRIGGER IF NOT EXISTS`).

**Recovery.**

```bash
# 1. Confirm trigger state.
polylogue ops diagnostics workload --json | jq .fts_trigger_state
# Expect all_present=true. If `missing` is non-empty, continue.

# 2. Start daemon convergence. Startup/read paths restore the FTS invariant.
polylogued run

# 3. Verify.
polylogue ops diagnostics workload --json | jq .fts_trigger_state.all_present
# Expect: true.
```

If FTS remains non-ready after daemon convergence, the underlying issue is
structural (missing columns, corrupted index file, or a broken write path).
Stop the daemon, restore from backup or rebuild the affected index tier, and
open an issue with the probe output attached.

### Inspecting a raw-authority census

Raw source-to-index convergence records an immutable census in `source.db`.
Status and daemon receipts expose a bounded summary plus a URI such as
`polylogue://raw-authority-census/census:42:.../0`. Resolve that same URI from
the CLI without copying every plan into the status payload:

```bash
polylogue ops maintenance raw-authority-census \
  'polylogue://raw-authority-census/census:42:.../0' \
  --output-format json
```

The response includes bounded before/postflight plan summaries, counts,
digests, and a `detail_query_handle` for each plan. It deliberately does not
inline raw-ID lists, witnesses, preconditions, application receipts, or blocker
documents: one authority component may contain thousands of each.
`next_query_handle` advances across the plan inventory. `--limit` is bounded to
1–500; `--offset` can override the offset encoded in the URI.

Resolve a census or plan detail handle as bounded canonical-JSON text chunks:

```bash
polylogue ops maintenance raw-authority-detail \
  'polylogue://raw-authority-detail/census:42:.../raw-replay:.../current/0' \
  --chunk-chars 16384 \
  --output-format json
```

The first `current/0` read returns digest-bound continuation handles. If the
underlying outcome or blocker changes between chunks, the old continuation
fails closed; restart from the record's `current/0` handle.

Concatenate `chunk` values by following `next_query_handle`, then verify the
reconstructed document against `document_sha256`. The chunk size is bounded to
256–65,536 characters. MCP clients resolve both census and detail URIs through
their matching resource templates, so CLI and MCP expose the same complete but
bounded ledger.

Every receipt identifies its `mode` (`census`, `dry_run`, or `apply`), whether
the parser census was `quiescent`, and its lifecycle. Apply receipts remain
`planned` until every selected immutable plan has an outcome; startup recovery
then validates exact source, application/membership, accepted-head, and session
postconditions before marking an interrupted pass `executed`. Readiness never
reports a `planned` row as the latest completed census and exposes its pending
count separately. Finalization also proves that every retryable or
carried-forward plan has the identical immutable ID in the postflight census;
a partially applied component cannot be mislabeled as unchanged work.

Parser census itself advances through a bounded number of authority components
per pass. If uncensused components remain, the pass persists a non-quiescent
zero-plan census receipt and returns without replay; a later daemon tick resumes
from the per-raw current-parser receipts. Immutable plans are published only
after the complete transitive census is quiescent.

Raw-authority preview is the narrow exception to the generic read-only preview
rule above: it may durably record source-tier parser/census observations so a
moved-path component has one crash-safe identity across preview and apply. It
never selects or applies an index replay plan.

A stale precondition or incomplete application receipt creates a durable,
fail-closed blocker. List unresolved blockers before resolving one -- this is
the read-only discovery surface for an operator who does not already know an
exact `--blocker-id` (previously the only way to find one was page-walking
`raw-authority-census`/`raw-authority-detail` or writing an ad hoc script
against the live archive):

```bash
polylogue ops maintenance raw-authority-blockers --output-format json
```

Each row's `kind` distinguishes `stale_plan` (replan against current
source/index evidence is enough), `frontier_judgment` (requires an accepted
judgment assertion id plus `disposition=retain_canonical_authority`, per the
conflicting-authority frontier), and `frontier_obligation` (the other
frontier obligation states -- missing bytes, unresolved provenance, corrupt
-- which resolve like an ordinary blocker: no judgment assertion is
required). The listing is bounded to `--limit` (1-500, default 100) per
call; if the response's `truncated` field is `true`, pass
`--offset <next_offset>` to read the next page. After inspecting the census
URI and current evidence, explicitly reopen replanning with a recorded
rationale:

```bash
polylogue ops maintenance raw-authority-blocker-resolve \
  --blocker-id 'raw-authority-blocker:...' \
  --reason 'reviewed current source/index evidence; replan from this state' \
  --yes
```

Resolution never applies the stale plan. It stores the replacement plan
witness in the resolution receipt; the next ordinary convergence pass plans
and validates current evidence normally. Both commands route through
`OperationExecutor`/`BlockerResolveActuator` (polylogue-t46.9 phase 3):
PREPARE previews the exact blocker target and EXECUTE requires a
confirm-flag-strength authorization bound to that plan's hash, refusing
(`preview_stale`) if the blocker was concurrently resolved between preview
and confirm.

### Measuring Codex UUID-title coverage

Codex sessions without a resolvable title (thread name / authored history /
`state_5.sqlite` title / a human-authored message) fall back to their native
UUID as `title` (polylogue-ih67). `polylogue ops diagnostics
codex-title-census` reports corpus-wide resolved/unresolved counts without
reading message content or file paths -- only the `sessions` table's
`title`/`title_source`/`message_count`/`authored_user_message_count` columns:

```bash
polylogue ops diagnostics codex-title-census --json
```

Every still-UUID-titled session is classified by structural reason rather
than an undifferentiated "unresolved" count: `no_messages_materialized` (the
raw record produced zero messages), `no_human_authored_message` (messages
exist but none are human-authored -- no message-text fallback is possible),
`not_yet_reprocessed_with_assembly` (a human-authored message exists but
`title_source` was never stamped -- an ordinary `reprocess` pass should
resolve it), and `human_authored_present_synthesis_failed` (enrichment ran
but title synthesis produced nothing usable, e.g. whitespace-only text).

Save a snapshot before a reprocess pass and compare after:

```bash
polylogue ops diagnostics codex-title-census --save /tmp/before.json
# ... run polylogue ops reprocess or polylogued run ...
polylogue ops diagnostics codex-title-census --save /tmp/after.json
polylogue ops diagnostics codex-title-census --compare /tmp/before.json /tmp/after.json
```

### Draining the convergence-debt queue

**Symptoms.** `polylogue ops diagnostics workload` reports a non-trivial
`convergence_debt` section. `polylogue analyze` shows derived
materialization counts (`session_profile`, `actions`,
`work_threads`) lagging behind `sessions`.

**Root cause.** The daemon's inline convergence loops process a
bounded slice each cycle. If ingest outpaced the loop (initial
backfill of a large archive, bulk re-import, schema bump) the
remaining backlog will not drain inside one cycle.

**Recovery.**

```bash
# 1. Snapshot the workload before.
polylogue ops diagnostics workload --json > /tmp/before.json

# 2. Run daemon convergence. It drains raw materialization, FTS, embeddings,
#    and ordinary derived read models in bounded batches.
polylogued run

# 3. Snapshot after and diff.
polylogue ops diagnostics workload --json > /tmp/after.json
polylogue ops diagnostics workload --compare /tmp/before.json /tmp/after.json
```

Expect `convergence_debt.delta` to be negative across each stage.
If a stage's delta is zero or positive, that target's repair function
is not draining the backlog — capture the `FailureSample` block and
escalate.

### Rolling back a bad schema upgrade

**Symptoms.** Polylogue refuses to start after a schema bump:
`SchemaVersionError: database is version N, code expects version M`.
Polylogue uses durability-keyed schema versioning (see
[internals.md § Schema Versioning Model](internals.md#schema-versioning-model)):
derived tiers rebuild, while durable `source.db`, `user.db`, and `audit.db` may
advance only through explicit additive numbered migrations. There is no
auto-downgrade.

**Root cause.** A new release advanced one tier's schema version and the
database is on the previous version. There is no reverse in-place migration.

**Recovery.**

```bash
# 1. Confirm the version mismatch.
polylogue --version
sqlite3 ~/.local/share/polylogue/index.db "PRAGMA user_version;"

# 2. STOP the daemon to release exclusive locks.
systemctl --user stop polylogued.service

# 3. Classify the tier before acting.

# 3a. Code rollback (preferred when a release just went out and you
#     have not yet relied on any new feature):
#     install the previous polylogue version, leave the database
#     alone, restart the daemon.

# 3b. Derived-tier forward rebuild: keep source/user/audit/embedding tiers safe,
#     move the mismatched index database aside, and re-ingest/rederive
#     the rebuildable index with the new polylogue binary.
cp ~/.local/share/polylogue/index.db /tmp/index-before-rebuild.db
# ...run the documented re-ingest/rederive flow for the release, verify
# it opens cleanly with the new polylogue binary, then restart production.

# 3c. Durable-tier additive migration: keep the daemon stopped, create and
#     scratch-verify a minimal backup, then use its authenticated receipt.
polylogue ops backup --output-dir /path/to/staging \
  --profile user_overlays --verify
polylogue ops maintenance migrate-tier user \
  --backup-manifest /path/to/staging/polylogue-archive-*/manifest.json \
  --output-format json

# 4. Restart and verify.
systemctl --user start polylogued.service
polylogue ops doctor
```

The daemon and `migrate-tier` command share the stable
`<archive-root>/.archive-ownership.lock` archive lease. `daemon.pid` is process
metadata only and is never reclaimed by unlinking it as a lock. A crash during
the train apply phase leaves a checksummed manifest under
`.maintenance-state/durable-change-trains/`; the next daemon startup acquires
the same archive lease, reconciles the interrupted version, and persists the
recovery evidence before opening normal archive components.

Never hand-edit a tier or use a plain manifest as migration authority. A
durable migration requires a successful scratch-restore receipt authenticated
by the exact live tier's local key; public hashes and an in-memory "Verification: OK" are
insufficient. Keep the backup as an independent copied file set: linked or
symlinked tiers are rejected because they do not survive mutation of the live
database. If release notes provide neither an additive durable migration
nor a derived-tier rebuild plan, keep the daemon stopped and roll back the
binary.

### Proving an archive is coherent after a rebuild or restore

**Symptoms.** None yet — this is the proactive check to run *before* symptoms
appear, immediately after any operation that replaces or promotes a whole
tier: a derived-tier rebuild (`polylogue ops reset --index && polylogued run`),
a durable-tier migration (previous runbook), or a full restore from backup.

**Why this matters.** A blue-green index rebuild can leave the conventional
`index.db` path stale while `.index-active-pointer` already points at the
promoted generation (polylogue-k8kj: an interrupted rebuild left a fresh
process silently reading a near-empty 4-session file instead of the real
18,796-session archive). A restore can silently drop rows a backup profile
never covered. `verify-archive` turns the manual "does this look right?"
inspection into one repeatable, extensible command instead of an ad hoc
sequence of `sqlite3` queries re-derived by hand each time.

**Recovery / verification.**

```bash
# Run every check; --strict also fails on warnings (e.g. missing sqlite_stat1).
polylogue ops maintenance verify-archive --output-format json | jq .

# Or narrow to the checks most relevant to the operation just performed:
polylogue ops maintenance verify-archive --check tier-schema --check pointer-coherence
```

A clean run (`"blocking": false`) is the proof the archive is coherent: every
tier is present at its current schema version, the active pointer and
conventional path agree, source-vs-index materialization has no gaps or
orphans, FTS parity holds archive-wide, and lineage references resolve. A
non-zero exit means read the failing check's `evidence` payload (id samples,
worst-offending sessions, tier paths) before deciding whether the drift is
expected mid-rebuild noise or a real regression — do not silently retry.

### Investigating a stuck source

**Symptoms.** A source family stops producing new sessions even
though source files are present. `polylogue sources` shows a source
with stale `last_seen`. Daemon logs show repeated parse errors for
the same artifact id.

**Recovery.**

```bash
# 1. Identify the stuck source.
polylogue sources --output-format json | jq '.[] | select(.healthy==false)'

# 2. Inspect raw-artifact failures from that source.
polylogue ops diagnostics workload --json \
  | jq '.recent_attempts[] | select(.source_paths[]? | contains("PATH"))'

# 3. Pull the raw artifact directly to inspect it.
curl -sf "http://127.0.0.1:8765/api/raw_artifacts/<artifact_id>" | jq .

# 4. If the artifact is malformed at the source layer (truncated
#    JSONL, missing required field), the fix is upstream — fix the
#    source file, then ask the running daemon to import it:
polylogue import <path-to-source>

# 5. If the artifact is fine but the parser rejects it, the fix is
#    in the parser. File an issue with the provider and artifact details.

# 6. While the upstream fix is in flight, you can tombstone the
#    bad session so it stops blocking convergence:
polylogue ops reset --session <conv_id>
```

Do **not** reach for `polylogue ops maintenance run` to "fix" a stuck
source. Maintenance operates over already-ingested rows; if the rows
are not in the archive yet, maintenance has nothing to do.

### Recovering a corrupt blob store

**Symptoms.** `polylogue ops doctor` reports unreadable blobs. Session
exports fail with "blob not found". `polylogue ops diagnostics workload`
shows divergence between `blob_links` count and the count of files
under `blob/`.

**Root cause.** A blob file under `<archive_root>/blob/ab/cdef...`
was deleted, partially overwritten, or its prefix shard directory
permissions changed. Or: a GC pass with a known orphan-detection bug
([#818](https://github.com/Sinity/polylogue/issues/818)) deleted a
blob that was still referenced.

**Recovery.**

```bash
# 1. Stop the daemon to halt new writes.
systemctl --user stop polylogued.service

# 2. Snapshot the GC generation state to capture the age-floor gate's
#    high-water mark at the time (GC has no lease state — see
#    docs/internals.md "GC concurrency model").
polylogue ops diagnostics workload --json | jq '{gc: .gc_state}'

# 3. Identify the affected sessions.
polylogue ops doctor --schemas --blob-integrity --format json \
  | jq '.unreadable_blobs[]'

# 4. If you have a recent backup, restore just the blob store.
#    The blob store is content-addressed, so per-blob restore is
#    safe — the hash is the address.
restic restore latest --target / --include /path/to/archive_root/blob

# 5. If the blob is gone for good, the session referencing it
#    cannot be exported. Tombstone it so it stops blocking exports
#    and import from the original source if available:
polylogue ops reset --session <conv_id>
polylogue import <path-to-source>

# 6. After recovery, GC the orphan references that point at the
#    now-missing blobs.
polylogue ops maintenance run --target orphaned_blobs

# 7. Restart the daemon.
systemctl --user start polylogued.service
```

If the corruption is the result of a known GC race (PR
[#1002](https://github.com/Sinity/polylogue/pull/1002) closed the
primary one, but [#818](https://github.com/Sinity/polylogue/issues/818)
tracks remaining classes), attach the lease/GC probe snapshot from
step 2 to that issue so the GC pass that mis-classified the blob can
be reproduced.
