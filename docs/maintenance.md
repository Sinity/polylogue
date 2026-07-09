# Maintenance

This guide is for operators choosing between
`polylogue ops maintenance preview`, `polylogue ops maintenance plan`,
`polylogue ops maintenance run`, `polylogue ops reset`, and "do nothing — the
daemon will catch up." It also collects runbook recipes for the most
common operational incidents.

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
| `archive_cleanup` | cleanup | **yes** | `orphaned_messages`, `orphaned_content_blocks`, `empty_sessions`, `orphaned_attachments`, `orphaned_blobs` |
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
  `plan` is a dry-run summary; `run` is the only mutating verb.
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

### `polylogue ops maintenance run` — execute

Runs the resolved targets. Per-target failures are isolated as
`FailureSample` entries; one failing target does not abort the rest.
Use `--dry-run` to combine the safety of `plan` with the full
execution-path code, or pass `--operation-id <uuid>` together with
`--resume` to pick up an interrupted operation.

```bash
polylogue ops maintenance run --dry-run
polylogue ops maintenance run --target session_insights --output-format json
```

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

**Root cause.** FTS5 uses `content='messages'` content-rowid
triggers. Bulk operations suspend those triggers for speed and
rebuild the index afterward. A SIGKILL during the suspension window
leaves FTS out of sync. The daemon startup check and FTS convergence
loop normally restore the triggers, but a long-running operation
that suspends them and dies without re-enabling them will leave the
index stale across daemon restarts.

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
Polylogue uses fresh-first schema versioning, not in-place upgrade chains
(see [internals.md § Schema Versioning
Model](internals.md#schema-versioning-model)) — there is no
auto-downgrade.

**Root cause.** A new release advanced `SCHEMA_VERSION` and the
database is on the previous version. There is no reverse in-place upgrade.

**Recovery.**

```bash
# 1. Confirm the version mismatch.
polylogue --version
sqlite3 ~/.local/share/polylogue/index.db "PRAGMA user_version;"

# 2. STOP the daemon to release exclusive locks.
systemctl --user stop polylogued.service

# 3. Decide: roll the code back, or rebuild the changed tier forward.
#    There is no third option — fresh-first schema versioning is
#    explicit about rejecting mismatched databases.

# 3a. Code rollback (preferred when a release just went out and you
#     have not yet relied on any new feature):
#     install the previous polylogue version, leave the database
#     alone, restart the daemon.

# 3b. Forward rebuild: keep the source/user/embedding tiers safe,
#     move the mismatched index database aside, and re-ingest/rederive
#     the rebuildable index with the new polylogue binary.
cp ~/.local/share/polylogue/index.db /tmp/index-before-rebuild.db
# ...run the documented re-ingest/rederive flow for the release, verify
# it opens cleanly with the new polylogue binary, then restart production.

# 4. Restart and verify.
systemctl --user start polylogued.service
polylogue ops doctor
```

Polylogue does not maintain an in-place upgrade chain for ordinary schema
bumps. If the release notes do not provide an explicit, reviewed
transition, **do not** hand-edit the schema. Move the mismatched tier
aside, protect `source.db` and `user.db`, and rebuild from source.

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
