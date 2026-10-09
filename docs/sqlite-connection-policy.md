# SQLite connection, WAL and read-frame policy

`polylogue/storage/sqlite/connection_profile.py` is the single owner of
connection profiles, named timeout classes, WAL checkpoint escalation, the
checkpoint hold budget and read-frame lifetime. `wal_checkpoint.py` executes
the checkpoint half of that policy and declares none of it.

These are the *mechanisms*. The guarantee each of them is chosen to meet --
what a process crash costs, what a power loss costs, what reconstructs a tier
afterwards, and where a caller may certify retention -- is
[Durability by tier](durability-by-tier.md).

## Named timeout classes

Callers select a role, never a lock-wait duration.

| Class | Lock wait | Role | Frame policy |
| --- | --- | --- | --- |
| `interactive-read` | 5 s | read | live generation, rebind after 30 s, cancellable |
| `background-read` | 30 s | read | live generation, rebind after 300 s, cancellable |
| `offline-bulk` (read) | 30 s | read | live generation, rebind after 300 s, cancellable |
| `publication` | 30 s | write | daemon writer, WAL, `synchronous=NORMAL` |
| `offline-bulk` (write) | 30 s | write | owned inactive generation, `journal_mode=MEMORY`, `EXCLUSIVE` |

`READ_PROFILES` and `WRITE_PROFILES` are separate maps: a single merged map
shadowed the offline-bulk read profile with the bulk-build writer.

`SEALED_READ_CONNECTION_PROFILE` is the only profile carrying SQLite's
`immutable=1`. `open_readonly_connection` selects it whenever a caller asks for
immutability and refuses immutability against any other live-generation
profile, so "immutable" can only ever mean a sealed generation and never "the
caller intends not to write".

A file is not sealed while a non-empty `-wal` or rollback journal sits beside
it: an `immutable=1` reader reads the main file alone and would skip that
committed state. `open_readonly_connection`, `attach_readonly_database` and
`open_sealed_staging_connection` refuse such a file with
`LiveGenerationImmutableError` (`immutable_over_live_state`). Freezing a
snapshot is an exclusive-boundary checkpoint that folds the WAL into the main
file, after which a sealed `ReadFrame` binds that file's generation identity.
A caller that must read a live tier's committed WAL state uses a live
`mode=ro` profile instead.

Historical continuity liveness classification uses the dedicated
`open_sealed_staging_connection` factory. It opens `mode=ro&immutable=1` with
the bounded offline cache/time profile and `temp_store=MEMORY`, but leaves
`query_only` off so classifier candidate tables can live in SQLite's private
TEMP schema. A fail-closed authorizer is installed before the connection is
returned: it permits only main/TEMP reads, the classifier's pure function
allowlist, transactions/savepoints, read-only schema/data PRAGMAs, and TEMP
table/index staging. Main-schema writes, ATTACH/DETACH, unsafe PRAGMA
assignments, virtual tables, triggers, extensions, and unlisted operations are
denied. This exception is not a general read-profile relaxation; ordinary
readers continue to require `query_only=ON`.

## The writable-open boundary

Archive writers admit their configured root before native construction through
`connection_profile.py` or their specialized destination owner. Bootstrap,
durable trains, embedding candidates and descriptor-bound generation checkpoints
retain their native opening policy and perform the same root admission.
Authenticated detached population requires its exact owned destination and root
lease before copying. The daemon arms lease enforcement; it does not replace
`sqlite3.connect` process-wide.

The existing layering gate rejects locally resolved writable tier opens without
an earlier explicit archive admission in their owner function. It tracks local
paths and lexical statement order, not control-flow dominance or arbitrary
runtime values. This positive regression check supplements the owning factory's
runtime root enforcement. Read-only, scratch and
external databases keep their own destination contracts.

The CLI uses one per-open residency and archive-ownership check whether a daemon
was present at entry or arrives later. Its process-wide residency interceptor
admits configured-archive writes only from that archive's daemon coordinator,
and admits separately owned scratch archives.
Its filename/URI classifier belongs to `maintenance/offline_guard.py`; it does
not grant a write lease.

## Checkpoint ownership

A process that runs the recurring coordinator claims it with
`arm_recurring_checkpoint_owner()`; `polylogued` does so for its whole
lifetime. Under an armed owner every writable connection sets
`wal_autocheckpoint = 0`, because an implicit autocheckpoint runs inside
whichever writer's commit crossed the page threshold and charges its time to
that writer's publication hold. Any other process keeps the bounded 10,000-page
default: a one-shot CLI or API writer has no recurring owner to defer to.

| Escalation | Modes attempted | Where it belongs |
| --- | --- | --- |
| `recurring` | PASSIVE | the daemon's 5-minute coordinator, and the cold-build pass boundary (`ArchiveStore.finish_active_cold_build`) — both against a live archive |
| `quiescent` | PASSIVE, RESTART | a declared quiescent boundary |
| `exclusive` | PASSIVE, RESTART, TRUNCATE | seal, shutdown, offline generation lifecycle |

A busy result retains the WAL and reports blockers; it never loops, retries or
escalates to outlast a live reader. Blocker collection walks `/proc`, so it is
opt-in and off for any interactive route. `blocking_read_frames` is the cheap
half of the same evidence and is always on: it names the read frames *in this
process* that held an open read transaction over the tier, with their declared
maximum age. In the daemon the `/proc` walk usually answers "polylogued", which
is not actionable; the frame list says which snapshot, how old, and against
what bound.

The recurring sweep covers all six archive tiers, including `audit.db`.

Checkpoint hold is budgeted separately from publication:
`CHECKPOINT_HOLD_BUDGET_S` (20 s) against `maintenance.wal_checkpoint` in
`daemon/write_coordinator.py`, below the 120 s general maintenance budget.

## Read frames

### What a snapshot actually costs

Two measurements on a synthetic WAL archive with `wal_autocheckpoint=0` (what
the armed recurring owner leaves every daemon writer) and a 512-byte row
payload.

40k rows written, then one PASSIVE:

| Reader shape | PASSIVE result |
| --- | --- |
| none | checkpointed 6079 of 6079 frames |
| idle `mode=ro` connection | checkpointed 6079 of 6079 frames |
| one lazily stepped cursor | checkpointed **0** of 6079 frames |

Eight bursts of 5k rows with one PASSIVE after each:

| Reader shape | WAL bytes, first burst → eighth |
| --- | --- |
| idle `mode=ro` connection | 2,962,312 → 2,978,792 (plateau) |
| one lazily stepped cursor | 2,962,312 → 23,776,552 (8.0x, monotonic) |

So an idle read-only connection is not a cost, and connection lifetime is not
the thing to bound. What pins WAL frames is an **open read transaction** — in
practice a cursor still being stepped — and while one is held the recurring
PASSIVE owner reclaims nothing and the WAL grows by the full concurrent write
volume. `sqlite3.Connection.in_transaction` cannot see this (it stays `False`
for a SELECT in autocommit while the cursor holds frames), so the frame has to
own the stepping in order to bound it.

### The bound

`ReadFrame` binds a read connection to a generation identity — the file's
`(st_dev, st_ino)`, which is what a generation-pointer swap moves and what stays
comparable between two connections — for the age its profile declares. Past that
age `frame.connection` raises `ReadFrameExpiredError` rather than serving a
reader that pins WAL frames indefinitely. `rebind()` reopens against the current
generation and starts a new incarnation; a sealed frame refuses, having nothing
to rebind to, and refuses too while a stream is in flight rather than closing
the connection under an in-flight cursor.

`ReadFrame.stream()` is the only supported way to hold a cursor open across
other work. It re-checks the declared age between rows, and on expiry closes the
cursor *before* `ReadFrameExpiredError` reaches the caller — so the typed
refusal ends the WAL pin instead of naming it and continuing to cause it.
Its SQLite progress handler also interrupts a single expensive row computation
when that computation itself crosses the frame age; the stream maps that
interrupt to the same typed expiry and closes its cursor.

A live-generation profile with `max_snapshot_age_s=None` is refused at
construction: that shape is exactly the unbounded snapshot the class exists to
prevent, and a sealed generation is the only thing that may be unbounded. A
caller whose work legitimately outlives its class default extends the bound
explicitly — `read_frame(..., max_snapshot_age_s=..., reason=...)`, where the
reason is required, travels into the expiry message and into the registry. There
is deliberately no spelling for "no bound".

`live_read_frames()` and `pinning_read_frames(path)` are the process-local
snapshot registry. Every open frame is in it; `pinning_read_frames` narrows to
the ones holding a `stream` cursor, which is the only state that provably pins.
The recurring checkpoint owner reads it to name what blocked it.

Content freshness inside one generation is a separate question, answered only by
the open connection: `PRAGMA data_version` is explicitly not meaningful across
connections, so `revalidate()` compares it against this connection's own
baseline and the generation identity against the file.

A `ReadContinuation` carries the anchor query proving its position, plus the
frame incarnation that produced it. `resume()` rebinds an expired frame, then
skips the anchor check only when nothing has moved at all — same generation,
same incarnation, no observed commit. Otherwise it re-proves the anchor and
raises `StaleContinuationError` if the position no longer holds; it never
advances or rewinds a position to make one fit.

## Connection inventory

Counts over `polylogue/` at the head that last revised this section, excluding
the policy module itself. The inventory is review evidence: it records what is
migrated and what remains, and is deliberately not a source-text gate.

| Shape | Count |
| --- | --- |
| Declared read factory call (`open_readonly_connection`, `open_profiled_connection`, `read_frame`, `one_shot_diagnostic_read`) | 286 |
| Declared write factory call (`open_connection`, `open_daemon_connection`, `open_isolated_write_connection`, `open_source_tier_write_connection`) | 78 |
| Direct `sqlite3.connect(` | 145 |
| Direct `mode=ro` open or URI | 55 |
| Hand-built `immutable=1` URI | 5 (two files, both classified below) |
| Raw `PRAGMA busy_timeout` outside the policy module | 5 (four files, all classified below) |
| `PRAGMA wal_checkpoint` outside `wal_checkpoint.py` | 0 |

The `aiosqlite` read pool in `storage/sqlite/async_sqlite.py` applies the read
profile's pragmas, attaches sibling tiers through `mode=ro` URIs and installs
the same read authorizer as the synchronous factory, so it is a profiled reader
even though it cannot call the synchronous factory.

Every other direct open is one of these roles, not a profiled archive reader:

| Role | Where | Why it keeps its own connection |
| --- | --- | --- |
| Private scratch, spool or spill database owned by one pass | `pipeline/ids.py`, `operations/{daemon_ingest,ingest_inputs}.py`, `sources/{prepared_jsonl,prepared_message_sink,tool_outcomes,assembly_claude_code}.py`, `sources/parsers/claude/stream_scratch.py`, `sources/live/tool_result_sidecars.py`, `storage/sqlite/archive_tiers/{write,revision_governance}.py`, `storage/sqlite/session_shard.py`, `sources/revision_backfill.py`, `archive/session_revision_membership.py` | not an archive tier; its lifetime is the owning pass |
| In-memory database | schema identity, disposition, inventory and manifest builders; durable change train; migration runner | no file |
| External or provider database | `sinex/service.py`, `sources/{assembly_codex,sqlite_export,sqlite_snapshot}.py`, `schemas/source_cache.py`, `schemas/source_inference.py`, `browser_capture/capture_jobs.py` | not an archive tier; opened under that source's own contract |
| Tier writer under a held lease | `storage/{blob_integrity,blob_publication,raw_reconciler}.py` (index exclusion lock), `analysis/claude_workflow_materializer.py`, `sources/live/hook_paste_enrichment.py`, `operations/{route_observation,mutation_actuators}.py`, `storage/sqlite/archive_tiers/{archive,user_write,bootstrap}.py`, `storage/sqlite/durable_change_train.py` | root-bound admission before native construction |
| Sealed or anchored copy | `storage/embeddings/generations.py` (unpublished generation, loads sqlite-vec), `storage/sqlite/audit_leaf.py`, `storage/index_generation.py` (descriptor-bound exclusive checkpoint) | the file is proven immutable or exclusively owned before the open |
| Demo, scenario and schema-generation tooling | `demo/`, `scenarios/corpus.py`, `schemas/generation/`, `operations/canonical_archive_ingest.py` (one-shot ownership probe) | development tooling or a probe outside the archive read path |
| Archive reader not yet migrated | `sources/live/{batch,watcher,batch_observability}.py`, `storage/raw_retention.py` | live-intake and retention readers; migrate with those modules' next owner change |
