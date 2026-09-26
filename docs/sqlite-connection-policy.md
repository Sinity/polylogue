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

`write_lease.py` makes every declared write-mode factory take the lease.
`write_guard.py` makes that boundary *total*: while the daemon holds its
process-lifetime writer ownership it also installs a guard over
`sqlite3.connect`, so a writable open whose file name is one of the six tiers
asserts the lease before the connection exists. Roughly seventy production
sites open `sqlite3.connect` directly; without the guard, a writer that reaches
a tier that way contends through the busy timeout instead of being refused, and
the factory census cannot see it.

Read-only opens (`mode=ro`, `immutable=1`), in-memory databases and non-archive
files -- spill databases, provider caches, the Sinex database, export
destinations -- are untouched. Tier membership is decided by file name, not by
directory, so a generation build or a staging copy named `index.db` is also a
guarded open; the guard therefore asserts only that *a* lease is held and
leaves archive-root binding to the factories, which know which archive they
were asked for.

`declared_unguarded_write(reason)` is the single named bypass, for the
authorities that own an archive without a daemon: first-time bootstrap,
offline exclusive rebuild, migration behind its own backup, and test fixtures.
It is thread-local, so one bootstrap never unlocks a concurrent writer.

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
| `exclusive` | PASSIVE, RESTART, TRUNCATE | seal, shutdown, offline generation lifecycle, backup snapshot |

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

Counts over `polylogue/` at the head that introduced this document. The
inventory is review evidence: it records what is migrated and what remains, and
is deliberately not a source-text gate.

| Shape | Count |
| --- | --- |
| Declared read factory (`open_readonly_connection`, `open_profiled_connection`, `read_frame`) | 165 |
| Declared write factory (`open_connection`, `open_daemon_connection`, `open_isolated_write_connection`) | 60 |
| Hand-built `immutable=1` URI | 0 |
| Direct `mode=ro` open | 155 |
| Raw `PRAGMA busy_timeout` outside the policy module | 9 (8 files) |
| `PRAGMA wal_checkpoint` outside `wal_checkpoint.py` | 0 |

Migrated in this pass: `daemon/backup.py` (live-tier snapshot writer and the
pre-migration backup reader), `security/secret_scan.py`,
`security/excision.py`'s writer, `api/archive.py`'s two-tier audit seam (to a
pair of read frames), `operations/mutation_actuators.py`,
`operations/raw_authority_verdict_cache.py`,
`archive/query/source_freshness.py`, and every hand-built
`immutable=1` wrapper in `storage/blob_integrity.py`,
`storage/artifacts/inspection.py`, `storage/sqlite/migration_runner.py`,
`sources/sqlite_snapshot.py`, `sources/parsers/{codex_state,hermes_state,hermes_verification}.py`,
`maintenance/embedding_preservation.py` and
`operations/durable_change_train.py`.

The archive store's source-tier version probe, persistent read-only source
handle, raw-artifact and hook-event readers, and operation-debt probe now use
the named factory. They keep their previous schema-validation behavior so a
diagnostic read can still report a stale or absent tier through its caller.

The API, CLI status, `operations/archive_debt.py` and `daemon/similarity.py`
readers already used the declared factories and named classes.

Classified but not migrated, with the reason each keeps its own connection:

| Site | Classification |
| --- | --- |
| `sources/revision_backfill.py` spill connections | non-archive scratch database owned by one pass |
| `sinex/service.py` | an external Sinex database, not an archive tier |
| `storage/blob_publication.py`, `daemon/convergence_stages.py` | one-tier writers inside a held lease |
| `storage/sqlite/archive_tiers/{archive,user_write}.py` | the tier writers the profiles are applied *by* |
| `storage/embeddings/status_payload.py` | diagnostic status read over a possibly-absent tier |

The remaining direct `mode=ro` opens are concentrated in `storage/blob_gc.py`,
`storage/raw_authority.py`, `daemon/convergence_stages.py` and
`sources/live/`. They are one-shot
maintenance and reconciliation reads over a single tier; migrating them is a
mechanical follow-up, not a correctness gap, because none of them holds a frame
across a request boundary.
