# SQLite connection, WAL and read-frame policy

`polylogue/storage/sqlite/connection_profile.py` is the single owner of
connection profiles, named timeout classes, WAL checkpoint escalation, the
checkpoint hold budget and read-frame lifetime. `wal_checkpoint.py` and
`read_frame.py` execute that policy; neither declares any of it.

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
| `recurring` | PASSIVE | the daemon's 5-minute coordinator, against a live archive |
| `quiescent` | PASSIVE, RESTART | a declared quiescent boundary |
| `exclusive` | PASSIVE, RESTART, TRUNCATE | seal, shutdown, offline generation lifecycle, backup snapshot |

A busy result retains the WAL and reports blockers; it never loops, retries or
escalates to outlast a live reader. Blocker collection walks `/proc`, so it is
opt-in and off for any interactive route.

Checkpoint hold is budgeted separately from publication:
`CHECKPOINT_HOLD_BUDGET_S` (20 s) against `maintenance.wal_checkpoint` in
`daemon/write_coordinator.py`, below the 120 s general maintenance budget.

## Read frames

`ReadFrame` binds a read connection to a generation identity
(`(st_dev, st_ino, PRAGMA data_version)`) for the age its profile declares.
Past that age `frame.connection` raises `ReadFrameExpiredError` rather than
serving a reader that pins WAL frames indefinitely. `rebind()` reopens against
the current generation; a sealed frame refuses, having nothing to rebind to.

A `ReadContinuation` carries the anchor query proving its position. `resume()`
rebinds an expired frame, then either confirms the continuation is still
equivalent or raises `StaleContinuationError` — it never advances or rewinds a
position to make one fit.

## Connection inventory

Counts over `polylogue/` at the head that introduced this document. The
inventory is review evidence: it records what is migrated and what remains, and
is deliberately not a source-text gate.

| Shape | Count |
| --- | --- |
| Declared read factory (`open_readonly_connection`, `open_profiled_connection`, `read_frame`) | 164 |
| Declared write factory (`open_connection`, `open_daemon_connection`, `open_isolated_write_connection`) | 60 |
| Hand-built `immutable=1` URI | 0 |
| Direct `mode=ro` open | 156 |
| Raw `PRAGMA busy_timeout` outside the policy module | 9 |
| `PRAGMA wal_checkpoint` outside `wal_checkpoint.py` | 0 |

Migrated in this pass: `daemon/backup.py` (live-tier snapshot writer and the
pre-migration backup reader), `security/secret_scan.py`,
`security/excision.py`'s writer, `cli/read_views/streaming_markdown.py` (to a
read frame), `archive/query/source_freshness.py`, and every hand-built
`immutable=1` wrapper in `storage/blob_integrity.py`,
`storage/artifacts/inspection.py`, `storage/sqlite/migration_runner.py`,
`sources/sqlite_snapshot.py`, `sources/parsers/{codex_state,hermes_state,hermes_verification}.py`,
`maintenance/embedding_preservation.py`, `operations/durable_change_train.py`,
`operations/archive_root_relocation.py` and
`operations/historical_source_continuity_recovery.py`.

The API, CLI status, `operations/archive_debt.py` and `daemon/similarity.py`
readers already used the declared factories and named classes.

Classified but not migrated, with the reason each keeps its own connection:

| Site | Classification |
| --- | --- |
| `sources/revision_backfill.py` spill connections | non-archive scratch database owned by one pass |
| `sinex/service.py` | an external Sinex database, not an archive tier |
| `storage/blob_publication.py`, `storage/raw_convergence.py`, `daemon/convergence_stages.py` | one-tier writers inside a held lease |
| `storage/sqlite/archive_tiers/{archive,user_write}.py` | the tier writers the profiles are applied *by* |
| `storage/embeddings/status_payload.py` | diagnostic status read over a possibly-absent tier |

The remaining direct `mode=ro` opens are concentrated in `storage/blob_gc.py`,
`storage/raw_authority.py`, `storage/raw_convergence.py`,
`daemon/convergence_stages.py` and `sources/live/`. They are one-shot
maintenance and reconciliation reads over a single tier; migrating them is a
mechanical follow-up, not a correctness gap, because none of them holds a frame
across a request boundary.
