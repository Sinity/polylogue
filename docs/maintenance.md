# Maintenance

This guide is for operators choosing between the read-only inspection
verbs under `polylogue ops maintenance`, the remaining guarded
durable-evidence verbs, `polylogue ops reset`, and "do nothing — the daemon
will catch up." It also collects runbook recipes for the most common
operational incidents. Derived state (`index.db`, FTS, insights,
embeddings) is rebuilt only by ordinary daemon convergence; there is no
manual rebuild or repair verb.

There is no supported archive-clone operation that rebinds all durable-train
authority state. A copied `.maintenance-state/durable-change-trains/.bootstrap`
and released train manifests remain bound to the original root and durable-tier
inode identities. Keep a copied archive offline rather than
starting it as a rehearsal. For a fresh archive, use the fresh-start path and
ingest only declared external source files; do not carry over train state.

## Durable schema changes

Durable tiers (`source.db`, `user.db`, `audit.db`) evolve only by numbered
migrations under `storage/sqlite/migrations/{source,user,audit}/`, each with
its `NNN.train.json` change-train sidecar. A fresh archive is created at every
tier's current version when the daemon first opens its root, and bootstrap
creates all durable tiers together under one pending intent.

When `polylogued run` opens an archive whose durable tier stands below the
version this runtime declares, it applies the pending trains before it serves
anything, under the same exclusive archive ownership
(`polylogue/daemon/durable_migrations.py`). An additive step
(`-- migration-safety: additive-no-backup`) runs directly. A step that changes
existing data runs only behind a backup the daemon takes under
`.maintenance-state/pre-migration-backups/` and scratch-verifies; the train
binds that backup's authenticated receipt to the exact pre-apply bytes, and a
failed backup refuses startup. There is no command that initializes an
archive, applies a migration, or recreates a missing durable tier. A durable
tier newer than this runtime, or one with no declared route, is refused with a
typed `SchemaSkew`; a lost durable tier is refused by name and never recreated.

Authenticated source maintenance writes a typed refresh receipt that binds its
predecessor authority and the exact durable-train manifest hashes before and
after the refresh. Successive refreshes therefore validate as one unbranched
transition chain ending at the exact current manifest; matching only the
current source hash or archive identity is not authority.

For the conceptual model behind derived insights and the FTS / blob
substrate, see [architecture.md](architecture.md) and
[internals.md](internals.md). For daemon ownership of the inline
maintenance loop, see [daemon.md](daemon.md).

## Ownership

`polylogued` owns every archive write. Its convergence loops drain raw
materialization, FTS, embeddings, and derived read models in bounded
batches from durable source evidence; a stale or missing derived row is
converged, never repaired by hand. The `ops maintenance` verbs below are
read-only inspection, or guarded, receipted recovery of durable evidence
that convergence cannot re-derive (blob quarantine, raw-authority ledger
recovery).
A WAL checkpoint is not a maintenance operation: ingest runs bounded passive
checkpoints after commits, the daemon runs periodic truncate checkpoints, and
status/metrics report WAL pressure.

The order of preference is: **do nothing → daemon → guarded recovery →
reset**. Reset is the only one that destroys primary data.

`polylogue ops reset` runs in the daemon, which holds every archive tier
database open. A reset that names `index.db` or `ops.db` (`--index`,
`--database`, `--all`) is therefore staged: the daemon records the authorized
plan and deletes nothing, and the next `polylogued` start deletes the whole
plan, including its other targets, before it opens any tier. Restart the
daemon to apply it; bootstrap then recreates the deleted tiers and the empty
index cold-builds from `source.db`. A reset only ever deletes `index.db` and
`ops.db`: `source.db`, `user.db` and `audit.db` are durable, an established
archive missing one refuses to open, and `embeddings.db` holds purchased
vectors. A target that is or holds any of those is refused with
`reset_unresettable_archive_tier` before any audit row. To start over, move
the whole archive root aside (see "Fresh-start archive after a reset ruling").

## Subcommands

### Blob-reference integrity — classification only

`polylogue ops maintenance blob-reference-debt` is read-only: it classifies
any referenced blob missing from the store. There is no repair command. No
current write path produces this debt -- raw-deleting routes remove the
`blob_refs` row with its raw row, publication reserves bytes before the
referencing row commits, and GC reclaims only unreferenced hashes -- so a
non-zero count is a producer defect to fix at its source, and the daemon's
expensive health check reports it as an error.

### `polylogue ops maintenance verify-archive` — coherence gate

Read-only. Runs a fixed registry of independent checks over the whole
archive and reports each as `ok`/`warning`/`error`/`skip` plus evidence
numbers, never just a boolean. This is the repeatable substitute for the
manual checklist an operator used to run by hand after an index reset
and reconvergence or a full restore — "does the archive prove its own restore?"

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
| `source-index-coverage` | Every raw logical head is materialized, has an explicit terminal disposition, or is quarantined, and every index session's `raw_id` still resolves to a real raw row (orphans). The raw source population, not the derived census ledger, defines the coverage universe. |
| `source-conservation` | Every acquired source item (each `raw_sessions` row, hook event, history sidecar) is materialized or carries a typed exclusion citing its rule (revision superseded, byte-duplicate receipt, parse failure, validation rejection, declared non-session artifact kind, decode failure, census verdict, pending); a raw row whose source file no longer exists on disk is `source_missing` when its raw payload bytes are still retained and `source_lost` when they are not. A raw acquired from inside an export bundle records an `archive!member` or `archive:member` coordinate: the on-disk probe resolves it to its container and requires the member to be present in it, so a bundle member is neither reported lost while its archive is on disk nor conserved by container existence alone. A demonstrably non-ZIP container has no retained member inventory; missing members use the retained-byte distinction above. Permission and I/O faults are blocking `source_unavailable` evidence, retried by a new audit, and never prove retention or loss. Two rules cite another owner's durable ledger: `authority_blocked_head` (warning) is a raw an unresolved `raw_authority_blockers` row names as the accepted revision head while the index materialized a different raw of the same logical source, and `quarantined_cohort_unmaterialized` (blocking) is a raw whose `raw_session_memberships` rows are all quarantined with no revision of the logical source indexed at all. Reverse: every session traces to a raw row that is not a declared non-session artifact (phantom sessions, polylogue-b508, are reported and never deleted), and every message, block, and attachment ref traces to its owner. An attachment with no ref splits on `ref_count` and the owner gap its session's writer recorded in `attachment_owner_gaps`: `attachment_unowned` (ref_count 0, owner ambiguous or never linked by the provider) is explained, `attachment_owner_missing` (the provider named a message the index does not hold) blocks, and `attachment_unreferenced` (non-zero ref_count) blocks. Unexplained, unclassified, lost-source, quarantined-cohort, orphan, and phantom terms block; pending and authority-blocked are warnings. The acceptance instrument for a rebuilt archive: zero blocking terms. |
| `reasoning-conservation` | Every reasoning witness inside the acquired bytes of a materialized coding-origin raw reaches a thinking block of the exact session and message that carries it. Witnesses are selected structurally from the payload -- Claude Code `thinking` segments (text-bearing, signature-only, empty) and standalone Codex `reasoning` records (summary-bearing, content-bearing, opaque) -- never from the parser, the index, or any identity in the file. Denominators and outcomes are reported per origin and per variant: material surviving under a non-thinking kind is `reasoning_kind_collapsed`, an absent carrier or lost material blocks, a declared origin that contributed no readable evidence is `origin_evidence_absent`, and a bounded run that truncated is `scan_truncated`. One origin's conserved witnesses can never stand in for another's lost ones. Reads every selected blob of the two largest origins, so it is declared on the cross-tier candidate route only, never the routine live route. |
| `fts-parity` | `messages_fts` exactly covers its source `blocks` rows, archive-wide, with the worst-offending sessions surfaced by name. |
| `lineage-sanity` | `session_links.resolved_dst_session_id` and `branch_point_message_id` resolve to real sessions/messages (the latter is deliberately not a foreign key — see the data-model docs). |
| `planner-stats` | `sqlite_stat1` covers `blocks`/`messages`/`session_links`/`action_pairs` (warn-level: a fresh generation without `ANALYZE` picks pathological query plans, polylogue-l3tk class). |

The configured frontier is a current source projection, not a cross-run
registry of optional child roots. It omits hook carrier or pending children
that are currently absent, while requiring the primary hook spool root to
be resolvable. Cold-build baselines retain accepted revisions through their
pending receipt and candidate generation, and source-conservation checks
durable acquired Source evidence independently; those are the owners of
previously observed item obligations.

Canonical provider roots come from resolved source paths as well as the
runtime source list, so an unreadable configured root remains in the frontier
as `UNAVAILABLE` and blocks source-conservation. A definitely absent optional
canonical path that has not entered the runtime source list is omitted; a root
already admitted by that list remains in the denominator if it disappears
before observation. The CLI builds the frontier only for a full verification
or when `source-conservation` is selected. Frontier members are observed into
a private disk-backed SQLite spool, and source-conservation joins its paged
members to `raw_sessions` through a file-backed TEMP projection while the
archive tiers remain read-only. The CLI closes the frontier after the check;
the connection owns and removes the TEMP projection.

Exit code is non-zero when any check reports `error` (or, with `--strict`,
`warning`). A single check's failure — including a tier database being
temporarily busy under a concurrent rebuild — never aborts the rest; each
check independently reports its own outcome.

---

## Runbooks

### Cold build source baseline and archive handoff

A daemon cold build captures `source-baseline.json` inside its inactive index generation before intake starts. It uses the effective typed `WatchSource` roots and the same discovery walk as file intake, before cursor filtering. The receipt records accepted revisions, excluded paths, aliases, and discovery faults. Promotion refuses an unresolved fault or an accepted revision absent from durable `source.db`. The daemon also refuses it, with settlement reason `active_coverage_incomplete` (`polylogue/operations/cold_build_coverage.py`), while the candidate lacks any session the active generation serves from a raw that `source.db` still retains, such as a manual import or a transcript deleted from its source root; the candidate stays inactive and the active generation keeps serving. Files arriving after capture are outside that generation's denominator.

The generation receipt is backed by a private pending receipt at `.maintenance-state/production-source-baseline/pending.json`. A retry unions newly discovered accepted revisions with earlier accepted revisions before allocating another generation, so a file deleted after an interrupted attempt remains an unmet obligation. Unresolved discovery faults also carry forward until the same coordinate is observed as accepted, as an independently accepted alias, or as a recovered watched root. Successful promotion clears the pending receipt. ZIP obligations use the exact per-record hashes and source indexes emitted by production acquisition, including split conversation members and declared artifacts. A file that grew after it was baselined is satisfied only by retained bytes that begin with exactly the baselined bytes: a larger whole-file capture of the same path, or a whole-file capture followed by byte-proven, contiguous append tails; verification streams and hashes that prefix. A staged SQLite import is baselined at the original database path its provenance sidecar names, which is the path acquisition retains it under.

The fresh-start reset preserves previous Polylogue tiers and the blob store intact solely as salvage evidence. It does not publish a source-baseline receipt from those tiers or use their databases for rollback or readback. Explicitly declared original source files remain inputs, including hook carriers, browser capture payloads and staged exports whose directories happen to be under the old archive root. Preserve those originals, copy their declared layouts into the new intake roots, and run any hook compaction only on the copies. Copying source bytes does not import an old tier, receipt, capture registry or generation authority. Normal cold-build baseline and promotion checks apply to the declared input set.

The runbooks below assume:

- You have a recent local backup (`polylogue ops backup` — see
  [daemon.md § Operator-Owned Tasks](daemon.md#operator-owned-tasks)).
- You can stop the daemon if a runbook requires exclusive write
  access (`systemctl --user stop polylogued.service`).
- You ran `polylogue ops doctor` first to confirm the symptom matches the
  runbook.

### Recovering from a stale FTS index

**Symptoms.** Search returns fewer hits than expected for known
strings. `polylogue ops doctor` reports a `messages_fts` discrepancy.
`polylogue ops diagnostics workload`
shows non-empty `fts_trigger_state.missing` or `regressed` triggers.

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
Stop the daemon, restore or rebuild the affected index tier, and open an issue
with the probe output attached.

### Reading the drift-magnitude trend

`polylogue ops diagnostics workload` reports the *current* FTS freshness
state as a boolean. The `fts_drift_samples` and `schema_drift_samples`
ledgers in `ops.db` record a time series of the same counters on every
convergence pass; `polylogue ops diagnostics drift` is their operator-facing
read (polylogue-g31s).

```bash
# FTS drift magnitude per surface plus schema-drift classifications per origin.
polylogue ops diagnostics drift --since-hours 168

# Machine-readable, including the per-surface magnitude series.
polylogue ops diagnostics drift --format json | jq '.fts[].magnitudes'
```

A magnitude series that is flat and non-zero across many passes means
convergence is running but not closing the gap; a series that rises means a
writer is outpacing the index. Both are structural, not transient.

### Inspecting the raw-authority frontier

`polylogue ops maintenance raw-authority-frontier` asks the daemon to run the
`maintenance.raw-authority-frontier` operation, which classifies every accepted
frontier head and every terminal supersession in one pass. The census publishes
durable blockers into `source.db`, so it runs under the daemon's writer and the
command refuses when no daemon serves the archive:

```bash
polylogue ops maintenance raw-authority-frontier --output-format json
```

The pass is not recorded. Its `pass_id` is a content address over the
inspected inventory, so two passes that observe the same frontier name the
same pass and neither writes a row (polylogue-6kur ruling 2026-09-15 retired
the per-pass census ledger, which re-recorded the entire pending plan set on
every inspection). Each returned item carries its own state, reason, evidence
digest, input raw IDs and preconditions, so a caller needs no separate detail
handle to see a component's evidence.

The pass classifies; it never schedules a remedy. `proven_current` and
`superseded` are terminal facts. `missing_bytes_reacquire` is the one
retryable state, and what retries it is ordinary acquisition.
`unresolved_provenance` and `corrupt` are typed permanent refusals: only new
bytes or new evidence changes them. polylogue-6kur deleted the actuator
taxonomy, the executability gate and the operator-judgment promotion loop that
used to dress these states as queued repairs -- every one of those named a
remedy no code in the tree could run.

What the pass *does* publish is durable: every blocking item gets a
`raw_authority_blockers` row, and an obligation that current evidence
disproves is tombstoned in the same transaction. A blocking item's
`evidence_ref` is the blocker ID that now carries its plan snapshot.

Parser census itself advances through a bounded number of authority components
per pass. Uncensused components remain pending and a later daemon tick resumes
from the per-raw current-parser receipts.

Raw-authority inspection is the narrow exception to the generic read-only
preview rule above: it may durably record source-tier parser/census
observations so a moved-path component has one crash-safe identity across
inspections. It never selects or applies an index replay plan.

A stale precondition or incomplete application receipt creates a durable,
fail-closed blocker. List unresolved blockers before resolving one -- this is
the read-only discovery surface for an operator who does not already know an
exact `--blocker-id` (the alternative is an ad hoc script against the live
archive):

```bash
polylogue ops maintenance raw-authority-blockers --output-format json
```

Each row's `kind` describes how the resolver reads its stored snapshot, not a
different effect: `frontier_obligation` carries the current frontier plan
shape, and `stale_plan` is a durable row whose snapshot predates it, which the
resolver re-derives from live evidence instead of trusting. The listing is
bounded to `--limit` (1-500, default 100) per
call; if the response's `truncated` field is `true`, pass
`--offset <next_offset>` to read the next page. After inspecting the
blocker's own plan snapshot and current evidence, explicitly resolve it with
a recorded rationale:

```bash
polylogue ops maintenance raw-authority-blocker-resolve \
  --blocker-id 'raw-authority-blocker:...' \
  --reason 'reviewed current source/index evidence; replan from this state' \
  --yes
```

Resolution applies no remedy. It stores the replacement plan witness in the
resolution receipt and tombstones the blocked state; whatever the obligation
named is discharged by ordinary acquisition or derivation, or the next census
pass republishes it. Both commands route through
`OperationExecutor`/`BlockerResolveActuator` (polylogue-t46.9 phase 3):
PREPARE previews the exact blocker target and EXECUTE requires a
confirm-flag-strength authorization bound to that plan's hash, refusing
(`preview_stale`) if the blocker was concurrently resolved between preview
and confirm.

### Raw-authority frontier ownership

Nothing applies a frontier plan. `polylogue ops maintenance
raw-authority-frontier` has the daemon inspect and publish obligations; it has no plan
selector and no apply option, and what it records is the durable blocker set,
not a census row. Raw materialization itself is owned by the canonical raw
derivation (`polylogue/storage/derived/raw.py`) under the daemon writer
coordinator, which replays one authority component per call from ordinary
durable evidence.

### Interrupted mutations

An executor-routed mutation that is interrupted before audit finalization
leaves a nonterminal `operation_runs` row. The daemon resolves these at
startup under its own writer lease (`polylogued run` ->
`recover_interrupted_operations`). A reset of archive tier files resolves
earlier, at `apply_staged_archive_resets`, before any tier opens; anywhere
else it stays pending. A later mutation request first resolves every dead run
whose family its process has loaded; it refuses while dead work it cannot
route overlaps its targets or would delete archive files.

Every mutation family declares a recovery route
(`polylogue/operations/mutation_replay.py`). Most re-apply the recorded plan:
their `apply` converges from any state an interrupted apply of the same plan
can leave. The annotation batch import commits in one transaction, so its
batch row shows whether it landed. The outcome is terminal and never `unknown`:
`recovered_complete`, `recovered_absent`, `recovery_not_replayable` (a family
or version this runtime no longer declares, or an ingest whose request was
stopped), or `recovery_replay_failed` with the error. None of them blocks a
later mutation of the same targets, and there is no operator adjudication
route.

An interrupted ingest whose request was never stopped is not terminalized by
generic recovery. The daemon's ingest owner (the operation runtime behind the
API) claims each such run under a new attempt when it starts and drives the
accepted generation from its retained manifest through enumeration,
materialization, profile convergence and finalization, the same phases a
fresh request runs. The original input path is not read again. The original
request id then reads `running` and, once the owner finalizes, the terminal
`completed` or `degraded` receipt. A generation the owner cannot drive (its
enumeration decoder is gone, its retained rows are damaged) ends `failed`; an
owner shutdown leaves the run for the next start.

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
`threads`) lagging behind `sessions`.

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

# 3c. Durable tier: no durable migration is declared, so a durable-tier
#     mismatch means a different runtime wrote it. Roll the binary back.

# 4. Restart and verify.
systemctl --user start polylogued.service
polylogue ops doctor
```

The daemon holds the stable `<archive-root>/.archive-ownership.lock` archive
lease. `daemon.pid` is process metadata only and is never reclaimed by
unlinking it as a lock. Never hand-edit a tier. If release notes provide
neither an additive durable migration nor a derived-tier rebuild plan, keep
the daemon stopped and roll back the binary.

### Fresh-start archive after a reset ruling

The reset creates a new empty archive with all six tier files at `PRAGMA user_version=1`. Move previous tiers and the blob store aside intact solely as salvage evidence. Do not migrate, import, copy forward, qualify for rollback, or read back their databases as part of this reset. Preserve and copy explicitly declared original inputs before intake, including archive-local source spools. Let the production daemon acquire those copies, converge derived tiers and verify the new archive with `polylogue ops maintenance verify-archive`.

General durable migration and runtime recovery policy still applies to archives intentionally retained for operation. It does not provide a rollback route for the preserved pre-reset archive or alter the fresh-start procedure.

### Proving an archive is coherent after a rebuild or restore

**Symptoms.** None yet — this is the proactive check to run *before* symptoms
appear, immediately after any operation that replaces or promotes a whole
tier: a derived-tier rebuild (`polylogue ops reset --index`, then a daemon restart),
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
though source files are present. `polylogue ops status --json` reports
stale ingestion progress for the source. Daemon logs show repeated parse
errors for the same artifact id.

**Recovery.**

```bash
# 1. Identify the stuck source.
polylogue ops status --json | jq '.components[]? | select(.state != "ready")'

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

# 6. Restart the daemon. The daemon-owned blob-GC loop waits for the initial
#    watcher registration event, or proceeds after the daemon's 1800-second gate timeout,
#    before starting its periodic interval.

systemctl --user start polylogued.service

# 7. After watcher registration or timeout, the first bounded blob-GC pass
#    waits one 900-second interval. Each pass reclaims at most 200 blobs;
#    eligible leftovers are handled by later passes. Manual blob reclamation
#    is not a supported route, and reservation TTLs must not be inferred.

```

If the corruption is the result of a known GC race (PR
[#1002](https://github.com/Sinity/polylogue/pull/1002) closed the
primary one, but [#818](https://github.com/Sinity/polylogue/issues/818)
tracks remaining classes), attach the lease/GC probe snapshot from
step 2 to that issue so the GC pass that mis-classified the blob can
be reproduced.
