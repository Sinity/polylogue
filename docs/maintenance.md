# Maintenance

This guide is for operators choosing between the read-only inspection
verbs under `polylogue ops maintenance`, the guarded durable-evidence
recovery verbs, `polylogue ops reset`, and "do nothing — the daemon will
catch up." It also collects runbook recipes for the most common
operational incidents. Derived state (`index.db`, FTS, insights,
embeddings) is rebuilt only by ordinary daemon convergence; there is no
manual rebuild or repair verb.

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

## Rehearsal clones and audit adoption

A FICLONE or copied archive has new source/user device-inode authority and must
fail closed on the existing audit-adoption receipt. For a rehearsal, create the
clone through the authorized established-archive adoption route, using a
verified `full_evidence` backup and
`migrate-tier audit --adopt-established-audit`; do not copy an adoption receipt
into a new authority set. A normal inactive index-generation build and
promotion keeps source/user authority in place, so the adopted audit receipt
must remain valid across that promotion; the regression test
`test_audit_adoption_continuity_survives_index_generation_promotion` proves
that production path.

Authenticated source maintenance writes a typed refresh receipt that binds its
predecessor authority and the exact durable-train manifest hashes before and
after the refresh. Successive refreshes therefore validate as one unbranched
transition chain ending at the exact current manifest; matching only the
current source hash or archive identity is not authority.

### Deploying a package that owns the live durable schemas

Confirm that the package being deployed owns the live durable schemas
(`source.db`, `user.db`, `audit.db`). `index.db` may be behind: the daemon
rebuilds that derived tier from source through ordinary convergence.

For a safe deployment recovery, first choose the exact target package commit.
With the daemon stopped, create a fresh verified full-evidence backup. If the
preflight reports that a newly introduced durable tier is absent, initialize
only that absent file through the archive ownership gate. This recovery path is
allowed only for a completely unadopted private archive directory. Any sibling
durable tier, active-index pointer, or durable change-train marker proves that
the archive already has an identity, so the command refuses to create the
missing file and leaves it absent:

```bash
polylogue ops maintenance migrate-tier audit --initialize-missing --output-format json
```

The flag builds the canonical database in memory, writes it into an anonymous
inode, and requires filesystem support for `O_TMPFILE`. If the filesystem does
not support anonymous temporary files, the command fails closed and leaves the
tier absent. It fsyncs the image, publishes it with a no-replace hard link, then
fsyncs the directory. It refuses any existing target including one created
concurrently, and never replaces durable data.

If publication fails after the file becomes visible, JSON output carries a
`durable_recovery` object. A state of `uncertain` means the command preserved a
visible tier because it could not prove a pathname still names its inode.
Inspect the reported target and remove it manually before retrying.

For each existing tier that the selected package reports behind, run its numbered
migration with the verified full-evidence backup manifest:

```bash
polylogue ops maintenance migrate-tier source --backup-manifest /path/to/verified-full-backup/manifest.json --output-format json
polylogue ops maintenance migrate-tier user --backup-manifest /path/to/verified-full-backup/manifest.json --output-format json
polylogue ops maintenance migrate-tier audit --backup-manifest /path/to/verified-full-backup/manifest.json --output-format json
```

Deploy that exact package after every required durable migration, then
start the daemon; ordinary convergence rebuilds `index.db` from `source.db`.
`polylogue ops status` must show no durable-tier mismatch after the deploy.

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
that convergence cannot re-derive (blob quarantine, durable-tier
migrations, raw-authority ledger recovery).
A WAL checkpoint is not a maintenance operation: ingest runs bounded passive
checkpoints after commits, the daemon runs periodic truncate checkpoints, and
status/metrics report WAL pressure.

The order of preference is: **do nothing → daemon → guarded recovery →
reset**. Reset is the only one that destroys primary data.

## Subcommands

### `polylogue ops maintenance blob-disposition` — physical namespace disposition

One-time transition tooling for the blob-store maneuver. `plan` is read-only:
it walks the complete physical namespace and gives every object exactly one
disposition proven against a configured source — `source_present`,
`superseded_prefix`, `restore_required`, `unreferenced`, or `unresolved`. A
plan is acceptable at zero unresolved members with every non-blob namespace
entry explained, and its digest binds the archive identity, the namespace, the
denominators, and every member outcome.

`unreferenced` is the terminal outcome for an object no durable relation
names. A blob is published before the row that owns it, and reference-dropping
repairs strand objects by design, so an unnamed object is daemon GC's to
collect: this plan records it, never removes it, and never blocks on it.

The plan splits the redundant population two ways. `reclaimable_bytes` counts
the `source_present` and `superseded_prefix` members no durable row
references; `retained_by_reference_bytes` counts the ones a durable reference
still names. Only the first half can go, whatever the second half's proofs say.

```bash
polylogue ops maintenance blob-disposition plan \
  --archive-root /path/to/archive \
  --output /path/to/disposition-plan.json --output-format json
polylogue ops maintenance blob-disposition apply \
  --archive-root /path/to/archive \
  --plan /path/to/disposition-plan.json \
  --authorized-digest <digest of the reviewed plan> \
  --receipt /path/to/new/disposition-receipt.json --active
```

`restore` is the additive half on its own: it publishes sole-copy carriers
into their ordinary spool and deletes nothing.

`apply` does two things, in an order that cannot lose material. It restores
every `restore_required` carrier into its ordinary spool and reads the
published material back — the capture receiver publishes acquired bytes
verbatim, so a restored capture is verified byte-for-byte, while a hook event
is verified through the production read route that derives the fields the
spool file does not carry. One provider session keeps one capture artifact, so
a carrier arriving at an occupied artifact name is a revision the spool
converges: the newer or richer capture is published and the rest report
`restoration_superseded`, which is a completed restoration because the
artifact holding that identity carries the material. Only a malformed
envelope, a genuinely different session claiming the artifact name, or the
spool quota refuses a carrier. It then deletes, through the canonical blob-GC
seam, every member no durable row references: the same objects recurring GC
would take, plus the namespace's non-blob entries (a SQLite `-wal` or `-shm`
stranded beside a content-addressed object, whose bytes are that object's
identity, so the sidecar has no owner).

Deletion is bounded by unreferencedness, not by disposition. A referenced
object is never deleted whatever its proof; an `unresolved` orphan nothing
names goes, because being unreferenced is its disposition. The plan's
unresolved count gates nothing — what stays in the namespace after a pass is
what a durable row still references, and the receipt's `cohorts` block says
how much of that is still unexplained.

`apply` is a dry rehearsal without `--active`, and the rehearsal reports the
totals and counts its active twin would: it resolves every restoration
destination and evaluates the same admission rule, carrying what it would have
published so a second carrier of one identity converges in the rehearsal
exactly as it does in the run. A stale digest, a namespace that is not the
plan's, a drifted denominator, or an active archive writer refuses the run
before any effect. A carrier whose restoration did not complete keeps its blob
and reports `blocked`. A pass interrupted part-way is resumed by re-running
it: an object a previous pass deleted is reported `retained_absent`, and a
carrier an earlier pass already restored is proven at the spool it was
restored into rather than published a second time.

The receipt records, per member: hash, cohort, referenced flag, size, source
path, restoration outcome and spool path, and terminal outcome; plus `counts`,
`cohorts` (everything deleted and everything left, per disposition, with
bytes), `totals` (blob count and bytes in the namespace before and after),
`restorations` (every sole copy and where the spool now holds it), and
`reference_relations` — the durable relations a deleted member's
`referenced: false` was decided against.

Hook-event and browser-capture carriers are proven by the owning production
read route, not by bytes: acquisition derives fields the spool file does not
carry, so byte equality would misreport reproducible material as a sole copy.
The same reasoning governs the three provers for material no filesystem walk
can hash. A payload synthesized from a database row is reproduced by re-running
the production encoding over the live state database. An attachment extracted
from an account export is proven against a member of the retained export
archive, selected by the member's uncompressed size and decided by a fresh
SHA-256 — pass `--export-archive-root` to `plan`, and again to `restore` and
`apply`, which cannot revalidate a proof whose prover they were not given. A
whole-session carrier whose source was rewritten in place is proven by the
normalized session contribution both sides produce through the live detector,
parser and admission: the source proves the carrier when it reproduces every
stored session and no stored axis is missing from it.

A non-blob entry inside the namespace blocks acceptance until it carries
positive evidence of what it is. The one explained shape is a SQLite sidecar
named after a blob that is still present, written beside the object by a
reader that opened the stored database in place.

Deletion trigger: this command, both maintenance modules, and their tests are
removed with the terminal disposition receipt. The recurring liveness,
publication, GC, and spool-admission laws stay with their owners.


### Blob-reference integrity — preview and apply pairs

`polylogue ops maintenance blob-reference-debt` and
`blob-reference-recovery-plan` are read-only classification/planning
commands. The two commands that can actually mutate the archive each
follow the same preview/apply split: a dedicated
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
| `source-conservation` | Every acquired source item (each `raw_sessions` row, hook event, history sidecar) is materialized or carries a typed exclusion citing its rule (revision superseded, byte-duplicate receipt, parse failure, validation rejection, declared non-session artifact kind, decode failure, census verdict, pending); a parsed raw whose content-addressed payload is unavailable is `missing_blob` and names source re-acquisition, while a raw row whose source file no longer exists on disk is `source_missing` when its raw payload bytes are still retained and `source_lost` when they are not. Two rules cite another owner's durable ledger: `authority_blocked_head` (warning) is a raw an unresolved `raw_authority_blockers` row names as the accepted revision head while the index materialized a different raw of the same logical source, and `quarantined_cohort_unmaterialized` (blocking) is a raw whose `raw_session_memberships` rows are all quarantined with no revision of the logical source indexed at all. Reverse: every session traces to a raw row that is not a declared non-session artifact (phantom sessions, polylogue-b508, are reported and never deleted), and every message, block, and attachment ref traces to its owner. An attachment with no ref splits on `ref_count`: `attachment_unowned` (ref_count 0) is explained and `attachment_unreferenced` (non-zero ref_count) blocks — `ref_count` distinguishes the two only in an archive written throughout by the current sweep, so on a legacy archive `plan_orphaned_attachment_relink` is the instrument that types each ref-less row. Unexplained, unclassified, missing-blob, lost-source, quarantined-cohort, orphan, and phantom terms block; pending and authority-blocked are warnings. The acceptance instrument for a rebuilt archive: zero blocking terms. |
| `reasoning-conservation` | Every reasoning witness inside the acquired bytes of a materialized coding-origin raw reaches a thinking block of the exact session and message that carries it. Witnesses are selected structurally from the payload -- Claude Code `thinking` segments (text-bearing, signature-only, empty) and standalone Codex `reasoning` records (summary-bearing, content-bearing, opaque) -- never from the parser, the index, or any identity in the file. Denominators and outcomes are reported per origin and per variant: material surviving under a non-thinking kind is `reasoning_kind_collapsed`, an absent carrier or lost material blocks, a declared origin that contributed no readable evidence is `origin_evidence_absent`, and a bounded run that truncated is `scan_truncated`. One origin's conserved witnesses can never stand in for another's lost ones. Reads every selected blob of the two largest origins, so it is declared on the cross-tier candidate route only, never the routine live route. |
| `fts-parity` | `messages_fts` exactly covers its source `blocks` rows, archive-wide, with the worst-offending sessions surfaced by name. |
| `lineage-sanity` | `session_links.resolved_dst_session_id` and `branch_point_message_id` resolve to real sessions/messages (the latter is deliberately not a foreign key — see the data-model docs). |
| `planner-stats` | `sqlite_stat1` covers `blocks`/`messages`/`session_links`/`action_pairs` (warn-level: a fresh generation without `ANALYZE` picks pathological query plans, polylogue-l3tk class). |
| `counts-summary` | Archive-wide session/message/block counts and an origin breakdown — the numbers-freeze starting point for an operator handoff. |

Exit code is non-zero when any check reports `error` (or, with `--strict`,
`warning`). A single check's failure — including a tier database being
temporarily busy under a concurrent rebuild — never aborts the rest; each
check independently reports its own outcome.

---

## Runbooks

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

`polylogue ops maintenance raw-authority-frontier` classifies every accepted
frontier head and every terminal supersession in one pass:

```bash
polylogue ops maintenance raw-authority-frontier --output-format json
```

The pass is not recorded. Its `pass_id` is a content address over the
inspected inventory, so two passes that observe the same frontier name the
same pass and neither writes a row (polylogue-6kur ruling 2026-09-15 retired
the per-pass census ledger, which re-recorded the entire pending plan set on
every inspection). Each returned item carries its own state, actuator,
reason, evidence digest, input raw IDs, preconditions and strategy witness,
so a caller needs no separate detail handle to see a component's evidence.

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

Each row's `kind` distinguishes `frontier_judgment` (requires an accepted
judgment assertion id plus `disposition=retain_canonical_authority`, per the
conflicting-authority frontier) from `frontier_obligation` (the other
frontier obligation states -- missing bytes, unresolved provenance, corrupt
-- which resolve like an ordinary blocker: no judgment assertion is
required). The listing is bounded to `--limit` (1-500, default 100) per
call; if the response's `truncated` field is `true`, pass
`--offset <next_offset>` to read the next page. After inspecting the
blocker's own plan snapshot and current evidence, explicitly reopen
replanning with a recorded rationale:

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

### Raw-authority frontier ownership and recovery

Routine raw-authority frontier application is daemon-owned. The daemon selects
only executable proof-backed plans under the writer coordinator and validates
the typed application receipt. `polylogue ops maintenance raw-authority-frontier`
inspects only; it has no manual plan selector or apply option, and what it
records is the durable blocker set, not a census row.

### `polylogue ops maintenance operation-recovery` - inspecting and adjudicating interrupted mutations

An executor-routed mutation that is interrupted before audit finalization
leaves a nonterminal `operation_runs` row. The daemon classifies these once,
explicitly, at startup under its own writer lease (`polylogued run` ->
`recover_interrupted_operations`); it is not a side effect of constructing an
`OperationExecutor`, so a request handler never reclassifies anything.

What the archive can actually classify by itself is narrow, and the daemon
does not pretend otherwise:

- **`mutate-delete-session`** is auto-classifiable. A session either exists or
  does not, so committed target state is real postcondition evidence and the
  domain inspector can confirm applied, not-applied, or partial.
- **Every other routed family**, and any operation version this build no
  longer recognizes, **fails closed**. It is terminalized as an
  operator-blocking `unknown` with `terminal_reason = 'recovery_unknown'`.
  Create/update and mixed-effect families have no such postcondition oracle,
  and are never guessed or silently retried.

Classification is one-pass and idempotent. A classified run is terminal, so
restarting the daemon over an already-recovered archive appends no further
`recovery_classified` events.

Recovery terminal reasons deliberately keep an operation visible to bounded
inspection:

- `recovery_unknown` keeps the run visible to overlap detection, so a later
  mutation touching the same targets is refused rather than racing an effect
  nobody has proved.
- `recovered_applied` installs the duplicate-effect barrier: a confirmed
  applied recovery refuses a second attempt at the same semantic effect.

- `recovered_partial:<action>` preserves the declared `retry-exact`, `forward`,
  or `rollback` continuation. A retryable partial remains eligible for the
  same exact plan; a forward or rollback continuation remains blocked until an
  operator adjudicates it.

These states are adjudicable, which keeps them from becoming permanent wedges.
Adjudication is an offline operator route with no writer lease of its own, so
it refuses to run beside a live `polylogued`:

```bash
# Inspect only.
polylogue ops maintenance operation-recovery \
  --operation-id operation:... --output-format json

# Adjudicate, naming every durable target exactly once.
polylogue ops maintenance operation-recovery \
  --operation-id operation:... \
  --target-outcome session:claude:abc=not-applied \
  --reason "verified against source export; the delete never committed" \
  --confirm
```

A run whose plan resolved to zero durable targets never installs a barrier at
all -- it has no target rows to overlap and no semantic effect to duplicate --
and `--confirm --reason` with no `--target-outcome` closes it.

The same two operations are reachable over MCP as
`maintenance(operation="recovery_status", ...)` and
`maintenance(operation="recovery_adjudicate", ..., confirm=true)`, under the
same confirmation gate and the same offline-writer exclusion.

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
