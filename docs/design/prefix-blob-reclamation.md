# Prefix-blob reclamation — reference-blob representation for byte-proven superseded revisions

Status: **draft / proposed** (design before implementation)
Tracking: polylogue-vzn6 (this design). Cross-refs: polylogue-nh44 / #3146
(the streamed byte-proof machinery this design reuses), polylogue-83u (blob
evidence epic), polylogue-869u (census memo by blob_hash).
Evidence: measured read-only against the live archive
(`/home/sinity/.local/share/polylogue` → `/realm/db/polylogue`), 2026-07-19.

## The problem

`source.db`'s `raw_sessions` table snapshots every observed file state as a
full, independent content-addressed blob. This is deliberate: it distinguishes
append-growth from history rewrite, keeps idempotent re-ingest honest, and
gives replay authority (`classify_raw_revision_cohort`,
`plan_revision_replay`) a durable evidential trail. But when a file is
captured repeatedly while it only grows (a live Codex rollout, a live Claude
Code project transcript), every capture before the last one is a **strict
byte prefix** of a later capture — and #3146 already proves this on the read
path with zero heuristics, via streamed byte comparison
(`classify_historical_full_revision_streams`,
`polylogue/archive/revision_authority.py`). Once that proof exists, the older
bytes are redundant by construction: they can be reconstructed on demand by
slicing the successor's blob to the recorded length, and the slice's own hash
re-verifies the original identity.

Blob GC (`polylogue/storage/blob_gc.py`) correctly refuses to collect these
blobs today, because `raw_sessions.blob_hash` (and `blob_refs`) still point at
them — from GC's point of view they are simply referenced. This design adds
the missing piece: a durable **reference-blob representation** that lets a
byte-proven superseded blob's physical bytes be reclaimed while every
consumer that asks for that hash keeps getting byte-identical content,
transparently.

## Evidence pack

### Decomposing the "45.2GB" figure

The bead (and #3146's PR body) both cite **45.2GB of 97.4GB (46%)** as "raw
bytes that are superseded full snapshots of files that grew over time." That
number was computed by `#3146`'s problem statement as *(total stored bytes) −
(newest-revision-per-cohort bytes)* — i.e. group by cohort, subtract the
largest member, treat the rest as reclaimable. It does **not** verify that
the non-newest members actually form a clean byte-prefix chain; it just
assumes growth-cohort membership implies it.

Actually running the byte-proof (`classify_historical_full_revision_streams`)
against the live archive shows this assumption is mostly false. Only a small
minority of same-identity cohorts are genuine linear append-growth chains —
most are non-monotonic (edits, resegmentation, coincidental path/identity
reuse) and are correctly quarantined as ambiguous by the existing classifier.
**The real reclaimable set under the strict byte-proof rule is ~17.6GB, not
45.2GB** — about 18% of the archive's 97.4GB, not 46%.

Three buckets, computed against `/realm/db/polylogue/source.db` + blob store
(101,347 `raw_sessions` rows, 97.42GB total blob bytes) on 2026-07-19:

| Bucket | What it is | Rows | Bytes | Method |
|---|---|---:|---:|---|
| **A** — already classified | `revision_kind='full'`, `revision_authority='byte_proven'`, not the cohort's terminal (leaf) member | 992 | 10.48 GB | Trust the durable `predecessor_raw_id`/`baseline_raw_id` chain already written by `classify_raw_revision_cohort`. No bytes read. |
| **B** — typed, needs a fresh proof | `logical_source_key` assigned, `revision_kind='full'`, cohort has ≥2 members but not all already `byte_proven` (default-quarantined, never reclassified) | 318 reclaimable / 3,432 cohorts checked (50 proven, 3,382 quarantined) | 7.17 GB | Ran the real `classify_historical_full_revision_streams` against the live blob store for every such cohort. |
| **C** — never-typed (`revision_kind='unknown'`) | Legacy/backfill raws, grouped by `source_path` per the `classify_untyped_full_revision_groups` equivalence rule | 11 reclaimable / 3,240 cohorts checked (4 proven, 3,236 quarantined) | 0.0006 GB | Same streamed proof, path-grouped candidates. |
| **Total** | | **1,321** | **17.65 GB** (18.12% of archive) | |

The B/C pass took ~70s wall time reading the live blob store (read-only,
`mode=ro` on `source.db`, `BlobStore.open()` on the blob root) — the streamed
proof is I/O-bound, not compute-bound, consistent with #3146's own
measurement.

**Cross-check**: `1,321` reclaimable rows collapse to `1,129` distinct blob
hashes (some hashes are shared: a strict-prefix chain member can be
byte-identical to another chain member, e.g. a repeated no-op capture).
Verified zero of the reclaimable hashes are also the *sole* physical backing
for a hash used by a non-reclaimable `raw_sessions` row (`unsafe_hash_overlap
= 0`) — every reclaimable hash's only "non-self" `blob_refs` collisions are
themselves `ref_type='raw_payload'` rows for **other, independent** raw_ids
that happen to share identical bytes (11 such cases, `foreign_blob_refs_overlap
= 11`) — this is the ordinary CAS-dedup case (two unrelated captures
producing byte-identical content, e.g. a shared empty-session skeleton), not
a chain-membership case, and it does **not** block reclamation — see
"Read-path audit" below for why.

### Distribution

By origin (reclaimable bytes): **codex-session 15.26GB (86.4%), claude-code-session
2.39GB (13.6%)**. Codex rollouts dominate because Codex's live-capture watcher
re-snapshots the whole rollout file far more often relative to its total
growth than Claude Code's transcript capture does.

### Top cohorts (by reclaimable bytes, buckets A+B combined per `logical_source_key`)

| Reclaimable | Members | Origin | Path |
|---:|---:|---|---|
| 7.91 GB (3.80GB bucket A + 4.11GB bucket B) | 391 | codex-session | `.../2026/07/12/rollout-...-019f5562-33d8-7cf2-becc-d8cabc96e894.jsonl` |
| 2.49 GB | 6 | codex-session | `.../2026/06/29/rollout-...-019f12b5-1a85-7b42-858e-44eccf8469dc.jsonl` |
| 2.13 GB | 101 | codex-session | `.../2026/07/12/rollout-...-019f579a-9800-7542-ab6e-b6a90d81026a.jsonl` |
| 778 MB | 54 | claude-code-session | `.../projects/-realm-project-polylogue/22155309-....jsonl` |
| 625 MB | 53 | codex-session | `.../2026/07/13/rollout-...-019f59ac-a602-7ca0-872f-db5ba6a93070.jsonl` |
| 518 MB | 25 | claude-code-session | `.../projects/-realm-project-polylogue/3b0038ec-....jsonl` |
| 370 MB | 28 | codex-session | `.../2026/07/13/rollout-...-019f5a33-fbf0-7592-b6b5-1879838b5079.jsonl` |
| 361 MB | 7 | codex-session | `.../2026/07/14/rollout-...-019f6295-ae1d-7d81-aab0-a5927ae3de6f.jsonl` |
| 235 MB | 22 | codex-session | `.../2026/07/13/rollout-...-019f58fd-62b7-7d63-8cf0-b1a8616602f1.jsonl` |
| 197 MB | 20 | claude-code-session | `.../projects/-realm-project-sinex/29c9c033-....jsonl` |

The single biggest cohort (one live Codex rollout, still growing) alone
accounts for 7.91GB — 45% of the entire reclaimable set. This matches #3146's
own anecdote ("one Codex rollout file alone: 800 stored revisions / 6.2GB for
a file whose final size is ~8MB") in shape, though the exact bytes have grown
since that PR landed (this file is still being actively captured).

Reproduction: the three probe scripts (`evidence_pack.py`, `prove_mixed.py`,
`prove_unknown.py`, `final_evidence.py`) and their raw JSON results are
preserved for audit; they run entirely read-only (`mode=ro` SQLite URIs,
`BlobStore.open()` reads) against the live archive and import the project's
own `classify_historical_full_revision_streams` rather than reimplementing
the proof.

## The reference-blob representation

### Where the reference row lives

**A new table, not `blob_refs`.** `blob_refs` (source.db) is a
referrer-identity relation — "this `blob_hash` is pointed at by this
`(ref_type, ref_id)`" — used by GC's reference-counting and dedup bookkeeping.
A prefix reference is a different axis entirely: "this `blob_hash`'s *content*
is reconstructible from elsewhere." Overloading `blob_refs`' closed
`ref_type` CHECK (`raw_payload | attachment | sidecar`) to also mean
"reconstruction recipe" would conflate two independent facts about a hash.

Proposed additive migration (source schema v14, `014_blob_prefix_references.sql`):

```sql
CREATE TABLE IF NOT EXISTS blob_prefix_references (
    blob_hash            BLOB PRIMARY KEY CHECK(length(blob_hash) = 32),
    byte_length          INTEGER NOT NULL CHECK(byte_length > 0),
    successor_blob_hash  BLOB NOT NULL CHECK(length(successor_blob_hash) = 32),
    proven_at_ms         INTEGER NOT NULL CHECK(proven_at_ms >= 0),
    reclaimed_at_ms       INTEGER CHECK(reclaimed_at_ms IS NULL OR reclaimed_at_ms >= proven_at_ms)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_blob_prefix_references_successor
ON blob_prefix_references(successor_blob_hash);
```

- `blob_hash` — the **reclaimed** blob's own hash (its permanent, re-verifiable
  identity; `raw_sessions.blob_hash` for the superseded row is left
  **unchanged** — its identity fact doesn't change, only how its bytes are
  served).
- `byte_length` — the prefix length to slice from the successor (== the
  reclaimed row's own recorded `blob_size`; redundant with `raw_sessions` by
  design, so a reference row is independently interpretable without a join).
- `successor_blob_hash` — the **immediate** next hop in the byte-proven chain
  (not necessarily the cohort's current terminal leaf — see "chained
  resolution" below).
- `proven_at_ms` — when the streamed byte-proof was last confirmed. This is
  audit metadata; the row is only ever inserted after a successful proof, so
  there is no separate "verified" flag to gate on.
- `reclaimed_at_ms` — `NULL` until a later confirmation sweep observes the
  physical file is actually gone (pure audit bookkeeping; nothing depends on
  this being set promptly — see crash-window analysis below).

This is purely additive (`CREATE TABLE`/`CREATE INDEX`) and needs no backup
manifest by itself, per the durable-tier regime (`docs/internals.md`). The
**consent gate applies to the reclamation *operation*** (irreversible
physical `unlink`), not to this schema change — see below.

### Chained resolution, not always-flatten-to-leaf

A cohort's chain can be deep (391 members in the top cohort above). Pointing
every reference row at the *current* physical leaf would need every existing
reference row **rewritten** whenever a new leaf supersedes the old one —
write amplification proportional to reclaimed history depth on every single
new capture of an actively-growing file (the common case). Instead, each
reference row points at its **immediate successor** in the proven chain
(exactly the edge `classify_historical_full_revision_streams` already
produces as `predecessor_raw_id`, inverted). Reads resolve iteratively:
follow `successor_blob_hash` hops until landing on a hash with either (a) a
physical file on disk, or (b) no reference row and no physical file (a hard
error — should be provably unreachable given the write ordering below).
Bounded by a depth guard (recommend 4,096 — the largest observed real cohort
is 391 members with substantial headroom) to convert a would-be infinite loop
(a corrupted/cyclic reference — should never happen given CAS immutability,
see below) into a loud error rather than a hang.

This is safe *because* `BlobStore` blobs are immutable once published
("Files are never modified after creation" — `blob_store.py`): once a
successor's bytes are proven to contain a given prefix, that fact can never
be invalidated by the successor changing underneath the reference row. There
is no TOCTOU window here at all — a whole class of race that a mutable-file
reference scheme would have to guard against doesn't exist.

## Read-path audit

Every consumer of raw payload bytes was enumerated (not just the "publisher"
surface the bead names):

**Publisher-mediated (archive-owned, already funnel through one class):**
`ArchiveStore.raw_revision_material` / `open_raw_revision_material` /
`blob_path_for_hash` (`storage/sqlite/archive_tiers/archive.py`) all call
`self._blob_publisher.open(blob_hash)` / `.read_all(...)` / `.blob_path(...)`
— i.e. `ArchiveBlobPublisher` (`storage/blob_publication.py`), which
subclasses `BlobStore`.

**Direct `BlobStore`/`get_blob_store()` callers — a wider surface than the
bead's framing implies.** Fifteen-plus call sites read blob bytes via
`blob_store.blob_path(hash)` directly rather than through the publisher:
`daemon/backup.py`, `pipeline/services/validation_runtime.py`,
`sources/revision_backfill.py`, `storage/raw_retention.py`,
`storage/artifacts/inspection.py`, `schemas/sampling_db.py`,
`schemas/validation/corpus.py`, `storage/blob_integrity.py`,
`sources/source_parsing.py`, `sources/live/batch.py`, `storage/repair.py`,
`storage/raw_reconciler.py`, `storage/sqlite/queries/artifacts.py`,
`pipeline/services/ingest_worker.py`. Several of these are exactly the
maintenance/forensic tools (repair, reconciler, inspection, integrity audit)
that legitimately need to open **any** raw_id's exact original bytes on
operator demand — including a superseded one.

**Design decision: push reference-resolution into the shared `BlobStore`
base class, not just `ArchiveBlobPublisher`.** `BlobStore` is deliberately
filesystem-only today (no DB awareness) and is the common ancestor both
`ArchiveBlobPublisher` and every direct caller above route through via
`get_blob_store()`. Threading an *optional* resolver hook into `BlobStore`
(a callback `hash -> (successor_hash, byte_length) | None`, `None` by
default — every ad hoc/test/demo `BlobStore(root)` construction is
unaffected) and wiring it in `get_blob_store()`'s singleton construction
(consulting the sibling `source.db`'s `blob_prefix_references` table, one
indexed lookup per miss) makes every one of the fifteen-plus call sites
reference-aware for free, through one code change, instead of fifteen
individual audits.

- `open(hash)` / `read_all(hash)` / `read_prefix(hash, n)`: if the physical
  file exists, unchanged fast path (the overwhelming majority of blobs,
  always). Otherwise resolve the reference chain and return a bounded
  reader/slice.
- `exists(hash)`: true if the physical file exists **or** a reference row
  resolves.
- `blob_path(hash)` — **the one method that cannot serve a reference
  transparently as a permanent path.** Its contract today is "a real,
  reusable, indefinitely-valid filesystem path" (used verbatim for
  `sqlite3.connect()` on Hermes `state.db`/`verification_evidence.db` blobs,
  streaming JSONL parses, and as a cache key by some callers). For a
  reference-backed hash there is no single real file containing exactly
  those bytes — only a slice of a larger file. The only honest option is to
  spill a **bounded, private, ungoverned-by-CAS temp file** (mirroring the
  existing `prepare_from_bytes`/`prepare_from_path` tempfile pattern) sliced
  from the successor, and return that path — explicitly **not** publishing it
  back into the CAS namespace (that would silently re-materialize the exact
  bytes just reclaimed, defeating the point).

**This is the single riskiest open question in this design** (see "Open
questions" below): callers of `blob_path()` today implicitly assume the
returned path is a permanent, reusable CAS artifact. A caller that caches the
path, hardlinks it, or assumes repeated calls for the same hash return the
same path will observe different (though byte-identical-content) behavior
for a reference-backed hash — a spilled temp file, a fresh one on each call
unless we add a caller-visible spill cache. None of the fifteen-plus call
sites were audited line-by-line for this assumption in this design-only
lane; that audit is deferred to the implementation phase and should be its
own explicit checklist item, not an assumed side effect of this document.

## Write/GC interaction — lease-safe reclamation ordering

**Reuse `blob_gc.py`'s existing hardened deletion path; do not build a
parallel one.** The bead's own phrasing — "mark physical blob collectible →
existing GC collects" — points at the right shape: extend
`_reference_surfaces`/`_still_referenced` (`storage/blob_gc.py`) with one more
clause, and let GC's already-audited candidate-discovery,
`MIN_AGE_S`-gated, `BEGIN IMMEDIATE`-locked, publication-reservation-checked
unlink + `gc_generations` bookkeeping do the actual deletion:

```python
# in _archive_reference_surfaces / _reference_surfaces, before checking
# raw_sessions / blob_refs / attachments:
if _table_exists(conn, "blob_prefix_references"):
    if conn.execute(
        "SELECT 1 FROM blob_prefix_references WHERE blob_hash = ? LIMIT 1",
        (blob_bytes,),
    ).fetchone() is not None:
        return []  # reconstructible: not "still referenced" for GC purposes
```

`raw_sessions.blob_hash`/`blob_refs` rows for the reclaimed hash are
deliberately left pointing at it (identity fact, unchanged) — this clause is
what tells GC those rows no longer need the *physical* bytes at that path.
Everything downstream of `_reference_surfaces` in `run_blob_gc_report` is
untouched: candidate discovery still walks the filesystem, the age gate still
applies, the publication-reservation check still applies, the destructive
recheck-and-unlink still happens under the same source-tier write lock.

Ordering (matching the bead's requested sequence, corrected for logical
dependency — verification must precede the durable write that makes a hash
collectible):

1. **Verify** the slice (streamed byte comparison, read-only, no side
   effect) — this already happened for buckets A/B/C above.
2. **Write the reference row** (single transaction on `source.db`). The
   instant this commits, the hash becomes GC-collectible via the clause
   above.
3. **Existing GC eventually collects** the physical file, on its own
   schedule, via its own hardened path, with its own `gc_generations`
   accounting.
4. **Confirmation sweep** (new, cheap, idempotent, run as part of the same
   maintenance pass or on demand): for every `blob_prefix_references` row
   with `reclaimed_at_ms IS NULL`, check `store.exists(blob_hash)`; if
   `False`, stamp `reclaimed_at_ms`. Pure audit bookkeeping.

### Crash-window analysis

- **Crash after verify, before the reference row commits**: no durable state
  changed; the physical blob is untouched and fully self-sufficient. Safe —
  idempotent, re-run classification later.
- **Crash after the reference row commits, before GC ever runs**: reads for
  that raw_id/hash still find the physical file present (GC hasn't touched
  it) — both the direct-filesystem path and the reference-aware path
  succeed, redundantly. No data loss, no inconsistency.
- **New race introduced by the added GC clause**: a `blob_prefix_references`
  insert racing a concurrent GC pass. SQLite's WAL snapshot isolation means a
  GC pass that started before the insert's commit simply won't see the new
  row and will correctly still treat the hash as referenced via
  `raw_sessions` (its existing check) — it skips deletion this pass and
  tries again next time. No unsafe window is introduced by this change; GC's
  own existing recheck-under-lock already handles the symmetric case
  (something becoming referenced between shortlist and delete).
- **Crash mid-GC-unlink**: unchanged from today — this is GC's existing,
  already-audited crash safety (unlink is attempted only under
  `BEGIN IMMEDIATE`, `gc_generations` row commits atomically with the
  transaction that recorded the deletion).
- **Crash after physical unlink, before the confirmation sweep stamps
  `reclaimed_at_ms`**: harmless. `reclaimed_at_ms` is pure audit metadata —
  reads already redirect via the reference row regardless of its value. The
  next sweep notices the file is gone and stamps it then. Idempotent.
- **The one invariant that must never be violated**: a reference row must
  never be written for a slice that has not been freshly, successfully
  proven against the *current* successor bytes. Because `BlobStore` blobs are
  write-once/immutable, "freshly" only matters for the successor existing at
  all at verification time — there is no possibility of the successor's
  *content* changing between verify and write.

## Consent gate

The **schema addition** (`blob_prefix_references`) is ordinary additive
source-tier DDL and needs no special gate beyond the existing migration
discipline. The **reclamation operation** (the physical `unlink` that GC
performs once a hash is marked collectible) is the actually destructive,
irreversible act — once the file inode is gone, only a backup restores the
byte-exact original. This should be gated exactly like other destructive
durable-tier changes: a **verified backup manifest** (`daemon/backup.py`,
`backup_archive`/`_verify_backup_result`) run immediately before the first
production reclamation pass, plus an explicit operator go-ahead (not
automatic — no daemon-scheduled auto-reclamation in v1). `backup_archive`
already builds its blob inventory from `source.db`'s referenced-hash set
(`_source_blob_inventory`); a backup taken *after* reclamation naturally
excludes physically-deleted hashes and includes the successor + reference
rows, which together are sufficient to reconstruct the original bytes —
this needs no special-casing in the backup tool, only confirmation (via the
verification plan below) that the reconstruction is actually byte-exact
before the first physical deletion is allowed to proceed.

## Verification plan

Two invariants must be checked, at two grain levels:

1. **Byte-identity**: for every `blob_prefix_references` row, slicing the
   resolved chain (following `successor_blob_hash` hops to a physical file,
   reading the first `byte_length` bytes) and re-hashing must equal
   `blob_hash`. This is the same comparison the streamed proof already makes
   at write time; the verification command re-runs it independently,
   distrusting the durable row, as a standing integrity check (mirrors
   `BlobStore.verify_all`'s "distrust the DB, trust only re-derived bytes"
   posture).
2. **Read-path parity**: for a sample of reference-backed hashes, call every
   audited read entry point (`open`, `read_all`, `read_prefix`, `blob_path`)
   and assert byte-for-byte equality against the pre-reclamation content
   (captured once, before the first real reclamation, from a snapshot/backup
   — this is exactly why the consent-gated backup must precede reclamation).

Two modes:
- **Sampled** (fast, safe to run frequently / in CI against the demo corpus):
  a bounded random sample of reference rows, full re-hash.
- **Full** (slow, operator-triggered, mirrors `verify_all`'s "checked /
  checked_bytes / failures" report shape): every reference row, every time.
  Cost scales with total reclaimed bytes read back (17.65GB today), not
  archive size — cheaper than a full blob-store `verify_all` pass.

## Forward-fix sibling: acquisition-side append-delta storage

Only 2,303 of 101,347 rows (~2.3%) are `revision_kind='append'` — i.e. the
capture pipeline already has a working append-delta path
(`sources/live/append_ingest.py`, `_AppendPlan`) but it activates far less
than the growth-cohort evidence would suggest it should. Root cause, traced
through `_append_plan` (`sources/live/batch.py:2132`): append planning
requires a **pre-existing cursor** from `CursorStore.get_record(path)` with a
matching `parser_fingerprint` and non-`None` `content_fingerprint`. The
`ingest_cursor` table backing `CursorStore` lives in **`ops.db`** — the
**disposable** tier (`docs/architecture.md`'s five-tier table: "ops.db |
disposable | ingest cursors, ..."). Every `ops.db` reset (index rebuilds,
schema mismatches, `polylogue ops reset`) wipes cursor state, forcing the
*next* observation of every currently-growing file back onto the full-capture
path — even though the file itself hasn't changed shape at all. Given how
often `ops.db` gets reset relative to how often a real Codex/Claude session
file grows, this fully explains the observed ~2%/98% append/full split: the
append path isn't broken, its *continuity* is anchored to the wrong
durability tier.

**Sketch of the fix**: when `_append_plan` finds no usable `ops.db` cursor,
fall back to reconstructing an equivalent cursor from **durable** evidence
already in `source.db` before giving up and falling back to a full capture —
specifically, the accepted chain's current head (`raw_revision_replay_plan`
/ `classify_raw_revision_cohort`'s already-durable `predecessor_raw_id` /
`baseline_raw_id` / `source_revision` / `blob_size` columns) already carries
everything `_append_plan` needs (`byte_offset`, `content_fingerprint`
equivalent, `parser_fingerprint` match) to resynthesize a `CursorRecord`
without touching `ops.db` at all. This is additive to `_append_plan` (a
secondary lookup path, tried only when the primary disposable-tier cursor is
absent), not a schema change, and does not weaken the existing
`RawRevisionAuthority` model — `classify_raw_revision_cohort` remains the
sole authority for what's accepted; this only changes *how eagerly* the
append path is attempted before falling back to a full capture.

**Decision recorded**: implement this as a **follow-up**, not in this design
lane (which is measurement + reference-blob design only). Filing as a
follow-up bead is appropriate scope — it's a distinct, independently
shippable perf/correctness fix to the *forward* accumulation rate, orthogonal
to reclaiming the *existing* backlog this design addresses.

## Anti-goals

- **No heuristic similarity reclamation.** Only the streamed byte-proof
  (`classify_historical_full_revision_streams`) authorizes a reference row.
  Same-size, same-prefix-hash-sampled, or "looks like a rename" heuristics
  are explicitly out of scope — the evidence pack above shows *most*
  same-identity/same-path cohorts are **not** clean prefix chains (3,382 of
  3,432 typed cohorts, 3,236 of 3,240 untyped cohorts, quarantined on first
  real check) — heuristics would be wrong most of the time here.
- **No reclamation of a quarantined/ambiguous proof chain.** A cohort that
  fails `classify_historical_full_revision_streams` (branching, non-monotonic
  sizes, a byte mismatch) gets zero reference rows, full stop — it remains
  exactly as durable as it is today.
- **Retired-generation cleanup is not this bead.** `.index-generations.retired-*`
  directories and similar retired-tier debris are a separate, already-tracked
  concern; this design touches only `source.db`'s raw-payload blob substrate.

## Open questions / risks (ranked)

1. **`blob_path()`'s permanent-path contract** (see "Read-path audit") is the
   riskiest unresolved point: fifteen-plus call sites assume a real, stable
   CAS file back a returned path; a spilled ephemeral file for reference
   hashes may violate assumptions none of them were individually audited for
   in this design-only lane. Recommend the implementation phase begin with
   an explicit per-call-site audit (does it re-call `blob_path()` for the
   same hash and expect path stability? does it hardlink or cache the path
   long-term?) before writing the spill-file code.
2. **Depth-bound tuning** for chained resolution (recommended 4,096) is a
   round-number guess against today's largest observed cohort (391 members);
   should be revisited if a materially deeper chain is observed in practice.
3. **Confirmation-sweep cadence** (`reclaimed_at_ms` stamping) is unspecified
   here beyond "idempotent, run as part of maintenance" — needs a concrete
   scheduling decision (daemon convergence stage vs. explicit CLI command)
   in the implementation phase.
