# Exact FTS identity ledger handoff

## Outcome

This patch implements bead `polylogue-1xc.12` as an end-to-end index-tier change. It replaces count-parity as the message-search correctness proof with an exact binding among the canonical block row, the FTS5 docsize row, and an FTS-owned identity row. It also makes drift magnitude durable and observable, routes exact audit and repair through the daemon’s sole writer, and adds production-DDL regressions plus a Hypothesis state machine.

The implementation deliberately preserves the existing contentless FTS5 shape and tokenizer. `messages_fts` remains `content=''`, `contentless_delete=1`, with `unicode61 remove_diacritics 2`; no Porter stemmer and no FTS column removal are included.

## Snapshot identity

The supplied project-state archive identifies:

- repository: `sinity/polylogue`
- branch: `master`
- commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`
- subject: `fix(repair): harden raw authority convergence (#3046)`
- captured at: `2026-07-17T17:45:03Z`
- archive manifest dirty flag: `true`

The supplied `polylogue-branch-delta.patch` is zero bytes, and the extracted tracked tree had no diff against the named commit. The implementation was developed on local branch `work/fts-identity` from that exact commit. The manifest’s dirty flag therefore appears to describe excluded or untracked snapshot material rather than an omitted tracked patch.

## Evidence inspected

The implementation followed the complete `polylogue-1xc.12` record and its parent architecture records, plus the parallel `polylogue-wmsc` DerivationKey design and pending index-window bead `polylogue-wohv`. Source inspection covered canonical index DDL, schema lifecycle declarations, all message FTS trigger definitions, synchronous and asynchronous FTS repair/rebuild paths, full-session replacement, revision application, startup readiness, convergence stages, the daemon write coordinator, status/health payloads, Prometheus emission, the ops tier, and archive write effects.

Relevant history included:

- `74f045dd0` — consolidated FTS trigger DDL into one source;
- `25bea6f03` — pinned Polish folding and `unicode61 remove_diacritics 2`;
- `ac9cfeb0b` — introduced the current contentless message FTS shape;
- `640655861` — added chunked missing-row repair;
- `cd46fcfb9` and `ee5a50741` — hardened and grounded FTS freshness reporting.

## Mechanism

### Identity ledger and derivation identity

Index schema version advances from 38 to 39. Canonical DDL adds:

```sql
CREATE TABLE IF NOT EXISTS messages_fts_identity (
    rowid INTEGER PRIMARY KEY,
    block_id TEXT NOT NULL UNIQUE,
    source_hash BLOB NOT NULL CHECK(length(source_hash) > 0),
    recipe_id TEXT NOT NULL CHECK(recipe_id != '')
) STRICT;
```

The ledger is FTS-owned derived state. It does not introduce durable-tier migration machinery or a shared scheduler.

The parallel `polylogue-wmsc` implementation was absent from the snapshot. This patch therefore mirrors its storage-neutral value semantics locally with `FtsDerivationKey(subject, source_identity, recipe_identity, output_contract)`, while retaining an FTS-specific table and lifecycle. The interface assumption is explicit and mechanically narrow:

- subject: `block/search_text`;
- source identity: domain-separated canonical block source identity;
- recipe identity: a stable SHA-256 of the canonical derivation descriptor;
- output contract: the contentless postings plus rowid/block/source/recipe ledger contract.

The recipe ID in this patch is:

`sha256:4c9224c45596b37eabe5efcf5ebe24fcab3ffe9d7ff76d0343068b83593585a6`

The canonical descriptor includes selector, fold SQL, tokenizer, content mode, indexed/UNINDEXED column contract, subject, output contract, and version. Any change to those semantics changes the hash and becomes exact recipe drift. Storing the 71-byte hash rather than the 480-byte descriptor avoids repeating a long recipe document in every ledger row.

For normal production rows, source identity is `0x00 || blocks.content_hash`. Minimal recovery fixtures or anomalous legacy rows without a usable content hash use `0x01 || UTF8(blocks.search_text)`. The prefix prevents a text payload from colliding with a true content hash, and the fallback is exact rather than a lossy sentinel.

### Trigger transaction boundary

The canonical block insert, delete, and update triggers now maintain three projections in one originating SQLite statement transaction:

1. `messages_fts` postings and docsize shadow;
2. `messages_fts_identity` rowid identity;
3. the constant-size `fts_freshness_state` counters, but only while the prior row is a trusted exact-ready state.

An indexed insert adds one ledger insertion. An indexed delete adds one ledger deletion. A text-to-text update performs one ledger deletion and one insertion. Empty-to-text and text-to-empty transitions adjust all three cardinalities consistently. If any FTS, ledger, or freshness statement fails, the source block statement rolls back.

A retained ledger row makes rowid rebinding fail closed. If a delete trigger is missing and SQLite reuses the old rowid for a replacement block, the new ledger insert conflicts with the ledger primary key and rolls back the replacement insert instead of silently binding retired terms to the replacement block.

When the durable state was already stale, hot writes do not guess new archive-wide magnitudes. The physical trigger work still runs, but the recorded magnitudes remain the last exact sample and `exact=0` is preserved until an explicit audit.

### Exact freshness state

`fts_freshness_state` now carries:

- `state` and `exact`;
- `checked_at`;
- `source_rows`, `indexed_rows`, `ledger_rows`;
- `missing_rows`, `excess_rows`;
- `identity_mismatch_rows`;
- `source_mismatch_rows`;
- `recipe_mismatch_rows`;
- `duplicate_rows`;
- `repair_generation`;
- `detail`.

A ready message surface requires all of the following:

- `blocks` source, `messages_fts`/docsize, identity ledger, and all three production triggers exist;
- source, docsize, and ledger cardinalities are equal;
- every indexable block rowid has both a docsize row and ledger row;
- no observed docsize/ledger rowid lacks an indexable source row;
- ledger `block_id` equals canonical `block_id` for the same rowid;
- source and recipe identities match;
- all classified drift magnitudes are zero;
- the sample is exact.

Ordinary search readiness, daemon status, health, and metrics read only the single durable state row plus constant-size `sqlite_master` structure checks. They do not scan `blocks`, `messages_fts_docsize`, or `messages_fts_identity`. Exact joins are confined to explicit audit and repair operations.

### Exact audit and repair

The exact audit materializes the unique affected rowids in a temporary primary-key table. It classifies missing, excess, block identity, source identity, recipe identity, and duplicate drift. Repair then replaces postings and ledger rows for that exact set inside one savepoint, restores production triggers, recomputes the exact invariant, and advances `repair_generation` only after successful convergence.

If message FTS storage or the ledger is structurally absent, the same route atomically resets and rebuilds both. Full rebuild drops/recreates `messages_fts`, clears and repopulates `messages_fts_identity`, restores triggers, verifies the exact invariant, and records the successful generation before releasing the savepoint.

The batched missing-row and excess-row helpers were also changed from FTS-only operations to postings-plus-ledger operations. Each bounded window uses one rowid set and one savepoint, so a failure in either half cannot leave a partially repaired batch pending for commit.

### Full-session replacement

The high-throughput full-session replacement path temporarily suspends block FTS triggers. It now:

1. counts the old indexable rows for the exact session ID;
2. removes old postings and identity rows while old canonical rowids still exist;
3. rewrites messages and blocks with block triggers suspended;
4. inserts new postings and identity rows from the new canonical rows;
5. applies the measured old/new delta to the durable state only if the pre-replacement trigger set and state were trusted;
6. restores canonical triggers in the transaction’s `finally` path.

No prefix parsing of colon-delimited IDs is used, so a session such as `a` cannot consume rows belonging to `a:b`. If the trigger set was already incomplete, the path restores it and runs a global exact audit/repair before returning rather than manufacturing exactness from a scoped operation.

### Daemon convergence and operational history

Startup first trusts the O(1) row only when the state, table set, and trigger set all prove readiness. Otherwise it performs one exact reconciliation. Drift up to 10,000 affected rowids is repaired atomically in place. Larger drift records `fts_surface` convergence debt for the dedicated repair route instead of performing an unbounded startup rewrite.

The daemon also schedules an exact audit every six hours through `DaemonWriteCoordinator`, preserving the single-writer invariant. Periodic inline repair is bounded at 10,000 affected rowids; dedicated surface-debt convergence removes that ceiling. Exact audit receipts record before/ready/after phases in `ops.db`.

`ops.db` adds `fts_drift_samples` without changing its disposable schema version. Samples include every drift class, cardinalities, exactness, repair generation, phase, and sample time. Retention is bounded independently per surface at 512 rows, so one noisy surface cannot evict another surface’s complete window.

### Gauges

The existing Prometheus text path now emits:

- `polylogue_fts_freshness_ready{surface="messages_fts"}`;
- `polylogue_fts_drift_exact{surface="messages_fts"}`;
- `polylogue_fts_drift_rows{surface="messages_fts",kind="missing"}`;
- `polylogue_fts_drift_rows{surface="messages_fts",kind="excess"}`;
- `polylogue_fts_drift_rows{surface="messages_fts",kind="identity_mismatch"}`;
- `polylogue_fts_drift_rows{surface="messages_fts",kind="source_mismatch"}`;
- `polylogue_fts_drift_rows{surface="messages_fts",kind="recipe_mismatch"}`;
- `polylogue_fts_drift_rows{surface="messages_fts",kind="duplicate"}`;
- `polylogue_fts_drift_rows{surface="messages_fts",kind="total"}`;
- `polylogue_fts_source_rows{surface="messages_fts"}`;
- `polylogue_fts_indexed_rows{surface="messages_fts"}`;
- `polylogue_fts_identity_rows{surface="messages_fts"}`;
- `polylogue_fts_repair_generation{surface="messages_fts"}`;
- `polylogue_fts_last_exact_check_timestamp_seconds{surface="messages_fts"}`.

The existing trigger-presence gauges remain available.

## Rowid-reuse vulnerability: ghost and catch

The named regression first installs legacy count-only insert/delete triggers. It indexes block A containing `alpha`, drops the delete arm, deletes A, and inserts block B containing `beta` at A’s reused rowid. There is one canonical source row and one docsize row, so count parity reports success. Both `alpha` and `beta`, however, resolve through that rowid and cite block B. Retired content is therefore attributed to the replacement block.

The production-ledger regression repeats the missing-delete-arm sequence. A’s identity row remains, so B’s insert conflicts on `messages_fts_identity.rowid`. SQLite rolls back B’s source insertion and all trigger effects. A separate exact-audit regression constructs already-existing equal-count corruption, proves `identity_mismatch_rows=1`, repairs the single affected rowid, removes the retired `alpha` hit, and retains the current `beta` hit.

## Property-test design

`tests/property/test_fts_identity_state_machine.py` executes the real production table and trigger DDL. Its state machine performs arbitrary indexed and empty-text inserts, updates, deletes, full-session replacements, rowid-reuse delete/insert sequences, savepoint rollbacks, source-identity corruption, and recipe-identity corruption.

After every completed rule it asserts:

- desired indexable block rowids equal docsize rowids equal ledger rowids;
- ledger block/source/recipe identities equal canonical values;
- unique marker terms for current rows resolve to the correct canonical rowid;
- retired marker terms no longer match;
- the durable state is exact-ready with matching cardinalities and zero drift.

The deterministic tests state the representative mutation that would invalidate each result: removing a trigger arm, removing the ledger rowid or unique block constraint, removing a block/source/recipe comparison, replacing trusted-state gating with unconditional counter arithmetic, or adding a data scan to the O(1) status/metric seam.

## Write amplification and storage cost

The algorithmic cost is O(1) per affected block event: one ledger row operation for insert/delete, two for a text-to-text update, and one constant-size freshness-row update while exact state is trusted. Exact joins are not on the hot path.

A five-repetition synthetic SQLite microbenchmark used 10,000 indexed rows per run and compared the pre-ledger FTS-only trigger shape with production ledger plus freshness triggers. Median results were:

| Operation | FTS-only | Ledger + state | Ratio | Added time per block |
|---|---:|---:|---:|---:|
| Insert 10,000 | 0.1822 s | 0.2635 s | 1.446x | 8.13 µs |
| Update 10,000 | 0.3436 s | 0.4460 s | 1.298x | 10.24 µs |
| Delete 10,000 | 0.0590 s | 0.1242 s | 2.105x | 6.52 µs |

The delete percentage is large because the baseline delete is very short; the absolute median delta is about 65 ms for 10,000 rows. Measured database storage after insertion increased from 2,134,016 to 3,907,584 bytes, or 177.36 bytes per indexed block in this fixture. The ledger had exactly 10,000 rows after insertion, and both identity and docsize returned to zero after deletion.

These figures are a synthetic microbenchmark, not a measurement of the operator’s live archive, filesystem, WAL pattern, or daemon workload.

## Changed files

Core FTS and schema:

- `polylogue/storage/fts/sql.py`
- `polylogue/storage/fts/freshness.py`
- `polylogue/storage/fts/fts_lifecycle.py`
- `polylogue/storage/fts/dangling_repair.py`
- `polylogue/storage/fts/session_repair.py`
- `polylogue/storage/sqlite/archive_tiers/index.py`
- `polylogue/storage/sqlite/lifecycle.py`

Archive write and exact-proof routes:

- `polylogue/storage/sqlite/archive_tiers/write.py`
- `polylogue/storage/sqlite/archive_tiers/revision_application.py`

Daemon and observability:

- `polylogue/daemon/cli.py`
- `polylogue/daemon/convergence_stages.py`
- `polylogue/daemon/fts_startup.py`
- `polylogue/daemon/fts_status.py`
- `polylogue/daemon/metrics.py`

Ops history:

- `polylogue/storage/sqlite/archive_tiers/ops.py`
- `polylogue/storage/sqlite/archive_tiers/ops_write.py`

Tests were updated or added under `tests/property`, `tests/unit/storage`, `tests/unit/daemon`, and `tests/unit/archive`. No production module was added, so the topology projection does not require a new-node regeneration.

## Acceptance matrix

| Acceptance criterion | Result | Evidence |
|---|---|---|
| AC1: strict identity ledger and expanded exact state | Satisfied | Canonical index v39 DDL; ledger constraints; exact state columns and seed row |
| AC2: same-trigger maintenance and atomic rebuild | Satisfied | Production insert/delete/update DDL; savepoint rebuild/reset; full-session direct replacement; rollback tests |
| AC3: exact rowid/block/source/recipe comparison | Satisfied | Sync/async exact invariant queries; named equal-count rowid-reuse and source/recipe regressions |
| AC4: bounded periodic convergence and idempotent receipts | Satisfied | Six-hour coordinator audit; 10,000-row inline ceiling; dedicated unbounded debt repair; before/after ops samples |
| AC5: O(1) status/metrics plus bounded operational history | Satisfied | Trace-callback tests forbid source/docsize/ledger scans; per-surface 512-row retention; all drift gauges |
| AC6: real-trigger Hypothesis state machine | Satisfied | Production DDL state machine covering all requested mutation classes and posting markers |
| AC7: anti-vacuity dependencies | Satisfied | Named mutation/removal conditions in deterministic/property test docstrings and assertions |
| AC8: consume shared DerivationKey semantics without shared lifecycle | Satisfied with explicit interface assumption | Local frozen value mirrors subject/source/recipe/output fields because `polylogue-wmsc` code was absent |

## Apply and deployment order

1. Apply `PATCH.diff` to commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`.
2. Review or rebase any concurrent index-schema work before merging. This patch owns canonical index version 39 in the supplied snapshot.
3. Run the repository’s focused/static gates and schema-versioning policy.
4. Reset/rebuild the derived index tier through the existing index rebuild route. Do not add an in-place migration helper for a v38 database.
5. Start the daemon and allow startup exact reconciliation to certify the new postings/ledger pair.
6. Observe the new drift and generation gauges and retained `ops.db` samples during rollout.

The pending `polylogue-wohv` bead proposes dropping four write-only UNINDEXED columns from the contentless FTS table. That is a legitimate same-tier batching opportunity for the integrator, but this mission explicitly forbids changing FTS shape, so it is not included. If batched before merge, regenerate the recipe descriptor/hash and keep the exact-ledger semantics unchanged.

## Risks and limitations

The patch was not exercised against the operator’s live archive, live daemon, NixOS deployment, browser sources, secrets, or real corpus. Those checks remain unverified by design.

The complete repository test suite was not run. A focused affected-surface selection produced 320 passing tests, and the final patch was independently applied to a detached base checkout where 85 critical tests passed. The repository-wide `devtools verify --quick` run completed whole-tree Ruff format and lint, then exceeded the command ceiling during whole-repository mypy. Strict mypy over every modified production module passed. A standalone schema policy check passed. A prior generated-surface check completed without drift; a later combined re-run exceeded its command ceiling while entering the broad render command and produced no generated-file changes.

The local `FtsDerivationKey` should be replaced by or adapted to the eventual shared `polylogue-wmsc` value type when that work lands, but the four field semantics are isolated and documented. No shared storage or lifecycle is assumed.

Another implementation iteration is unlikely to add substantial behavior unless the integrator supplies the concurrent shared-key or FTS-shape patch. The remaining value is a small merge/certification pass: reconcile concurrent index version ownership, run the full repository/Nix lane, and benchmark a representative archive on the target filesystem.
