# Source, Bead, and history evidence

## Mission authority

The attached mission requires an index-tier `messages_fts_identity` ledger keyed by rowid with unique block identity and source/recipe identity; exact O(1)-readable freshness state; drift magnitude gauges; real-trigger Hypothesis tests including rowid reuse; canonical DDL plus index version bump; unchanged tokenizer/FTS shape; and repair/rebuild integration in the existing transaction boundary.

The complete `polylogue-1xc.12` Bead record adds eight acceptance criteria. The implementation maps directly to them:

1. strict identity ledger and durable exact-state fields;
2. same-trigger and same-transaction maintenance, including rollback and recipe changes;
3. exact desired/source/recipe comparison rather than count parity;
4. bounded periodic convergence with before/after receipts and idempotence;
5. drift gauges and bounded operational history without request-time scans;
6. a real-trigger Hypothesis state machine;
7. anti-vacuity failures when a trigger/check/audit dependency is removed;
8. shared DerivationKey value semantics without shared storage or lifecycle.

## Snapshot contradiction resolved

The supplied manifest says the source worktree was dirty. The supplied branch-delta patch is zero bytes, and the extracted tracked checkout is clean at `536a53efac0cbe4a2473ad379e4db49ef3fce74d`. No tracked dirty patch was available to preserve. The implementation therefore names the commit as patch base and records the dirty-flag discrepancy instead of inventing missing source changes.

## Canonical FTS evidence

`polylogue/storage/sqlite/archive_tiers/index.py` is the canonical index-tier schema. At the base commit it defined a contentless `messages_fts` table with:

- `block_id`, `message_id`, `session_id`, and `block_type` as UNINDEXED columns;
- indexed `text`;
- `content=''`;
- `contentless_delete=1`;
- `unicode61 remove_diacritics 2`.

`polylogue/storage/fts/sql.py` is the single source of trigger DDL after history commit `74f045dd0`. The base trigger key was `blocks.rowid`, and the base readiness checks primarily compared `blocks` counts with `messages_fts_docsize` counts. This establishes the exact dependency identified by the mission: docsize proves rowid presence, not logical block identity.

Search joins FTS matches back to canonical blocks by rowid. A stale posting at a reused rowid can therefore cite the new block even though the term came from the deleted block. The deterministic regression reproduces that behavior with equal counts.

## Source and recipe identity evidence

The canonical `blocks` table carries a block content hash in the current schema. The ledger uses that as the preferred source identity, with explicit domain separation. The exact UTF-8 `search_text` fallback is retained only for minimal recovery schemas or anomalous NULL-hash rows so exact comparison remains possible.

The derivation recipe is not a manually incremented label. It is the SHA-256 of a canonical JSON descriptor containing the source selector, fold SQL, tokenizer, contentless mode, FTS column contract, subject, output contract, and version. That makes recipe drift mechanically tied to the actual write-side semantics.

## Parallel DerivationKey record

The `polylogue-wmsc` Bead specifies a storage-neutral frozen value carrying:

- subject;
- source identity;
- computational recipe identity;
- output contract.

It explicitly says consumers own their storage, scheduling, and lifecycle. No implementation of that shared value was present in the supplied source. The patch therefore introduces an FTS-local frozen value with exactly those four semantic fields and documents the interface assumption. It does not add a parallel shared framework or claim compatibility with unavailable code beyond that value shape.

## Repair and convergence evidence

The existing repair implementation lives in `polylogue/storage/fts/fts_lifecycle.py` and is called by dangling repair, convergence stages, startup readiness, archive writes, and revision application. The patch extends those routes rather than building a second repair service.

The daemon’s write coordinator is the sole writer seam. The periodic exact audit is scheduled from `polylogue/daemon/cli.py` and calls the canonical audit through `DaemonWriteCoordinator`. Startup uses durable state only when structure and trigger checks also pass; otherwise it performs exact reconciliation and bounded repair. Larger drift becomes `fts_surface` convergence debt and is handled by the dedicated unbounded surface repair.

## Observability evidence

The established Prometheus-style emission seam is `polylogue/daemon/metrics.py`. The patch reads `fts_freshness_state` and emits per-surface gauges there. `polylogue/daemon/fts_status.py` uses the same bounded durable projection for status and health. Trace-callback tests prove these normal paths do not query the canonical block set, docsize rows, or identity rows.

The disposable ops tier already holds bounded operational samples and daemon events. `fts_drift_samples` is added there with per-surface retention rather than creating a new database or durable migration regime.

## Schema lifecycle evidence

The repository has two schema regimes. `index.db` is rebuildable and must change through canonical DDL, version declaration, and reset/rebuild—not an upgrade helper. The patch changes `INDEX_SCHEMA_VERSION` to 39 and declares a `REBUILD_FTS` delta for postings, ledger, freshness state, and message triggers. The schema policy reports no derived-tier upgrade helper and no undeclared delta.

The base snapshot was already at index version 38 but lacked a corresponding lifecycle declaration. The patch adds a conservative v38 declaration for the action/delegation projection boundary before declaring v39. It does not claim a live in-place executor for v38; older indexes still follow the rebuild regime.

## Pending same-tier batching opportunity

`polylogue-wohv` is an open P4 bead observing that the four UNINDEXED content columns in the contentless FTS table are write-only and proposing their removal in the next batched index window. That work would also require a rebuild and recipe change. It was not included because this mission explicitly says not to change FTS schema shape. The integrator can batch it with v39 before merge, provided the recipe descriptor/hash and tests are updated together.

## Historical evidence

Selected history relevant to current authority:

- `ac9cfeb0b` established the current contentless message FTS table and rowid-based relationship;
- `25bea6f03` pinned `unicode61 remove_diacritics 2` and Polish folding behavior;
- `74f045dd0` consolidated canonical trigger DDL into `storage/fts/sql.py`;
- `640655861` added bounded missing-row repair;
- `4e746622d`, `cd46fcfb9`, and `ee5a50741` evolved FTS synchronization and freshness checks.

The patch preserves those decisions and extends their production routes.

## Findings corrected during implementation

Several composition defects were found and repaired before packaging:

- the initial scoped full-session replacement draft changed postings without changing identity rows or durable counters;
- the initial batched repair draft still repaired FTS without the ledger;
- missing-trigger restoration could leave an old exact marker trusted;
- a small startup drift initially caused a whole-message-index reset rather than affected-rowid repair;
- the first recipe representation repeated a long descriptor in every identity row;
- initial ops retention was global rather than independently bounded per surface;
- the first property invariant compared rowid sets but did not prove posting terms were current;
- a status closure captured a loop variable incorrectly;
- successful operational repair samples were moved after the index commit boundary.

The final patch addresses each finding and the detached-checkout verification exercises the repaired composition.

## Evidence limitations

No evidence was available from the operator’s live database, daemon process, deployment host, or private corpus. The supplied archive is source authority only. Performance evidence is synthetic and included with its method and limits. The parallel shared-key implementation and any concurrent index-schema branch were not supplied, so integration with those changes remains an integrator responsibility rather than an implied claim.
