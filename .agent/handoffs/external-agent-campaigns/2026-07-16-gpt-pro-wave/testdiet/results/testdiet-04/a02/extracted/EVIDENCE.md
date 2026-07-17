# Source, Bead, and history evidence

## Authority resolution

The supplied full snapshot identifies `master` at `b9052e09103502017c0f510ecc699aac395de23c`, generated on 2026-07-17. Its overview says `dirty=true`, but its branch-delta artifacts are empty: no commits, no changed files, and a zero-byte patch against `origin/master`.

Reconstructing the packaged Git authority showed only tracked omissions created by the snapshot packaging boundary: 1,228 `.agent` paths and 10 other hidden/lock paths. The snapshot also explicitly reports 1,123,292,939 bytes of ignored local state. Because there was no reproducible product-source dirty patch, the named commit is the patch authority. The supplied archive and task inputs are not present in `PATCH.diff` or the result ZIP.

## Architectural decision consumed

`architecture/05-derived-freshness.md` settles four important points:

1. Currentness requires exact source identity, recipe identity, output contract, and active generation.
2. Counts, timestamps, and boolean flags are insufficient proofs.
3. A small typed `DerivationKey` vocabulary should be shared, while domain storage ledgers remain separate.
4. Incremental, targeted, rebuild, sync, async, and restarted routes should converge to equivalent canonical facts for a fixed source snapshot and recipe.

The patch follows that decision directly. It does not reopen the rejected timestamp, row-count, global-schema-only, or universal-ledger alternatives.

## Embedding findings

### Source findings

The archive embedding system had multiple selectors with an `include_stale_checks` escape. The Bead records correctly described the consequence: aggregate status and matching message counts could conceal missing/replaced metadata, and backlog/preflight/manual routes could bypass content-hash comparison. The selected relation also used metadata joins that could lose a source message entirely when the stored message identity changed but the count remained the same.

Terminal status writes had a separate ordering problem. A previous model-identity guard protected one success route, but a non-retryable old failure could still write `needs_reindex=0` and an error after newer content/config debt existed. The failure lifecycle itself was durable, but it did not identify which desired key and generation it was allowed to affect.

### Bead findings

`polylogue-wmsc` requires one indexed stale predicate for convergence, backlog, manual embedding, and preflight; complete content-and-recipe identity; and conditional success/error writes. Its later notes explicitly consolidate the same-count changed-text regression and the earlier success race into one generation invariant.

`polylogue-303r.7` requires the complete model-effect recipe rather than a model-name-only cache key. The implemented recipe fields are canonicalization, selector, chunking, provider, model revision, dimensions, task input type, normalization, tool implementation, and input/schema version. Eligibility is intentionally separate.

`polylogue-1dk1` owns orphan deletion across authoritative index generations. Its notes explicitly preserve identity-present content-hash mismatches for re-embedding rather than deletion. That is why this patch does not add a second orphan reconciler.

### Implementation consequence

The patch adds embedding-owned desired state and attempt receipts, refreshes the desired key around provider work, and makes both success and failure writes conditional on exact ownership. It changes the shared selector relation rather than patching each caller with independent logic.

## FTS findings

### Source findings

`messages_fts` is contentless. Its row count and `docsize` count cannot identify which block a reused rowid represents. SQLite can delete one block and later assign the same rowid to a different block; a stale FTS row can then pass count parity while serving the wrong evidence.

The repository already had consolidated canonical FTS trigger DDL (`74f045dd0`), which is the correct seam for adding identity maintenance. Existing readiness paths, however, could trust a recorded freshness row and same-named triggers without proving that the trigger programs contained the current identity write arms.

Targeted repair also required care: repairing one session can make that session coherent without proving no unrelated global debt remains. A targeted operation must not overwrite the global ready projection with a local result.

### Bead findings

`polylogue-1xc.12` names rowid reuse as the keystone defect and requires a rowid-to-block identity ledger, exact source/recipe comparisons, periodic exact reconciliation, and domain-owned lifecycle. It also asks for Prometheus gauges, retained `ops.db` samples, and a Hypothesis state machine.

The task’s coherent survivor slice is the identity/repair/currentness boundary. The observability and broad property-testing items are not necessary to make stale values stop presenting as current, so they remain explicitly unclaimed rather than being represented by scaffolding.

### Implementation consequence

The patch creates `messages_fts_identity`, extends freshness counters, maintains the ledger with the real triggers and rebuild paths, compares persisted trigger programs, and records global readiness only after a global exact audit. The independent survivor relation proves the ledger against `blocks` and `docsize` rather than using production counters as its oracle.

## Provider-usage insight findings

### Source findings

The existing `provider_usage` insight self-heal work correctly re-derived `session_model_usage` through the shared session insight rebuild. Its freshness stamp, however, was materializer-version-centric even though the result also depended on provider usage event facts, message token fallback rows, and the pricing catalog.

A pricing catalog change can alter monetary outputs without changing session content or materializer version. A session-level content hash is therefore not a complete source/recipe identity for this projection.

### Bead findings

`polylogue-f2qv.5` establishes the production rebuild route and the requirement that stale or zeroed usage rows self-heal. Its coordinator note also establishes the project’s fresh-first derived schema policy: widening or changing canonical DDL must force a derived tier rebuild instead of silently reopening old schema.

### Implementation consequence

The patch extends the existing `insight_materialization` receipt with exact source, recipe, output, state, and generation fields; adds source invalidation triggers; incorporates a stable packaged pricing catalog hash; and stamps the receipt in the same rebuild transaction as the usage result.

## History findings

The following history informed the implementation boundaries:

- `74f045dd0` consolidated FTS trigger DDL, so the identity arms belong in the canonical trigger source rather than a new trigger framework.
- `4177544ce` made the embedding failure/orphan lifecycle actionable and generation-aware at the archive boundary; this patch extends freshness ownership without replacing that lifecycle.
- `25bea6f03` made `pl_fold` part of all FTS write paths, so fold behavior is part of the recipe identity.
- Current convergence and repair history already centralizes daemon ownership; the patch wires exact freshness into those routes rather than adding another scheduler.
- The base commit `b9052e091` bounds raw maintenance admission; no work here weakens those admission limits.

## Contradictions and resolutions

### Snapshot dirty flag versus empty branch delta

The snapshot says dirty, but branch-delta evidence is empty and the reconstructed differences are archive omissions/local state. Resolution: base the patch on the named commit and exclude all packaging omissions.

### Full Bead scope versus mission-sized coherent slice

`polylogue-1xc.12` includes metrics, history retention, and a broad property state machine. The mission asks for the largest coherent implementation that prevents stale facts from presenting as current. Resolution: implement the exact identity, trigger, repair, startup, and test witnesses; identify observability/property additions as remaining scope.

### Generic embedding compatibility route

The repository has both archive embedding materialization and a generic repository/vector-provider route. Only the archive route has the exact message content hashes and split-tier schema needed for this seam. Resolution: make archive production routes exact and generation-conditional; retain compatibility behavior for generic callers and document that new archive callers must pass attempts.

### Orphan cleanup versus content/recipe freshness

`polylogue-1dk1` removes identities absent from an authoritative index generation, while this task handles identities that still exist but whose source or recipe changed. Resolution: compose with the existing reconciler and avoid a second deletion lifecycle.

### Dependency order

The task requires integration only after testdiet-02 and testdiet-03 are adjudicated. Those patches were not present in the isolated worktree. Resolution: produce an apply-ready patch against the named snapshot and require semantic conflict resolution after the prerequisite foundations, especially around schema generations, convergence, and rebuild publication.

## Evidence that would falsify the design

The implementation should be rejected or revised if any of the following is demonstrated:

- an archive embedding production caller can complete without capturing and passing a derivation attempt;
- a declared output-affecting embedding parameter is absent from the recipe;
- a same-count source replacement is not selected by the shared pending relation;
- an old success or failure changes status/currentness after a newer key or generation exists;
- FTS readiness can become true with same-named legacy trigger bodies, a missing identity table, rowid/block mismatch, source mismatch, or recipe mismatch;
- targeted repair can mark unrelated global FTS debt ready;
- a provider usage source or pricing catalog change leaves an old receipt current;
- old v37/v2 derived tiers can be silently reopened as v38/v3 without rebuild.

The survivor tests and modified-route suites directly exercise these falsifiers, except for live archive scale and deployment behavior, which remain unverified.
