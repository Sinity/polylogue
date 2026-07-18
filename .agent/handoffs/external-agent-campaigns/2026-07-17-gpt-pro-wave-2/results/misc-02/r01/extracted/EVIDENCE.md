# EVIDENCE — `polylogue-wmsc`

## Snapshot authority

| Item | Observed value |
|---|---|
| Supplied project-state archive | `/mnt/data/polylogue-all.tar(133).gz` |
| Archive SHA-256 | `ec3fa193b2f99a6daee0bc620af4f69b5728834e99f8196792a17c3fbae11155` |
| Mission file | `/mnt/data/misc-02-embedding-freshness(2).md` |
| Mission SHA-256 | `6bed92923ef359f274a4c4362d2258bb5f9fe582f195c093d982415eade05a15` |
| Git commit | `536a53efac0cbe4a2473ad379e4db49ef3fce74d` |
| Ref identity | `master`, `origin/master` |
| Commit subject | `fix(repair): harden raw authority convergence (#3046)` |
| Commit time | `2026-07-17T18:55:47+02:00` |
| Initial tracked state | Clean |
| Draft branch | `work/polylogue-wmsc` |

The snapshot's dirty/project-state marker was attributable to untracked archive/context material. The implementation began from an unchanged tracked tree at the named commit. No supplied archive, project-state input, daemon state, credentials, or copied database appears in the result package.

## Repository rules and architecture inspected

- Root `CLAUDE.md` and `AGENTS.md`: repository ownership and verification rules; substrate-first implementation; canonical generated artifacts must be refreshed rather than hand-maintained inconsistently.
- `polylogue/storage/sqlite/archive_tiers/embeddings.py`: canonical DDL authority for the rebuildable embeddings tier.
- `docs/schema.md`: tier boundaries and rebuildability/cost statement.
- Archive-tier bootstrap and sqlite-vec initialization paths: schema version is checked at initialization; no migration chain is required for a rebuildable tier.
- `devtools/build_topology_projection.py`, `docs/plans/topology-target.yaml`, and `docs/topology-status.md`: generated ownership/projection evidence.
- `.agent/handoffs/external-agent-campaigns/2026-07-16-gpt-pro-wave/testdiet/context/testsuite_diet/architecture/05-derived-freshness.md`: later architecture decision requiring exact derivation key plus active generation, domain-specific ledgers, conditional old-worker completion, and retryability separate from freshness.

The Diet decision directly rejects boolean `needs_reindex`, row/count parity, timestamps, global schema version alone, and a universal derivation ledger as sufficient/current architecture.

## Beads read in full

### `polylogue-wmsc`

Direct implementation authority. Material findings:

- A `DerivationKey` has exactly four conceptual components: subject/grain, exact source identity, complete computational recipe identity, and output contract.
- Generation, producer/resource data, eligibility/privacy, lifecycle, retryability, and result/output hash are separate.
- Per-source convergence, bulk backlog, manual embed, and preflight must use one indexed stale predicate on the same snapshot.
- Success and terminal error cannot clear freshness after a later source/recipe key or generation.
- Terminal/non-retryable disposition is scoped to the failed key/generation.
- Mutation proof is required for every recipe field, each caller bypass, and conditional terminal writes.
- FTS must be able to reuse the value shape without sharing embedding storage/scheduling/lifecycle.

### `polylogue-303r.7`

Recipe naming authority. It declares input digest, canonicalization, record/chunk selector, chunking version, provider, model/revision, dimensions, task/input type, normalization, tool implementation, and input/schema version as computational request identity. It states that `output_hash` is result integrity, while privacy/access/retention/deletion are eligibility/lifecycle metadata.

The patch maps those names into `EmbeddingRecipe` and a separate output contract. No second effect-envelope or universal receipt framework is introduced.

### `polylogue-1xc.12`

Shared-shape coordination authority. It requires FTS to consume the same derivation value semantics while retaining an FTS-owned identity ledger and lifecycle. This drove the neutral names `DerivationSubject`, `DerivationIdentity`, `DerivationKey`, and `DerivationKeyLike`, with no imports from embeddings in the shared file.

### `polylogue-iqd3`

The old terminal error race. It identified `mark_session_embedding_error(..., retryable=False)` unconditionally clearing `needs_reindex` after a concurrent config mark. The bead is closed as superseded by `wmsc`; the stronger fix scopes every failure and resolution projection to generation plus exact key rather than comparing only model strings.

### `polylogue-0k6`

The split-tier same-ID/same-count full-replace regression. It requires stale selection and replacement—not duplication—of old vector/meta output after changed content. It is closed as absorbed by `wmsc`. The delivered test uses the canonical selector and generation-guarded full-session writer against real embeddings DDL/sqlite-vec.

## Production source findings

| Route/file | Finding before patch | Consequence | Implemented authority |
|---|---|---|---|
| `materialization.select_pending_archive_session_window` | Public `include_stale_checks` allowed callers to disable current-content comparison. | One function exposed two incompatible definitions of pending. | Public bypass removed; v3 exact predicate plus always-content-aware legacy fallback. |
| Per-source convergence in `daemon/convergence_stages.py` | Called the content-aware path. | It was the sole caller capable of detecting same-count content replacement. | Same exact-key predicate as all other selectors; sibling recipe ledger reconciled first. |
| Daemon backlog in `daemon/embedding_backlog.py` | Passed `include_stale_checks=False`. | Stale vectors could be omitted from automatic catch-up. | Passes configured `EmbeddingRecipe`; exact predicate selects content/recipe/materialization drift. |
| Manual backfill in `cli/commands/embed.py` | Passed `include_stale_checks=False`. | Operator remediation could report no work for stale content. | Uses preflight model/dimension recipe and the same selector. |
| Preflight in `storage/embeddings/preflight.py` | Passed `include_stale_checks=False`. | Cost/window estimate disagreed with convergence and could authorize a misleading no-op. | Uses the configured recipe and same selector snapshot. |
| `count_archive_embedding_session_state` and status payload | Duplicated approximate status/count semantics; exact refinement was detail-only. | Readiness could still call stale sessions embedded even after selector repair. | Shared predicate supplies exact fresh/pending/current-key-blocked counts under bounded progress handling. |
| `_record_archive_embedding_success` path | Previous race fix compared only the model at terminal write. | Content, selector, dimensions, revision, schema, or other recipe changes could still be clobbered; vector writes could precede terminal authority. | Attempt token captures key/generation; full session is staged and atomically published only after exact guard. |
| `mark_session_embedding_error` and failure resolution | Non-retryable/unscoped receipts could clear compatibility freshness without proving they belonged to current work. | An old terminal error or acknowledgement could erase a newer config/content mark. | Keyed conditional state update; stale/unscoped receipts remain superseded audit evidence only; generation-zero resolution is legacy-scoped. |
| Config reconciliation | Production checks opened `index.db`, while derivation/status state normally lives in sibling `embeddings.db`. | A correct generation mechanism placed on the wrong tier would not protect production. | Reconcile sibling embeddings tier, loading sqlite-vec for dimension handling, before source selection. |
| Canonical embeddings DDL | v2 had status/meta/failure lifecycle but no exact active session derivation generation. | Counts/flags/model strings could not prove semantic currentness. | v3 domain ledger and per-message key/recipe/generation metadata; rebuild route. |

## Shared-predicate evidence

The v3 predicate derives `desired_messages` from the production archive embeddable-message relation and groups one `desired_sessions` source snapshot. It then classifies each eligible session:

- **fresh**: exact desired key/source/recipe/output match, active state `succeeded`, clean compatibility status with exact count, and every desired message has matching content hash, message key, recipe, and active generation;
- **blocked**: exact current key with `failed_terminal`;
- **pending**: neither fresh nor blocked.

This classification is used by the selector and exact status counts. The four scheduling callers differ only in their requested IDs/bounds/rebuild mode, not freshness semantics.

The predicate deliberately treats missing ledger state, source drift, recipe/output drift, retryable failure, superseded work, status drift, and missing/mismatched message materialization as pending. An old terminal failure is not blocking after the desired key changes.

## History inspected

### `d07627ebeffb41100d4da82a1390dd8645d4dcae` — 2026-07-03

`fix(embeddings): detect same-count message edits`

This commit changed archive message content hashes to include stored role/type/material-origin and text/block content, then taught archive pending selection to compare message metadata hashes. It established the right content fact but did not remove the three call-site bypasses.

### `b998ec4cfc28cde2606c89631a451b3b0133fd47` — 2026-07-02

`fix(embeddings): reselect stale clean archive sessions`

This commit made clean rollup/status candidates undergo exact refinement and aligned status detail with the embeddable-message relation. It is the immediate ancestry of the content-aware branch now made mandatory.

### `16fdb0fcacba55f9640222a3e0e52984f6c414aa` — 2026-07-09

`fix(storage): gate embedding success write against config-change race (#2616)`

This commit compared the producing model with current configuration before clearing `needs_reindex`. Its message explicitly documented the identical terminal-error gap as `polylogue-iqd3` and chose no schema generation. Later `wmsc` and the Diet architecture decision supersede that local choice: exact source plus full recipe plus output contract and generation are now required.

### `4177544ce7c24b51ef56a8e875116c0e6ae9978c` — 2026-07-13

`fix(embeddings): make embedding lifecycle actionable (#2796)`

This introduced inspectable failure history, requeue/acknowledge/supersession, status surfaces, and orphan-generation authority. The patch preserves those lifecycle records and scopes their projection to the failed derivation generation/key.

## Contradictions and resolutions

1. **Earlier model-only terminal comparison versus current exact-key authority.** The earlier commit intentionally avoided a schema generation and covered one race. `wmsc` and the later Diet decision require a complete key/generation. Current authority wins; the patch replaces model-only terminal authority rather than layering another boolean check.

2. **One selector API with an optional content bypass versus “one predicate.”** Three production callers used the bypass. Retaining the argument would preserve competing authorities and make future regressions one keyword away. The argument is removed; legacy fallback remains content-aware.

3. **Status aggregates versus selector semantics.** Fixing only selection would still let readiness claim convergence from flags/counts. Exact status counts now call the same predicate, bounded by the existing SQLite progress mechanism.

4. **Config reconciliation against the index connection versus split-tier ownership.** Production source facts are in `index.db`; embedding generation/status/meta live in sibling `embeddings.db`. The patch moves recipe reconciliation to the owner tier before selection while retaining same-file behavior for legacy fixtures.

5. **`polylogue-0k6`'s conditional suggestion to upsert by `(session_id, position)` versus current identities.** Current source uses stable message IDs and vector storage keyed by message ID. The stronger current-source-safe operation is an exact, generation-guarded full-session replacement: delete prior session vectors/meta, validate the full desired source, and write one current row per message. No new positional identity or vector format is invented.

6. **Documentation said embeddings schema v1 while source was already v2.** Canonical source at the snapshot is authoritative. The patch bumps source from v2 to v3 and updates documentation directly to v3.

7. **A universal derivation table would simplify generic queries.** Both `wmsc`, `1xc.12`, and the Diet decision explicitly reject it. The shared module contains values/protocols only; `embedding_derivation_state` remains embedding-owned.

## Schema and compatibility evidence

- Snapshot canonical source declared `EMBEDDINGS_SCHEMA_VERSION = 2`.
- Patch declares version 3 and edits only canonical DDL; no migration module or `ALTER TABLE` chain was added.
- New v3 data is rebuildable and storage-local:
  - session derivation state with exact hashes/generation/attempt state;
  - per-message recipe/key/generation metadata;
  - failure receipt identity columns;
  - pending-state index.
- Minimal/pre-v3 test fixtures still select through the existing content-aware fallback.
- v2 failure fixtures remain readable through column introspection and generation-zero defaults.
- Once a session has v3 keyed state, unscoped legacy writes are audit-only and cannot alter freshness.

## Verification evidence

- Consolidated focused compatibility run: `174 passed, 2 warnings in 9.51s` across eleven directly affected API/CLI/daemon/MCP/storage files.
- Clean detached-worktree application and direct invariant/writer run: `29 passed, 2 warnings`.
- `git diff --check`: clean.
- Changed Python compilation: clean.
- Generated topology status check: clean.
- Topology verification: non-blocking; zero orphan/missing/conflict/kernel findings; nine pre-existing storage-root TBDs.
- Synthetic read-only audit: one eligible changed-content session was exact-pending, old-bypass-fresh, counted as `missed_by_old_bypass`, and source-drifted; partition check true.
- Locked dev-extra hydration attempted and blocked by external DNS while fetching `virtualenv==21.2.0`; no claim of full Ruff/mypy/repository-gate execution.

Full commands, mutation mapping, environment details, and remaining live checks are in `TESTS.md`.

## Evidence not available

No access was available to the operator's live archive, running daemon, deployment, provider credentials, or Voyage API. Consequently this package does not claim:

- a measured live stale-session count;
- production query-plan/runtime data;
- provider spend/rebuild completion;
- daemon postflight or live status convergence;
- an in-place migration result (none is designed).

`HANDOFF.md` supplies the required read-only audit harness so the integrator can convert the historical drift into concrete counts without granting mutation authority.
