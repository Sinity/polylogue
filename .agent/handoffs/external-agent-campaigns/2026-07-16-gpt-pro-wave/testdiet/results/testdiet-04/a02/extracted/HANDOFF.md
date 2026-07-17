# Testdiet 04 handoff: monotonic derived freshness

## Mission outcome

This package implements a coherent survivor law for three current Polylogue derivation domains: archive embeddings, contentless message FTS, and the `provider_usage` session insight. A derived result is current only when its exact source identity, full computational recipe identity, output contract, and active domain generation agree. Counts, timestamps, trigger names, and `needs_reindex` remain compatibility or operational signals; none is accepted as sufficient currentness proof.

The implementation deliberately introduces one small storage-neutral value vocabulary—`DerivationKey` and `DerivationAttempt`—while retaining independent storage, scheduling, repair, and lifecycle ledgers for each domain. It does not add a universal derivation table or a Diet-specific freshness token.

## Snapshot identity

The supplied Chisel snapshot was generated at `2026-07-17T10:53:28Z` from `/realm/project/polylogue` and identifies:

- branch: `master`
- commit: `b9052e09103502017c0f510ecc699aac395de23c`
- commit subject: `fix(daemon): bound raw maintenance admission (#2975)`
- commit date: `2026-07-17T08:28:35+02:00`
- snapshot flag: `dirty=true`
- branch delta against `origin/master`: zero commits, zero changed files, zero-byte patch

The dirty marker was not a reproducible source patch. The archive reported more than 1.1 GB of ignored local state, and its reconstructed working tree omitted 1,238 tracked packaging/context files—1,228 under `.agent/`, plus `.beads`, `.claude`, `flake.lock`, and `uv.lock`. No product-source branch delta was supplied. Accordingly, `PATCH.diff` is based on the named commit, not on those archive omissions.

Implementation work was preserved on local branch `testdiet-04-derived-freshness`, still based at the exact snapshot commit. The package patch changes 36 paths, with 3,233 insertions and 433 deletions. It is 229,082 bytes and was generated with binary hunks for the repository’s binary-attributed generated topology files.

Input identities inspected:

- `00-polylogue-all.tar(21).gz`: `c895e9096613dada7fa0e8c65dbdc768f88b118d6c096fabb1d5c83215e9515f`
- `00-slightly-stale-context-testsuite-diet.tar(21).gz`: `d4fa7fc31c70ff30db076e526836ffc81b8124d3f54c7add5e58d3f57c9dcbff`
- `04-derived-freshness(1).md`: `72b34413851a796814cf1bddde0821317ed836382dd55857239598b4c4efe072`

## Evidence inspected

The implementation was derived from the mission prompt, `architecture/05-derived-freshness.md`, repository instructions in `CLAUDE.md` and `TESTING.md`, the current production routes, existing tests, generated schema/topology surfaces, Git history around the affected subsystems, and the complete relevant Bead records.

The main Bead authorities were:

- `polylogue-wmsc`: one embedding content-and-recipe invariant, shared selection semantics, and generation-conditional success/error writes;
- `polylogue-1xc.12`: FTS rowid/block/source/recipe identity, exact reconciliation, and the warning that equal counts cannot detect rowid reuse;
- `polylogue-303r.7`: complete embedding model-effect recipe identity;
- `polylogue-f2qv.5`: provider-usage self-healing through the session insight materializer;
- `polylogue-1dk1`: the already-existing orphan embedding reconciliation boundary, kept separate from this work.

Relevant source/history included the archive embedding materializer and write tier, all archive embedding selectors, the FTS DDL/trigger/lifecycle/startup/repair paths, the insight materialization registry and rebuild path, pricing catalog seeding, daemon convergence, schema bootstrap policy, and prior history including FTS trigger consolidation (`74f045dd0`) and embedding lifecycle work (`4177544ce`).

## Mechanism

### Shared typed identity

`polylogue/storage/derivation_identity.py` adds immutable `DerivationKey` and `DerivationAttempt` values. A key contains only:

- logical subject and grain;
- exact source identity;
- complete computational recipe identity;
- output contract.

Generation and attempt identity are captured separately. Eligibility, privacy, retention, producer/resource data, retry policy, and output integrity remain outside the key. `canonical_identity()` produces stable namespaced SHA-256 identities from explicitly named fields. `attempt_matches_current()` is the common terminal-write predicate.

### Archive embeddings

A new embeddings-owned freshness module builds the exact source identity from ordered embeddable `message_id`/`content_hash` pairs. Its recipe declares canonicalization, selector, chunking, provider, model revision, dimensions, task input type, normalization, tool implementation, and input/schema versions. The output contract names SQLite vector float encoding and dimensions.

`embeddings.db` schema version advances from 2 to 3 and gains:

- `embedding_derivation_state`, one desired/current state per session;
- `embedding_derivation_attempts`, durable attempt receipts keyed by attempt ID and generation.

Every production archive attempt captures its key and generation before provider computation. Before either success or failure is committed, the current source/config key is refreshed. Success performs the vector/meta replacement and currentness promotion in one transaction only when the captured key, generation, and active attempt still match. Failure records compatibility status and the failure lifecycle only when the same predicate succeeds. A superseded worker records its attempt as `superseded` but cannot clear, degrade, reopen, or supersede newer debt.

All archive selectors now use the same source-aware stale relation. The old `include_stale_checks=False` escape no longer bypasses semantic freshness, and metadata is left-joined so an equal-count replacement with a different message identity is selected. Backlog, convergence, manual archive embedding, and preflight therefore consume the same pending relation.

`needs_reindex` remains a projection for existing APIs and operators. Retryability remains independent: a current retryable failure stays pending; a current non-retryable failure is degraded; neither disposition authorizes a stale attempt.

### Message FTS

Index schema version advances from 37 to 38. The patch adds an FTS-owned rebuildable ledger:

`messages_fts_identity(rowid PRIMARY KEY, block_id UNIQUE, source_hash, recipe_id, output_contract)`.

The recipe identity covers the `blocks.search_text` projection, `pl_fold` implementation version, tokenizer configuration, and FTS output contract. Production insert/update/delete triggers maintain the contentless FTS table and identity ledger together, including empty-text transitions.

Readiness no longer trusts table counts or trigger names. It verifies the exact persisted trigger programs for the observed `blocks` schema and performs a stable read-transaction comparison among:

- desired non-empty blocks: rowid, block ID, source hash, recipe, output contract;
- observed identity ledger rows;
- FTS `docsize` row presence.

The resulting invariant distinguishes missing rows, excess rows, rowid/block mismatches, source mismatches, and recipe/output mismatches. Equal-count rowid reuse therefore fails. Full and targeted repair paths rebuild FTS and identity rows together. Targeted repair does not claim global readiness; startup and periodic exact reconciliation record a global snapshot only after the full relation has been checked. Same-named legacy trigger programs cannot authorize a previously recorded ready state.

### `provider_usage` insight

The existing `provider_usage` materializer depended on more than session content: provider event facts, message token fallback rows, the materializer definition, and the packaged pricing catalog. The patch adds an exact source-and-recipe receipt to the existing domain table rather than creating a new global ledger.

The source identity hashes all provider usage event fields and message usage fields read by the materializer in deterministic order. The recipe includes the materializer version, provider-event fold, token-lane mapping, message fallback, model resolution, pricing catalog hash, pricing algorithm, and input schema. Source-table triggers mark an existing receipt pending and advance its generation. Rebuild computes and stamps the exact current key in the same SQLite write transaction as the materialized usage rows. Daemon stale selection now checks the exact receipt, so a pricing catalog change or source fact change cannot leave an old usage projection current merely because its row count or materializer version matches.

## Decisions and boundaries

The following decisions are intentional:

1. There is no universal derivation table. Embeddings use their own state/attempt tables, FTS uses a rebuildable identity ledger and freshness projection, and `provider_usage` extends its existing insight materialization receipt.
2. `output_hash` is result integrity evidence, not a lookup key.
3. Authorization and retention changes can deny work without changing computational identity.
4. Index and embeddings schema bumps follow Polylogue’s derived-tier fresh-first policy. Existing v37 index tiers and v2 embeddings tiers must not be silently treated as v38/v3.
5. The existing orphan embedding reconciler remains the deletion/rebuild-generation boundary. This patch handles identity-present content/recipe replacement and terminal ordering; it does not create a competing orphan lifecycle.
6. The generic non-archive `embed_session_sync` compatibility route remains outside the archive derivation ledger. The production archive route always supplies a captured attempt. Direct compatibility calls to `mark_session_embedding_error()` and `record_embedding_failure()` without an attempt retain legacy behavior for callers that do not participate in the archive freshness protocol.

## Changed files

Shared identity and embedding freshness:

- `polylogue/storage/derivation_identity.py`
- `polylogue/storage/embeddings/freshness.py`
- `polylogue/storage/embeddings/materialization.py`
- `polylogue/storage/embeddings/preflight.py`
- `polylogue/storage/sqlite/archive_tiers/embedding_write.py`
- `polylogue/storage/sqlite/archive_tiers/embeddings.py`
- `polylogue/daemon/embedding_backlog.py`
- `polylogue/daemon/convergence_stages.py`
- `polylogue/daemon/cli.py`

FTS identity, repair, startup, and schema:

- `polylogue/storage/fts/sql.py`
- `polylogue/storage/fts/freshness.py`
- `polylogue/storage/fts/fts_lifecycle.py`
- `polylogue/storage/repair.py`
- `polylogue/daemon/fts_startup.py`
- `polylogue/storage/sqlite/archive_tiers/index.py`
- `polylogue/storage/sqlite/archive_tiers/write.py`

Provider usage freshness:

- `polylogue/storage/insights/session/provider_usage_freshness.py`
- `polylogue/storage/insights/session/rebuild.py`
- `polylogue/storage/insights/session/runtime.py`
- `polylogue/storage/sqlite/archive_tiers/pricing_seed.py`

Documentation and topology:

- `docs/schema.md`
- `docs/internals.md`
- `devtools/build_topology_projection.py`
- `docs/plans/topology-target.yaml`
- `docs/topology-status.md`

Tests:

- `tests/unit/storage/test_derived_freshness_survivor.py`
- `tests/unit/storage/test_archive_tiers_embedding_write.py`
- `tests/unit/storage/test_embedding_contracts.py`
- `tests/unit/storage/test_embedding_needs_reindex_race_evidence.py`
- `tests/unit/storage/test_fts_bloat_invariants.py`
- `tests/unit/storage/test_repair.py`
- `tests/unit/storage/test_schema_policy_contracts.py`
- `tests/unit/storage/test_session_insight_refresh.py`
- `tests/unit/daemon/test_convergence_stages.py`
- `tests/unit/daemon/test_daemon_cli.py`
- `tests/unit/daemon/test_embedding_convergence_progress.py`

## Acceptance matrix

| Requirement | Result | Evidence |
| --- | --- | --- |
| Typed key separates source, recipe, output, and subject from generation/lifecycle | Implemented | `derivation_identity.py`; strict new-module mypy check |
| No universal derivation lifecycle | Implemented | Three domain-owned receipt mechanisms |
| Archive embedding selectors share one exact stale relation | Implemented | Materialization selector, backlog, convergence, preflight tests |
| Every declared embedding recipe field changes identity | Implemented | Field-by-field survivor test |
| Eligibility is outside computational identity | Implemented | Recipe-component contract test |
| Old success cannot clear or reopen newer debt | Implemented | Direct state witness and production archive writer witness |
| Old terminal error cannot degrade or clear newer debt | Implemented | Direct state witness and production `record_embedding_failure` witness |
| Same-count message replacement is selected | Implemented | Left-join source relation witness |
| FTS equal-count rowid reuse is detected | Implemented | Real-trigger independent relation plus reuse/repair witness |
| FTS source/recipe/trigger-program drift is detected | Implemented | Exact ledger, recipe mismatch, same-name legacy-trigger tests |
| Targeted repair cannot claim global readiness early | Implemented | Repair/startup/convergence tests |
| Provider usage source or price recipe change marks pending | Implemented | Source-trigger and pricing-catalog receipt witness |
| Rebuild stamps current provider usage receipt atomically | Implemented | Session insight rebuild route tests |
| Restart/startup sees exact FTS state | Implemented for synthetic archives | Daemon startup tests; no operator live archive run |
| Full Bead `wmsc` live census categories | Not included | Requires a separate status/census surface pass |
| Full Bead `1xc.12` Prometheus gauges, bounded `ops.db` history, and Hypothesis state machine | Not included | The core identity/repair survivor slice is implemented; observability/property expansion remains separate |

## Apply order

1. Adjudicate and integrate the prerequisite `testdiet-02` and `testdiet-03` foundations. This patch was validated only against the named base commit, so resolve any overlapping schema, convergence, or rebuild-survivor hunks before applying it.
2. Apply `PATCH.diff` to `b9052e09103502017c0f510ecc699aac395de23c` with `git apply --check` followed by `git apply`.
3. Regenerate/rebuild the derived index tier at schema v38 and embeddings tier at schema v3 according to the repository’s fresh-first schema policy. Do not mark old tiers current by editing `user_version`.
4. Run the focused commands in `TESTS.md`, then the repository-pinned `devtools verify --quick` and affected testmon gate in the integration worktree.
5. Deploy only after startup exact reconciliation and daemon convergence are observed on a representative rebuilt archive. Keep the previous derived generation available until postflight proves the new receipts and search/readiness state.

## Risks

The primary operational risk is rebuild cost: both schema versions change, and large archives must create the FTS identity ledger and embedding derivation receipts. Exact FTS reconciliation scans the desired and observed relations when a proof is required; readiness metrics continue to consume recorded state rather than scanning on every scrape, but startup/periodic proof should be measured on the operator archive.

The patch touches central write and repair paths and may overlap prerequisite lanes. Integration conflicts must be resolved semantically, not by choosing one side wholesale. In particular, preserve any newer generation/publication seams from testdiet-02/03 while retaining this patch’s exact key and conditional terminal predicates.

A compatibility caller that invokes embedding terminal helpers without a captured attempt still uses legacy status semantics. That is intentional for non-archive callers, but new archive production call sites must always pass an attempt.

Pricing catalog identity is derived from the packaged catalog. Any future dynamic catalog source must supply a stable catalog snapshot identity rather than silently bypassing this recipe field.

## Verification performed and remaining

Completed verification is detailed in `TESTS.md`. The final worktree has 291 passing tests across disjoint modified daemon, embedding, FTS/repair/schema, insight, and survivor groups. The nine-test survivor module also passed after applying `PATCH.diff` to a clean detached worktree. Ruff formatting, Ruff lint, whitespace validation, strict type checking of the three new modules, patch application, and byte-for-byte patch/content comparison passed. The relevant topology renderers were rerun twice and produced identical hashes.

Not performed: the complete repository test suite, the repository-pinned full mypy/`devtools verify` environment, a full `render all --check` completion, Nix/deployment checks, a live daemon run, a live archive rebuild, large-archive resource measurement, Prometheus/ops history validation, or integration with testdiet-02/03. A full `render all --check` attempt passed the CLI reference, CLI output schemas, OpenAPI, and devtools reference checks before the external 180-second command budget ended during later unrelated rendering; the task-relevant topology renderers were independently stable.

## Value of another iteration

A small repair iteration has low expected value for the implemented invariant. The highest-value small additions would be a Hypothesis state machine over the real FTS triggers and a repository-pinned full verification run after dependency integration.

A substantial second pass would be justified only to close the broader open Bead scope: add the bounded embedding freshness census categories, FTS Prometheus gauges and retained `ops.db` drift samples, large-archive reconciliation measurements, and live migration/postflight evidence. That is materially more work than polishing this patch and should be treated as a separate observability and rollout tranche.
