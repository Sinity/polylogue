# Source, Beads, and history evidence

## Snapshot authority

The Chisel manifest reports:

- project: `polylogue`;
- source: `/realm/project/polylogue`;
- generated: `2026-07-18T013442Z`;
- branch: `master`;
- commit: `bf8191b3f56aa40da8f271df7f3385c712825497`;
- dirty: `true`.

`origin/master` resolves to the same commit and the bundled branch-delta patch/log are empty. The source authority is therefore the commit plus its dirty worktree, not an unmerged branch history.

The dirty worktree diff contains only:

- `polylogue/archive/query/unit_results.py` — replaces an unsafe literal cast with explicit field narrowing;
- `polylogue/daemon/http.py` — narrows a socket before `select.select`;
- `polylogue/hooks/__init__.py` — replaces a cast from an environment string with explicit supported harness values.

The reconstructed dirty patch is 2,458 bytes, SHA-256 `c5cf563e1cedb52a9e5d4b780eb09d48fa28ed606cd8c680926b3701215de4fe`. None of those edits is included as a new lane change in `PATCH.diff`.

## Native substrate findings

### Query definitions and materialized evidence already exist

`polylogue/storage/sqlite/query_objects.py` owns content-addressed canonical query definitions, named query refs, immutable result-set manifests and members, retained query runs, watched baselines, and query evaluation receipts. `polylogue/core/query_identity.py` supplies protocol-versioned canonical query identity. This made the query-object module the native storage layer for analysis definitions and run receipts; a separate report registry would duplicate durable identity and provenance behavior.

`polylogue/archive/query/production_evaluator.py`, `polylogue/daemon/convergence_standing_queries.py`, and `tests/unit/daemon/test_standing_queries_default_evaluator.py` show that watched queries can reach production materialization and evaluation receipts without a fake evaluator. The synthetic real-route test reuses that route.

### Assertions own claims, not procedures or executions

`polylogue/storage/sqlite/archive_tiers/user_write.py` persists findings through the unified assertion lifecycle. `polylogue/storage/sqlite/finding_provenance.py` reconstructs finding evidence and staleness from structured refs. This is the correct home for a materialized finding, but not for the full analysis definition or execution receipt.

The user-tier source comment and `polylogue-rxdo.8` both encode the same separation: settings/procedure/execution state must not be forced into claims merely to avoid a table.

### Object refs and product reads already have extension points

`polylogue/core/refs.py` already registered `analysis` among analysis-provenance ref kinds. The smallest grammar extension is one sibling `analysis-run` kind.

`polylogue/api/archive.py` already dispatches `resolve_ref()` by object-ref kind and `polylogue/surfaces/payloads.py` owns bounded typed payloads. Adding typed definition/run reads there gives the new archive objects a real consumer without creating speculative list or report APIs.

### Public claims are not in the authoritative source tree

Commit `922aa2297a93b976ad8d690844a9510fa19dd1d8` exists on another bundled ref with subject `feat(legibility): render public claims from FINDING assertions, not a hand ledger`. Its relevant paths include:

- `polylogue/insights/measurement/public_claims.py`
- `polylogue/storage/sqlite/archive_tiers/user_write.py`
- `polylogue/storage/sqlite/finding_provenance.py`
- public-claims tests and devtools renderers.

Those files/semantics were inspected as history, not treated as source authority. The patch does not invent their API. The integrating lane should rebase the two overlapping storage files if that branch lands first and make the existing public-claims consumer read the structured `analysis_run_ref`.

## Why a durable migration is unavoidable

Before this patch, authoritative source reports `USER_SCHEMA_VERSION = 9`. Existing user migrations are numbered and contiguous through:

- `007_query_objects.sql`
- `008_query_provenance_hardening.sql`
- `009_query_holdout_policy.sql`

The migration loader/policy requires durable versions to advance one step at a time and requires verified backup authorization unless a migration explicitly carries an allowed no-backup marker. Fresh canonical DDL is separately defined in `archive_tiers/user.py` and migration tests compare it with upgraded databases.

A content-addressed analysis definition and immutable run receipt cannot be represented faithfully by an existing query row because they are not query plans, and cannot be represented faithfully by an assertion because they are procedure/execution objects rather than claims. A run also needs a normalized one-to-many relation to exact named query/result/evaluation inputs. Embedding those inputs as opaque assertion JSON would lose relational binding, foreign-key integrity, raw SQL enforcement, and reusable typed reads.

The smallest justified durable extension is therefore:

- `analysis_definitions` — immutable content-addressed definition;
- `analysis_runs` — immutable receipt header and run-level stamps;
- `analysis_run_inputs` — exact named/ordered evidence bindings.

The patch supplies both numbered migration DDL and fresh DDL parity. It does not claim rollout admission.

## Beads findings

### `polylogue-rxdo.8` — analysis recipes/runs

Status: open, priority 3, updated `2026-07-14T15:17:14Z`.

Load-bearing findings:

- analyses are interactive DAGs;
- recipes/definitions are procedure and runs are execution state;
- assertions are claims and must not substitute for those objects;
- YAML is import/export serialization only;
- runs must cite query runs/batches/evidence and a later rerun must remain diffable;
- implementation was previously deferred because no durable user-tier window had been declared.

The patch follows the object distinction and rerun-history requirement. It does not add YAML import/export because the mission is the reusable archive representation and real-route proof, not the full recipe composer lifecycle.

### `polylogue-60i5` — durable-tier change train

Status: open, priority 1, updated `2026-07-14T23:24:27Z`.

Load-bearing requirements:

- derive the shipped version from source/live state rather than stale notes;
- admit only stabilized typed protocols with runtime read/write wiring and behavioral proof;
- reserve one slot and one writer;
- land one contiguous numbered migration plus fresh-DDL parity;
- authorize against an authenticated verified backup;
- apply under stopped-daemon/single-writer authority;
- prove integrity, row parity, restart, and runtime convergence;
- late riders move to the next train.

The package satisfies the implementation-side prerequisites but cannot reserve/admit the operator's live slot. It therefore labels v10 as a draft slot and requires renumbering if occupied.

### `polylogue-37t.14` — shared evidence-integrity evaluator

Status: open, priority 1, updated `2026-07-15T19:23:11Z`.

This record says one provider-neutral evaluator must own support, partial support, stale evidence, cycles, closed loops, unresolved refs, frame incompleteness, private-held evidence, and grounding compatibility. Existing assertion/finding/query/public stores remain fact owners.

Consequently this patch only records and resolves exact evidence ancestry. It does not interpret a resolvable finding as supported. The existing provenance reader's `current` verdict is retained with that limited meaning.

### `polylogue-rxdo.2` — privacy/retention/excision

Status: open, priority 1, updated `2026-07-15T18:14:51Z`.

The record says promoted query evidence still lacks complete privacy classification, retention, and excision behavior. The analysis run therefore records explicit privacy and retention/excision stamps in its receipt identity, but this patch does not pretend that recording a policy ref implements the policy actuator.

### `polylogue-rxdo.3`, `.4`, and `.5`

- `.3`: evaluation-world/query-run envelopes across all surfaces remain incomplete; the analysis receipt records the world needed for its own immutable identity without claiming cross-surface completion.
- `.4`: findings/public claims are future consumers of stronger evidence verdicts; this patch supplies structured run provenance for that future consumer.
- `.5`: production default standing-query evaluator injection is landed. Live daemon ingest-to-converger activation and scoped dependency epochs remain open; the authored test directly invokes the real convergence stage.

## History findings

| Commit | Date | Relevance |
|---|---|---|
| `88155768a0578a3b50c8577153f3643350ebc1fc` | `2026-07-12` | Added analysis-provenance object-ref kinds. |
| `61a2808d42ada503c0514d3858ad5c27110c7cf2` | `2026-07-13` | Established finding candidates as assertion-backed claims. |
| `a952221cdcc4813ffcc4c9c18c4fd8981d5bbb2a` | `2026-07-13` | Materialized watched-query relations and baselines. |
| `89166362b9aee8c304b27a69f68ec1b74606f634` | `2026-07-14` | Added production canonical-plan evaluation and finding provenance. |
| `922aa2297a93b976ad8d690844a9510fa19dd1d8` | `2026-07-18` | Adds public-claims rendering on another ref; not authoritative source. |

The progression supports extending the query/evaluation substrate for procedure/execution identity and keeping materialized claims in assertions.

## Contradictions and stale evidence

`AGENTS.md`, `CLAUDE.md`, and `docs/schema.md` describe `user.db` as schema v6. Authoritative executable source is already v9 before this patch, with migrations 007–009 and tests/status fixtures for that state. Current source wins. The package does not use those stale prose versions to select a slot and does not broaden scope into rewriting the repository's full schema documentation.

Older `polylogue-rxdo.8` design language names `analysis_recipes`, while the mission asks for a content-addressed analysis definition. The patch uses `analysis_definitions` because it is the protocol identity being archived and because no recipe runtime/composer exists in authoritative source. It preserves the important type boundary from the Bead: definition/procedure and run/execution are DB objects, findings are assertions.

The Bead's old phrase “batch with v5” is stale. The executable shipped source is v9 and `polylogue-60i5` explicitly says to re-derive the window rather than trust stale version labels.

## Evidence deliberately not inspected or generated

- No private live archive was opened.
- No private claim-vs-evidence report was regenerated.
- No operator daemon, browser, deployment, secrets, or database backup was accessed.
- No classifier/calibration semantics were changed.
- No live support verdict was inferred from synthetic evidence.
- No repository tests were run.

## Patch evidence

`PATCH.diff` is a unified binary-capable diff against the authoritative snapshot baseline. At package construction it was 113,601 bytes with SHA-256 `85609cb2230b50646ba1196ce54b71d6c95ed3835890df83eaf58442e44bf1c1`.

It contains exactly 15 paths and no supplied archive, manifest, Beads export, mission prompt, or copied snapshot file. A detached baseline apply check and byte-for-byte comparison of all resulting paths passed.
