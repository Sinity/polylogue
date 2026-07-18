# Lane A handoff — first-party analysis evidence kernel

## Mission and result

This package implements a reusable archive-native representation for analysis evidence in Polylogue. It does not recreate a private claim-vs-evidence report and it does not alter classifier semantics.

The implementation adds:

1. a content-addressed analysis definition with stable identity;
2. an immutable analysis-run receipt whose identity covers the exact query result sets, query evaluation receipts, archive generations, evaluator, evaluation world, privacy stamp, retention/excision stamp, protocol, and run timestamp;
3. materialized analysis findings that remain ordinary `AssertionKind.FINDING` assertions but are rejected unless they cite a persisted analysis run that binds the same query/result-set evidence; and
4. typed public reads for `analysis:<hash>` and `analysis-run:ar_<hash>`, plus structured finding provenance that exposes the analysis-run citation rather than treating the finding body as sufficient evidence.

The storage extension is deliberately placed in the existing user-tier query-object/assertion substrate. It is not a report registry, a YAML runner, or an assertion-payload substitute for procedure/execution state.

## Authoritative snapshot identity

The attached Chisel snapshot, not prior branch memory, was treated as authority.

| Item | Identity |
|---|---|
| Manifest source | `/realm/project/polylogue` |
| Manifest generated | `2026-07-18T013442Z` |
| Branch | `master` |
| Git commit | `bf8191b3f56aa40da8f271df7f3385c712825497` |
| Commit subject | `feat: land WebUI v2 scaffold, design system, and generated client (#3074)` |
| Commit parent | `4b574ce66533ac114961a9533ba2f2c4a0c45b83` |
| Commit date | `2026-07-18T02:39:50+02:00` |
| Dirty | yes |
| `origin/master` | `bf8191b3f56aa40da8f271df7f3385c712825497` |
| Branch delta vs `origin/master` | empty |

The authoritative dirty patch contains only:

- `polylogue/archive/query/unit_results.py`
- `polylogue/daemon/http.py`
- `polylogue/hooks/__init__.py`

That dirty patch is 2,458 bytes with SHA-256 `c5cf563e1cedb52a9e5d4b780eb09d48fa28ed606cd8c680926b3701215de4fe`.

For deterministic patch construction only, the authoritative worktree state was committed locally as synthetic baseline `291c57effbb8483d39e08cc5e215fa9f35819fdf` (`chisel-authoritative-baseline`). That commit has parent `bf8191…` and contains exactly the three pre-existing dirty paths above. `PATCH.diff` is therefore against the complete named snapshot state—HEAD plus its authoritative dirty patch—and does not reproduce or overwrite that dirty patch.

## Evidence inspected before choosing the mechanism

Repository instructions and schema policy:

- `AGENTS.md`
- `CLAUDE.md`
- `docs/schema.md`
- `polylogue/storage/sqlite/archive_tiers/bootstrap.py`
- `polylogue/storage/sqlite/archive_tiers/migrations.py`
- `polylogue/storage/sqlite/migration_policy.py`
- `polylogue/storage/sqlite/migrations/user/007_query_objects.sql`
- `polylogue/storage/sqlite/migrations/user/008_query_provenance_hardening.sql`
- `polylogue/storage/sqlite/migrations/user/009_query_holdout_policy.sql`
- `polylogue/storage/sqlite/archive_tiers/user.py`

Native query/evaluation/assertion/public-read route:

- `polylogue/core/query_identity.py`
- `polylogue/core/refs.py`
- `polylogue/storage/sqlite/query_objects.py`
- `polylogue/archive/query/production_evaluator.py`
- `polylogue/daemon/convergence_standing_queries.py`
- `polylogue/daemon/convergence_stages.py`
- `polylogue/storage/sqlite/archive_tiers/user_write.py`
- `polylogue/storage/sqlite/finding_provenance.py`
- `polylogue/surfaces/payloads.py`
- `polylogue/api/archive.py`
- `tests/unit/daemon/test_standing_queries_default_evaluator.py`
- existing storage, migration, assertion, facade, and CLI status tests modified by this patch

Beads records:

- `polylogue-rxdo.8` — analysis recipes/runs must be DB-native procedure/execution objects; assertion payloads and YAML-only state are explicitly rejected.
- `polylogue-60i5` — all durable-tier evolution must use one declared/reserved change train, a contiguous numbered migration, fresh-DDL parity, authenticated backup authorization, stopped-daemon/single-writer apply, and restart/convergence proof.
- `polylogue-37t.14` — the shared support/drift/cycle/grounding evaluator remains open; this patch does not invent those semantics.
- `polylogue-rxdo.2` — full privacy/retention/excision behavior for promoted query evidence remains open.
- `polylogue-rxdo.3` — cross-surface evaluation-world/query-run envelopes remain incomplete.
- `polylogue-rxdo.4` — finding/public support verdict consumer context.
- `polylogue-rxdo.5` — production default standing-query evaluator is landed; live ingest-loop activation and scoped epoch work remain open.

Relevant history:

- `88155768a0578a3b50c8577153f3643350ebc1fc` — analysis-provenance object-ref kinds.
- `61a2808d42ada503c0514d3858ad5c27110c7cf2` — finding candidates persisted through assertions.
- `a952221cdcc4813ffcc4c9c18c4fd8981d5bbb2a` — watched query relation materialization.
- `89166362b9aee8c304b27a69f68ec1b74606f634` — production evaluator and finding provenance.
- `922aa2297a93b976ad8d690844a9510fa19dd1d8` — public claims rendered from findings; this commit exists on another ref and is not present in the authoritative source tree.

## Mechanism

### Content-addressed definitions

`put_analysis_definition()` canonicalizes a JSON-object definition under protocol `polylogue.analysis-definition.v1`. Canonicalization normalizes Unicode to NFC, rejects non-string or normalization-colliding object keys, rejects non-finite floats, and hashes the protocol-wrapped canonical JSON. Equivalent key ordering and canonically equivalent Unicode produce the same `analysis:<sha256>` identity. Reads recompute and verify the content hash and protocol metadata.

### Immutable run receipts

`put_analysis_run()` accepts one or more named `AnalysisRunInput` values. Every input must resolve to an existing query object, result-set manifest, and query evaluation receipt, and the result set and receipt must bind the same query/result-set pair. Inputs are sorted by normalized input key before hashing.

The `analysis-run` receipt digest covers:

- analysis definition hash;
- each exact input key, query hash, result-set ID, and evaluation-receipt ID;
- source, user, and index archive generations;
- evaluator ref;
- evaluation frame, runtime build, model refs, and parameter document;
- privacy classification, policy ref, raw-content flag, and detail document;
- retention class, retention policy, excision policy, and detail document;
- receipt protocol version; and
- `created_at_ms`.

Repeating the exact receipt is idempotent. Any changed bound component creates another `analysis-run:ar_<sha256>` row. Existing rows are not upserted or overwritten. Typed reads reconstruct the canonical receipt and verify its ID.

### Findings remain claims, with mandatory evidence

The patch adds generic `finding_kind="analysis"` to the existing assertion lifecycle. Such a finding must provide:

- a registered `query:<hash>` ref;
- a registered `result-set:<id>` ref; and
- a registered, persisted `analysis-run:<receipt>` ref.

Before writing the assertion, production code verifies that the cited run contains the exact query/result-set pair. The run ref is included in the assertion value and evidence refs. A missing run citation, missing run, malformed ref kind, or mismatched result set fails closed.

The finding's public payload remains structured provenance. The finding body is not exposed as its evidence payload. The resolver reports the query, result set, and analysis-run evidence as separately resolvable refs.

### Typed consumer

The only new product-facing read seam is the existing `Polylogue.resolve_ref()` route:

- `analysis:<hash>` returns `payload_kind="analysis-definition"`.
- `analysis-run:ar_<hash>` returns `payload_kind="analysis-run-receipt"` with exact inputs and stamps.
- `finding:<assertion-id>` now includes `analysis_run_ref` in `payload_kind="finding-provenance"`.

No list/report registry or speculative command surface was added.

## Durable migration classification

**Classification: additive durable user-tier migration, draft `user.db` v9 → v10.**

A durable migration is necessary because the current architecture and `polylogue-rxdo.8` explicitly distinguish:

- assertions as claims;
- analysis definitions as procedures; and
- analysis runs as execution records.

The existing assertion payload cannot safely own immutable procedure/execution identity without violating that boundary. Existing query tables cannot represent an analysis definition or a run that binds multiple named query result/evaluation inputs plus evaluator and policy stamps. The smallest native extension is therefore two parent tables plus one exact-input child table beside the query-object substrate.

The migration is intentionally not marked `additive-no-backup`; it requires the repository's verified user-tier backup authorization. Fresh DDL and migration DDL are kept in parity.

**Rollout gate:** this package does not claim that user slot 010 has been admitted by `polylogue-60i5`. Before local integration, the durable change-train owner must re-derive the shipped live version, reserve/admit the slot and rider, and either accept `010` or renumber this migration if another lane has occupied v10. This is a deployment coordination limitation, not an excuse to hide the required schema.

## Changed files

| Path | Purpose |
|---|---|
| `polylogue/storage/sqlite/migrations/user/010_analysis_evidence_kernel.sql` | Additive v10 migration for definitions, runs, exact inputs, indexes, binding triggers, and update immutability. |
| `polylogue/storage/sqlite/archive_tiers/user.py` | Fresh user-tier DDL parity and `USER_SCHEMA_VERSION = 10`. |
| `polylogue/storage/sqlite/query_objects.py` | Canonical definition/run types, content hashing, exact receipt persistence/read verification, and evaluation-receipt reads. |
| `polylogue/core/refs.py` | Register `analysis-run` as an object-ref kind. |
| `polylogue/storage/sqlite/archive_tiers/user_write.py` | Require and verify run evidence for generic analysis findings. |
| `polylogue/storage/sqlite/finding_provenance.py` | Resolve and expose `analysis_run_ref` as finding evidence. |
| `polylogue/surfaces/payloads.py` | Typed definition, run, input, generation, evaluation-world, privacy, and retention payloads. |
| `polylogue/api/archive.py` | Public typed resolvers for definitions/runs and structured finding provenance wiring. |
| `tests/unit/api/test_analysis_evidence_kernel.py` | Privacy-safe synthetic real-route fixture and end-to-end acceptance test. |
| `tests/unit/storage/test_query_objects.py` | Stable identity, idempotency, component-isolated receipt hashing, immutable history, and read verification coverage. |
| `tests/unit/storage/test_durable_migrations.py` | v10 backup/migration/fresh-DDL parity and raw SQL trigger enforcement. |
| `tests/unit/storage/test_archive_tiers_assertions.py` | Fresh user-tier object/trigger inventory. |
| `tests/unit/api/test_facade_contracts.py` | Move `analysis` out of pending-ref behavior. |
| `tests/unit/cli/__snapshots__/test_plain_cli_snapshots.ambr` | Current user-tier version status v10. |
| `docs/plans/topology-target.yaml` | Regenerated topology projection after production changes. |

`FILES/` is omitted because the unified patch fully disambiguates every change.

## Acceptance matrix

| Requirement | Production mechanism | Proof authored | Verification status |
|---|---|---|---|
| Definition has stable identity | NFC/protocol canonical JSON and SHA-256 identity; idempotent insert; read-side hash verification | `test_analysis_definitions_and_runs_are_content_addressed_immutable_history`; real-route equivalent Unicode/key-order definition | Standalone production acceptance probe passed; test suite not run |
| Run binds exact inputs, evaluator/world, privacy/retention, and archive generations | `analysis_runs` + `analysis_run_inputs`; existing query/result/evaluation FKs; SQL evidence-match trigger; digest covers every receipt component; read-side digest verification | Component-isolation assertions change generation, evaluator, world, privacy, retention, and input independently | Standalone storage/public probe passed for exact bound inputs; authored test suite not run |
| Finding cannot be read as evidence-free prose | Analysis finding writer requires a run and exact query/result membership; run added to evidence; finding resolver returns provenance and omits body prose | Missing-run and mismatched-run rejection in real-route test; public resolver assertions | Standalone acceptance probe rejected missing/mismatched evidence and returned three resolvable refs with no body text |
| Changed corpus/evaluator creates a distinguishable receipt without overwriting history | Content-addressed INSERT-only receipts; changed input/evaluator/world/generation hashes differently | Real convergence route ingests a second synthetic session and changes evaluator; storage test asserts eight component/input variants | Standalone acceptance probe produced two distinct, concurrently readable run refs; full real-route test not run |
| Migration follows native durable route | Numbered `010` migration, fresh DDL parity, version/status fixture update, backup-required migration policy | Migration/fresh schema parity and raw SQL trigger tests | Fresh v10 initialization, integrity, and FK probes passed; admission/live migration unverified |
| Typed read exists only for a real consumer | Existing public `resolve_ref()` gains definition/run payloads; finding consumer gains run provenance | Facade and real-route assertions | Runtime resolver probe passed after repairing a Pydantic recursive-alias import defect |

## Apply order

1. Confirm the target worktree is the attached snapshot state: commit `bf8191…` plus the three-path dirty patch with SHA-256 `c5cf…`.
2. Re-derive the current local `user.db` shipped/live version and inspect the active `polylogue-60i5` durable change-train manifest.
3. Reserve and admit this rider. If slot 010 is occupied, rename `010_analysis_evidence_kernel.sql`, change `USER_SCHEMA_VERSION`, update version assertions/snapshot, and regenerate the patch/topology as one coherent train.
4. Apply `PATCH.diff` from the repository root with `git apply --check PATCH.diff && git apply PATCH.diff`.
5. Run the focused tests and gates listed in `TESTS.md` before merge.
6. Produce an authenticated verified backup receipt for the exact live `user.db` bytes.
7. Stop the daemon and all other writers; apply the one admitted user-tier migration under single-writer authority.
8. Verify pre/post versions, `PRAGMA integrity_check`, `PRAGMA foreign_key_check`, row parity for existing tables, fresh-DDL parity, and the definition/run/finding behavior.
9. Restart and prove schema/runtime convergence and typed `resolve_ref()` reads.

## Verification performed

No pytest/devtools test command was run, as required by the task contract. The authored tests remain for the integrating lane.

The following non-test checks were performed against the final implementation:

- `ruff format --check` over all 12 changed Python files: passed.
- `ruff check` over all 12 changed Python files: passed.
- strict `mypy` over all seven changed production modules: passed with no issues.
- Python byte-compilation over all changed Python files: passed.
- `devtools render topology-projection --check`: passed; 1,034 rows, nine pre-existing TBD ownership entries.
- `devtools render topology-status --check`: passed; no status-file drift.
- `git diff --check chisel-authoritative-baseline`: passed.
- import/runtime construction of the new payload models: passed.
- fresh production user-tier initializer: v10, required tables/triggers present, `PRAGMA integrity_check = ok`, zero foreign-key violations.
- standalone production storage/public-resolver acceptance probe: stable definition identity; missing and mismatched run citations rejected; two changed-corpus/evaluator receipts persisted; old receipt remained readable; definition, both runs, and finding resolved with typed payloads; finding exposed no body prose and had three resolvable evidence refs.
- `PATCH.diff` applied cleanly to a separate detached worktree at `chisel-authoritative-baseline`; all 15 resulting file hashes matched the implementation worktree; `git diff --check` passed there.

One initial fresh-schema probe used the system interpreter and stopped before opening a database because that interpreter lacked `aiosqlite`; the same production initializer passed under the repository `.venv`. A later public resolver probe exposed a real Pydantic recursion failure caused by using the recursive `JSONDocument` alias directly as model fields. The patch was corrected to use bounded mapping fields with explicit JSON-document validators, after which import, strict typing, and runtime resolution passed.

## Risks and limitations

- **Migration train not admitted:** v10 is an implementation-ready draft slot, not a claim of live change-train authority. This is the primary integration gate.
- **No live archive verification:** no operator daemon, browser, secrets, deployment, private archive, or live `user.db` was accessed. Live backup, migration, restart, and convergence remain unverified.
- **Tests authored but not executed:** static checks and standalone production probes passed; the repository test runner and full gate remain the integrating lane's responsibility.
- **Public claims branch absent from source authority:** commit `922aa…` modifies `user_write.py` and `finding_provenance.py` on another ref. If it lands first, rebase these two seams and make its renderer consume `analysis_run_ref`; do not duplicate public-claim logic.
- **Support semantics remain out of scope:** `staleness_verdict="current"` means declared refs resolve under the current provenance reader. It is not a proof that the finding is supported. `polylogue-37t.14` remains the authority for shared support, cycle, grounding, and drift verdicts.
- **Privacy/retention enforcement remains out of scope:** the run records explicit stamps. It does not implement retention scheduling, private evidence promotion policy, or excision propagation; `polylogue-rxdo.2` owns that work.
- **Delete semantics are intentionally not blocked by immutability triggers:** updates are prohibited and content is verified on read, while future controlled retention/excision must retain a deletion path. The current foreign keys restrict deleting definitions/runs that still have relational children; assertion evidence refs are not foreign keys, so the future excision actuator must define downstream handling.
- **Documentation version drift exists upstream:** `AGENTS.md`, `CLAUDE.md`, and `docs/schema.md` still describe `user.db` v6, while authoritative source was already v9. This patch updates executable version/status surfaces only rather than expanding scope into a broad documentation repair.
- **Live ingest loop remains unverified:** the synthetic authored test invokes the real convergence stage directly. It does not prove that a deployed daemon calls that stage at the right point in its ingest loop.

## Dominated deletions

None proposed. No existing tests, helpers, schemas, or routes were deleted. The only pending behavior removed is `analysis` from the public pending-ref set because it now has a real resolver.

## Value of another iteration

If slot 010 is free and focused tests reveal only ordinary defects, another iteration is a small repair pass, plausibly adding about 10–20% value through test fixes and local train metadata.

A substantial second pass, plausibly adding 30–50% value, is justified if any of these conditions are true: v10 is occupied and the package must join another migration train; the public-claims branch lands and requires semantic composition; the shared evidence-integrity evaluator becomes available; or retention/excision enforcement must be wired rather than merely stamped. Those are integration changes, not missing scaffolding in this package.
