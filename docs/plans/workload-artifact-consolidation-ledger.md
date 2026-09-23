# Workload artifact consolidation ledger

Census and disposition record for the deterministic workload-artifact
substrate (`tests/infra/workload_artifacts.py`,
`tests/infra/workload_declarations.py`). Measured at `a3d6ffe2b`.

This ledger is a disposition record, not a refactoring plan. Each candidate
below carries a decision and, where deferred, a named owner.

## Owners

| Concern | Canonical owner |
| --- | --- |
| Deterministic workload shape and named profiles | `tests/infra/workload_declarations.py` |
| Artifact identity, publication, lease, clone, GC | `tests/infra/workload_artifacts.py` |
| Production archive write seam | `polylogue/storage/sqlite/archive_tiers/write.py` via `tests/infra/live_ingest.py` |
| Shared storage-record and DB seeding primitives | `tests/infra/storage_records.py` |
| Query-law semantics and pathology vocabulary | `tests/infra/query_contract.py` |

## Census

`tests/infra` recursive, `*.py`, first-parent `master`:

| date | commit | files | LOC | top-level defs | `workload_artifacts.py` | `workload_declarations.py` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2026-08-25 | `037a64f23` | 88 | 26,601 | 902 | 3,245 | — |
| 2026-09-01 | `5be5954e3` | 94 | 28,352 | 969 | 3,349 | — |
| 2026-09-05 | `1a53a1f61` | 95 | 28,486 | 981 | 3,381 | — |
| 2026-09-08 | `521888986` | 101 | 32,043 | 1,076 | 3,542 | — |
| 2026-09-13 | `a3d6ffe2b` | 104 | 33,367 | 1,106 | 3,191 | 509 |

The central hotspot shrank from its 3,542-line peak to 3,191 by extracting 509
lines of declarations into a separate owner, so the workload pair now totals
3,700 against a 3,542 peak in one file. `tests/infra` as a whole grew 25% over
the same window. Consolidation has redistributed the hotspot; it has not yet
reduced the aggregate.

Deleted since the campaign began: `tests/infra/pathology_zoo.py` (commit
`eb8be9427`, dissolved into law-owned material). No `pathology_composer.py`
remains.

### Module depth, fan-in and fan-out

Fan-in counts repository modules importing the module; `fan-out` counts its
`tests.infra` imports.

| module | LOC | fan-in | fan-out | routes through manifest |
| --- | ---: | ---: | ---: | --- |
| `workload_artifacts` | 3,191 | 21 | 2 | canonical |
| `storage_records` | 1,531 | 144 | 1 | primitive beneath it |
| `reference_model` | 1,105 | 4 | 0 | n/a (oracle) |
| `convergence_harness` | 1,035 | 17 | 1 | no |
| `inferred_corpus` | 1,075 | 3 | 0 | n/a |
| `corpus_program` | 1,022 | 1 | 0 | no |
| `whale_fixtures` | 609 | 4 | 1 | partial |
| `source_composer` | 526 | 5 | 1 | n/a (writes no DB) |
| `workload_declarations` | 509 | 17 | 0 | canonical |
| `query_corpus` | 502 | 2 | 2 | no |
| `archive_scenarios` | 456 | 25 | 2 | no |
| `integration_profile` | 270 | 3 | 3 | yes |
| `archive_templates` | 182 | 36 | 1 | yes |
| `live_ingest` | 170 | 73 | 0 | production writer seam |
| `shared_session_archives` | 116 | 3 | 0 | yes |
| `corpus_fixtures` | 110 | 9 | 4 | yes |
| `benchmark_archives` | 52 | 3 | 2 | yes |

## Dispositions

| candidate | finding | disposition | owner |
| --- | --- | --- | --- |
| `archive_scenarios.py` — `ArchiveScenario.seed`, `seed_archive_scenarios`, `seed_workspace_scenarios` | Originally recorded as a "true direct-index seeder" | **Refuted — see the retirement pass.** `seed` writes no SQL of its own; it delegates to `SessionBuilder` and `ArchiveStore.add_user_tags`, and the "25 importers" count is fan-in on two pure helpers. | itself |
| `whale_fixtures.py` — `acquire_codex_revision_chain`, `copy_sqlite_database` | Drives real `AcquisitionService`; clones via `archive_templates` but publishes no manifest | **Refused with a reason — see the retirement pass.** Wrapping a builder that deliberately uses the production acquisition seam is a design change, not a deletion. | itself |
| `query_corpus.py` — `build_query_corpus` | Successor to the retired `pathology_zoo`; writes via `live_ingest.write_index_session` | **Keep.** Not a duplicate: it materializes into a caller-supplied root through the production writer seam and caches nothing, so there is no artifact identity to own. Its pathology vocabulary is law-owned (`query_contract.REQUIRED_PATHOLOGIES`). | `query_contract.py` |
| `corpus_program.py` | Opens `SQLiteBackend`/`ArchiveStore` directly | **Keep.** Explicitly delegates effects to production acquisition, parsing, convergence and hook seams; owns no cached artifact. | itself |
| `live_ingest.py` | Raw `sqlite3` connections into `index.db` | **Not a duplicate.** This is the single test seam onto `write_parsed_session_to_archive`, the production choke point. Retiring it would create duplication, not remove it. | itself |
| `storage_records.py` — `SessionBuilder` | Widest seeding primitive (fan-in 144) | **Keep.** The primitive the manifest route is built on, not a competitor to it. | itself |
| `reference_model.py` | Matched an archive-construction grep | **Not a duplicate.** A declared oracle that builds no archive. | itself |
| `convergence_harness.py` | Adapts production writers and ops ledger | **Keep.** Owns no alternate convergence state machine by its own contract. | itself |
| `source_composer.py` | Matched on `session_count` | **Not a duplicate.** Composes in-memory arrangements and writes no database; `session_count` is a bundle-shape parameter, not a scale tier. | itself |
| `integration_profile._SCALE_MINIMUM_MESSAGES` | `smoke`/`representative`/`archive-shaped`/`stress` minimums | **Keep.** Tier-naming residue used only as a validation gate over `CorpusSpec`s that materialize through `build_seeded_archive`. Not an alternate construction route. | itself |

Every candidate in this table is a deliberate production seam, a shared
primitive, or an oracle; none of them is a duplicate to retire. The code that
*was* retired came from a separate orphan census, recorded under "Retirement
pass" below.

## Criterion status

- **AC6 — satisfied.** No generic pathology dimension survives in the workload
  substrate: `pathology_zoo.py` is deleted and no `pathology_composer.py`
  exists. `PathologyName` and `REQUIRED_PATHOLOGIES` live in
  `tests/infra/query_contract.py` beside the query laws that consume them,
  which AC5 requires rather than forbids. No arbitrary session-count tier
  remains — the surviving `session_count` is a bundle-shape parameter in
  `source_composer.py`, and the named profiles are the operational
  smoke/representative/archive-shaped/stress set. Malformed material survives
  only as fuzz corpora and Hypothesis strategies beside their owning laws, not
  as a historical-output profile.
- **AC11 — reporting satisfied; the deletion half requires a rewrite.** The
  orphan retirement pass below is complete, but the remaining named fixture
  modules are not duplicates: `archive_scenarios.py` is the shared scenario
  and tag-seeding seam used by six consumers, while `whale_fixtures.py`
  drives the production `AcquisitionService` for scale/revision coverage.
  Deleting either module would remove coverage or force a new test-only
  replacement, which conflicts with this bead's own prohibition on inventing
  another fixture substrate. The deletion criterion is therefore
  **AC-REWRITE**, not an open implementation task.
- **AC2 — see the retirement pass below.** The two candidates this ledger
  first recorded as "true duplicates" are re-examined there: one is refuted on
  evidence, one is refused with a reason.

## Retirement pass

Census and refutation above were docs-only. This section records the deletion
that followed, measured on the same denominators.

### What was deleted, and what absorbed it

Selection rule: a symbol was eligible only if a whole-repository reference
count — every `*.py`, `*.md` and `*.toml` token, including the defining
module's own body — found exactly one occurrence, the definition itself. An
orphan carries no obligation, so nothing needed to move for it. Where an
obligation existed independently, the owner that keeps it is named.

| deleted | kind | what keeps the obligation |
| --- | --- | --- |
| `tests/infra/schema_inference.py` (whole module, 65 lines) | direct-index seeder: raw `INSERT INTO raw_sessions / sessions / messages / blocks` against `source.db` and `index.db`, no manifest, no importer | nothing to keep — no test referenced it. The schema-inference gate routes are covered by `tests/unit/core/test_schema_*`, which use the production `polylogue.schemas.operator.schema_inference` module |
| `source_builders.admit_provider_source_packages` (12 lines) | second `parse_sources_archive` admission wrapper, no caller | `workload_artifacts` is the canonical admission route; it admits through `provider_source_package` → `admitted_sources` → `parse_sources_archive`. Both of those stay live and are exercised by `tests/unit/infra/test_source_builders.py` |
| `corpus_fixtures.integration_archive` (fixture, 4 lines) | session-scoped fixture building a whole heterogeneous archive; requested by no test, and not registered in `shared_session_archives()` | AC8's obligations are owned by `tests/unit/infra/test_integration_profile.py`, which calls `build_integration_archive` and `default_integration_selection` directly |
| `mcp.make_mock_filter`, `mcp.make_simple_session` (57 lines) | unreferenced mock builders | — |
| `strategies/filters.pagination_filter_chain_strategy`, `date_range_filter_chain_strategy` (32 lines) | unreferenced Hypothesis strategies | — |
| `large_batches.write_jsonl_file` (7 lines) | unreferenced writer | — |
| `pty_scenarios.EventKind` (1 line) | unreferenced alias | — |

`integration_archive` also violated the invariant `shared_session_archives`
states in its own docstring — "an archive reachable from a fixture cannot be
absent from the warm-up". `tests/unit/infra/test_shared_session_archives.py`
enforces that only over builders declared inside
`shared_session_archives.py`, so it did not catch a session fixture building
elsewhere. Deleting the fixture restores the invariant. No construction time
was recovered, because no test requested the fixture and it therefore never
built: the saving is a removed latent unwarmed build, not a measured one.

### Before / after

Measured against `origin/master` at `b8375e111`, which this branch is rebased onto.

| denominator | before | after | delta |
| --- | ---: | ---: | ---: |
| `tests/infra` `*.py` files | 106 | 105 | −1 |
| `tests/infra` LOC | 34,084 | 33,882 | −202 |
| `tests/infra` top-level defs | 1,126 | 1,118 | −8 |
| `workload_artifacts.py` + `workload_declarations.py` | 3,701 | 3,701 | 0 |
| modules with raw `INSERT INTO sessions/messages/blocks` | 2 | 1 | −1 |
| unreferenced public symbols in `tests/infra` | 165 lines / 7 files | 0 | −165 |

Construction cost is unchanged and honestly so: no archive-building call site
on a live path was removed, so no build count, byte count or wall time moved.
The one construction this pass removes was never reachable. Claiming a
construction-cost reduction here would be false.

The workload pair did not shrink. It is already the narrow substrate AC1
describes, and this bead explicitly refuses migrating domain builders into it
to make a number fall.

### Why the aggregate rose, and what the honest denominator is

`tests/infra` grew 28,486 → 34,084 between 2026-09-05 (`1a53a1f61`) and
`b8375e111`. Attributing that to workload-artifact consolidation is a
measurement error. Sixteen modules were added in that window, and the bulk of
the growth is the query-law family — `query_census`, `query_contract`,
`query_corpus`, `query_differential`, `query_field_laws`,
`surface_differential`, 4,280 lines — which belongs to a different
campaign and is law-owned material AC5 requires to live beside its laws.

The denominator this bead can move is the workload substrate and its
duplicate builders, not the `tests/infra` aggregate.

### AC2 candidates, re-examined

- `archive_scenarios.py` — **refuted, not deferred.** The earlier finding
  called `ArchiveScenario.seed` a "direct-index seeder"; it is not. `seed`
  delegates to `SessionBuilder` — the shared primitive this ledger keeps — and
  to `ArchiveStore.add_user_tags`, the same production API the public tag
  route uses. It writes no SQL of its own. The "25 importing modules" figure
  is also the wrong measure: those 25 almost all import the two pure helpers
  `native_session_id_for` and `open_index_db`, which construct nothing. The
  seeding path has six consumers in total (`tests/infra/continuity.py`,
  `tests/infra/surfaces.py`, `tests/infra/test_archive_scenarios.py`,
  `tests/unit/test_cross_surface_agreement.py`,
  `tests/unit/storage/test_tag_contracts.py`,
  `tests/unit/surfaces/test_public_fact_parity.py`). Nothing is duplicated, so
  nothing is retired.
- `whale_fixtures.py` — **refused with a reason, unchanged.** Publishing a
  manifest around `acquire_codex_revision_chain` is a design change to a
  builder that deliberately drives the production `AcquisitionService`, not a
  deletion. `copy_sqlite_database` is a six-line `shutil` helper, not an
  alternate clone substrate. This bead's own caution — do not create a narrow
  leaf without a true duplicate — applies.

### AC11 deletion re-evaluation (2026-09-23)

The current heads measure 456 lines for `tests/infra/archive_scenarios.py`
and 609 lines for `tests/infra/whale_fixtures.py`. Their consumers and
responsibilities are distinct from the canonical workload-artifact owner:

* `archive_scenarios.py` delegates through `SessionBuilder` and the
  production `ArchiveStore` tag API. Its six seeding consumers depend on that
  direct scenario shape, while most of its wider fan-in uses pure identity or
  connection helpers. Removing it would not remove duplicate artifact
  construction; it would remove the shared scenario contract.
* `whale_fixtures.py` exercises the production acquisition path and keeps
  revision-chain/append behaviour observable at scale. Wrapping it in a
  `CorpusArtifactManifest` would change its acquisition contract, not retire a
  duplicate builder.

No coverage-preserving successor exists at this head. The measured deletion
target is therefore not achievable without a new design decision. Rewrite
AC11 to require the completed orphan-retirement ledger and an explicit
disposition for these two retained seams, rather than requiring their
deletion. This is an AC-REWRITE, not an attempted implementation.

### Unsatisfied, with the remaining action

The 2026-09-06 note requires consolidation to remove the shared seeded-artifact
startup serialization behind
`/realm/tmp/polylogue-seeded-artifacts/.cleanup.lock` and the per-artifact
locks, and to keep cleanup off the hot path for independent read-only
consumers. **This is not satisfied and was not attempted here.** It cannot be
settled by reading code: the original finding was a live measurement (4–8
minutes of pre-xdist startup against ~17 seconds of selected test time, two
slots, one in `locks_lock_inode_wait` and one in Btrfs metadata flush). The
remaining action is a live focused-run measurement of startup wall time and
lock wait with two or more pytest slots, before and after moving cleanup off
the acquisition path, on the same host and filesystem. Anything short of that
would be a guess.

## AC13 wording

AC13 as written asks for controlled mutations that "substitute profile output
for an independent oracle". That clause is not implementable against this
substrate, and the reason is a deliberate design property rather than a gap:
`CorpusArtifactManifest.__post_init__` calls `reject_semantic_metadata` on its
receipt, files and resources, and `SEMANTIC_METADATA_PREFIXES` refuses any
`expected_`, `oracle_`, `pathology_` or `case_` key. There is no semantic
profile output to substitute, so a test for that clause would prove nothing.

The manifest's only expectation-shaped field, `SyntheticArtifactFacts.
expected_session_id`, is a planted-versus-materialized construction echo
checked by `_validate_facts`, not a domain verdict.

The other four mutation classes are implementable and already covered by
hostile tests in `tests/unit/infra/test_workload_artifacts.py`: artifact
identity (`test_profile_identity_controls_published_artifact_reuse`,
`test_seeded_archive_key_changes_with_derived_schema_identity`), cache
corruption (`test_seeded_archive_rejects_corrupt_published_cache_and_rebuilds`,
`test_seeded_archive_memo_rejects_same_size_database_corruption`,
`test_seeded_archive_rejects_forged_receipt_with_recomputed_manifest`), clone
isolation (`test_clone_rejects_tampered_copy_and_cleans_output`,
`test_seeded_archive_clone_leaves_unrelated_siblings_live`) and lease integrity
(`test_query_only_lease_refuses_mutated_or_replaced_source`).

Proposed replacement wording:

> 13. Controlled mutations that bypass a production parser, alter artifact
> identity, corrupt cache content, or weaken clone or lease isolation each fail
> their owning test. The substrate carries no semantic output to mutate — the
> manifest refuses expectation-shaped fields — so oracle substitution is proved
> instead by the schema-rejection test in AC4, and independent semantic oracles
> remain the property of their owning domain laws.
