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
| `archive_scenarios.py` — `ArchiveScenario.seed`, `seed_archive_scenarios`, `seed_workspace_scenarios` | True direct-index seeder: writes through `SessionBuilder` + `ArchiveStore` with no manifest | **Deferred retirement.** A genuine duplicate under the "direct-index seeders" clause, but 25 importing modules make the migration broad and test-file-wide. Not retired here. | `polylogue-1xc.14.1` follow-up leaf |
| `whale_fixtures.py` — `acquire_codex_revision_chain`, `copy_sqlite_database` | Drives real `AcquisitionService`; clones via `archive_templates` but publishes no manifest | **Deferred.** Scale-outlier artifacts are cacheable and should carry a manifest, but the builder deliberately uses the production acquisition seam; wrapping it is a design change, not a deletion. | follow-up leaf |
| `query_corpus.py` — `build_query_corpus` | Successor to the retired `pathology_zoo`; writes via `live_ingest.write_index_session` | **Keep.** Not a duplicate: it materializes into a caller-supplied root through the production writer seam and caches nothing, so there is no artifact identity to own. Its pathology vocabulary is law-owned (`query_contract.REQUIRED_PATHOLOGIES`). | `query_contract.py` |
| `corpus_program.py` | Opens `SQLiteBackend`/`ArchiveStore` directly | **Keep.** Explicitly delegates effects to production acquisition, parsing, convergence and hook seams; owns no cached artifact. | itself |
| `live_ingest.py` | Raw `sqlite3` connections into `index.db` | **Not a duplicate.** This is the single test seam onto `write_parsed_session_to_archive`, the production choke point. Retiring it would create duplication, not remove it. | itself |
| `storage_records.py` — `SessionBuilder` | Widest seeding primitive (fan-in 144) | **Keep.** The primitive the manifest route is built on, not a competitor to it. | itself |
| `reference_model.py` | Matched an archive-construction grep | **Not a duplicate.** A declared oracle that builds no archive. | itself |
| `convergence_harness.py` | Adapts production writers and ops ledger | **Keep.** Owns no alternate convergence state machine by its own contract. | itself |
| `source_composer.py` | Matched on `session_count` | **Not a duplicate.** Composes in-memory arrangements and writes no database; `session_count` is a bundle-shape parameter, not a scale tier. | itself |
| `integration_profile._SCALE_MINIMUM_MESSAGES` | `smoke`/`representative`/`archive-shaped`/`stress` minimums | **Keep.** Tier-naming residue used only as a validation gate over `CorpusSpec`s that materialize through `build_seeded_archive`. Not an alternate construction route. | itself |

No code was retired under this ledger. Every candidate is either a deliberate
production seam, a shared primitive, an oracle, or a migration too broad to
perform inside a census.

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
- **AC11 — satisfied by this ledger** for the reporting obligation. The
  deletion obligation it shares with AC2 is not met: the aggregate did not
  fall.
- **AC2 — open.** Two true duplicates (`archive_scenarios`, `whale_fixtures`)
  are recorded above rather than retired.

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
