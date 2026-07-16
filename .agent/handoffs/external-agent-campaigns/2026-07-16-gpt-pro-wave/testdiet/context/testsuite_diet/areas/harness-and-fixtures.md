---
created: 2026-07-16
purpose: Bounded execution packet for corpus cache integrity and fixture consolidation
status: prepared-awaiting-realized-baseline-reconciliation-and-sensitivity
project: polylogue
---

# Harness and fixtures

Generated dossier:
[`../dossiers/seeded-artifact-integrity.md`](../dossiers/seeded-artifact-integrity.md).

## Exact authority and ownership

Authoritative current routes and symbols:

- `tests/conftest.py:seeded_db`, `_managed_pytest_temp_root`,
  `_clone_archive_template`;
- `tests/infra/corpus_fixtures.py:_seed_db`, `corpus_seeded_db`;
- `polylogue/scenarios/corpus.py:CorpusSpec`,
  `build_default_corpus_specs`;
- `polylogue/schemas/synthetic/runtime.py:_generate_from_schema`;
- `tests/infra/semantic_facts.py:ArchiveFacts`, `SessionFacts`.

Treat the completed `polylogue-1xc.14.1` outcome as upstream authority. Read but
do not edit `polylogue/schemas/generation/workload_profiles.py`,
`polylogue/schemas/generation/archive_workload_profile.py`,
`polylogue/schemas/field_stats/distributions.py`, the correlated synthetic
generation route, the merged C-03/scale canaries, and their focused tests.
Also consume the `polylogue-1xc.14` workload receipts and the complete
seed/testmon proof receipts owned by `polylogue-b054.1.1.3` through `.5`.

This packet owns only residual cache key extension, atomic publication,
archive validation, immutable cloning, independent canary facts, and obsolete
fixture subtraction. It does not own observed distributions, workload or
archive identity, scale tiers, canary definitions, resource accounting,
affected-selection mutation proof, or xdist hang repetition.

Proposed survivor tests may live in
`tests/unit/core/test_corpus_artifacts.py` and must exercise the real generated
provider artifacts and ingest route. If the realized baseline already provides
the same publication/clone law, Sol deletes or narrows the proposed file rather
than adding a duplicate. Exact worker write files are assigned only after the
merged-source dossier refresh.

## Current evidence

- `tests/conftest.py` publishes the shared seeded archive by touching
  `.build.done` after converting per-file raw-store and ingest exceptions into
  warnings.
- Reuse checks are marker/file/blob presence; the cache identity is the
  checkout-path hash, not schema/generator/profile inputs.
- `tests/infra/corpus_fixtures.py` generates provider wire artifacts and uses a
  real archive ingest path, but has only four direct use sites.
- `tests/infra/scale_fixtures.py` delegates to
  `tests/benchmarks/conftest.py:_seed_realistic_db`, a separate distribution
  that directly constructs normalized storage records.
- `seeded_repository` and `seeded_db_writable` have no test consumers.
- `tests/infra/state_machines.py` has no consumers (124 physical / 103 nonblank
  lines).
- `tests/infra/growth_budgets.py` is consumed only by its own 62-line unit test
  and no product/scale test (198 physical / 156 nonblank combined).

Regenerate the infra consumer inventory with:

```bash
python .agent/scratch/testsuite_diet/census/test_infra_usage.py
```

The output is routing evidence. Dynamic/plugin loading and indirect imports
must be checked before deleting any zero-consumer module.

## Stronger replacement

Adapt one realized named workload/canary through the real
acquire/parse/materialize/index route into a content-addressed immutable cache.
Extend the shared receipt with archive validation and attach independently
planted expected facts. Expose only:

- immutable base archive fixture by upstream workload identity;
- private writable clone fixture;
- planted-facts reader independent of production query/storage code.

Keep direct normalized-record builders for narrow storage writer unit tests.
They are not a substitute for composition corpora and need not be forced
through provider parsing.

## Required mutations/failures

- make one provider ingest fail: no valid manifest/cache hit may result;
- change an upstream workload/profile/build/archive or schema/generator digest:
  cache must invalidate;
- kill the builder before publish: readers must ignore the partial directory;
- corrupt a finished DB or manifest count: validation must reject reuse;
- clone then mutate: base archive remains unchanged;
- run concurrent builders for one key: exactly one valid artifact is published.

The exact first four mutations and their expected failure reasons are mandatory
before this packet becomes survivor-execution-grade. Their realized failures
are mandatory before any deletion job becomes certified. Reuse
`b054.1.1.4`'s real production-mutation/testmon mechanism where applicable. A
helper-only unit test, wrapper mock, or self-authored manifest validator does
not satisfy them.

## Permitted focused commands

```bash
devtools test tests/unit/core/test_corpus_artifacts.py
devtools test -k seeded_db
```

Workers do not run `devtools verify`; Sol uses the merged `b054` receipts and
runs `devtools verify --all` after the harness wave integrates.

## Deletion targets after replacement

- unused root fixtures;
- obsolete sentinel/cache-key logic;
- the benchmark-only hand-built “realistic” seeder once benchmark profiles use
  the common artifact;
- unused lifecycle harness;
- generic growth-budget abstraction and its self-test;
- duplicated scale fixture plumbing and weak bounded/nonempty smoke tests.

Do not delete `SessionBuilder`/normalized record helpers wholesale: many unit
tests legitimately exercise the storage writer below provider parsing.

## Economics baseline

- `tests/infra/`: 8,265 nonblank / 9,883 physical lines;
- root `tests/conftest.py`: 914 nonblank / 1,124 physical lines;
- confirmed inert lifecycle/growth islands: 259 nonblank / 322 physical lines
  including the growth self-test;
- unused root fixture bodies: small confirmed lead, exact removal LOC to be
  recorded in the implementation diff;
- corpus/scale/benchmark consolidation: do not forecast until replacement
  profile and consumer inventory are designed.
