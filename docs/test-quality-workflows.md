# Test Quality Workflows

This document is the canonical operator reference for correctness runs,
mutation checkpoints, benchmark campaigns, and the explicitly accepted slow
portion of the suite.

## Canonical Commands

### Full correctness run

```bash
nix develop -c pytest -q -n 0
```

Current baseline on this branch after Phase 7:

- `3487 passed, 1 warning in 269.56s (0:04:29)`
- measured with `nix develop -c pytest -q -n 0 --durations=30`

### Fast local run

Use this when iterating locally and you do not need the explicit slow proofs or
benchmarks.

```bash
nix develop -c pytest -q -n 0 -m "not slow and not benchmark"
```

### Validation lanes

Use the validation-lane runner when you want named operator surfaces instead of
remembering individual test files or campaign commands.

```bash
python -m devtools.run_validation_lanes --list
python -m devtools.run_validation_lanes --lane machine-contract
python -m devtools.run_validation_lanes --lane query-routing
python -m devtools.run_validation_lanes --lane frontier-local
python -m devtools.run_validation_lanes --lane frontier-extended
python -m devtools.run_validation_lanes --lane live-exercises --dry-run
```

Lane intent:

- `machine-contract`: root CLI JSON success/failure and runtime-health machine surfaces
- `query-routing`: query-first integration plus route-planning/unit proofs
- `tui`: Textual Mission Control interaction/state coverage
- `chaos`: ingestion hostility, interruption, and chronology suites
- `frontier-local`: machine + query + TUI + chaos
- `frontier-extended`: `frontier-local` plus fast scale and the small long-haul campaign
- `live-exercises`: explicit operator lane for read-only live archive QA exercises

### Focused mutation checkpoints

```bash
nix develop -c python -m devtools.mutmut_campaign list
nix develop -c python -m devtools.mutmut_campaign run <campaign>
nix develop -c python -m devtools.mutmut_campaign index
```

Durable mutation ledgers live in:

- [mutation-testing-baseline.md](./mutation-testing-baseline.md)
- [`docs/mutation-campaigns/`](./mutation-campaigns/README.md)

### Benchmark campaigns

```bash
nix develop -c python -m devtools.benchmark_campaign list
nix develop -c python -m devtools.benchmark_campaign run search-filters
nix develop -c python -m devtools.benchmark_campaign run storage
nix develop -c python -m devtools.benchmark_campaign run pipeline
nix develop -c python -m devtools.benchmark_campaign compare \
  docs/benchmark-campaigns/<baseline>.json \
  docs/benchmark-campaigns/<candidate>.json
nix develop -c python -m devtools.benchmark_campaign index
```

Durable benchmark artifacts live in:

- [`docs/benchmark-campaigns/`](./benchmark-campaigns/README.md)

## Benchmark Policy

Benchmark comparisons are intentionally not part of generic hosted CI. The repo
runs on heterogeneous machines and shared runners are too noisy for reliable
latency budgets.

Policy:

- correctness, lint, and typecheck fail CI by default
- mutation campaigns are explicit operator workflows, not default CI jobs
- benchmark comparisons are operator-run or fixed-hardware workflows
- benchmark regressions should warn at `10%` and fail at `20%` on stable
  hardware, which is the default policy encoded in
  [`devtools/benchmark_campaign.py`](../devtools/benchmark_campaign.py)

## Slow-Test Disposition

The slowest non-benchmark tests currently fall into four buckets.

### Explicit slow proofs

These are intentionally marked `slow` because they are valuable but expensive,
and they are not needed in the fast local lane.

- [`test_workflows.py`](/realm/project/polylogue/tests/integration/test_workflows.py)
  Comprehensive end-to-end provider workflows.
- [`test_schema_generation.py`](/realm/project/polylogue/tests/unit/core/test_schema_generation.py)
  Database-backed provider schema generation proof.
- [`test_schema_validation.py`](/realm/project/polylogue/tests/unit/core/test_schema_validation.py)
  Raw-corpus verification path using persisted payload-provider filtering.
- [`test_scale.py`](/realm/project/polylogue/tests/unit/storage/test_scale.py)
  Fixed performance-budget checks.
- [`test_vec.py`](/realm/project/polylogue/tests/unit/storage/test_vec.py)
  Vector-provider retry and degradation proofs.

### Kept in the default correctness lane

These are expensive, but they are core generated-graph/storage correctness
proofs rather than optional long proofs. They stay in the default lane.

- [`test_store_ops.py`](/realm/project/polylogue/tests/unit/storage/test_store_ops.py)
  Generated repository graph/view/session/projection contracts.
- [`test_store_ops.py`](/realm/project/polylogue/tests/unit/storage/test_store_ops.py)
  Backend ordering, filtering, deletion, and tag-distribution laws.
- [`test_resilience.py`](/realm/project/polylogue/tests/unit/pipeline/test_resilience.py)
  Acquisition/planning invariants that still catch meaningful regressions.

## Slowest Observed Non-Benchmark Tests

From the latest `--durations=30` run:

| Test | Time | Disposition |
| --- | ---: | --- |
| `test_verify_raw_corpus_uses_persisted_payload_provider_for_filters` | 6.96s | marked `slow` |
| `TestPerformanceBudget::test_fts_search_budget` | 5.05s | already `slow` |
| `TestPerformanceBudget::test_list_performance_budget` | 4.88s | already `slow` |
| `test_repository_views_agree_on_generated_graph` | 4.08s | keep in default lane |
| `test_generate_schema_from_db[codex]` | 3.97s | marked `slow` via test |
| `TestPerformanceBudget::test_get_many_performance_budget` | 3.87s | already `slow` |
| `test_repository_tree_methods_preserve_root_and_closure` | 3.82s | keep in default lane |
| `test_repository_lookup_views_and_projection_agree_on_generated_id` | 3.78s | keep in default lane |
| `test_repository_provider_filters_are_subset_and_count_consistent` | 3.69s | keep in default lane |
| `test_backend_list_tags_matches_generated_tag_distribution` | 3.62s | keep in default lane |

## Operator Guidance

When changing code in a narrow domain:

1. run the targeted `pytest -q -n 0` slice for that domain
2. run the corresponding mutation campaign if that domain has one
3. run the corresponding benchmark campaign only if the change touches a hot
   path already represented in `tests/benchmarks`
4. rerun the full correctness lane before closing the work

When a new slow test appears in the `--durations` report:

1. decide whether it is an optional heavy proof or a default correctness proof
2. mark it `slow` only in the first case
3. otherwise keep it in the default lane and document why it earns that cost
