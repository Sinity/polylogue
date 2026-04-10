# Test Quality Workflows

This document is the canonical operator reference for correctness runs,
mutation checkpoints, benchmark campaigns, and the explicitly accepted slow
portion of the suite.

## Canonical Commands

### Full correctness run

```bash
nix develop -c pytest -q -n 0
```

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
nix develop -c python -m devtools.run_validation_lanes --list
nix develop -c python -m devtools.run_validation_lanes --lane machine-contract
nix develop -c python -m devtools.run_validation_lanes --lane frontier-local
nix develop -c python -m devtools.run_validation_lanes --lane live-exercises --dry-run
```

Treat `--list` as the source of truth for the active lane catalog rather than
copying the full lane set into static documentation.

Lane intent:

- `machine-contract`: root CLI JSON success/failure and runtime-health machine surfaces
- `query-routing`: query-first integration plus route-planning/unit proofs
- `showcase-baselines`: registry-derived tier-0 CLI help and source-baseline drift check
- `pipeline-probe-chatgpt`: short synthetic ChatGPT parse probe with explicit runtime/RSS budgets
- `semantic-stack`: harmonization, semantic facts/profile convergence, proof, and contract inventory
- `source-provider-fidelity`: local source traversal, Drive/runtime boundaries, and provider-ingest fidelity
- `maintenance-control-plane`: health, maintenance selection, cache/live provenance, and publication maintenance summaries
- `archive-data-products`: durable archive products, consumer contracts, product-aware grouped stats, and health/governance surfaces
- `semantic-product-normalization`: semantic/session product normalization, operator/toolchain narrowing, schema contracts, and provider parser cleanup
- `evidence-tier-contracts`: explicit evidence-tier product contracts, chronology fields, and durable evidence payload/query surfaces
- `inference-tier-contracts`: inference-tier work-event/phase/profile contracts with confidence/provenance-bearing semantic payloads
- `mixed-consumer-contracts`: CLI, facade, MCP, and health surfaces consuming the same evidence/inference product model
- `retrieval-band-readiness`: transcript/evidence/inference retrieval-band readiness, embedding stats, and health exposure
- `evidence-stewardship-contracts`: `evidence-tier-contracts` plus `inference-tier-contracts`, `mixed-consumer-contracts`, and `retrieval-band-readiness`
- `heuristic-inference-contracts`: support/confidence/provenance contracts for heuristic profile, work-event, and phase products
- `probabilistic-enrichment-contracts`: durable enrichment-product contract lane across CLI, library, sync, repository, MCP, and retrieval-health surfaces
- `governed-cleanup-contracts`: archive-debt lineage, maintenance preview/apply semantics, and cleanup-governance contract lane
- `evidence-stewardship-live`: bounded live archive lane for tiered product views, live session-product repair, health, and retrieval-band budgets
- `evidence-stewardship-hardening`: `evidence-stewardship-contracts` plus `evidence-stewardship-live`
- `live-products-enrichments`: bounded live archive lane for enrichment-product reads
- `probabilistic-enrichment-live`: bounded live archive lane for enrichment products, retrieval bands, and health surfaces
- `governed-cleanup-live`: bounded live archive lane for cleanup debt preview/validation governance and maintenance budgets
- `probabilistic-enrichment-hardening`: heuristic inference + probabilistic enrichment + governed cleanup closure lane
- `semantic-product-live`: bounded live archive lane for normalized product surfaces, debt governance, and maintenance preview/budgets
- `semantic-product-hardening`: `semantic-product-normalization` plus `semantic-product-live`
- `runtime-substrate-contracts`: local closure lane for the decomposed query/runtime/product/maintenance contract surfaces
- `runtime-substrate-live`: bounded live archive dogfooding for runtime-substrate retrieval, governance, and memory budgets
- `runtime-substrate-hardening`: `runtime-substrate-contracts` plus `runtime-substrate-live`
- `source-runtime-governance`: local closure lane for source/provider fidelity plus runtime maintenance convergence
- `retrieval-dogfood`: action-aware query truth, grouped retrieval stats, archive health, and MCP retrieval payloads
- `embeddings-coverage`: embedding readiness/coverage stats and embed command contracts
- `archive-intelligence`: local closure lane for retrieval and embedding readiness
- `archive-data-products-live`: local durable-product contract lane plus bounded live durable-product dogfood
- `domain-read-model-contracts`: local closure lane for domain-banded products, analytics, consumer contracts, and archive-debt governance
- `domain-read-model-live`: bounded live archive lane for durable products, analytics/debt products, and maintenance governance
- `domain-read-model-stewardship`: `domain-read-model-contracts` plus `domain-read-model-live`
- `tui`: Textual Mission Control interaction/state coverage
- `chaos`: ingestion hostility, interruption, and chronology suites
- `frontier-local`: machine + query + semantic + TUI + chaos
- `frontier-extended`: `frontier-local` plus fast scale and the small long-haul campaign
- `live-archive-small`: bounded live archive embedding/retrieval/health dogfood
- `live-products-small`: bounded live archive durable-product and grouped-stats dogfood
- `live-governance-small`: bounded live archive health plus maintenance-preview dogfood
- `memory-budget`: explicit RSS budget check for a representative live retrieval query
- `maintenance-memory-budget`: explicit RSS budget check for the live maintenance preview surface
- `live-maintenance-preview`: explicit operator lane for machine-readable repair/cleanup preview output
- `live-exercises`: explicit operator lane for read-only live archive QA exercises

### Focused mutation checkpoints

```bash
nix develop -c python -m devtools.mutmut_campaign list
nix develop -c python -m devtools.mutmut_campaign run <campaign>
nix develop -c python -m devtools.mutmut_campaign index
```

Durable mutation ledgers live in:

- [mutation-testing-baseline.md](./mutation-testing-baseline.md)
- local artifact output under `artifacts/mutation-campaigns/`

### Fast pipeline probes

Use the real pipeline probe before running long-haul campaigns or touching hot
paths. It writes the normal `archive_root/runs/run-*.json` artifact and emits
stage timing plus RSS metrics in one JSON summary.

There are now two useful tiers:

- Synthetic smoke probes: tiny generated corpora for fast budget checks in validation lanes.
- Archive-subset probes: replay a persisted, isolated subset of already-acquired raw rows from a real archive into a secondary workspace, with parse and validation state reset so post-acquire stages run again.

```bash
nix develop -c python -m devtools.pipeline_probe --provider chatgpt --count 5 --stage parse --workdir /tmp/polylogue-probe
nix develop -c python -m devtools.pipeline_probe --provider claude-code --count 3 --stage all --workdir /tmp/polylogue-probe-cc --json-out /tmp/polylogue-probe-cc.json
nix develop -c python -m devtools.pipeline_probe --provider chatgpt --count 5 --stage parse --max-total-ms 10000 --max-peak-rss-mb 512
nix develop -c python -m devtools.pipeline_probe --input-mode archive-subset --source-db /home/sinity/.local/share/polylogue/polylogue.db_ --sample-per-provider 50 --stage parse --workdir /tmp/polylogue-probe-real --manifest-out /tmp/polylogue-probe-real.json
nix develop -c python -m devtools.pipeline_probe --input-mode archive-subset --manifest-in /tmp/polylogue-probe-real.json --stage parse --workdir /tmp/polylogue-probe-replay
nix develop -c python -m devtools.pipeline_probe --input-mode archive-subset --source-db /home/sinity/.local/share/polylogue/polylogue.db_ --provider claude-code --provider codex --sample-per-provider 25 --stage parse
```

Archive-subset mode fails loudly when the selected source archive has no usable
raw rows, instead of silently “benchmarking” an empty archive. If your live
archive is not on the default `polylogue.db` path, pass `--source-db`
explicitly.

### Showcase baseline drift

Use this to verify the tier-0 CLI help surface tracked by the registered root
command set. Missing or newly added commands show up as explicit drift.

```bash
nix develop -c python -m devtools.verify_showcase
nix develop -c python -m devtools.verify_showcase --update
```

### Benchmark campaigns

```bash
nix develop -c python -m devtools.benchmark_campaign list
nix develop -c python -m devtools.benchmark_campaign run search-filters
nix develop -c python -m devtools.benchmark_campaign run storage
nix develop -c python -m devtools.benchmark_campaign run pipeline
nix develop -c python -m devtools.benchmark_campaign compare \
  artifacts/benchmark-campaigns/<baseline>.json \
  artifacts/benchmark-campaigns/<candidate>.json
nix develop -c python -m devtools.benchmark_campaign index
```

Durable benchmark artifacts live in:

- local artifact output under `artifacts/benchmark-campaigns/`

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

- [`test_workflows.py`](../tests/integration/test_workflows.py)
  Comprehensive end-to-end provider workflows.
- [`test_schema_generation.py`](../tests/unit/core/test_schema_generation.py)
  Database-backed provider schema generation proof.
- [`test_schema_validation.py`](../tests/unit/core/test_schema_validation.py)
  Raw-corpus verification path using persisted payload-provider filtering.
- [`test_scale.py`](../tests/unit/storage/test_scale.py)
  Fixed performance-budget checks.
- [`test_vec.py`](../tests/unit/storage/test_vec.py)
  Vector-provider retry and degradation proofs.

### Kept in the default correctness lane

These are expensive, but they are core generated-graph/storage correctness
proofs rather than optional long proofs. They stay in the default lane.

- [`test_store_ops.py`](../tests/unit/storage/test_store_ops.py)
  Generated repository graph/view/session/projection contracts.
- [`test_store_ops.py`](../tests/unit/storage/test_store_ops.py)
  Backend ordering, filtering, deletion, and tag-distribution laws.
- [`test_resilience.py`](../tests/unit/pipeline/test_resilience.py)
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
