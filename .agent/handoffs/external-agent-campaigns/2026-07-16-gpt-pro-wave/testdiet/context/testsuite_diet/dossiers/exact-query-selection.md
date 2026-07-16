---
cluster: exact-query-selection
readiness: prepared-not-execution-grade
git_head: 21f78b4db2ba62ff44b5f16dfab96067bc249b4c
generated_by: census/dossier.py
---

# Exact query selection, composition, and work bounds

> Evidence packet, not a deletion verdict or coverage gate.

## Responsibility

Public query expressions select exactly the planted fact set; membership, counts, partitions, pages, preview/apply, and bounded work agree through real routes.

## Readiness

`prepared-not-execution-grade`

- cluster prerequisite requires coordinator receipt: seeded-artifact-integrity
- upstream Bead contract requires merged-source coordinator receipt: polylogue-1xc.14.1
- upstream Bead contract requires merged-source coordinator receipt: polylogue-b054.1.1.4
- no realized sensitivity artifact found for the cluster

## Baseline dependencies

- `seeded-artifact-integrity`: Requires the cache adapter and independent facts attached to a realized named workload canary.
- `polylogue-1xc.14.1`: Begin with the realized C-03 exact-session actions canary, scale/selectivity tier, and workload receipt rather than designing a parallel corpus.
- `polylogue-b054.1.1.4`: Reuse the real testmon database and reversible production-mutation proof for sensitivity.

## Authoritative routes

| Path | Exists | Resolved symbols | Missing symbols |
| --- | --- | --- | --- |
| `polylogue/archive/query/expression.py` | yes | `parse_expression_ast`, `compile_expression`, `compile_expression_into` | — |
| `polylogue/archive/query/spec.py` | yes | `build_query_spec_from_params`, `query_spec_to_plan` | — |
| `polylogue/archive/query/plan.py` | yes | `SessionQueryPlan` | — |
| `polylogue/archive/query/archive_execution.py` | yes | — | — |
| `polylogue/cli/query.py` | yes | — | — |

## Independent obligations

- canonical and native identity remain exact under ambiguity
- boolean precedence, quoting, escaping, and diagnostics stay public-contract correct
- membership equals count and grouped partitions
- concatenated pages equal stable unpaged order
- preview and apply share effective selection
- irrelevant archive growth preserves result and bounded SQLite work
- unique FTS, vector, lineage, cancellation, and terminal-unit branches survive

## Proposed survivor tests

- tests/unit/cli/test_query_composition_laws.py::test_c03_workload_truth_across_repository_facade_cli_and_http
- tests/unit/cli/test_query_composition_laws.py::test_query_partitions_pages_and_preview_apply_agree
- tests/unit/cli/test_query_composition_laws.py::test_irrelevant_growth_preserves_bounded_work
- reviewed parser diagnostic decision table in test_query_expression.py

## Sensitivity witnesses

- temporary-production-mutation: Drop one structural predicate or weaken exact identity to suffix matching; manifest membership must fail.
- work-bound-mutation: Reuse the C-03 global-first ranking mutant or an equivalent production mutation; the exact-session irrelevant-growth VM-step law must fail.

Realized artifacts: `0`.

## Candidate scope

Tests: `tests/unit/cli/test_query_expression.py`, `tests/unit/cli/test_query_exec_laws.py`

Helpers: `tests/infra/query_cases.py`, `tests/infra/surfaces.py`, `tests/infra/semantic_facts.py`

Planned: `tests/infra/query_manifest_oracle.py`, `tests/unit/cli/test_query_composition_laws.py`

Avoid: `polylogue/mcp`, `polylogue/web`, `tests/unit/mcp`

## Deletion candidates requiring dominance proof

- mock-forwarding permutations in tests/unit/cli/test_query_exec_laws.py
- repetitive one-field lowering examples dominated by the reviewed decision table
- tests/infra/query_cases.py shadow fields after public-expression migration
- tests/infra/surfaces.py partial query translator after manifest oracle migration

## Evidence inventory

- pytest receipts: `1`
- testmon available/matching tests: `True` / `1403`
- coverage contexts available: `False`
- coupling findings: `9`
- fixture inventory rows: `3`
- mutation artifacts: `0`

## Recent path history

- `f0c1b489b84cd04aac840315e7e55fa23eb97e39	2026-07-16	fix: restore archive contract verification (#2932) (#2932)`
- `b6c78adfcd666358307daf64ac97e8d695a8b854	2026-07-16	feat(archive): expose exact-source freshness (#2924)`
- `5d99611f4aaca2eabbc8621a173692140ed165d3	2026-07-15	refactor(polylogue-dab): stop materializing run-projection cache rows (#2898)`
- `b6d851a933d3861404be38c895b1dc7cfe3bca82	2026-07-15	chore(protocols): prune zero-consumer protocols, fix dangling cursor mapping (#2906)`
- `d068d64821c6dc440013a28134e2f245fe7074b2	2026-07-14	refactor(architecture): sqlite leak sweep, staleness unify, control-center decomposition (#2900)`
- `89166362b9aee8c304b27a69f68ec1b74606f634	2026-07-14	feat(query-dsl,daemon,api): real production query evaluator + finding provenance (rxdo cluster) (#2899)`
- `13d19ae36c2bfbc60d1197390573f5beed08e953	2026-07-13	fix(actions): read legacy Codex commands without rewriting evidence (#2855)`
- `219869f660358df09946b726a0ca213a1e0c43bf	2026-07-13	fix(actions): expose Codex exec payloads as commands (#2853) (#2853)`
- `cc0999befd726bcd1c3f63c4ab8d35986bee7064	2026-07-13	refactor(storage): make source filters origin-native (#2820)`
- `a952221cdcc4813ffcc4c9c18c4fd8981d5bbb2a	2026-07-13	feat(query): materialize watched query relations (#2826)`
- `3082c72f0c046d184e6c5088d31cf87faac548e6	2026-07-13	feat(cli): add hot daemon read routing (#2827)`
- `58691ab16f8258397b32c0ba5806df5d3c30656b	2026-07-13	feat(insights): compile deterministic cohort manifests (#2775)`

## Permitted worker checks

```bash
devtools test tests/unit/cli/test_query_composition_laws.py
devtools test -k 'query_expression and (precedence or quoting or exact)'
```

The coordinator must refresh this dossier after the upstream merge, after collecting
per-test coverage contexts, and after sensitivity execution.
