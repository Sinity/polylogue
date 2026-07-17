---
created: 2026-07-16
purpose: Execution packet for query selection, terminal actions, and cross-surface composition tests
status: prepared-awaiting-foundation-and-sensitivity
project: polylogue
---

# Query composition

Generated dossier:
[`../dossiers/exact-query-selection.md`](../dossiers/exact-query-selection.md).

## Exact authority and obligations

Read the public expression/lowering/execution route under
`polylogue/archive/query/`, the repository query readers, and
`polylogue/cli/query.py`. The exact current test/helper ownership surface
is:

- `tests/infra/query_cases.py:ArchiveQueryCase`;
- `tests/infra/surfaces.py:_query_case_to_spec`, `RepositorySurface`,
  `FacadeSurface`, `CLISurface`, `DaemonHTTPSurface`;
- `tests/unit/cli/test_query_expression.py`;
- `tests/unit/cli/test_query_exec_laws.py`;
- planned `tests/infra/query_manifest_oracle.py` and
  `tests/unit/cli/test_query_composition_laws.py`.

Independent obligations are exact canonical/native selection, ambiguity,
boolean precedence and diagnostics, membership/count agreement, partitioning,
stable paging, preview/apply identity, FTS/vector/lineage branches, and bounded
work under irrelevant growth. MCP and the current web reader are avoided
rewrite boundaries.

## Current cluster

- `tests/unit/cli/test_query_expression.py` (~6.5k lines): valuable grammar and
  execution branches mixed with repetitive one-field lowering cases and
  dropped-table absence fossils.
- `tests/unit/cli/test_query_exec_laws.py`: 66 tests, roughly 43 using patched
  or mocked collaborators; many prove forwarding rather than selection.
- `tests/infra/query_cases.py`, `tests/infra/surfaces.py`, and
  `tests/infra/strategies/filters.py`: partial shadow translations that omitted
  exact session identity, units/stages/projections, and current origin
  vocabulary.

## Build first

Create one discriminating micro archive through the real route. Its independent
fact manifest includes hit/miss facts for every selection dimension,
canonical/native IDs, an ambiguous suffix, tools/results/errors, lineage,
dates, tags, paths, and planted FTS terms. A small evaluator over the manifest,
not production SQL/lowering, computes expected IDs, counts, and partitions.

Run laws through repository, Python facade, CLI, HTTP, and rewritten MCP only
when it exists:

- membership equals manifest truth;
- count equals membership;
- groups/facets partition the selected set;
- concatenated pages equal unpaged order;
- explain preserves effective selection;
- preview/apply target the same selection;
- exact/selective work stays bounded after irrelevant archive growth.

## Retain

Grammar diagnostics for quoting/escaping/precedence/errors; unique vector,
lineage, FTS, terminal-unit, cancellation, and ambiguity branches; exact public
payload snapshots only where serialized representation is the contract.

## Delete after dominance proof

Repeated one-field lowering examples may become one reviewed decision table.
Delete mock-forwarding permutations, exact helper-call/source assertions,
dropped-private-table absence checks, and partial cross-surface translators
once the real-route laws kill the historical dropped-filter and unbounded-work
mutations.

## Survivor and sensitivity contract

The proposed survivor is a table/property-driven composition law using public
query expressions and a planted-fact evaluator that never imports production
lowering or SQL. Retain parser error/security examples and unique terminal
branches outside that law.

Required sensitivity:

- temporarily drop one structural predicate during lowering;
- replace exact identity matching with suffix matching;
- remove one page continuation field;
- grow irrelevant sessions and observe the SQLite VM-work bound fail when
  selection is applied after an archive-wide operation.

Permitted worker commands:

```bash
devtools test tests/unit/cli/test_query_composition_laws.py
devtools test -k 'query_expression and (precedence or quoting or exact)'
```
