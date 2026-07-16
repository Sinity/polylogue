# 014. polylogue-f2qv.4 — Single pricing source of truth: LiteLLM catalog, drop tokencost, last-path-segment match

Priority/type/status: **P2 / task / open**. Lane: **02-usage-cost-honesty**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

PROBLEM. Memory (cost/usage research 2026-06-28) records: LiteLLM is the sole pricing source, tokencost must be dropped, and model-name resolution should match the LAST path segment of the model id. A vendored LiteLLM price catalog was committed (67dd9e64c) covering gpt-5.x / codex / deepseek. Any residual second pricing table (tokencost or a hardcoded map) will drift against it.

FILES. The LiteLLM catalog module and its resolver; any remaining tokencost import or hardcoded per-model price map; pyproject dependency on tokencost. Cross-check cost rollup builders resolve through the single resolver.

ALGORITHM. All model->rate lookups go through one resolver keyed on the last path segment of the model id (e.g. vendor/family/model-name -> model-name). Remove tokencost from dependencies and imports. Add a test that every model observed in the live archive resolves to a LiteLLM rate or a labelled unknown (never a silent second-table value), and that no second price map exists.

PITFALLS. Model ids carry provider prefixes and dated suffixes; last-segment match must handle both. Unknown models must surface as an explicit caveat, not a $0 or a stale fallback price.

## Acceptance criteria

grep shows tokencost is removed from dependencies and imports; a single LiteLLM-backed resolver owns all model->rate lookups via last-path-segment match; a test asserts no second price table exists and that live-archive models resolve or are labelled unknown. Cost surfaces consume only this resolver.

## Static mechanism / likely defect

Mechanism from bead: a LiteLLM catalog exists, but tokencost/hardcoded maps may remain. Model ids can be vendor-prefixed, so resolver must match the last path segment rather than failing or resolving a parent path.

## Source anchors to inspect first

- `polylogue/storage/usage.py:478` — stale provider rollup stats are materialized/read here.
- `polylogue/storage/usage.py:797` — full stale diagnostics path can become expensive.
- `polylogue/storage/usage.py:891` — comments indicate corrected rows with cached/reasoning partitions.
- `polylogue/storage/usage.py:1033` — model rollup stats aggregate input/output/cache lanes.
- `polylogue/archive/semantic/cost_compute.py` — Inspect pricing/provenance computation before changing cost views.
- `polylogue/archive/semantic/subscription_pricing.py` — Subscription-credit view belongs here, distinct from API-list-equivalent cost.
- `scripts/cost_accounting_demo.py` — Existing demo captures usage/cost accounting expectations.

## Implementation plan

1. Implementation shape:
2. 1. `rg 'tokencost|PRICE|pricing|cost_per'` across `polylogue`, `scripts`, `pyproject.toml`.
3. 2. Remove `tokencost` dependency/imports.
4. 3. Create exactly one resolver module/function. It should normalize model ids by exact id first, then last path segment, then aliases if declared.
5. 4. Resolver result should be a structured object: rate fields + source/catalog version + `unknown` reason.
6. 5. Update all cost surfaces to call the resolver. Do not leave emergency hardcoded maps in report scripts.

## Tests to add

- `vendor/family/model-name` resolves via last segment when catalog contains `model-name`.
- observed live-archive model ids either resolve or become labelled unknown.
- no `tokencost` import/dependency remains.
- a fake second map in a fixture/test helper is caught by the no-second-price-table test.

## Verification commands

- ``devtools test tests/unit/usage tests/unit/storage -k 'pricing or litellm or cost'` plus `rg tokencost pyproject.toml polylogue scripts` returns no production hits.`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
