# 13. polylogue-f2qv.4 — Use one LiteLLM-backed pricing resolver and remove tokencost/second maps

Priority: **P2**  
Lane: **usage-cost-correctness**  
Readiness: **ready-now / grep-and-contract**

Depends on packet(s): polylogue-f2qv.2

## Why this is urgent / critical-path

Two pricing catalogs drift. Cost outputs must say one authoritative API-equivalent price source or labelled unknown.

## Static diagnosis / likely mechanism

Mechanism from bead: a LiteLLM catalog exists, but tokencost/hardcoded maps may remain. Model ids can be vendor-prefixed, so resolver must match the last path segment rather than failing or resolving a parent path.

## Implementation plan

Implementation shape:
1. `rg 'tokencost|PRICE|pricing|cost_per'` across `polylogue`, `scripts`, `pyproject.toml`.
2. Remove `tokencost` dependency/imports.
3. Create exactly one resolver module/function. It should normalize model ids by exact id first, then last path segment, then aliases if declared.
4. Resolver result should be a structured object: rate fields + source/catalog version + `unknown` reason.
5. Update all cost surfaces to call the resolver. Do not leave emergency hardcoded maps in report scripts.
6. Add a test that greps or introspects for forbidden second-table symbols/imports.

## Test plan

Tests:
- `vendor/family/model-name` resolves via last segment when catalog contains `model-name`.
- observed live-archive model ids either resolve or become labelled unknown.
- no `tokencost` import/dependency remains.
- a fake second map in a fixture/test helper is caught by the no-second-price-table test.

## Verification command / proof

`devtools test tests/unit/usage tests/unit/storage -k 'pricing or litellm or cost'` plus `rg tokencost pyproject.toml polylogue scripts` returns no production hits.

## Pitfalls

Do not mix this with subscription-credit math. This packet owns API-list-equivalent price lookup only.

## Files/functions to inspect or touch

- `pyproject.toml`
- `polylogue/**/pricing*.py`
- `polylogue/storage/usage.py`
- `scripts/agent_forensics.py or promoted report`
- `LiteLLM catalog module`
