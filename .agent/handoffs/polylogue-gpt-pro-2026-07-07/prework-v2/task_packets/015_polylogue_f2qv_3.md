# 015. polylogue-f2qv.3 — Dual cost view: API-list-equivalent and subscription-credit reported separately

Priority/type/status: **P2 / feature / open**. Lane: **02-usage-cost-honesty**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

PROBLEM. cost_usd is API-list-price-equivalent and OVERSTATES actual subscription spend: on Claude Max/Pro cache reads are free and the credit formula differs from list pricing. Memory (reference_claude_subscription_credit_pricing) also records a credit-rate 5x-output bug. Reporting a single number conflates two genuinely different accounting regimes.

FILES. cost rollup surfaces (cost_rollups / session_costs / cost_outlook MCP tools and their storage builders); the subscription credit-rate constants/formula. Depends on the LiteLLM single-pricing-source child for the API-equivalent leg and on the disjoint-lane child to know which tokens are free-on-subscription.

ALGORITHM. Compute two figures per session/day/origin: (1) api_equivalent_usd = sum(lane_tokens * LiteLLM_rate) counting cache reads at list price; (2) subscription_credit = credit formula with cache reads zeroed on subscription tiers and the corrected (non-5x) output credit rate. Surface both as distinct fields; never silently substitute one for the other. Document the plan-tier assumption driving the subscription view.

PITFALLS. The 5x-output credit-rate error must be fixed with a regression test. Do not apply the free-cache-read rule to API-tier sessions. Keep the two views additive-separable so a caller can choose.

## Acceptance criteria

Cost surfaces return api_equivalent_usd and subscription_credit as distinct fields; a test asserts they differ correctly for a session with cache reads (subscription view < API view). The credit-rate 5x-output error is fixed and locked by a test. Live archive shows both views for Claude and Codex sessions with cache-heavy inputs.

## Static mechanism / likely defect

Mechanism from bead: cache reads dominate token volume, but subscription plans may meter cache reads differently or not at all. Prior memory also records a 5x output credit-rate bug. Correct reports need two accounting regimes, not one blended number.

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
2. 1. Add fields such as `api_equivalent_usd`, `api_price_source`, `subscription_credit_estimate`, `subscription_credit_source`, and `cost_view_caveat` to session/day/origin cost surfaces.
3. 2. API-equivalent view uses the LiteLLM resolver from f2qv.4 and disjoint token lanes from f2qv.2.
4. 3. Subscription-credit view zeroes/free-prices lanes according to declared plan assumptions; cache-read treatment must be explicit.
5. 4. Correct the output credit rate bug and lock it with a fixture.
6. 5. Render both fields side by side. Never label subscription credits as dollars unless converted by a separate plan-capacity model.

## Tests to add

- cache-heavy session has `subscription_credit_estimate < api_equivalent_usd` under Claude Max assumptions.
- output credit-rate fixture catches the old 5x bug.
- unknown model leaves API field unknown/partial rather than false zero.
- JSON schemas/docs show both views.

## Verification commands

- ``devtools test tests/unit/storage/test_usage*.py tests/unit/cli/test_usage*.py -k 'subscription or api_equivalent or credit or cache'``

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
