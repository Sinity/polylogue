# 14. polylogue-f2qv.3 — Report API-equivalent dollars and subscription credits as separate fields

Priority: **P2**  
Lane: **usage-cost-correctness**  
Readiness: **ready-now after lanes/pricing**

Depends on packet(s): polylogue-f2qv.2, polylogue-f2qv.4

## Why this is urgent / critical-path

A single `cost_usd` field conflates API list-equivalent value with actual subscription metering. That produces persuasive but misleading financial claims.

## Static diagnosis / likely mechanism

Mechanism from bead: cache reads dominate token volume, but subscription plans may meter cache reads differently or not at all. Prior memory also records a 5x output credit-rate bug. Correct reports need two accounting regimes, not one blended number.

## Implementation plan

Implementation shape:
1. Add fields such as `api_equivalent_usd`, `api_price_source`, `subscription_credit_estimate`, `subscription_credit_source`, and `cost_view_caveat` to session/day/origin cost surfaces.
2. API-equivalent view uses the LiteLLM resolver from f2qv.4 and disjoint token lanes from f2qv.2.
3. Subscription-credit view zeroes/free-prices lanes according to declared plan assumptions; cache-read treatment must be explicit.
4. Correct the output credit rate bug and lock it with a fixture.
5. Render both fields side by side. Never label subscription credits as dollars unless converted by a separate plan-capacity model.
6. Keep unknown models/unknown plan assumptions as labelled unknown/partial, not zero.

## Test plan

Tests:
- cache-heavy session has `subscription_credit_estimate < api_equivalent_usd` under Claude Max assumptions.
- output credit-rate fixture catches the old 5x bug.
- unknown model leaves API field unknown/partial rather than false zero.
- JSON schemas/docs show both views.

## Verification command / proof

`devtools test tests/unit/storage/test_usage*.py tests/unit/cli/test_usage*.py -k 'subscription or api_equivalent or credit or cache'`

## Pitfalls

Do not remove the old cost field abruptly without compatibility plan; either deprecate it or make it a wrapper with a caveat. Do not pair all-provider token totals with priced-only dollars.

## Files/functions to inspect or touch

- `polylogue/storage/usage.py`
- `polylogue/storage/sqlite/archive_tiers/write.py`
- `polylogue/cli usage/analyze surfaces`
- `MCP cost tools`
- `scripts/agent_forensics.py or report surface`
