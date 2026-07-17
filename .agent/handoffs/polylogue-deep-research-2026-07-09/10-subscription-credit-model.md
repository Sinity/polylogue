---
created: "2026-06-28"
purpose: "Full Claude subscription credit-model spec for Polylogue's subscription-reality cost view"
status: "complete"
project: "polylogue"
---

# Claude Subscription Credit Model — Spec for Polylogue Cost View

## Context / Problem

Polylogue's `cost_usd` (`session_model_usage.cost_usd`) is **API-list-price
equivalent**. For an operator running Claude Code on a Max plan this *overstates*
real spend by a large factor, because on subscription plans:

- **cache reads cost ZERO** (API charges ~10% of input rate; in agentic loops
  Claude Code re-reads the full context on every tool call, so cache reads
  dominate token volume — this is where the overstatement comes from);
- spend is bounded by a **flat monthly fee + credit pool**, not metered $;
- there are **5-hour rolling windows** and **weekly caps**, so $ is the wrong
  unit entirely.

Goal: present an **API-equivalent** view (value extracted) alongside a
**subscription-reality** view (credits consumed, plan-months) side by side.

## Sources (with URLs and authority)

1. **she-llac.com / claude-counter** — primary reverse-engineered model.
   - https://she-llac.com/claude-limits (Jan 2026, reverse-engineered from
     *unrounded* usage floats; validated by the author, **NOT official**).
   - https://github.com/she-llac/claude-counter — browser extension reading
     Claude's `/usage` endpoint + live SSE `message_limit` data; tracks the
     200k context bar, cache timer, 5-hour session bar, 7-day weekly bar.
   - This is the authority for the **credit formula and per-token rates**.
2. **Anthropic official** (numbers deliberately NOT published as fixed values):
   - https://support.claude.com/en/articles/12429409-manage-usage-credits-for-paid-claude-plans
     — usage credits = "billed at standard API rates"; mechanics only, no token math.
   - https://support.claude.com/en/articles/11049741-what-is-the-max-plan
   - https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work
   - https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan
3. **Third-party limit explainers** (corroborate structure, not exact pools):
   - https://www.morphllm.com/claude-code-usage-limits — confirms Anthropic
     stopped publishing fixed counts; capacity is expressed as **multipliers**
     (Pro = 1×, Max 5× = 5×, Max 20× = 20×); 5-hour window doubled May 6 2026;
     two weekly limits (all-models + Sonnet-only) since Aug 2025; a *separate*
     Agent-SDK/non-interactive monthly credit ($20/$100/$200) added Jun 15 2026.
   - https://tokenmix.ai/blog/complete-claude-limits-guide-2026-tokens-uploads-5-hour
   - https://www.truefoundry.com/blog/claude-code-limits-explained

**Authority caveat:** Anthropic publishes only the **1×/5×/20× multipliers** and
"billed at API rates." All absolute credit-pool and per-token-credit numbers
below are reverse-engineered (she-llac) — keep them dated and overridable.

## Credit Formula

```
credits = ceil(  input_tokens       × in_rate
               + output_tokens      × out_rate
               + cache_read_tokens  × 0           # FREE on subscription
               + cache_write_tokens × in_rate )   # cache write = input rate
```

Rates are expressed as `numerator / 15` credits per token (divisor 15 keeps the
ratios integer-friendly). Ratios mirror API list prices: output = 5× input,
Opus = 5× Haiku.

| Model  | input credits/tok | output credits/tok | cache_read | cache_write |
|--------|-------------------|--------------------|-----------|-------------|
| Haiku  | 2/15 ≈ 0.1333     | 10/15 ≈ 0.6667     | 0         | = input     |
| Sonnet | 6/15 = 0.4        | 30/15 = 2.0        | 0         | = input     |
| Opus   | 10/15 ≈ 0.6667    | 50/15 ≈ 3.3333     | 0         | = input     |

### Plan credit pools (she-llac, non-authoritative)

| Plan     | $/mo | 5-hour cap | weekly cap   | monthly pool (≈) | max_instances |
|----------|------|-----------|--------------|------------------|---------------|
| Pro      | 20   | 550k      | 5M           | 21.7M            | 1             |
| Max 5×   | 100  | 3.3M      | 41.667M      | 180.6M           | 5             |
| Max 20×  | 200  | 11M       | 83.333M      | 361.1M           | 20            |

Reference scale: Max 20× monthly ≈ **541.7M Opus-input** tokens OR **108.3M
Opus-output** tokens. (Pool ratios in the curated numbers are ~8.3× and ~16.6×
of Pro, which do NOT match the official 1/5/20 multipliers — treat the absolute
pools as best-effort reverse-engineering, and the multipliers as the only
official anchor.)

**API-equivalent value of plans** (lower bounds): Pro 8.1×, Max 5× 13.5×,
Max 20× 13.5×. Warm-cache agentic loops reach ~36× because cache reads are free.

## What Polylogue Already Has vs Missing

Already implemented (good prior art, mostly matches this spec):

- `polylogue/archive/semantic/subscription_pricing.py`
  - `SubscriptionTier` + `SUBSCRIPTION_TIERS` (pro/max_5x/max_20x) with the same
    pools (21.7M / 180.6M / 361.1M) and max_instances (1/5/20).
  - `ModelCreditRate` with `cache_read_credits=0` (free) ✓, `cache_write_credits`
    = input rate ✓, divisor 15 ✓, `credits_for()` does per-lane `ceil`.
  - `MODEL_CREDIT_RATES` for opus-4-6/4-5, sonnet-4-6/4-5, haiku-4-5.
  - `compute_credit_cost()`, `get_credit_rate()`, `SubscriptionCostEstimate`.
  - Provenance constants: `SUBSCRIPTION_CATALOG_PROVENANCE`,
    `..._EFFECTIVE_DATE = "2026-05-07"`, `SUBSCRIPTION_SOURCE_URL`.
- `polylogue/cost/plans.py` — typed `SubscriptionPlan` + `WELL_KNOWN_PLANS`
  curated seed (claude-pro/max-5x/max-20x as `credits` quota basis; ChatGPT/
  Copilot/Gemini with no quota), cycle math, user-config override.
- `polylogue/archive/semantic/cost_compute.py` — already computes per-model
  `api_cost_usd`, `credit_cost`, and `subscription_equivalent_usd` per session,
  side by side. Subscription-$ is derived as `credit_cost / 21.7M × $20`
  (Pro-credit-value basis).
- `subscription_models.py` / `outlook.py` — `UsageOutlookPayload` carries
  `api_equivalent_usd_total` AND `subscription_equivalent_usd_total`, plus
  credits_used/remaining/burn-rate/exhaustion projection. The side-by-side
  contract already exists at the outlook layer.

**Bug / gap — per-token output rate is WRONG in code.**
`MODEL_CREDIT_RATES` sets `output_credits == input_credits` for every model
(Opus 10/10, Sonnet 6/6, Haiku 2/2). The reverse-engineered model says **output
= 5× input** (Opus 50/15, Sonnet 30/15, Haiku 10/15). Current code therefore
*understates output credit cost by 5×*. Fix: set output_credits to 50/30/10 (or
keep numerator and multiply: out_rate = 5 × in_rate). This is the single most
important correction.

Other gaps:

- No encoded **5-hour** and **weekly** caps (only monthly pool). The window/cap
  structure (550k/5M, 3.3M/41.667M, 11M/83.333M) is not modeled, so
  "are you about to hit a weekly cap?" cannot be answered. `cost/plans.py`
  models one monthly cycle only.
- `subscription_equivalent_usd` is hardcoded to the **Pro** credit value
  (`/ 21.7M × $20`) in `cost_compute.py:133` regardless of the operator's
  actual plan. For a Max-20× operator the correct basis is
  `credit_cost / 361.1M × $200`. Should be plan-parametric.
- No **plan-month** presentation unit (corpus credits ÷ plan monthly pool).
- Two-weekly-limit split (all-models vs Sonnet-only) and the separate
  Agent-SDK monthly credit are not represented.
- ChatGPT cross-ref export (`/realm/inbox/download`, "High Lifetime Token Use"
  goblin — display counters exclude cache reads, understating ~100–700×) is a
  red-team companion, not yet wired as a calibration source.

## Subscription-Reality Cost Spec (proposed)

1. **Per message → credits** via the formula above, using provider-reported
   token lanes (input/output/cache_read/cache_write) already harmonized in
   `cost_compute.py`. Cache-read lane multiplied by 0.
2. **Per session/day/corpus**: sum credits per model; keep
   `api_equivalent_usd` (existing) as the parallel column.
3. **Plan-month conversion** (the headline unit):
   `plan_months = total_credits / SUBSCRIPTION_TIERS[plan].credit_pool`.
   For the corpus on Max 20× (361.1M/mo) this yields the "≈ N Max-20× plan-months"
   figure. Also expose `subscription_equivalent_usd = plan_months ×
   monthly_fee_usd` (plan-parametric — fixes the hardcoded Pro basis).
4. **Cap pressure** (new): compare rolling 5-hour and 7-day credit sums against
   the window caps to flag throttle risk; emit as outlook annotations.
5. **Presentation — always side by side, never collapse:**
   - `api_equivalent_usd` — "value extracted at list price" (what it would cost
     on the API). Honest framing: this is NOT spend for a subscriber.
   - `subscription_credits` + `plan_months` + `subscription_equivalent_usd`
     under the operator's actual plan — the real cost basis.
   - Stamp both with provenance + effective date + the non-authoritative notice
     (the reverse-engineered numbers WILL drift; keep overridable in
     `polylogue.toml`).

## Immediate Action Items

1. Fix `MODEL_CREDIT_RATES` output rates to 5× input (data correctness bug).
2. Make `subscription_equivalent_usd` plan-parametric (drop hardcoded Pro basis
   in `cost_compute.py:133`).
3. Add 5-hour / weekly caps to the tier/plan model for throttle-pressure views.
4. Add a `plan_months` rollup as the corpus headline subscription unit.
