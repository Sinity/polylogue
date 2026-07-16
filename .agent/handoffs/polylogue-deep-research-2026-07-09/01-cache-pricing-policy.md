---
created: "2026-06-28"
purpose: "Define the single explicit cache-class pricing policy for re-pricing the whole archive"
status: "active"
project: "polylogue"
---

# Cache-Class Pricing Policy (archive re-pricing)

## Context

A full LiteLLM price catalog was vendored at
`polylogue/archive/semantic/data/litellm_model_prices.json` (2918 keys, ~1.6 MB).
Pricing code: `polylogue/archive/semantic/pricing.py`,
`polylogue/archive/semantic/cost_compute.py`,
`polylogue/archive/semantic/subscription_pricing.py`. Token classing/audit:
`polylogue/storage/usage.py` and the Codex lane mapper in
`polylogue/storage/sqlite/archive_tiers/write.py:_provider_usage_disjoint_lanes`.

The question: what is the ONE documented cache-class pricing policy Polylogue
should adopt when re-pricing the whole archive.

---

## 1. How the vendored LiteLLM JSON encodes cache pricing

All values are **USD per single token** (multiply by 1e6 for per-1M). The
cache-relevant fields per model entry are:

| JSON field | Meaning |
|---|---|
| `input_cost_per_token` | Fresh (uncached) input price. |
| `output_cost_per_token` | Output price. |
| `cache_read_input_token_cost` | Price to *read* a cache hit (discounted input). |
| `cache_creation_input_token_cost` | Price to *write*/create a 5-min-TTL cache entry. |
| `cache_creation_input_token_cost_above_1hr` | Price for the 1-hr-TTL cache write tier. |

### Anthropic — carries the full cache triple

```json
"claude-opus-4-8": {
  "litellm_provider": "anthropic",
  "input_cost_per_token": 5e-06,
  "output_cost_per_token": 2.5e-05,
  "cache_read_input_token_cost": 5e-07,
  "cache_creation_input_token_cost": 6.25e-06,
  "cache_creation_input_token_cost_above_1hr": 1e-05
}
"claude-sonnet-4-5": {
  "litellm_provider": "anthropic",
  "input_cost_per_token": 3e-06, "output_cost_per_token": 1.5e-05,
  "cache_read_input_token_cost": 3e-07,
  "cache_creation_input_token_cost": 3.75e-06,
  "cache_creation_input_token_cost_above_1hr": 6e-06
}
```

Anthropic is the only family that reliably ships `cache_creation_input_token_cost`
(the cache-WRITE lane). The `_above_1hr` tier exists but Polylogue's loader does
**not** read it (Claude Code uses the default 5-min TTL).

### OpenAI / gpt-5.x / codex — cache-READ only, NO creation field

```json
"gpt-5":         {"litellm_provider":"openai","input_cost_per_token":1.25e-06,"output_cost_per_token":1e-05,"cache_read_input_token_cost":1.25e-07}
"gpt-5.1-codex": {"litellm_provider":"openai","input_cost_per_token":1.25e-06,"output_cost_per_token":1e-05,"cache_read_input_token_cost":1.25e-07}
"gpt-5-codex":   {"litellm_provider":"openai","input_cost_per_token":1.25e-06,"output_cost_per_token":1e-05,"cache_read_input_token_cost":1.25e-07}
"codex-mini-latest": {"litellm_provider":"openai","input_cost_per_token":1.5e-06,"output_cost_per_token":6e-06,"cache_read_input_token_cost":3.75e-07}
"o3":            {"litellm_provider":"openai","input_cost_per_token":2e-06,"output_cost_per_token":8e-06,"cache_read_input_token_cost":5e-07}
```

There is **no `cache_creation_input_token_cost`** on OpenAI entries — OpenAI does
not charge for cache writes; caching is automatic and you only pay the discounted
read. This is correct, not a gap.

### Gemini / DeepSeek — cache-READ only (same shape as OpenAI)

```json
"gemini-2.5-pro":   {"litellm_provider":"vertex_ai-language-models","input_cost_per_token":1.25e-06,"output_cost_per_token":1e-05,"cache_read_input_token_cost":1.25e-07}
"gemini-2.0-flash": {"litellm_provider":"vertex_ai-language-models","input_cost_per_token":1e-07,"output_cost_per_token":4e-07,"cache_read_input_token_cost":2.5e-08}
"deepseek-chat":    {"litellm_provider":"deepseek","input_cost_per_token":2.8e-07,"output_cost_per_token":4.2e-07,"cache_read_input_token_cost":2.8e-08}
```

### Older models — NO cache fields at all

`gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`, `chatgpt-4o-latest` carry only
`input_cost_per_token`/`output_cost_per_token` — no cache fields whatsoever
(these predate prompt caching).

### How the loader ingests this (`pricing.py:_load_litellm_catalog`)

```python
cache_read_usd_per_1m  = float(entry.get("cache_read_input_token_cost") or 0.0)     * 1_000_000
cache_write_usd_per_1m = float(entry.get("cache_creation_input_token_cost") or 0.0) * 1_000_000
```

- `cache_read_input_token_cost` → `ModelPricing.cache_read_usd_per_1m`
- `cache_creation_input_token_cost` → `ModelPricing.cache_write_usd_per_1m`
- `cache_creation_input_token_cost_above_1hr` → **ignored** (5-min TTL assumed)
- Missing field → `0.0` (so OpenAI cache-write lane = $0, which is the correct
  billing behaviour, not a defect).
- Keys stored both fully-qualified (`openai/gpt-5`) and bare (`gpt-5`); bare
  collisions resolve to the first non-Azure provider.
- `_CURATED_PRICING` (hand-verified) overrides the catalog and is the source of
  truth for the Claude models the operator actually uses.

---

## 2. How Polylogue classes token columns today

### Pricing/cost-compute model (`CostUsagePayload`, 4 lanes)

`pricing.py` works in exactly four token classes:
`input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`.
`_cost_components()` bills them as **four independent additive lanes**:

```
cost = input*input_rate
     + output*output_rate
     + cache_read*cache_read_rate
     + cache_write*cache_write_rate     (all /1e6)
```

This is the **Anthropic convention**: `input` means FRESH/uncached input;
cache-read is a separate discounted lane; cache-write (creation) is a separate
premium lane.

### Storage/audit model (`storage/usage.py:UsageCounters`, 6 lanes)

The provider-usage audit surface carries six lanes:
`input_tokens`, `output_tokens`, `cached_input_tokens`, `cache_write_tokens`,
`reasoning_output_tokens`, `total_tokens`. Note `cached_input` (storage) ==
`cache_read` (pricing); `reasoning_output` is a sub-lane of output and is NOT a
separate billable term.

### Per-provider native token semantics (from `usage.py` coverage matrix)

- **Claude Code** (`claude-code-session`): exact `message.usage`. Native fields
  `cache_read_input_tokens` and `cache_creation_input_tokens` preserved as
  `cache_read`/`cache_write`, **not folded into input/output**. `input` is
  already fresh/uncached on the wire.
- **Codex** (`codex-session`): exact `token_count`. Native `input_tokens`
  is **inclusive of `cached_input_tokens`** (~96% of input is cached on the real
  corpus), and `output_tokens` is **inclusive of `reasoning_output_tokens`**.
- **ChatGPT export / Claude.ai export**: estimate-only; no cache lanes.
- **Gemini (AI Studio Drive / CLI), Hermes**: partial; no verified cache lanes.
- **Antigravity / unknown**: unsupported; no token telemetry.

### The Codex disjoint-lane fix (load-bearing, `write.py:_provider_usage_disjoint_lanes`)

Because the pricing model bills `input` and `cache_read` as separate additive
lanes (Anthropic convention) but Codex reports `input` *inclusive* of cached,
the materializer subtracts cached out of input BEFORE pricing:

```
fresh_input = max(input_with_cached - cache_read, 0)
return (fresh_input, output_with_reasoning, cache_read, cache_write)
```

Without this, cached tokens were billed twice (full input rate + cache-read
rate) — historically an ~8x Codex input-cost inflation. `reasoning` is left
inside `output` (never added again). This transform is applied at
materialization time, so `session_model_usage.input_tokens` already holds
fresh/uncached input.

---

## 3. RECOMMENDED POLICY (single, explicit)

**Adopt the Anthropic four-lane additive convention archive-wide, where
`input` always means fresh/uncached input.** This is what `_cost_components`
already implements; the policy is to make it the documented, uniform contract
and ensure every provider's tokens are normalized into it before pricing.

### Canonical mapping table

| Polylogue lane | Catalog price field → ModelPricing | Billing rule |
|---|---|---|
| `input_tokens` (fresh, uncached) | `input_cost_per_token` → `input_usd_per_1m` | `tokens * rate` |
| `output_tokens` (incl. reasoning) | `output_cost_per_token` → `output_usd_per_1m` | `tokens * rate` |
| `cache_read_tokens` (= `cached_input`) | `cache_read_input_token_cost` → `cache_read_usd_per_1m` | `tokens * rate` |
| `cache_write_tokens` (cache creation) | `cache_creation_input_token_cost` → `cache_write_usd_per_1m` | `tokens * rate` |

### Normalization rules (apply BEFORE pricing)

1. **Fresh-input invariant.** `input_tokens` fed to the price model must be
   uncached. Anthropic/Claude already reports it that way. Codex must run
   `_provider_usage_disjoint_lanes` (subtract cached from input) — already done
   at materialization. Any future provider that reports input inclusive of
   cache must do the same subtraction.
2. **Reasoning stays inside output.** Never add `reasoning_output_tokens` as a
   separate billed term; it is already in `output_tokens`.
3. **Cache-read vs fresh-input.** Cache reads are a separate, discounted lane
   (typically 0.1x input for Anthropic/OpenAI/Gemini). They are NEVER billed at
   the fresh-input rate.

### Missing-field fallback (the explicit rule)

When a model entry lacks a cache field, **treat that lane's rate as 0.0** (the
loader's current behaviour) — but interpret it correctly:

- **Missing `cache_creation_input_token_cost` (OpenAI/Gemini/DeepSeek):** correct
  by design. Those providers do not charge for cache creation; cache_write
  tokens are usually absent there anyway. Rate 0.0 = accurate.
- **Missing `cache_read_input_token_cost` (gpt-4, gpt-4-turbo, gpt-3.5-turbo,
  chatgpt-4o-latest):** these predate prompt caching; cache_read tokens should
  be ~0 for them. If cache_read tokens *are* present for such a model, fall back
  to billing them at the **fresh-input rate** (conservative — never silently
  free) rather than 0.0. (This is a deliberate divergence from the current blunt
  0.0 default and is the one behavioural change the policy recommends.)
- **Model entirely absent from catalog:** status `unavailable`, reason
  `no_price` — do not guess. (Already implemented.)

### Subscription vs API distinction (must be carried as separate bases)

The catalog cache rates are **API-list-equivalent** pricing. They do NOT reflect
what the operator actually pays on a Claude Max/Pro subscription:

- On Claude Code subscription credit accounting
  (`subscription_pricing.py:ModelCreditRate`), **cache reads cost 0 credits**
  (`cache_read_credits=0`) — i.e. cache reads are FREE on Max/Pro. Cache writes
  cost credits.
- Therefore re-pricing MUST populate both bases on `CostBasisPayload`:
  - `api_equivalent_usd` / `catalog_priced_usd`: cache_read billed at the
    catalog `cache_read_usd_per_1m` (list price).
  - `subscription_equivalent_usd`: cache_read billed at $0 (free on plan);
    derived from the credit model, not the list catalog.
- Reports must label which basis they show. "API-equivalent" overstates real
  subscription spend, primarily because of free cache reads.

### Provenance / re-pricing

Stamp every re-priced row with
`_PRICE_SNAPSHOT_VERSION = f"{CATALOG_PROVENANCE}-{CATALOG_EFFECTIVE_DATE}"`
(already done in `cost_compute.py`) so catalog-version drift is auditable and a
future catalog refresh produces distinguishable rows. Provider-reported exact
costs still win over catalog pricing; curated overrides win over the vendored
catalog.

---

## 4. Archive models with NO cache pricing in the catalog

Best-effort from catalog inspection over archive-relevant models. Two distinct
"no cache pricing" cases:

### (a) Model missing from the vendored catalog entirely

| Model | Notes |
|---|---|
| `claude-3-5-sonnet-20241022` | **Covered by `_CURATED_PRICING`** (3.0/15.0, cache 0.3/3.75). No gap. |
| `claude-3-5-haiku-20241022` | **Covered by `_CURATED_PRICING`** (0.8/4.0, cache 0.08/1.0). No gap. |
| `o1-mini` | In `_CURATED_PRICING` but **without cache rates** (3.0/12.0, cache 0/0). |
| `gemini-1.5-pro` | In `_CURATED_PRICING` but **without cache rates** (3.5/10.5, cache 0/0). |
| `gemini-1.5-flash` | In `_CURATED_PRICING` but **without cache rates** (0.075/0.3, cache 0/0). |

Catalog name drift is the cause: the vendored JSON keys these under different
aliases (e.g. dated/regional variants); the curated overrides are what make them
priceable. The curated Gemini 1.5 and o1-mini entries carry input/output only —
cache lanes default to 0.

### (b) In catalog but no cache fields (legacy, pre-caching)

| Model | cache_read | cache_creation |
|---|---|---|
| `gpt-4` | absent | absent |
| `gpt-4-turbo` | absent | absent |
| `gpt-3.5-turbo` | absent | absent |
| `chatgpt-4o-latest` | absent | absent |

For these, cache tokens are expected to be ~0; if any appear, apply the
fresh-input fallback from §3.

### (c) Cache-READ present, cache-WRITE absent (correct, not a gap)

All current OpenAI (`gpt-5*`, `gpt-5*-codex`, `gpt-4o*`, `o1`, `o3*`, `o4-mini`,
`codex-mini-latest`), Gemini (`2.5-pro`, `2.5-flash`, `2.0-flash`,
`gemini-3-pro-preview`), and DeepSeek models. Cache-write lane = 0.0 by design.

---

## Outcome / recommendation in one line

Bill four disjoint additive lanes (fresh-input / output-incl-reasoning /
cache-read / cache-write) under the Anthropic convention; normalize every
provider to fresh-input before pricing (Codex subtracts cached); map cache_read
← `cache_read_input_token_cost` and cache_write ← `cache_creation_input_token_cost`;
missing cache_read on a legacy model falls back to the input rate (never silent
$0), missing cache_write stays $0 (correct for OpenAI/Gemini); always carry
both api-equivalent (cache reads at list) and subscription-equivalent (cache
reads free on Claude Max/Pro) bases.
