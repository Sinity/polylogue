---
created: "2026-06-28"
purpose: "Pricing-source coverage for models present in the real operator archive (litellm vendored JSON vs tokencost 0.1.26)"
status: complete
project: polylogue
---

# Pricing Coverage Matrix — archive models vs litellm JSON vs tokencost 0.1.26

## Method / sources

- **Archive (real corpus):** `/home/sinity/.local/share/polylogue/index.db` (38 GB), opened
  read-only (`mode=ro&immutable=1`). The configured `$POLYLOGUE_ARCHIVE_ROOT=/tmp/polylogue-archive`
  is the demo archive (660 KiB) — NOT used; the operator corpus is the data-home `index.db`.
- Model volumes from two tables:
  - `session_provider_usage_events` (PUE) — provider-reported token events; this is the *priced* lane
    for Codex/gpt models (Claude rows here carry 0 tokens; their tokens land in SMU).
  - `session_model_usage` (SMU) — per-session rollups, includes Claude `cost_usd` where present.
- **litellm:** vendored `polylogue/archive/semantic/data/litellm_model_prices.json` (1.5 MB).
  Matched by exact key, `.`→`-` normalization, and **any path-prefix** (`openai/`, `chatgpt/`, …)
  via last-segment index. Cache = has `cache_read_input_token_cost` / `cache_creation_input_token_cost`.
- **tokencost 0.1.26:** not installed in any env; pulled the wheel
  (`pip download tokencost==0.1.26`) and read its bundled `tokencost/model_prices.json` directly.
- Also noted: `index.db` carries its own materialized `model_prices` table (26 rows) — a small
  curated catalog (claude-3.x/4.x family, gemini-1.5/2.x, gpt-3.5/4/4o, o1/o3). It does **not**
  contain any gpt-5.x or codex entries, so it is not the pricing source for the high-volume models.

## Coverage matrix (models with real token volume — the ones that matter)

Token column = (PUE + SMU) input+output tokens, in billions. Cache = litellm has cache fields.

| model | PUE rows | SMU rows | tok (B) | in litellm (key) | cache | in tokencost | recommended source |
|---|---:|---:|---:|---|:--:|---|---|
| gpt-5.4 | 899,093 | 531 | 756,906 | ✅ `gpt-5.4` | Y | ❌ | **litellm** |
| gpt-5.5 | 258,180 | 426 | 251,386 | ✅ `gpt-5.5` | Y | ❌ | **litellm** |
| gpt-5.2 | 112,286 | 29 | 43,999 | ✅ `gpt-5.2` | Y | ❌ | **litellm** |
| gpt-5.2-codex | 95,422 | 88 | 15,753 | ✅ `gpt-5.2-codex` | Y | ❌ | **litellm** |
| gpt-5-codex | 250,179 | 454 | 8,668 | ✅ `gpt-5-codex` | Y | ❌ | **litellm** |
| gpt-5.1-codex-max | 60,015 | 70 | 8,654 | ✅ `gpt-5.1-codex-max` | Y | ❌ | **litellm** |
| gpt-5.3-codex | 84,626 | 405 | 7,480 | ✅ `gpt-5.3-codex` | Y | ❌ | **litellm** |
| gpt-5.1-codex | 40,612 | 22 | 4,657 | ✅ `gpt-5.1-codex` | Y | ❌ | **litellm** |
| gpt-5.3-codex-spark | 20,847 | 241 | 1,214 | ✅ `chatgpt/gpt-5.3-codex-spark` | Y | ❌ | **litellm** (chatgpt/ prefix) |
| gpt-5.1 | 7,052 | 10 | 155 | ✅ `gpt-5.1` | Y | ❌ | **litellm** |
| gpt-5.1-codex-mini | 294 | 7 | 46 | ✅ `gpt-5.1-codex-mini` | Y | ❌ | **litellm** |
| gpt-5.4-mini | 1,477 | 47 | 17 | ✅ `gpt-5.4-mini` | Y | ❌ | **litellm** |
| deepseek-v4-pro | 142,771 | 776 | 0.47 | ✅ `deepseek-v4-pro` | Y | ❌ | **litellm** |
| deepseek-v4-flash | 44,200 | 542 | 0.18 | ✅ `deepseek-v4-flash` | Y | ❌ | **litellm** |
| claude-opus-4-8 | 86,354 | 407 | 0.14 | ✅ `claude-opus-4-8` | Y | ❌ | **litellm** |
| gpt-5 | 18 | 59 | 0.11 | ✅ `gpt-5` | Y | ✅ `gpt-5` | litellm (parity) |
| claude-opus-4-7 | 114,819 | 669 | 0.07 | ✅ `claude-opus-4-7` | Y | ❌ | **litellm** |
| claude-opus-4-6 | 155,249 | 1,884 | 0.07 | ✅ `claude-opus-4-6` | Y | ❌ | **litellm** |
| claude-haiku-4-5-20251001 | 136,970 | 2,940 | 0.05 | ✅ `claude-haiku-4-5-20251001` | Y | ❌ | **litellm** |
| claude-sonnet-4-6 | 131,217 | 1,033 | 0.04 | ✅ `claude-sonnet-4-6` | Y | ❌ | **litellm** |
| claude-opus-4-5-20251101 | 83,849 | 587 | 0.02 | ✅ `claude-opus-4-5-20251101` | Y | ❌ | **litellm** |
| claude-sonnet-4-20250514 | 102,291 | 155 | 0.02 | ✅ | Y | ✅ | litellm (parity) |
| claude-opus-4-20250514 | 89,416 | 237 | 0.02 | ✅ | Y | ✅ | litellm (parity) |
| claude-sonnet-4-5-20250929 | 39,054 | 560 | 0.01 | ✅ `claude-sonnet-4-5-20250929` | Y | ❌ | **litellm** |
| claude-fable-5 | 1,363 | 3 | 0.003 | ✅ `claude-fable-5` | Y | ❌ | **litellm** |
| claude-opus-4-1-20250805 | 8,924 | 20 | 0.002 | ✅ | Y | ✅ | litellm (parity) |
| claude-3-7-sonnet-20250219 | 946 | 8 | 0.001 | ✅ | Y | ✅ | litellm (parity) |
| gemini-3.1-pro-preview | 0 | 2 | 0.0 | ✅ `gemini-3.1-pro-preview` | Y | ❌ | litellm |

**Every model carrying real token volume is covered by the vendored litellm JSON, with cache pricing fields.**

## Legacy / zero-volume models (SMU-only, 0 priced tokens — ChatGPT web/older API labels)

litellm covers the genuine API names: `gpt-4` (715), `gpt-4o` (512), `gpt-4o-mini` (13),
`gpt-5-pro` (142, no cache fields), `gpt-5-mini` (4), `o1` (99), `o3` (148), `o3-mini` (2),
`o4-mini` (2). tokencost also covers most of these plus `o1-mini`/`o1-preview`.

## Coverage holes (priced by NEITHER source)

All true holes have **0 provider-usage token volume** — they are ChatGPT consumer-web surface
labels (no API token accounting exists for them anywhere) or non-priceable tags:

- **`local-llama`** (4 PUE / 2 SMU, ~0 tok) — local model, intentionally unpriced.
- **`alpha:n7jupd:alpha.agent_with_prompt_expansion_2`** (2 SMU) — internal/experimental tag.
- ChatGPT web product variants with no token accounting (all 0 priced tokens):
  `gpt-5-1`, `gpt-5-1-instant`, `gpt-5-1-pro`, `gpt-5-1-thinking`, `gpt-5-2`, `gpt-5-2-instant`,
  `gpt-5-2-pro`, `gpt-5-2-thinking`, `gpt-5-3`, `gpt-5-4-pro`, `gpt-5-4-thinking`,
  `gpt-5-5-instant`, `gpt-5-5-pro`, `gpt-5-5-thinking`, `gpt-5-auto-thinking`, `gpt-5-instant`,
  `gpt-5-t-mini`, `gpt-5-thinking`, `o4-mini-high`, `o3-mini-high`, `research`.
  (Note: these are dash-form web aliases, e.g. `gpt-5-2-pro`; the dotted API forms `gpt-5.2` etc.
  ARE priced. The dash variants are distinct ChatGPT-UI session labels.)
- `gpt-4-gizmo` (70), `gpt-4-5` (108), `gpt-4-1` (28), `gpt-4-browsing` (23),
  `gpt-4-code-interpreter` (15), `gpt-4-plugins` (9), `gpt-4-dalle` (5),
  `gpt-4o-canmore` (5), `gpt-4o-jawbone` (1) — ChatGPT GPTs/tool-mode web labels.
- `text-davinci-002-render`, `-render-paid`, `-render-sha` (84) — legacy ChatGPT web engine ids.

None of these holes are billing-significant: zero priced tokens, so they contribute nothing
to cost rollups regardless of pricing source.

## Conclusions

1. **litellm vendored JSON is the correct/sufficient pricing source.** It covers 100% of
   archive models that carry token volume, including the recent ones (gpt-5.4, gpt-5.5,
   gpt-5-codex, gpt-5.3-codex-spark, claude-opus-4-8, deepseek-v4-*), all with cache pricing.
2. **tokencost 0.1.26 lags badly** — it has NONE of gpt-5.x, gpt-5.x-codex, claude-opus-4-5/6/7/8,
   deepseek-v4-*, claude-fable-5. Its only overlap with the archive's recent models is the
   May-2025 dated Claude snapshots (`claude-opus-4-20250514`, `claude-opus-4-1-20250805`,
   `claude-sonnet-4-20250514`, `claude-3-7-sonnet-20250219`) plus legacy gpt-4/4o/o-series.
   It must not be relied on for the high-volume corpus.
3. **No real (token-bearing) coverage hole exists.** Every "neither" model has 0 provider-usage
   tokens; they are ChatGPT-web product labels and local/experimental tags, not API-priced models.
4. **Prefix gotcha:** litellm keys are namespaced (`openai/…`, `chatgpt/…`). A naive exact-match
   lookup falsely flags `gpt-5.3-codex-spark` as missing — it lives under `chatgpt/gpt-5.3-codex-spark`.
   Any pricing lookup against this JSON must match on the last path segment, not just the bare name.
