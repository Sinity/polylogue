---
created: "2026-06-28"
purpose: "Index of the cost/usage analytics program research wave (12 subagents)"
status: active
project: polylogue
branch: feature/dogfood/cost-usage-analytics-program
---

# Cost/Usage Analytics Program — Research Wave Index

12 read-only research subagents run 2026-06-28 against the live 38 GB archive,
advancing the program agenda A–G (see `../000-capability-log.md`). Each file holds
the full analysis; headlines + actionability below.

| # | File | Headline finding | Actionable now? |
|---|------|------------------|-----------------|
| 01 | `01-cache-pricing-policy.md` | `_cost_components` already bills 4 disjoint lanes correctly (Anthropic convention). Gap: legacy models (gpt-4/4o/3.5/o1-mini) lack `cache_read` → must fall back to fresh-input rate, never silent $0. Carry both `api_equivalent` + `subscription_equivalent` bases. | Small fix: legacy cache_read fallback |
| 02 | `02-codex-token-overcount.md` | **376.6B `session_model_usage` = STALE pre-`3938bc6c2` rows** (re-ingest clears). Residual **per-model partitioning bug**: `_aggregate_provider_usage_into_model_usage` (write.py:2311-2418) + `usage.py:962-1000` split a *session-global* cumulative counter by `model_name` → 199.6B vs correct 177.1B. 270.8B = SUM(last_*) = request-volume, must be relabeled. | YES (= #2472) |
| 03 | `03-server-tool-use-capture.md` | `message.usage.server_tool_use.{web_search_requests,web_fetch_requests}` dropped at `code_parser.py:304-307`. Tier A fix: add to event payload → rides into `payload_json`, NO schema bump. | YES (small, no bump) |
| 04 | `04-crossverify-sources.md` | Authoritative sources mapped: `stats-cache.json` (per-model 4-lane lifetime, costUSD=0), `usage-data/session-meta/*.json` (per-session join bridge), `state_5.sqlite threads.tokens_used` (cumulative, cache-incl). `logs_2.sqlite`/`history.jsonl` = no token ledger. | Design for step D |
| 05 | `05-fold-forensics-into-insights.md` | 16 analytics; most already materialized (`cost_rollups` incl. `sub_usd`!). 4 genuine gaps: monthly rollup, model×month table, session-cost/length distributions, cache-amplification ratio. Script's hand-rolled she-llac rates are redundant. | Step E (= #2480) |
| 06 | `06-atropos-export.md` | Atropos `ScoredDataGroup` TypedDict captured verbatim. Polylogue has no token-ids/masks → honest **Viewer tier** (`tokens=[]`, scores from pathologies). `polylogue export atropos --group-by session\|lineage`. Vendor the TypedDict. | Step F design ready |
| 07 | `07-reprice-in-place-perf.md` | Reprice is **sub-second**: only `session_model_usage` (~15.7K) + `session_profiles` (~16.4K) carry cost; 2.97M-row events table has NO cost col (red herring). Price table already in DB. Python-driven per-distinct-model UPDATEs (model_name stored RAW needs `_normalize_model`). | Implements after #2472 |
| 08 | `08-ingestion-snappiness-dogfood.md` | `full.index.full_replace` FTS suspend/insert/restore is the re-ingest tax: large Codex files up to **37 min/file**. `live_ingest_attempt` already carries files/s, MB/s, per-stage timings, RSS, cgroup mem. `ops diagnostics workload --compare` diffs it. (ties #2391) | Measurement plan ready |
| 09 | `09-pricing-coverage-matrix.md` | **litellm is the SOLE correct pricing source** — 100% of token-bearing models incl. gpt-5.4/5.5/codex, opus-4-8, deepseek-v4. **tokencost lags badly → drop reliance.** Gotcha: litellm keys are prefix-namespaced (`chatgpt/gpt-5.3-codex-spark`) → lookup MUST match last path segment. | YES (verify lookup; drop tokencost) |
| 10 | `10-subscription-credit-model.md` | **BUG: `MODEL_CREDIT_RATES` output_credits = input_credits (missing 5× multiplier)** → understates output 5×. `subscription_equivalent_usd` hardcodes Pro basis. Credit formula + plan pools documented (Pro 21.7M, Max5× 180.6M, Max20× 361.1M). | YES (isolated bug) |
| 11 | `11-program-issue-map.md` | Next-3: **#2472** (logical-basis token count = unfinished half of B), **#2475** (memoize composition + FTS docsize scan — land before big re-ingest), **#2470** (compose forks on ALL read surfaces — #2469 wired only 2 of ~5 → CLI/exports truncate fork transcripts). #818/#1503 already closed. C has no issue → file one. | Roadmap |
| 12 | `12-lineage-validation.md` | Baseline (pre-re-ingest, user_version=11): **16,483 physical vs 8,809 logical sessions (1.87×)**; 286 Codex continuations = 1.77M msgs (~31% of all rows); 187 Claude acompact = 63.9K. ~7,450 Task subagents are LEGIT. Snapshot sample transcripts NOW for exact post-re-ingest equality check. | Validation gated on re-ingest |

## Highest-confidence shippable now (no re-ingest needed)
1. **#10 credit 5× output bug** — isolated, wrong-by-construction.
2. **#03 server_tool_use Tier-A capture** — parser-only, no schema bump.
3. **#09 litellm last-segment lookup** + drop tokencost reliance — verify + tighten.
4. **#01 legacy cache_read fallback** — small pricing correctness.
5. **#2470 read-path composition** — correctness regression surface from #2469's partial wiring.

## Gated on operator-run re-ingest (`ops reset --database && polylogued run`)
- #2472 per-model partitioning + clearing stale `session_model_usage` (#02)
- reprice-in-place (#07), cross-verify (#04, #2481), lineage validation (#12)
- #2475 composition memoization should land BEFORE that re-ingest (perf)
