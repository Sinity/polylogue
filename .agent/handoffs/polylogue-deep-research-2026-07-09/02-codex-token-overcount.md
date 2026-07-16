---
created: "2026-06-28"
purpose: "Root-cause the Codex token-accounting discrepancy: session_model_usage 376.6B vs provider-events last_* 270.8B vs authoritative 139.3B"
status: "complete"
project: "polylogue"
---

# Codex token over-count — root cause

## TL;DR

There is no single bug; there is a **ladder of four measures**, each over-counting
the one below it for a distinct, identifiable reason. Authoritative
`~/.codex/state_5.sqlite threads.tokens_used` = **139.3B** is the *latest cumulative
`total_token_usage.total_tokens` per thread*. Polylogue already stores that exact value
correctly in `session_provider_usage_events.total_tokens`; taking the latest reproduces
it (median per-thread ratio **1.000**, top thread matches to the token). Every larger
internal number is an aggregation that fails to dedupe a session-global cumulative
counter.

| Measure | Value | Why it is larger |
| --- | --- | --- |
| authoritative `threads.tokens_used` | 139.3B | latest cumulative total per thread |
| polylogue latest `total_tokens` (per **session**) | 140.6B (joined) / 177.1B (all 2243 codex sessions) | matches authoritative on joined threads; the all-sessions figure is inflated by **lineage duplication (#2467)** |
| current materializer output (per **session,model**) | **199.6B** | partitions a session-global cumulative by `model_name` → multi-model sessions (95) double/triple-count |
| `session_model_usage` rows **in the live DB** | **376.6B** | **stale rows predating the `3938bc6c2` disjoint-lanes fix**: cached input counted twice (input still includes cached, AND cache_read = cached) |
| `_provider_event_stats.provider_request_usage` = SUM(last_*) | **270.8B** | sums every per-request `last_*`; each Codex request re-includes the ~96% cached prefix |

All numbers above were measured directly against the live `index.db`
(38 GB, opened read-only) and `state_5.sqlite`.

## Evidence (live archive, read-only)

- `sessions` origin counts: `codex-session` = 2436 (authoritative threads = 2289).
- `session_model_usage` for codex: input=191.4B, output=0.7B, cache_read=184.6B,
  cache_write=0 → **ALLSUM = 376.6B** (matches the reported figure).
- `session_provider_usage_events` (token_count, 1.84M rows):
  - `SUM(last_total_tokens)` = **270.8B** (matches reported), `SUM(last_input)`=268.3B,
    `SUM(last_cached)`=257.6B → cached is **96%** of last_input.
  - `SUM(total_tokens)` = 1,105,425B (cumulative summed naively → meaningless; nobody
    should ever `SUM()` the `total_*` columns).
- Latest `total_tokens` per session (max position) = 177.1B over all codex sessions.
- **Join on thread id** (`session_id = codex-session:<thread-uuid>`), 2147 common threads:
  - `SUM(auth.tokens_used)` = 139.3B
  - `SUM(polylogue latest cumulative total_tokens)` = **140.6B**, median ratio **1.000**,
    top thread 4.448B == 4.448B exactly.
  - `SUM(polylogue summed last_total)` = 234.2B, median ratio 1.017 (aggregate 1.68×
    because long, many-turn sessions dominate).
- Codex **messages carry no per-message token counts** (0 rows with tokens) → the
  per-message aggregation path is not a contributor for Codex.

## What the code does

### Parser — emits BOTH per-request and cumulative
`polylogue/sources/parsers/codex.py`
- `_codex_token_usage_payload` (150-167): reads `input_tokens`, `cached_input_tokens`,
  `output_tokens`, `reasoning_output_tokens`, `total_tokens`.
- `_compact_response_payload` token_count branch (268-289): emits `last_token_usage`
  (per-request, 280-281) and `total_token_usage` (cumulative, 282-283). Codex `input`
  is **inclusive of cached**, `output` is **inclusive of reasoning** (confirmed: cached
  = 96% of input on the corpus).

### Event writer — stores both lanes faithfully
`polylogue/storage/sqlite/archive_tiers/write.py:2235-2277` `_write_provider_usage_event`
writes `last_*` and `total_*` columns. This is correct; the raw evidence is intact.

### Bug 1 (stale rows, explains 376.6B): disjoint-lanes fix not applied to stored rows
`_provider_usage_disjoint_lanes` (2280-2308) subtracts cached from input
(`fresh_input = max(input - cache_read, 0)`, line 2307) so cached is not billed twice.
This is commit `3938bc6c2`. **The live `index.db` rows predate it**: stored
input(fresh)=191.4B but cache_read=184.6B, i.e. input was NOT reduced by cache
(if the fix were applied, fresh would be ~7B). Replaying the *current* algorithm over
the live events yields **199.6B, not 376.6B** — proving the stored rows are stale.

### Bug 2 (per-model partition, 199.6B vs 177.1B): cumulative counter split by model
`_aggregate_provider_usage_into_model_usage` (2311-2418) keeps
`latest_total_by_model[model_name]` (2348, 2372) — the latest cumulative **per model**.
But `total_token_usage` is a **single session-global running counter**, not per-model.
A session whose token_count events carry different `model_name` values over its life
(model upgrade mid-session, or label drift) records each model's near-final cumulative
and **sums them** (2390-2402). 95 multi-model codex sessions inflate 177.1B → 199.6B.
The appended path (2421-2516) and the report's `_provider_cumulative_usage`
(`usage.py:962-1000`, keyed by `model_key` at 970/988) have the identical partition flaw.

### Bug 3 (mislabeled, 270.8B): summing per-request last_*
`polylogue/storage/usage.py:_provider_event_stats` (913-959) builds
`provider_request_usage` as `SUM(last_*)` across every token_count event (927-928).
For Codex this is a **request-volume** metric (each request re-sends the cached prefix),
NOT "tokens used". Summed it is 270.8B and must never be presented as a session/thread
token total.

### Residual (177.1B vs 139.3B): lineage duplication (#2467)
Per-session latest cumulative is still 1.27× authoritative because forked/resumed
Codex sessions carry cumulative totals that include the inherited parent prefix.
Joining on thread id collapses this to 140.6B ≈ 139.3B. Out of scope for token
accounting per se; the fix is logical-session/thread dedupe (#2467).

## Fix plan

1. **Primary, no code change — re-ingest / rebuild the index tier.** The live
   `session_model_usage` (376.6B) is stale, predating `3938bc6c2`. Per the
   schema-versioning re-ingest policy: `polylogue ops reset --database && polylogued run`.
   After rebuild the figure drops to ~199.6B (the current algorithm's output), at which
   point Bug 2 becomes the dominant internal error.

2. **Bug 2 — change per-(session,model) to per-session latest cumulative.**
   In `write.py` `_aggregate_provider_usage_into_model_usage` (2311-2418) and
   `_aggregate_appended_provider_usage_into_model_usage` (2421-2516): track ONE latest
   cumulative `total_*` tuple per session (highest `position`), attribute it to the
   `model_name` carried on that final cumulative event, and write a single rollup row.
   Keep the per-model split only for the `summed_last_by_model` fallback (sessions with
   no `total_*` at all). Apply the same change to `usage.py:_provider_cumulative_usage`
   (962-1000): drop `model_key` from the dedupe key, or pick the latest-position row per
   session and label by its model. Expected: 199.6B → 177.1B.

3. **Bug 3 — relabel, do not sum for totals.** `usage.py:_provider_event_stats`
   `provider_request_usage` (913-959) is per-request volume; surfaces must present the
   cumulative (latest `total_*`) measure as "tokens used" and `SUM(last_*)` only as
   "request volume / cache churn". Do not change the SUM itself; change its
   interpretation in any caller that treats it as a token total.

4. **Residual — #2467 lineage dedupe** collapses 177.1B → ~139.3B by counting logical
   threads, not physical sessions. Track there, not here.

## One-line correctness invariant for tests
For Codex: per-session "tokens used" MUST equal the `total_tokens` of the
**highest-position** token_count event for that session (not a SUM of `last_*`, not a
per-model sum of `total_*`). On the corpus this reproduces `state_5.sqlite.tokens_used`
to median ratio 1.000.
