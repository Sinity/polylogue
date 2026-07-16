---
created: "2026-06-28"
purpose: "Schema docs + cross-verification design for external token/cost authoritative sources vs Polylogue derived numbers"
status: "complete"
project: "polylogue"
---

# Cross-Verification Sources: Authoritative Token/Cost Data vs Polylogue Derived

Read-only investigation. None of the source files were modified. They are the
operator's live data and are **not yet ingested** into Polylogue.

Goal: a feature that compares Polylogue's *derived* token/cost numbers
(`session_model_usage`, `session_provider_usage_events`) against *authoritative
external* numbers the providers themselves emit.

---

## 1. Claude — `~/.config/claude/stats-cache.json`

- Mode: `0600`, ~64 KB, single JSON object. Written by Claude Code itself
  (`version: 4`, `lastComputedDate: 2026-06-21`). This is the CLI's own rolled-up
  usage cache — authoritative for *aggregate* Claude Code token usage.
- `costUSD` is **always 0** across every model (subscription; CLI does not price).
  So this source is authoritative for **tokens**, never for dollars.

### Top-level keys

| Path | Type | Meaning |
| --- | --- | --- |
| `version` | int | cache schema version (currently 4) |
| `lastComputedDate` | str `YYYY-MM-DD` | last recompute; **freshness anchor** |
| `dailyActivity` | list[221] | per-day `{date, messageCount, sessionCount, toolCallCount}` |
| `dailyModelTokens` | list[215] | per-day `{date, tokensByModel: {model: tokens}}` time-series |
| `modelUsage` | dict[14 models] | **lifetime per-model token totals** (the key table) |
| `totalSessions` | int | 2522 |
| `totalMessages` | int | 1555901 |
| `longestSession` | dict | `{sessionId, duration, messageCount, timestamp}` |
| `firstSessionDate` | str ISO | `2025-05-21T...` |
| `hourCounts` | dict[`"0".."23"`] | message count by hour-of-day |
| `totalSpeculationTimeSavedMs` | int | speculative-decoding savings |

### `modelUsage[<model>]` — authoritative lifetime per-model totals

Exact paths (e.g. `modelUsage["claude-sonnet-4-5-20250929"]`):

| Field | Type | Notes |
| --- | --- | --- |
| `inputTokens` | int | **non-cached** input only (small; e.g. 1.5M) |
| `outputTokens` | int | output tokens |
| `cacheReadInputTokens` | int | cache reads — dominant lane (e.g. 2.19B) |
| `cacheCreationInputTokens` | int | cache writes (e.g. 162M) |
| `webSearchRequests` | int | usually 0 |
| `costUSD` | int | **always 0** (subscription, not priced) |
| `contextWindow` | int | 0 in cache |
| `maxOutputTokens` | int | 0 in cache |

Aggregate sanity (all 14 models): input ≈ 674.5M, output ≈ 298.3M,
cacheRead ≈ 151.4B. The four token lanes map 1:1 onto Polylogue's
`session_model_usage` lanes (see §5). Note: `inputTokens` here is **input
EXCLUDING cache** — important; do not compare it to a Polylogue "input that
includes cache" number.

Models present include dated (`claude-sonnet-4-5-20250929`,
`claude-opus-4-1-20250805`, `claude-3-7-sonnet-20250219`...), undated rolling
(`claude-opus-4-6/4-7/4-8`, `claude-sonnet-4-6`), and non-Anthropic
(`deepseek-v4-pro`, `deepseek-v4-flash`) — DeepSeek routed through Claude Code.

### `dailyModelTokens[i]` — per-day per-model time series

```
{ "date": "2026-06-15",
  "tokensByModel": { "claude-opus-4-8": 8021112, "claude-haiku-4-5-...": 242360, ... } }
```
Single scalar per (day, model). **No in/out/cache split per day** — only the
lifetime `modelUsage` has the 4-lane split. This is the spine for a
**daily-granularity** cross-check; `modelUsage` is the **lifetime** cross-check.

---

## 2. Claude — `~/.config/claude/usage-data/`

Per-session sidecar JSON written by Claude Code, **with per-session token
counts** — this is the bridge to Polylogue's per-session rows (stats-cache only
has lifetime + daily, not per-session tokens).

- `facets/` — 100 files `<session_id>.json`: LLM-graded session quality
  (`underlying_goal`, `goal_categories`, `outcome`, `claude_helpfulness`,
  `session_type`, `friction_*`, `brief_summary`, `session_id`). **No tokens** —
  qualitative only. Not a cost-verify source, but useful enrichment.
- `report.html` — 67 KB static rendered report (the `claude usage` UI dump).
- `session-meta/` — **414 files `<session_id>.json` with per-session metrics**:

  | Field | Notes |
  | --- | --- |
  | `session_id` | joins to Polylogue `sessions.session_id` for Claude Code |
  | `project_path` | cwd |
  | `start_time` ISO, `duration_minutes` | |
  | `user_message_count`, `assistant_message_count` | |
  | `tool_counts` {tool: n}, `languages`, `git_commits`, `git_pushes` | |
  | **`input_tokens`**, **`output_tokens`** | per-session totals (NO cache split here) |
  | `user_interruptions`, `user_response_times`, `tool_errors`, ... | behavioral |

`session-meta.input_tokens/output_tokens` is the **authoritative per-session**
Claude token number — the one to diff against Polylogue's
`session_model_usage` summed per `session_id`. Caveat: session-meta has only
input+output (no cache lanes), and 414 meta files vs 2522 lifetime sessions
(only a recent window is retained).

---

## 3. Codex — `~/.codex/state_5.sqlite`

sqlx-managed (`_sqlx_migrations`). The authoritative Codex per-thread token
source. Tables: `threads`, `thread_dynamic_tools`, `thread_spawn_edges`,
`agent_jobs`, `agent_job_items`, `backfill_state`,
`remote_control_enrollments`, `external_agent_config_imports`.

### `threads` (2289 rows) — the cross-verify table

Relevant columns:

| Column | Type | Notes |
| --- | --- | --- |
| `id` | TEXT PK | thread/session id → Polylogue Codex `sessions.session_id` |
| `rollout_path` | TEXT | path to the `~/.codex/sessions/**.jsonl` rollout |
| `created_at`, `updated_at` | INTEGER (unix s) | also `_ms` variants via triggers |
| `model` | TEXT (nullable) | e.g. `gpt-5.5`, `gpt-5.4`, `gpt-5.2-codex` |
| `model_provider` | TEXT | `openai` / `local` |
| **`tokens_used`** | INTEGER | **authoritative cumulative token total for the thread** |
| `cwd`, `title`, `first_user_message`, `preview` | | |
| `source`, `thread_source`, `reasoning_effort`, `cli_version` | | |
| `git_sha`, `git_branch`, `git_origin_url`, `archived`, `archived_at` | | |

**Critical semantics for `tokens_used`** (matches MEMORY note on Codex token
semantics): it is a **single cumulative scalar** — no in/out/cache breakdown,
and it **includes cached input** (Codex input ≈ 96% cache). Values are huge:
top thread 4.45B tokens; lifetime sum across threads ≈ 139.3B.

Aggregation:
```sql
-- per model
SELECT model, model_provider, COUNT(*) threads, SUM(tokens_used) tokens
FROM threads GROUP BY model, model_provider ORDER BY tokens DESC;
-- per thread (direct join key to polylogue)
SELECT id, model, tokens_used FROM threads;
```
Per-model totals observed: blank-model (older threads, no `model` populated)
`openai` 1621 threads / 65.6B; `gpt-5.5` 432 / 40.8B; `gpt-5.4` 119 / 31.6B;
plus codex/spark/mini variants and 2 `local-llama`. The 1621 blank-model
threads are a real gap — those cannot be attributed to a model from this table.

`thread_spawn_edges(parent_thread_id, child_thread_id, status)` mirrors
Polylogue's subagent/fork lineage — useful to avoid double-counting spawned
children when rolling thread tokens up to a logical session.

### Authoritative-source caveat
The richest Codex signal Polylogue actually parses is the per-event
`token_count` records inside the rollout JSONL (`~/.codex/sessions/` exists,
years 2025/2026) → these feed `session_provider_usage_events`. `state_5`
`tokens_used` is the **independent cross-check**: same provider, different
accounting path (the daemon's own running tally). MEMORY notes per-thread median
ratio 1.00 between them — so equality is the expected baseline.

---

## 4. Codex — `~/.codex/logs_2.sqlite` and `~/.codex/history.jsonl`

Neither is a clean token ledger; documented for completeness / to rule out.

- **`logs_2.sqlite`** (516 MB): tables `_sqlx_migrations`, `logs` (185,768
  rows). `logs` schema: `id, ts, ts_nanos, level, target, feedback_log_body,
  module_path, file, line, thread_id, process_uuid, estimated_bytes`. Targets
  are OTLP/trace/daemon (`codex_otel.*`, `codex_api::endpoint::*`,
  `app_server.request{...}`). `feedback_log_body` is free-form trace text, not
  structured token fields. **Not** a token source — skip for cross-verify
  (would require fragile body parsing). `thread_id` joins to `threads.id` if
  ever needed for trace correlation.
- **`history.jsonl`** (27 MB, 16,020 lines): one object per line
  `{session_id, ts, text}` — the user **prompt history** only (e.g. `"update"`).
  No tokens, no assistant content. Useful only as a prompt-presence/ordering
  cross-check, not for cost.

---

## 5. Cross-Verification Feature Design

### Principle
For each provider, pick the *authoritative external* number and the *Polylogue
derived* number computed over the **same scope** (lifetime / daily / per
session / per model), normalize lane definitions, then flag discrepancies above
a threshold. Every comparison must state its lane contract because the failure
mode here is lane-mismatch (cache in/out, reasoning tokens), not arithmetic.

### Polylogue derived side (recap of columns)
- `session_model_usage(session_id, model_name, input_tokens, output_tokens,
  cache_read_tokens, cache_write_tokens, message_count, cost_usd, cost_credits,
  cost_provenance)` — per (session, model) rollup. **This is the primary
  comparison surface.**
- `session_provider_usage_events(session_id, position, provider_event_type,
  model_name, last_*_tokens, total_input/output/cached_input/cache_write/
  reasoning_output/total_tokens, ...)` — per-event raw provider usage
  (`token_count` for Codex, `message_usage` for Claude). The `total_*` columns
  are the running/cumulative form; `last_*` is the latest snapshot.

### Mapping table

| Provider | Scope | Authoritative external | Polylogue derived | Lane contract |
| --- | --- | --- | --- | --- |
| Claude | lifetime / model | `stats-cache modelUsage[m].{inputTokens, outputTokens, cacheReadInputTokens, cacheCreationInputTokens}` | `SUM(session_model_usage.{input,output,cache_read,cache_write}) GROUP BY model_name` over Claude-origin sessions | 4 lanes map 1:1. `inputTokens` = **non-cache** input only → compare to `input_tokens`, NOT input+cache. |
| Claude | daily / model | `stats-cache dailyModelTokens[d].tokensByModel[m]` (scalar = in+out+cache?) | per-day SUM of all four lanes per model from `session_model_usage` joined to session date | Daily scalar has no lane split; compare the **summed total** lane, allow loose threshold. |
| Claude | per session | `usage-data/session-meta/<sid>.json.{input_tokens, output_tokens}` | `SUM(session_model_usage.{input,output}) WHERE session_id=sid` | session-meta has **no cache lanes** → compare input & output only. |
| Codex | per thread | `state_5.threads.tokens_used` (cumulative, **includes cache**) | `session_provider_usage_events` latest `total_tokens` for the session (or `SUM(session_model_usage.input+output+cache_read+cache_write)`) | Both include cache → compare the **all-in total**; do NOT split. |
| Codex | per model | `SUM(threads.tokens_used) GROUP BY model` | `SUM(all lanes) GROUP BY model_name` over Codex sessions | Exclude the 1621 blank-`model` threads from per-model (un-attributable); keep them in the lifetime total. |
| Codex | lineage | `thread_spawn_edges` parent/child | Polylogue logical-session lineage | De-dup spawned children before rollup to avoid double count. |

### Join keys
- Claude Code: `sessions.session_id` ↔ `session-meta` filename / `facets.session_id`.
  stats-cache has **no** session ids except `longestSession.sessionId`.
- Codex: `threads.id` ↔ Polylogue Codex `sessions.session_id`. Also
  `threads.rollout_path` ↔ Polylogue source raw artifact path.

### Cost note
Neither authoritative source carries trustworthy dollars (`stats-cache.costUSD`
= 0; Codex has no cost column). Cross-verification is **token-only**; cost
verification stays internal to Polylogue's price-catalog pricing
(`session_model_usage.cost_usd/cost_credits/cost_provenance`). Do not try to
diff dollars against these sources.

### Discrepancy thresholds (proposed)
Report a comparison as one of `match / drift / mismatch / missing`:

- **match**: |Δ| / max(authoritative, derived) ≤ **1%** *per lane* (token
  counters are integer-exact in principle; allow 1% for ingest-window edges and
  off-by-a-few-events).
- **drift (warn)**: 1% < relative Δ ≤ **10%** on any lane — usually a partial
  ingest window or a freshness skew (`lastComputedDate` vs Polylogue ingest
  cursor). Surface, don't fail.
- **mismatch (error)**: relative Δ > **10%** on a lane, OR a **lane-direction
  inversion** (e.g. Polylogue shows cache≈0 where authoritative shows cache
  dominant → fabricated/zeroed lane bug), OR Codex per-thread ratio outside
  **[0.8, 1.25]** (baseline expectation is ratio 1.00).
- **missing**: session/model present in one side, absent in the other. Expected
  asymmetries to whitelist: stats-cache 2522 lifetime sessions vs 414
  session-meta files (retention window); Codex 1621 blank-`model` threads
  (un-attributable per-model, fine in lifetime total).

### Freshness gating
Always read `stats-cache.lastComputedDate` and compare to the Polylogue ingest
high-water mark; if authoritative is staler than derived (or vice versa), scope
the comparison to the overlapping date window before computing Δ, otherwise the
newer side trivially "mismatches". Codex `state_5` is live (updated continuously)
so gate Codex comparisons on `threads.updated_at`.

### Suggested surface
A read-only `devtools` or `polylogue ops diagnostics` command that:
1. loads each authoritative source (no ingestion, no writes to source files),
2. computes the derived side from `index.db`,
3. emits a JSON report: per (provider, scope, lane) `{authoritative, derived,
   delta, rel, status}` with the whitelisted asymmetries annotated.
This mirrors the existing `ops diagnostics workload` evidence-snapshot pattern.
