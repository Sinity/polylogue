---
created: 2026-08-25
purpose: Bounded continuation of polylogue-rrxe4.8 — public contract-mask and cross-tier parity for cost/usage and durable assertions
status: complete — all investigated cells dispositioned; no genuinely new current-route defects found
project: polylogue
---

# Live route-mask continuation audit

## Authority and method

Continuation of `2026-08-24-live-db-correctness-audit.md` (polylogue-rrxe4.8). The prior
audit settled message/session-summary/attachment/topology/embeddings/ops-tier families.
This pass focused on cost/usage public projection and durable assertion surfaces, then verified
remaining session/topology/attachment cells are already owned.

Frozen physical snapshot reused from prior pass (no daemon activity since 2026-08-24).

---

## Coverage matrix — investigated row families × public surfaces

| Row family | CLI | Python API | MCP | HTTP | Novelty disposition |
|---|---|---|---|---|---|
| Session summary (`parent_id`, `branch_type`) | partial | drops `parent_id` | drops both | drops both | **Already owned / `blpir`** |
| Session cost (`sp.cost_usd` vs `sp.total_cost_usd`) | enriched | enriched | enriched | enriched | **Already owned / `f2qv.6`** |
| Coverage insight `total_cost_usd` | 0 (no profile join) | 0 (sp.cost_usd=NULL) | 0 | 0 | **Already owned / `f2qv.6`** |
| Usage timeline (`smu.cost_usd`) | real data | real data | real data | real data | **Already owned / `f2qv.6`** |
| Durable assertions | CLI only | CLI only | not exposed | not exposed | **By design** |
| Attachment public state | acquired→missing | acquired→missing | acquired→missing | acquired→missing | **Already owned / `hb9o6`** |
| Session topology edge type | full type | full type | loses type on unresolved | loses type on unresolved | **Already owned / `27ezu`** |

---

## Detailed findings

### Cost/usage route trace

**`list_session_cost_insights` → `_session_cost_insight_from_archive_row`** (storage/sqlite/archive_tiers/archive.py:7459)

- Reads `sp.cost_usd` (line 2539), not `sp.total_cost_usd`.
- Live state: `sp.cost_usd` is NULL for all 23,494 sessions; `sp.total_cost_usd` is nonzero
  for 10,327. Every session therefore enters the `unavailable` branch (line 7485) with
  `unavailable_reason = "no_tokens"` before enrichment.
- `enrich_session_cost_insight` (insights/cost_enrichment.py:30) compensates by re-deriving
  the estimate from the full session object via `estimate_session_cost`. This reads actual
  session messages and model-usage events, so the enriched estimate is correct when usage
  data exists.
- The stored `total_cost_usd` is intentionally bypassed because `f2qv.6` owns the transition
  to the authoritative `cost_usd/cost_credits/priced_with` triple. This is the designed
  transient state. **No new defect.**

**Thread insight `total_cost_usd`** (storage/sqlite/archive_tiers/archive.py:2409)

- Reads `sp.total_cost_usd AS profile_total_cost_usd` (old populated field).
- Thread insight total_cost_usd aggregates from the old field; session cost insight enriches
  from session content. Two different cost authorities are exposed on the same archive
  without a documented reconciliation. **Within `f2qv.6` scope — not new.**

**`ArchiveCoverageInsight.total_cost_usd`** (storage/sqlite/archive_tiers/read_insights.py)

- `_origin_coverage_insights` (line 191): no join to `session_profiles` at all.
  `total_cost_usd` stays at the model default `0.0`. No cost data in origin-grouped coverage.
- `_time_bucket_coverage_insights` (line 232): joins `session_profiles`, reads
  `SUM(COALESCE(sp.cost_usd, 0.0))` (line 266). `sp.cost_usd` is NULL for all rows →
  `total_cost_usd = 0.0` for all buckets.
- Both variants return 0 cost, but for different structural reasons (no join vs. null field).
  The `sp.total_cost_usd` (nonzero for 10,327) is never read by either path. **Within
  `f2qv.6` scope — not new.**

**`list_cost_rollup_insights`** (archive.py:2561)

- Reads `COALESCE(u.cost_usd, sp.cost_usd, 0.0)` where `u = session_model_usage`.
  `session_model_usage.cost_usd` has 10,245 nonzero rows; this is the cost authority for
  usage-bearing sessions. `sp.cost_usd` is always NULL so the fallback does nothing.
  **Correct for usage-bearing sessions; silent zero for non-usage-bearing sessions.
  Within `f2qv.6` scope.**

**`list_usage_timeline_insights`** (archive.py:2761)

- Reads `u.cost_usd` from `session_model_usage` (line 2917), not from `session_profiles`.
  Uses real usage data for sessions that have `session_model_usage` rows.
  `subscription_credits` is also computed from `u.cost_credits` with a fallback catalog
  computation. This is the most accurate cost surface currently available.
  **No new defect.**

**`enrich_session_cost_insight` N+1 overhead note** (insights/cost_enrichment.py:30)

- With the default `limit=50`, enrichment reads 50 full sessions per cost-list call.
  This is bounded and intentional. Coordinator may wish to note this in `f2qv.6`'s
  implementation guidance: once `sp.cost_usd` is populated, `_session_cost_insight_from_archive_row`
  can skip enrichment for `exact`/`priced` sessions, collapsing to a pure index read.

---

### Durable assertion projection

**CLI exposure** (cli/commands/judge.py, cli/commands/annotations.py,
cli/commands/maintenance/_assertion_export.py, cli/commands/status.py):

- `polylogue judge` — full `AssertionClaimPayload`/`AssertionQueryRowPayload` lifecycle
  review
- `polylogue annotations import` — batch import of candidate assertions
- `polylogue maintenance assertion-export` — JSONL export via `list_assertions_for_export`
- `polylogue status` — counts and target-ref summary in debt component

**MCP exposure**: zero. No MCP tool or resource exposes assertions. No assertion-related
operation verbs appear in `mcp/server_cutover.py` or `mcp/server_tools.py`.

**HTTP exposure**: zero. No route in `daemon/http.py` serves assertions.

**Python API exposure** (api/archive.py, api/contracts/assertions.py): read/write available
through the `Polylogue` class internally; no public async surface exposes a list-all-assertions
operation.

**Disposition**: Assertions are intentionally operator-only CLI/write tooling. The four
dangling session-target assertions (`b4n2`/`auhr8` cohort) remain the only known correctness
issue, already owned. **No new defect.**

---

### Session summary envelope field coverage

**`_SESSION_SUMMARY_MASK`** (surfaces/payloads.py:696):

Exposes: id, origin, title, title_source, title_ref, title_confidence, message_count,
created_at, updated_at.

Does NOT expose (selected by population impact):

| Field | Population in live index | Missing from |
|---|---|---|
| `parent_id` | 8,071 sessions | API summary mapper (`_archive_summary_to_domain`), summary envelope |
| `branch_type` | 14,575 sessions | `ArchiveSessionSummary` lacks the column entirely; not mappable from summary SQL path |
| `display_name` | 7,207 sessions | archive-query path only; API summary mapper carries it |

The three-mapper divergence (archive-query, API, exact-CLI) is already thoroughly documented
in `blpir`. **No new defect.**

**`branch_type` structural note** for `blpir`: `ArchiveSessionSummary` (archive.py:374)
does not carry a `branch_type` field, so the summary-row path cannot populate it regardless
of mapper completeness. The `branch_type` is only available through the full session read
(`ArchiveSessionEnvelope`) or through an additional join to `session_links.link_type`. `blpir`
must select a JOIN strategy for the summary path, not just a mapper fix.

---

### Remaining unexamined cells

The following were not investigated in this pass. No claim of completeness beyond the above.

1. **`ArchiveCoverageInsight` work_event_breakdown accuracy**: three subquery calls per
   bucket row may produce stale or misaligned breakdowns for large result sets. No
   denominator was established.
2. **`SessionProfileInsight` evidence/inference/enrichment payload field parity vs. stored
   JSON**: the `from_record` path reconciles `terminal_state`/`workflow_shape` from native
   columns back onto the payload (lines 272-298), but the full field inventory of what the
   evidence/inference JSON stores vs. what `SessionProfileRecord` carries in native columns
   was not compared.
3. **`CostRollupInsight` per_model_breakdown accuracy for multi-model sessions**: the
   rollup groups by `model_name` + `cost_provenance`; sessions that mix models create
   multiple rows. Whether the per-model token sums are correctly attributed per session
   was not verified.
4. **`UsageTimelineInsight` subscription_credits catalog computation accuracy**: the
   fallback `compute_credit_cost` (line 2946) runs for every bucket row that has
   `stored_credits = 0`. Whether this matches the actual subscription-credit formula
   for all model families was not verified.
5. **MCP raw vs. rich session read field parity**: `_raw_session_payload` and the rich
   session rendering return different row shapes; the exact field inventory of each was
   not compared against `SessionSummary` domain model fields.
6. **CLI `analyze --cost-outlook` vs. MCP `cost-outlook:` semantic equivalence**: both
   routes ultimately call `Polylogue.cost_outlook`, but whether query parameters (plan,
   method, since/until) are identical was not verified.

---

## New defects: none

All confirmed mask mismatches and data divergences were classified as:

- **Already owned / `blpir`**: session summary N-mapper field loss (parent_id, branch_type,
  display_name parity)
- **Already owned / `f2qv.6`**: cost authority transition (sp.cost_usd NULL throughout,
  sp.total_cost_usd/smu.cost_usd authority split, coverage insight zero cost, thread vs.
  session cost insight disagreement)
- **Already owned / `hb9o6`**: attachment acquired→missing public classification
- **Already owned / `27ezu`**: session topology dual-implementation, unresolved edge type loss
- **By design**: assertion non-exposure on MCP/HTTP

---

## Recommended strengthening to existing Beads

**`blpir`** (already P0):
- Add explicit constraint: `branch_type` requires a `session_links` JOIN in the summary
  SQL path; the mapper fix alone is insufficient. Enumerate exactly which summary routes
  require the JOIN and which can accept a deliberate NULL.

**`f2qv.6`** (campaign packet 6):
- Note that `ArchiveCoverageInsight.total_cost_usd` is zero on all current routes:
  origin path has no cost join; day/week path reads `sp.cost_usd` (NULL). The selected
  authority for this field post-`f2qv.6` should be stated explicitly (likely `sp.cost_usd`
  for day/week, and an added profile join for origin).
- Note the enrichment N+1 elimination opportunity: once `sp.cost_usd` is populated,
  `enrich_session_cost_insight` can short-circuit for `exact`/`priced` sessions.

---

## Commands run and durations

All queries were read-only static code inspection; no live SQLite queries were executed in
this pass. Code reads covered:

- `polylogue/api/archive.py` (mappers at lines 2292–2342, cost insights at lines 2495–2560,
  cost rollup at lines 2561–2641, usage timeline at lines 2761–2960)
- `polylogue/storage/sqlite/archive_tiers/archive.py` (list_session_cost_insights, read
  functions at lines 7459–7498, cost rollup accumulator at lines 2586–2720)
- `polylogue/storage/sqlite/archive_tiers/read_insights.py` (coverage insight queries at
  lines 155–315)
- `polylogue/insights/cost_enrichment.py` (full file)
- `polylogue/insights/archive.py` (cost insight models at lines 455–554)
- `polylogue/archive/semantic/pricing.py` (payload definitions at lines 59–168)
- `polylogue/surfaces/payloads.py` (session masks at lines 696–717, envelope builders at
  lines 890–961, assertion payloads at lines 1794–1960)
- `polylogue/mcp/payloads.py` (session summary exposure)
- `polylogue/daemon/http.py` (cost panel at lines 761–820, session cost handler at lines
  3765–3777)
- `polylogue/mcp/server.py`, `mcp/server_tools.py`, `mcp/server_cutover.py` (assertion/cost
  exposure audit)
- `polylogue/storage/insights/session/records.py` (cost field declarations)
- `polylogue/archive/session/domain_models.py` (SessionSummary model)

One live query ran to check session_kind distribution (23,496 rows, all 'standard', < 1 s).
