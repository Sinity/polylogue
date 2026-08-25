---
created: 2026-08-25
purpose: Bounded continuation-2 of polylogue-rrxe4.8 — the six remaining unexamined cells from continuation-1
status: complete — all six cells dispositioned; one genuine current-route defect found
project: polylogue
---

# Live route-mask continuation audit — pass 2

## Authority and method

Direct continuation of `2026-08-25-live-route-mask-continuation.md` (polylogue-rrxe4.8 pass 1).
Frozen evidence head: `bc270dfb3be48c9b4c2fdd24919f5d349cc17751`. No daemon activity; no
live archive queries executed (all evidence from static code inspection at exact line numbers).
Code files inspected are listed under Commands/Durations.

---

## Coverage matrix — six remaining cells × public surfaces

| Cell | Route | CLI | API | MCP | HTTP | Novelty disposition |
|---|---|---|---|---|---|---|
| 1. `work_event_breakdown` N+1 | day/week bucket subquery | all surfaces (dict) | all surfaces | all surfaces | all surfaces | **Benign — N+1 known; no correctness defect** |
| 2. `SessionProfileInsight` field parity | native vs JSON payload reconciliation | inference JSON | inference JSON | inference JSON | inference JSON | **By design — reconciled fields are ranking denorms only** |
| 3. `CostRollupInsight` per_model overwrite | accumulator assignment | rollup surface | rollup surface | rollup surface | rollup surface | **Genuine current-route defect — contradicts rigor contract** |
| 4. `UsageTimelineInsight` subscription_credits | catalog fallback | timeline surface | timeline surface | timeline surface | timeline surface | **Known limitation — already documented in rigor.py:807** |
| 5. MCP raw vs rich session payload | list vs resolve_ref path | N/A | N/A | list (raw) / get (rich) | N/A | **Already owned / `blpir`** |
| 6. `analyze --cost-outlook` vs MCP `cost-outlook:` | CLI vs MCP envelope | CLI JSON + availability | N/A | MCP — no availability | N/A | **Behavioral difference — benign by design** |

---

## Detailed findings

---

### Cell 1 — ArchiveCoverageInsight `work_event_breakdown`: query shape, alignment, N+1

**Code**: `_coverage_work_event_breakdown` (read_insights.py:387–408).
Called at read_insights.py:296–297 for each row returned by `_time_bucket_coverage_insights`.

**Query shape** (lines 397–406):
```sql
SELECT e.work_event_type, COUNT(*) AS count
FROM sessions s
JOIN session_work_events e ON e.session_id = s.session_id
WHERE strftime(?, s.sort_key_ms / 1000, 'unixepoch') = ?
  [AND s.origin = ?] [AND s.sort_key_ms >= ?] [AND s.sort_key_ms <= ?]
GROUP BY e.work_event_type
ORDER BY count DESC, e.work_event_type
```

**N+1 pattern**: Two sibling functions fire on the same bucket loop: `_coverage_repos_active`
(lines 411–432) and `_coverage_origin_breakdown` (lines 435–455). Each bucket row causes 3
extra queries. For an unbounded `list_archive_coverage_insights(group_by="day")` call over
a one-year window, that is 365 × 3 = 1,095 extra queries. No minimum-bucket-count guard exists.

**Bucket alignment**: Correct. `_coverage_bucket_filter` (lines 365–384) re-applies the same
`strftime(bucket_format, sort_key_ms/1000, 'unixepoch') = bucket` predicate used in the outer
query. Since the outer `GROUP BY bucket` computed the same strftime expression, the subquery
exactly isolates each bucket's sessions. No misalignment possible from the strftime re-evaluation.

**Stale / duplicate states**: No stale state — this is a live synchronous query over the same
connection snapshot. No deduplication needed: each `session_work_events` row is a distinct event.

**Per-bucket count vs session count denominator mismatch**: `work_event_breakdown` values are
event counts (not distinct session counts). The outer `session_count` field is a distinct-session
count. These are different denominators and not presented as equivalent. No mismatch defect.

**Provenance gap (origin path)**: `_origin_coverage_insights` (lines 191–230) does NOT call
`_coverage_work_event_breakdown` at all. `ArchiveCoverageInsight.work_event_breakdown` defaults
to `None` in origin-grouped results. This is an asymmetry between `group_by="origin"` (no
breakdown) and `group_by="day"/"week"` (breakdown included). No surface documents this asymmetry.

**Public parity**: The `dict[str, int]` result passes through unchanged to CLI, API, MCP, HTTP.
No additional surface-level masking.

**Disposition**: **Benign.** N+1 is a performance concern not a correctness defect. Bucket
alignment is correct. The origin-vs-time-bucket asymmetry on `work_event_breakdown` is by design
(origin path omits the subquery). No data-integrity defect exists.

---

### Cell 2 — `SessionProfileInsight` field parity: evidence/inference/enrichment JSON vs native columns

**Code**: `SessionProfileRecord` (storage/insights/session/records.py:23–113),
`SessionProfileInsight.from_record` (insights/archive.py:255–312).

**Full field inventory** of `SessionProfileRecord` native columns that have parallel
representation in the JSON payloads:

| Native column | Payload type | Field name in payload | Reconciled in `from_record`? |
|---|---|---|---|
| `terminal_state` | `SessionInferencePayload` | `terminal_state` | **Yes** (line 274) |
| `terminal_state_confidence` | `SessionInferencePayload` | `terminal_state_confidence` | **Yes** (line 275) |
| `terminal_state_method` | `SessionInferencePayload` | `terminal_state_method` | **Yes** (line 276) |
| `workflow_shape` | `SessionInferencePayload` | `workflow_shape` | **Yes** (line 277) |
| `workflow_shape_confidence` | `SessionInferencePayload` | `workflow_shape_confidence` | **Yes** (line 278) |
| `engaged_duration_ms` | `SessionInferencePayload` | `engaged_duration_ms` | **No** |
| `tool_active_duration_ms` | `SessionInferencePayload` | `tool_active_duration_ms` | **No** |
| `work_event_count` | `SessionInferencePayload` | `work_event_count` | **No** |
| `phase_count` | `SessionInferencePayload` | `phase_count` | **No** |
| `total_cost_usd` | `SessionEvidencePayload` | `total_cost_usd` | **No** |

The 5 reconciled fields are explicitly identified in CLAUDE.md as "ranking signals" (terminal_state,
workflow_shape). The un-reconciled fields (engaged_duration_ms etc.) are denorm native columns for
query predicates; `from_record` does not reconcile them because they are not exposed through
ranking predicates that would diverge from the JSON.

The writer (`storage/insights/session/storage.py:235–287` region) keeps native columns and JSON
payloads in sync on every write, so divergence requires an out-of-band native-column repair that
doesn't re-materialize the inference JSON. No such repair path has been observed or documented.

**Enrichment reconciliation** (lines 281–298): `enrichment.objective_posture` is recomputed
from the reconciled `inference.terminal_state` on every read. This is correct and documented
(polylogue-37t.23).

**Disposition**: **By design.** The reconciliation targets the 5 ranking-signal fields that must
be queryable from native columns. Un-reconciled fields have no native-column authority divergence
path because the writer is the only source. No new defect.

---

### Cell 3 — `CostRollupInsight` per_model_breakdown: provenance-overwrite defect

**Code**: `list_cost_rollup_insights` (archive.py:2561–2759), `_CostRollupAccumulator` (archive.py:341–363).

**SQL grouping** (lines 2609–2614):
```sql
GROUP BY s.origin,
         u.model_name,
         COALESCE(CASE WHEN u.cost_usd IS NOT NULL THEN u.cost_provenance
                       ELSE sp.cost_provenance END, 'unknown')
```
One SQL row per `(origin, model_name, effective_provenance)`.

**Python accumulator key** (line 2652):
```python
key = (source_name, normalized_model or model_name)
```
One accumulator per `(origin, normalized_model)`.

**Outer totals**: accumulated correctly via `entry.session_count += session_count`,
`entry.basis = entry.basis.plus(basis)`, `entry.total_usd += stored_cost_usd` (lines 2693–2711).

**per_model breakdown** (line 2714):
```python
entry.per_model[(model_name, normalized_model)] = CostModelBreakdown(
    model_name=model_name,
    normalized_model=normalized_model,
    usage=usage,
    basis=basis,
    total_usd=stored_cost_usd,
    session_count=session_count,
)
```
This is an **assignment, not accumulation**. When the same `(model_name, normalized_model)` key
appears in two SQL rows (same model, different provenances), only the **last** row's values survive.

**Concrete failure scenario**:
- Origin=`claude`, model=`claude-sonnet-4-6` exists with provenance `exact` (rows from 2024,
  provider-reported) and `estimated` (rows from 2023, catalog-priced).
- SQL emits two rows: `(claude, claude-sonnet-4-6, exact, 300 sessions, $X)` and
  `(claude, claude-sonnet-4-6, estimated, 150 sessions, $Y)`.
- Both map to accumulator key `(claude, claude-sonnet-4-6)`.
- After processing: `entry.session_count = 450` (correct), `entry.total_usd = X+Y` (correct).
- `entry.per_model[("claude-sonnet-4-6", "claude-sonnet-4-6")]` = whichever row processed last:
  `CostModelBreakdown(session_count=150, total_usd=Y)` (the estimated row, last in SQL order).
- `per_model_breakdown[0].session_count (150) ≠ entry.session_count (450)` — inconsistency.

**Rigor contract gap**: `rigor.py:763–787` declares per_model_breakdown values as
"direct grouped aggregates" via `_true_zero_paths`. The overwrite means they are NOT always
direct grouped aggregates when a model spans multiple provenance groups.

**Surface exposure**: `per_model_breakdown` is a consumer field on `CostRollupInsight`
(rigor.py:716) and reaches all surfaces unchanged.

**Live denominator**: Without a live query (within the 90-second constraint), the exact count
of (origin, model_name) pairs with mixed provenances in `session_model_usage` is unknown.
The defect is latent until such rows exist, which is possible when import batches differ in
cost_provenance assignment. `sp.cost_usd` is NULL for all sessions (established in pass 1),
so the fallback `COALESCE(CASE WHEN u.cost_usd IS NOT NULL ... ELSE sp.cost_provenance ...)` 
collapses to `u.cost_provenance` only — making the denominator depend on
`session_model_usage.cost_provenance` diversity per (origin, model) group.

**Bead search**: Not found in existing Bead set for this session. `f2qv.6` owns cost-column
selection; `rrxe4.8` is the audit parent. This per_model overwrite is independent of which
column to read.

**Classification**: **Genuine current-route defect.** The `per_model_breakdown` accumulator
silently underreports per-model session counts and costs when the same (origin, model) has
rows with different cost_provenance values. Outer CostRollupInsight totals are unaffected.

**Draft Bead contract** (do not create; coordinator action required):

```
Title: CostRollupInsight per_model_breakdown overwrites on provenance split
Route: list_cost_rollup_insights → _CostRollupAccumulator.per_model → per_model_breakdown
       archive.py:2714 (assignment) vs :2693–2711 (correct accumulation)
Defect: entry.per_model[(model_name, normalized_model)] = ... replaces instead of
        accumulating when (origin, model_name) spans multiple cost_provenance SQL groups.
        Outer entry totals are correct; per_model detail is wrong.
Rigor contract claim violated: rigor.py:763–787 "direct grouped aggregates".
TDD: test with fixture: two session_model_usage rows with same (session.origin, model_name)
     but different cost_provenance. Expect per_model_breakdown[0].session_count to equal
     the sum of both rows' counts. Currently fails: shows only the last row's count.
Mutant: change line 2714 from assignment to additive accumulation (using CostBasisPayload.plus
        and CostUsagePayload.plus). The test above must turn green.
Data query (read-only verification):
  SELECT s.origin, u.model_name,
         COALESCE(CASE WHEN u.cost_usd IS NOT NULL THEN u.cost_provenance
                       ELSE sp.cost_provenance END, 'unknown') AS prov,
         COUNT(DISTINCT u.session_id) AS n
  FROM session_model_usage u
  JOIN sessions s ON s.session_id = u.session_id
  LEFT JOIN session_profiles sp ON sp.session_id = s.session_id
  GROUP BY s.origin, u.model_name, prov
  HAVING n > 0
  ORDER BY s.origin, u.model_name, prov;
  -- Any (origin, model_name) appearing in 2+ rows with different prov values is live evidence.
Deletion trigger: defect resolved when per_model_breakdown[model].session_count sums correctly
                  across all provenance groups for a given (origin, model_name) pair.
```

---

### Cell 4 — `UsageTimelineInsight` subscription_credits fallback accuracy

**Code**: archive.py:2944–2952 (cost_rows accumulation loop).

**Fallback trigger**: `if not float(row["stored_credits"] or 0.0)` — fires for both NULL and
stored_credits=0. For a genuinely zero-credit session (e.g. a 0-message session), this would
invoke `compute_credit_cost`, which would still return 0 for that session since zero tokens
produce zero credits. Net effect: no overcounting in the zero-session case.

**`compute_credit_cost` call** (lines 2946–2952):
```python
item.subscription_credits += compute_credit_cost(
    _normalize_model(str(row["model_name"] or "")),
    int(row["input_tokens"] or 0),
    int(row["output_tokens"] or 0),
    0,                           # cache_read_tokens hardcoded
    int(row["cache_write_tokens"] or 0),
)
```

**cache_read_tokens hardcoded to 0**: `MODEL_CREDIT_RATES` (subscription_pricing.py:84–105)
sets `cache_read_credits=0` for all defined models (subscription plans include free cache reads
per Anthropic's pricing page). Passing 0 for cache_read_tokens is therefore correct.

**Model catalog coverage** (subscription_pricing.py:84–105):
- `claude-opus-4-6`, `claude-opus-4-5`: 10/50 input/output per 15 tokens
- `claude-sonnet-4-6`, `claude-sonnet-4-5`: 6/30 per 15 tokens
- `claude-haiku-4-5`: 2/10 per 15 tokens
- All other models (pre-4.5, non-Claude): `get_credit_rate` returns `None` → `compute_credit_cost` returns 0

**Known limitation**: rigor.py:806–813 explicitly documents that `subscription_credits` uses
`compute_credit_cost` as a fallback and is "indistinguishable in the payload from a genuinely
stored credit figure." This is already acknowledged; `cost_provenance_counts` is the intended
disambiguation signal.

**Disposition**: **Known limitation already documented in rigor.py:807.** No new defect.
The hardcoded `cache_read_tokens=0` is correct given the subscription-plan pricing model.

---

### Cell 5 — MCP raw vs rich session payload masks

**Two MCP read paths for a session ref**:

**Path A — list/search/query results** (`archive_summary_payload`, archive_support.py:158–172):
Built from `ArchiveSessionSummary` (a SQL summary row). Produces `MCPSessionSummaryPayload`:
- Fields present: `id`, `origin` (from session.origin), `title` (from `display_label || title`),
  `message_count`, `target_ref`, `anchor`, `created_at`, `updated_at`
- Fields absent: `title_source`, `title_ref`, `title_confidence` (not on ArchiveSessionSummary)
- `actions` uses `reader_session_actions()` default

**Path B — resolve_ref via `get(ref="session:<id>")` or `read(ref="session:")`**:
`_resolve_session_object_ref` (api/archive.py:4224–4264):
1. `archive.list_summaries(session_id=..., limit=1)` → `ArchiveSessionSummary`
2. `_archive_summary_to_domain(summaries[0])` → full `SessionSummary` domain model
3. `session_summary_envelope_from_summary(domain_summary)` → `SessionSummaryEnvelope`
   - Fields present: `id`, `origin`, `title`, **`title_source`**, **`title_ref`**,
     **`title_confidence`**, `message_count`, `target_ref`, `anchor`, `actions`, `created_at`, `updated_at`

**Divergence**: Path A lacks `title_source`, `title_ref`, `title_confidence`; Path B carries them.
The richer route (Path B) is reachable via `get(ref="session:<id>")`.

**Deliberate compactness vs silent omission**: `ArchiveSessionSummary` is a shallow SQL summary
row that does not carry title provenance signals. Path A uses it for all list/search results.
Path B incurs the extra `_archive_summary_to_domain` cost to hydrate these fields.

**Disposition**: **Already owned / `blpir`** (three-mapper divergence: archive-query path,
API-summary-mapper path, exact-CLI path). The MCP list path is the "archive-query" variant
that loses title_source/ref/confidence. No new defect.

**blpir strengthening note**: The richer MCP route (Path B, `get(ref="session:...")`) 
produces full `SessionSummaryEnvelope` fields. Any `blpir` fix that adds title_source etc.
to the list-path mapper should verify Path A's `archive_summary_payload` function
(archive_support.py:158) is also updated, since it uses `ArchiveSessionSummary` directly
and will need a `_archive_summary_to_domain` conversion or a JOIN in the underlying summary query.

---

### Cell 6 — CLI `analyze --cost-outlook` vs MCP `cost-outlook:` equivalence

**Shared production route**: Both call `Polylogue.cost_outlook(plan_name, method=projection_method)`.

**Parameter parity**:

| Parameter | CLI | MCP |
|---|---|---|
| `plan_name` | `--plan` (required with `--cost-outlook`) | `plan_name = ref.removeprefix("cost-outlook:")` |
| `method` | `--method` (choice: linear/trailing-7d-mean/eom-naive) | `projection` param, `ProjectionMethod(method)` |
| default method | `ProjectionMethod.linear` | `ProjectionMethod.linear` |
| method validation | `click.Choice` → UsageError | `ProjectionMethod(method)` → ValueError → `error_json` |
| empty plan name | `raise UsageError` (CLI guard at line 1954) | `error_json("cost-outlook ref requires a plan name")` |
| unknown plan | `PlanLookupError` → `ClickException` | `PlanLookupError` → `error_json(invalid_argument)` |
| since/until | no (cost_outlook is plan-based, not range-filtered) | no |
| pagination | no | no |

**Envelope difference — success case**:
- CLI JSON (line 1983–1993): `{**outlook.model_dump(mode="json"), "availability": {state, elapsed_s, ...}}`
- MCP (line 204): `hooks.json_payload(outlook, exclude_none=True)` — **no `availability` field**

**Envelope parity — None case (no cycle window)**:
- CLI JSON: `{"outlook": null, "availability": {...}}` ✓
- MCP: `MCPRootPayload(root={"outlook": None, "availability": {...}})` ✓ (equivalent)

**Behavioral difference**: MCP success response omits `availability`, which carries:
- `state` ("ready"/"degraded"/etc.), `elapsed_s`, `deadline_s`, `detail`, `remediation`
- MCP clients cannot determine readiness state or computation timing for a successful projection.

**Why benign**: `_cost_outlook_payload` docstring (server_cutover.py:172–177) says "Mirrors the
CLI `analyze --cost-outlook` call shape" — the shared production route is honored. The `availability`
field is a CLI-specific timing diagnostic (populated from `time.perf_counter()` in the CLI,
unavailable in async MCP context). MCP exposes the outlook data itself without the timing wrapper.

**Disposition**: **Behavioral difference — benign by design.** Both surfaces call the same
`Polylogue.cost_outlook` route with identical defaults and semantics. The availability envelope
is a CLI timing diagnostic, not part of the projection contract. No new defect.

---

## Summary: new defects found

| # | Defect | Route | Blast radius |
|---|---|---|---|
| **1** | `per_model_breakdown` overwrites on provenance split | archive.py:2714, `list_cost_rollup_insights` | All surfaces reading `CostRollupInsight.per_model_breakdown`; outer totals correct |

---

## Existing Beads confirmed as owning residue

| Cell | Owning Bead | Evidence |
|---|---|---|
| MCP raw vs rich session payload | `blpir` | title_source/title_ref/title_confidence absent from list path; archive_summary_payload uses ArchiveSessionSummary |
| cost_outlook envelope | (no Bead — benign) | CLI timing wrapper vs MCP compact response; shared production route |

---

## Recommended strengthening to existing Beads

**`blpir`** (already noted in pass 1): add that `archive_summary_payload`
(archive_support.py:158) also requires a fix — it uses `ArchiveSessionSummary` directly and
will not gain title_source/ref/confidence from a mapper fix alone. Either add a
`_archive_summary_to_domain` call or add the fields to the underlying ArchiveSessionSummary
SQL query.

**`rrxe4.8`** (parent audit Bead): reference the `per_model_breakdown` defect (draft above)
as a new finding requiring a child Bead. The fix is a targeted accumulation change at
archive.py:2714 with a fixture-based TDD case.

---

## Commands run and durations

All evidence is from static code inspection; no live SQLite queries were executed in this pass.

Files read (with key line ranges):

| File | Lines inspected | Purpose |
|---|---|---|
| `storage/sqlite/archive_tiers/read_insights.py` | 1–458 (full) | Cells 1 (work_event_breakdown), reviewed prior-pass coverage insight |
| `storage/insights/session/records.py` | 1–179 (full) | Cell 2 (native column inventory) |
| `insights/archive_models.py` | 62–311 | Cell 2 (payload field inventory) |
| `insights/archive.py` | 116–312 | Cell 2 (from_record reconciliation) |
| `storage/sqlite/archive_tiers/archive.py` | 341–363, 2561–2759, 2761–2990 | Cells 3 (cost rollup), 4 (usage timeline) |
| `archive/semantic/subscription_pricing.py` | 1–170 (full) | Cell 4 (catalog coverage) |
| `mcp/server_cutover.py` | 165–204, 750–918 | Cells 5 (MCP session paths), 6 (cost-outlook MCP) |
| `api/archive.py` | 4109–4264 | Cell 5 (resolve_ref session path) |
| `mcp/archive_support.py` | 155–205 | Cell 5 (archive_summary_payload) |
| `mcp/payloads.py` | 44, 87–114, 216–393 | Cell 5 (MCPSessionSummaryPayload) |
| `surfaces/payloads.py` | 696–720, 880–960, 1163–1220 | Cell 5 (SESSION_SUMMARY_MASK, SessionSummaryEnvelope) |
| `cli/query_verbs.py` | 1733–2017 | Cell 6 (CLI cost-outlook path) |
| `insights/rigor.py` | 655–794 | Cell 3 (rigor contract for cost_rollups per_model_breakdown) |

Investigation: ~28 minutes static analysis. Synthesis: ~10 minutes. Total: ~38 minutes.

---

## Remaining unexamined cells

None. All six cells from pass 1's "remaining unexamined" list are now dispositioned.

---

## Completeness claim

This pass covers exactly the six cells listed in pass 1 as unexamined. No completeness claim
beyond those cells. The broader rrxe4.8 audit scope (e.g. daemon HTTP surfaces, full MCP
operation dispatcher coverage, embeddings search path, export bundles) was not re-examined.
