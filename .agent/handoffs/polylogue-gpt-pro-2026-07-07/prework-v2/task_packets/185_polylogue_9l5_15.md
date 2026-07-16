# 185. polylogue-9l5.15 — Triage frontier: worth_reviewing_score + TRIAGED lifecycle — an inbox that empties

Priority/type/status: **P2 / task / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

A context-free frontier over all ~16K logical sessions (inverts the cwd-coupled find_resume_candidates): time-invariant worth_reviewing_score materialized with a decomposable breakdown (unresolved blockers, open questions, decision density, terminal state), collapsed by logical_session_id; inverted-U staleness applied at READ time (materialized staleness goes stale). TRIAGED assertion kind (resumed / wont_resume / archived / snoozed:<until>) makes it a true inbox that empties via WHERE NOT EXISTS triaged; snooze-with-wake. Honesty corrections from review: hard-zero ONLY truly disposable sessions — in-flight and superseded branches become VISIBLE demoted buckets, never hidden drops (a queue that hides rows falsely looks empty); low-confidence enrichment factors visibly marked and down-weighted. Cross-tier caveat: triaged-filtering is a runtime query method, never a persistent view over ATTACHed user.db (SQLite forbids it).

## Existing design note

The score is a NAMED-FEATURE LINEAR COMBINATION with per-feature contributions visible in the payload (rigor doctrine: no opaque scores) — every feature is a computable structural signal that already exists or has an owning bead: unacknowledged_failure (tool_result_is_error=1 with no subsequent success of a normalized-same command in-session — the 'failed and moved on' signature), abnormal_termination (session ends inside a tool loop / no assistant close), cost_outlier (session cost above p95 for its repo x workflow-shape cohort), correction_density (operator corrections per authored-user message), pathology_hits (get_pathologies count), duration_outlier (wall-clock p95 cohort-relative), zero_outcome (no commit/file-write/verify success evidence in a session whose prompt implies a work task — evidence tiers from 9l5.13 spans). Weights start hand-set, tuned only against operator triage decisions once TRIAGED data exists (the lifecycle IS the label source; no invented ground truth). LIFECYCLE: worth_reviewing surfaces sessions into a triage view (CLI + webui inbox); operator verdicts (reviewed-useful / reviewed-noise / ignore-kind) are assertions (kind=judgment, scope=session) that (a) empty the inbox and (b) accumulate into the weight-tuning set. Emission is coverage-gated per feature: a session missing cost evidence gets score WITHOUT cost_outlier and the payload says so (insufficient_evidence per feature, never fabricated).

## Acceptance criteria

Frontier returns logical representatives with score breakdown + confidence; triage/snooze removes rows via runtime method; disposable clean-finish rows zero out while blocker sessions surface; demoted buckets visible. Verify: fixture corpus + scorer tests.

## Static mechanism / likely defect

Issue description localizes the mechanism: A context-free frontier over all ~16K logical sessions (inverts the cwd-coupled find_resume_candidates): time-invariant worth_reviewing_score materialized with a decomposable breakdown (unresolved blockers, open questions, decision density, terminal state), collapsed by logical_session_id; inverted-U staleness applied at READ time (materialized staleness goes stale). TRIAGED assertion kind (resumed / wont_resume / archived / snoozed:<until>) makes it a true inbox that empties via WHERE NOT EXISTS triaged; snooze-w… Design direction: The score is a NAMED-FEATURE LINEAR COMBINATION with per-feature contributions visible in the payload (rigor doctrine: no opaque scores) — every feature is a computable structural signal that already exists or has an owning bead: unacknowledged_failure (tool_result_is_error=1 with no subsequent success of a normalized-same command in-session — the 'failed and moved on' signature), abnormal_termination (session ends …

## Source anchors to inspect first

- `polylogue/archive/actions/followup.py` — Action/followup classification is a real structural analytics input.
- `polylogue/archive/actions/fields.py` — Action fields determine what can be measured without prose heuristics.
- `polylogue/insights/registry.py:294` — Insight registry should become measure/product registry input.
- `scripts/agent_forensics.py` — Existing forensics script is a proof artifact and candidate product surface.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## Implementation plan

1. The score is a NAMED-FEATURE LINEAR COMBINATION with per-feature contributions visible in the payload (rigor doctrine: no opaque scores) — every feature is a computable structural signal that already exists or has an owning bead: unacknowledged_failure (tool_result_is_error=1 with no subsequent success of a normalized-same command in-session — the 'failed and moved on' signature), abnormal_termination (session end…
2. Weights start hand-set, tuned only against operator triage decisions once TRIAGED data exists (the lifecycle IS the label source
3. no invented ground truth).
4. LIFECYCLE: worth_reviewing surfaces sessions into a triage view (CLI + webui inbox)
5. operator verdicts (reviewed-useful / reviewed-noise / ignore-kind) are assertions (kind=judgment, scope=session) that (a) empty the inbox and (b) accumulate into the weight-tuning set.
6. Emission is coverage-gated per feature: a session missing cost evidence gets score WITHOUT cost_outlier and the payload says so (insufficient_evidence per feature, never fabricated).

## Tests to add

- Acceptance proof: Frontier returns logical representatives with score breakdown + confidence
- Acceptance proof: triage/snooze removes rows via runtime method
- Acceptance proof: disposable clean-finish rows zero out while blocker sessions surface
- Acceptance proof: demoted buckets visible.
- Acceptance proof: Verify: fixture corpus + scorer tests.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
