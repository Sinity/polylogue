# 116. polylogue-9l5.7 — Statistics substrate + measure registry: uncertainty primitives with construct-validity metadata

Priority/type/status: **P2 / feature / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **blocked-hard**.

Hard blockers: polylogue-9l5.19

## What the bead says

The keystone of the analytics tower. Today every number the archive emits is a point estimate with no uncertainty, and construct validity is a discipline (footnotes written by hand in campaign reports) rather than a mechanism. Two deliverables: (1) honest statistical primitives available wherever aggregates compose — proportions with Wilson intervals, mean/median/percentiles with n and CI, two-sample comparisons with effect size + test, histogram/ECDF buckets; (2) the MEASURE REGISTRY: every analytic registers a declaration — construct it operationalizes, formula, evidence tier (structural / provider-reported / derived / heuristic), sample-frame requirements, known confounds — and the composition layer enforces it: a cross-origin cost comparison without coverage tiers refuses to render as a bare number; every output carries its tier footnote automatically (generalizing the 9l5.2 pattern). insight_rigor_audit extends to audit the registry.

## Existing design note

(1) polylogue/analytics/stats.py: pure functions over sequences — wilson_interval(k,n), quantiles_with_ci (bootstrap or order-statistic CIs), two_proportion_test, mann_whitney (rank test avoids normality assumptions on latency/cost distributions), cliffs_delta effect size, histogram_buckets. scipy.stats behind the [analytics] extra with hand-rolled fallbacks for the handful used in core paths (Wilson and bootstrap are 20 lines each — core stays dependency-lean). (2) MeasureSpec (declare-once discipline, o21): name, construct, unit, formula ref, evidence_tier, required_coverage (e.g. priced-provenance-only), confounds list, output schema. Registered like query units; query_units/completions expose them so agents can DISCOVER what is measurable and at what validity before designing an analysis — the informed-construction affordance. (3) DSL integration (after fnm.1 aggregates): measure stages compose — 'sessions where repo:X | measure silent_proceed_rate by model | compare origin:claude-code-session vs codex-session' emits rates + CIs + test + tier footnotes. Multiple-comparison honesty: when a group-by fans out >5 comparisons, render Benjamini-Hochberg-adjusted flags, not raw stars. (4) Renderers show uncertainty by default: rate -> '24.1% [22.9, 25.3] n=5000 (structural)'; --point-only to suppress. Pitfall: do NOT attach CIs to full-population counts (no sampling error) — the registry marks census vs sample measures.

## Acceptance criteria

polylogue/analytics/stats.py exists with property tests (hypothesis: interval coverage on synthetic distributions). At least 5 existing analytics re-registered as MeasureSpecs with tiers. A cross-origin comparison without coverage labels is refused at composition with an actionable error. One DSL query composes measure+group+compare and renders CIs + tier footnotes on the seeded corpus.

## Static mechanism / likely defect

Issue description localizes the mechanism: The keystone of the analytics tower. Today every number the archive emits is a point estimate with no uncertainty, and construct validity is a discipline (footnotes written by hand in campaign reports) rather than a mechanism. Two deliverables: (1) honest statistical primitives available wherever aggregates compose — proportions with Wilson intervals, mean/median/percentiles with n and CI, two-sample comparisons with effect size + test, histogram/ECDF buckets; (2) the MEASURE REGISTRY: every analytic registers a d… Design direction: (1) polylogue/analytics/stats.py: pure functions over sequences — wilson_interval(k,n), quantiles_with_ci (bootstrap or order-statistic CIs), two_proportion_test, mann_whitney (rank test avoids normality assumptions on latency/cost distributions), cliffs_delta effect size, histogram_buckets. scipy.stats behind the [analytics] extra with hand-rolled fallbacks for the handful used in core paths (Wilson and bootstrap a…

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

1. (1) polylogue/analytics/stats.py: pure functions over sequences — wilson_interval(k,n), quantiles_with_ci (bootstrap or order-statistic CIs), two_proportion_test, mann_whitney (rank test avoids normality assumptions on latency/cost distributions), cliffs_delta effect size, histogram_buckets.
2. scipy.stats behind the [analytics] extra with hand-rolled fallbacks for the handful used in core paths (Wilson and bootstrap are 20 lines each — core stays dependency-lean).
3. (2) MeasureSpec (declare-once discipline, o21): name, construct, unit, formula ref, evidence_tier, required_coverage (e.g.
4. priced-provenance-only), confounds list, output schema.
5. Registered like query units
6. query_units/completions expose them so agents can DISCOVER what is measurable and at what validity before designing an analysis — the informed-construction affordance.
7. (3) DSL integration (after fnm.1 aggregates): measure stages compose — 'sessions where repo:X | measure silent_proceed_rate by model | compare origin:claude-code-session vs codex-session' emits rates + CIs + test + tier footnotes.

## Tests to add

- Acceptance proof: polylogue/analytics/stats.py exists with property tests (hypothesis: interval coverage on synthetic distributions).
- Acceptance proof: At least 5 existing analytics re-registered as MeasureSpecs with tiers.
- Acceptance proof: A cross-origin comparison without coverage labels is refused at composition with an actionable error.
- Acceptance proof: One DSL query composes measure+group+compare and renders CIs + tier footnotes on the seeded corpus.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
