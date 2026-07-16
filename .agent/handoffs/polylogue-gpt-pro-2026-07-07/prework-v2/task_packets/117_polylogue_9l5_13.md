# 117. polylogue-9l5.13 — activity_spans materializer: edit/test/build/idle/delegate intervals with evidence tiers

Priority/type/status: **P2 / task / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **blocked-hard**.

Hard blockers: polylogue-9l5.19

## What the bead says

The missing bridge between raw structure and "so what": a derived queryable relation of time-bounded work spans composed OVER existing substrate (actions keystone fields, phases 5-min-gap intervals, weak work-event labels, observed events, run projection) — a normalizer/composer, not a new capture pipeline. Span kinds start coarse and construct-valid: read_search, edit, build_compile, test, debug, review_vcs, delegate, synthesize, idle_gap, tool_wait, llm_wait, unknown. LOAD-BEARING DESIGN CHOICE: span kind is SEPARATE from evidence_tier — a test span from a pytest command with exit code (structural) is a different epistemic object than a debug span from prose containing "debug" (heuristic); no heuristic-only span renders untiered. Algorithm: versioned command-classifier alphabet (pytest/devtools/ruff/mypy recognition already exists in transforms — promote it out of ad hoc code) -> gap-split at phase threshold (gaps become idle spans, never vanish into duration math; separate idle_gap/tool_wait/llm_wait/human_absence when evidence supports, never one blended idle score) -> merge adjacent same-kind events -> attach NEXT structural outcome to each span (edit spans get their following test/verify result — enables recovery-latency and verification-discipline measures) -> caveats on every degraded input (turn-axis-only when timestamps missing, unknown outcome for unpaired tools per 9l5.6 doctrine). Relation designed as reusable work-trace (context packs, replay, delegation analytics consume it), filed under 9l5 as its first customer. tool_episodes (9l5.6) is the atomic layer below; activity_spans is temporal composition above it.

## Acceptance criteria

Seeded corpus produces spans with evidence refs; >threshold gaps are idle spans; structural test failure yields kind=test outcome=failed; activity-spans where session.repo:X | group by kind | sum duration_ms works (DSL terminal unit + fields registered as part of this bead); heuristic-classified spans carry the tier visibly. Verify: materializer fixtures + query-unit tests.

## Static mechanism / likely defect

Issue description localizes the mechanism: The missing bridge between raw structure and "so what": a derived queryable relation of time-bounded work spans composed OVER existing substrate (actions keystone fields, phases 5-min-gap intervals, weak work-event labels, observed events, run projection) — a normalizer/composer, not a new capture pipeline. Span kinds start coarse and construct-valid: read_search, edit, build_compile, test, debug, review_vcs, delegate, synthesize, idle_gap, tool_wait, llm_wait, unknown. LOAD-BEARING DESIGN CHOICE: span kind is SEP…

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

1. Register the measure/outcome with evidence tier, denominator, and uncertainty.
2. Materialize only after source units are stable.
3. Add fixture proving empty/uncovered samples do not become zeros.
4. Render caveats in CLI/report/web outputs.

## Tests to add

- Acceptance proof: Seeded corpus produces spans with evidence refs
- Acceptance proof: >threshold gaps are idle spans
- Acceptance proof: structural test failure yields kind=test outcome=failed
- Acceptance proof: activity-spans where session.repo:X | group by kind | sum duration_ms works (DSL terminal unit + fields registered as part of this bead)
- Acceptance proof: heuristic-classified spans carry the tier visibly.
- Acceptance proof: Verify: materializer fixtures + query-unit tests.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
