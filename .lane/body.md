## Summary

The current master contains the shared `Page[T]` pagination contract and its
surface adapters for candidate review, facets, pathology reports, daemon CLI
responses, and MCP personal-state listings.

## Problem

Bounded reads previously derived totals and completion from returned page
lengths, consumed facet limits while computing scoped denominators, and exposed
MCP continuations that could not fetch the next page.

## Solution

Use an explicit matched denominator and returned count, preserve full scoped
facet inputs, carry pathology matched/analyzed counts, honor daemon-clamped
limits when calculating continuation offsets, and accept offset continuations
for all MCP personal-state projections.

## Verification

- `uv run devtools test tests/unit/mcp/test_query_gap_projections.py tests/unit/cli/test_daemon_golden_parity.py tests/unit/cli/test_assertion_candidates.py tests/unit/insights/test_pathology.py tests/unit/insights/test_postmortem.py tests/unit/api/test_assertion_candidate_evidence_disclosure.py -x` — 43 passed.
- `uv run devtools verify --quick` — every quick stage passed.

## Residual risk

The broader selected batch stopped at an unrelated inherited daemon cost
assertion: expected `0.0`, received `None` in
`test_web_reader.py::TestCockpitAggregateRoutes::test_evidence_summary_matches_structural_tool_relations[0-2-expected_outcomes0]`.
