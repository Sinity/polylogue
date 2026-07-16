# 183. polylogue-9l5.8 — Temporal analytics: trends, rolling baselines, changepoint detection

Priority/type/status: **P2 / feature / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **blocked-hard**.

Hard blockers: polylogue-9l5.7

## What the bead says

The archive spans years of daily work but has no time-axis analytics beyond day/week summaries: no trend ('is my silent-proceed rate improving?'), no baseline ('is today's cost anomalous vs my trailing month?'), no changepoint ('did failure rates shift when I switched models / enabled hooks / upgraded the harness?'). Changepoints are the construct-valid way to talk about interventions without a controlled experiment — locate the shift, then check whether it coincides with a known event (harness release, config commit) rather than eyeballing dashboards.

## Existing design note

(1) Series builder as a DSL stage: any measure over any unit grouped into time buckets -> a typed series ('measure tool_failure_rate window week over 2026') — one series primitive, every measure gains a time axis for free (composability payoff). (2) Rolling baselines: trailing-window median/MAD bands (robust to heavy-tailed cost/latency); points outside k*MAD flag as anomalies — same machinery serves cost_outlook upgrades and daemon health anomaly lines (cursor_lag_baseline already does a bespoke version — converge it onto this). (3) Changepoint detection: offline PELT or binary segmentation on series (ruptures library under [analytics]; fallback: simple binary segmentation is ~60 lines); output = candidate changepoints WITH the honesty rail: each changepoint is a CANDIDATE annotated with nearby known events (model switch from session metadata, config commits via 7xv, harness version changes from hook events) — never auto-asserted as causal. (4) Seasonality: day-of-week / hour-of-day profiles (circular means) for session volume, cost, failure rates — descriptive only, confound-flagged (workload mix shifts with time). (5) Surfaces: analyze projection + webui sparklines (dataviz-lite in the workbench header is a natural consumer); series render as terminal sparklines in CLI (--plain: table).

## Acceptance criteria

A series stage composes with any registered measure on the seeded corpus. Rolling-baseline anomaly flags reproduce a seeded anomaly scenario. Changepoint output on a synthetic step-series locates the step and renders it as candidate + nearby-events annotation, never as a causal claim.

## Static mechanism / likely defect

Issue description localizes the mechanism: The archive spans years of daily work but has no time-axis analytics beyond day/week summaries: no trend ('is my silent-proceed rate improving?'), no baseline ('is today's cost anomalous vs my trailing month?'), no changepoint ('did failure rates shift when I switched models / enabled hooks / upgraded the harness?'). Changepoints are the construct-valid way to talk about interventions without a controlled experiment — locate the shift, then check whether it coincides with a known event (harness release, config com… Design direction: (1) Series builder as a DSL stage: any measure over any unit grouped into time buckets -> a typed series ('measure tool_failure_rate window week over 2026') — one series primitive, every measure gains a time axis for free (composability payoff). (2) Rolling baselines: trailing-window median/MAD bands (robust to heavy-tailed cost/latency); points outside k*MAD flag as anomalies — same machinery serves cost_outlook up…

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

1. (1) Series builder as a DSL stage: any measure over any unit grouped into time buckets -> a typed series ('measure tool_failure_rate window week over 2026') — one series primitive, every measure gains a time axis for free (composability payoff).
2. (2) Rolling baselines: trailing-window median/MAD bands (robust to heavy-tailed cost/latency)
3. points outside k*MAD flag as anomalies — same machinery serves cost_outlook upgrades and daemon health anomaly lines (cursor_lag_baseline already does a bespoke version — converge it onto this).
4. (3) Changepoint detection: offline PELT or binary segmentation on series (ruptures library under [analytics]
5. fallback: simple binary segmentation is ~60 lines)
6. output = candidate changepoints WITH the honesty rail: each changepoint is a CANDIDATE annotated with nearby known events (model switch from session metadata, config commits via 7xv, harness version changes from hook events) — never auto-asserted as causal.
7. (4) Seasonality: day-of-week / hour-of-day profiles (circular means) for session volume, cost, failure rates — descriptive only, confound-flagged (workload mix shifts with time).

## Tests to add

- Acceptance proof: A series stage composes with any registered measure on the seeded corpus.
- Acceptance proof: Rolling-baseline anomaly flags reproduce a seeded anomaly scenario.
- Acceptance proof: Changepoint output on a synthetic step-series locates the step and renders it as candidate + nearby-events annotation, never as a causal claim.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
