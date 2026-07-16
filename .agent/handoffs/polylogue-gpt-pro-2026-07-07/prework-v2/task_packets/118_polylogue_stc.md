# 118. polylogue-stc — Experiment hosting: declared arms, preregistered metrics, paired analysis, agent-buildable

Priority/type/status: **P2 / feature / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **blocked-hard**.

Hard blockers: polylogue-9l5.7

## What the bead says

Generalize what cfk/jxe did by hand into substrate: an experiment is a first-class declared object — hypothesis, arms, assignment rule, PREREGISTERED metrics (declared before data collection, timestamped — the construct-validity teeth), sample-size intent, analysis plan — and the archive hosts its lifecycle: assignment, observation collection (sessions tagged to arms), paired/grouped analysis through the measure registry, and a cold-reader-gateable report. Agent affordance is the point (operator ask): agents should be able to CONSTRUCT well-formed experiments informedly — query the measure registry for what is measurable at what validity, draft the prereg, run the arms, and produce the analysis — so self-experimentation (37t.9 context-spec variation, prompt A/B, harness comparisons) stops being bespoke campaign scripting.

## Existing design note

(1) ExperimentSpec as a user.db artifact (assertion-adjacent, judgment-visible): hypothesis, arms (name + treatment description), assignment (manual | alternating | by-session-property), preregistered metrics (each a registry measure ref + direction + minimum-interesting-effect), planned n, analysis plan (paired vs unpaired, test choice via 9l5.7). Prereg timestamp is the assertion created_at — post-hoc metric additions are visibly post-hoc (labeled exploratory). (2) Lifecycle tools (CLI + MCP): experiment define / assign <session-ref> <arm> / status (n per arm, power-ish progress vs planned n) / analyze (runs the plan: per-metric effect + CI + test, paired where declared; exploratory section separated) / report (markdown artifact, cold-reader-gate ready, .agent/demos pattern). (3) Assignment evidence: arm membership is an assertion row with evidence ref to the session — auditable, revocable. (4) Agent flow: the registry + query_units expose measures and their validity metadata; a well-formed spec is constructible from one MCP conversation; malformed specs (unregistered metric, no direction, n=1 with unpaired plan) are refused with actionable errors. (5) First consumers: the uplift re-run (cfk) migrates onto this; 37t.9 prompt/context experiments; harness A/B (same task class, model arms). Non-goal: automatic arm assignment inside agent harnesses — assignment stays explicit/observable.

## Acceptance criteria

cfk's protocol is expressible as an ExperimentSpec and its analysis reproduces via experiment analyze. An agent (via MCP) can define a valid two-arm experiment end-to-end against the seeded corpus; malformed specs are refused with the missing field named. Prereg vs exploratory metrics render separately in the report.

## Static mechanism / likely defect

Issue description localizes the mechanism: Generalize what cfk/jxe did by hand into substrate: an experiment is a first-class declared object — hypothesis, arms, assignment rule, PREREGISTERED metrics (declared before data collection, timestamped — the construct-validity teeth), sample-size intent, analysis plan — and the archive hosts its lifecycle: assignment, observation collection (sessions tagged to arms), paired/grouped analysis through the measure registry, and a cold-reader-gateable report. Agent affordance is the point (operator ask): agents shoul… Design direction: (1) ExperimentSpec as a user.db artifact (assertion-adjacent, judgment-visible): hypothesis, arms (name + treatment description), assignment (manual | alternating | by-session-property), preregistered metrics (each a registry measure ref + direction + minimum-interesting-effect), planned n, analysis plan (paired vs unpaired, test choice via 9l5.7). Prereg timestamp is the assertion created_at — post-hoc metric addit…

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

1. (1) ExperimentSpec as a user.db artifact (assertion-adjacent, judgment-visible): hypothesis, arms (name + treatment description), assignment (manual | alternating | by-session-property), preregistered metrics (each a registry measure ref + direction + minimum-interesting-effect), planned n, analysis plan (paired vs unpaired, test choice via 9l5.7).
2. Prereg timestamp is the assertion created_at — post-hoc metric additions are visibly post-hoc (labeled exploratory).
3. (2) Lifecycle tools (CLI + MCP): experiment define / assign <session-ref> <arm> / status (n per arm, power-ish progress vs planned n) / analyze (runs the plan: per-metric effect + CI + test, paired where declared
4. exploratory section separated) / report (markdown artifact, cold-reader-gate ready, .agent/demos pattern).
5. (3) Assignment evidence: arm membership is an assertion row with evidence ref to the session — auditable, revocable.
6. (4) Agent flow: the registry + query_units expose measures and their validity metadata
7. a well-formed spec is constructible from one MCP conversation

## Tests to add

- Acceptance proof: cfk's protocol is expressible as an ExperimentSpec and its analysis reproduces via experiment analyze.
- Acceptance proof: An agent (via MCP) can define a valid two-arm experiment end-to-end against the seeded corpus
- Acceptance proof: malformed specs are refused with the missing field named.
- Acceptance proof: Prereg vs exploratory metrics render separately in the report.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
