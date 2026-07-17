# 113. polylogue-1vpm.1 — Delegation derived unit: materializer + query unit + delegation-card projection

Priority/type/status: **P2 / task / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

First-class delegations rows in index.db (derived, recomputable, extractor-versioned): delegation identity prefers (parent_session_id, tool_use_block_id) — never prompt text (identical prompts are different delegations). Row carries parent/child session+run refs, instruction/result block refs, task_id/tool_id, delegation_kind (subagent|background-agent|sidecar-report|async-task|unknown), harness, subagent_type/model/family, status, link_status (resolved|unresolved|inferred|quarantined), confidence, evidence+artifact refs. Extraction rules with per-provider confidence: Claude Task tool_use or subagent_type/agent_type input (agent-acompact-* excluded — continuation not delegation); Codex requires source.subagent.thread_spawn for kind=subagent; session_runs.role=subagent as neutral evidence. Every delegation ATTEMPT gets a row even with no resolved child (link_status=unresolved) or failed-delegation behavior is invisible. Then: delegation query unit (rows/count/group/select, joins assertion labels by target), delegation-card projection (instruction, parent context window, child output, PARENT-USE window — did the parent consume or ignore the result — artifacts, annotations, provenance), target_kind=delegation registered for assertions. Enables delegation-yield analytics (child cost vs parent-use rate; result_status only from actions.is_error/exit_code — unknown never enters an ROI denominator) and the orchestrator-rhetoric demo generalized beyond Fable (Fable is a cohort, not a feature). Verbatim spec: bundles/rnd-bundle-4-of-6.md L723-980.

## Acceptance criteria

Fixtures: Claude Task pair, acompact exclusion, Codex spawn, unresolved child, no false subagent from forked_from_id; delegations where parent.repo:X and status:failed works; card renders bounded (full prompts only under explicit opt-in); index bump batched. Verify: unit fixtures + query-unit tests.

## Static mechanism / likely defect

Issue description localizes the mechanism: First-class delegations rows in index.db (derived, recomputable, extractor-versioned): delegation identity prefers (parent_session_id, tool_use_block_id) — never prompt text (identical prompts are different delegations). Row carries parent/child session+run refs, instruction/result block refs, task_id/tool_id, delegation_kind (subagent|background-agent|sidecar-report|async-task|unknown), harness, subagent_type/model/family, status, link_status (resolved|unresolved|inferred|quarantined), confidence, evidence+artifa…

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

- Acceptance proof: Fixtures: Claude Task pair, acompact exclusion, Codex spawn, unresolved child, no false subagent from forked_from_id
- Acceptance proof: delegations where parent.repo:X and status:failed works
- Acceptance proof: card renders bounded (full prompts only under explicit opt-in)
- Acceptance proof: index bump batched.
- Acceptance proof: Verify: unit fixtures + query-unit tests.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
