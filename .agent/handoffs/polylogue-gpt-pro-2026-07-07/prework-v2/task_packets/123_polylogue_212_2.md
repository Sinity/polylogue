# 123. polylogue-212.2 — D1 'The receipts': claim-vs-evidence on a real PR

Priority/type/status: **P2 / task / open**. Lane: **12-external-legibility-demos**. Release: **L-legibility-demos**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Pick a merged agent-authored PR; resolve PR -> authoring session via session_commits/session_repos; get_postmortem_bundle; render two columns: claimed (PR-body sentences: 'tests pass') vs observed (actions rows: the pytest invocation, exit_code, duration — drillable to the raw tool_result block). A PR body audited against ground truth in ~10 seconds. Nearly free: all reads exist. Tell the deleted-prose-miner story as part of the demo (why this exists).

## Existing design note

A demo: pick a merged agent-authored PR, resolve PR->authoring session via session_commits/session_repos, run get_postmortem_bundle, and render two columns, claimed (PR-body sentences like 'tests pass') vs observed (actions rows: the pytest invocation, exit_code, duration, drillable to the raw tool_result block). Audits a PR body against ground truth in ~10 seconds. All reads exist; tell the deleted-prose-miner story as motivation.

## Acceptance criteria

1. For a chosen merged agent-authored PR, the demo resolves the authoring session from session_commits/session_repos and produces a two-column claim-vs-evidence view: PR-body claim sentences beside the observed actions rows (invocation, exit_code, duration), drillable to the raw tool_result block. 2. The demo composes only existing reads (get_postmortem_bundle) with no new query machinery and includes the deleted-prose-miner motivation. Verify: run against a real merged PR and its authoring session (recorded artifact); the drill-through resolves to an actual tool_result block.

## Static mechanism / likely defect

Issue description localizes the mechanism: Pick a merged agent-authored PR; resolve PR -> authoring session via session_commits/session_repos; get_postmortem_bundle; render two columns: claimed (PR-body sentences: 'tests pass') vs observed (actions rows: the pytest invocation, exit_code, duration — drillable to the raw tool_result block). A PR body audited against ground truth in ~10 seconds. Nearly free: all reads exist. Tell the deleted-prose-miner story as part of the demo (why this exists). Design direction: A demo: pick a merged agent-authored PR, resolve PR->authoring session via session_commits/session_repos, run get_postmortem_bundle, and render two columns, claimed (PR-body sentences like 'tests pass') vs observed (actions rows: the pytest invocation, exit_code, duration, drillable to the raw tool_result block). Audits a PR body against ground truth in ~10 seconds. All reads exist; tell the deleted-prose-miner stor…

## Source anchors to inspect first

- `README.md` — Public claims should be grounded through the claims ledger.
- `docs/agent-forensics.md` — Existing forensics docs are a pattern for proof artifacts.
- `docs/demo.md` — Demo docs should depend on evidence/citation machinery.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.
- `polylogue/archive/actions/followup.py` — Action/followup classification is a real structural analytics input.
- `polylogue/archive/actions/fields.py` — Action fields determine what can be measured without prose heuristics.
- `polylogue/insights/registry.py:294` — Insight registry should become measure/product registry input.
- `scripts/agent_forensics.py` — Existing forensics script is a proof artifact and candidate product surface.

## Implementation plan

1. A demo: pick a merged agent-authored PR, resolve PR->authoring session via session_commits/session_repos, run get_postmortem_bundle, and render two columns, claimed (PR-body sentences like 'tests pass') vs observed (actions rows: the pytest invocation, exit_code, duration, drillable to the raw tool_result block).
2. Audits a PR body against ground truth in ~10 seconds.
3. All reads exist
4. tell the deleted-prose-miner story as motivation.

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: For a chosen merged agent-authored PR, the demo resolves the authoring session from session_commits/session_repos and produces a two-column claim-vs-evidence view: PR-body claim sentences beside the observed actions rows (invocation, exit_code, duration), drillable to the raw tool_result block.
- Acceptance proof: 2.
- Acceptance proof: The demo composes only existing reads (get_postmortem_bundle) with no new query machinery and includes the deleted-prose-miner motivation.
- Acceptance proof: Verify: run against a real merged PR and its authoring session (recorded artifact)
- Acceptance proof: the drill-through resolves to an actual tool_result block.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --docs --demo`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
