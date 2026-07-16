# 169. polylogue-37t.7 — Close the failure loop: verify postmortem -> next session's context seed

Priority/type/status: **P2 / feature / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

workspace failure-context produces an envelope (testmon graph + git history + fixtures for a failing test); the pytest supervisor produces a postmortem (.cache/verify/) — neither flows into the next agent session. Build the splice: a compile_context seed constructed from the latest verify postmortem + failure-context envelope, injectable via the SessionStart hook or an explicit `polylogue context --from-verify` entry. The obvious first consumer is the devloop itself after a red verify run.

## Existing design note

`workspace failure-context` produces an envelope (testmon graph + git history + fixtures for a failing test) and the pytest supervisor produces a postmortem (.cache/verify/) — neither flows into the next agent session. Build the splice: a compile_context seed constructed from the latest verify postmortem + failure-context envelope, injectable via the SessionStart hook or an explicit `polylogue context --from-verify`. First consumer: the devloop itself after a red verify run. Test discipline: session-cut recovery drills (chaos-lane, yeq) — deliberately kill sessions mid-work and measure whether the next session recovers unprompted from injected context alone.

## Acceptance criteria

- A compile_context seed is constructed from the latest verify postmortem (.cache/verify/) + the `workspace failure-context` envelope (testmon graph + git history + fixtures), injectable via the SessionStart hook or an explicit `polylogue context --from-verify` entry point.
- The first consumer is wired: the devloop injects the seed after a red verify run.
- Verify: `polylogue context --from-verify` on a real red postmortem emits a seed containing the failing test plus the implicated files; `devtools test <context seed test>` green.
- Session-cut recovery drills (chaos-lane, yeq) are run: sessions are killed mid-work and the next session's unprompted recovery from injected context alone is measured; the recovery rate is recorded as the loop's KPI.

## Static mechanism / likely defect

Issue description localizes the mechanism: workspace failure-context produces an envelope (testmon graph + git history + fixtures for a failing test); the pytest supervisor produces a postmortem (.cache/verify/) — neither flows into the next agent session. Build the splice: a compile_context seed constructed from the latest verify postmortem + failure-context envelope, injectable via the SessionStart hook or an explicit `polylogue context --from-verify` entry. The obvious first consumer is the devloop itself after a red verify run. Design direction: `workspace failure-context` produces an envelope (testmon graph + git history + fixtures for a failing test) and the pytest supervisor produces a postmortem (.cache/verify/) — neither flows into the next agent session. Build the splice: a compile_context seed constructed from the latest verify postmortem + failure-context envelope, injectable via the SessionStart hook or an explicit `polylogue context --from-verify`…

## Source anchors to inspect first

- `polylogue/coordination/envelope.py` — Coordination envelope model exists; harden it as the shared payload.
- `polylogue/coordination/payloads.py` — Coordination payload types should stay small and evidence-ref oriented.
- `polylogue/coordination/rendering.py` — Rendered advisories should be scheduler-mediated, not chat spam.
- `tests/unit/coordination/test_envelope.py` — Existing envelope tests are the starting verification lane.
- `polylogue/mcp/server_prompts.py:219` — MCP prompt registration exists and can surface cookbook/roles.
- `polylogue/cli/commands/agents.py` — CLI agent commands are the operator-facing entry point.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:31` — ASSERTION_DEFAULT_STATUS is ACTIVE, so missing status currently means trusted active.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:641` — upsert_blackboard_note passes author_kind and no explicit status into upsert_assertion.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — upsert_assertion is the single write chokepoint to patch.
- `polylogue/api/contracts/assertions.py` — Check public assertion request/response contract after changing default status behavior.

## Implementation plan

1. `workspace failure-context` produces an envelope (testmon graph + git history + fixtures for a failing test) and the pytest supervisor produces a postmortem (.cache/verify/) — neither flows into the next agent session.
2. Build the splice: a compile_context seed constructed from the latest verify postmortem + failure-context envelope, injectable via the SessionStart hook or an explicit `polylogue context --from-verify`.
3. First consumer: the devloop itself after a red verify run.
4. Test discipline: session-cut recovery drills (chaos-lane, yeq) — deliberately kill sessions mid-work and measure whether the next session recovers unprompted from injected context alone.

## Tests to add

- Acceptance proof: A compile_context seed is constructed from the latest verify postmortem (.cache/verify/) + the `workspace failure-context` envelope (testmon graph + git history + fixtures), injectable via the SessionStart hook or an explicit `polylogue context --from-verify` entry point.
- Acceptance proof: The first consumer is wired: the devloop injects the seed after a red verify run.
- Acceptance proof: Verify: `polylogue context --from-verify` on a real red postmortem emits a seed containing the failing test plus the implicated files
- Acceptance proof: `devtools test <context seed test>` green.
- Acceptance proof: Session-cut recovery drills (chaos-lane, yeq) are run: sessions are killed mid-work and the next session's unprompted recovery from injected context alone is measured
- Acceptance proof: the recovery rate is recorded as the loop's KPI.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
