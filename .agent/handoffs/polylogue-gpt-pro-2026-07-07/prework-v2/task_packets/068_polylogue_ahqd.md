# 068. polylogue-ahqd — Observe MCP write adoption after role rollout

Priority/type/status: **P1 / task / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **blocked-hard**.

Hard blockers: polylogue-s7ae.2

## What the bead says

Why: polylogue-27p made full/evidence/browser agent profiles write-capable and mutation rows author-attributed, but the current Codex process predates the Home Manager activation. The adoption proof should be collected from a freshly launched agent session using the write-role MCP server so the archive's affordance-usage report contains real MCP write calls rather than unit-test or shell simulations. What: launch or wait for a fresh full-profile agent, perform benign record_correction/add_tag/blackboard_post writes against a demo or clearly marked test session, then run devtools workspace affordance-usage or the equivalent Polylogue query to show the write calls and author refs.

## Acceptance criteria

A freshly launched full/evidence/browser agent session performs benign record_correction, add_tag, and blackboard_post MCP calls; the resulting archive rows carry the authoring session ref; an affordance-usage artifact/report shows those write calls; lean profile remains read-only.

## Static mechanism / likely defect

Issue description localizes the mechanism: Why: polylogue-27p made full/evidence/browser agent profiles write-capable and mutation rows author-attributed, but the current Codex process predates the Home Manager activation. The adoption proof should be collected from a freshly launched agent session using the write-role MCP server so the archive's affordance-usage report contains real MCP write calls rather than unit-test or shell simulations. What: launch or wait for a fresh full-profile agent, perform benign record_correction/add_tag/blackboard_post write…

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

1. Confirm all write paths go through the candidate/non-injected assertion policy.
2. Make scheduler/context-ledger the only injection route.
3. Add a minimal operator-visible envelope/payload and one proof fixture.
4. Verify two-agent or simulated-agent flow before broad rollout.

## Tests to add

- Acceptance proof: A freshly launched full/evidence/browser agent session performs benign record_correction, add_tag, and blackboard_post MCP calls
- Acceptance proof: the resulting archive rows carry the authoring session ref
- Acceptance proof: an affordance-usage artifact/report shows those write calls
- Acceptance proof: lean profile remains read-only.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
