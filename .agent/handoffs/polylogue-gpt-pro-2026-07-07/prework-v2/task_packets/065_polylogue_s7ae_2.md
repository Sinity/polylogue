# 065. polylogue-s7ae.2 — Pre-deployment MCP and hook coordination batch

Priority/type/status: **P1 / task / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **blocked-hard**.

Hard blockers: polylogue-pj8

## What the bead says

Why: the coordination program will require MCP prompt/tool updates and harness/hook rollout. Deployment should not happen piecemeal after every small MCP change. Before asking for a Sinnix/Home Manager switch or other deployment, batch all MCP-related code/config/test work that can be completed locally, including Beads hook health integration and subtle Polylogue hook affordances. If deployment is the only remaining step, record that state and move on to other work.

## Existing design note

Audit and implement the deployment-sensitive pieces in one pass: Polylogue MCP tools/prompts for coordination views, server tool contract registration, CLI/openapi/generated schema updates, Sinnix MCP registry implications if needed, Beads git hook health detection/reporting, and hook-mediated coordination source points. Keep hooks subtle: mostly silent evidence capture/liveness updates; visible advisories only through the context scheduler and only for material events such as direct messages, same-resource activity, stale roots, or merge/integration state. Install/verify Beads git hooks as part of the actual implementation lane, but treat hook installation as environment setup plus proof, not as the coordination ontology. Do not deploy until all predeploy MCP/hook code paths and tests are done; then leave a bead note that deployment is ready/needed and continue other non-deployment work.

## Acceptance criteria

MCP prompt/tool surface for coordination is implemented or explicitly delegated to the envelope bead with no remaining predeploy MCP code gaps. Generated MCP/OpenAPI/CLI schemas are refreshed where required. Beads hook health is visible in devloop review/status and the coordination envelope when Beads is present. Beads git hooks are installed/verified in the Polylogue checkout or a precise blocker is recorded. Hook-based coordination capture/advisory paths are designed and tested without noisy hardcoded workflow policing. Focused tests and generated checks pass. The bead notes explicitly say either 'pre-deployment complete; deployment required' or list remaining pre-deployment work; agents must not request deployment until the former is true.

## Static mechanism / likely defect

Issue description localizes the mechanism: Why: the coordination program will require MCP prompt/tool updates and harness/hook rollout. Deployment should not happen piecemeal after every small MCP change. Before asking for a Sinnix/Home Manager switch or other deployment, batch all MCP-related code/config/test work that can be completed locally, including Beads hook health integration and subtle Polylogue hook affordances. If deployment is the only remaining step, record that state and move on to other work. Design direction: Audit and implement the deployment-sensitive pieces in one pass: Polylogue MCP tools/prompts for coordination views, server tool contract registration, CLI/openapi/generated schema updates, Sinnix MCP registry implications if needed, Beads git hook health detection/reporting, and hook-mediated coordination source points. Keep hooks subtle: mostly silent evidence capture/liveness updates; visible advisories only thro…

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

1. Audit and implement the deployment-sensitive pieces in one pass: Polylogue MCP tools/prompts for coordination views, server tool contract registration, CLI/openapi/generated schema updates, Sinnix MCP registry implications if needed, Beads git hook health detection/reporting, and hook-mediated coordination source points.
2. Keep hooks subtle: mostly silent evidence capture/liveness updates
3. visible advisories only through the context scheduler and only for material events such as direct messages, same-resource activity, stale roots, or merge/integration state.
4. Install/verify Beads git hooks as part of the actual implementation lane, but treat hook installation as environment setup plus proof, not as the coordination ontology.
5. Do not deploy until all predeploy MCP/hook code paths and tests are done
6. then leave a bead note that deployment is ready/needed and continue other non-deployment work.

## Tests to add

- Acceptance proof: MCP prompt/tool surface for coordination is implemented or explicitly delegated to the envelope bead with no remaining predeploy MCP code gaps.
- Acceptance proof: Generated MCP/OpenAPI/CLI schemas are refreshed where required.
- Acceptance proof: Beads hook health is visible in devloop review/status and the coordination envelope when Beads is present.
- Acceptance proof: Beads git hooks are installed/verified in the Polylogue checkout or a precise blocker is recorded.
- Acceptance proof: Hook-based coordination capture/advisory paths are designed and tested without noisy hardcoded workflow policing.
- Acceptance proof: Focused tests and generated checks pass.
- Acceptance proof: The bead notes explicitly say either 'pre-deployment complete
- Acceptance proof: deployment required' or list remaining pre-deployment work

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
