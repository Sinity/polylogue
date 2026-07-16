# 071. polylogue-rii.1 — Agent work-event write-leg -> session_events -> materialized read-models

Priority/type/status: **P2 / feature / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

record_work_event/emit_decision write surface routed through the existing idempotent ingest seam (no parallel writer); flows into the run-projection read models. Today agents can only record_correction/blackboard_post/tag — there is no 'I ran this tool / spawned this subagent / decided X' write. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

Route through the existing idempotent ingest seam (write_raw_and_parsed / the daemon ingest path) — no parallel writer (gh#2459 body is code-grounded here). Surface: MCP tools record_work_event/emit_decision (mutation role) accepting typed events (tool run, subagent spawn, decision, artifact change) with evidence/session refs; land in session_events; run-projection read models pick them up through the normal materializer. MCP registration trap: EXPECTED_TOOL_NAMES + TOOL_CONTRACT + role gating + render openapi/cli-output-schemas regen (see bd memories). Acceptance: an agent posts a work event mid-session; it is queryable via observed-events within one convergence cycle; re-posting is idempotent.

## Acceptance criteria

- MCP tools record_work_event / emit_decision are registered with the mutation role: EXPECTED_TOOL_NAMES + TOOL_CONTRACT updated, role gating enforced, and `devtools render openapi && devtools render cli-output-schemas` regenerated with `devtools render all --check` clean.
- Typed events (tool run, subagent spawn, decision, artifact change) with evidence/session refs route through the existing idempotent ingest seam (write_raw_and_parsed / the daemon ingest path) into session_events — no parallel writer (grep confirms reuse).
- Behavior test: an agent posts a work event mid-session and it is queryable via observed-events (session_work_events / DSL) within one convergence cycle; re-posting the same event is idempotent (no duplicate row). `devtools test <mcp work-event test>` green.

## Static mechanism / likely defect

Issue description localizes the mechanism: record_work_event/emit_decision write surface routed through the existing idempotent ingest seam (no parallel writer); flows into the run-projection read models. Today agents can only record_correction/blackboard_post/tag — there is no 'I ran this tool / spawned this subagent / decided X' write. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Route through the existing idempotent ingest seam (write_raw_and_parsed / the daemon ingest path) — no parallel writer (gh#2459 body is code-grounded here). Surface: MCP tools record_work_event/emit_decision (mutation role) accepting typed events (tool run, subagent spawn, decision, artifact change) with evidence/session refs; land in session_events; run-projection read models pick them up through the normal materia…

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

1. Route through the existing idempotent ingest seam (write_raw_and_parsed / the daemon ingest path) — no parallel writer (gh#2459 body is code-grounded here).
2. Surface: MCP tools record_work_event/emit_decision (mutation role) accepting typed events (tool run, subagent spawn, decision, artifact change) with evidence/session refs
3. land in session_events
4. run-projection read models pick them up through the normal materializer.
5. MCP registration trap: EXPECTED_TOOL_NAMES + TOOL_CONTRACT + role gating + render openapi/cli-output-schemas regen (see bd memories).
6. Acceptance: an agent posts a work event mid-session
7. it is queryable via observed-events within one convergence cycle

## Tests to add

- Acceptance proof: MCP tools record_work_event / emit_decision are registered with the mutation role: EXPECTED_TOOL_NAMES + TOOL_CONTRACT updated, role gating enforced, and `devtools render openapi && devtools render cli-output-schemas` regenerated with `devtools render all --check` clean.
- Acceptance proof: Typed events (tool run, subagent spawn, decision, artifact change) with evidence/session refs route through the existing idempotent ingest seam (write_raw_and_parsed / the daemon ingest path) into session_events — no parallel writer (grep confirms reuse).
- Acceptance proof: Behavior test: an agent posts a work event mid-session and it is queryable via observed-events (session_work_events / DSL) within one convergence cycle
- Acceptance proof: re-posting the same event is idempotent (no duplicate row).
- Acceptance proof: `devtools test <mcp work-event test>` green.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
