# 167. polylogue-37t.3 — Reboot-with-refs: session self-compaction protocol

Priority/type/status: **P2 / feature / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Agent reboots into a fresh session carrying all prose verbatim with every tool exchange collapsed to a one-line expandable ref — better than harness compaction because refs resolve via resolve_ref. Raw-log 06-29: refs over stripping; hierarchical expansion affordances.

## Existing design note

Flow: agent calls MCP compile_context with ContextSpec(seed_refs=[current session], purpose=continue) + a new prose_with_refs segment profile -> markdown with authored prose verbatim, tool_use/result collapsed to '-> [tool:Bash exit:0 4.2s] pytest ... <ref:action:...>'; new harness session; SessionStart hook injects when POLYLOGUE_REBOOT_FROM=<session_id> marker present (source field startup|resume|clear|compact; don't tax ordinary startups). VERIFY hookSpecificOutput.additionalContext field names against current Claude Code docs. Bundle header: 'expand any ref via resolve_ref before assuming content'. Lineage: session_links inheritance='spawned-fresh' link_type='continuation' new->old via record_manual_continuation (child, parent); also write the first handoff-kind assertion (kind exists, unwired). Budget rule: prose verbatim to 60% of budget, then oldest prose -> one-line recaps; keep first user message + last N turns verbatim; reuse ContextOmission with reason=budget.

## Acceptance criteria

- compile_context with a new prose_with_refs segment profile emits authored prose verbatim while every tool_use/result collapses to a one-line '<ref:action:...>' marker. Verify: pytest asserts each emitted ref resolves via resolve_ref back to the original block.
- Budget rule enforced: prose verbatim to 60% of budget, then oldest prose collapses to one-line recaps; first user message + last N turns kept verbatim; overflow recorded as ContextOmission(reason=budget) (ContextOmission at context/compiler.py:48). Test over a large seed session.
- New session lineage recorded: session_links inheritance='spawned-fresh', link_type='continuation' via record_manual_continuation(child, parent), and the first handoff-kind assertion is written. Verify: get_session_topology shows the continuation edge.
- hookSpecificOutput.additionalContext field names verified against current Claude Code SessionStart docs and the verification recorded in the PR.

## Static mechanism / likely defect

Issue description localizes the mechanism: Agent reboots into a fresh session carrying all prose verbatim with every tool exchange collapsed to a one-line expandable ref — better than harness compaction because refs resolve via resolve_ref. Raw-log 06-29: refs over stripping; hierarchical expansion affordances. Design direction: Flow: agent calls MCP compile_context with ContextSpec(seed_refs=[current session], purpose=continue) + a new prose_with_refs segment profile -> markdown with authored prose verbatim, tool_use/result collapsed to '-> [tool:Bash exit:0 4.2s] pytest ... <ref:action:...>'; new harness session; SessionStart hook injects when POLYLOGUE_REBOOT_FROM=<session_id> marker present (source field startup|resume|clear|compact; do…

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

1. Flow: agent calls MCP compile_context with ContextSpec(seed_refs=[current session], purpose=continue) + a new prose_with_refs segment profile -> markdown with authored prose verbatim, tool_use/result collapsed to '-> [tool:Bash exit:0 4.2s] pytest ...
2. <ref:action:...>'
3. new harness session
4. SessionStart hook injects when POLYLOGUE_REBOOT_FROM=<session_id> marker present (source field startup|resume|clear|compact
5. don't tax ordinary startups).
6. VERIFY hookSpecificOutput.additionalContext field names against current Claude Code docs.
7. Bundle header: 'expand any ref via resolve_ref before assuming content'.

## Tests to add

- Acceptance proof: compile_context with a new prose_with_refs segment profile emits authored prose verbatim while every tool_use/result collapses to a one-line '<ref:action:...>' marker.
- Acceptance proof: Verify: pytest asserts each emitted ref resolves via resolve_ref back to the original block.
- Acceptance proof: Budget rule enforced: prose verbatim to 60% of budget, then oldest prose collapses to one-line recaps
- Acceptance proof: first user message + last N turns kept verbatim
- Acceptance proof: overflow recorded as ContextOmission(reason=budget) (ContextOmission at context/compiler.py:48).
- Acceptance proof: Test over a large seed session.
- Acceptance proof: New session lineage recorded: session_links inheritance='spawned-fresh', link_type='continuation' via record_manual_continuation(child, parent), and the first handoff-kind assertion is written.
- Acceptance proof: Verify: get_session_topology shows the continuation edge.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
