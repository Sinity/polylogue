# 070. polylogue-t8t — Agent workflow catalog: walk the seven core flows end-to-end, fix what breaks

Priority/type/status: **P2 / task / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Affordances exist in pieces; nobody has walked the actual workflows end-to-end: (1) RESUME — arrive in repo, recover context, continue; (2) FORENSIC DEBUG — did a past session touch this; (3) PRIOR ART — has anything explored this approach; (4) DECISION LOOKUP — what did we decide and why; (5) POSTMORTEM WRITE — close the loop after failure (37t.7); (6) COST CHECK — what has this repo/task cost; (7) SELF-INSPECTION — agent reads its own live session mid-flight (raw-log 05-28; needs hook-fresh ingest + get_session on self). Walk each as a real agent over MCP against the live archive, time it, file every gap — the empirical complement to the cookbook (pj8).

## Existing design note

(1) Each flow = a workflow registry entry (product/workflows.py REQUIRED_WORKFLOW_IDS) with intent, tool sequence, envelope shapes, round-trip token cost, latency budget (20d.14). (2) Execute from a real Claude Code session via polylogue MCP on this machine; archive the walk transcript as evidence. (3) Verify-or-refute known suspects: self-inspection freshness (searchable within seconds?), search-within-session ergonomics, resume-brief token cost vs preamble budget, cost-check cold latency. (4) Output: per flow PASS+timing or a filed bead; catalog doc renders from the registry (drift-checked) and becomes the reference the cookbook prompts point at.

## Acceptance criteria

Seven registry entries; seven archived walk transcripts; every gap filed as a linked bead; rendered catalog lists measured tokens+latency per flow; self-inspection demonstrates an agent reading its own in-progress session.

## Static mechanism / likely defect

Issue description localizes the mechanism: Affordances exist in pieces; nobody has walked the actual workflows end-to-end: (1) RESUME — arrive in repo, recover context, continue; (2) FORENSIC DEBUG — did a past session touch this; (3) PRIOR ART — has anything explored this approach; (4) DECISION LOOKUP — what did we decide and why; (5) POSTMORTEM WRITE — close the loop after failure (37t.7); (6) COST CHECK — what has this repo/task cost; (7) SELF-INSPECTION — agent reads its own live session mid-flight (raw-log 05-28; needs hook-fresh ingest + get_session … Design direction: (1) Each flow = a workflow registry entry (product/workflows.py REQUIRED_WORKFLOW_IDS) with intent, tool sequence, envelope shapes, round-trip token cost, latency budget (20d.14). (2) Execute from a real Claude Code session via polylogue MCP on this machine; archive the walk transcript as evidence. (3) Verify-or-refute known suspects: self-inspection freshness (searchable within seconds?), search-within-session ergo…

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

1. (1) Each flow = a workflow registry entry (product/workflows.py REQUIRED_WORKFLOW_IDS) with intent, tool sequence, envelope shapes, round-trip token cost, latency budget (20d.14).
2. (2) Execute from a real Claude Code session via polylogue MCP on this machine
3. archive the walk transcript as evidence.
4. (3) Verify-or-refute known suspects: self-inspection freshness (searchable within seconds?), search-within-session ergonomics, resume-brief token cost vs preamble budget, cost-check cold latency.
5. (4) Output: per flow PASS+timing or a filed bead
6. catalog doc renders from the registry (drift-checked) and becomes the reference the cookbook prompts point at.

## Tests to add

- Acceptance proof: Seven registry entries
- Acceptance proof: seven archived walk transcripts
- Acceptance proof: every gap filed as a linked bead
- Acceptance proof: rendered catalog lists measured tokens+latency per flow
- Acceptance proof: self-inspection demonstrates an agent reading its own in-progress session.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
