# 067. polylogue-s7ae.5 — Live proof: two agents, separate worktrees, one repo — overlap, message, context, handoff

Priority/type/status: **P1 / task / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **blocked-hard**.

Hard blockers: polylogue-37t.11, polylogue-s7ae.3

## What the bead says

_No description in export._

## Existing design note

Realize the epic s7ae HEADLINE acceptance line ('A live proof demonstrates at least two agents on one repo with separate worktrees, visible overlap/resource awareness, a scoped coordination message, context injection, and a handoff packet') — currently owned by no child. Build a reproducible proof (run script + captured JSON artifacts) demonstrating two agents (e.g. Claude + Codex) on ONE repo in SEPARATE git worktrees, showing via the coordination envelope: (a) mutual peer + same-repo-agent + resource-episode awareness (process-table overlaps already shipped in s7ae.1); (b) at least one SCOPED coordination message posted by one agent and observed as delivered/addressed in the other's envelope (s7ae.3); (c) context injection into the second agent recorded via the 37t.11 scheduler/ledger; (d) a handoff packet (.agent/conductor-devloop/*.md) produced by one agent and referenced by both. Implement as a deterministic devtools workspace proof (mirror the existing devtools/workspace proof-artifact pattern, e.g. degraded-archive-proof) plus an operator run script; capture before/after `polylogue agents status --json` envelopes from both agents as the evidence artifact. Pitfalls: overlap is awareness-not-blocker (assert non-blocking); the proof must run without private corpus; the coordination message must actually round-trip through the message store (no fabricated delivery). Also depends on the new archive session-evidence composition child (this batch) so the captured envelope carries session-tree lineage, and on 37t.11 for the context-injection leg.

## Acceptance criteria

A committed, reproducible proof exists (run script + captured before/after JSON envelope artifacts under a devtools workspace path) demonstrating: two agents on one repo in separate worktrees; each envelope shows the other as a same-repo peer with overlap + resource-episode awareness; exactly one scoped coordination message posted and observed as delivered/addressed in the recipient's envelope; context injection recorded via the 37t.11 ledger; a handoff packet produced and referenced by both agents. The proof runs from one documented command and the epic s7ae live-proof acceptance line is explicitly marked satisfied by this artifact. Deps: s7ae.1 (envelope/overlap/handoff), s7ae.3 (scoped message), 37t.11 (context injection), and the archive session-evidence composition child (session-tree lineage).

## Static mechanism / likely defect

Design direction: Realize the epic s7ae HEADLINE acceptance line ('A live proof demonstrates at least two agents on one repo with separate worktrees, visible overlap/resource awareness, a scoped coordination message, context injection, and a handoff packet') — currently owned by no child. Build a reproducible proof (run script + captured JSON artifacts) demonstrating two agents (e.g. Claude + Codex) on ONE repo in SEPARATE git worktr…

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

1. Realize the epic s7ae HEADLINE acceptance line ('A live proof demonstrates at least two agents on one repo with separate worktrees, visible overlap/resource awareness, a scoped coordination message, context injection, and a handoff packet') — currently owned by no child.
2. Build a reproducible proof (run script + captured JSON artifacts) demonstrating two agents (e.g.
3. Claude + Codex) on ONE repo in SEPARATE git worktrees, showing via the coordination envelope: (a) mutual peer + same-repo-agent + resource-episode awareness (process-table overlaps already shipped in s7ae.1)
4. (b) at least one SCOPED coordination message posted by one agent and observed as delivered/addressed in the other's envelope (s7ae.3)
5. (c) context injection into the second agent recorded via the 37t.11 scheduler/ledger
6. (d) a handoff packet (.agent/conductor-devloop/*.md) produced by one agent and referenced by both.
7. Implement as a deterministic devtools workspace proof (mirror the existing devtools/workspace proof-artifact pattern, e.g.

## Tests to add

- Acceptance proof: A committed, reproducible proof exists (run script + captured before/after JSON envelope artifacts under a devtools workspace path) demonstrating: two agents on one repo in separate worktrees
- Acceptance proof: each envelope shows the other as a same-repo peer with overlap + resource-episode awareness
- Acceptance proof: exactly one scoped coordination message posted and observed as delivered/addressed in the recipient's envelope
- Acceptance proof: context injection recorded via the 37t.11 ledger
- Acceptance proof: a handoff packet produced and referenced by both agents.
- Acceptance proof: The proof runs from one documented command and the epic s7ae live-proof acceptance line is explicitly marked satisfied by this artifact.
- Acceptance proof: Deps: s7ae.1 (envelope/overlap/handoff), s7ae.3 (scoped message), 37t.11 (context injection), and the archive session-evidence composition child (session-tree lineage).

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
