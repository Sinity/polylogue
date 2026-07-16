# 066. polylogue-s7ae.3 — Coordination messages and subtle scheduler-mediated advisories

Priority/type/status: **P1 / feature / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **blocked-hard**.

Hard blockers: polylogue-37t.11

## What the bead says

Why: multi-agent cooperation needs lightweight communication and awareness, but not a noisy chatroom or hardcoded workflow police. Agents should be able to leave scoped messages, receive direct or relevant notices, and see overlap/resource awareness. Hooks should mostly capture facts silently; visible output should be rare, bounded, and mediated by the context scheduler.

## Existing design note

Realize the coordination-specific parts of existing beads 1hj (blackboard as agent comms), bfv (advisory hooks), d1y (hook install/liveness), and 37t.11 (ContextSource scheduler/ledger). Prefer reusing blackboard/user rows for CoordinationMessage. Addressing scopes: repo, work-item, session-tree, direct session/agent, path/surface, resource scope, broadcast only when explicitly requested. Delivery: SessionStart and on-demand context snapshots first; mid-session advisories only for direct messages or high-value material changes. Advisory examples are generic facts, not policy: sibling touched same surface, same archive/resource episode active, merge slot held, hook/daemon/archive root stale, direct message addressed to this agent. Same-file editing is overlap awareness, not a block. Command/test/build/import/daemon activity is modeled as ActivityEpisode with optional heuristic family; unknown commands still surface as resource-scoped activity. Context scheduler owns token budget, dedup, cooldown, trust class, and ledger; hooks do not assemble their own context.

## Acceptance criteria

Agents can post and receive scoped coordination messages with refs/provenance, using existing blackboard/user-state machinery where viable. The coordination envelope exposes unread/addressed messages and recent advisories. Hook-triggered visible advisories are bounded, rare, and emitted only through the scheduler/ledger path. Tests cover direct message delivery, repo/work-item scoped delivery, TTL/expiry or equivalent boundedness, same-surface overlap as non-blocking awareness, generic resource episode warning, and no noisy injection when there is no material signal. MCP/CLI expose message/advisory read paths without requiring Beads; Beads-backed scopes work when available.

## Static mechanism / likely defect

Issue description localizes the mechanism: Why: multi-agent cooperation needs lightweight communication and awareness, but not a noisy chatroom or hardcoded workflow police. Agents should be able to leave scoped messages, receive direct or relevant notices, and see overlap/resource awareness. Hooks should mostly capture facts silently; visible output should be rare, bounded, and mediated by the context scheduler. Design direction: Realize the coordination-specific parts of existing beads 1hj (blackboard as agent comms), bfv (advisory hooks), d1y (hook install/liveness), and 37t.11 (ContextSource scheduler/ledger). Prefer reusing blackboard/user rows for CoordinationMessage. Addressing scopes: repo, work-item, session-tree, direct session/agent, path/surface, resource scope, broadcast only when explicitly requested. Delivery: SessionStart and …

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

1. Realize the coordination-specific parts of existing beads 1hj (blackboard as agent comms), bfv (advisory hooks), d1y (hook install/liveness), and 37t.11 (ContextSource scheduler/ledger).
2. Prefer reusing blackboard/user rows for CoordinationMessage.
3. Addressing scopes: repo, work-item, session-tree, direct session/agent, path/surface, resource scope, broadcast only when explicitly requested.
4. Delivery: SessionStart and on-demand context snapshots first
5. mid-session advisories only for direct messages or high-value material changes.
6. Advisory examples are generic facts, not policy: sibling touched same surface, same archive/resource episode active, merge slot held, hook/daemon/archive root stale, direct message addressed to this agent.
7. Same-file editing is overlap awareness, not a block.

## Tests to add

- Acceptance proof: Agents can post and receive scoped coordination messages with refs/provenance, using existing blackboard/user-state machinery where viable.
- Acceptance proof: The coordination envelope exposes unread/addressed messages and recent advisories.
- Acceptance proof: Hook-triggered visible advisories are bounded, rare, and emitted only through the scheduler/ledger path.
- Acceptance proof: Tests cover direct message delivery, repo/work-item scoped delivery, TTL/expiry or equivalent boundedness, same-surface overlap as non-blocking awareness, generic resource episode warning, and no noisy injection when there is no material signal.
- Acceptance proof: MCP/CLI expose message/advisory read paths without requiring Beads
- Acceptance proof: Beads-backed scopes work when available.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
