# 160. polylogue-s7ae — Agent coordination substrate: evidence-backed multi-agent work without tracker lock-in

Priority/type/status: **P1 / epic / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **epic-needs-child-closure**.

## What the bead says

Why: Polylogue should make concurrent agent work operational, not merely visible. The target is a general coordination evidence layer over sessions, topology, repos/worktrees, work items, activity/resource episodes, context injection, messages, handoffs, and proof evidence. Beads is an important adapter when present, but the system must degrade to GitHub/git/session inference without Beads. The concrete Claude+Codex same-repo workflow should be realizable inside this substrate, not bolted into Polylogue as a special workflow.

## Existing design note

Core shape: add a reusable coordination envelope, not a web-only mission-control feature. The envelope composes existing Polylogue evidence: sessions, topology_edges, tool/action blocks, session events, context compiler/ledger, blackboard/user rows, daemon/hook liveness, git/worktree state, and optional task-system adapters. WorkItemRef is source-agnostic (beads|github|git|inferred|none) with provenance/confidence. CoordinationMessage should reuse blackboard/user-state machinery where viable. ActivityEpisode should reuse action/tool/session event evidence and add only missing normalization for resource scope/liveness. Hooks are subtle: capture facts and update presence quietly; visible advisories are bounded and scheduler-mediated. Surfaces are projections: CLI JSON, MCP prompts/tools, web mission control, context-source injection, and demos. Extant beads to realize under this program: bby.9 for the coordination envelope/web+CLI projection, pj8 for MCP prompt discoverability, ahqd for MCP write-adoption proof, 37t.11 for scheduler/ledger integration, d1y for hook installation/liveness, and bby.11 only where the web architecture must carry the projection.

## Acceptance criteria

A typed coordination envelope exists and is queryable without assuming Beads. It joins active/historical agent session trees with repo/worktree/branch, optional work item refs, activity/resource episodes, coordination messages/advisories, context-flow refs, proof/outcome summaries, and freshness/provenance/confidence. CLI and MCP expose bounded agent-grade views (status, self, work-item/current, conflicts/overlap, handoff, watch) with JSON-first output. Web mission control renders the same envelope rather than owning a separate ontology. Context injection uses the 37t.11 scheduler and ledger. Beads integration enriches the envelope when available, including hook health, gates, merge slot, claims, and dependencies; without Beads, git/GitHub/session inference still works. A live proof demonstrates at least two agents on one repo with separate worktrees, visible overlap/resource awareness, a scoped coordination message, context injection, and a handoff packet. Before any deployment/switch is requested, all MCP-related code/config/tests in this program are completed and recorded; if deployment is the remaining step, note that explicitly in this bead and move to other work.

## Static mechanism / likely defect

Issue description localizes the mechanism: Why: Polylogue should make concurrent agent work operational, not merely visible. The target is a general coordination evidence layer over sessions, topology, repos/worktrees, work items, activity/resource episodes, context injection, messages, handoffs, and proof evidence. Beads is an important adapter when present, but the system must degrade to GitHub/git/session inference without Beads. The concrete Claude+Codex same-repo workflow should be realizable inside this substrate, not bolted into Polylogue as a speci… Design direction: Core shape: add a reusable coordination envelope, not a web-only mission-control feature. The envelope composes existing Polylogue evidence: sessions, topology_edges, tool/action blocks, session events, context compiler/ledger, blackboard/user rows, daemon/hook liveness, git/worktree state, and optional task-system adapters. WorkItemRef is source-agnostic (beads|github|git|inferred|none) with provenance/confidence. …

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

1. Core shape: add a reusable coordination envelope, not a web-only mission-control feature.
2. The envelope composes existing Polylogue evidence: sessions, topology_edges, tool/action blocks, session events, context compiler/ledger, blackboard/user rows, daemon/hook liveness, git/worktree state, and optional task-system adapters.
3. WorkItemRef is source-agnostic (beads|github|git|inferred|none) with provenance/confidence.
4. CoordinationMessage should reuse blackboard/user-state machinery where viable.
5. ActivityEpisode should reuse action/tool/session event evidence and add only missing normalization for resource scope/liveness.
6. Hooks are subtle: capture facts and update presence quietly
7. visible advisories are bounded and scheduler-mediated.

## Tests to add

- Acceptance proof: A typed coordination envelope exists and is queryable without assuming Beads.
- Acceptance proof: It joins active/historical agent session trees with repo/worktree/branch, optional work item refs, activity/resource episodes, coordination messages/advisories, context-flow refs, proof/outcome summaries, and freshness/provenance/confidence.
- Acceptance proof: CLI and MCP expose bounded agent-grade views (status, self, work-item/current, conflicts/overlap, handoff, watch) with JSON-first output.
- Acceptance proof: Web mission control renders the same envelope rather than owning a separate ontology.
- Acceptance proof: Context injection uses the 37t.11 scheduler and ledger.
- Acceptance proof: Beads integration enriches the envelope when available, including hook health, gates, merge slot, claims, and dependencies
- Acceptance proof: without Beads, git/GitHub/session inference still works.
- Acceptance proof: A live proof demonstrates at least two agents on one repo with separate worktrees, visible overlap/resource awareness, a scoped coordination message, context injection, and a handoff packet.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.
- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
