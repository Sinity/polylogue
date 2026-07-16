# 073. polylogue-x4s — Express devloop state in Polylogue substrate (dogfood target)

Priority/type/status: **P2 / task / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Raw-log 2026-07-03: 'perhaps devloops themselves could be expressed in sinex and/or polylogue and/or beads?'. Beads now owns task state. The remaining half: focus transitions, handoffs, proof claims, and velocity notes should eventually be archive/assertion data rather than markdown sidecars only. Candidate first slice: devloop-log dual-writes an assertion (kind=NOTE, author=devloop) with evidence refs to the producing session. Route through the substrate write-leg rather than a parallel writer.

## Existing design note

The full argument (fables devloop reading): the conductor's own memory is the silo it is fighting — ACTIVE-LOOP.md, OPERATING-LOG.md, HANDOFF-LATEST.md, EVENTS.jsonl live outside the archive while the product has the native home sitting unwired: handoff and run_state assertion kinds exist in the enum with NO writer (user_write.py has no helper for them), blackboard has an unresolved filter, and get_resume_brief/compose_context_preamble already do provenance-cited handoffs. Target: conductor-on-assertions — active-loop state, handoffs, and focus transitions written as assertions in user.db, recovered at session start through the product's own context compilation instead of 'read these 11 files in order'. The post-compaction discipline exists because agent context is lossy — that is the product's founding problem, currently solved with markdown instead of the product. Sequence: (1) first writers for handoff/run_state kinds (dual-write from devloop-handoff/devloop-focus, markdown stays authoritative), (2) a conductor context profile in compile_context, (3) flip authority once recovery quality is proven, keeping markdown as a rendered VIEW of the assertions rather than the source. Coordinate with the beads split: beads owns task state; assertions own narrative/handoff/decision state (see the beads-vs-assertions decision bead).

## Acceptance criteria

- First writers for the handoff/run_state assertion kinds are added to user_write.py (the kinds already exist in the enum with no writer — grep confirms the new helpers); devloop-handoff / devloop-focus dual-write into user.db while markdown stays authoritative.
- A conductor context profile is added to compile_context that recovers active-loop state, handoffs, and focus transitions at session start (via get_resume_brief / compose_context_preamble, with provenance).
- Authority flips to assertions once recovery quality is proven; markdown becomes a rendered VIEW of the assertions, not the source.
- Migration-ladder rungs each retire their file in the same PR that ships the replacement (rung1 beads-native loop replacing ACTIVE-LOOP.md; rung2 OPERATING-LOG entries -> work-event/assertion writes rendered as a log view; rung3 handoff packets -> read-packages / resume briefs; rung4 devloop-status -> a polylogue status profile + `bd ready` join) — no deprecation theater.
- Coordinated with the beads-vs-assertions split (beads own task state; assertions own narrative/handoff/decision state).
- `devtools test <writer + context profile tests>` green; a session-start recovery reconstructs conductor state from the assertions.

## Static mechanism / likely defect

Issue description localizes the mechanism: Raw-log 2026-07-03: 'perhaps devloops themselves could be expressed in sinex and/or polylogue and/or beads?'. Beads now owns task state. The remaining half: focus transitions, handoffs, proof claims, and velocity notes should eventually be archive/assertion data rather than markdown sidecars only. Candidate first slice: devloop-log dual-writes an assertion (kind=NOTE, author=devloop) with evidence refs to the producing session. Route through the substrate write-leg rather than a parallel writer. Design direction: The full argument (fables devloop reading): the conductor's own memory is the silo it is fighting — ACTIVE-LOOP.md, OPERATING-LOG.md, HANDOFF-LATEST.md, EVENTS.jsonl live outside the archive while the product has the native home sitting unwired: handoff and run_state assertion kinds exist in the enum with NO writer (user_write.py has no helper for them), blackboard has an unresolved filter, and get_resume_brief/comp…

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

1. The full argument (fables devloop reading): the conductor's own memory is the silo it is fighting — ACTIVE-LOOP.md, OPERATING-LOG.md, HANDOFF-LATEST.md, EVENTS.jsonl live outside the archive while the product has the native home sitting unwired: handoff and run_state assertion kinds exist in the enum with NO writer (user_write.py has no helper for them), blackboard has an unresolved filter, and get_resume_brief/co…
2. Target: conductor-on-assertions — active-loop state, handoffs, and focus transitions written as assertions in user.db, recovered at session start through the product's own context compilation instead of 'read these 11 files in order'.
3. The post-compaction discipline exists because agent context is lossy — that is the product's founding problem, currently solved with markdown instead of the product.
4. Sequence: (1) first writers for handoff/run_state kinds (dual-write from devloop-handoff/devloop-focus, markdown stays authoritative), (2) a conductor context profile in compile_context, (3) flip authority once recovery quality is proven, keeping markdown as a rendered VIEW of the assertions rather than the source.
5. Coordinate with the beads split: beads owns task state
6. assertions own narrative/handoff/decision state (see the beads-vs-assertions decision bead).

## Tests to add

- Acceptance proof: First writers for the handoff/run_state assertion kinds are added to user_write.py (the kinds already exist in the enum with no writer — grep confirms the new helpers)
- Acceptance proof: devloop-handoff / devloop-focus dual-write into user.db while markdown stays authoritative.
- Acceptance proof: A conductor context profile is added to compile_context that recovers active-loop state, handoffs, and focus transitions at session start (via get_resume_brief / compose_context_preamble, with provenance).
- Acceptance proof: Authority flips to assertions once recovery quality is proven
- Acceptance proof: markdown becomes a rendered VIEW of the assertions, not the source.
- Acceptance proof: Migration-ladder rungs each retire their file in the same PR that ships the replacement (rung1 beads-native loop replacing ACTIVE-LOOP.md
- Acceptance proof: rung2 OPERATING-LOG entries -> work-event/assertion writes rendered as a log view
- Acceptance proof: rung3 handoff packets -> read-packages / resume briefs

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
