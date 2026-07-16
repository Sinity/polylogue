# 162. polylogue-37t — Agent context/memory loop: declared claims -> judgment -> preamble -> reboot

Priority/type/status: **P2 / epic / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **epic-needs-child-closure**.

## What the bead says

The judged-memory loop: agents declare structured claims, the operator judges them, active claims compile into context preambles, and sessions reboot into compact evidence packs. Substrate exists (assertions, compile_context, compose_context_preamble, SessionStart hook); these children wire the loop closed. Raw-log design criteria (2026-06-29): entries timestamped, expiry metadata, navigable origin refs, restrained injection with expandable indices/refs.

## Existing design note

Epic spine: the loop closes when a declared claim travels end-to-end through all four named stages against the live archive. Stage owners: CLAIMS = 37t.2 (author-declared markers -> candidate assertions) + 37t.1 (assertion consumer wiring); JUDGMENT = operator bulk review/accept/reject of candidate assertions (currently unowned by any child — needs a judgment-queue bead); PREAMBLE = 37t.4 (SessionStart rollout) refactored onto 37t.11 (ContextSource scheduler/arbiter); REBOOT = 37t.3 (reboot-with-refs). 37t.11 is the coherence spine every preamble/recall source registers against. Each named stage must map to an open child with non-null acceptance; no stage may survive only as prose in a sibling's design field.

## Acceptance criteria

- A seeded end-to-end scenario test demonstrates one claim flowing claims->judgment->preamble->reboot: an agent emits a declared marker (37t.2) that lands as a candidate assertion, the operator accepts it via the judgment queue, it appears as a ref in a compiled SessionStart preamble for the matching repo (37t.4/37t.11), and it survives a reboot-with-refs handoff (37t.3) resolvable via resolve_ref. Verify: a pytest covering the four-stage path (e.g. tests/unit/context/test_judged_memory_loop.py) plus MCP resolve_ref on the emitted ref.
- Every named stage (claims/judgment/preamble/reboot) has an owning open bead with non-null acceptance_criteria; no stage survives only as prose in a sibling's design field. Verify: `bd show` on each child confirms acceptance present.
- The compiled preamble honors the raw-log restraint criteria: entries timestamped, expiry metadata present, origin refs navigable, injection is indices/refs not dumps. Verify: asserted by the 37t.4 preamble test.

## Static mechanism / likely defect

Issue description localizes the mechanism: The judged-memory loop: agents declare structured claims, the operator judges them, active claims compile into context preambles, and sessions reboot into compact evidence packs. Substrate exists (assertions, compile_context, compose_context_preamble, SessionStart hook); these children wire the loop closed. Raw-log design criteria (2026-06-29): entries timestamped, expiry metadata, navigable origin refs, restrained injection with expandable indices/refs. Design direction: Epic spine: the loop closes when a declared claim travels end-to-end through all four named stages against the live archive. Stage owners: CLAIMS = 37t.2 (author-declared markers -> candidate assertions) + 37t.1 (assertion consumer wiring); JUDGMENT = operator bulk review/accept/reject of candidate assertions (currently unowned by any child — needs a judgment-queue bead); PREAMBLE = 37t.4 (SessionStart rollout) refa…

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

1. Epic spine: the loop closes when a declared claim travels end-to-end through all four named stages against the live archive.
2. Stage owners: CLAIMS = 37t.2 (author-declared markers -> candidate assertions) + 37t.1 (assertion consumer wiring)
3. JUDGMENT = operator bulk review/accept/reject of candidate assertions (currently unowned by any child — needs a judgment-queue bead)
4. PREAMBLE = 37t.4 (SessionStart rollout) refactored onto 37t.11 (ContextSource scheduler/arbiter)
5. REBOOT = 37t.3 (reboot-with-refs).
6. 37t.11 is the coherence spine every preamble/recall source registers against.
7. Each named stage must map to an open child with non-null acceptance

## Tests to add

- Acceptance proof: A seeded end-to-end scenario test demonstrates one claim flowing claims->judgment->preamble->reboot: an agent emits a declared marker (37t.2) that lands as a candidate assertion, the operator accepts it via the judgment queue, it appears as a ref in a compiled SessionStart preamble for the matching repo (37t.4/37t.11), and it survives a reboot-with-refs handoff (37t.3) resolvable via resolve_ref.
- Acceptance proof: Verify: a pytest covering the four-stage path (e.g.
- Acceptance proof: tests/unit/context/test_judged_memory_loop.py) plus MCP resolve_ref on the emitted ref.
- Acceptance proof: Every named stage (claims/judgment/preamble/reboot) has an owning open bead with non-null acceptance_criteria
- Acceptance proof: no stage survives only as prose in a sibling's design field.
- Acceptance proof: Verify: `bd show` on each child confirms acceptance present.
- Acceptance proof: The compiled preamble honors the raw-log restraint criteria: entries timestamped, expiry metadata present, origin refs navigable, injection is indices/refs not dumps.
- Acceptance proof: Verify: asserted by the 37t.4 preamble test.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.
- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
