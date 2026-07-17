# 165. polylogue-37t.1 — Assertions: consumer wiring + lifecycle tightening for unified overlays

Priority/type/status: **P2 / feature / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Assertion substrate is the live path; remaining work is consumer wiring + lifecycle (promotion, staleness, expiry). Unwired kinds exist: handoff, prompt_eval, highlight. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

Wiring points (verify current state first): unwired AssertionKinds handoff/prompt_eval/highlight need first writers + surface registration (user_audit every-kind-has-a-surface invariant will force the surface entry; scope_ref/author_ref must use a registered ObjectRef kind — see #2383 memory). Lifecycle: add staleness/expiry semantics to claims consumed by the preamble compiler (ASSERTION_CLAIM_KINDS reads ACTIVE only — extend with expiry check rather than a new status); judgment surfaces: list/accept/reject exist via MCP assertion tools — the gap is operator-ergonomic review flow (bulk judge candidate batches). Raw-log criteria: timestamped entries, expiry metadata, navigable origin refs.

## Acceptance criteria

- First writers exist for the three currently-unwired AssertionKinds (handoff, prompt_eval, highlight — present-but-writerless at core/enums.py:409/423/426) and each passes the user_audit every-kind-has-a-surface invariant with a registered ObjectRef scope_ref/author_ref. Verify: the user_audit surface reports zero surfaceless kinds; pytest asserts a written row per kind.
- Preamble-consumed claims gain expiry: the ASSERTION_CLAIM_KINDS admission read (user_write.py:~1520) extends its ACTIVE-only filter with an expiry check (no new status). Test: an expired claim is excluded from the preamble compiler input.
- Content-hash invariant: recording or expiring a claim never mutates sessions.content_hash. Verify: pytest mirroring tests/unit/insights/test_feedback.py.
- Operator judgment ergonomics: a bulk accept/reject-over-a-candidate-batch flow is demonstrated, or the scope is explicitly split to the judgment-queue child bead and referenced from here.

## Static mechanism / likely defect

Issue description localizes the mechanism: Assertion substrate is the live path; remaining work is consumer wiring + lifecycle (promotion, staleness, expiry). Unwired kinds exist: handoff, prompt_eval, highlight. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Wiring points (verify current state first): unwired AssertionKinds handoff/prompt_eval/highlight need first writers + surface registration (user_audit every-kind-has-a-surface invariant will force the surface entry; scope_ref/author_ref must use a registered ObjectRef kind — see #2383 memory). Lifecycle: add staleness/expiry semantics to claims consumed by the preamble compiler (ASSERTION_CLAIM_KINDS reads ACTIVE on…

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

1. Wiring points (verify current state first): unwired AssertionKinds handoff/prompt_eval/highlight need first writers + surface registration (user_audit every-kind-has-a-surface invariant will force the surface entry
2. scope_ref/author_ref must use a registered ObjectRef kind — see #2383 memory).
3. Lifecycle: add staleness/expiry semantics to claims consumed by the preamble compiler (ASSERTION_CLAIM_KINDS reads ACTIVE only — extend with expiry check rather than a new status)
4. judgment surfaces: list/accept/reject exist via MCP assertion tools — the gap is operator-ergonomic review flow (bulk judge candidate batches).
5. Raw-log criteria: timestamped entries, expiry metadata, navigable origin refs.

## Tests to add

- Acceptance proof: First writers exist for the three currently-unwired AssertionKinds (handoff, prompt_eval, highlight — present-but-writerless at core/enums.py:409/423/426) and each passes the user_audit every-kind-has-a-surface invariant with a registered ObjectRef scope_ref/author_ref.
- Acceptance proof: Verify: the user_audit surface reports zero surfaceless kinds
- Acceptance proof: pytest asserts a written row per kind.
- Acceptance proof: Preamble-consumed claims gain expiry: the ASSERTION_CLAIM_KINDS admission read (user_write.py:~1520) extends its ACTIVE-only filter with an expiry check (no new status).
- Acceptance proof: Test: an expired claim is excluded from the preamble compiler input.
- Acceptance proof: Content-hash invariant: recording or expiring a claim never mutates sessions.content_hash.
- Acceptance proof: Verify: pytest mirroring tests/unit/insights/test_feedback.py.
- Acceptance proof: Operator judgment ergonomics: a bulk accept/reject-over-a-candidate-batch flow is demonstrated, or the scope is explicitly split to the judgment-queue child bead and referenced from here.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
