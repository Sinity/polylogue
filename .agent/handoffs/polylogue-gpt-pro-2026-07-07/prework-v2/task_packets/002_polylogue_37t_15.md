# 002. polylogue-37t.15 — Single agent-write chokepoint in upsert_assertion: non-user authors => CANDIDATE + inject:false, always

Priority/type/status: **P1 / task / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Known live hole (R&D-confirmed): blackboard_post lets an agent write author_kind=agent rows that land status=ACTIVE — an agent claim can self-inject as authoritative TODAY. Fix at the ONE chokepoint, not per-path: inside upsert_assertion, coerce ALL non-user authors to CANDIDATE + inject:false + promotion_required, never resurrecting a terminal-judged row. Every writer (transform/pathology/goal/decision/recipe/distillery/recall/blackboard/annotation-import) inherits the invariant; enforcing per-recipe provably leaves holes (blackboard was the counterexample). This is the QUOTED->OPERATOR promotion gate that 37t.11 injection-security depends on, and it is frontier-executable NOW independent of the verdict substrate.

## Existing design note

coerce_agent_authored(assertion) applied inside upsert_assertion (both storage twins — sync archive_tiers AND async mixins, per the twins trap); terminal-judged detection via existing judgment rows; deterministic-detector carve-out only via an explicit allowlist argument, not author_kind sniffing. Regression test recreates the blackboard_post ACTIVE hole.

## Acceptance criteria

blackboard_post as agent lands candidate+inject:false; a rejected candidate re-upserted by an agent stays rejected; user-authored writes unaffected; both storage paths covered. Verify: focused user_write + blackboard tests.

## Static mechanism / likely defect

`upsert_assertion` defaults missing status to ACTIVE and `upsert_blackboard_note` does not override it, so non-user blackboard/MCP writes can become trusted active assertions by omission.

## Source anchors to inspect first

- `polylogue/storage/sqlite/archive_tiers/user_write.py:31` — ASSERTION_DEFAULT_STATUS is ACTIVE, so missing status currently means trusted active.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:641` — upsert_blackboard_note passes author_kind and no explicit status into upsert_assertion.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — upsert_assertion is the single write chokepoint to patch.
- `polylogue/api/contracts/assertions.py` — Check public assertion request/response contract after changing default status behavior.

## Implementation plan

1. Patch `upsert_assertion` as the only safety chokepoint. Normalize `author_kind`; only exact `user` may default to active/injected.
2. For any non-user author, coerce missing or active status to candidate and set context/injection flags false unless a reviewed user action promotes it.
3. Fetch existing assertion status before upsert and forbid agent writes from resurrecting rejected/deleted/superseded terminal states.
4. Make all blackboard/MCP/API call paths rely on this same function; remove caller-specific safety guesses.

## Tests to add

- Unit test agent blackboard post without status: row is candidate, inject false, provenance preserved.
- Unit test user write without status: keeps existing user-default active semantics if intended.
- Regression: rejected candidate cannot be overwritten by an agent into active.
- MCP/API write tests prove non-user authors are demoted even when they request active.

## Verification commands

- ``devtools test tests/unit/storage/test_user_state_contracts.py tests/unit/storage/test_archive_tiers_assertions.py -k 'assertion or blackboard or candidate'``

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
