# 062. polylogue-37t.11 — Context scheduler: one arbiter for everything that enters an agent's context

Priority/type/status: **P1 / feature / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **blocked-hard**.

Hard blockers: polylogue-37t.12, polylogue-37t.15

## What the bead says

The missing 40% of the OS-vision design, and the coherence fix for a real fragmentation risk: as of today SEVEN independent mechanisms want to write into agent context, each with its own budget rules — repo brief + resume delta (37t.4), semantic recall (mhx.4), SRS-due lessons (rvh), blackboard messages (1hj), PreToolUse/prompt advisories (bfv), compaction re-grounding (gjg), and the affordance-index pointer (pj8). Built separately they will fight for the same tokens, double-inject, and be untunable as a whole. The OS analogy names the fix: injection sources are processes, context budget is memory, and there must be ONE scheduler that allocates it — plus a LEDGER recording every allocation so 'what was in this session's context and why' is a queryable fact rather than a reconstruction.

## Existing design note

(1) SOURCE PROTOCOL: each injector registers (declare-once) as a ContextSource with: moment (session-start | pre-compact-resume | mid-session-advisory | on-demand), priority class (correctness > directives > recall > ambient), a propose() returning candidate items (each: content-or-ref, token cost, relevance score, source ref, expiry), and a degrade order (full -> ref-only -> drop). Existing beads become sources, not owners of budgets: 37t.4's sections, mhx.4's recall hits, rvh's due lessons, 1hj's messages, gjg's re-grounding items. (2) SCHEDULER: per moment, allocate the scoped budget (y4c per-repo/per-moment budgets) across classes by fixed proportions with borrowing (unused directive budget flows to recall), then within class by score; produce the final assembly deterministically (same inputs -> same context, testable). Hard invariants: never exceed moment budget; every included item carries its resolve_ref; refs-over-bodies below a per-item threshold (jgp). (3) LEDGER: every allocation decision (included/degraded/dropped, with scores and budget state) written as a context-injection event keyed to the target session — the session's 'memory map'. Read surfaces: 'polylogue context ledger <session>' + a webui panel on the session Info tab; the compaction-loss forensics (gjg) and uplift instrumentation read it (arm evidence = ledger rows). (4) MID-SESSION moments share the same arbiter with tiny budgets (advisory = one item) so cooldown/dedup state is global — an advisory about X suppresses a blackboard item about X (cross-source dedup by content-hash/ref). (5) IMPLEMENTATION HOME: extends context/compiler.py (compile_context already does budgeted segment assembly — the scheduler generalizes its admission logic and hoists it above the preamble composer); the SessionStart hook calls ONE entrypoint. (6) Sequencing: lands BEFORE mhx.4/rvh/1hj/bfv wire their injection legs — each of those beads' design now says 'register as a ContextSource' (notes added); 37t.4 ships first as the initial two sources and gets refactored onto the protocol here.

## Acceptance criteria

ContextSource protocol + scheduler in context/compiler.py with deterministic assembly (property test: same inputs -> byte-identical context); 37t.4's sections migrated as the first two sources; budget invariants enforced (property test: never exceeds moment budget at any source combination); ledger rows written per injection and readable via CLI + MCP; cross-source dedup demonstrated (advisory suppresses same-ref blackboard item in a seeded scenario).

## Static mechanism / likely defect

Issue description localizes the mechanism: The missing 40% of the OS-vision design, and the coherence fix for a real fragmentation risk: as of today SEVEN independent mechanisms want to write into agent context, each with its own budget rules — repo brief + resume delta (37t.4), semantic recall (mhx.4), SRS-due lessons (rvh), blackboard messages (1hj), PreToolUse/prompt advisories (bfv), compaction re-grounding (gjg), and the affordance-index pointer (pj8). Built separately they will fight for the same tokens, double-inject, and be untunable as a whole. Th… Design direction: (1) SOURCE PROTOCOL: each injector registers (declare-once) as a ContextSource with: moment (session-start | pre-compact-resume | mid-session-advisory | on-demand), priority class (correctness > directives > recall > ambient), a propose() returning candidate items (each: content-or-ref, token cost, relevance score, source ref, expiry), and a degrade order (full -> ref-only -> drop). Existing beads become sources, no…

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

1. (1) SOURCE PROTOCOL: each injector registers (declare-once) as a ContextSource with: moment (session-start | pre-compact-resume | mid-session-advisory | on-demand), priority class (correctness > directives > recall > ambient), a propose() returning candidate items (each: content-or-ref, token cost, relevance score, source ref, expiry), and a degrade order (full -> ref-only -> drop).
2. Existing beads become sources, not owners of budgets: 37t.4's sections, mhx.4's recall hits, rvh's due lessons, 1hj's messages, gjg's re-grounding items.
3. (2) SCHEDULER: per moment, allocate the scoped budget (y4c per-repo/per-moment budgets) across classes by fixed proportions with borrowing (unused directive budget flows to recall), then within class by score
4. produce the final assembly deterministically (same inputs -> same context, testable).
5. Hard invariants: never exceed moment budget
6. every included item carries its resolve_ref
7. refs-over-bodies below a per-item threshold (jgp).

## Tests to add

- Acceptance proof: ContextSource protocol + scheduler in context/compiler.py with deterministic assembly (property test: same inputs -> byte-identical context)
- Acceptance proof: 37t.4's sections migrated as the first two sources
- Acceptance proof: budget invariants enforced (property test: never exceeds moment budget at any source combination)
- Acceptance proof: ledger rows written per injection and readable via CLI + MCP
- Acceptance proof: cross-source dedup demonstrated (advisory suppresses same-ref blackboard item in a seeded scenario).

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
