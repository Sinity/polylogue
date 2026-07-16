# 166. polylogue-37t.2 — Inline annotation protocol: agent-authored structure in plain prose

Priority/type/status: **P2 / feature / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Agents write structured markers in prose; extraction at block enrichment turns them into candidate assertions with evidence refs. Author-declared notation, not heuristic mining — the construct-validity-safe way to get structure out of prose.

## Existing design note

PROTOCOL DESIGN (2026-07-03, generalizing the marker idea into a composable protocol): (1) SYNTAX: line-anchored sigil markers — '::kind(args): body' on its own line, inline '[[kind: body]]' for short spans. Chosen for: harness-agnostic (plain text works in ANY provider incl. web chats), streaming-safe (line-complete before parse), markdown-inert (harmless where uninterpreted), collision-resistant (escape via '\::'). Decide the final sigil after a corpus collision scan — grep the live archive for candidate-prefix false positives; evidence over taste. (2) KIND REGISTRY (declare-once, o21): each kind declares payload schema, extraction target, lifecycle — note/claim/lesson/decision -> candidate assertions with evidence ref = containing message; predict(p, horizon, resolver) -> prediction ledger (calibration bead); handoff(...) -> reboot/handoff hints (37t.3); anchor(name) -> named refs other markers cite; bead(title, prio?) -> candidate bead in the discovered-work flow (4c0); eval(score, dim) -> self-assessment rows (37t.9's PROMPT_EVAL becomes one kind). New kinds are registry entries, not parser changes. (3) COMPOSABILITY: markers carry refs (session:/message:/assertion:/bead ids) linking structure across the corpus; scoping via anchor + explicit ref, NOT syntactic nesting — the grammar stays line-local and trivial; complexity lives in the registry. (4) EXTRACTION at block enrichment (structural): each marker -> typed row with exact message/block provenance; malformed markers extract as kind=malformed with raw text (never silently dropped — agents learn from feedback, and malformed rate is itself a quality measure). (5) ADOPTION LOOP: spec ships as an agent skill + preamble one-liner (pj8/37t.4); adoption rate and kind distribution are 9l5.7 measures; the experiment machinery (stc) can A/B protocol-on vs off. (6) CONSTRUCT VALIDITY: author-declared structure is the honest tier between raw prose and tool calls — extraction is exact (no NLP), authorship explicit, and the tier label 'agent-declared' distinguishes it from 'structural' outcomes in every downstream measure; agents can be wrong or game it — calibration closes that loop.

## Acceptance criteria

- Final sigil chosen after a corpus collision scan: grep the live archive for candidate-prefix false positives and record the scan result in the PR.
- Line-anchored '::kind(args): body' and inline '[[kind: body]]' parse at block enrichment into typed rows with exact message/block provenance; malformed markers extract as kind=malformed with raw text (never silently dropped) and the malformed rate is a recorded measure. Verify: pytest over fixtures covering well-formed, malformed, markdown-inert, streaming-split, and '\::' escaped inputs.
- Kind registry is declare-once: adding a kind (note/claim/lesson/decision/predict/handoff/anchor/bead/eval) is a registry entry, not a parser change. Verify: a structure/property test asserts a new kind touches only the registry module.
- Extracted candidates carry an evidence ref to the containing message and land as candidate-status assertions (not active). Verify: pytest asserts status and ref on an extracted marker.

## Static mechanism / likely defect

Issue description localizes the mechanism: Agents write structured markers in prose; extraction at block enrichment turns them into candidate assertions with evidence refs. Author-declared notation, not heuristic mining — the construct-validity-safe way to get structure out of prose. Design direction: PROTOCOL DESIGN (2026-07-03, generalizing the marker idea into a composable protocol): (1) SYNTAX: line-anchored sigil markers — '::kind(args): body' on its own line, inline '[[kind: body]]' for short spans. Chosen for: harness-agnostic (plain text works in ANY provider incl. web chats), streaming-safe (line-complete before parse), markdown-inert (harmless where uninterpreted), collision-resistant (escape via '\::')…

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

1. PROTOCOL DESIGN (2026-07-03, generalizing the marker idea into a composable protocol): (1) SYNTAX: line-anchored sigil markers — '::kind(args): body' on its own line, inline '[[kind: body]]' for short spans.
2. Chosen for: harness-agnostic (plain text works in ANY provider incl.
3. web chats), streaming-safe (line-complete before parse), markdown-inert (harmless where uninterpreted), collision-resistant (escape via '\::').
4. Decide the final sigil after a corpus collision scan — grep the live archive for candidate-prefix false positives
5. evidence over taste.
6. (2) KIND REGISTRY (declare-once, o21): each kind declares payload schema, extraction target, lifecycle — note/claim/lesson/decision -> candidate assertions with evidence ref = containing message
7. predict(p, horizon, resolver) -> prediction ledger (calibration bead)

## Tests to add

- Acceptance proof: Final sigil chosen after a corpus collision scan: grep the live archive for candidate-prefix false positives and record the scan result in the PR.
- Acceptance proof: Line-anchored '::kind(args): body' and inline '[[kind: body]]' parse at block enrichment into typed rows with exact message/block provenance
- Acceptance proof: malformed markers extract as kind=malformed with raw text (never silently dropped) and the malformed rate is a recorded measure.
- Acceptance proof: Verify: pytest over fixtures covering well-formed, malformed, markdown-inert, streaming-split, and '\::' escaped inputs.
- Acceptance proof: Kind registry is declare-once: adding a kind (note/claim/lesson/decision/predict/handoff/anchor/bead/eval) is a registry entry, not a parser change.
- Acceptance proof: Verify: a structure/property test asserts a new kind touches only the registry module.
- Acceptance proof: Extracted candidates carry an evidence ref to the containing message and land as candidate-status assertions (not active).
- Acceptance proof: Verify: pytest asserts status and ref on an extracted marker.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
