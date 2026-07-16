# 074. polylogue-4c0 — Beads-native work loop: session<->bead cross-links and archive-rendered work history

Priority/type/status: **P2 / task / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **blocked-hard**.

Hard blockers: polylogue-7fj

## What the bead says

Beads and the archive already observe the same work from two sides but never join: a bead's history (claims, closes, reasons) names no sessions; a session's transcript contains bd commands the archive does not structurally extract. Joining them makes both better: bd show could point at the sessions that did the work (with the postmortem one hop away); polylogue could render a bead's full work history (every session that touched it, what changed — yrx — what it cost, what failed); close reasons become claims checkable against session evidence (the lnd doctrine's claim-vs-evidence seam, made real); and the devloop's next-action choice can weigh 'this bead already burned 3 sessions and $4 without closing' — cost-aware scheduling.

## Existing design note

(1) EXTRACTION: bd invocations are shell tool calls with structured output — a block enricher recognizes bd claim/close/create/update and materializes session<->bead edge rows (bead id, verb, timestamp, session ref); zero heuristics, the commands are structural. (2) READ SURFACES: 'polylogue bead <id>' (or a DSL unit: beads where id:X | sessions) renders the work history envelope: sessions, durations, cost, changes summary, close reason vs evidence; MCP twin for agents. (3) BEADS SIDE: a bd-side pointer needs no bd fork — the devloop convention writes the session ref into close reasons/notes automatically via a Stop-hook helper (the hook knows the session id and the claimed bead). (4) VERIFICATION SEAM: close-reason claims cross-checked against the linked sessions' structural evidence (tests actually ran? files actually changed?) — a claim-vs-evidence variant scoped to bead closures; surfaces as an audit measure, not a gate. (5) 7fj (beads-history ingestion) is the substrate dependency: issues.jsonl + Dolt history land as an evidence source; this bead builds the join + surfaces on top.

## Acceptance criteria

On the live archive: session<->bead edges materialize for the recent devloop sessions; the bead work-history envelope renders for a real closed bead with sessions, cost, and changes; a close-reason cross-check runs for one campaign bead and reports agreement; Stop-hook writes the session ref into bd notes on claim/close.

## Static mechanism / likely defect

Issue description localizes the mechanism: Beads and the archive already observe the same work from two sides but never join: a bead's history (claims, closes, reasons) names no sessions; a session's transcript contains bd commands the archive does not structurally extract. Joining them makes both better: bd show could point at the sessions that did the work (with the postmortem one hop away); polylogue could render a bead's full work history (every session that touched it, what changed — yrx — what it cost, what failed); close reasons become claims checka… Design direction: (1) EXTRACTION: bd invocations are shell tool calls with structured output — a block enricher recognizes bd claim/close/create/update and materializes session<->bead edge rows (bead id, verb, timestamp, session ref); zero heuristics, the commands are structural. (2) READ SURFACES: 'polylogue bead <id>' (or a DSL unit: beads where id:X | sessions) renders the work history envelope: sessions, durations, cost, changes …

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

1. (1) EXTRACTION: bd invocations are shell tool calls with structured output — a block enricher recognizes bd claim/close/create/update and materializes session<->bead edge rows (bead id, verb, timestamp, session ref)
2. zero heuristics, the commands are structural.
3. (2) READ SURFACES: 'polylogue bead <id>' (or a DSL unit: beads where id:X | sessions) renders the work history envelope: sessions, durations, cost, changes summary, close reason vs evidence
4. MCP twin for agents.
5. (3) BEADS SIDE: a bd-side pointer needs no bd fork — the devloop convention writes the session ref into close reasons/notes automatically via a Stop-hook helper (the hook knows the session id and the claimed bead).
6. (4) VERIFICATION SEAM: close-reason claims cross-checked against the linked sessions' structural evidence (tests actually ran? files actually changed?) — a claim-vs-evidence variant scoped to bead closures
7. surfaces as an audit measure, not a gate.

## Tests to add

- Acceptance proof: On the live archive: session<->bead edges materialize for the recent devloop sessions
- Acceptance proof: the bead work-history envelope renders for a real closed bead with sessions, cost, and changes
- Acceptance proof: a close-reason cross-check runs for one campaign bead and reports agreement
- Acceptance proof: Stop-hook writes the session ref into bd notes on claim/close.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
