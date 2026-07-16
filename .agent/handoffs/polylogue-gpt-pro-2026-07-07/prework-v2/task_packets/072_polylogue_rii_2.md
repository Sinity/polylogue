# 072. polylogue-rii.2 — Materialize hook events + OTLP spans into queryable evidence

Priority/type/status: **P2 / feature / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **blocked-hard**.

Hard blockers: polylogue-rii.1

## What the bead says

Hook events are captured as raw blobs but never materialized (~95% of hook-only signal invisible: tool annotations, pre-MCP output, permission decisions, cwd changes, subagent lifecycle); OTLP spans likewise. Both converge on the write-leg contract. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

Code-confirmed gaps (gh#2461, re-locate lines): archive/artifact_taxonomy/runtime.py:~209 classifies HOOK_EVENT with parse_as_session=False (stored as raw blobs, used only for paste enrichment); artifact_taxonomy/support.py:~82 looks_like_hook_event hardcodes provider in ('claude-code','codex'). Fix: materialize hook events through the write-leg contract into session_events/ObservedEvents keyed to the owning session (session id is in the hook payload); un-hardcode the provider check via the taxonomy. OTLP: spans already land in ops.db via the receiver — project them into queryable evidence the same way rather than a second reader. ~95% of hook-only signal (tool annotations, pre-MCP output, permission decisions, cwd changes, subagent lifecycle) becomes visible.

## Acceptance criteria gap

This active bead lacks acceptance criteria in the export. Add checkable acceptance criteria before coding unless this packet explicitly supplies a temporary gate.

## Static mechanism / likely defect

Issue description localizes the mechanism: Hook events are captured as raw blobs but never materialized (~95% of hook-only signal invisible: tool annotations, pre-MCP output, permission decisions, cwd changes, subagent lifecycle); OTLP spans likewise. Both converge on the write-leg contract. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Code-confirmed gaps (gh#2461, re-locate lines): archive/artifact_taxonomy/runtime.py:~209 classifies HOOK_EVENT with parse_as_session=False (stored as raw blobs, used only for paste enrichment); artifact_taxonomy/support.py:~82 looks_like_hook_event hardcodes provider in ('claude-code','codex'). Fix: materialize hook events through the write-leg contract into session_events/ObservedEvents keyed to the owning session…

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

1. Code-confirmed gaps (gh#2461, re-locate lines): archive/artifact_taxonomy/runtime.py:~209 classifies HOOK_EVENT with parse_as_session=False (stored as raw blobs, used only for paste enrichment)
2. artifact_taxonomy/support.py:~82 looks_like_hook_event hardcodes provider in ('claude-code','codex').
3. Fix: materialize hook events through the write-leg contract into session_events/ObservedEvents keyed to the owning session (session id is in the hook payload)
4. un-hardcode the provider check via the taxonomy.
5. OTLP: spans already land in ops.db via the receiver — project them into queryable evidence the same way rather than a second reader.
6. ~95% of hook-only signal (tool annotations, pre-MCP output, permission decisions, cwd changes, subagent lifecycle) becomes visible.

## Tests to add

- Candidate-write safety test.
- Scheduler context-ledger determinism test.
- Simulated two-agent coordination envelope test.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
