# 168. polylogue-37t.6 — Session-aware devshell entry: surface what the last agent session left behind

Priority/type/status: **P2 / feature / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

On cd/direnv entry, print what the last agent session in this cwd left: unresolved blackboard blocker/question notes, the last session's terminal state, resume candidates for this directory. All reads exist (blackboard_list with unresolved filter; find_resume_candidates already scores cwd at 0.15 weight) — this is a status-line/devshell-hook integration away. Keep it one bounded line + an expand command; restrained-injection rule applies.

## Existing design note

On cd/direnv entry, print one bounded line summarizing what the last agent session in this cwd left behind: unresolved blackboard blocker/question notes (blackboard_list unresolved filter), the last session's terminal state, and resume candidates for this directory (find_resume_candidates, which already scores cwd at 0.15 weight). All reads exist; this is a devshell-hook / status-line integration. Keep it one bounded line plus an expand command and apply the restrained-injection rule (no noisy dumps; suppress when there is nothing to report).

## Acceptance criteria

1. A devshell/direnv entry hook prints a single bounded line for the current cwd combining unresolved blackboard-note count, the last session's terminal state, and the top resume candidate(s), using existing reads (blackboard_list unresolved filter, find_resume_candidates) with no new query machinery. 2. An expand command shows the full detail; the entry line stays one line and suppresses itself when there is nothing to report (restrained injection). Verify: run the hook in a cwd with a known last session plus an unresolved blackboard note and confirm the summary line and expand output; `devtools test` selection on the integration helper.

## Static mechanism / likely defect

Issue description localizes the mechanism: On cd/direnv entry, print what the last agent session in this cwd left: unresolved blackboard blocker/question notes, the last session's terminal state, resume candidates for this directory. All reads exist (blackboard_list with unresolved filter; find_resume_candidates already scores cwd at 0.15 weight) — this is a status-line/devshell-hook integration away. Keep it one bounded line + an expand command; restrained-injection rule applies. Design direction: On cd/direnv entry, print one bounded line summarizing what the last agent session in this cwd left behind: unresolved blackboard blocker/question notes (blackboard_list unresolved filter), the last session's terminal state, and resume candidates for this directory (find_resume_candidates, which already scores cwd at 0.15 weight). All reads exist; this is a devshell-hook / status-line integration. Keep it one bounde…

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

1. On cd/direnv entry, print one bounded line summarizing what the last agent session in this cwd left behind: unresolved blackboard blocker/question notes (blackboard_list unresolved filter), the last session's terminal state, and resume candidates for this directory (find_resume_candidates, which already scores cwd at 0.15 weight).
2. All reads exist
3. this is a devshell-hook / status-line integration.
4. Keep it one bounded line plus an expand command and apply the restrained-injection rule (no noisy dumps
5. suppress when there is nothing to report).

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: A devshell/direnv entry hook prints a single bounded line for the current cwd combining unresolved blackboard-note count, the last session's terminal state, and the top resume candidate(s), using existing reads (blackboard_list unresolved filter, find_resume_candidates) with no new query machinery.
- Acceptance proof: 2.
- Acceptance proof: An expand command shows the full detail
- Acceptance proof: the entry line stays one line and suppresses itself when there is nothing to report (restrained injection).
- Acceptance proof: Verify: run the hook in a cwd with a known last session plus an unresolved blackboard note and confirm the summary line and expand output
- Acceptance proof: `devtools test` selection on the integration helper.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
