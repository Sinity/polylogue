# 173. polylogue-37t.4 — SessionStart preamble opt-in rollout (polylogue + sinnix repos)

Priority/type/status: **P2 / task / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **blocked-hard**.

Hard blockers: polylogue-37t.12

## What the bead says

Wire compose_context_preamble into SessionStart hooks per-project via .claude/settings.json, polylogue + sinnix first (operator decision). Preamble presence is arm B of the uplift experiment; rollout and experiment reinforce each other. Restrained injection — indices/refs over dumps (raw-log criterion).

## Existing design note

Wire compose_context_preamble into SessionStart hooks per-project via .claude/settings.json, polylogue + sinnix first. Session-start mechanics (2026-07-03 research): (1) SOURCE-AWARE — the SessionStart payload carries source (startup|resume|clear|compact): fresh startup gets the repo brief (active beads pointer, last session outcome, judged lessons for this repo); resume gets a delta-since-last-session (new sessions/commits/beads touching this repo since the resumed session's end); compact gets NOTHING from polylogue (bd prime already reinjects task state; double-injection burns budget — jgp restraint). (2) RELEVANCE GATING — inject only when the cwd maps to a repo with archive history; otherwise stay silent. (3) BUDGET — hard token cap in the hook (start ~600 tokens), indices/refs over content, every line resolvable via resolve_ref. (4) FRESHNESS — cache the compiled preamble keyed by (repo, archive cursor); regenerate only when the cursor moved (the uplift pilot's staleness lesson, yps metadata is the input). (5) ESCAPE HATCH — POLYLOGUE_PREAMBLE=off env kills injection without editing settings. (6) MEASUREMENT — the hook logs its own injection as a hook event so preamble presence/size/latency is queryable; this is arm instrumentation for the uplift re-run (cfk). Current local state: sessionstart-polylogue-recall.sh ships session LISTS (recall), not the compiled preamble — this bead upgrades that hook, it does not add a second one.

## Acceptance criteria

- compose_context_preamble is wired into the existing SessionStart hook (upgrading sessionstart-polylogue-recall.sh, not adding a second hook) for polylogue + sinnix. Verify: hook fires and injects on `polylogue` SessionStart in each repo; pytest asserts the single-hook path.
- Source-aware branching verified by test: source=startup injects the repo brief, source=resume injects a since-last-session delta, source=compact injects zero polylogue bytes.
- Hard token cap enforced (default ~600): a test that an oversized brief degrades to refs rather than exceeding the cap.
- Relevance gate: a cwd with no archive history produces zero injection (test).
- Escape hatch: POLYLOGUE_PREAMBLE=off suppresses injection without editing settings (test asserts no bytes emitted).
- Instrumentation (arm B for cfk): the hook logs its own injection as a queryable hook event carrying presence/size/latency.
- Freshness cache keyed by (repo, archive cursor) regenerates only when the cursor moved (test asserts no regeneration on an unchanged cursor).

## Static mechanism / likely defect

Issue description localizes the mechanism: Wire compose_context_preamble into SessionStart hooks per-project via .claude/settings.json, polylogue + sinnix first (operator decision). Preamble presence is arm B of the uplift experiment; rollout and experiment reinforce each other. Restrained injection — indices/refs over dumps (raw-log criterion). Design direction: Wire compose_context_preamble into SessionStart hooks per-project via .claude/settings.json, polylogue + sinnix first. Session-start mechanics (2026-07-03 research): (1) SOURCE-AWARE — the SessionStart payload carries source (startup|resume|clear|compact): fresh startup gets the repo brief (active beads pointer, last session outcome, judged lessons for this repo); resume gets a delta-since-last-session (new sessions…

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

1. Wire compose_context_preamble into SessionStart hooks per-project via .claude/settings.json, polylogue + sinnix first.
2. Session-start mechanics (2026-07-03 research): (1) SOURCE-AWARE — the SessionStart payload carries source (startup|resume|clear|compact): fresh startup gets the repo brief (active beads pointer, last session outcome, judged lessons for this repo)
3. resume gets a delta-since-last-session (new sessions/commits/beads touching this repo since the resumed session's end)
4. compact gets NOTHING from polylogue (bd prime already reinjects task state
5. double-injection burns budget — jgp restraint).
6. (2) RELEVANCE GATING — inject only when the cwd maps to a repo with archive history
7. otherwise stay silent.

## Tests to add

- Acceptance proof: compose_context_preamble is wired into the existing SessionStart hook (upgrading sessionstart-polylogue-recall.sh, not adding a second hook) for polylogue + sinnix.
- Acceptance proof: Verify: hook fires and injects on `polylogue` SessionStart in each repo
- Acceptance proof: pytest asserts the single-hook path.
- Acceptance proof: Source-aware branching verified by test: source=startup injects the repo brief, source=resume injects a since-last-session delta, source=compact injects zero polylogue bytes.
- Acceptance proof: Hard token cap enforced (default ~600): a test that an oversized brief degrades to refs rather than exceeding the cap.
- Acceptance proof: Relevance gate: a cwd with no archive history produces zero injection (test).
- Acceptance proof: Escape hatch: POLYLOGUE_PREAMBLE=off suppresses injection without editing settings (test asserts no bytes emitted).
- Acceptance proof: Instrumentation (arm B for cfk): the hook logs its own injection as a queryable hook event carrying presence/size/latency.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
