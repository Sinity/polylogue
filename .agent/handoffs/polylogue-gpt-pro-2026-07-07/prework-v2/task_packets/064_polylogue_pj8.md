# 064. polylogue-pj8 — Agent query cookbook: MCP prompts + skill recipes as the discoverability layer

Priority/type/status: **P1 / feature / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Agents use what is in their face and skip what requires invention (jgp doctrine). The MCP server exposes ~61 read tools; nothing teaches an agent WHICH five matter for the common intents: 'what was I doing in this repo', 'postmortem the last failed session', 'what did we decide about X', 'what failed recently and was never acknowledged', 'find the session where we touched file Y'. server_prompts.py exists but the prompt surface is thin, and there is no harness-side skill teaching Polylogue idioms the way the beads skill teaches bd.

## Existing design note

Three thin layers over existing capability, no new query machinery: (1) MCP prompts: register ~6 intent-named prompts (resume-context, postmortem-last, decisions-about, unacknowledged-failures, sessions-touching-file, cost-of) that expand to the right tool-call sequences with cwd/repo prefilled — prompts are the MCP-native discoverability channel. (2) A 'polylogue' harness skill (dots/claude/skills + codex overlay, sinnix-side) with the same recipes in agent-readable form plus the two rules agents get wrong (archive root env var; refs over dumps). (3) The SessionStart preamble (37t.4) ends with a one-line affordance index pointing at those prompts — injection makes the surface ambient. Acceptance: affordance-usage report shows tool diversity rising in agent sessions (baseline: today's usage is dominated by search/get_session). Keep total prompt count small — the cookbook is a curation, not another catalog.

## Acceptance criteria

- ~6 intent-named MCP prompts are registered (resume-context, postmortem-last, decisions-about, unacknowledged-failures, sessions-touching-file, cost-of) that expand to the correct tool-call sequences with cwd/repo prefilled; total prompt count stays small (curation, not another catalog).
- The prompt set includes the coordination intents over the shared envelope (agent_status, agent_self, work_item/current packet, coordination_hazards, addressed_messages, handoff) per the s7ae coordination update.
- A `polylogue` harness skill (dots/claude/skills + codex overlay, sinnix-side) carries the same recipes plus the two rules agents get wrong (archive-root env var; refs over dumps).
- The SessionStart preamble (37t.4) ends with a one-line affordance index pointing at those prompts.
- MCP prompt/tool code and generated contracts are complete BEFORE any deployment batch (EXPECTED_TOOL_NAMES / prompt registry + `devtools render openapi` + `render all --check` clean); if only deployment remains it is recorded explicitly on the bead.
- `devtools workspace affordance-usage` shows tool diversity rising versus the search/get_session-dominated baseline.

## Static mechanism / likely defect

Issue description localizes the mechanism: Agents use what is in their face and skip what requires invention (jgp doctrine). The MCP server exposes ~61 read tools; nothing teaches an agent WHICH five matter for the common intents: 'what was I doing in this repo', 'postmortem the last failed session', 'what did we decide about X', 'what failed recently and was never acknowledged', 'find the session where we touched file Y'. server_prompts.py exists but the prompt surface is thin, and there is no harness-side skill teaching Polylogue idioms the way the beads… Design direction: Three thin layers over existing capability, no new query machinery: (1) MCP prompts: register ~6 intent-named prompts (resume-context, postmortem-last, decisions-about, unacknowledged-failures, sessions-touching-file, cost-of) that expand to the right tool-call sequences with cwd/repo prefilled — prompts are the MCP-native discoverability channel. (2) A 'polylogue' harness skill (dots/claude/skills + codex overlay, …

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

1. Three thin layers over existing capability, no new query machinery: (1) MCP prompts: register ~6 intent-named prompts (resume-context, postmortem-last, decisions-about, unacknowledged-failures, sessions-touching-file, cost-of) that expand to the right tool-call sequences with cwd/repo prefilled — prompts are the MCP-native discoverability channel.
2. (2) A 'polylogue' harness skill (dots/claude/skills + codex overlay, sinnix-side) with the same recipes in agent-readable form plus the two rules agents get wrong (archive root env var
3. refs over dumps).
4. (3) The SessionStart preamble (37t.4) ends with a one-line affordance index pointing at those prompts — injection makes the surface ambient.
5. Acceptance: affordance-usage report shows tool diversity rising in agent sessions (baseline: today's usage is dominated by search/get_session).
6. Keep total prompt count small — the cookbook is a curation, not another catalog.

## Tests to add

- Acceptance proof: ~6 intent-named MCP prompts are registered (resume-context, postmortem-last, decisions-about, unacknowledged-failures, sessions-touching-file, cost-of) that expand to the correct tool-call sequences with cwd/repo prefilled
- Acceptance proof: total prompt count stays small (curation, not another catalog).
- Acceptance proof: The prompt set includes the coordination intents over the shared envelope (agent_status, agent_self, work_item/current packet, coordination_hazards, addressed_messages, handoff) per the s7ae coordination update.
- Acceptance proof: A `polylogue` harness skill (dots/claude/skills + codex overlay, sinnix-side) carries the same recipes plus the two rules agents get wrong (archive-root env var
- Acceptance proof: refs over dumps).
- Acceptance proof: The SessionStart preamble (37t.4) ends with a one-line affordance index pointing at those prompts.
- Acceptance proof: MCP prompt/tool code and generated contracts are complete BEFORE any deployment batch (EXPECTED_TOOL_NAMES / prompt registry + `devtools render openapi` + `render all --check` clean)
- Acceptance proof: if only deployment remains it is recorded explicitly on the bead.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
