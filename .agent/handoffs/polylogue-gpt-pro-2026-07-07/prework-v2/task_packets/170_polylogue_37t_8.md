# 170. polylogue-37t.8 — Resume routing: map a session to the harness invocation that reopens it

Priority/type/status: **P2 / feature / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Genuinely-missing item: nothing owns 'reopen this session in its harness' — claude --resume <id> vs the codex equivalent, per origin; plus detecting an already-open interactive session (the kitty/hyprland control plane can answer that on this machine, but keep that integration optional/pluggable). Natural terminal action for the continue verb and the last mile of the resumption loop: find ... then continue should end with the session actually open.

## Existing design note

Add a mapping from an archived session to the harness invocation that reopens it, per origin: `claude --resume <id>` for Claude Code, the Codex equivalent, etc. Optionally detect an already-open interactive session via the kitty/hyprland control plane, kept behind a pluggable/optional interface. This is the last mile of the `continue` verb: `find ... then continue` should end with the session actually open (or the exact reopen command emitted).

## Acceptance criteria

1. A resume-routing helper maps (origin, native session id) to the concrete harness reopen command, covering at least Claude Code (`claude --resume <id>`) and Codex, with an explicit unsupported/unknown result for origins that have no reopen path. 2. The `continue` action (or `find ... | continue`) emits or executes the correct reopen invocation for the selected session. 3. Optional already-open detection sits behind a pluggable interface and degrades cleanly with no hard dependency on kitty/hyprland. Verify: `devtools test` selection on the resume-routing module asserts the per-origin command mapping for fixture sessions; a manual `continue` on a real Claude and Codex session opens it (recorded in the PR).

## Static mechanism / likely defect

Issue description localizes the mechanism: Genuinely-missing item: nothing owns 'reopen this session in its harness' — claude --resume <id> vs the codex equivalent, per origin; plus detecting an already-open interactive session (the kitty/hyprland control plane can answer that on this machine, but keep that integration optional/pluggable). Natural terminal action for the continue verb and the last mile of the resumption loop: find ... then continue should end with the session actually open. Design direction: Add a mapping from an archived session to the harness invocation that reopens it, per origin: `claude --resume <id>` for Claude Code, the Codex equivalent, etc. Optionally detect an already-open interactive session via the kitty/hyprland control plane, kept behind a pluggable/optional interface. This is the last mile of the `continue` verb: `find ... then continue` should end with the session actually open (or the e…

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

1. Add a mapping from an archived session to the harness invocation that reopens it, per origin: `claude --resume <id>` for Claude Code, the Codex equivalent, etc.
2. Optionally detect an already-open interactive session via the kitty/hyprland control plane, kept behind a pluggable/optional interface.
3. This is the last mile of the `continue` verb: `find ...
4. then continue` should end with the session actually open (or the exact reopen command emitted).

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: A resume-routing helper maps (origin, native session id) to the concrete harness reopen command, covering at least Claude Code (`claude --resume <id>`) and Codex, with an explicit unsupported/unknown result for origins that have no reopen path.
- Acceptance proof: 2.
- Acceptance proof: The `continue` action (or `find ...
- Acceptance proof: | continue`) emits or executes the correct reopen invocation for the selected session.
- Acceptance proof: 3.
- Acceptance proof: Optional already-open detection sits behind a pluggable interface and degrades cleanly with no hard dependency on kitty/hyprland.
- Acceptance proof: Verify: `devtools test` selection on the resume-routing module asserts the per-origin command mapping for fixture sessions

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
