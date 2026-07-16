# 063. polylogue-d1y — polylogue hooks install: one-command harness wiring + hook liveness monitoring

Priority/type/status: **P1 / feature / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Hooks are the highest-fidelity capture channel (event-granularity, 100% coverage vs ~79% post-hoc per docs/hooks.md) and the enabling substrate for context injection — yet wiring them is manual settings.json surgery per harness, per machine, per event type (16 Claude Code events, 6 Codex), and NOTHING notices when they stop firing (harness update, moved script, broken PATH): capture silently degrades to post-hoc JSONL discovery. On this very machine only a recall hook + two agent-event hooks are wired — not even the recommended starter set.

## Existing design note

(1) INSTALL: 'polylogue hooks install [--harness claude-code|codex] [--events recommended|all|<list>]' — idempotently merges the polylogue-hook entries into the harness settings (respects existing hooks, writes the minimal diff, --dry-run shows it); 'polylogue hooks status' shows wired-vs-recommended per harness with the exact missing entries; uninstall symmetric. Settings-file formats are harness-version-dependent — VERIFY current schemas at build time and encode per-harness adapters, not one template. (2) LIVENESS: the daemon knows which harnesses are active (it ingests their JSONL); cross-check per session: sessions from harness X arriving with zero hook events while hooks claim to be installed = hook-flow gap -> health alert (daemon health surface + status line + /metrics gauge). Per-event-type observed-rate over a trailing window catches partial breakage (e.g. Stop firing but PreToolUse gone after a harness update renamed an event). (3) COVERAGE REPORT: 'polylogue hooks status --coverage' = per-harness event-type table: wired / observed-last-7d / enrichment value (from docs/hooks.md roles), so expanding coverage is a decision from evidence. (4) The install path is also the distribution story for 37t.4's SessionStart preamble and the advisory hooks — one command turns a fresh machine into a fully-instrumented one; that makes it a 3tl demo asset too ('polylogue hooks install' inside the one-command demo tour).

## Acceptance criteria

On a clean settings.json, hooks install --harness claude-code --events recommended wires the starter set; a second run produces zero diff. hooks status shows wired vs observed-last-7d per event type. With hooks wired and the script broken, the daemon raises a hook-flow health alert within one session.

## Static mechanism / likely defect

Issue description localizes the mechanism: Hooks are the highest-fidelity capture channel (event-granularity, 100% coverage vs ~79% post-hoc per docs/hooks.md) and the enabling substrate for context injection — yet wiring them is manual settings.json surgery per harness, per machine, per event type (16 Claude Code events, 6 Codex), and NOTHING notices when they stop firing (harness update, moved script, broken PATH): capture silently degrades to post-hoc JSONL discovery. On this very machine only a recall hook + two agent-event hooks are wired — not even t… Design direction: (1) INSTALL: 'polylogue hooks install [--harness claude-code|codex] [--events recommended|all|<list>]' — idempotently merges the polylogue-hook entries into the harness settings (respects existing hooks, writes the minimal diff, --dry-run shows it); 'polylogue hooks status' shows wired-vs-recommended per harness with the exact missing entries; uninstall symmetric. Settings-file formats are harness-version-dependent …

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

1. (1) INSTALL: 'polylogue hooks install [--harness claude-code|codex] [--events recommended|all|<list>]' — idempotently merges the polylogue-hook entries into the harness settings (respects existing hooks, writes the minimal diff, --dry-run shows it)
2. 'polylogue hooks status' shows wired-vs-recommended per harness with the exact missing entries
3. uninstall symmetric.
4. Settings-file formats are harness-version-dependent — VERIFY current schemas at build time and encode per-harness adapters, not one template.
5. (2) LIVENESS: the daemon knows which harnesses are active (it ingests their JSONL)
6. cross-check per session: sessions from harness X arriving with zero hook events while hooks claim to be installed = hook-flow gap -> health alert (daemon health surface + status line + /metrics gauge).
7. Per-event-type observed-rate over a trailing window catches partial breakage (e.g.

## Tests to add

- Acceptance proof: On a clean settings.json, hooks install --harness claude-code --events recommended wires the starter set
- Acceptance proof: a second run produces zero diff.
- Acceptance proof: hooks status shows wired vs observed-last-7d per event type.
- Acceptance proof: With hooks wired and the script broken, the daemon raises a hook-flow health alert within one session.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
