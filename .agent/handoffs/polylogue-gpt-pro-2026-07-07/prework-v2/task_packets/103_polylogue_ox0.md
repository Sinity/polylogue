# 103. polylogue-ox0 — Codex deep integration: state DBs as authoritative source + AppServer live lane

Priority/type/status: **P2 / task / open**. Lane: **11-interoperability-origin**. Release: **K-interop-origin**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Codex writes far more than rollout JSONL: state_5.sqlite (authoritative session/thread state — already used read-only by cost reconciliation lpl), goals_1.sqlite (goal/plan state — unexplored), history.jsonl, hooks.json, prompts/, rules/, skills/ (agent-config, 7aw), shell_snapshots/. And Codex ships an APP-SERVER: a JSON-RPC interface (used by IDE integrations) exposing thread lifecycle, streamed turn events, approval requests, and programmatic session control — which makes it BOTH a richer capture channel (live events with structure the JSONL flattens) AND the native remote-control lane for Codex (2n6's 'analogous for Codex' without terminal injection). This is the same pattern as the Hermes bridge: product-local state is authoritative; snapshot files are the fallback adapter.

## Existing design note

Three sub-lanes, sequenced: (1) STATE-DB IMPORTER (mirror of fs1.1): state_5.sqlite as authoritative Codex source — threads, turns, token counters with the disjoint-lane semantics already encoded in memory, parent/fork relations for lineage (4ts cross-check); goals_1.sqlite explored and mapped (VERIFY schema — undocumented); rollout JSONL demotes to fallback. Read via the copy-to-scratch discipline (live-locked DB). (2) APP-SERVER CAPTURE: a daemon-side client subscribing to the event stream during live Codex sessions -> event-granularity ingest (approvals, exec begin/end, plan updates — the Codex analogue of hook events); VERIFY the current protocol shape from the installed codex binary (codex app-server --help / the IDE extension's usage; the protocol is versioned and moves fast). (3) APP-SERVER CONTROL: 2n6's Codex leg — start/resume threads programmatically with a prompt, making 'continue this session' native instead of kitty injection. Fidelity declarations per lane (fs1.3 pattern); OriginSpec packaging (2qx) when it lands. Sequencing: (1) is pure value now; (2)+(3) after protocol verification, likely one investigation spike then two small implementation beads.

## Acceptance criteria

state_5.sqlite importer materializes threads/turns/lineage on the live machine and reconciles with existing rollout-derived sessions (dedup by content, no double-count — token totals match lpl reconciliation); goals_1.sqlite mapped with a written schema note; app-server protocol spike artifact committed (capabilities, version, event vocabulary) with go/no-go for lanes 2 and 3.

## Static mechanism / likely defect

Issue description localizes the mechanism: Codex writes far more than rollout JSONL: state_5.sqlite (authoritative session/thread state — already used read-only by cost reconciliation lpl), goals_1.sqlite (goal/plan state — unexplored), history.jsonl, hooks.json, prompts/, rules/, skills/ (agent-config, 7aw), shell_snapshots/. And Codex ships an APP-SERVER: a JSON-RPC interface (used by IDE integrations) exposing thread lifecycle, streamed turn events, approval requests, and programmatic session control — which makes it BOTH a richer capture channel (live … Design direction: Three sub-lanes, sequenced: (1) STATE-DB IMPORTER (mirror of fs1.1): state_5.sqlite as authoritative Codex source — threads, turns, token counters with the disjoint-lane semantics already encoded in memory, parent/fork relations for lineage (4ts cross-check); goals_1.sqlite explored and mapped (VERIFY schema — undocumented); rollout JSONL demotes to fallback. Read via the copy-to-scratch discipline (live-locked DB).…

## Source anchors to inspect first

- `polylogue/sources/dispatch.py` — Current origin/source dispatch logic; target for OriginSpec consolidation.
- `polylogue/sources/import_preflight.py` — Preflight/readiness should report origin strictness and ambiguity.
- `polylogue/sources/provider_completeness.py` — Provider completeness is adjacent to OriginSpec readiness.
- `polylogue/sources/parsers/base.py` — Parser base contracts should be folded into OriginSpec.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## Implementation plan

1. Three sub-lanes, sequenced: (1) STATE-DB IMPORTER (mirror of fs1.1): state_5.sqlite as authoritative Codex source — threads, turns, token counters with the disjoint-lane semantics already encoded in memory, parent/fork relations for lineage (4ts cross-check)
2. goals_1.sqlite explored and mapped (VERIFY schema — undocumented)
3. rollout JSONL demotes to fallback.
4. Read via the copy-to-scratch discipline (live-locked DB).
5. (2) APP-SERVER CAPTURE: a daemon-side client subscribing to the event stream during live Codex sessions -> event-granularity ingest (approvals, exec begin/end, plan updates — the Codex analogue of hook events)
6. VERIFY the current protocol shape from the installed codex binary (codex app-server --help / the IDE extension's usage
7. the protocol is versioned and moves fast).

## Tests to add

- Acceptance proof: state_5.sqlite importer materializes threads/turns/lineage on the live machine and reconciles with existing rollout-derived sessions (dedup by content, no double-count — token totals match lpl reconciliation)
- Acceptance proof: goals_1.sqlite mapped with a written schema note
- Acceptance proof: app-server protocol spike artifact committed (capabilities, version, event vocabulary) with go/no-go for lanes 2 and 3.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
