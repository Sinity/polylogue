# 082. polylogue-4ts.5 — Compaction boundary-range columns + effective-context derivation

Priority/type/status: **P2 / feature / open**. Lane: **03-lineage-compaction-truth**. Release: **F-lineage-compaction**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

session_events boundary_start/end_position + boundary_message_id; get_effective_context(session, at_position) = what the model actually saw vs the full composed prefix. Schema bump + re-ingest. Surfaces: view=effective_context; precise replaced-range signal for stale_context pathology. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

Design from gh#2478 (code-grounded): add session_events.boundary_start_position/boundary_end_position (message range a compaction replaces, in the session's own position coordinate) + boundary_message_id (the materialized summary). Parser computes the range while walking records: start = prev boundary end + 1, end = message_position - 1; writer applies position_offset. Read helper get_effective_context(session, at_position) returns [summary] + post-boundary messages (what the model actually saw) vs the full composed prefix used for forks. Index-tier schema bump + re-ingest plan in the PR body (fresh-first doctrine; batch with other pending index bumps if possible). Surfaces: read view 'effective_context' (CLI/MCP/daemon auto-register via read_view_registry); stale_context pathology gets the precise replaced-range signal.

## Acceptance criteria

1. session_events gains boundary_start_position, boundary_end_position, and boundary_message_id (index-tier schema bump, with the rebuild/re-ingest plan stated in the PR body per fresh-first doctrine and batched with other pending index bumps where possible). 2. The parser populates the range while walking records — start = prev boundary end + 1, end = message_position - 1, with position_offset applied — for Codex `compacted` records and Claude inline / agent-acompact-* boundaries. 3. get_effective_context(session, at_position) returns [summary] + post-boundary messages (what the model saw) and a fixture asserts it differs from the full-composed fork prefix at the boundary. 4. A `read --view effective_context` surface auto-registers across CLI/MCP/daemon via read_view_registry. 5. The stale_context pathology consumes the precise replaced-range signal. Verify: focused parser + read-view tests pass (`devtools test` selection on the parser and read_view_registry files); a live re-ingest shows boundary_* rows populated for known compaction sessions.

## Static mechanism / likely defect

Issue description localizes the mechanism: session_events boundary_start/end_position + boundary_message_id; get_effective_context(session, at_position) = what the model actually saw vs the full composed prefix. Schema bump + re-ingest. Surfaces: view=effective_context; precise replaced-range signal for stale_context pathology. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Design from gh#2478 (code-grounded): add session_events.boundary_start_position/boundary_end_position (message range a compaction replaces, in the session's own position coordinate) + boundary_message_id (the materialized summary). Parser computes the range while walking records: start = prev boundary end + 1, end = message_position - 1; writer applies position_offset. Read helper get_effective_context(session, at_p…

## Source anchors to inspect first

- `polylogue/archive/session/threads.py` — Session/thread lineage read and composition model.
- `polylogue/insights/topology.py` — Topology/lineage derived insight code.
- `polylogue/daemon/lineage_startup.py` — Daemon lineage startup/convergence path.
- `polylogue/archive/coverage.py` — Completeness/truncation cues live here.
- `polylogue/insights/postmortem.py` — Compaction/continuation postmortem evidence is mined here.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## Implementation plan

1. Design from gh#2478 (code-grounded): add session_events.boundary_start_position/boundary_end_position (message range a compaction replaces, in the session's own position coordinate) + boundary_message_id (the materialized summary).
2. Parser computes the range while walking records: start = prev boundary end + 1, end = message_position - 1
3. writer applies position_offset.
4. Read helper get_effective_context(session, at_position) returns [summary] + post-boundary messages (what the model actually saw) vs the full composed prefix used for forks.
5. Index-tier schema bump + re-ingest plan in the PR body (fresh-first doctrine
6. batch with other pending index bumps if possible).
7. Surfaces: read view 'effective_context' (CLI/MCP/daemon auto-register via read_view_registry)

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: session_events gains boundary_start_position, boundary_end_position, and boundary_message_id (index-tier schema bump, with the rebuild/re-ingest plan stated in the PR body per fresh-first doctrine and batched with other pending index bumps where possible).
- Acceptance proof: 2.
- Acceptance proof: The parser populates the range while walking records — start = prev boundary end + 1, end = message_position - 1, with position_offset applied — for Codex `compacted` records and Claude inline / agent-acompact-* boundaries.
- Acceptance proof: 3.
- Acceptance proof: get_effective_context(session, at_position) returns [summary] + post-boundary messages (what the model saw) and a fixture asserts it differs from the full-composed fork prefix at the boundary.
- Acceptance proof: 4.
- Acceptance proof: A `read --view effective_context` surface auto-registers across CLI/MCP/daemon via read_view_registry.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
