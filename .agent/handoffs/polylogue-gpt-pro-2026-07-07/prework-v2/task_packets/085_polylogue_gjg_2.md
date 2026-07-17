# 085. polylogue-gjg.2 — Pre-compaction snapshot capture: hook payload when available, manifest-of-refs otherwise, honesty ladder always

Priority/type/status: **P2 / task / open**. Lane: **03-lineage-compaction-truth**. Release: **F-lineage-compaction**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Two-level snapshotting with snapshot_source as a FIRST-CLASS honesty axis, not a footnote: precompact-hook (strongest — the actual assembled context payload, blob-stored content-addressed, claim: this WAS model context) > jsonl-boundary (manifest of composed-transcript message refs up to the boundary; claim limited to archive-composed transcript, NOT model context) > reconstructed-composed-context (weakest) > none (epidemiology only). Do not store duplicated text when a manifest of message/block/blob refs suffices; exact payload blobs only from the hook. Loss-forensics claims MUST downgrade wording per snapshot_source.

## Existing design note

PreCompact hook wiring rides d1y (hooks install — existing gjg dependency). VERIFY the current Claude Code PreCompact payload actually carries assembled context before promising the strongest rung (known open question; the hook catalog moves). Codex equivalent via app-server events is ox0 territory. Blob dedup makes repeated compactions near-free; source-tier hook event row (raw_hook_events exists) links the blob.

## Acceptance criteria

A live compaction on the operator machine lands either a hook snapshot or a labeled jsonl-boundary manifest; every snapshot row carries source+confidence; no unlabeled reconstruction. Verify: live dogfood compaction + fixture for the fallback.

## Static mechanism / likely defect

Issue description localizes the mechanism: Two-level snapshotting with snapshot_source as a FIRST-CLASS honesty axis, not a footnote: precompact-hook (strongest — the actual assembled context payload, blob-stored content-addressed, claim: this WAS model context) > jsonl-boundary (manifest of composed-transcript message refs up to the boundary; claim limited to archive-composed transcript, NOT model context) > reconstructed-composed-context (weakest) > none (epidemiology only). Do not store duplicated text when a manifest of message/block/blob refs suffices… Design direction: PreCompact hook wiring rides d1y (hooks install — existing gjg dependency). VERIFY the current Claude Code PreCompact payload actually carries assembled context before promising the strongest rung (known open question; the hook catalog moves). Codex equivalent via app-server events is ox0 territory. Blob dedup makes repeated compactions near-free; source-tier hook event row (raw_hook_events exists) links the blob.

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

1. PreCompact hook wiring rides d1y (hooks install — existing gjg dependency).
2. VERIFY the current Claude Code PreCompact payload actually carries assembled context before promising the strongest rung (known open question
3. the hook catalog moves).
4. Codex equivalent via app-server events is ox0 territory.
5. Blob dedup makes repeated compactions near-free
6. source-tier hook event row (raw_hook_events exists) links the blob.

## Tests to add

- Acceptance proof: A live compaction on the operator machine lands either a hook snapshot or a labeled jsonl-boundary manifest
- Acceptance proof: every snapshot row carries source+confidence
- Acceptance proof: no unlabeled reconstruction.
- Acceptance proof: Verify: live dogfood compaction + fixture for the fallback.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
