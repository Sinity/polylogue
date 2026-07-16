# 083. polylogue-4ts.6 — Lineage composition silently truncates transcripts; surface a completeness signal

Priority/type/status: **P2 / bug / open**. Lane: **03-lineage-compaction-truth**. Release: **F-lineage-compaction**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

storage/sqlite/queries/message_query_reads.py:get_messages composes a prefix-sharing child's full logical transcript (parent prefix up to branch_point + child tail). Two paths silently return an INCOMPLETE transcript with no signal: (1) _depth >= _MAX_LINEAGE_DEPTH (=64, line 59/116) forces edge=None -> returns only the deepest session's own tail, dropping all ancestors beyond 64 (reachable via long Claude Code acompact chains). (2) found=False dangling branch point (140-141) returns child  — but a prefix-sharing child's own rows are ONLY its divergent tail (shared prefix dropped at write per #2467), so the reader gets a transcript starting mid-conversation. Neither surfaces incompleteness: the read envelope has no lineage-completeness field (the api/archive.py:1866  flag is postmortem-bundle cap, unrelated). For a system-of-record this is a construct-validity hole — a partial transcript is served as if whole. FIX: carry a typed completeness signal on the composed read (e.g. lineage_complete: bool + truncation_reason in {depth_limit, dangling_branch_point}); log at depth-limit hit; consider raising/removing the 64 cap now that composition is iterative, or making it explicit debt. Relates to the dangling-branch-point repair (9p0y) which reduces path (2)'s frequency but does not add the signal.

## Acceptance criteria

get_messages / read_archive_session_envelope returns (or the envelope carries) a completeness indicator; a depth>64 chain and a dangling-branch-point session both report lineage_complete=false with a reason, and the depth-limit hit is logged. Consumers (reader, MCP get_messages, context-image) can distinguish a complete logical transcript from a truncated one. Verify: unit tests constructing a >64 chain and a dangling branch point, asserting the completeness signal + log.

## Static mechanism / likely defect

Bead title says silently truncated transcripts need a completeness signal. Likely sources: provider export truncation, partial file capture, continuation branches, max-message reads, or parser fallback. The exact current code path needs `rg 'truncated|complete|partial|max_messages|limit' polylogue/sources polylogue/storage polylogue/read` before patching.

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

1. Implementation shape:
2. 1. Identify all parser/read paths that can return partial sessions: provider export flags, capture payload bounds, read limits, branch-local raw logs, and daemon read pagination.
3. 2. Add a `transcript_completeness` enum or structured field: `complete`, `partial_export`, `parser_partial`, `read_limited`, `unknown`, with reason/source.
4. 3. Store completeness in session/profile/read payloads.
5. 4. Context packs and reports must show the signal and avoid strong claims over partial sessions.
6. 5. Parser fixtures should set completeness from provider metadata where available; read-limit completeness is render-time, not stored source truth.

## Tests to add

- parser fixture with known truncated export stores partial reason.
- daemon/CLI read with `limit` says read-limited while source remains complete.
- context pack includes completeness caveat.
- complete sessions remain uncluttered/complete.

## Verification commands

- ``devtools test tests/unit/sources tests/unit/read tests/unit/daemon -k 'complete or completeness or truncated or partial'``

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
