# 25. polylogue-4ts.6 — Expose transcript completeness instead of silently reading truncated sessions

Priority: **P2**  
Lane: **lineage-truth**  
Readiness: **needs-source-confirmation then model patch**

## Why this is urgent / critical-path

A transcript reader that returns a partial session without saying so creates false context packs and misleading evidence.

## Static diagnosis / likely mechanism

Bead title says silently truncated transcripts need a completeness signal. Likely sources: provider export truncation, partial file capture, continuation branches, max-message reads, or parser fallback. The exact current code path needs `rg 'truncated|complete|partial|max_messages|limit' polylogue/sources polylogue/storage polylogue/read` before patching.

## Implementation plan

Implementation shape:
1. Identify all parser/read paths that can return partial sessions: provider export flags, capture payload bounds, read limits, branch-local raw logs, and daemon read pagination.
2. Add a `transcript_completeness` enum or structured field: `complete`, `partial_export`, `parser_partial`, `read_limited`, `unknown`, with reason/source.
3. Store completeness in session/profile/read payloads.
4. Context packs and reports must show the signal and avoid strong claims over partial sessions.
5. Parser fixtures should set completeness from provider metadata where available; read-limit completeness is render-time, not stored source truth.

## Test plan

Tests:
- parser fixture with known truncated export stores partial reason.
- daemon/CLI read with `limit` says read-limited while source remains complete.
- context pack includes completeness caveat.
- complete sessions remain uncluttered/complete.

## Verification command / proof

`devtools test tests/unit/sources tests/unit/read tests/unit/daemon -k 'complete or completeness or truncated or partial'`

## Pitfalls

Keep source completeness separate from projection/read-limit completeness. A user selecting first 100 messages did not make the source incomplete.

## Files/functions to inspect or touch

- `polylogue/sources/parsers/*`
- `polylogue/archive/read*`
- `polylogue/daemon/http.py read routes`
- `profile/session models`
- `context pack renderers`
