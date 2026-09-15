# Transcript-window read: who owns what

Scope: the request "give me messages `[offset, offset + limit)` of one
session", on the four public surfaces (Python API, CLI, MCP, HTTP daemon).
The companion request — the session summary/list — is covered by the
oracle-backed differential in `tests/infra/surface_differential.py`; this sheet
covers only the transcript window, and states honestly which parts of the
shared-execution goal are done and which are not.

## Before / after

| Surface | Entry point | Executes through | Window expressible? |
| --- | --- | --- | --- |
| Python API | `Polylogue.get_messages_paginated` | `repository.get_messages_paginated` | limit + offset — yes (unchanged) |
| CLI | `read <ref> --view messages --limit --offset` | `lower_session_read` → `session.read` operation → `ArchiveStore.read_session_page` | limit + offset + snapshot-bound continuation (unchanged) |
| MCP | `read(ref, view="messages", …)` | `Polylogue.get_messages_paginated` | **before: first page only** (`offset` hard-coded to `0`, no parameter, continuation refused); **after: limit + offset + `message-offset:` continuation** |
| HTTP | `GET /api/sessions/:id/messages?limit=&offset=` | `Polylogue.get_messages_paginated` | limit + offset — yes (unchanged) |

What actually changed here is MCP, and it is a relocation of nothing: no
branch moved surfaces. The MCP messages view previously could not name any
window but the first page, so the four surfaces did not answer the same
request at all. It now can, and
`tests/integration/test_transcript_window_differential.py` holds them to one
answer for the same `(ref, limit, offset)`.

The `offset` vocabulary is deliberately shared rather than reinvented: MCP's
`message-offset:<n>` continuation is the same decimal form its `topology` view
already uses (`node-offset:<n>`), so no surface mints a token another surface
cannot read.

## Two owners remain, and that is the open work

There are still **two** implementations of "window one session's messages":

1. the `session.read` declared operation
   (`polylogue/operations/daemon_reads.py::_session_read_payload`), reached by
   the CLI, which carries the snapshot-bound `QueryContinuation` and the
   operation-kernel read control (cancellation, deadline, pinned snapshot);
2. `Polylogue.get_messages_paginated`
   (`polylogue/api/archive.py`) → `repository.get_messages_paginated`, reached
   by the Python API, MCP and HTTP, which carries `limit`/`offset` and
   `LineageCompleteness` but no continuation token and no query-transaction
   read control.

Route (2) additionally exposes `message_role` / `message_type` /
`material_origin` filters that route (1) does not, so they are not yet
interchangeable implementations of one contract; collapsing them is a
selection-vocabulary change, not a call-site swap.

Consequences that are real today, not hypothetical:

- Only the CLI can resume a transcript window across a concurrent write with
  epoch validation. The other three resume by re-asking for an offset, which
  a write landing between pages can shift.
- Only route (1) runs inside `QueryExecutionContext`, so cancellation and
  deadline behaviour on a long transcript window differ by surface.

These are the remaining holes for the "one executable read algebra" goal
(polylogue-4p1) and they are recorded here rather than papered over.
Unifying them means deciding which owner keeps the richer message-filter
vocabulary and giving the losing route's callers the winner's envelope —
work that belongs to its own task, because it changes the HTTP webui payload
(`semantic_entries`, semantic-card placement) that
`/api/sessions/:id/messages` serves.
