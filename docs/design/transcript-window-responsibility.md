# Transcript-window read: who owns what

Scope: the request "give me messages `[offset, offset + limit)` of one
session", on the four public surfaces (Python API, CLI, MCP, HTTP daemon).
The companion request — the session summary/list — is covered by the
oracle-backed differential in `tests/infra/surface_differential.py`; this sheet
covers only the transcript window.

## One execution route

`polylogue/operations/transcript_window.py` owns the transcript window. Every
public surface reaches it, and none of them decides a window itself:

| Surface | Entry point | Reaches the route through |
| --- | --- | --- |
| Python API | `Polylogue.read_transcript_window` | `message_transcript_window` |
| CLI | `read <ref> --view messages [--limit --offset \| --continuation]` | `run_messages` → `Polylogue.read_transcript_window` |
| MCP | `read(ref, view="messages", limit=, offset=, continuation=)` | `Polylogue.read_transcript_window` |
| HTTP | `GET /api/sessions/:id/messages?limit=&offset=&continuation=` (and the same window behind `/api/sessions/:id/read?view=messages`) | `Polylogue.read_transcript_window`, or `read_transcript_window_sync` on the web reader's pinned archive |

The route owns the four things that must not differ by surface:

* **window arithmetic** — `[offset, offset + limit)`, `next_offset`, `complete`;
* **snapshot binding** — the archive epoch is read before *and* after the
  storage read, so a continuation is only minted for a window composed against
  one snapshot;
* **epoch validation** — a resume whose token was issued against an older
  snapshot raises `QueryContinuationStaleError` (CLI: a typed
  `query_continuation_stale` refusal; MCP: the same error code; HTTP: 409);
* **continuation vocabulary** — one opaque `QueryContinuation` token, carrying
  the `session-owner-v1` projection. A token minted on any surface decodes and
  resumes on every other, and it is the same token the `sessions.read` declared
  operation mints.

The selection vocabulary is preserved rather than dropped: `message_role`,
`message_type` and `material_origin` are fields of the request contract
(`SessionRead`), so they are part of the token's request identity — a
continuation minted for a filtered read cannot resume an unfiltered one.

## What stays with each surface, and why

Each surface keeps its **row projection**. That is a rendering difference, not
an execution one, and collapsing it would drop real fields:

* the CLI, MCP and Python API answer with domain `Message` rows
  (`message_row_envelope_from_domain`);
* the HTTP web reader answers with composed `ArchiveMessageRow` rows, which
  additionally carry `source_session_id` / `inherited_prefix` (lineage-prefix
  provenance the domain `Message` model does not represent) and the stored
  per-message `word_count` the reader totals against the session row. It also
  computes `semantic_entries` / `semantic_cards` / `semantic_card_suppressed`
  placement over the bounded page.

The webui consumes `GET /api/sessions/:id/read?view=messages`
(`webui/src/contracts/session-read.ts`), which is served by the same two
payload builders as `/api/sessions/:id/messages`. Their payload fields are
unchanged by the unification; `next_offset` and `continuation` were added
alongside, so the consumer is additive-compatible.

## Storage compatibility remains below the route

`session.read` (`polylogue/operations/daemon_reads.py::_session_read_payload`)
still serves the CLI's **session document** read — the whole-session body with
blocks and session identity, composed by `ArchiveStore.read_session_page`. It
is a different product from the message-row window: different rows, different
envelope, different consumer. It keeps its own continuation because it runs
inside the operation kernel against a pinned snapshot.

`Polylogue.get_messages_paginated` remains a public compatibility method, but
it now projects the result of `read_transcript_window`; it no longer owns
selection, window arithmetic, snapshot binding, or continuation. The shared
route reads the repository directly. `tests/unit/operations/test_transcript_window_route.py`
enforces that no public surface calls the facade storage method directly.

## Verification

* `tests/integration/test_transcript_window_differential.py` — ids, total,
  tiling, **rank and provenance**, and snapshot-bound continuation minting
  across API/CLI/MCP/HTTP.
* `tests/unit/archive/query/test_continuation_surface_parity.py` — the same
  token resumes identically on all four surfaces, and a token issued before a
  write lands is refused identically by all four.
* `tests/unit/operations/test_transcript_window_route.py` — the route census:
  re-splitting it turns this red.
