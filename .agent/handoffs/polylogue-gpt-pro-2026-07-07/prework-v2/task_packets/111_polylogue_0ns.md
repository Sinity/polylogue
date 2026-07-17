# 111. polylogue-0ns — Bound archive embedding work within large sessions

Priority/type/status: **P2 / task / open**. Lane: **09-embeddings-retrieval**. Release: **J-embeddings-retrieval**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Why: while verifying live daemon convergence on 2026-07-04, a forced embedding debt drain could run longer than the outer daemon session window because _embed_archive_sessions_sync checks _DAEMON_EMBED_STOP_AFTER_SECONDS only between sessions, while embed_archive_session_sync can process a very large session internally. What needs to be done: make archive embedding resumable/bounded within a single huge session, or have the daemon select message windows instead of whole-session units so automatic catch-up remains responsive under very large Codex/Claude sessions.

## Existing design note

Make archive embedding bounded within a single large session so a forced debt drain cannot exceed the daemon window. Root cause: _embed_archive_sessions_sync checks _DAEMON_EMBED_STOP_AFTER_SECONDS only between sessions, while embed_archive_session_sync processes a whole session internally. Fix option (a): check the stop-after deadline inside embed_archive_session_sync at message-window granularity and persist a resumable position; or (b) have the daemon select message windows (via select_pending_archive_session_window) instead of whole-session units. Files: the daemon embed loop (_embed_archive_sessions_sync / embed_archive_session_sync) and the pending-window selection helper.

## Acceptance criteria

1. embed_archive_session_sync honors _DAEMON_EMBED_STOP_AFTER_SECONDS (or an equivalent deadline) at message-window granularity within one session and records a resumable position, so the next daemon tick continues the same session rather than restarting it. 2. Regression test: a synthetic session larger than one embedding window, with the stop-after deadline set below the whole-session cost, produces a partial embed that resumes and completes across ticks with no unbounded single-session run. Verify via `devtools test` selection on the daemon embed path. 3. Live/seeded check: a forced embedding debt drain returns within the configured window bound and `polylogue ops embed status --detail` shows monotonic progress across bounded runs.

## Static mechanism / likely defect

Issue description localizes the mechanism: Why: while verifying live daemon convergence on 2026-07-04, a forced embedding debt drain could run longer than the outer daemon session window because _embed_archive_sessions_sync checks _DAEMON_EMBED_STOP_AFTER_SECONDS only between sessions, while embed_archive_session_sync can process a very large session internally. What needs to be done: make archive embedding resumable/bounded within a single huge session, or have the daemon select message windows instead of whole-session units so automatic catch-up remains … Design direction: Make archive embedding bounded within a single large session so a forced debt drain cannot exceed the daemon window. Root cause: _embed_archive_sessions_sync checks _DAEMON_EMBED_STOP_AFTER_SECONDS only between sessions, while embed_archive_session_sync processes a whole session internally. Fix option (a): check the stop-after deadline inside embed_archive_session_sync at message-window granularity and persist a res…

## Source anchors to inspect first

- `polylogue/api/embeddings.py` — Public embedding API seam.
- `polylogue/cli/commands/embed.py` — Operator embedding command and dry-run planner hooks.
- `polylogue/archive/query/retrieval.py` — Retrieval composition seam for FTS/vector/hybrid.
- `polylogue/archive/query/retrieval_search.py` — Search/retrieval runtime implementation.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Make archive embedding bounded within a single large session so a forced debt drain cannot exceed the daemon window.
2. Root cause: _embed_archive_sessions_sync checks _DAEMON_EMBED_STOP_AFTER_SECONDS only between sessions, while embed_archive_session_sync processes a whole session internally.
3. Fix option (a): check the stop-after deadline inside embed_archive_session_sync at message-window granularity and persist a resumable position
4. or (b) have the daemon select message windows (via select_pending_archive_session_window) instead of whole-session units.
5. Files: the daemon embed loop (_embed_archive_sessions_sync / embed_archive_session_sync) and the pending-window selection helper.

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: embed_archive_session_sync honors _DAEMON_EMBED_STOP_AFTER_SECONDS (or an equivalent deadline) at message-window granularity within one session and records a resumable position, so the next daemon tick continues the same session rather than restarting it.
- Acceptance proof: 2.
- Acceptance proof: Regression test: a synthetic session larger than one embedding window, with the stop-after deadline set below the whole-session cost, produces a partial embed that resumes and completes across ticks with no unbounded single-session run.
- Acceptance proof: Verify via `devtools test` selection on the daemon embed path.
- Acceptance proof: 3.
- Acceptance proof: Live/seeded check: a forced embedding debt drain returns within the configured window bound and `polylogue ops embed status --detail` shows monotonic progress across bounded runs.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
