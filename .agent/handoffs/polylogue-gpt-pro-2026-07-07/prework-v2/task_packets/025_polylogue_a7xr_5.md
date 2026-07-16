# 025. polylogue-a7xr.5 — FTS trigger DDL declared twice: archive_tiers/index.py vs fts_lifecycle repair copies

Priority/type/status: **P2 / task / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Same class as the closed fts_freshness_state double-declaration, three more objects: trigger DDL for messages_fts/session_work_events_fts/threads_fts lives in BOTH storage/sqlite/archive_tiers/index.py (:307-324, :729-767, :449-464) and storage/fts/fts_lifecycle.py (:198-233+ as _BLOCKS/_SESSION_WORK_EVENT/_THREAD trigger DDL constants used by drop-and-recreate repair). Byte-equivalent today; any future edit forks trigger behavior between fresh DBs and repaired DBs. No test couples the two sources.

## Existing design note

Move trigger DDL lists to storage/fts/sql.py (already holds FTS_INDEX_EXISTS_SQL) as the single source; archive_tiers/index.py composes its DDL script from them; fts_lifecycle imports them. Derived-tier regime: pure code move, no schema bump (emitted DDL identical — assert via normalized-text comparison in the PR). Relates 1xc.12 (drift gauges family).

## Acceptance criteria

rg finds each trigger body in exactly one module; a drift test asserts fresh-DB and repair-path trigger text are identical (normalized); rebuild + repair smoke green. Verify: devtools test -k fts.

## Static mechanism / likely defect

Issue description localizes the mechanism: Same class as the closed fts_freshness_state double-declaration, three more objects: trigger DDL for messages_fts/session_work_events_fts/threads_fts lives in BOTH storage/sqlite/archive_tiers/index.py (:307-324, :729-767, :449-464) and storage/fts/fts_lifecycle.py (:198-233+ as _BLOCKS/_SESSION_WORK_EVENT/_THREAD trigger DDL constants used by drop-and-recreate repair). Byte-equivalent today; any future edit forks trigger behavior between fresh DBs and repaired DBs. No test couples the two sources. Design direction: Move trigger DDL lists to storage/fts/sql.py (already holds FTS_INDEX_EXISTS_SQL) as the single source; archive_tiers/index.py composes its DDL script from them; fts_lifecycle imports them. Derived-tier regime: pure code move, no schema bump (emitted DDL identical — assert via normalized-text comparison in the PR). Relates 1xc.12 (drift gauges family).

## Source anchors to inspect first

- `polylogue/storage/sqlite/archive_tiers/index.py:296` — messages_fts table DDL lives in archive_tiers/index.py.
- `polylogue/storage/fts/sql.py:11` — messages_fts DDL copy also exists in storage/fts/sql.py.
- `polylogue/storage/fts/fts_lifecycle.py:292` — ensure_fts_triggers_sync owns runtime repair/recreation.
- `polylogue/storage/fts/fts_lifecycle.py:512` — Thread FTS rebuild logic is duplicated in lifecycle repair path.
- `polylogue/daemon/convergence_stages.py:988` — daemon convergence has another repair/readiness path for archive messages_fts.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Move trigger DDL lists to storage/fts/sql.py (already holds FTS_INDEX_EXISTS_SQL) as the single source
2. archive_tiers/index.py composes its DDL script from them
3. fts_lifecycle imports them.
4. Derived-tier regime: pure code move, no schema bump (emitted DDL identical — assert via normalized-text comparison in the PR).
5. Relates 1xc.12 (drift gauges family).

## Tests to add

- Acceptance proof: rg finds each trigger body in exactly one module
- Acceptance proof: a drift test asserts fresh-DB and repair-path trigger text are identical (normalized)
- Acceptance proof: rebuild + repair smoke green.
- Acceptance proof: Verify: devtools test -k fts.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
