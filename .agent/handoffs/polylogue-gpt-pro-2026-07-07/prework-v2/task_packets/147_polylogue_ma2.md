# 147. polylogue-ma2 — Add FK-supporting index for web_content_constructs message cleanup

Priority/type/status: **P2 / task / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Live v24 rebuild evidence showed ChatGPT full-replace rows spending seconds in append.index.full_replace.delete_messages. Source review found web_content_constructs.message_id has an ON DELETE CASCADE FK to messages(message_id) but no supporting index; active EXPLAIN planned SELECT 1 FROM web_content_constructs WHERE message_id = ? as SCAN web_content_constructs over about 89k rows. This should be a schema-index slice after v24 convergence, not during the active rebuild.

## Existing design note

After polylogue-6h7 closes, add a canonical index on web_content_constructs(message_id) in the index tier DDL, bump INDEX_SCHEMA_VERSION once as part of a batched schema slice, document the re-ingest plan in docs/internals.md, and add a focused planner/DDL test proving message-id lookups no longer scan the table. Coordinate with any other index-tier changes so one rebuild covers all of them.

## Acceptance criteria

EXPLAIN QUERY PLAN for SELECT 1 FROM web_content_constructs WHERE message_id = ? LIMIT 1 uses the new index on a seeded/current archive; full_replace delete_messages stage timing no longer shows web_content_constructs-driven table scans; schema version docs include the re-ingest plan; no in-place migration helper is added.

## Static mechanism / likely defect

Issue description localizes the mechanism: Live v24 rebuild evidence showed ChatGPT full-replace rows spending seconds in append.index.full_replace.delete_messages. Source review found web_content_constructs.message_id has an ON DELETE CASCADE FK to messages(message_id) but no supporting index; active EXPLAIN planned SELECT 1 FROM web_content_constructs WHERE message_id = ? as SCAN web_content_constructs over about 89k rows. This should be a schema-index slice after v24 convergence, not during the active rebuild. Design direction: After polylogue-6h7 closes, add a canonical index on web_content_constructs(message_id) in the index tier DDL, bump INDEX_SCHEMA_VERSION once as part of a batched schema slice, document the re-ingest plan in docs/internals.md, and add a focused planner/DDL test proving message-id lookups no longer scan the table. Coordinate with any other index-tier changes so one rebuild covers all of them.

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

1. After polylogue-6h7 closes, add a canonical index on web_content_constructs(message_id) in the index tier DDL, bump INDEX_SCHEMA_VERSION once as part of a batched schema slice, document the re-ingest plan in docs/internals.md, and add a focused planner/DDL test proving message-id lookups no longer scan the table.
2. Coordinate with any other index-tier changes so one rebuild covers all of them.

## Tests to add

- Acceptance proof: EXPLAIN QUERY PLAN for SELECT 1 FROM web_content_constructs WHERE message_id = ? LIMIT 1 uses the new index on a seeded/current archive
- Acceptance proof: full_replace delete_messages stage timing no longer shows web_content_constructs-driven table scans
- Acceptance proof: schema version docs include the re-ingest plan
- Acceptance proof: no in-place migration helper is added.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
