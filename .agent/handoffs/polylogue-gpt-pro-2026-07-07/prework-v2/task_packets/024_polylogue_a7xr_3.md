# 024. polylogue-a7xr.3 — message_type_backfill reconstructs prose unordered and unfiltered; message-prose SQL exists 5x

Priority/type/status: **P2 / bug / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

VERIFIED LIVE 2026-07-06: storage/message_type_backfill.py:54-64 claims (comment) to concatenate block text in position order, but its GROUP_CONCAT has no inner ORDER BY — SQLite GROUP_CONCAT is unordered, so the #839 classifier can receive scrambled prose. It also omits the block_type='text' filter (thinking/tool text leaks into classification) and uses a single-newline separator, while the embeddings/demo family (storage/embeddings/materialization.py:535/754/923, demo/seed.py:607, demo/constructs.py:240) uses double-newline + block_type filter + min-length HAVING. Five paste sites, one concept, one real ordering bug — and demo/constructs.py exists to VERIFY the embedding selector but pastes the SQL instead of importing it, so drift silently breaks the verification.

## Existing design note

message_prose_sql(alias, *, separator, block_types, min_chars) fragment builder next to archive_embeddable_message_where (the factoring pattern already proven there); backfill gains ordered concatenation via correlated subquery (SELECT GROUP_CONCAT(text, sep) FROM (SELECT text FROM blocks WHERE message_id=m.message_id AND ... ORDER BY position)); all five sites compose the builder; demo/constructs.py imports it (verification becomes real).

## Acceptance criteria

One builder; backfill output for a multi-block fixture is position-ordered (regression test with 3+ blocks inserted out of order); block_type filter applied on the classifier path; embeddings selection output unchanged (golden). Verify: devtools test -k 'backfill or message_type or embeddable'.

## Static mechanism / likely defect

Issue description localizes the mechanism: VERIFIED LIVE 2026-07-06: storage/message_type_backfill.py:54-64 claims (comment) to concatenate block text in position order, but its GROUP_CONCAT has no inner ORDER BY — SQLite GROUP_CONCAT is unordered, so the #839 classifier can receive scrambled prose. It also omits the block_type='text' filter (thinking/tool text leaks into classification) and uses a single-newline separator, while the embeddings/demo family (storage/embeddings/materialization.py:535/754/923, demo/seed.py:607, demo/constructs.py:240) uses do… Design direction: message_prose_sql(alias, *, separator, block_types, min_chars) fragment builder next to archive_embeddable_message_where (the factoring pattern already proven there); backfill gains ordered concatenation via correlated subquery (SELECT GROUP_CONCAT(text, sep) FROM (SELECT text FROM blocks WHERE message_id=m.message_id AND ... ORDER BY position)); all five sites compose the builder; demo/constructs.py imports it (ver…

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
- `polylogue/core/dates.py:10` — parse_date has no injected clock parameter.
- `polylogue/core/dates.py:37` — RELATIVE_BASE uses ambient datetime.now.
- `polylogue/archive/query/expression.py:2440` — Query grammar recognizes relative-date literals.
- `polylogue/archive/query/spec.py:498` — SessionQuerySpec.from_params is the central query-spec constructor.
- `polylogue/insights/temporal_source.py:66` — classify_profile_hwm_source promotes any updated_at to provider_ts.
- `polylogue/insights/temporal_source.py:97` — classify_aggregate_hwm_source currently collapses all non-empty source updates to provider_ts.

## Implementation plan

1. message_prose_sql(alias, *, separator, block_types, min_chars) fragment builder next to archive_embeddable_message_where (the factoring pattern already proven there)
2. backfill gains ordered concatenation via correlated subquery (SELECT GROUP_CONCAT(text, sep) FROM (SELECT text FROM blocks WHERE message_id=m.message_id AND ...
3. ORDER BY position))
4. all five sites compose the builder
5. demo/constructs.py imports it (verification becomes real).

## Tests to add

- Acceptance proof: One builder
- Acceptance proof: backfill output for a multi-block fixture is position-ordered (regression test with 3+ blocks inserted out of order)
- Acceptance proof: block_type filter applied on the classifier path
- Acceptance proof: embeddings selection output unchanged (golden).
- Acceptance proof: Verify: devtools test -k 'backfill or message_type or embeddable'.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
