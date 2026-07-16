# 141. polylogue-a7xr.11 — Prune protocols.py zero-consumer protocols + dead repo kwarg query surface + cursor mapping bug

Priority/type/status: **P2 / chore / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

VERIFIED 2026-07-06: 6 of 14 protocols in protocols.py have zero consumers anywhere (SessionReader, SearchStore, ArchiveMessageQueryStore, SemanticArchiveQueryStore, SessionSemanticStatsStore, SessionArchiveReadStore) — violating the module's own docstring rule ('only protocols with 2+ implementations earn their existence'). The 18-filter-kwarg signature is spelled out 3x in SessionReader alone. The repo kwarg methods are equally dead: RepositoryArchiveQueryMixin.list (docstring-example-only), .count (zero callers), .list_summaries (sole caller iter_summary_pages, itself zero callers). All real traffic goes SessionRecordQuery -> list_by_query/count_by_query. The SessionListQueryKwargs/SessionCountQueryKwargs TypedDicts are a pure 1:1 re-expansion consumed once. LATENT BUG (verified): archive/query/fields.py:797 maps record_attr='cursor' but SessionRecordQuery has no cursor field — dataclasses.replace would TypeError if a plan ever carried a cursor; unreachable today, proving the mapping is dead.

## Existing design note

Delete the 6 unconsumed protocols, the repo list/list_summaries/count kwarg wrappers + iter_summary_pages, the two TypedDicts (pass SessionRecordQuery through at query_store_archive.py:70-84), and either the cursor field-spec entry or add the cursor field deliberately (decide with rxdo pagination needs — a real cursor concept may arrive with result-set pagination; if so, wire it properly instead of deleting). KEEP protocols with real consumers: SessionQueryRuntimeStore, SessionOutputStore, SessionArchiveStatsStore, TagStore, RawPersistenceStore, RawValidationStore (genuine test double). mypy --strict is the net.

## Acceptance criteria

protocols.py contains only consumed protocols (each with a named consumer in a comment); dead kwarg surface gone; cursor mapping resolved (deleted or actually wired); mypy strict green; ~600 LOC removed. Verify: devtools verify.

## Static mechanism / likely defect

Issue description localizes the mechanism: VERIFIED 2026-07-06: 6 of 14 protocols in protocols.py have zero consumers anywhere (SessionReader, SearchStore, ArchiveMessageQueryStore, SemanticArchiveQueryStore, SessionSemanticStatsStore, SessionArchiveReadStore) — violating the module's own docstring rule ('only protocols with 2+ implementations earn their existence'). The 18-filter-kwarg signature is spelled out 3x in SessionReader alone. The repo kwarg methods are equally dead: RepositoryArchiveQueryMixin.list (docstring-example-only), .count (zero callers… Design direction: Delete the 6 unconsumed protocols, the repo list/list_summaries/count kwarg wrappers + iter_summary_pages, the two TypedDicts (pass SessionRecordQuery through at query_store_archive.py:70-84), and either the cursor field-spec entry or add the cursor field deliberately (decide with rxdo pagination needs — a real cursor concept may arrive with result-set pagination; if so, wire it properly instead of deleting). KEEP p…

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

1. Delete the 6 unconsumed protocols, the repo list/list_summaries/count kwarg wrappers + iter_summary_pages, the two TypedDicts (pass SessionRecordQuery through at query_store_archive.py:70-84), and either the cursor field-spec entry or add the cursor field deliberately (decide with rxdo pagination needs — a real cursor concept may arrive with result-set pagination
2. if so, wire it properly instead of deleting).
3. KEEP protocols with real consumers: SessionQueryRuntimeStore, SessionOutputStore, SessionArchiveStatsStore, TagStore, RawPersistenceStore, RawValidationStore (genuine test double).
4. mypy --strict is the net.

## Tests to add

- Acceptance proof: protocols.py contains only consumed protocols (each with a named consumer in a comment)
- Acceptance proof: dead kwarg surface gone
- Acceptance proof: cursor mapping resolved (deleted or actually wired)
- Acceptance proof: mypy strict green
- Acceptance proof: ~600 LOC removed.
- Acceptance proof: Verify: devtools verify.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
