# 022. polylogue-a7xr.1 — Sweep remaining sqlite3 connection leaks: 'with sqlite3.connect()' commits but never closes

Priority/type/status: **P2 / bug / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

Python's sqlite3 Connection context manager commits/rolls back the TRANSACTION on __exit__ but does NOT close the connection — a well-known trap. insights/otlp_correlation.py:116 already documents it and uses contextlib.closing; but ~9 other sites still leak: coordination/envelope.py:591 (_sqlite_user_version, leaks 3 conns PER envelope build — agent-polled hot path), api/user_state_resolver.py:59/67/91 (per user-state read), api/archive.py:2931/4626, archive/raw_payload/decode.py:309, storage/repair.py:112, demo/seed.py:82. Connections leak until GC -> ResourceWarnings, fd pressure under sustained load. FIX: wrap each in contextlib.closing(sqlite3.connect(...)) (or try/finally: conn.close()), matching the otlp_correlation.py fix and the try/finally pattern already used by _archive_evidence_payloads right below the leaking _sqlite_user_version. Follows the 2026-05-31 ResourceWarning/conn-leak remediation — these are the stragglers + new coordination-code regressions.

## Acceptance criteria

Every 'with sqlite3.connect(...)' in non-test polylogue/ either closes via contextlib.closing/try-finally or is justified; a ResourceWarning-as-error test run over the coordination-envelope and user_state_resolver hot paths shows no leaked connections. Verify: rg 'with sqlite3.connect' polylogue/ --type py -g '!*test*' returns only closing()-wrapped forms; pytest -W error::ResourceWarning on the touched paths passes.

## Static mechanism / likely defect

Design direction: Python's sqlite3 Connection context manager commits/rolls back the TRANSACTION on __exit__ but does NOT close the connection — a well-known trap. insights/otlp_correlation.py:116 already documents it and uses contextlib.closing; but ~9 other sites still leak: coordination/envelope.py:591 (_sqlite_user_version, leaks 3 conns PER envelope build — agent-polled hot path), api/user_state_resolver.py:59/67/91 (per user-st…

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

1. Python's sqlite3 Connection context manager commits/rolls back the TRANSACTION on __exit__ but does NOT close the connection — a well-known trap.
2. insights/otlp_correlation.py:116 already documents it and uses contextlib.closing
3. but ~9 other sites still leak: coordination/envelope.py:591 (_sqlite_user_version, leaks 3 conns PER envelope build — agent-polled hot path), api/user_state_resolver.py:59/67/91 (per user-state read), api/archive.py:2931/4626, archive/raw_payload/decode.py:309, storage/repair.py:112, demo/seed.py:82.
4. Connections leak until GC -> ResourceWarnings, fd pressure under sustained load.
5. FIX: wrap each in contextlib.closing(sqlite3.connect(...)) (or try/finally: conn.close()), matching the otlp_correlation.py fix and the try/finally pattern already used by _archive_evidence_payloads right below the leaking _sqlite_user_version.
6. Follows the 2026-05-31 ResourceWarning/conn-leak remediation — these are the stragglers + new coordination-code regressions.

## Tests to add

- Acceptance proof: Every 'with sqlite3.connect(...)' in non-test polylogue/ either closes via contextlib.closing/try-finally or is justified
- Acceptance proof: a ResourceWarning-as-error test run over the coordination-envelope and user_state_resolver hot paths shows no leaked connections.
- Acceptance proof: Verify: rg 'with sqlite3.connect' polylogue/ --type py -g '!*test*' returns only closing()-wrapped forms
- Acceptance proof: pytest -W error::ResourceWarning on the touched paths passes.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
