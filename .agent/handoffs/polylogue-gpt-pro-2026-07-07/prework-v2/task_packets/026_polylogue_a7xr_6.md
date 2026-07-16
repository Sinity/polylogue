# 026. polylogue-a7xr.6 — parse_archive_datetime: 6 copies, one with different tz semantics (naive/aware time bomb)

Priority/type/status: **P2 / bug / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Divergence audit: identical _parse_archive_datetime copies in context/selection.py:285, mcp/archive_support.py:492, cli/read_views/standard.py:232, api/archive.py:514, archive/query/archive_execution.py:113 (naive stays naive; empty string raises) vs a DIVERGENT copy in storage/insights/session/rebuild.py:763 (empty->None; naive FORCED to UTC). The same stored string parses to offset-naive or offset-aware depending on surface — a latent TypeError (cannot compare naive and aware) across insight vs read paths. Also _iso_from_epoch_ms x5 with a strict/lenient split (daemon/provenance.py:84, storage/embeddings/status_payload.py:338 lenient; three strict one-liners).

## Existing design note

core/timestamps.py is the designated home (docstring: unified timestamp parsing, all operations UTC): add parse_archive_datetime() with the rebuild copy's UTC-forcing semantics (matches the module contract) + iso_from_epoch_ms(); delete all copies. Audit each call site for naive-datetime comparisons that silently relied on naive semantics (mypy + tests are the net). Part of the cpf temporal doctrine surface.

## Acceptance criteria

One definition each; all six+five sites import core/timestamps; a test asserts the parsed value is ALWAYS tz-aware UTC; no naive-vs-aware comparison remains reachable (grep + focused tests). Verify: devtools test -k timestamp.

## Static mechanism / likely defect

Issue description localizes the mechanism: Divergence audit: identical _parse_archive_datetime copies in context/selection.py:285, mcp/archive_support.py:492, cli/read_views/standard.py:232, api/archive.py:514, archive/query/archive_execution.py:113 (naive stays naive; empty string raises) vs a DIVERGENT copy in storage/insights/session/rebuild.py:763 (empty->None; naive FORCED to UTC). The same stored string parses to offset-naive or offset-aware depending on surface — a latent TypeError (cannot compare naive and aware) across insight vs read paths. Also … Design direction: core/timestamps.py is the designated home (docstring: unified timestamp parsing, all operations UTC): add parse_archive_datetime() with the rebuild copy's UTC-forcing semantics (matches the module contract) + iso_from_epoch_ms(); delete all copies. Audit each call site for naive-datetime comparisons that silently relied on naive semantics (mypy + tests are the net). Part of the cpf temporal doctrine surface.

## Source anchors to inspect first

- `polylogue/context/selection.py:285` — One of several _parse_archive_datetime copies.
- `polylogue/mcp/archive_support.py:492` — MCP copy of _parse_archive_datetime.
- `polylogue/api/archive.py:514` — API copy of _parse_archive_datetime.
- `polylogue/cli/read_views/standard.py:232` — CLI read-view copy of _parse_archive_datetime.
- `polylogue/archive/query/archive_execution.py:113` — Archive query execution copy of _parse_archive_datetime.
- `polylogue/storage/insights/session/rebuild.py:763` — Divergent storage/insight copy that forces naive timestamps to UTC.
- `polylogue/daemon/provenance.py:84` — One of several _iso_from_epoch_ms copies.
- `polylogue/storage/embeddings/status_payload.py:338` — Lenient _iso_from_epoch_ms copy in embeddings status payload.
- `polylogue/core/dates.py:10` — parse_date has no injected clock parameter.
- `polylogue/core/dates.py:37` — RELATIVE_BASE uses ambient datetime.now.
- `polylogue/archive/query/expression.py:2440` — Query grammar recognizes relative-date literals.
- `polylogue/archive/query/spec.py:498` — SessionQuerySpec.from_params is the central query-spec constructor.
- `polylogue/insights/temporal_source.py:66` — classify_profile_hwm_source promotes any updated_at to provider_ts.
- `polylogue/insights/temporal_source.py:97` — classify_aggregate_hwm_source currently collapses all non-empty source updates to provider_ts.

## Implementation plan

1. core/timestamps.py is the designated home (docstring: unified timestamp parsing, all operations UTC): add parse_archive_datetime() with the rebuild copy's UTC-forcing semantics (matches the module contract) + iso_from_epoch_ms()
2. delete all copies.
3. Audit each call site for naive-datetime comparisons that silently relied on naive semantics (mypy + tests are the net).
4. Part of the cpf temporal doctrine surface.

## Tests to add

- Acceptance proof: One definition each
- Acceptance proof: all six+five sites import core/timestamps
- Acceptance proof: a test asserts the parsed value is ALWAYS tz-aware UTC
- Acceptance proof: no naive-vs-aware comparison remains reachable (grep + focused tests).
- Acceptance proof: Verify: devtools test -k timestamp.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
