# 095. polylogue-20d.5 — Finish streaming reads: composed transcripts, messages --full writer, origin-filtered pagination SQL

Priority/type/status: **P2 / task / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Residue of the streaming-export slice: lineage-composed transcript streaming falls back to the eager path; read --view messages --full --to file lacks a true writer/iterator renderer; material-origin-filtered pagination is eager until SQL owns the predicate.

## Existing design note

Three eager fallbacks to close (prior-audit evidence, re-locate): (1) lineage-composed transcript streaming falls back to the eager path — extend the streaming writer landed in a9dc3f274 to composed (parent-prefix + tail) reads; (2) read --view messages --full --to file lacks a true writer/iterator renderer — same pattern; (3) material-origin-filtered message pagination hydrates eagerly until SQL owns the predicate — push material_origin into the repository pagination SQL (pattern: a17e3af95 routed ordinary paginated reads through repository pagination). Verify each with a live-archive file export timing + RSS bound, plus focused unit tests on the streaming/pagination modules.

## Acceptance criteria

- Lineage-composed transcript streaming uses the streaming writer (extend the a9dc3f274 pattern) for composed (parent-prefix + tail) reads — no eager full-materialization fallback remains (grep the composed read path).
- `read --view messages --full --to <file>` uses a true iterator/writer renderer rather than eager buffering.
- Material-origin-filtered message pagination pushes `material_origin` into the repository pagination SQL (pattern a17e3af95); hydration no longer filters in Python.
- Each of the three is verified with a live-archive file export showing bounded peak RSS (flat vs message count) with export timing recorded, plus focused unit tests on the streaming/pagination modules (`devtools test <streaming/pagination modules>` green).

## Static mechanism / likely defect

Issue description localizes the mechanism: Residue of the streaming-export slice: lineage-composed transcript streaming falls back to the eager path; read --view messages --full --to file lacks a true writer/iterator renderer; material-origin-filtered pagination is eager until SQL owns the predicate. Design direction: Three eager fallbacks to close (prior-audit evidence, re-locate): (1) lineage-composed transcript streaming falls back to the eager path — extend the streaming writer landed in a9dc3f274 to composed (parent-prefix + tail) reads; (2) read --view messages --full --to file lacks a true writer/iterator renderer — same pattern; (3) material-origin-filtered message pagination hydrates eagerly until SQL owns the predicate …

## Source anchors to inspect first

- `CONTRIBUTING.md:102` — Derived-tier schema changes require rebuild/blue-green planning.
- `AGENTS.md:168` — Agent guidance says schema mismatch should rebuild or blue-green-replace derived tiers.
- `polylogue/cli/commands/reset.py` — Current reset/rebuild commands are the operator path to replace derived tiers.
- `polylogue/daemon/convergence_stages.py` — Daemon convergence/readiness state should represent generation progress honestly.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.
- `polylogue/storage/sqlite/archive_tiers/index.py:296` — messages_fts table DDL lives in archive_tiers/index.py.
- `polylogue/storage/fts/sql.py:11` — messages_fts DDL copy also exists in storage/fts/sql.py.
- `polylogue/storage/fts/fts_lifecycle.py:292` — ensure_fts_triggers_sync owns runtime repair/recreation.
- `polylogue/storage/fts/fts_lifecycle.py:512` — Thread FTS rebuild logic is duplicated in lifecycle repair path.
- `polylogue/daemon/convergence_stages.py:988` — daemon convergence has another repair/readiness path for archive messages_fts.

## Implementation plan

1. Three eager fallbacks to close (prior-audit evidence, re-locate): (1) lineage-composed transcript streaming falls back to the eager path — extend the streaming writer landed in a9dc3f274 to composed (parent-prefix + tail) reads
2. (2) read --view messages --full --to file lacks a true writer/iterator renderer — same pattern
3. (3) material-origin-filtered message pagination hydrates eagerly until SQL owns the predicate — push material_origin into the repository pagination SQL (pattern: a17e3af95 routed ordinary paginated reads through repository pagination).
4. Verify each with a live-archive file export timing + RSS bound, plus focused unit tests on the streaming/pagination modules.

## Tests to add

- Acceptance proof: Lineage-composed transcript streaming uses the streaming writer (extend the a9dc3f274 pattern) for composed (parent-prefix + tail) reads — no eager full-materialization fallback remains (grep the composed read path).
- Acceptance proof: `read --view messages --full --to <file>` uses a true iterator/writer renderer rather than eager buffering.
- Acceptance proof: Material-origin-filtered message pagination pushes `material_origin` into the repository pagination SQL (pattern a17e3af95)
- Acceptance proof: hydration no longer filters in Python.
- Acceptance proof: Each of the three is verified with a live-archive file export showing bounded peak RSS (flat vs message count) with export timing recorded, plus focused unit tests on the streaming/pagination modules (`devtools test <streaming/pagination modules>` green).

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
