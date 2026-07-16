# 089. polylogue-1xc.8 — Schema rebuild-safety scenario

Priority/type/status: **P2 / task / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

scenario-coverage.yaml gap 'schema-rebuild-safety' orphaned on gh#590. A scenario proving derived-tier rebuild (index/embeddings) from durable source/user evidence is lossless and idempotent, and durable-tier additive migration preserves user.db assertions. Ties 1xc.7 scale-regression lane + z7rv migration framework.

## Acceptance criteria

A rebuild-safety scenario resets a derived tier and rebuilds from source, asserting byte/row parity + no user.db loss; a durable additive migration round-trips behind the backup gate. Verify: the scenario under devtools lab lanes.

## Static mechanism / likely defect

Design direction: scenario-coverage.yaml gap 'schema-rebuild-safety' orphaned on gh#590. A scenario proving derived-tier rebuild (index/embeddings) from durable source/user evidence is lossless and idempotent, and durable-tier additive migration preserves user.db assertions. Ties 1xc.7 scale-regression lane + z7rv migration framework.

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

1. scenario-coverage.yaml gap 'schema-rebuild-safety' orphaned on gh#590.
2. A scenario proving derived-tier rebuild (index/embeddings) from durable source/user evidence is lossless and idempotent, and durable-tier additive migration preserves user.db assertions.
3. Ties 1xc.7 scale-regression lane + z7rv migration framework.

## Tests to add

- Acceptance proof: A rebuild-safety scenario resets a derived tier and rebuilds from source, asserting byte/row parity + no user.db loss
- Acceptance proof: a durable additive migration round-trips behind the backup gate.
- Acceptance proof: Verify: the scenario under devtools lab lanes.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
