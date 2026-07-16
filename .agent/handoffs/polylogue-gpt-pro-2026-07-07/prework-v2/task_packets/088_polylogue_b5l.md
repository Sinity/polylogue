# 088. polylogue-b5l — Blue-green index rebuilds: fresh-first without downtime

Priority/type/status: **P1 / task / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **blocked-hard**.

Hard blockers: polylogue-20d.15

## What the bead says

Operator re-think directive (2026-07-03): the fresh-first doctrine's CORRECTNESS half is right (no in-place migration chains, derived tiers rebuild from source) but its OPERATIONAL half is wrong — a schema bump currently means 'ops reset --index && polylogued run' with the archive degraded for the whole rebuild (observed live: 20-40 min at 12-15 rows/s, every surface confused meanwhile). Rebuilds should be non-events: build the NEW index beside the OLD, serve the old until the new converges, swap atomically. Fresh-first stays; downtime goes.

## Existing design note

(1) MECHANISM: index tier gets generation-suffixed files (index.g42.db); a tiny pointer (ops.db row + symlink for external readers) names the active generation. Rebuild = daemon builds g43 from source.db in the background (at 20d.15 bulk speed, with idle IO class) while all reads keep hitting g42; when g43 reaches convergence parity (raw-materialization complete, FTS ready — the 4bu snapshot is the readiness oracle), swap the pointer under a write pause measured in milliseconds, then delete g42 after a grace window. (2) WRITES DURING REBUILD: live ingest double-writes to both generations during the window (idempotent-by-content-hash makes replay-instead-of-double-write an acceptable simpler alternative: finish bulk, then replay the delta since rebuild start from source.db — pick by measured window size). (3) SCOPE: index.db only (embeddings.db already tier-reset-able and its rebuild is money-bounded not time-bounded; source/user are never rebuilt). (4) SURFACES: ops reset --index becomes 'schedule blue-green rebuild' with the old behavior behind --offline; status/webui show 'rebuilding generation 43: N% (serving 42)' via the convergence snapshot. (5) DOCTRINE EDIT: internals.md schema-versioning section replaces 'operator moves the index aside' with the generation flow; the lab policy lint (no upgrade chains) is UNCHANGED — this is still rebuild-from-source, just concurrent. Depends on 20d.15 (a 5-minute rebuild makes the double-write window trivial) and 4bu (readiness oracle).

## Acceptance criteria

A schema bump on the seeded corpus (and then the live archive) completes with zero failed queries: reads served continuously from the old generation, swap under 100ms write pause, delta replayed, old generation reaped. ops reset --index --offline preserves the old path. internals.md updated; the 20-40min degraded window is gone from the operator runbook.

## Static mechanism / likely defect

Current derived-tier reset/rebuild doctrine can produce long degraded windows. Blue-green means building a fresh generation beside the served one and swapping only after convergence proof.

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

1. Define index generation metadata: generation id, schema version, source snapshot, build status, replay cursor, ready flag, active pointer.
2. Build new `index.db` generation out-of-place while serving old generation.
3. Replay writes or pause briefly at swap; document exact consistency model.
4. Make daemon/web/CLI readiness report old generation served/new generation building, not archive-ready over partial corpus.

## Tests to add

- Schema bump fixture builds new generation and serves old until ready.
- Crash mid-build leaves old generation active.
- Swap is atomic; post-swap row counts/materialization checks pass.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
