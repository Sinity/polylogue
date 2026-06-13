# GitHub triage plan — Polylogue Archive Workbench

This plan is prepared for `Sinity/polylogue`. It uses the open-issue export generated on 2026-05-23 as the dedupe baseline.

Use two actions:

- `create`: open a new focused issue when the work is not already represented clearly.
- `comment`: add the issue body as a design/acceptance update to an existing open issue when a matching issue already exists.

Before running the helper script, refresh the GitHub issue list and confirm that referenced issues are still open.

## Existing issue anchors

- #993 — web reader advanced functionality tracking.
- #1205 / #865 — reader visual smoke and degraded-state coverage.
- #873 / #1267 / #1420 — ranked search explanation, facets, pagination, lane preservation.
- #866 / #1261 — session identity, lineage graph, typed topology read model.
- #1418 / #1419 / #1415 / #1414 — OpenAPI, surface parity, machine errors, shared list response.
- #1446 / #1447 / #1321 / #999 — daemon health, catch-up, WAL/memory, metrics, operations.
- #958 — CLI polish and command ergonomics.

## Suggested milestones

1. `MK4.1 Reader Object Spine`
2. `MK4.2 Materials and Native Lineage`
3. `MK4.3 Workspaces and Context Composer`
4. `MK4.4 Operations and Surface Parity`

## Suggested labels

`type:feat`, `type:design`, `type:test`, `area:daemon`, `area:site`, `area:storage`, `area:cli`, `area:mcp`, `area:schema`, `theme:ux`, `theme:evidence-trust`, `theme:operations`, `theme:command-surface`, `phase:mk4`.

## Creation order

Open or update in this order:

1. `POLY-MK4-001` and `POLY-MK4-002` — reader contracts and action registry.
2. `POLY-MK4-003` and `POLY-MK4-004` — session reader and visual/state fixtures.
3. `POLY-MK4-005` and `POLY-MK4-006` — paste/attachment materials.
4. `POLY-MK4-007` and `POLY-MK4-008` — topology detail and native fork lens.
5. Workspaces/context/search/operations/surface parity follow once the reader spine is concrete.

Do not open every target issue as an immediate implementation promise. The manifest includes target-only work so it can be tracked cleanly, not because it should all start now.
