# GitHub triage plan — Polylogue Archive Workbench

This plan is a historical design-pack artifact prepared for `Sinity/polylogue`.
It used an open-issue export generated on 2026-05-23 as the dedupe baseline;
that baseline is stale. Do not use this file as a current issue map.

Use two actions:

- `create`: open a new focused issue when the work is not already represented clearly.
- `comment`: add the issue body as a design/acceptance update to an existing open issue when a matching issue already exists.

Before mining any idea from this pack, refresh the GitHub issue list, read the
current issue bodies, and fold useful detail into the live owning issue instead
of creating a parallel roadmap.

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
