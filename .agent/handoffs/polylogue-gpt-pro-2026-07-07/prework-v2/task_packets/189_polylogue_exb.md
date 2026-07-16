# 189. polylogue-exb — Layering: substrate rings import the api facade (6 sites, 2 private-symbol reaches)

Priority/type/status: **P2 / task / open**. Lane: **11-interoperability-origin**. Release: **K-interop-origin**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The architecture says surfaces adapt over substrate, but the dependency arrow runs backwards in at least six places: storage/embeddings/preflight.py imports select_pending_embedding_session_window from polylogue.api; storage/embeddings/materialization.py and insights/correlation_view.py import api.sync.bridge.run_coroutine_sync; storage/repair.py imports the PRIVATE api.archive._rebuild_archive_session_insights; sources/live/batch.py imports the whole Polylogue facade; pipeline/run_stages.py imports the PRIVATE api.archive._active_archive_root. Most are function-local imports — the classic cycle-hiding smell. The layering linter blesses all of it because docs/plans/layering.yaml disallow lists for storage/pipeline/sources/insights omit polylogue/api. Consequence: the facade cannot be decomposed or slimmed while the substrate calls up into it, and every one of these edges is a latent import cycle.

## Existing design note

Per-site relocation, then close the gate: (1) _active_archive_root -> config/core (it is runtime-root resolution, nothing facade-y about it); (2) run_coroutine_sync -> a core/asyncbridge module (two substrate rings need it; the api.sync home is an accident of history); (3) _rebuild_archive_session_insights: repair orchestration needs the insight-rebuild primitive — move the primitive into insights/ or pipeline/ and have BOTH api and repair call it downward; (4) select_pending_embedding_session_window -> storage.embeddings owns pending-window selection already (sql.py) — the api re-export should be the alias, not the source; (5) sources/live/batch.py Polylogue facade use is the hard one: identify which facade methods it actually calls and inject them as a narrow protocol from the daemon composition root instead of importing the facade (dependency injection at the call boundary). Then: add polylogue/api to the disallow lists for storage/pipeline/sources/insights in layering.yaml so devtools verify layering enforces the direction permanently; the function-local-import trick stops working because the linter reads imports statically wherever they occur (verify it catches function-local imports; if not, fix the linter first). Sequence before the facade decomposition bead — decomposition is impossible while substrate calls up.

## Acceptance criteria

The six inward imports are relocated (grep for 'polylogue.api' under storage/, sources/, insights/, pipeline/ returns nothing, including function-local). layering.yaml disallows polylogue/api for all four substrate rings and devtools verify layering passes. No behavior change: testmon-affected suite green.

## Static mechanism / likely defect

Issue description localizes the mechanism: The architecture says surfaces adapt over substrate, but the dependency arrow runs backwards in at least six places: storage/embeddings/preflight.py imports select_pending_embedding_session_window from polylogue.api; storage/embeddings/materialization.py and insights/correlation_view.py import api.sync.bridge.run_coroutine_sync; storage/repair.py imports the PRIVATE api.archive._rebuild_archive_session_insights; sources/live/batch.py imports the whole Polylogue facade; pipeline/run_stages.py imports the PRIVATE ap… Design direction: Per-site relocation, then close the gate: (1) _active_archive_root -> config/core (it is runtime-root resolution, nothing facade-y about it); (2) run_coroutine_sync -> a core/asyncbridge module (two substrate rings need it; the api.sync home is an accident of history); (3) _rebuild_archive_session_insights: repair orchestration needs the insight-rebuild primitive — move the primitive into insights/ or pipeline/ and …

## Source anchors to inspect first

- No precise source anchor was localized in this static pass. Start from the bead description and repository search.

## Implementation plan

1. Per-site relocation, then close the gate: (1) _active_archive_root -> config/core (it is runtime-root resolution, nothing facade-y about it)
2. (2) run_coroutine_sync -> a core/asyncbridge module (two substrate rings need it
3. the api.sync home is an accident of history)
4. (3) _rebuild_archive_session_insights: repair orchestration needs the insight-rebuild primitive — move the primitive into insights/ or pipeline/ and have BOTH api and repair call it downward
5. (4) select_pending_embedding_session_window -> storage.embeddings owns pending-window selection already (sql.py) — the api re-export should be the alias, not the source
6. (5) sources/live/batch.py Polylogue facade use is the hard one: identify which facade methods it actually calls and inject them as a narrow protocol from the daemon composition root instead of importing the facade (dependency injection at the call boundary).
7. Then: add polylogue/api to the disallow lists for storage/pipeline/sources/insights in layering.yaml so devtools verify layering enforces the direction permanently

## Tests to add

- Acceptance proof: The six inward imports are relocated (grep for 'polylogue.api' under storage/, sources/, insights/, pipeline/ returns nothing, including function-local).
- Acceptance proof: layering.yaml disallows polylogue/api for all four substrate rings and devtools verify layering passes.
- Acceptance proof: No behavior change: testmon-affected suite green.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
