# 146. polylogue-6wnh — Bound thread refresh cost for large Codex appends

Priority/type/status: **P2 / task / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Current 3wb closure evidence shows the old 260s append.index.graph_resolve rebuild tail is not active on recent daemon appends, but the worst current graph_resolve sample is still dominated by append.index.graph_resolve.thread_refresh: 3.020976s of a 3.040429s graph_resolve step on a 340.8 MB Codex append. This is not a P1 blocker while raw replay backlog is zero and recent samples are bounded, but it is the concrete next optimization if thread_refresh becomes the next tail.

## Existing design note

Use /realm/tmp/polylogue-workload-graph-tail-499e26363.json as the initial evidence. Inspect the thread_refresh implementation behind append.index.graph_resolve.thread_refresh; determine whether it can refresh only affected thread/session rows instead of rebuilding broader thread projections. Add timing/profiling evidence before editing. Preserve lineage/thread correctness and do not skip real topology updates. If the implementation is already incremental, add a regression diagnostic/SLO around the current bounded timing instead of changing code.

## Acceptance criteria

A focused benchmark or live diagnostic shows thread_refresh cost on giant Codex append/replay rows; either the implementation becomes incremental and the worst recent 340 MiB-class thread_refresh path is materially reduced, or the bead records why the current cost is the correct bounded floor with a guardrail that would catch a regression toward the 260s class.

## Static mechanism / likely defect

Issue description localizes the mechanism: Current 3wb closure evidence shows the old 260s append.index.graph_resolve rebuild tail is not active on recent daemon appends, but the worst current graph_resolve sample is still dominated by append.index.graph_resolve.thread_refresh: 3.020976s of a 3.040429s graph_resolve step on a 340.8 MB Codex append. This is not a P1 blocker while raw replay backlog is zero and recent samples are bounded, but it is the concrete next optimization if thread_refresh becomes the next tail. Design direction: Use /realm/tmp/polylogue-workload-graph-tail-499e26363.json as the initial evidence. Inspect the thread_refresh implementation behind append.index.graph_resolve.thread_refresh; determine whether it can refresh only affected thread/session rows instead of rebuilding broader thread projections. Add timing/profiling evidence before editing. Preserve lineage/thread correctness and do not skip real topology updates. If t…

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

1. Use /realm/tmp/polylogue-workload-graph-tail-499e26363.json as the initial evidence.
2. Inspect the thread_refresh implementation behind append.index.graph_resolve.thread_refresh
3. determine whether it can refresh only affected thread/session rows instead of rebuilding broader thread projections.
4. Add timing/profiling evidence before editing.
5. Preserve lineage/thread correctness and do not skip real topology updates.
6. If the implementation is already incremental, add a regression diagnostic/SLO around the current bounded timing instead of changing code.

## Tests to add

- Acceptance proof: A focused benchmark or live diagnostic shows thread_refresh cost on giant Codex append/replay rows
- Acceptance proof: either the implementation becomes incremental and the worst recent 340 MiB-class thread_refresh path is materially reduced, or the bead records why the current cost is the correct bounded floor with a guardrail that would catch a regression toward the 260s class.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
