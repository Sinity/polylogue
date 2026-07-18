# Wave 2 evidence-admission audit — 2026-07-18

This ledger prevents returned agent work from becoming anonymous archive
material. A package is preserved even when its patch is stale; source review
and current-owner acceptance criteria decide admission.

## Absorbed or terminally classified

- `misc-01` is superseded by the merged configuration-closure work.
- `support-d` is superseded by PR #3080: its compact synthetic crash/restart
  fixed-point proof landed. This does **not** satisfy the separate July-scale,
  resource-envelope, and daemon-health scope of `polylogue-hjpx.2`.
- `webui-05` is structurally incomplete; retain it as evidence only.

## Inputs delivered to active owners

- Lane A: `support-a` (analysis evidence kernel), `ann-05` (claims/evidence
  view), and `mandate-02` (observed work effects). These are design and
  acceptance inputs; durable-schema patches are not admitted wholesale.
- Lane B: `support-b` (Hermes verification-ledger and fidelity harness). The
  owner decides any selective adoption against the landed runtime-watcher
  source.
- Lane C: `support-c` (API/MCP/HTTP transaction certification), `mcp-01`,
  `mcp-04`, `lin-02`, and `mandate-03`. Use their discovery, continuation,
  continuity-oracle, and terminal-gate matrices; do not replay their stale
  patches alongside the canonical current transaction.
- Lane E: `webui-01`, `webui-02`, `webui-03/r01`, `webui-03/r02`, `webui-04`,
  `webui-07`, and `webui-08`. The replacement `webui-03/r02` is the preferred
  external search reference because it understands the scaffold, generated
  client, and design system. Lane E remains sole owner of `daemon/http.py`.

## Deferred, still valuable packets

- Lane F candidates: `perf-01`, `perf-02`, `perf-03`, and `perf-04`.
- Future annotation work: `ann-01`, `ann-02`, and `ann-03`.
- Future lineage and orchestration work: `lin-01`, `lin-03`, and `lin-04`.
- Future substrate/retrieval work: `misc-02` and `misc-03`.
- Future product/research work: `res-04`, `mcp-02`, and `mcp-03`.

## Admission rule

Before a future owner reuses a deferred package, read its `receipt.json`,
`HANDOFF.md`, and current source. A `snapshot_mismatch` or
`needs_rebase_review` package may contribute requirements, tests, and design
arguments, but its patch is never applied blindly. Update the receipt state
only when a concrete current-master admission or supersession is evidenced.
