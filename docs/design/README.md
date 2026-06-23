# Polylogue Design Direction

This directory holds product and interaction design material that is specific
enough to guide implementation. Architecture-level contracts still live in
`docs/architecture*.md`; this area is for the user-facing reader, workbench,
and surface behavior those contracts need to support.

## Design Inputs

| Input | Status | Purpose |
|------|--------|---------|
| [Query-action workflows](query-action-workflows.md) | Moved pointer | Redirects to the generated product contract at `docs/product/workflows.md`, which is rendered from live workflow and action registries. |
| [MK3 design pack](mk3/README.md) | Historical source material | Archive workbench direction for the daemon web reader, multi-chat workspace, topology, paste and attachment rendering, user state, realtime, degraded states, and implementation slices. |

## How To Use These Docs

Use the MK3 pack as reference material, not as dispatch truth. The current
implementation order lives in [`docs/execution-plan.md`](../execution-plan.md)
and the open GitHub issues. If a design detail still matters, fold it into the
owning issue or current docs before implementing it.

The bundled MK3 boards and screenshots are useful for visual vocabulary and
fixture ideas, but they do not override the live query, route, assertion, and
workbench contracts in the codebase.
