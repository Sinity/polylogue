# Polylogue Design Direction

This directory holds product and interaction design material that is specific
enough to guide implementation. Architecture-level contracts still live in
`docs/architecture*.md`; this area is for the user-facing reader, workbench,
and surface behavior those contracts need to support.

## Canonical surface design

| Pack | Status | Purpose |
|------|--------|---------|
| [MK3 design pack](mk3/README.md) | Canonical current target | Archive workbench direction for the daemon web reader, multi-chat workspace, topology, paste and attachment rendering, user state, realtime, degraded states, and implementation slices. |
| [MK2 coding-agent pack](mk2/coding-agent-pack/README.md) | Historical input | Earlier daemon-first reader shell and verification handoff. Keep for context, but new reader planning should start from MK3. |

## How to use MK3

Use [MK3 north star](mk3/docs/00-mk3-north-star.md) for product intent,
[MK3 data model proposal](mk3/docs/03-data-model-mk3.md) for missing
contracts, and [MK3 API and component contracts](mk3/docs/09-api-and-component-contracts.md)
for surface boundaries. The execution order lives in
[`docs/execution-plan.md`](../execution-plan.md), where MK3 has been mapped
onto the existing open issues instead of maintained as a parallel roadmap. The
bundled [visual board](mk3/index.html) and screenshots under `mk3/screens/`
are evidence inputs for the reader visual smoke work.
