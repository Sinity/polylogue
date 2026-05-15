# Polylogue MK3 design pack

MK3 treats Polylogue as an archive workbench, not only a chat reader. The design covers search, conversation reading, multi-chat workspaces, lineage/topology, paste and attachment handling, user-state workflows, derived insights, live/degraded states, and the exact data-model gaps that block those views.

The pack is deliberately implementation-shaped. It separates what can be rendered today from what needs new contract work, and it keeps every advanced panel honest by marking current, proposed, and deferred states.

Open `index.html` for the visual board. The markdown files are the handoff surface for implementers.

## Pack index

| Document | Purpose |
|----------|---------|
| [00 - MK3 north star](docs/00-mk3-north-star.md) | Product direction and design rules. |
| [01 - Current substrate audit](docs/01-current-substrate-audit.md) | What the current archive, daemon, and reader can assume. |
| [02 - View inventory](docs/02-view-inventory.md) | Complete MK3 view set and state expectations. |
| [03 - Data model proposal](docs/03-data-model-mk3.md) | TargetRef, topology, paste, attachment, user-state, and message envelope contracts. |
| [04 - Message rendering and actions](docs/04-message-rendering-and-actions.md) | Transcript rendering, copy, folds, and action rail behavior. |
| [05 - Paste and attachment rendering](docs/05-paste-and-attachment-rendering.md) | First-class paste spans and attachment states. |
| [06 - Topology continuation/forking](docs/06-topology-continuation-forking.md) | Continuation, fork, sidechain, and subagent modeling. |
| [07 - Multi-chat workspace](docs/07-multi-chatlog-workspace.md) | Tabs, stack, compare, and timeline modes. |
| [08 - State matrix](docs/08-state-matrix.md) | Ready, degraded, partial, stale, and error states. |
| [09 - API and component contracts](docs/09-api-and-component-contracts.md) | HTTP endpoints, mutation envelopes, and component inventory. |
| [10 - Implementation slices](docs/10-implementation-slices.md) | Sequenced implementation plan from target refs through design assurance. |
| [11 - Little details](docs/11-little-details.md) | Copy feedback, chip order, density, persistence, privacy, and keyboard shape. |

## Visual evidence

The board in [index.html](index.html) is the interactive reference. Static
screenshots live in [screens/](screens/) and cover reader, stack, topology,
attachments, states, and palette.

## Execution

MK3 is source material, not a separate planning silo. The active execution
sequence is folded into [the repository execution plan](../../execution-plan.md)
and the relevant GitHub issues. Use this pack for product and contract detail;
use the issue-set for ownership and closeout.
