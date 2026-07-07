---
created: "2026-06-30T20:52:42+02:00"
purpose: "Record the active /realm/inbox reorganization for the Polylogue devloop"
status: "active"
project: "polylogue"
---

# Inbox Reorg Assimilation

## Current Layout

The inbox cleanup moved historical Polylogue/Sinex material out of loose root
paths. Future evidence reads should use the new routing files first:

- `/realm/inbox/project-devloops/README.md` — historical devloop packets,
  raw exports, conductor source notes, and older reports.
- `/realm/inbox/project-artifacts/README.md` — downloaded patches, prompt
  exports, browser-profile preserves, legacy archives, and cross-stack
  methodology/forensics inputs.

Active shelves remain top-level and should not be treated as historical dumps:

- `/realm/inbox/polylogue-conductor-devloop`
- `/realm/inbox/sinex-conductor-devloop`
- `/realm/inbox/demos_polylogue`
- `/realm/inbox/demos_sinex`
- `/realm/inbox/codices`

## Operational Consequence

Older notes that mention `/realm/inbox/download` or loose inbox-root files are
historical provenance, not current routing instructions. When the devloop needs
to re-read a brief, patch, prompt export, or older raw devloop artifact, first
consult the two new README files and then follow their organized subpaths.

Do not rewrite old provenance notes merely to hide their original source path.
Do update active instructions, handoffs, and future demo manifests so they do
not imply the old loose layout is still current.

## Process Rule

Before a slice that uses external briefs, downloaded materials, old devloop
exports, or historical reports:

1. Read the relevant `project-devloops` or `project-artifacts` README.
2. Record the current routed path in the operating log or artifact summary.
3. Keep active conductor/demo outputs in their top-level shelves unless the
   shelf itself is intentionally being reorganized.
