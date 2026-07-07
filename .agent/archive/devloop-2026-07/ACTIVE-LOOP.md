# Active Loop

## Current Objective

Conduct the Polylogue dogfood/demo devloop indefinitely: continuously choose the
highest-value live-archive capability slice, produce inspectable artifacts
proving Polylogue improves agents with real history, collapse silos into general
acquisition/query/projection/rendering substrate, verify on the canonical active
archive or live browser capture, maintain timestamped operating logs and
handoffs, adversarially review archive/process/resource state, and use each
loop's evidence to reprioritize while maximizing devloop velocity.

## Current Slice

fs1.1 Hermes state.db importer current-internals verification

## Other In-Progress Beads

- `polylogue-fs1.1` is the current slice: verify current Hermes state.db
  internals, then import that database through the canonical acquisition,
  parsing, and archive write path without creating a Hermes-only silo.
- `polylogue-bby.9` remains in progress as the mission-control renderer bead.
  The archive-evidence fields landed in `polylogue-s7ae.4`; residual before
  closure is first-class subagent dispatch prompt / returned-final-message
  rendering and the stronger live multi-agent proof.

## Meta Origin

no

## Current Focus

Focus: Direction -> Evidence

Trigger: fs1.1 Hermes state.db importer current-internals verification

Decision: Gather current evidence before editing, then record the next material focus switch.

## Accepted Warnings

None recorded for this slice. If `devloop-review` warnings are consciously accepted, record only current-slice exceptions here; historical proofs and completed slices belong in `OPERATING-LOG.md` or `DEMO-RADAR.md`.

## Next Action

Continue in `Evidence` mode: gather current evidence before editing.

## Do Not Drift

- Do not reintroduce `/realm/tmp/polylogue-dev/archive` as a live database root.
- Do not quote counts without archive root and schema version.
- Do not preserve compatibility endpoints, flags, or DTOs just because removal
  is broader than the current file.
- Stay on the current long-lived branch for ordinary loop work; commit logical,
  proven chunks by path and avoid worktrees unless isolation is actually needed.
- Use compile/test/daemon wait time for ahead work in this checkout. A failed
  proof can be retried after batched fixes; it should not freeze the loop.
- Do not overcorrect into "demo instead of substrate." Demonstrated value is the
  forcing function, but substrate repair is the right slice whenever broken
  archive/query/rendering state would make a demo false or fragile.
- Add slice-specific guardrails deliberately; do not inherit stale guardrails
  from the previous slice.
