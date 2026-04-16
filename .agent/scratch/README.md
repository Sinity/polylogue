# Scratch Notes

This directory is organized by note role, not by date alone.

## Layout

- `active/`
  - operational notes that are still being updated while work is in flight
  - logs, audits, ongoing campaigns
- `plans/`
  - execution-ready plans and roadmaps
  - these should be the first stop when deciding what to build next
- `completed/`
  - finished analyses and already-landed architectural phases
  - use these to separate historical rationale from still-active work
- `concepts/`
  - architectural ideas, streamlining proposals, and higher-level design work
  - useful for strategy, but not necessarily ready to implement as-is
- `handoffs/`
  - reserved for reset-safe recovery notes when a dedicated handoff is warranted
  - currently unused; the active operator brief is the live reset-safe entrypoint
- `rejected/`
  - ideas that were explicitly considered and then ruled out
- `archive/`
  - closed historical notes and old bundles kept for reference only

## Fast Paths

If resuming runtime/product work after context loss, read:

1. `active/009-operator-brief.md`
2. `plans/008-verification-architecture-plan.md`
3. `completed/019-finished-analysis-and-landed-phases.md`
4. `active/005-thorough-vetting-log.md`

If working on architecture/verification strategy, read:

1. `concepts/010-schema-native-scenario-design.md`
2. `concepts/011-verification-streamlining-and-maps.md`
3. `concepts/012-architecture-streamlining-critique.md`
4. `completed/014-unifying-architecture-direction.md`
5. `completed/018-authored-scenario-roots.md`
6. `completed/019-finished-analysis-and-landed-phases.md`

## Current Notes

### Active

- `active/005-thorough-vetting-log.md`
- `active/009-operator-brief.md`

### Plans

- `plans/007-vetting-and-hardening-plan.md`
- `plans/008-verification-architecture-plan.md`
- `plans/017-module-layout-and-migration-for-unification.md`

### Completed

- `completed/013-existing-capabilities-and-xtask-lessons.md`
- `completed/014-unifying-architecture-direction.md`
- `completed/015-unifying-vertical-slice-plan.md`
- `completed/016-control-plane-compiler-architecture.md`
- `completed/018-authored-scenario-roots.md`
- `completed/019-finished-analysis-and-landed-phases.md`

### Concepts

- `concepts/002-scenario-driven-quality-frontier.md`
- `concepts/010-schema-native-scenario-design.md`
- `concepts/011-verification-streamlining-and-maps.md`
- `concepts/012-architecture-streamlining-critique.md`

### Handoff

- no dedicated live handoff note
- use `active/009-operator-brief.md` instead
