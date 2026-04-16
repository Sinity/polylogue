---
created: "2026-04-13T16:40:00+02:00"
purpose: "Separate finished analysis and already-landed architectural phases from still-active scratch work"
status: "complete"
project: "polylogue"
---

# Finished Analysis And Landed Phases

This note exists to stop treating already-landed analysis as if it were still
open planning work.

The scratch tree had drifted into a mixed state:

- some notes are still active operating context
- some notes are execution plans that have already been substantially realized
- some notes are historical concept work that remains useful as rationale but
  should not be read as pending backlog

This file is the separation layer.

## Finished Enough To Treat As Historical

These notes describe work that has already materially landed and should now be
read as implementation history or design rationale, not as the next thing to do
by default.

### `completed/015-unifying-vertical-slice-plan.md`

Status:

- materially completed as a vertical-slice record
- now functions more as a landed-phase ledger than as an open plan

Why:

- it records implemented action-event and raw-state artifact-path unification
- it lists concrete runtime/storage/control-plane files where the slice landed
- the latest branch work has moved past the "should we prove this?" stage

Use it for:

- reconstructing what actually landed
- tracing where the unifying slice touched runtime and control-plane code

Do not use it as:

- the default "next work item" note

### `completed/018-authored-scenario-roots.md`

Status:

- historical rationale for already-landed scenario-root work

Why:

- the named scenario source root, inferred corpus scenario root, and executable
  authored scenario root are now present in code
- this note is still useful, but mostly as explanation of what was done

Use it for:

- understanding why `NamedScenarioSource`, `ExecutableScenario`, and
  `CorpusScenario` exist

### Earlier unification concept notes

These remain useful as design rationale, but they are no longer the sharpest
place to resume implementation from:

- `completed/013-existing-capabilities-and-xtask-lessons.md`
- `completed/014-unifying-architecture-direction.md`
- `completed/016-control-plane-compiler-architecture.md`

Why:

- their key claims have been partially absorbed into the current scenario,
  catalog, and execution-runtime substrate
- they still matter, but primarily as architectural memory

## Still Active

These notes still represent genuinely live work or current operator context.

### Runtime/product operating context

- `active/005-thorough-vetting-log.md`
- `active/009-operator-brief.md`

These are still live because they track:

- real archive behavior
- open operator defects
- drift that is not yet structurally resolved
- reset-safe branch context

### Live plans

- `plans/007-vetting-and-hardening-plan.md`
- `plans/008-verification-architecture-plan.md`
- `plans/017-module-layout-and-migration-for-unification.md`

These are still live because they contain real remaining work:

- runtime vetting and memory/perf follow-through
- broader verification architecture beyond the landed slices
- remaining module-placement and ownership decisions

### Still-live concepts

- `concepts/002-scenario-driven-quality-frontier.md`
- `concepts/010-schema-native-scenario-design.md`
- `concepts/011-verification-streamlining-and-maps.md`
- `concepts/012-architecture-streamlining-critique.md`

These are still live because they describe broader future-state questions that
have not yet been fully compiled into the repo.

## Practical Reading Order Now

If resuming concrete implementation work:

1. `active/009-operator-brief.md`
2. `plans/008-verification-architecture-plan.md`
3. this file

If reconstructing what already landed in the architecture/unification wave:

1. this file
2. `completed/015-unifying-vertical-slice-plan.md`
3. `completed/018-authored-scenario-roots.md`

## Repository-Analysis Correction

No, the previous state was not the best possible state of the scratch analyses.

The main weakness was:

- too much implemented work was still sitting inside "plans" and "concepts"
  without an explicit historical split

That made it harder to tell:

- what is still open
- what is rationale only
- what has already been compiled into code

This file is the first correction for that.
