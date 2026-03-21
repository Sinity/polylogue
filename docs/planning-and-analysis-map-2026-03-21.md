# Polylogue Planning And Analysis Map

Date: 2026-03-21
Status: current planning index

This document organizes the repo's planning, backlog, audit, and scratch-note
material into one readable map.

It exists because most planning state is in docs rather than inline code TODOs.
A repository-wide scan found little meaningful inline TODO/FIXME debt in main
code; the real backlog/navigation problem is document sprawl.

## How To Use This Map

Read in this order:

1. current execution program
2. current executed subprograms
3. broader strategic references
4. audits and raw design inputs
5. historical closure material only if you are doing archaeology

## Canonical Current Entry Points

| Document | Role | Current status |
| --- | --- | --- |
| [intentional-forward-program-2026-03-21.md](./intentional-forward-program-2026-03-21.md) | Live execution queue for the post-2026-03-19 planning wave | Current canonical program |
| [artifact-cohort-control-plane-program-2026-03-21.md](./artifact-cohort-control-plane-program-2026-03-21.md) | Executed subprogram covering durable artifact/cohort/proof surfaces | Executed and retained as concrete shape |

## Strategic Reference Programs

These are still important, but they are not the live queue.

| Document | Role | Current status |
| --- | --- | --- |
| [canonical-archive-platform-program-2026-03-19.md](./canonical-archive-platform-program-2026-03-19.md) | Broad north-star architecture program | Strategic reference |
| [refactoring-first-streamlining-program-2026-03-19.md](./refactoring-first-streamlining-program-2026-03-19.md) | Maximal simplification/refactoring reservoir | Strategic reference, much absorbed into the intentional-forward program |
| [testing-reliability-expansion-program-2026-03-14.md](./testing-reliability-expansion-program-2026-03-14.md) | Broad testing/showcase/runtime reliability program | Active backlog reservoir, not the live queue |
| [artifact-and-semantic-proof-program-2026-03-19.md](./artifact-and-semantic-proof-program-2026-03-19.md) | Narrower proof-oriented architecture program | Reference; artifact half partly executed, semantic-preservation half still future |
| [artifact-and-semantic-proof-commit-plan-2026-03-19.md](./artifact-and-semantic-proof-commit-plan-2026-03-19.md) | Concrete commit decomposition of the proof program | Historical execution slice/reference |

## Audits And Raw Design Inputs

These are inputs into planning, not the current queue themselves.

| Document | Role | Current status |
| --- | --- | --- |
| [2026-03-19-polylogue-architectural-anatomy-and-pathology-audit.md](./2026-03-19-polylogue-architectural-anatomy-and-pathology-audit.md) | Code-outward architectural audit | Reference input |
| [2026-03-19-testing-research-across-sinity-repos.md](./2026-03-19-testing-research-across-sinity-repos.md) | Cross-repo deep-research testing audit | Reference input |
| [test-ideas-dialogue.md](./test-ideas-dialogue.md) | Raw design dialogue covering testing/schema ambitions | Input transcript, not a direct plan |
| [testing-gaps-according-to-gemini-still-2.md](./testing-gaps-according-to-gemini-still-2.md) | Gap memo derived from external analysis | Input memo, not canonical backlog |

## Historical Closure And Recovery Docs

These are useful for archaeology and evidence, but they are not active backlog
authorities anymore.

| Document | Role |
| --- | --- |
| [remaining-workload-tracker-2026-03-05.md](./remaining-workload-tracker-2026-03-05.md) | Historical closure/backlog tracker from the schema-validation wave |
| [tasklist-master-2026-03-06.md](./tasklist-master-2026-03-06.md) | Compact closure checkpoint from the same wave |
| [workload-closure-2026-03-06.md](./workload-closure-2026-03-06.md) | Closure note |
| [workload-closure-2026-03-07.md](./workload-closure-2026-03-07.md) | Follow-up closure note |
| [session-recovery-2026-03-05.md](./session-recovery-2026-03-05.md) | Context-compaction recovery note |
| [workload-schema-qa-2026-03-05.md](./workload-schema-qa-2026-03-05.md) | Historical schema-QA workload note |
| [demo-parse-validate-audit-2026-03-05.md](./demo-parse-validate-audit-2026-03-05.md) | Historical parse/validate audit |
| [task22-test-audit-2026-03-05.md](./task22-test-audit-2026-03-05.md) | Historical test audit |
| [triage-comment-grouping-2026-03-07.md](./triage-comment-grouping-2026-03-07.md) | Historical triage note |
| [schema-composition-and-quarantine-report-2026-03-06.md](./schema-composition-and-quarantine-report-2026-03-06.md) | Historical schema report |

## Active Scratch Notes

Scratch notes are not the same as committed docs, but they currently contain
real design state and should be read intentionally rather than rediscovered.

Current active or still-live scratch set:

- `.claude/scratch/019-polylogue-architecture-audit-2026-03-20.md` - living local architecture audit with post-2026-03-21 addenda
- `.claude/scratch/018-wave0-schema-package-design.md` - implementation-facing schema package/version correction design
- `.claude/scratch/026-schema-taxonomy-and-versioning.md` - current schema taxonomy/versioning working note
- `.claude/scratch/013-fluff-audit.md` - codebase and plan anti-fluff criteria
- `.claude/scratch/012-cohesion-and-observability-impact.md` - cohesion/LOC impact estimate for the broader platform program

Useful completed scratch references:

- `.claude/scratch/017-schema-state-audit.md`
- `.claude/scratch/016-schema-centrality-vs-overreach.md`
- `.claude/scratch/015-repo-structure-overwhelm-audit.md`
- `.claude/scratch/014-radical-simplification-audit.md`

Archived scratch wave plans remain under `.claude/scratch/archive/`.

## Generated Validation Artifacts

These are outputs and evidence, not planning authorities:

- [mutation-campaigns/README.md](./mutation-campaigns/README.md)
- [benchmark-campaigns/README.md](./benchmark-campaigns/README.md)

## Current Open Frontier

As of this map:

- the current open frontier in the canonical program is Step 5 from [intentional-forward-program-2026-03-21.md](./intentional-forward-program-2026-03-21.md): publication and repo-shape cleanup
- the main still-active adjacent design lane not yet fully absorbed into a committed canonical program is schema package/version correction, currently best captured by `.claude/scratch/018-wave0-schema-package-design.md` and `.claude/scratch/026-schema-taxonomy-and-versioning.md`
- testing/showcase expansion remains a backlog reservoir in [testing-reliability-expansion-program-2026-03-14.md](./testing-reliability-expansion-program-2026-03-14.md), not the immediate execution queue

## Maintenance Rule

When adding a new planning or audit document, classify it explicitly as one of:

- current execution program
- executed subprogram
- strategic reference
- audit/input
- historical closure
- generated evidence

Then update this map so the repo keeps one intentional planning surface.
