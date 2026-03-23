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

1. current planned execution program
2. most recent executed implementation programs
3. broader strategic references and next-frontier reservoir
4. audits and raw design inputs
5. historical closure material only if you are doing archaeology

## Canonical Current Entry Points

| Document | Role | Current status |
| --- | --- | --- |
| [programs/core-architecture-convergence-program-2026-03-23.md](./programs/core-architecture-convergence-program-2026-03-23.md) | Current planned execution program for collapsing the remaining parallel truth surfaces in query, storage, CLI front-door, showcase, schema, and package-root API shape | Planned; canonical next implementation campaign |
| [programs/runtime-contract-and-validation-lanes-program-2026-03-22.md](./programs/runtime-contract-and-validation-lanes-program-2026-03-22.md) | Most recent executed closure slice for the remaining machine/runtime/testing frontier | Executed; current canonical record of the runtime-contract and validation-lane closure |
| [programs/read-surface-proof-and-showcase-hardening-program-2026-03-22.md](./programs/read-surface-proof-and-showcase-hardening-program-2026-03-22.md) | Most recent executed proof program covering query summary/list surfaces, stream surfaces, MCP read payloads, and showcase hardening | Executed; current canonical record of the read-surface proof lane |
| [programs/multi-surface-semantic-proof-program-2026-03-22.md](./programs/multi-surface-semantic-proof-program-2026-03-22.md) | Executed semantic/export proof program extending proof from canonical markdown to broader export/query surfaces | Executed; predecessor proof-lane record |
| [programs/schema-package-authority-program-2026-03-22.md](./programs/schema-package-authority-program-2026-03-22.md) | Most recent major implementation program for finishing schema package/version authority | Executed; predecessor canonical record |
| [programs/semantic-proof-and-showcase-proof-lanes-program-2026-03-22.md](./programs/semantic-proof-and-showcase-proof-lanes-program-2026-03-22.md) | Executed semantic-preservation proof slice for canonical markdown plus proof-lane showcase coverage | Executed and retained as the current semantic-proof implementation record |
| [programs/intentional-forward-program-2026-03-21.md](./programs/intentional-forward-program-2026-03-21.md) | Executed umbrella program for the post-2026-03-19 planning wave | Executed umbrella; Step 6 split into the schema package authority program |
| [programs/artifact-cohort-control-plane-program-2026-03-21.md](./programs/artifact-cohort-control-plane-program-2026-03-21.md) | Executed subprogram covering durable artifact/cohort/proof surfaces | Executed and retained as concrete shape |
| [programs/publication-control-plane-program-2026-03-22.md](./programs/publication-control-plane-program-2026-03-22.md) | Executed subprogram covering typed site publication manifests and durable publication records | Executed and retained as the Step 5 publication slice |
| [programs/site-and-repo-shape-streamlining-program-2026-03-22.md](./programs/site-and-repo-shape-streamlining-program-2026-03-22.md) | Executed slice for finishing Step 5 through site decomposition and repo-shape slimming | Executed and retained as the site/repo-shape slice |

## Strategic Reference Programs

These are still important, but they are not the live queue.

| Document | Role | Current status |
| --- | --- | --- |
| [programs/canonical-archive-platform-program-2026-03-19.md](./programs/canonical-archive-platform-program-2026-03-19.md) | Broad north-star architecture program | Strategic reference |
| [programs/refactoring-first-streamlining-program-2026-03-19.md](./programs/refactoring-first-streamlining-program-2026-03-19.md) | Maximal simplification/refactoring reservoir | Strategic reference, much absorbed into the intentional-forward program |
| [programs/testing-reliability-expansion-program-2026-03-14.md](./programs/testing-reliability-expansion-program-2026-03-14.md) | Broad testing/showcase/runtime reliability program | Active backlog reservoir, not the live queue |
| [programs/artifact-and-semantic-proof-program-2026-03-19.md](./programs/artifact-and-semantic-proof-program-2026-03-19.md) | Narrower proof-oriented architecture program | Reference; artifact half executed, semantic-preservation half partly executed |
| [programs/artifact-and-semantic-proof-commit-plan-2026-03-19.md](./programs/artifact-and-semantic-proof-commit-plan-2026-03-19.md) | Concrete commit decomposition of the proof program | Historical execution slice/reference |

## Audits And Raw Design Inputs

These are inputs into planning, not the current queue themselves.

| Document | Role | Current status |
| --- | --- | --- |
| [analysis/2026-03-19-polylogue-architectural-anatomy-and-pathology-audit.md](./analysis/2026-03-19-polylogue-architectural-anatomy-and-pathology-audit.md) | Code-outward architectural audit | Reference input |
| [analysis/2026-03-19-testing-research-across-sinity-repos.md](./analysis/2026-03-19-testing-research-across-sinity-repos.md) | Cross-repo deep-research testing audit | Reference input |
| [analysis/test-ideas-dialogue.md](./analysis/test-ideas-dialogue.md) | Raw design dialogue covering testing/schema ambitions | Input transcript, not a direct plan |
| [analysis/testing-gaps-according-to-gemini-still-2.md](./analysis/testing-gaps-according-to-gemini-still-2.md) | Gap memo derived from external analysis | Input memo, not canonical backlog |

## Historical Closure And Recovery Docs

These are useful for archaeology and evidence, but they are not active backlog
authorities anymore.

| Document | Role |
| --- | --- |
| [archive/2026-03-05-07-closure-wave/remaining-workload-tracker-2026-03-05.md](./archive/2026-03-05-07-closure-wave/remaining-workload-tracker-2026-03-05.md) | Historical closure/backlog tracker from the schema-validation wave |
| [archive/2026-03-05-07-closure-wave/tasklist-master-2026-03-06.md](./archive/2026-03-05-07-closure-wave/tasklist-master-2026-03-06.md) | Compact closure checkpoint from the same wave |
| [archive/2026-03-05-07-closure-wave/workload-closure-2026-03-06.md](./archive/2026-03-05-07-closure-wave/workload-closure-2026-03-06.md) | Closure note |
| [archive/2026-03-05-07-closure-wave/workload-closure-2026-03-07.md](./archive/2026-03-05-07-closure-wave/workload-closure-2026-03-07.md) | Follow-up closure note |
| [archive/2026-03-05-07-closure-wave/session-recovery-2026-03-05.md](./archive/2026-03-05-07-closure-wave/session-recovery-2026-03-05.md) | Context-compaction recovery note |
| [archive/2026-03-05-07-closure-wave/workload-schema-qa-2026-03-05.md](./archive/2026-03-05-07-closure-wave/workload-schema-qa-2026-03-05.md) | Historical schema-QA workload note |
| [archive/2026-03-05-07-closure-wave/demo-parse-validate-audit-2026-03-05.md](./archive/2026-03-05-07-closure-wave/demo-parse-validate-audit-2026-03-05.md) | Historical parse/validate audit |
| [archive/2026-03-05-07-closure-wave/task22-test-audit-2026-03-05.md](./archive/2026-03-05-07-closure-wave/task22-test-audit-2026-03-05.md) | Historical test audit |
| [archive/2026-03-05-07-closure-wave/triage-comment-grouping-2026-03-07.md](./archive/2026-03-05-07-closure-wave/triage-comment-grouping-2026-03-07.md) | Historical triage note |
| [archive/2026-03-05-07-closure-wave/schema-composition-and-quarantine-report-2026-03-06.md](./archive/2026-03-05-07-closure-wave/schema-composition-and-quarantine-report-2026-03-06.md) | Historical schema report |

## Active Scratch Notes

Scratch notes are not the same as committed docs, but they currently contain
real design state and should be read intentionally rather than rediscovered.

Current active scratch set:

- `.claude/scratch/027-architecture-review-2026-03-23.md` - current code-outward architectural review feeding the convergence program
- `.claude/scratch/019-polylogue-architecture-audit-2026-03-20.md` - living local architecture audit with post-2026-03-21 addenda
- `.claude/scratch/018-wave0-schema-package-design.md` - implementation-facing schema package/version correction design
- `.claude/scratch/026-schema-taxonomy-and-versioning.md` - current schema taxonomy/versioning working note

Archived scratch references now include:

- `.claude/scratch/archive/2026-03-19-architecture-wave/012-cohesion-and-observability-impact.md`
- `.claude/scratch/archive/2026-03-19-architecture-wave/013-fluff-audit.md`
- `.claude/scratch/archive/2026-03-19-architecture-wave/014-radical-simplification-audit.md`
- `.claude/scratch/archive/2026-03-19-architecture-wave/015-repo-structure-overwhelm-audit.md`
- `.claude/scratch/archive/2026-03-19-architecture-wave/016-schema-centrality-vs-overreach.md`
- `.claude/scratch/archive/2026-03-19-architecture-wave/017-schema-state-audit.md`

Earlier scratch wave plans remain under `.claude/scratch/archive/`.

## Generated Validation Artifacts

These are outputs and evidence, not planning authorities:

- [mutation-campaigns/README.md](./mutation-campaigns/README.md)
- [benchmark-campaigns/README.md](./benchmark-campaigns/README.md)

## Current Open Frontier

As of this map:

- the next live implementation campaign is now [programs/core-architecture-convergence-program-2026-03-23.md](./programs/core-architecture-convergence-program-2026-03-23.md)
- that program specifically targets remaining architectural drag inside main code rather than a missing product surface:
  - query spec vs mutable filter execution
  - backend/query-store/repository overlap
  - Click front-door routing semantics
  - showcase/QA catalog and runner embedding
  - schema runtime vs tooling entanglement
  - over-broad package-root exports
- the publication-control-plane half of Step 5 from [programs/intentional-forward-program-2026-03-21.md](./programs/intentional-forward-program-2026-03-21.md) has been executed via [programs/publication-control-plane-program-2026-03-22.md](./programs/publication-control-plane-program-2026-03-22.md)
- the site/repo-shape half of Step 5 is now executed via [programs/site-and-repo-shape-streamlining-program-2026-03-22.md](./programs/site-and-repo-shape-streamlining-program-2026-03-22.md)
- schema package/version authority correction has now been executed via [programs/schema-package-authority-program-2026-03-22.md](./programs/schema-package-authority-program-2026-03-22.md)
- the older scratch notes `.claude/scratch/018-wave0-schema-package-design.md` and `.claude/scratch/026-schema-taxonomy-and-versioning.md` remain useful implementation references, but no longer serve as the live authority
- the next broad frontier is no longer schema authority; the canonical markdown semantic-proof slice has now been executed via [programs/semantic-proof-and-showcase-proof-lanes-program-2026-03-22.md](./programs/semantic-proof-and-showcase-proof-lanes-program-2026-03-22.md)
- the multi-surface proof lane is now executed via [programs/multi-surface-semantic-proof-program-2026-03-22.md](./programs/multi-surface-semantic-proof-program-2026-03-22.md), which extended semantic proof across export/query surfaces and routed that suite through QA/publication/showcase
- the read-surface proof and showcase-hardening lane is now executed via [programs/read-surface-proof-and-showcase-hardening-program-2026-03-22.md](./programs/read-surface-proof-and-showcase-hardening-program-2026-03-22.md), which covers query summary/list surfaces, stream surfaces, MCP read payloads, and matching showcase proof coverage
- the remaining runtime/testing reservoir has now been executed via [programs/runtime-contract-and-validation-lanes-program-2026-03-22.md](./programs/runtime-contract-and-validation-lanes-program-2026-03-22.md), which closes the root machine contract, query/TUI hardening, and named validation lanes for chaos/scale/live operator workflows
- the next frontier is therefore no longer the old testing-reliability backlog bucket; the live queue is now the convergence campaign above, and later work should be selected intentionally from the strategic-reference programs rather than assumed from unresolved reservoir notes

## Maintenance Rule

When adding a new planning or audit document, classify it explicitly as one of:

- current execution program
- executed subprogram
- strategic reference
- audit/input
- historical closure
- generated evidence

Then update this map so the repo keeps one intentional planning surface.
