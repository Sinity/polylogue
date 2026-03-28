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

1. most recent executed implementation program
2. current executed subprograms
3. broader strategic references and next-frontier reservoir
4. audits and raw design inputs
5. historical closure material only if you are doing archaeology

## Canonical Current Entry Points

| Document | Role | Current status |
| --- | --- | --- |
<<<<<<< HEAD
||||||| parent of ef1c8d35 (docs: close codebase-wide cleanup program)
| [programs/codebase-wide-topology-and-debt-retirement-program-2026-03-26.md](./programs/codebase-wide-topology-and-debt-retirement-program-2026-03-26.md) | Current whole-repository cleanup/refactoring program with explicit wave-based execution order and parallel workstreams across rendering, CLI/UI/sync, MCP/operations, storage, schema, sources, lib/domain, pipeline/showcase/site, and cross-cutting package-root, test, devtools, and documentation topology | Active; this is now the live whole-repository cleanup queue rather than a hotspot slice |
| [programs/rendering-operator-provider-and-runtime-topology-cleanup-program-2026-03-26.md](./programs/rendering-operator-provider-and-runtime-topology-cleanup-program-2026-03-26.md) | Absorbed predecessor cleanup/refactoring program for rendering/semantic-proof, operator adapters, storage write/rebuild bands, schema runtime analysis, provider/source boundaries, and domain/query-evidence topology narrowing | Absorbed predecessor; broadened into the codebase-wide topology/debt retirement queue |
| [programs/deep-query-service-and-schema-topology-cleanup-program-2026-03-26.md](./programs/deep-query-service-and-schema-topology-cleanup-program-2026-03-26.md) | Executed cleanup/refactoring program for query-engine, operator-command, repository/repair/build, raw-payload/pipeline, schema-toolchain, and sqlite-vec topology narrowing | Executed; query/service/schema/public roots were narrowed substantially, the named runtime-substrate contracts lane passed, and live archive debt/product status remained clean |
| [programs/product-and-runtime-topology-cleanup-program-2026-03-26.md](./programs/product-and-runtime-topology-cleanup-program-2026-03-26.md) | Executed cleanup/refactoring program for archive-product, health/debt, query-SQL, validation-lane, and declarative-root topology narrowing | Executed; archive-product/session-product/health/debt/template/validation roots were narrowed substantially, the named stewardship lane passed, and live archive debt remains clean |
| [programs/probabilistic-enrichment-and-governed-cleanup-program-2026-03-26.md](./programs/probabilistic-enrichment-and-governed-cleanup-program-2026-03-26.md) | Executed convergence program for heuristic-inference hardening, durable enrichment products, retrieval-band rollout, consumer/operator convergence, and governed live cleanup | Executed; enrichment products, retrieval-enrichment health, live governed cleanup apply/validation lineage, and named enrichment/cleanup lanes are now in place |
| [programs/evidence-and-stewardship-platform-convergence-program-2026-03-24.md](./programs/evidence-and-stewardship-platform-convergence-program-2026-03-24.md) | Executed convergence program for evidence-tier contracts, inference-tier governance, retrieval/embedding alignment, consumer contract convergence, and live semantic stewardship | Executed; durable evidence/inference product contracts, tiered profile retrieval, retrieval-band health, live migration compatibility, and named validation lanes are now in place |
| [programs/cleanup-and-architectural-debt-retirement-program-2026-03-24.md](./programs/cleanup-and-architectural-debt-retirement-program-2026-03-24.md) | Executed cleanup-only architectural debt-retirement record for mixed-role file breakup, ownership narrowing, public-root reduction, and deletion of structural overlap | Executed; high-leverage public roots, semantic runtime bands, search/runtime roots, schema/operator surfaces, and QA/reporting roots were narrowed or deleted |
| [programs/consumer-contracts-and-governed-live-cleanup-program-2026-03-24.md](./programs/consumer-contracts-and-governed-live-cleanup-program-2026-03-24.md) | Broader consumer/governance predecessor covering durable product contracts, governed destructive cleanup, stewardship history, and live archive governance | Absorbed predecessor; replaced as the live queue by the evidence/stewardship platform campaign |
| [programs/semantic-product-normalization-and-toolchain-convergence-program-2026-03-24.md](./programs/semantic-product-normalization-and-toolchain-convergence-program-2026-03-24.md) | Executed convergence program for semantic/session product normalization, operator/toolchain narrowing, schema convergence, and live cleanup governance | Executed; canonical normalization, session-product/toolchain decomposition, archive-debt governance stages, and semantic-product validation lanes are now in place |
| [programs/domain-read-model-and-live-archive-stewardship-program-2026-03-24.md](./programs/domain-read-model-and-live-archive-stewardship-program-2026-03-24.md) | Executed convergence program for domain-model decomposition, repository-read/product-query convergence, external consumer contracts, and live archive stewardship | Executed; domain models, repository reads, session-product query bands, archive-debt governance, provider analytics/debt products, and live stewardship lanes are now in place |
| [programs/runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md](./programs/runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md) | Executed runtime-substrate campaign for query/runtime decomposition, storage/module narrowing, contract hardening, and validation governance | Executed; query/runtime substrate decomposition, lifecycle/repair/store narrowing, semantic registry narrowing, backend/schema banding, MCP tool split, and named runtime-substrate lanes are now in place |
| [programs/archive-data-products-and-live-governance-program-2026-03-24.md](./programs/archive-data-products-and-live-governance-program-2026-03-24.md) | Executed convergence program for durable semantic/session products, external consumer contracts, cleanup lineage, retrieval/product convergence, and live governance lanes | Executed; durable session/work/tag/day/week products, public product contracts, live status lanes, and cleanup preview lineage are now in place |
| [programs/source-boundary-and-runtime-governance-program-2026-03-23.md](./programs/source-boundary-and-runtime-governance-program-2026-03-23.md) | Executed convergence program for source/provider boundary cleanup, runtime maintenance governance, derived-model freshness/provenance convergence, publication/runtime narrowing, and live-archive validation | Executed; source traversal split, typed maintenance control plane, publication maintenance summaries, and live governance lanes are now in place |
| [programs/platform-wide-architecture-and-refactoring-program-2026-03-23.md](./programs/platform-wide-architecture-and-refactoring-program-2026-03-23.md) | Executed implementation record for repository-wide architecture reduction, refactoring, module-topology cleanup, semantic-product consolidation, and lifecycle/health/repair hardening | Executed; query/filter shell convergence, backend narrowing, shared query construction, semantic-product consolidation, root narrowing, and live-maintenance hardening are now in place |
| [programs/archive-intelligence-platform-convergence-program-2026-03-23.md](./programs/archive-intelligence-platform-convergence-program-2026-03-23.md) | Executed convergence program for action/event semantics, retrieval/FTS/embedding control-plane work, unresolved runtime-state/schema/operator slices, and archive-scale validation | Executed; durable action-event read model, readiness-aware query fallback, retrieval-health surfaces, and archive-intelligence validation lanes are now in place |
| [programs/state-and-schema-platform-convergence-program-2026-03-23.md](./programs/state-and-schema-platform-convergence-program-2026-03-23.md) | Absorbed predecessor program for the runtime-state/schema half of the new integrated campaign | Planned reference; no longer the live queue on its own |
| [programs/semantic-stack-convergence-program-2026-03-23.md](./programs/semantic-stack-convergence-program-2026-03-23.md) | Most recent executed convergence program for harmonization, canonical semantic facts, downstream semantic products, and semantic proof/export contract convergence | Executed; current canonical record of the semantic-stack convergence campaign |
| [programs/schema-and-evidence-pipeline-convergence-program-2026-03-23.md](./programs/schema-and-evidence-pipeline-convergence-program-2026-03-23.md) | Executed convergence program for schema tooling, synthetic generation, raw-to-record preparation, proof workflows, and the evidence write path around the canonical package model | Executed; predecessor evidence-pipeline convergence campaign |
| [programs/core-architecture-convergence-program-2026-03-23.md](./programs/core-architecture-convergence-program-2026-03-23.md) | Executed convergence campaign for collapsing the remaining parallel truth surfaces in query, storage, CLI front-door, showcase, schema, and package-root API shape | Executed; canonical record of the convergence campaign |
=======
| [programs/codebase-wide-topology-and-debt-retirement-program-2026-03-26.md](./programs/codebase-wide-topology-and-debt-retirement-program-2026-03-26.md) | Executed whole-repository cleanup/refactoring record with explicit wave-based execution order, parallel workstreams, and final closure evidence across rendering, CLI/UI/sync, MCP/operations, storage, schema, sources, lib/domain, pipeline/showcase/site, and cross-cutting package-root, test, devtools, and documentation topology | Executed; the full hotspot wave is closed, the broad regression slice passed, named composite/live lanes passed, and representative live memory-budget proofs stayed within budget |
| [programs/rendering-operator-provider-and-runtime-topology-cleanup-program-2026-03-26.md](./programs/rendering-operator-provider-and-runtime-topology-cleanup-program-2026-03-26.md) | Absorbed predecessor cleanup/refactoring program for rendering/semantic-proof, operator adapters, storage write/rebuild bands, schema runtime analysis, provider/source boundaries, and domain/query-evidence topology narrowing | Absorbed predecessor; broadened into the codebase-wide topology/debt retirement queue |
| [programs/deep-query-service-and-schema-topology-cleanup-program-2026-03-26.md](./programs/deep-query-service-and-schema-topology-cleanup-program-2026-03-26.md) | Executed cleanup/refactoring program for query-engine, operator-command, repository/repair/build, raw-payload/pipeline, schema-toolchain, and sqlite-vec topology narrowing | Executed; query/service/schema/public roots were narrowed substantially, the named runtime-substrate contracts lane passed, and live archive debt/product status remained clean |
| [programs/product-and-runtime-topology-cleanup-program-2026-03-26.md](./programs/product-and-runtime-topology-cleanup-program-2026-03-26.md) | Executed cleanup/refactoring program for archive-product, health/debt, query-SQL, validation-lane, and declarative-root topology narrowing | Executed; archive-product/session-product/health/debt/template/validation roots were narrowed substantially, the named stewardship lane passed, and live archive debt remains clean |
| [programs/probabilistic-enrichment-and-governed-cleanup-program-2026-03-26.md](./programs/probabilistic-enrichment-and-governed-cleanup-program-2026-03-26.md) | Executed convergence program for heuristic-inference hardening, durable enrichment products, retrieval-band rollout, consumer/operator convergence, and governed live cleanup | Executed; enrichment products, retrieval-enrichment health, live governed cleanup apply/validation lineage, and named enrichment/cleanup lanes are now in place |
| [programs/evidence-and-stewardship-platform-convergence-program-2026-03-24.md](./programs/evidence-and-stewardship-platform-convergence-program-2026-03-24.md) | Executed convergence program for evidence-tier contracts, inference-tier governance, retrieval/embedding alignment, consumer contract convergence, and live semantic stewardship | Executed; durable evidence/inference product contracts, tiered profile retrieval, retrieval-band health, live migration compatibility, and named validation lanes are now in place |
| [programs/cleanup-and-architectural-debt-retirement-program-2026-03-24.md](./programs/cleanup-and-architectural-debt-retirement-program-2026-03-24.md) | Executed cleanup-only architectural debt-retirement record for mixed-role file breakup, ownership narrowing, public-root reduction, and deletion of structural overlap | Executed; high-leverage public roots, semantic runtime bands, search/runtime roots, schema/operator surfaces, and QA/reporting roots were narrowed or deleted |
| [programs/consumer-contracts-and-governed-live-cleanup-program-2026-03-24.md](./programs/consumer-contracts-and-governed-live-cleanup-program-2026-03-24.md) | Broader consumer/governance predecessor covering durable product contracts, governed destructive cleanup, stewardship history, and live archive governance | Absorbed predecessor; replaced as the live queue by the evidence/stewardship platform campaign |
| [programs/semantic-product-normalization-and-toolchain-convergence-program-2026-03-24.md](./programs/semantic-product-normalization-and-toolchain-convergence-program-2026-03-24.md) | Executed convergence program for semantic/session product normalization, operator/toolchain narrowing, schema convergence, and live cleanup governance | Executed; canonical normalization, session-product/toolchain decomposition, archive-debt governance stages, and semantic-product validation lanes are now in place |
| [programs/domain-read-model-and-live-archive-stewardship-program-2026-03-24.md](./programs/domain-read-model-and-live-archive-stewardship-program-2026-03-24.md) | Executed convergence program for domain-model decomposition, repository-read/product-query convergence, external consumer contracts, and live archive stewardship | Executed; domain models, repository reads, session-product query bands, archive-debt governance, provider analytics/debt products, and live stewardship lanes are now in place |
| [programs/runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md](./programs/runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md) | Executed runtime-substrate campaign for query/runtime decomposition, storage/module narrowing, contract hardening, and validation governance | Executed; query/runtime substrate decomposition, lifecycle/repair/store narrowing, semantic registry narrowing, backend/schema banding, MCP tool split, and named runtime-substrate lanes are now in place |
| [programs/archive-data-products-and-live-governance-program-2026-03-24.md](./programs/archive-data-products-and-live-governance-program-2026-03-24.md) | Executed convergence program for durable semantic/session products, external consumer contracts, cleanup lineage, retrieval/product convergence, and live governance lanes | Executed; durable session/work/tag/day/week products, public product contracts, live status lanes, and cleanup preview lineage are now in place |
| [programs/source-boundary-and-runtime-governance-program-2026-03-23.md](./programs/source-boundary-and-runtime-governance-program-2026-03-23.md) | Executed convergence program for source/provider boundary cleanup, runtime maintenance governance, derived-model freshness/provenance convergence, publication/runtime narrowing, and live-archive validation | Executed; source traversal split, typed maintenance control plane, publication maintenance summaries, and live governance lanes are now in place |
| [programs/platform-wide-architecture-and-refactoring-program-2026-03-23.md](./programs/platform-wide-architecture-and-refactoring-program-2026-03-23.md) | Executed implementation record for repository-wide architecture reduction, refactoring, module-topology cleanup, semantic-product consolidation, and lifecycle/health/repair hardening | Executed; query/filter shell convergence, backend narrowing, shared query construction, semantic-product consolidation, root narrowing, and live-maintenance hardening are now in place |
| [programs/archive-intelligence-platform-convergence-program-2026-03-23.md](./programs/archive-intelligence-platform-convergence-program-2026-03-23.md) | Executed convergence program for action/event semantics, retrieval/FTS/embedding control-plane work, unresolved runtime-state/schema/operator slices, and archive-scale validation | Executed; durable action-event read model, readiness-aware query fallback, retrieval-health surfaces, and archive-intelligence validation lanes are now in place |
| [programs/state-and-schema-platform-convergence-program-2026-03-23.md](./programs/state-and-schema-platform-convergence-program-2026-03-23.md) | Absorbed predecessor program for the runtime-state/schema half of the new integrated campaign | Planned reference; no longer the live queue on its own |
| [programs/semantic-stack-convergence-program-2026-03-23.md](./programs/semantic-stack-convergence-program-2026-03-23.md) | Most recent executed convergence program for harmonization, canonical semantic facts, downstream semantic products, and semantic proof/export contract convergence | Executed; current canonical record of the semantic-stack convergence campaign |
| [programs/schema-and-evidence-pipeline-convergence-program-2026-03-23.md](./programs/schema-and-evidence-pipeline-convergence-program-2026-03-23.md) | Executed convergence program for schema tooling, synthetic generation, raw-to-record preparation, proof workflows, and the evidence write path around the canonical package model | Executed; predecessor evidence-pipeline convergence campaign |
| [programs/core-architecture-convergence-program-2026-03-23.md](./programs/core-architecture-convergence-program-2026-03-23.md) | Executed convergence campaign for collapsing the remaining parallel truth surfaces in query, storage, CLI front-door, showcase, schema, and package-root API shape | Executed; canonical record of the convergence campaign |
>>>>>>> ef1c8d35 (docs: close codebase-wide cleanup program)
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

<<<<<<< HEAD
||||||| parent of ef1c8d35 (docs: close codebase-wide cleanup program)
- the current live implementation queue is now [programs/codebase-wide-topology-and-debt-retirement-program-2026-03-26.md](./programs/codebase-wide-topology-and-debt-retirement-program-2026-03-26.md)
- it is explicitly the whole-repository cleanup authority now, not just a broad hotspot queue:
  - every top-level runtime package is in scope
  - MCP and operations consumer surfaces are in scope
  - package-root import/re-export cleanup is in scope
  - tests, devtools, memory-budget probes, validation lanes, and operator docs are in scope
- the narrower queue in [programs/rendering-operator-provider-and-runtime-topology-cleanup-program-2026-03-26.md](./programs/rendering-operator-provider-and-runtime-topology-cleanup-program-2026-03-26.md) has been absorbed because it was still a frontier slice rather than a whole-codebase authority
- the just-executed cleanup program in [programs/deep-query-service-and-schema-topology-cleanup-program-2026-03-26.md](./programs/deep-query-service-and-schema-topology-cleanup-program-2026-03-26.md) closed the main structural drag around:
  - broad query planning/spec/retrieval roots
  - broad products/run/QA/check command-family roots
  - broad repository-archive, repair, builder, mapper, and store-core roots
  - broad raw-payload, validation, acquisition, and semantic service roots
  - broad schema toolchain roots for generation, roundtrip, audit, and sampling
  - the oversized sqlite-vec provider root
- the older cleanup program in [programs/product-and-runtime-topology-cleanup-program-2026-03-26.md](./programs/product-and-runtime-topology-cleanup-program-2026-03-26.md) closed the preceding structural drag around:
  - broad archive-product contract, mapper, and operator-support roots
  - broad session-product row/store/status roots
  - broad health/debt/MCP/product workflow and rendering roots
  - broad conversation/raw SQL roots
  - the oversized validation-lane registry
  - the remaining mixed declarative site template root
- the just-executed probabilistic/governed-cleanup program in [programs/probabilistic-enrichment-and-governed-cleanup-program-2026-03-26.md](./programs/probabilistic-enrichment-and-governed-cleanup-program-2026-03-26.md) closed the main semantic-governance gap around:
  - explicit support metadata for heuristic session products
  - durable probabilistic enrichment products queryable from CLI, library, sync, repository, and MCP
  - enrichment retrieval-band readiness and `embed --stats` health exposure
  - live governed cleanup apply plus validation lineage for orphaned content blocks and orphaned attachments
  - named local/live validation lanes for heuristic inference, enrichment contracts, and cleanup governance
- the live archive is now materially cleaner:
  - `orphaned_content_blocks=0`
  - `orphaned_attachments=0`
  - `products debt --json` reports zero actionable debt
  - all durable session-product and retrieval-enrichment bands are ready, while transcript embeddings remain intentionally pending
- the next cleanup drag is now explicitly codebase-wide:
  - rendering/proof/output roots across `rendering/`, `cli/`, and renderer-facing helpers
  - operator/control-plane roots across CLI, MCP, UI, sync, and machine/plain adapters
  - operations/archive-support roots consumed by public product/query surfaces
  - storage write/rebuild/status/DDL/query/search roots across repository, backend, lifecycle, and provider bands
  - schema runtime/tooling/verification/synthetic roots across the full `schemas/` subsystem
  - source/provider/parser/decode/Drive roots across the full `sources/` subsystem
  - domain/evidence/query model roots across `lib/` and archive-product entity contracts
  - pipeline/showcase/site/UI roots that still carry broad orchestration or page/report family logic
  - cross-cutting root/package/test/devtools/docs topology that must move with the above cleanup rather than lag behind it
- the cleanup-only program in [programs/cleanup-and-architectural-debt-retirement-program-2026-03-24.md](./programs/cleanup-and-architectural-debt-retirement-program-2026-03-24.md) is now executed
- it closed the main structural debt around:
  - broad query-execution and query-store ownership
  - broad semantic runtime roots for facts, profiles, phases, work events, and decisions
  - broad schema/operator roots for code detection, semantic inference, runtime schema authority, and schema command rendering
  - broad QA/reporting and search/runtime public roots
  - broad async-backend, lifecycle, facade, and semantic-surface declaration roots
- the absorbed consumer/governance predecessor remains [programs/consumer-contracts-and-governed-live-cleanup-program-2026-03-24.md](./programs/consumer-contracts-and-governed-live-cleanup-program-2026-03-24.md)
- the just-executed semantic-product campaign in [programs/semantic-product-normalization-and-toolchain-convergence-program-2026-03-24.md](./programs/semantic-product-normalization-and-toolchain-convergence-program-2026-03-24.md) is now executed
- it closed the main semantic/toolchain drag around:
  - noisy project/repo canonicalization leaking into durable tag/day/week outputs
  - overbroad session-product support/status ownership
  - broad archive-product operator reshaping and missing debt governance stages
  - mixed semantic proof catalog declarations and schema-tooling role bands
  - broad Claude/Drive parser seams that fed late-stage cleanup
  - missing campaign-specific validation lanes for semantic products and live governance
- the domain read-model campaign in [programs/domain-read-model-and-live-archive-stewardship-program-2026-03-24.md](./programs/domain-read-model-and-live-archive-stewardship-program-2026-03-24.md) is now executed
- it closed the broad read-model and stewardship gap around:
  - `lib/models.py` decomposition into conversation/message/attachment/model-support bands
  - repository read banding across archive/action/maintenance/product surfaces
  - session-product SQL/query decomposition into profile/timeline/thread/summary bands
  - archive-debt governance shared by health, maintenance preview, and product status
  - stable provider analytics and archive-debt products across CLI/library/sync/MCP
  - named live stewardship validation lanes exercised against the real archive
- the runtime-substrate campaign in [programs/runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md](./programs/runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md) is now executed
- it closed the main runtime-substrate drag around:
  - query planning/retrieval/grouping/output monoliths
  - session-product lifecycle and maintenance control-plane monoliths
  - broad store and backend/schema ownership
  - semantic surface registry execution vs declaration ownership
  - broad MCP tool registration wiring
  - missing campaign-specific validation lanes and live memory-governance proofs
- the next broad frontier is now higher-level and consumer-facing:
  - domain model decomposition around the remaining broad `lib/models.py` root
  - repository-read and session-product query banding
  - external consumer and analytics contract convergence
  - governed live archive stewardship for explicit cleanup debt such as orphaned content blocks
  - tighter product/retrieval/analytics convergence on durable read models
- the just-executed archive-data-products campaign in [programs/archive-data-products-and-live-governance-program-2026-03-24.md](./programs/archive-data-products-and-live-governance-program-2026-03-24.md) closed the broad product/governance gap around:
  - durable session/work/tag/day/week archive products
  - public consumer contracts across library, CLI, MCP, and sync surfaces
  - typed maintenance lineage and machine-readable product status
  - live archive session-product rebuilds and validation lanes
  - governed cleanup preview for live orphaned content-block debt
- the next integrated move is substrate/refactoring heavy:
  - decompose the remaining broad query, lifecycle, maintenance, backend, schema, and operator modules
  - harden public contracts so they compose smaller subsystems instead of broad omnibus files
  - keep memory/live-governance validation attached to those smaller runtime surfaces
- the source-boundary/runtime-governance campaign in [programs/source-boundary-and-runtime-governance-program-2026-03-23.md](./programs/source-boundary-and-runtime-governance-program-2026-03-23.md) is now executed
- it closed the source/runtime-governance drag around:
  - historical `sources/source.py` / `sources/drive_client.py` umbrella ownership
  - typed maintenance selection and explicit safe-vs-destructive operator semantics
  - canonical derived-model provenance/freshness reporting across health, publication, and machine output
  - clearer cached-vs-live operator output semantics
  - committed local/live governance lanes for source/provider fidelity and maintenance preview
- the platform-wide refactoring campaign in [programs/platform-wide-architecture-and-refactoring-program-2026-03-23.md](./programs/platform-wide-architecture-and-refactoring-program-2026-03-23.md) is now executed
- it closed the broad architectural drag around:
  - query/filter shell convergence onto immutable plans
  - backend/query/lifecycle narrowing and removal of leftover backend read wrappers
  - shared query-spec construction across CLI and MCP
  - consolidation of semantic session/work products into explicit builders
  - narrower package-root and synthetic-stack exports
  - live-maintenance hardening for action-event readiness, repair, and archive health
- the archive-intelligence campaign in [programs/archive-intelligence-platform-convergence-program-2026-03-23.md](./programs/archive-intelligence-platform-convergence-program-2026-03-23.md) is now executed
- it closed the dogfooded archive-intelligence frontier around:
  - first-class durable action events
  - action-aware retrieval and grouped stats
  - readiness-aware fallback when persisted action semantics are absent or stale
  - freshness/provenance-aware embedding retrieval health
  - named archive-intelligence validation and memory-budget lanes
- the semantic-stack convergence program is now executed via [programs/semantic-stack-convergence-program-2026-03-23.md](./programs/semantic-stack-convergence-program-2026-03-23.md)
- it closed the strongest remaining semantic-cluster drag around:
  - harmonization boundary cleanup around `schemas/unified.py` and `lib/provider_semantics.py`
  - a missing canonical semantic facts layer shared by profiles, tags, and proof
  - downstream semantic products that still composed several separate extraction passes
  - a large semantic proof stack that mixed facts, surface policy, and suite orchestration
  - semantic preservation/loss contracts that were not previously declared once in one registry
- the schema-and-evidence pipeline campaign is now executed via [programs/schema-and-evidence-pipeline-convergence-program-2026-03-23.md](./programs/schema-and-evidence-pipeline-convergence-program-2026-03-23.md)
- it closed the remaining evidence-pipeline drag around:
  - runtime-safe schema observation helpers living too close to tooling
  - monolithic schema generation orchestration
  - monolithic synthetic corpus generation
  - overbroad prepare/source transformation boundaries
  - fragmented artifact proof and verification workflows
  - an overbroad async SQLite evidence/write path
  - a missing named roundtrip proof lane for the full schema-and-evidence loop
- the convergence campaign in [programs/core-architecture-convergence-program-2026-03-23.md](./programs/core-architecture-convergence-program-2026-03-23.md) is now executed
- it closed the previously identified main-code drag around:
  - query spec vs mutable filter execution
  - backend/query-store/repository overlap
  - Click front-door routing semantics
  - showcase/QA catalog and runner embedding
  - schema runtime vs tooling entanglement
  - over-broad package-root exports
=======
- there is no newer cleanup/refactoring queue yet; the latest whole-repository cleanup authority is now the executed record [programs/codebase-wide-topology-and-debt-retirement-program-2026-03-26.md](./programs/codebase-wide-topology-and-debt-retirement-program-2026-03-26.md)
- that record is now closed with whole-repository proof rather than left as an open queue:
  - every top-level runtime package was included in scope
  - MCP and operations consumer surfaces were included in scope
  - package-root import/re-export cleanup was included in scope
  - tests, devtools, memory-budget probes, validation lanes, and operator docs moved with the code
  - the final Python hotspot scan is `count 0` at `>=250` lines
  - the final broad targeted regression slice passed: `952 passed`
  - the named composite lane `runtime-substrate-contracts` passed
  - the named live lane `evidence-stewardship-live` passed
  - the representative live memory proofs passed under budget
- the narrower queue in [programs/rendering-operator-provider-and-runtime-topology-cleanup-program-2026-03-26.md](./programs/rendering-operator-provider-and-runtime-topology-cleanup-program-2026-03-26.md) has been absorbed because it was still a frontier slice rather than a whole-codebase authority
- the just-executed cleanup program in [programs/deep-query-service-and-schema-topology-cleanup-program-2026-03-26.md](./programs/deep-query-service-and-schema-topology-cleanup-program-2026-03-26.md) closed the main structural drag around:
  - broad query planning/spec/retrieval roots
  - broad products/run/QA/check command-family roots
  - broad repository-archive, repair, builder, mapper, and store-core roots
  - broad raw-payload, validation, acquisition, and semantic service roots
  - broad schema toolchain roots for generation, roundtrip, audit, and sampling
  - the oversized sqlite-vec provider root
- the older cleanup program in [programs/product-and-runtime-topology-cleanup-program-2026-03-26.md](./programs/product-and-runtime-topology-cleanup-program-2026-03-26.md) closed the preceding structural drag around:
  - broad archive-product contract, mapper, and operator-support roots
  - broad session-product row/store/status roots
  - broad health/debt/MCP/product workflow and rendering roots
  - broad conversation/raw SQL roots
  - the oversized validation-lane registry
  - the remaining mixed declarative site template root
- the just-executed probabilistic/governed-cleanup program in [programs/probabilistic-enrichment-and-governed-cleanup-program-2026-03-26.md](./programs/probabilistic-enrichment-and-governed-cleanup-program-2026-03-26.md) closed the main semantic-governance gap around:
  - explicit support metadata for heuristic session products
  - durable probabilistic enrichment products queryable from CLI, library, sync, repository, and MCP
  - enrichment retrieval-band readiness and `embed --stats` health exposure
  - live governed cleanup apply plus validation lineage for orphaned content blocks and orphaned attachments
  - named local/live validation lanes for heuristic inference, enrichment contracts, and cleanup governance
- the live archive is now materially cleaner:
  - `orphaned_content_blocks=0`
  - `orphaned_attachments=0`
  - `products debt --json` reports zero actionable debt
  - all durable session-product and retrieval-enrichment bands are ready, while transcript embeddings remain intentionally pending
- the whole-repository cleanup drag named above is now retired as an active frontier
- the cleanup-only program in [programs/cleanup-and-architectural-debt-retirement-program-2026-03-24.md](./programs/cleanup-and-architectural-debt-retirement-program-2026-03-24.md) is now executed
- it closed the main structural debt around:
  - broad query-execution and query-store ownership
  - broad semantic runtime roots for facts, profiles, phases, work events, and decisions
  - broad schema/operator roots for code detection, semantic inference, runtime schema authority, and schema command rendering
  - broad QA/reporting and search/runtime public roots
  - broad async-backend, lifecycle, facade, and semantic-surface declaration roots
- the absorbed consumer/governance predecessor remains [programs/consumer-contracts-and-governed-live-cleanup-program-2026-03-24.md](./programs/consumer-contracts-and-governed-live-cleanup-program-2026-03-24.md)
- the just-executed semantic-product campaign in [programs/semantic-product-normalization-and-toolchain-convergence-program-2026-03-24.md](./programs/semantic-product-normalization-and-toolchain-convergence-program-2026-03-24.md) is now executed
- it closed the main semantic/toolchain drag around:
  - noisy project/repo canonicalization leaking into durable tag/day/week outputs
  - overbroad session-product support/status ownership
  - broad archive-product operator reshaping and missing debt governance stages
  - mixed semantic proof catalog declarations and schema-tooling role bands
  - broad Claude/Drive parser seams that fed late-stage cleanup
  - missing campaign-specific validation lanes for semantic products and live governance
- the domain read-model campaign in [programs/domain-read-model-and-live-archive-stewardship-program-2026-03-24.md](./programs/domain-read-model-and-live-archive-stewardship-program-2026-03-24.md) is now executed
- it closed the broad read-model and stewardship gap around:
  - `lib/models.py` decomposition into conversation/message/attachment/model-support bands
  - repository read banding across archive/action/maintenance/product surfaces
  - session-product SQL/query decomposition into profile/timeline/thread/summary bands
  - archive-debt governance shared by health, maintenance preview, and product status
  - stable provider analytics and archive-debt products across CLI/library/sync/MCP
  - named live stewardship validation lanes exercised against the real archive
- the runtime-substrate campaign in [programs/runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md](./programs/runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md) is now executed
- it closed the main runtime-substrate drag around:
  - query planning/retrieval/grouping/output monoliths
  - session-product lifecycle and maintenance control-plane monoliths
  - broad store and backend/schema ownership
  - semantic surface registry execution vs declaration ownership
  - broad MCP tool registration wiring
  - missing campaign-specific validation lanes and live memory-governance proofs
- the next broad frontier is now higher-level and consumer-facing:
  - domain model decomposition around the remaining broad `lib/models.py` root
  - repository-read and session-product query banding
  - external consumer and analytics contract convergence
  - governed live archive stewardship for explicit cleanup debt such as orphaned content blocks
  - tighter product/retrieval/analytics convergence on durable read models
- the just-executed archive-data-products campaign in [programs/archive-data-products-and-live-governance-program-2026-03-24.md](./programs/archive-data-products-and-live-governance-program-2026-03-24.md) closed the broad product/governance gap around:
  - durable session/work/tag/day/week archive products
  - public consumer contracts across library, CLI, MCP, and sync surfaces
  - typed maintenance lineage and machine-readable product status
  - live archive session-product rebuilds and validation lanes
  - governed cleanup preview for live orphaned content-block debt
- the next integrated move is substrate/refactoring heavy:
  - decompose the remaining broad query, lifecycle, maintenance, backend, schema, and operator modules
  - harden public contracts so they compose smaller subsystems instead of broad omnibus files
  - keep memory/live-governance validation attached to those smaller runtime surfaces
- the source-boundary/runtime-governance campaign in [programs/source-boundary-and-runtime-governance-program-2026-03-23.md](./programs/source-boundary-and-runtime-governance-program-2026-03-23.md) is now executed
- it closed the source/runtime-governance drag around:
  - historical `sources/source.py` / `sources/drive_client.py` umbrella ownership
  - typed maintenance selection and explicit safe-vs-destructive operator semantics
  - canonical derived-model provenance/freshness reporting across health, publication, and machine output
  - clearer cached-vs-live operator output semantics
  - committed local/live governance lanes for source/provider fidelity and maintenance preview
- the platform-wide refactoring campaign in [programs/platform-wide-architecture-and-refactoring-program-2026-03-23.md](./programs/platform-wide-architecture-and-refactoring-program-2026-03-23.md) is now executed
- it closed the broad architectural drag around:
  - query/filter shell convergence onto immutable plans
  - backend/query/lifecycle narrowing and removal of leftover backend read wrappers
  - shared query-spec construction across CLI and MCP
  - consolidation of semantic session/work products into explicit builders
  - narrower package-root and synthetic-stack exports
  - live-maintenance hardening for action-event readiness, repair, and archive health
- the archive-intelligence campaign in [programs/archive-intelligence-platform-convergence-program-2026-03-23.md](./programs/archive-intelligence-platform-convergence-program-2026-03-23.md) is now executed
- it closed the dogfooded archive-intelligence frontier around:
  - first-class durable action events
  - action-aware retrieval and grouped stats
  - readiness-aware fallback when persisted action semantics are absent or stale
  - freshness/provenance-aware embedding retrieval health
  - named archive-intelligence validation and memory-budget lanes
- the semantic-stack convergence program is now executed via [programs/semantic-stack-convergence-program-2026-03-23.md](./programs/semantic-stack-convergence-program-2026-03-23.md)
- it closed the strongest remaining semantic-cluster drag around:
  - harmonization boundary cleanup around `schemas/unified.py` and `lib/provider_semantics.py`
  - a missing canonical semantic facts layer shared by profiles, tags, and proof
  - downstream semantic products that still composed several separate extraction passes
  - a large semantic proof stack that mixed facts, surface policy, and suite orchestration
  - semantic preservation/loss contracts that were not previously declared once in one registry
- the schema-and-evidence pipeline campaign is now executed via [programs/schema-and-evidence-pipeline-convergence-program-2026-03-23.md](./programs/schema-and-evidence-pipeline-convergence-program-2026-03-23.md)
- it closed the remaining evidence-pipeline drag around:
  - runtime-safe schema observation helpers living too close to tooling
  - monolithic schema generation orchestration
  - monolithic synthetic corpus generation
  - overbroad prepare/source transformation boundaries
  - fragmented artifact proof and verification workflows
  - an overbroad async SQLite evidence/write path
  - a missing named roundtrip proof lane for the full schema-and-evidence loop
- the convergence campaign in [programs/core-architecture-convergence-program-2026-03-23.md](./programs/core-architecture-convergence-program-2026-03-23.md) is now executed
- it closed the previously identified main-code drag around:
  - query spec vs mutable filter execution
  - backend/query-store/repository overlap
  - Click front-door routing semantics
  - showcase/QA catalog and runner embedding
  - schema runtime vs tooling entanglement
  - over-broad package-root exports
>>>>>>> ef1c8d35 (docs: close codebase-wide cleanup program)
- the publication-control-plane half of Step 5 from [programs/intentional-forward-program-2026-03-21.md](./programs/intentional-forward-program-2026-03-21.md) has been executed via [programs/publication-control-plane-program-2026-03-22.md](./programs/publication-control-plane-program-2026-03-22.md)
- the site/repo-shape half of Step 5 is now executed via [programs/site-and-repo-shape-streamlining-program-2026-03-22.md](./programs/site-and-repo-shape-streamlining-program-2026-03-22.md)
- schema package/version authority correction has now been executed via [programs/schema-package-authority-program-2026-03-22.md](./programs/schema-package-authority-program-2026-03-22.md)
- the older scratch notes `.claude/scratch/018-wave0-schema-package-design.md` and `.claude/scratch/026-schema-taxonomy-and-versioning.md` remain useful implementation references, but no longer serve as the live authority
- the next broad frontier is no longer schema authority; the canonical markdown semantic-proof slice has now been executed via [programs/semantic-proof-and-showcase-proof-lanes-program-2026-03-22.md](./programs/semantic-proof-and-showcase-proof-lanes-program-2026-03-22.md)
- the multi-surface proof lane is now executed via [programs/multi-surface-semantic-proof-program-2026-03-22.md](./programs/multi-surface-semantic-proof-program-2026-03-22.md), which extended semantic proof across export/query surfaces and routed that suite through QA/publication/showcase
- the read-surface proof and showcase-hardening lane is now executed via [programs/read-surface-proof-and-showcase-hardening-program-2026-03-22.md](./programs/read-surface-proof-and-showcase-hardening-program-2026-03-22.md), which covers query summary/list surfaces, stream surfaces, MCP read payloads, and matching showcase proof coverage
- the remaining runtime/testing reservoir has now been executed via [programs/runtime-contract-and-validation-lanes-program-2026-03-22.md](./programs/runtime-contract-and-validation-lanes-program-2026-03-22.md), which closes the root machine contract, query/TUI hardening, and named validation lanes for chaos/scale/live operator workflows
- the next frontier is therefore no longer the old testing-reliability backlog bucket; future work should be selected intentionally from the strategic-reference programs rather than assumed from unresolved reservoir notes

## Maintenance Rule

When adding a new planning or audit document, classify it explicitly as one of:

- current execution program
- executed subprogram
- strategic reference
- audit/input
- historical closure
- generated evidence

Then update this map so the repo keeps one intentional planning surface.
