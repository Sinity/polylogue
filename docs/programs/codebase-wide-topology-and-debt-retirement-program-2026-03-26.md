[← Back to README](./README.md)

# Codebase-Wide Topology And Debt Retirement Program

Date: 2026-03-26
Status: active cleanup/refactoring program
Role: canonical whole-repository cleanup-only queue for every remaining architectural-debt band in Polylogue

Replaces as the live queue:

- `rendering-operator-provider-and-runtime-topology-cleanup-program-2026-03-26.md`

Absorbs as narrower predecessors:

- `rendering-operator-provider-and-runtime-topology-cleanup-program-2026-03-26.md`
- `deep-query-service-and-schema-topology-cleanup-program-2026-03-26.md`
- `product-and-runtime-topology-cleanup-program-2026-03-26.md`
- `cleanup-and-architectural-debt-retirement-program-2026-03-24.md`

Prerequisite executed programs:

- `probabilistic-enrichment-and-governed-cleanup-program-2026-03-26.md`
- `evidence-and-stewardship-platform-convergence-program-2026-03-24.md`
- `runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md`
- `platform-wide-architecture-and-refactoring-program-2026-03-23.md`
- the earlier executed convergence/refactoring records already indexed in `README.md`

## Repository Coverage Snapshot

The queue is whole-repository only if it names the actual repo surface area.
Current Python footprint by top-level package:

| Package | Files | Total lines | Current max root |
| --- | ---: | ---: | ---: |
| `storage/` | 132 | 18,269 | 380 |
| `schemas/` | 76 | 10,949 | 394 |
| `cli/` | 65 | 8,692 | 328 |
| `lib/` | 64 | 8,087 | 333 |
| `sources/` | 28 | 5,646 | 356 |
| `pipeline/` | 31 | 4,557 | 286 |
| `rendering/` | 18 | 3,764 | 445 |
| `showcase/` | 22 | 2,769 | 277 |
| `site/` | 13 | 1,805 | 271 |
| `mcp/` | 15 | 1,671 | 270 |
| `ui/` | 10 | 1,256 | 397 |
| `operations/` | 8 | 791 | 173 |

Cross-cutting surfaces are also in scope:

- package roots and re-export topology
- `devtools/` validation lanes and memory-budget surfaces
- test layout and fixtures that still mirror pre-cleanup topology
- docs/program indexes and operator references that would otherwise lag behind code movement

Nothing in the repo is implicitly exempt.

## Representative Current Hotspots

This is the concrete floor, not the ceiling, for the cleanup queue:

- `rendering/`: `semantic_proof_facts.py`, `renderers/html.py`, `semantic_proof_surfaces.py`, `semantic_proof.py`, `semantic_proof_models.py`
- `cli/`: `query_output.py`, `check_rendering.py`, `commands/embed.py`, `click_app.py`, `schema_rendering.py`
- `lib/`: `artifact_taxonomy.py`, `query_runtime.py`, `action_events.py`, `raw_payload_sampling.py`, `viewports.py`
- `storage/`: `action_event_rebuild.py`, `session_product_timeline_queries.py`, `embedding_stats.py`, `schema_ddl_products.py`, `store_runtime_records.py`
- `schemas/`: `field_stats.py`, `unified_provider_meta.py`, `observation.py`, `roundtrip_provider.py`, `synthetic/builders.py`
- `sources/`: `providers/claude_code.py`, `parsers/base.py`, `providers/gemini.py`, `providers/chatgpt.py`, `drive_source.py`
- `pipeline/`: `runner.py`, `services/planning.py`, `prepare_transform.py`, `services/validation_flow.py`, `run_stages.py`
- `showcase/`: `runner.py`, `showcase_report.py`, `qa_markdown.py`, `generators.py`, `workspace.py`
- `site/`: `builder.py`, `templates_index.py`, `templates_conversation.py`, `templates_dashboard.py`, `publication_flow.py`
- `mcp/`: `payloads.py`, `server_prompts.py`, `server_product_tools_session.py`, `server_product_tools_aggregate.py`, `server_query_tools.py`
- `ui/`: `facade.py`, `theme.py`, `__init__.py`
- `operations/`: `archive_product_support_debt.py`, `archive_product_support_session.py`, `archive_product_support_analytics.py`, `archive_search_support.py`, `archive.py`

## One-Line Goal

Retire architectural debt across every remaining top-level package and shared
governance surface, not just the latest cleanup hotspot.

## Why This Program Must Be Broader Than The Last Broadening

The previous expansion correctly stopped treating the rendering/operator/runtime
slice as the whole problem. It still was not strict enough.

The repo-wide cleanup authority has to do more than name the obvious biggest
files. It has to explicitly cover:

- every top-level runtime package
- MCP and operations surfaces used by external consumers
- package-root import/re-export cleanup
- tests, validation lanes, and docs that otherwise preserve old topology
- live-archive and memory governance attached to the same broad code moves

Without that, the queue remains only partially comprehensive and leaves adjacent
bands untouched whenever one subsystem gets cleaned up in isolation.

## Explicit Repository Scope

This program owns cleanup across:

- `polylogue/rendering/`
- `polylogue/cli/`
- `polylogue/lib/`
- `polylogue/storage/`
- `polylogue/schemas/`
- `polylogue/sources/`
- `polylogue/pipeline/`
- `polylogue/showcase/`
- `polylogue/site/`
- `polylogue/mcp/`
- `polylogue/ui/`
- `polylogue/operations/`
- package roots, re-export surfaces, and import topology
- `devtools/`, validation-lane registry, memory-budget probes, and test support
- operator/reference docs that need to move with the code

If a broad mixed-role root remains anywhere in those bands, it belongs here.

## Non-Goals

This program is still cleanup/refactoring only. It is not for:

- new retrieval features, new semantic inference features, or new LLM workflows
- new archive-product families or new live cleanup targets
- semantics changes justified only by moving files around
- compatibility wrappers, deprecation shells, or transition aliases
- renaming modules without shrinking or deleting structural overlap
- “cleanup” that edits one file while leaving the obvious sibling roots untouched

## Cleanup Thesis

Polylogue’s earlier waves removed the deepest substrate bottlenecks. The
remaining debt is now distributed across the whole repo:

1. rendering/proof/output families still carry broad mixed-role roots
2. CLI/UI/sync/MCP control-plane surfaces still bundle too much shaping logic
3. storage write/rebuild/query/status/DDL/search bands still keep broad roots
4. schema runtime/tooling/verification/synthetic bands still have broad helper authorities
5. provider/source/parser/Drive runtime still has oversized provider and support roots
6. domain/evidence/query model bands still conflate contracts, extraction, and runtime assembly
7. pipeline/showcase/site/UI orchestration bands still have broad execution/report/page roots
8. operations and consumer-facing helper bands still mirror pre-cleanup topology
9. package roots, tests, devtools, and docs still risk lagging behind the code cleanup unless made explicit

The queue is complete only when all nine bands are closed together.

## Architectural Rules

### 1. No Major Package Is Exempt

If a top-level package still has broad mixed-role roots, this program owns it.

### 2. Public Roots Must Be Thin Authorities

Root modules should bind a surface over narrower families, not remain broad
implementations that happen to be imported everywhere.

### 3. Domain Truth Must Stay Separate From Operator Formatting

Contracts, facts, read models, and lifecycle status should not keep reappearing
inside CLI, MCP, UI, or site-formatting roots.

### 4. Storage Layers Must Stay Distinct

DDL, row contracts, rebuild flows, repository reads/writes, SQL families, and
derived status must not remain braided together in broad roots.

### 5. Provider Runtime Must Stay Distinct From Parsing And Transport

Provider adapters, parser scaffolding, Drive/auth helpers, traversal logic, and
decode policy must not stay mixed in a few oversized modules.

### 6. Adjacent Sibling Roots Move Together

Do not narrow one surface in a band while leaving the obvious adjacent mixed-role
roots untouched.

### 7. Tests, Devtools, And Docs Move With Code

Cleanup is incomplete if old topology survives in tests, validation lanes,
operator references, or planning indexes.

### 8. Retained Broad Roots Need Explicit Justification

If a large root remains after the campaign, the execution record must explain
why it is now single-role rather than merely still broad.

### 9. Closure Requires Whole-Band Verification

Each phase must prove the entire affected band still passes, not just the one
module that was edited.

## Phase 1: Rendering, Proof, And Output Surface Cleanup

### Targets

- `polylogue/rendering/semantic_proof_facts.py`
- `polylogue/rendering/semantic_proof_surfaces.py`
- `polylogue/rendering/semantic_proof.py`
- `polylogue/rendering/semantic_proof_models.py`
- `polylogue/rendering/core.py`
- `polylogue/rendering/renderers/html.py`
- `polylogue/rendering/semantic_surface_canonical_declarations.py`
- `polylogue/rendering/semantic_surface_query_declarations.py`
- `polylogue/cli/query_output.py`
- `polylogue/cli/check_rendering.py`
- `polylogue/cli/schema_rendering.py`
- `polylogue/cli/query.py`
- `polylogue/cli/query_semantic_stats.py`
- `polylogue/cli/formatting.py`
- `polylogue/rendering/formatting.py`

### Main Work

- split proof facts, surface comparators, proof orchestration, and output
  shaping by coherent concern families
- reduce HTML/core renderer roots by page, section, or output family
- keep CLI output adapters thin rather than shadow-rendering the same concepts

### Acceptance

- rendering and proof roots are materially smaller across the whole band
- CLI output surfaces stop acting as parallel renderer authorities

## Phase 2: CLI, UI, Sync, And Operator Control-Plane Cleanup

### Targets

- `polylogue/cli/commands/embed.py`
- `polylogue/cli/commands/check.py`
- `polylogue/cli/click_app.py`
- `polylogue/cli/helpers.py`
- `polylogue/cli/check_workflow.py`
- `polylogue/cli/run_workflow.py`
- `polylogue/ui/facade.py`
- `polylogue/ui/theme.py`
- `polylogue/ui/__init__.py`
- `polylogue/sync.py`
- `polylogue/cli/machine_errors.py`

### Main Work

- narrow root CLI adapters by command family, machine/plain contract family,
  and reusable workflow/rendering support
- split UI roots by prompt, stream, non-TTY, stub, and theme concerns
- narrow sync and operator helper roots so transport convenience does not become
  another domain-shaping authority

### Acceptance

- operator roots are thinner across the full CLI/UI/sync stack
- no single operator control-plane root remains an omnibus file

## Phase 3: MCP And Operations Consumer Surface Cleanup

### Targets

- `polylogue/mcp/payloads.py`
- `polylogue/mcp/server_prompts.py`
- `polylogue/mcp/server_query_tools.py`
- `polylogue/mcp/server_product_tools_session.py`
- `polylogue/mcp/server_product_tools_aggregate.py`
- `polylogue/mcp/server_product_tools_maintenance.py`
- `polylogue/operations/archive.py`
- `polylogue/operations/archive_product_support_session.py`
- `polylogue/operations/archive_product_support_analytics.py`
- `polylogue/operations/archive_product_support_debt.py`
- `polylogue/operations/archive_search_support.py`

### Main Work

- narrow MCP payload, prompt, query-tool, and product-tool roots by one public
  contract family each
- keep operations helpers distinct from product contract shaping and consumer
  presentation logic
- remove any leftover parallel query/product shaping between MCP and operations

### Acceptance

- external consumer surfaces are thin and consistent across MCP and operations
- consumer-facing helper bands no longer mirror old broad runtime topology

## Phase 4: Storage Runtime, Write, Query, Rebuild, And Search Cleanup

### Targets

- `polylogue/storage/action_event_rebuild.py`
- `polylogue/storage/embedding_stats.py`
- `polylogue/storage/backends/schema_ddl_products.py`
- `polylogue/storage/backends/schema_ddl.py`
- `polylogue/storage/store_runtime_records.py`
- `polylogue/storage/store_products.py`
- `polylogue/storage/repository_product_reads.py`
- `polylogue/storage/repository_writes.py`
- `polylogue/storage/backends/async_sqlite.py`
- `polylogue/storage/fts_lifecycle.py`
- `polylogue/storage/search_providers/hybrid.py`
- `polylogue/storage/session_product_profile_rows.py`
- `polylogue/storage/session_product_timeline_rows.py`
- `polylogue/storage/session_product_row_support.py`
- `polylogue/storage/derived_status_products.py`
- `polylogue/storage/backends/queries/session_product_profile_queries.py`
- `polylogue/storage/backends/queries/session_product_timeline_queries.py`
- `polylogue/storage/backends/queries/messages.py`
- `polylogue/storage/backends/queries/attachments.py`
- `polylogue/storage/backends/queries/stats.py`
- `polylogue/storage/backends/queries/raw_reads.py`
- `polylogue/storage/backends/queries/raw_state.py`

### Main Work

- split rebuild flows by read-model family
- narrow records, DDL, SQL, status, and write-support families into clearer ownership bands
- reduce repository roots where too much storage policy still accumulates
- keep hybrid, FTS, and vector runtime surfaces separate from provider/runtime policy

### Acceptance

- no major storage band remains outside the cleanup queue
- repository, DDL, SQL, rebuild, and status responsibilities are clearer repo-wide

## Phase 5: Schema Runtime, Analysis, Verification, And Synthetic Cleanup

### Targets

- `polylogue/schemas/field_stats.py`
- `polylogue/schemas/observation.py`
- `polylogue/schemas/unified_provider_meta.py`
- `polylogue/schemas/roundtrip_provider.py`
- `polylogue/schemas/relational_inference.py`
- `polylogue/schemas/validator.py`
- `polylogue/schemas/operator_workflow.py`
- `polylogue/schemas/generation_annotations.py`
- `polylogue/schemas/generation_provider_bundle.py`
- `polylogue/schemas/verification_corpus.py`
- `polylogue/schemas/semantic_inference_scoring.py`
- `polylogue/schemas/synthetic/builders.py`
- `polylogue/schemas/synthetic/runtime.py`
- `polylogue/schemas/synthetic/relations.py`
- `polylogue/schemas/tooling_models.py`
- `polylogue/schemas/operator_models.py`
- `polylogue/schemas/verification_models.py`
- `polylogue/schemas/redaction_report.py`

### Main Work

- split broad analysis/runtime helpers by one coherent role each
- narrow verification and operator workflow roots where checks, hydration, and
  orchestration still sit together
- reduce synthetic/runtime helper roots that still act as broad utility bundles

### Acceptance

- schema runtime/tooling bands are cleaner across the full subsystem, not only
  the earlier generation and sampling slices

## Phase 6: Provider, Source, Parser, And Drive Boundary Cleanup

### Targets

- `polylogue/sources/providers/claude_code.py`
- `polylogue/sources/providers/gemini.py`
- `polylogue/sources/providers/chatgpt.py`
- `polylogue/sources/providers/codex.py`
- `polylogue/sources/parsers/base.py`
- `polylogue/sources/parsers/claude.py`
- `polylogue/sources/parsers/codex.py`
- `polylogue/sources/parsers/drive.py`
- `polylogue/sources/parsers/drive_support.py`
- `polylogue/sources/drive_source.py`
- `polylogue/sources/drive_auth.py`
- `polylogue/sources/drive.py`
- `polylogue/sources/decoders.py`
- `polylogue/sources/dispatch.py`
- `polylogue/sources/emitter.py`

### Main Work

- split provider adapters by one coherent runtime family rather than one broad provider root
- narrow shared parser scaffolding so provider-specific policy is not trapped in base/support modules
- reduce Drive/auth/source roots by traversal, auth, cache, and payload-shaping concern
- narrow decode, dispatch, and emitter roots where unrelated wire/runtime helpers still live together

### Acceptance

- provider/source/parser cleanup is whole-subsystem cleanup rather than selective hotspot work

## Phase 7: Domain, Evidence, Query, And Product-Entity Cleanup

### Targets

- `polylogue/lib/artifact_taxonomy.py`
- `polylogue/lib/action_events.py`
- `polylogue/lib/query_runtime.py`
- `polylogue/lib/query_plan.py`
- `polylogue/lib/raw_payload_sampling.py`
- `polylogue/lib/viewports.py`
- `polylogue/lib/projections.py`
- `polylogue/lib/semantic_fact_builders.py`
- `polylogue/lib/semantic_fact_models.py`
- `polylogue/lib/work_event_extraction.py`
- `polylogue/lib/message_models.py`
- `polylogue/lib/conversation_models.py`
- `polylogue/lib/session_profile_models.py`
- `polylogue/archive_product_entities.py`

### Main Work

- narrow taxonomy, evidence, query-runtime, and model roots by contract, support, and runtime family
- keep domain contracts separate from convenience assembly and operator-facing shaping helpers
- reduce product-entity roots where multiple contract families are still bundled together

### Acceptance

- domain/query/evidence cleanup is explicit across the remaining broad model roots

## Phase 8: Pipeline, Showcase, Site, And User-Facing Orchestration Cleanup

### Targets

- `polylogue/pipeline/runner.py`
- `polylogue/pipeline/services/planning.py`
- `polylogue/pipeline/prepare_transform.py`
- `polylogue/pipeline/services/validation_flow.py`
- `polylogue/pipeline/run_stages.py`
- `polylogue/pipeline/ids.py`
- `polylogue/pipeline/observers.py`
- `polylogue/showcase/runner.py`
- `polylogue/showcase/showcase_report.py`
- `polylogue/showcase/qa_markdown.py`
- `polylogue/showcase/generators.py`
- `polylogue/showcase/workspace.py`
- `polylogue/site/builder.py`
- `polylogue/site/templates_index.py`
- `polylogue/site/templates_conversation.py`
- `polylogue/site/templates_dashboard.py`
- `polylogue/site/publication_flow.py`

### Main Work

- narrow orchestration roots that still combine planning, execution, rendering,
  and artifact staging
- reduce showcase/site roots where page, report, generator, and workspace concerns are still mixed
- keep site/build/publication families aligned with the same decomposition style as the rest of the repo

### Acceptance

- pipeline, showcase, and site cleanup is explicit and whole-band rather than deferred

## Phase 9: Package-Root, Test, Devtools, And Documentation Topology Cleanup

### Targets

- `polylogue/__init__.py`
- `polylogue/lib/__init__.py`
- `polylogue/schemas/__init__.py`
- any remaining package-root or re-export surfaces affected by earlier phases
- `devtools/run_validation_lanes.py`
- memory-budget probes and live-governance helpers affected by the cleanup
- tests and fixtures that still encode superseded topology
- program indexes, CLI reference docs, and architecture references affected by the cleanup

### Main Work

- delete obsolete imports, re-exports, and compatibility-shaped surface area introduced by earlier waves
- keep validation lanes and memory-budget probes aligned with the new topology
- update tests and docs in the same wave so old ownership assumptions do not persist in support files

### Acceptance

- package/import surfaces do not lag behind subsystem breakups
- tests, devtools, and docs reflect the cleaned topology rather than preserving the old one

## Validation And Closure

This codebase-wide cleanup program is complete only when all of the following are true:

- every top-level package named above has been addressed where broad mixed-role roots remained
- MCP, operations, tests, devtools, and docs have moved with the code rather than lagging behind it
- any retained broad root has explicit execution-record justification that it is now genuinely single-role
- targeted regression slices across rendering, operator, MCP, storage, schema, sources, domain/query, pipeline/showcase/site/UI, and package-root/devtools bands pass
- at least one named composite validation lane and one live/archive-oriented validation lane pass after the breakup
- live archive health, debt, product status, and enrichment status remain clean via:
  - `python -m polylogue --plain products status --json`
  - `python -m polylogue --plain products debt --json`
  - `python -m polylogue --plain check --json`
  - `python -m polylogue --plain embed --stats --json`
- representative live operator commands remain within budget and `earlyoom` stays quiet under the representative lane/command set
- `ruff check`, `git diff --check`, and the relevant pytest slices pass
- the worktree ends clean, with obsolete wrappers and re-export cruft removed rather than preserved
