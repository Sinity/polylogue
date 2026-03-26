[← Back to README](./README.md)

# Deep Query Service And Schema Topology Cleanup Program

Date: 2026-03-26
Status: active cleanup/refactoring program
Role: cleanup-only broad queue after the executed product/runtime topology cleanup

Replaces as the live queue:

- `product-and-runtime-topology-cleanup-program-2026-03-26.md`

Prerequisite executed programs:

- `product-and-runtime-topology-cleanup-program-2026-03-26.md`
- `probabilistic-enrichment-and-governed-cleanup-program-2026-03-26.md`
- `evidence-and-stewardship-platform-convergence-program-2026-03-24.md`
- `cleanup-and-architectural-debt-retirement-program-2026-03-24.md`
- `runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md`

Primary current evidence:

- `polylogue/storage/search_providers/sqlite_vec.py` (`507`)
- `polylogue/rendering/semantic_proof_facts.py` (`445`)
- `polylogue/lib/raw_payload.py` (`442`)
- `polylogue/pipeline/services/validation.py` (`437`)
- `polylogue/lib/query_plan.py` (`429`)
- `polylogue/storage/repair_derived.py` (`427`)
- `polylogue/storage/repository_archive_reads.py` (`426`)
- `polylogue/cli/commands/products.py` (`412`)
- `polylogue/archive_product_builders.py` (`407`)
- `polylogue/storage/backends/queries/mappers_products.py` (`402`)
- `polylogue/ui/facade.py` (`397`)
- `polylogue/storage/store_core.py` (`397`)
- `polylogue/schemas/generation_support.py` (`409`)
- `polylogue/schemas/generation_analysis.py` (`386`)
- `polylogue/schemas/roundtrip_proof.py` (`383`)
- `polylogue/lib/query_retrieval.py` (`383`)
- `polylogue/pipeline/semantic.py` (`377`)

## One-Line Goal

Retire the next layer of broad mixed-role roots across query planning,
operator commands, repository/repair/build flows, raw-payload/pipeline
services, schema tooling, and the search-provider runtime.

## Why This Is The Right Next Cleanup

The just-executed cleanup program removed the broad archive-product, status,
health, MCP, template, and lane-registry roots. The next drag is deeper in the
runtime substrate:

- the query engine is still split across large plan/spec/retrieval modules with
  intertwined plan shape, SQL candidate logic, post-filter semantics, and
  grouped-stat helpers
- operator command roots such as `cli/commands/products.py` still carry too
  much command-definition and wiring density in one place
- repository archive reads, derived repair flows, archive-product aggregate
  builders, and product mappers still remain large cross-family ownership bands
- raw payload decoding and pipeline validation/semantic services still mix
  decode, inference, concurrency, and lifecycle policy in broad roots
- the schema generation and roundtrip stack still has several wide toolchain
  modules despite earlier authority/tooling cleanup
- `sqlite_vec.py` remains the single largest runtime file and still blends
  optional capability checks, SQL, lifecycle, and provider behavior

This queue is therefore cleanup-only again. Success is measured by narrower
ownership and deleted structural overlap, not by new semantic capability.

## Non-Goals

This program is not for:

- new enrichment features, embeddings, or LLM inference
- new archive-product families or live cleanup targets
- widening user-facing command surfaces beyond cleanup wiring
- compatibility aliases, deprecated shells, or transition shims
- schema behavior changes beyond what narrower module ownership requires

## Cleanup Thesis

Polylogue’s top-level product/runtime topology is now much cleaner. The next
cleanup burden sits one layer lower:

1. query planning/spec/retrieval still spread one execution model across too
   many large roots
2. operator command files still bundle too many command families and repeated
   wiring patterns
3. repository read, derived repair, aggregate build, and product mapper bands
   still carry too much cross-family logic
4. raw-payload and pipeline service roots still mix decoding, inference,
   concurrency, and lifecycle policy
5. schema toolchain internals still have several large mixed-role modules
6. the sqlite-vec provider remains an oversized runtime boundary

This campaign should end with smaller public roots, sharper ownership bands,
and fewer large modules where policy, transport, SQL, and formatting remain
interleaved.

## Architectural Rules

### 1. Query Planning Must Not Also Own Retrieval Policy

Plan shape, candidate selection, post-filter execution, grouped stats, and
retrieval-band policy should not remain braided together in one or two broad
modules.

### 2. Command Roots Must Not Also Be Workflow Registries

Click command modules should define operator surfaces, not carry repeated
query/build/render glue that belongs in narrower families.

### 3. Read, Repair, And Aggregate Build Bands Must Stay Separate

Repository reads, derived-repair flows, aggregate builders, and row mappers can
share contracts but should not keep re-deriving one another’s internals.

### 4. Decode And Validation Policy Must Not Be One File

Raw payload decoding, provider inference, sample extraction, validation
threading, and writeback policy should each have one obvious home.

### 5. Schema Toolchain Cleanup Must Remove Real Width

If a schema toolchain root remains large after this campaign, it should be
because one coherent family is still truly large, not because several distinct
concerns were left bundled together.

### 6. Optional Search Providers Need Clear Runtime Boundaries

`sqlite_vec.py` should not keep mixing capability detection, query strategy,
materialization support, and operator-facing policy in one oversized module.

## Phase 1: Query Engine Topology Cleanup

### Targets

- `polylogue/lib/query_plan.py`
- `polylogue/lib/query_spec.py`
- `polylogue/lib/query_retrieval.py`
- `polylogue/cli/query_grouped_stats.py`

### Main Work

- split immutable plan modeling from execution helpers still sitting inside
  `query_plan.py`
- reduce query-spec normalization vs validation vs compatibility translation
- narrow retrieval helpers by candidate SQL, hydration policy, and retrieval-band
  semantics
- keep grouped-stat shaping out of broader query execution internals

### Acceptance

- plan, retrieval, and grouped-stat roots each have one obvious authority
- no single query root remains a mixed model-plus-runtime-plus-policy band

## Phase 2: Operator Command And Workflow Cleanup

### Targets

- `polylogue/cli/commands/products.py`
- `polylogue/cli/commands/run.py`
- `polylogue/cli/check_workflow.py`
- `polylogue/cli/commands/qa.py`

### Main Work

- split broad command files by subcommand family or operator concern
- reduce repeated option-to-query/workflow glue where the same command shape is
  reimplemented several times
- keep machine/plain output wiring and operator workflow helpers in narrower
  bands instead of broad command roots

### Acceptance

- command modules are materially smaller
- command roots stop carrying repeated workflow/query construction patterns

## Phase 3: Read, Repair, And Aggregate Builder Cleanup

### Targets

- `polylogue/storage/repository_archive_reads.py`
- `polylogue/storage/repair_derived.py`
- `polylogue/archive_product_builders.py`
- `polylogue/storage/backends/queries/mappers_products.py`
- `polylogue/storage/store_core.py`

### Main Work

- split repository archive reads by conversation/message/tree/paging concern
- split derived repair flows by FTS, action-event, session-product, and
  maintenance target family
- narrow archive-product aggregate builders by rollup family
- reduce product-mapper width by product family or migrated-row concern
- keep low-level store models/support out of broad mixed roots

### Acceptance

- repository, repair, builder, and mapper roots each have one obvious family
- aggregate/build/repair logic no longer cross-carries unrelated product bands

## Phase 4: Raw Payload And Pipeline Service Cleanup

### Targets

- `polylogue/lib/raw_payload.py`
- `polylogue/pipeline/services/validation.py`
- `polylogue/pipeline/services/acquisition.py`
- `polylogue/pipeline/semantic.py`

### Main Work

- separate raw decode, provider inference, record-candidate logic, and sampling
  helpers
- split validation threading/CPU work from DB-write/lifecycle policy
- narrow acquisition service orchestration versus record persistence policy
- split semantic preparation helpers by evidence extraction, action shaping, and
  product write intent

### Acceptance

- decode, validation, acquisition, and semantic roots stop mixing several
  policy layers in one place
- pipeline service surfaces remain behavior-equivalent but materially smaller

## Phase 5: Schema Toolchain Topology Cleanup

### Targets

- `polylogue/schemas/generation_support.py`
- `polylogue/schemas/generation_analysis.py`
- `polylogue/schemas/generation_workflow.py`
- `polylogue/schemas/roundtrip_proof.py`
- `polylogue/schemas/audit.py`
- `polylogue/schemas/sampling.py`

### Main Work

- split schema generation helpers by evidence preparation, field analysis,
  package assembly, and output writing concern
- narrow roundtrip proof by models, fixtures, and execution helpers
- reduce audit and sampling roots where several schema-tooling concerns still
  sit together

### Acceptance

- schema generation/roundtrip/audit/sampling roots are materially smaller
- toolchain ownership bands are clearer than the current wide helper modules

## Phase 6: Search Provider And Runtime Boundary Cleanup

### Targets

- `polylogue/storage/search_providers/sqlite_vec.py`
- any adjacent helper roots exposed by the breakup

### Main Work

- separate capability detection, provider configuration, query execution,
  indexing/materialization support, and result shaping
- keep optional-provider fallback behavior explicit without leaving one
  oversized runtime root

### Acceptance

- `sqlite_vec.py` becomes a narrow provider root over smaller concern bands
- optional vector runtime behavior stays equivalent but more navigable

## Validation And Closure

This cleanup program is complete only when all of the following are true:

- the main broad roots above are materially smaller or deleted
- no new compatibility wrappers were added to preserve old shapes
- targeted regression slices covering query, commands, storage, pipeline,
  schema, and search-provider surfaces pass
- at least one named composite validation lane still passes after the breakup
- live archive health/debt/product status remains clean
- `earlyoom` stays quiet under the representative live lanes and budgets
