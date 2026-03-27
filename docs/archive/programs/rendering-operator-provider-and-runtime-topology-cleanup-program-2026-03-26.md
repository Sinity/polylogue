[← Back to README](./README.md)

# Rendering Operator Provider And Runtime Topology Cleanup Program

Date: 2026-03-26
Status: absorbed predecessor cleanup/refactoring program
Role: narrower predecessor broadened into the codebase-wide cleanup/refactoring queue

Replaced as the live queue by:

- `codebase-wide-topology-and-debt-retirement-program-2026-03-26.md`

Prerequisite executed programs:

- `deep-query-service-and-schema-topology-cleanup-program-2026-03-26.md`
- `product-and-runtime-topology-cleanup-program-2026-03-26.md`
- `probabilistic-enrichment-and-governed-cleanup-program-2026-03-26.md`
- `evidence-and-stewardship-platform-convergence-program-2026-03-24.md`
- `cleanup-and-architectural-debt-retirement-program-2026-03-24.md`
- `runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md`

Primary current evidence:

- `polylogue/rendering/semantic_proof_facts.py` (`445`)
- `polylogue/rendering/renderers/html.py` (`416`)
- `polylogue/ui/facade.py` (`397`)
- `polylogue/schemas/field_stats.py` (`394`)
- `polylogue/rendering/semantic_proof_surfaces.py` (`391`)
- `polylogue/storage/action_event_rebuild.py` (`380`)
- `polylogue/storage/backends/queries/session_product_timeline_queries.py` (`374`)
- `polylogue/schemas/unified_provider_meta.py` (`371`)
- `polylogue/storage/embedding_stats.py` (`370`)
- `polylogue/rendering/semantic_proof.py` (`370`)
- `polylogue/storage/backends/schema_ddl_products.py` (`367`)
- `polylogue/storage/store_runtime_records.py` (`362`)
- `polylogue/sources/providers/claude_code.py` (`356`)
- `polylogue/archive_product_entities.py` (`354`)
- `polylogue/sources/parsers/base.py` (`334`)
- `polylogue/rendering/semantic_proof_models.py` (`334`)
- `polylogue/lib/artifact_taxonomy.py` (`333`)
- `polylogue/schemas/observation.py` (`332`)
- `polylogue/rendering/core.py` (`332`)
- `polylogue/cli/query_output.py` (`328`)
- `polylogue/cli/check_rendering.py` (`325`)
- `polylogue/cli/commands/embed.py` (`323`)
- `polylogue/schemas/synthetic/builders.py` (`322`)
- `polylogue/storage/backends/async_sqlite.py` (`320`)
- `polylogue/schemas/generation_annotations.py` (`319`)
- `polylogue/sources/providers/gemini.py` (`318`)
- `polylogue/sources/providers/chatgpt.py` (`318`)
- `polylogue/schemas/relational_inference.py` (`316`)
- `polylogue/sources/drive_source.py` (`312`)
- `polylogue/storage/session_product_profile_rows.py` (`310`)
- `polylogue/lib/query_runtime.py` (`310`)
- `polylogue/storage/repository_product_reads.py` (`308`)
- `polylogue/rendering/semantic_surface_canonical_declarations.py` (`308`)
- `polylogue/schemas/validator.py` (`305`)
- `polylogue/schemas/operator_workflow.py` (`305`)
- `polylogue/storage/repository_writes.py` (`300`)
- `polylogue/sources/drive_auth.py` (`298`)
- `polylogue/storage/search_providers/hybrid.py` (`296`)
- `polylogue/cli/click_app.py` (`292`)
- `polylogue/lib/action_events.py` (`291`)
- `polylogue/storage/backends/queries/attachments.py` (`290`)
- `polylogue/pipeline/runner.py` (`286`)
- `polylogue/sources/parsers/claude.py` (`283`)
- `polylogue/lib/raw_payload_sampling.py` (`283`)
- `polylogue/cli/schema_rendering.py` (`281`)
- `polylogue/storage/session_product_timeline_rows.py` (`280`)
- `polylogue/showcase/runner.py` (`277`)
- `polylogue/storage/fts_lifecycle.py` (`273`)
- `polylogue/schemas/semantic_inference_scoring.py` (`273`)
- `polylogue/cli/helpers.py` (`273`)
- `polylogue/lib/viewports.py` (`272`)
- `polylogue/sources/decoders.py` (`271`)
- `polylogue/site/builder.py` (`271`)
- `polylogue/cli/commands/check.py` (`271`)
- `polylogue/sync.py` (`270`)
- `polylogue/sources/parsers/drive_support.py` (`270`)
- `polylogue/mcp/payloads.py` (`270`)

## One-Line Goal

Retire the remaining broad operator/rendering/provider/runtime roots so
Polylogue’s top half is no longer carried by large mixed-role files after the
substate/query/schema cleanup wave.

## Why This Is The Right Next Cleanup

The last cleanup queue closed the deepest obvious mixed-role seams around query
planning, command families, archive-read/repair/build bands, raw payload
decoding, pipeline services, schema tooling roots, and the sqlite-vec provider.

That leaves a different class of structural drag:

- rendering and semantic-proof roots still blend declaration, fact extraction,
  comparison logic, and output shaping
- operator-adapter roots still carry too much CLI/MCP/UI glue in single files
- storage/runtime write and rebuild paths still keep schema DDL, record
  definitions, rebuild orchestration, and lifecycle status too tightly coupled
- schema/runtime analysis still has several large helper roots whose concerns
  have never been decomposed because they were not on the earlier critical path
- provider/parser/source roots still bundle too much provider-specific shape,
  auth, traversal, and decode policy in large files
- several domain/query evidence roots are still oversized and sit directly on
  the semantic layer used across products, retrieval, and rendering

This queue is therefore cleanup-only again, but broader and more ambitious than
the last one. Success is measured by deleted overlap, materially smaller public
roots, and sharper ownership seams across the remaining top-heavy runtime
surfaces.

## Non-Goals

This program is not for:

- new retrieval features, new embeddings, or LLM-based enrichment
- new archive-product families or new live cleanup targets
- changing semantic meaning just to justify module moves
- backwards-compatibility shims, deprecated aliases, or “temporary” wrappers
- another plan-only wave that renames files without deleting structural overlap

## Cleanup Thesis

Polylogue’s lower-level substrate is now much cleaner. The next cleanup burden
is concentrated in the top-heavy execution and semantics bands:

1. rendering and semantic-proof execution still spread one concept across a set
   of broad fact/comparator/output roots
2. operator adapters still keep CLI/MCP/UI glue bundled in large roots
3. write/rebuild/status/storage roots still mix record definitions, DDL,
   lifecycle policy, and rebuild execution
4. schema/runtime analysis roots still remain broad because they predate the
   current decomposition style
5. provider/parser/source roots still carry too much provider-specific runtime
   behavior in single modules
6. domain/query evidence roots still conflate model contracts, extraction, and
   runtime composition

This campaign should end with smaller public roots, clearer bands between
declarations/runtime/output, and fewer places where the same semantic meaning
is translated repeatedly by different operator surfaces.

## Architectural Rules

### 1. Rendering Roots Must Not Also Declare Semantic Truth

Proof declarations, fact extraction, comparison policy, and renderer-specific
output shaping should not stay braided together.

### 2. Operator Adapters Must Not Also Be Domain Assemblers

CLI, MCP, UI, and sync roots should bind public surfaces, not re-derive domain
models or duplicate workflow composition.

### 3. Record Definitions Must Not Also Own Rebuild Strategy

Store records, DDL, rebuild flows, and derived-status logic can share
contracts, but should not stay in broad cross-family roots.

### 4. Provider Runtime Roots Must Stay Narrow

Provider-specific decode/auth/traversal shape should not remain spread across a
small number of oversized provider modules.

### 5. Cleanup Must Remove Real Width

Every phase should materially reduce at least one currently broad root rather
than only moving helpers around without shrinking the public authority.

## Phase 1: Semantic Proof And Rendering Topology Cleanup

### Targets

- `polylogue/rendering/semantic_proof_facts.py`
- `polylogue/rendering/semantic_proof_surfaces.py`
- `polylogue/rendering/semantic_proof.py`
- `polylogue/rendering/semantic_proof_models.py`
- `polylogue/rendering/core.py`
- `polylogue/rendering/renderers/html.py`
- `polylogue/rendering/semantic_surface_canonical_declarations.py`
- `polylogue/rendering/semantic_surface_query_declarations.py`

### Main Work

- split proof fact extraction by evidence family rather than one broad fact
  root
- narrow surface comparison helpers by surface family instead of one large
  mixed comparator root
- keep proof orchestration distinct from renderer-facing output shaping
- reduce HTML renderer width by page section/family instead of one large
  output root
- narrow semantic-surface declarations so catalog families are more obviously
  separated from canonical/query-specific declarations

### Acceptance

- proof roots are materially smaller and clearer by role
- rendering roots stop mixing declaration, extraction, comparison, and output
  formatting in one place

## Phase 2: Operator Adapter And Control-Plane Cleanup

### Targets

- `polylogue/cli/query_output.py`
- `polylogue/cli/check_rendering.py`
- `polylogue/cli/schema_rendering.py`
- `polylogue/cli/commands/embed.py`
- `polylogue/cli/commands/check.py`
- `polylogue/cli/click_app.py`
- `polylogue/cli/helpers.py`
- `polylogue/ui/facade.py`
- `polylogue/sync.py`
- `polylogue/mcp/payloads.py`

### Main Work

- narrow CLI output roots so summary/detail/JSON/plain rendering families stop
  accumulating into large adapter modules
- split root CLI command wiring from reusable option/parsing/render helpers
- reduce UI facade width by prompt family / non-TTY / stub / streaming concern
- narrow sync and MCP payload roots so public contract shaping is distinct from
  transport convenience helpers

### Acceptance

- operator roots bind public surfaces but do not keep reassembling domain logic
- no single operator adapter remains an omnibus rendering/control-plane file

## Phase 3: Storage Write, Rebuild, And Lifecycle Cleanup

### Targets

- `polylogue/storage/action_event_rebuild.py`
- `polylogue/storage/embedding_stats.py`
- `polylogue/storage/backends/schema_ddl_products.py`
- `polylogue/storage/store_runtime_records.py`
- `polylogue/storage/session_product_profile_rows.py`
- `polylogue/storage/session_product_timeline_rows.py`
- `polylogue/storage/repository_product_reads.py`
- `polylogue/storage/repository_writes.py`
- `polylogue/storage/backends/async_sqlite.py`
- `polylogue/storage/fts_lifecycle.py`
- `polylogue/storage/search_providers/hybrid.py`
- `polylogue/storage/backends/queries/session_product_profile_queries.py`
- `polylogue/storage/backends/queries/session_product_timeline_queries.py`

### Main Work

- split rebuild flows by read-model family instead of one large rebuild root
- narrow write/storage roots by contract family rather than broad repository
  write policy
- reduce DDL roots by product family and move reusable statement fragments out
- narrow lifecycle/query status helpers where sync/async or profile/timeline
  responsibilities still sit together
- keep hybrid/vector retrieval orchestration separate from provider/runtime
  policy

### Acceptance

- write/rebuild/status roots have one obvious ownership band each
- storage lifecycle policy is less spread across broad repository and backend
  modules

## Phase 4: Schema Runtime Analysis And Tooling Cleanup

### Targets

- `polylogue/schemas/field_stats.py`
- `polylogue/schemas/observation.py`
- `polylogue/schemas/relational_inference.py`
- `polylogue/schemas/validator.py`
- `polylogue/schemas/unified_provider_meta.py`
- `polylogue/schemas/operator_workflow.py`
- `polylogue/schemas/generation_annotations.py`
- `polylogue/schemas/semantic_inference_scoring.py`
- `polylogue/schemas/synthetic/builders.py`

### Main Work

- split field-stat collection, path walking, enum/value heuristics, and format
  detection into narrower internal families
- reduce observation/runtime schema extraction roots where provider/runtime
  concerns still accumulate
- narrow relational/validator/operator roots so orchestration and atomic checks
  are no longer braided together
- keep annotation/scoring/builder helpers separated by one coherent concern

### Acceptance

- schema analysis roots are materially smaller and role-specific
- runtime/tooling helper bands are clearer than the current wide authority set

## Phase 5: Provider, Parser, And Source Boundary Cleanup

### Targets

- `polylogue/sources/providers/claude_code.py`
- `polylogue/sources/providers/gemini.py`
- `polylogue/sources/providers/chatgpt.py`
- `polylogue/sources/parsers/base.py`
- `polylogue/sources/parsers/claude.py`
- `polylogue/sources/parsers/drive_support.py`
- `polylogue/sources/drive_source.py`
- `polylogue/sources/drive_auth.py`
- `polylogue/sources/decoders.py`

### Main Work

- split provider runtime adapters by content/event/tool/action family where
  they remain broad
- narrow parser/base roots so shared parser scaffolding does not also own
  provider-specific normalization
- reduce Drive source/auth/support roots by traversal, credentials, and payload
  shaping concern
- keep decode helpers focused on one wire/decompression family each

### Acceptance

- provider/source roots stop carrying several unrelated runtime concerns
- parser/shared helper roots are smaller and more obviously layered

## Phase 6: Domain And Query-Evidence Model Cleanup

### Targets

- `polylogue/lib/artifact_taxonomy.py`
- `polylogue/lib/action_events.py`
- `polylogue/lib/query_runtime.py`
- `polylogue/lib/viewports.py`
- `polylogue/lib/projections.py`
- `polylogue/lib/message_models.py`
- `polylogue/lib/conversation_models.py`
- `polylogue/archive_product_entities.py`

### Main Work

- narrow taxonomy/action-evidence roots by model/support/extraction concern
- reduce query-runtime composition width by separating plan execution helpers
  from runtime result shaping where still mixed
- split domain-model roots where contracts and convenience/runtime methods
  remain interleaved
- keep archive-product entity contracts distinct from formatting or productized
  convenience behavior

### Acceptance

- evidence/model roots become easier to inspect and less coupled to operator
  conveniences
- query/runtime and domain/model roots each have clearer authority boundaries

## Validation And Closure

This cleanup program is complete only when all of the following are true:

- the broad roots above are materially smaller or replaced by narrower families
- no compatibility wrappers or deprecated aliases were added
- targeted regression slices covering rendering, operator adapters, storage,
  schema analysis, provider/source, and domain/query evidence surfaces pass
- at least one named composite validation lane still passes after the breakup
- live archive health/debt/product status remains clean
- representative live validation or operator commands stay within memory budget
- `earlyoom` stays quiet under the representative lane/command set
