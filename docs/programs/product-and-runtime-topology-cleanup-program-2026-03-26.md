[← Back to README](./README.md)

# Product And Runtime Topology Cleanup Program

Date: 2026-03-26
Status: active cleanup/refactoring program
Role: cleanup-only broad queue after the executed probabilistic-enrichment wave

Replaces as the live queue:

- `probabilistic-enrichment-and-governed-cleanup-program-2026-03-26.md`

Prerequisite executed programs:

- `probabilistic-enrichment-and-governed-cleanup-program-2026-03-26.md`
- `evidence-and-stewardship-platform-convergence-program-2026-03-24.md`
- `cleanup-and-architectural-debt-retirement-program-2026-03-24.md`
- `semantic-product-normalization-and-toolchain-convergence-program-2026-03-24.md`
- `domain-read-model-and-live-archive-stewardship-program-2026-03-24.md`
- `runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md`

Primary current evidence:

- `devtools/run_validation_lanes.py` (`982`)
- `polylogue/storage/session_product_rows.py` (`825`)
- `polylogue/site/templates.py` (`643`)
- `polylogue/archive_products.py` (`599`)
- `polylogue/storage/backends/queries/mappers.py` (`594`)
- `polylogue/storage/backends/queries/conversations.py` (`558`)
- `polylogue/storage/backends/queries/raw.py` (`521`)
- `polylogue/storage/session_product_storage.py` (`496`)
- `polylogue/storage/derived_status.py` (`450`)
- `polylogue/storage/session_product_status.py` (`442`)
- `polylogue/health_archive.py` (`441`)
- `polylogue/mcp/server_product_tools.py` (`433`)
- `polylogue/operations/archive_product_support.py` (`429`)

## One-Line Goal

Delete the remaining broad mixed-role product/runtime roots so Polylogue’s
archive-product, health, query, and operator surfaces are composed of narrow
ownership bands rather than large cross-layer modules.

## Why This Is The Right Next Cleanup

The last campaign closed real product and governance gaps:

- enrichment products are durable and queryable
- cleanup lineage is real and validated on the live archive
- archive debt is zero-actionable for the known destructive targets

That means the next drag is not missing feature capability. It is structural
shape:

- archive-product contracts, row builders, storage, status, debt, health, and
  operator adapters still know too much about one another
- query SQL roots still remain too broad across conversation and raw bands
- validation-lane declarations have become another large mixed registry
- the remaining broad declarative/template roots should be narrowed while their
  semantics are still fresh

This program is therefore intentionally cleanup-only. Success is measured by
deletion, breakup, and narrower ownership, not by new product families.

## Non-Goals

This program is not for:

- new enrichment algorithms, embeddings, or LLM-assisted semantics
- widening consumer features beyond what cleanup wiring requires
- new destructive live cleanup targets
- compatibility wrappers or “legacy” shells that preserve old shapes
- another planning wave that renames modules without deleting overlap

## Cleanup Thesis

Polylogue now has the right product families. It still does not have the right
topology everywhere.

The remaining cleanup burden is concentrated in four broad seams:

1. archive-product contract/store/status/debt/health/operator overlap
2. query SQL and mapper overlap around conversation/raw/product read paths
3. validation-lane declaration/execution sprawl
4. large declarative roots that still bundle unrelated template or operator
   concerns

This campaign should end with smaller public roots, fewer mixed-role modules,
and fewer places where the same archive-product meaning is translated more than
once.

## Architectural Rules

### 1. Contracts Must Not Also Store

`archive_products.py` should define contracts, not own operator-level storage
or lineage translation behavior.

### 2. Row Building Must Not Also Govern Lifecycle

`session_product_rows.py`, `session_product_storage.py`, and the status/debt
stack should each have one job.

### 3. Health, Debt, And Operator Output Must Share Inputs, Not Logic

The same underlying status/debt source can feed CLI, MCP, health, and
publication surfaces, but each surface should not re-derive the same meanings
locally.

### 4. SQL Families Need Narrow Roots

Conversation reads, raw reads, product reads, and mapper logic should not keep
accumulating into large generic query modules.

### 5. Cleanup Must Remove Real Overlap

Each phase should end with at least one deleted helper family, deleted mixed
responsibility, or materially smaller public root.

## Phase 1: Archive Product Contract And Mapper Cleanup

### Targets

- `polylogue/archive_products.py`
- `polylogue/storage/backends/queries/mappers.py`
- `polylogue/operations/archive_product_support.py`

### Main Work

- split archive-product contracts by concern family instead of keeping one
  large omnibus contract root
- narrow mapper ownership so migrated-row reconstruction, product hydration,
  and inference/enrichment fallback translation do not remain one wide band
- reduce archive-product support helpers so operator-facing transforms are not
  bundled with record-to-product mapping internals

### Acceptance

- smaller contract roots with explicit concern boundaries
- mapper logic split by product family or translation role
- archive-product support no longer mixes debt lineage parsing, product
  hydration, and generic paging helpers in one place

## Phase 2: Session Product Build/Store/Status Cleanup

### Targets

- `polylogue/storage/session_product_rows.py`
- `polylogue/storage/session_product_storage.py`
- `polylogue/storage/session_product_status.py`
- `polylogue/storage/derived_status.py`

### Main Work

- split row construction into smaller profile/event/phase/enrichment builders
- split product persistence by row family instead of one large storage module
- narrow status reporting so row counts, FTS counts, readiness, and duplicate
  detection are not all mixed together
- narrow derived-status assembly so retrieval, session-product, and health-band
  assembly are clearly separated

### Acceptance

- row-builder, storage, and status modules each have one obvious authority
- enrichment-specific status logic is not smeared across unrelated status code
- product-write paths stop depending on broad all-family helpers

## Phase 3: Health, Debt, And Operator Surface Cleanup

### Targets

- `polylogue/health_archive.py`
- `polylogue/storage/archive_debt.py`
- `polylogue/mcp/server_product_tools.py`
- `polylogue/cli/products_workflow.py`
- `polylogue/cli/products_rendering.py`

### Main Work

- narrow archive health checks into smaller product/retrieval/debt groupings
- split archive-debt computation from lineage/governance formatting where those
  concerns are still bundled
- reduce MCP product tool registration into smaller concern families
- keep CLI workflow/rendering roots small by deleting repeated list/query glue

### Acceptance

- health, debt, CLI, and MCP all read the same narrow sources instead of
  carrying duplicated transform logic
- operator/product surfaces remain feature-equivalent but materially smaller

## Phase 4: Query SQL And Raw Read Cleanup

### Targets

- `polylogue/storage/backends/queries/conversations.py`
- `polylogue/storage/backends/queries/raw.py`
- any adjacent helper roots exposed by the breakup

### Main Work

- split conversation-query families by list/filter/search/projection concern
- split raw-query families by acquisition/provenance/payload/readiness concern
- remove broad helper overlap where conversation/raw paths still duplicate date,
  provider, or payload translation logic

### Acceptance

- smaller SQL roots with clearer ownership
- fewer cross-imports between conversation and raw query modules

## Phase 5: Validation Lane Registry Cleanup

### Targets

- `devtools/run_validation_lanes.py`

### Main Work

- separate lane declarations from lane-construction helpers and canned command
  families
- group live/archive/governance/memory lanes into explicit submodules rather
  than one ever-growing registry root
- keep the operator surface identical while reducing the single-file blast
  radius

### Acceptance

- named lanes stay stable
- the registry root becomes substantially smaller and more navigable

## Phase 6: Declarative Root Cleanup

### Targets

- `polylogue/site/templates.py`
- any other declarative root that remains large only because several unrelated
  declarations still live together

### Main Work

- split site template ownership by page family or template concern
- keep large declarative assets large only when they are genuinely one family,
  not because multiple unrelated declarations were left in place

### Acceptance

- fewer mixed page-family declarations in one module
- clearer template/page ownership without changing rendered output semantics

## Validation And Closure

This cleanup program is complete only when all of the following are true:

- the main broad product/runtime roots above are materially smaller or deleted
- no new compatibility facades were added to preserve old shapes
- named validation lanes still pass after the breakup
- live archive health/debt/product status remains clean
- `earlyoom` stays quiet under the representative live lanes and memory checks

## Recommended Execution Order

1. archive product contract and mapper cleanup
2. session product build/store/status cleanup
3. health, debt, and operator surface cleanup
4. query SQL and raw read cleanup
5. validation lane registry cleanup
6. declarative root cleanup
