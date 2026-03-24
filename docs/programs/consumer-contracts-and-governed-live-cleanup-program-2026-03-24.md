[← Back to README](./README.md)

# Consumer Contracts And Governed Live Cleanup Program

Date: 2026-03-24
Status: absorbed predecessor program
Role: broader contract/governance predecessor absorbed by the replacement queue

Absorbs and extends as the live queue:

- `semantic-product-normalization-and-toolchain-convergence-program-2026-03-24.md`
- `domain-read-model-and-live-archive-stewardship-program-2026-03-24.md`
- `runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md`
- `archive-data-products-and-live-governance-program-2026-03-24.md`
- `source-boundary-and-runtime-governance-program-2026-03-23.md`
- the still-relevant contract/governance reservoir from:
  - `platform-wide-architecture-and-refactoring-program-2026-03-23.md`
  - `canonical-archive-platform-program-2026-03-19.md`
  - `testing-reliability-expansion-program-2026-03-14.md`

Prerequisite executed programs:

- `semantic-product-normalization-and-toolchain-convergence-program-2026-03-24.md`
- `domain-read-model-and-live-archive-stewardship-program-2026-03-24.md`
- `runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md`
- `archive-data-products-and-live-governance-program-2026-03-24.md`
- `source-boundary-and-runtime-governance-program-2026-03-23.md`
- `archive-intelligence-platform-convergence-program-2026-03-23.md`
- `semantic-stack-convergence-program-2026-03-23.md`

Primary design inputs:

- `../planning-and-analysis-map-2026-03-21.md`
- the now-executed semantic-product normalization campaign
- live `products debt`, `products status`, `products tags`, and `check --json`
  output after the 2026-03-24 durable-product rebuild
- downstream pressure from Lynchpin and other consumers that should stop
  hydrating raw conversations to rebuild already-materialized products

## One-Line Goal

Make Polylogue’s durable products and maintenance surfaces the canonical
ecosystem contract for downstream consumers, while turning live archive cleanup
from a preview-only burden into a governed, validated, lineage-backed operator
workflow.

## Why This Is The Right Next Campaign

The last campaign fixed the normalization and decomposition problems inside
Polylogue’s semantic/session product layer. The remaining drag is now broader
and more operational:

- downstream consumers can read durable products, but the consumer contract
  story is still uneven across CLI, library, sync, MCP, and external repos
- live archive cleanup debt is explicit and previewable, but the system still
  lacks a fully governed apply-and-validate story for destructive targets
- maintenance lineage exists, but archive stewardship is not yet a first-class
  durable operational history that downstream tools can depend on
- product/retrieval/site/report surfaces still expose overlapping slices of the
  same archive truth with slightly different framing
- the live archive is now good enough that stewardship itself should become a
  productized platform concern, not just an internal repair mechanism

## Program Thesis

Polylogue should now converge around four explicit platform truths:

1. durable products are the archive API, not merely one rendering of it
2. live cleanup is a governed control plane with durable lineage and explicit
   validation, not a hidden side-effect of maintenance commands
3. downstream consumers should page stable product/read-model contracts instead
   of rebuilding semantics from raw conversations
4. archive stewardship signals should compose with retrieval, publication, and
   external analytics as one reusable operational substrate

## Architectural Rules

### 1. Durable Products Are The Public Default

If a consumer can be served by a durable product or archive-debt/readiness
surface, that should be the first-class contract. Raw conversation hydration is
the fallback, not the primary integration mode.

### 2. Destructive Cleanup Must Stay Governed

Preview, apply, validation, and resulting debt-state transitions must remain
separate and inspectable. Cleanup must not become an opaque helper under
general “repair.”

### 3. One Contract Per Semantic Product Family

Profiles, phases, work events, tags, summaries, maintenance runs, archive debt,
provider analytics, and any new stewardship products should have one canonical
contract reused across CLI, sync, library, MCP, and downstream consumption.

### 4. Live Archive History Must Become Queryable Stewardship State

Maintenance lineage and cleanup outcomes should not remain buried in operator
logs or one-off JSON output. Archive stewardship needs durable, queryable
history.

### 5. Validation Must Stay Live-Attached

This campaign is not done if only local tests pass. The real archive must stay
part of the closure mechanism through named lanes and bounded live workflows.

## Phase 1: Downstream Consumer Contract Convergence

### Goal

Make durable products the explicit integration path for external consumers and
remove remaining raw-conversation rebuilding where the product API is already
sufficient.

### Main Work

- audit current CLI/library/sync/MCP/export surfaces for raw-hydration fallback
  where durable product paging should exist
- widen product query/filter coverage where needed for external consumers:
  provider, canonical session date, first message time, work kind, project,
  repo, and readiness/debt scope
- harden pagination, filtering, and machine envelopes for the product families
  most likely to be consumed externally
- tighten the library docs and operator examples around those durable surfaces

### Acceptance Criteria

- external consumers no longer need private reconstruction for common
  session/profile/timeline/product use cases
- CLI, library, sync, and MCP expose aligned filters and payload structure for
  the same durable products
- live dogfooding and targeted downstream smoke can stay on durable product
  contracts

## Phase 2: Governed Cleanup Apply And Validation Workflow

### Goal

Turn live archive cleanup from preview-only governance into an explicit
apply-and-validate control plane.

### Main Work

- separate cleanup preview, apply, and validation result modeling more clearly
- persist durable post-apply validation lineage for destructive targets
- define cleanup-state transitions such as:
  - `unreviewed`
  - `previewed`
  - `applied`
  - `validated`
  - `regressed`
- add operator surfaces for:
  - targeted apply with explicit target selection
  - post-apply validation summaries
  - debt-state inspection by target/category
- keep destructive execution gated by explicit operator intent; no silent
  cleanup under generic repair paths

### Acceptance Criteria

- destructive cleanup has durable apply/validate lineage beyond preview records
- debt-state transitions are explicit and queryable
- the operator can distinguish safe repairs from destructive cleanup at every
  surface

## Phase 3: Archive Stewardship History As Durable Products

### Goal

Treat archive stewardship itself as a durable product family rather than only
as health/check output.

### Main Work

- design durable stewardship products for:
  - maintenance runs
  - cleanup apply/validation history
  - debt-state snapshots
  - derived-model readiness snapshots
- expose them consistently through products/library/sync/MCP surfaces
- ensure publication and site/report surfaces can consume those products
  directly instead of local reshaping

### Acceptance Criteria

- archive stewardship history is queryable and versioned like other product
  families
- operator and external-consumer surfaces can page stewardship history without
  parsing ad hoc check output

## Phase 4: Retrieval, Product, And Publication Convergence

### Goal

Reduce the remaining overlap where retrieval, grouped stats, products, and
publication surfaces each carry slightly different archive summaries.

### Main Work

- revisit grouped stats, provider analytics, archive debt, and publication
  summaries to ensure they consume the same durable product/readiness sources
- align site/publication summaries with archive stewardship and product
  contracts
- revisit embeddings/retrieval health where it should surface as stewardship or
  provider analytics rather than isolated status text

### Acceptance Criteria

- grouped stats, products, and publication summaries agree on the same archive
  truth
- archive health and product analytics stop drifting by output surface

## Phase 5: Provider And Ingestion Governance Convergence

### Goal

Tie source/provider refresh semantics more explicitly to the durable products
and stewardship plane.

### Main Work

- revisit provider/parser refresh points where a live rebuild or repair changes
  product truth but provenance does not make that obvious downstream
- connect source refresh/backfill history to stewardship lineage where
  appropriate
- ensure provider-specific cleanup or refresh steps do not bypass the durable
  product/control-plane story

### Acceptance Criteria

- refresh/reparse/rebuild actions are easier to trace from source to product
- live archive changes caused by source work are visible in stewardship history

## Phase 6: Validation, Memory, And Live Governance Expansion

### Goal

Keep this campaign attached to live proof, explicit budgets, and durable
operator lanes.

### Main Work

- add named lanes for:
  - consumer contract convergence
  - cleanup apply/validation workflows
  - stewardship history surfaces
  - bounded live cleanup governance
- keep memory budgets on the heaviest consumer/product/governance workflows
- run live archive dogfooding as part of closure, not as a postscript

### Acceptance Criteria

- the campaign has named local and live validation lanes
- destructive cleanup workflows are validated with bounded live proofs
- consumer-facing product contracts are exercised against the real archive

## Execution Order

1. downstream consumer contract convergence
2. governed cleanup apply and validation workflow
3. archive stewardship history as durable products
4. retrieval, product, and publication convergence
5. provider and ingestion governance convergence
6. validation, memory, and live governance expansion

## Definition Of Done

This campaign is done when:

- downstream consumers can stay on durable product contracts for the major
  session/timeline/stewardship use cases
- destructive cleanup has durable apply/validation lineage and explicit
  governance stages
- archive stewardship history is a first-class durable product family
- publication/retrieval/product analytics compose one canonical archive truth
- named validation lanes prove the campaign locally and against the live
  archive
