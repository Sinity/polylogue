# Code Navigation

This page is the shortest route from “I need to change Polylogue” to the code
that owns the change. It complements the system overview in
[Architecture](architecture.md), the durable decisions in
[Architecture Spine](architecture-spine.md), and the detailed implementation
landmarks in [Internals](internals.md).

The governing rule is simple:

> Put a change in the layer that owns its meaning, then adapt outward. Do not
> start from the CLI, daemon, or a repair command and work inward.

## Five-minute mental model

Polylogue is an evidence pipeline with split durability:

```text
provider bytes / browser capture / hook evidence
        │
        ▼
source acquisition and shape-based dispatch
        │   polylogue/sources/
        ▼
parse and normalize provider records
        │   polylogue/pipeline/ + polylogue/sources/parsers/
        ▼
durable source evidence + content-addressed blobs
        │   source.db + blob/
        ▼
archive-domain semantics and rebuildable projections
        │   polylogue/archive/ + index.db
        ▼
derived read models and convergence
        │   polylogue/insights/ + polylogue/daemon/convergence*.py
        ▼
CLI / API / MCP / HTTP / rendering surfaces
```

`polylogued` owns normal writes. `source.db`, `user.db`, and source blob bytes
are durable evidence. `index.db`, `embeddings.db`, insights, FTS, and most
status products are rebuildable. Maintenance code may verify or repair an
invariant, but it is never the normal home for new archive semantics.

## Read the code in this order

For a first repository pass, read these landmarks rather than traversing the
entire package tree:

1. [`polylogue/sources/dispatch.py`](../polylogue/sources/dispatch.py) —
   provider detection and lowering into parser-ready units.
2. [`polylogue/sources/origin_specs.py`](../polylogue/sources/origin_specs.py) —
   declared source/origin capabilities and parser bindings.
3. [`polylogue/pipeline/services/ingest_batch/`](../polylogue/pipeline/services/ingest_batch/) —
   acquire, parse, materialize, and index orchestration.
4. [`polylogue/storage/sqlite/archive_tiers/revision_governance.py`](../polylogue/storage/sqlite/archive_tiers/revision_governance.py) —
   the authority-sensitive source/index write boundary.
5. [`polylogue/storage/sqlite/archive_tiers/write.py`](../polylogue/storage/sqlite/archive_tiers/write.py) —
   normalized session writes into the rebuildable index tier.
6. [`polylogue/archive/query/expression.py`](../polylogue/archive/query/expression.py) —
   query-language semantics rather than surface parsing.
7. [`polylogue/operations/specs.py`](../polylogue/operations/specs.py) —
   declared multi-surface operations.
8. [`polylogue/daemon/convergence.py`](../polylogue/daemon/convergence.py) and
   [`convergence_stages.py`](../polylogue/daemon/convergence_stages.py) —
   bounded repair of rebuildable products after ingest.
9. [`polylogue/surfaces/payloads.py`](../polylogue/surfaces/payloads.py) —
   provider-neutral response payloads shared by public surfaces.
10. [`docs/plans/layering.yaml`](plans/layering.yaml) — enforced import and
    SQLite-writer ownership boundaries.

## Where a change belongs

| Change | First home | Required follow-through | Do not put it in |
| --- | --- | --- | --- |
| Add or change a provider/origin route | `sources/origin_specs.py`, `sources/dispatch.py`, and the owning parser | schema/package evidence, positive and collision-negative fixtures, production ingest test | a CLI switch or filename-only detector |
| Change durable source evidence | `storage/sqlite/archive_tiers/source.py` and `source_write.py` | additive source migration, backup-gated migration proof, source-authority tests | direct SQL in `pipeline/`, `daemon/`, or `maintenance/` |
| Change normalized session meaning or identity | owning parser plus `archive/` and `pipeline/ids.py` when hashes change | eager/streaming/replay equivalence, semantic fingerprint decision, candidate rebuild impact | a post-hoc repair that rewrites index rows |
| Change rebuildable index schema | `storage/sqlite/archive_tiers/index.py` and `storage/sqlite/lifecycle.py` | declared delta class, candidate/rebuild tests, generated schema docs | a durable migration chain |
| Change query semantics | `archive/query/` | SQL lowering and in-memory parity, discovery/reference regeneration, public result tests | bespoke CLI- or MCP-only filtering |
| Add a reusable operator workflow | `operations/` | operation declaration, ownership/authorization, thin CLI/API/MCP adapters | a large command handler that owns domain logic |
| Detect or repair violated invariants | `maintenance/` over typed `storage/` primitives | dry-run-first behavior, backup/ownership boundary, immutable receipt, red twin | the primary ingest/write path |
| Add a materialized derived read model | `insights/` plus `storage/insights/` | convergence stage, staleness model, rebuild and public-read tests | an ad hoc table queried only by one surface |
| Add a public payload or affordance | the owning surface package, such as `mcp/payloads.py`, then the relevant adapter | CLI/API/MCP/HTTP parity or an explicit structured exclusion | provider-specific dicts assembled independently per surface |
| Add a daemon loop | `daemon/` | ownership, bounded work, backoff, health/status evidence, interruption test | an unbounded background task with no convergence state |
| Add a cross-cutting shared type | `core/` only when it has no I/O and three or more otherwise-unrelated packages consume it | import-layer check and focused type tests | a new top-level package or loose module |
| Add a scenario, fixture, or proof | `scenarios/`, `demo/`, `tests/infra/`, or `devtools/` | production-route anti-vacuity and a named claim/invariant | hand-inserted rows that production can never create |

If a change does not fit a row, use the ordered placement decision in
[Architecture § Placement Rules](architecture.md#placement-rules). Ambiguity is
a reason to extend an existing package, not evidence that another top-level
package is needed.

## Package roles

The top-level packages are easier to understand as six roles. These are
navigation roles, not a second import policy; `docs/plans/layering.yaml` is the
enforced boundary authority.

### Foundation

- `core/` — dependency-light types, errors, enums, identity laws, and helpers.
- `declarations/` — declaration/derivation machinery shared by typed registries.
- `paths/` — canonical filesystem resolution and path sanitization.
- `artifacts/` — typed runtime-artifact descriptors and graphs.

### Acquisition and normalization

- `sources/` — source discovery, decoding, provider detection, and parsers.
- `browser_capture/` — local capture receiver and native-envelope handling.
- `hooks/` — hook evidence wiring and liveness projections.
- `pipeline/` — ingest orchestration and normalized identity construction.

### Archive substrate

- `storage/` — SQLite, schemas, migrations, queries, blobs, and writer
  primitives.
- `archive/` — archive-domain meaning over storage: identity, lineage, query,
  revision authority, and write effects.
- `operations/` — reusable multi-step workflows over archive/storage services.
- `annotations/` — schema-declared user assertions and annotation batches.
- `material_protocol/` — normalized-session interchange contract.
- `security/` — excision and secret-hygiene lifecycle.
- `sinex/` — durable publication obligations and settlement transport.

### Derived products

- `insights/` — materialized, rebuildable read models and their semantics.
- `context/` — context-oriented read views and evidence correlation.
- `cost/` — typed cost and subscription-plan computation.
- `readiness/` — consolidated capability and claim-readiness predicates.
- `product/` — executable product-workflow declarations.
- `coordination/` — coordination envelopes projected from archived evidence.

### Runtime and surfaces

- `daemon/` — the long-running writer, convergence owner, HTTP reader, and
  metrics runtime.
- `api/`, `cli/`, `mcp/` — public adapters over shared operations and payloads.
- `surfaces/` — provider-neutral payload and affordance contracts.
- `rendering/` — markdown/HTML/string rendering only.
- `agent_integration/` — packaged cold-start and native client integration.
- `telemetry/` — outbound telemetry projections.
- `ui/` — legacy terminal presentation facade retained for compatibility.

### Verification worlds

- `maintenance/` — fail-closed verification and operator-supervised repair over
  typed storage primitives; never the primary write path.
- `schemas/` — provider schema observation, inference, validation, and drift.
- `scenarios/` — reusable scenario declarations and executable workload worlds.
- `demo/` — deterministic private-data-free product demonstrations.
- `devtools/` and `tests/` — repository policy, generators, fixtures, and
  executable verification.

## Authority order

When sources disagree, use this order:

1. **Code, DDL, typed declarations, and production routes** define behavior.
2. **Versioned receipts and live evidence** establish what actually happened.
3. **Beads** owns unresolved work, dependencies, acceptance, and successors.
4. **Generated references** describe declarations and live command surfaces.
5. **Hand-written docs** explain rationale and navigation; they do not override
   code or a current receipt.
6. **Historical plans and audits** preserve context but are not current
   execution authority unless an active Bead explicitly adopts them.

Generated files say how to regenerate them in their header. Edit the source
registry or declaration, not the rendered output.

## Common traps

- **Raw SQL outside `storage/`.** Add or call a storage accessor instead.
- **Normal semantics in `maintenance/`.** Fix the write path; keep maintenance
  for diagnosis, one-shot repair, and recovery.
- **Inferring `Provider` from `Origin`.** The mapping is not injective. Preserve
  original acquisition evidence at wire boundaries.
- **Surface-specific copies of domain policy.** Put the rule in `archive/`,
  `operations/`, or a typed declaration and adapt it outward.
- **Direct writes from a CLI or worker.** The daemon and owned maintenance
  boundaries are the mutation authority.
- **Green synthetic tests standing in for a production route.** Every proof
  needs an anti-vacuity path that would fail if the production seam were
  bypassed.
- **Running operational commands from a dirty or differently pinned checkout.**
  Bind live work to the selected package SHA, archive identity, and receipt.
- **Adding another top-level package because placement is unclear.** Apply the
  decision procedure first; uncertainty usually reveals a missing boundary in
  an existing package.

## Verification by change type

| Change | Minimum focused verification |
| --- | --- |
| Documentation navigation | `devtools render docs-surface --check` |
| Package/import boundary | `devtools verify layering` |
| Durable or derived schema | `devtools lab policy schema-versioning` plus the owning migration/rebuild tests |
| CLI/API/MCP contract | owning focused tests plus generated reference checks |
| Archive invariant or maintenance route | red-twin test, real command dispatch, and receipt validation |
| Parser or identity semantics | provider fixture, eager/streaming/replay equivalence, and content-hash/fingerprint tests |
| Any merge candidate | `devtools verify --quick` plus the PR's affected-area tests |

For the complete verification model, see [Testing](../TESTING.md) and
[Developer Tools](devtools.md).
