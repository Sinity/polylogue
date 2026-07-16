---
created: 2026-07-16
purpose: Route test-suite diet work by behavioral responsibility and production authority
status: active-planning-map
project: polylogue
---

# Behavioral capability map

This map answers “where does the behavior actually live, what can fail, and
what is the strongest present proof?” It is a planning index, never a coverage
catalog, completeness gate, or quality score. A row with strong proof can still
hide a composition escape; a row with weak proof does not authorize deletion.
The cross-cutting regression mechanisms are defined in
[`12-bead-regression-class-map.md`](12-bead-regression-class-map.md); the two
maps must be reconciled when preparing a dossier because subsystem ownership
alone does not expose repeated failure classes.

| Responsibility | Authoritative production route | Important failure modes | Strongest existing proof | Known escapes | Intended stronger form |
| --- | --- | --- | --- | --- | --- |
| Provider normalization | `polylogue/sources/dispatch.py:detect_provider`, `_lower_payload_specs`, `_parse_lowered_spec`; provider parsers under `polylogue/sources/` | loose detector steals tighter shape, loss of provider fields, wrong material origin, bundle/session split drift | parser fixtures, schema-driven `SyntheticCorpus`, provider semantic-fact comparisons | provider examples rarely compose detector ambiguity, bundle lowering, and downstream facts | one provider-family blueprint through real detection/lowering/parser with ambiguity negatives and independent normalized facts |
| Ingestion | `polylogue/pipeline/services/ingest_batch/`; `polylogue/operations/specs.py`; source/index writer boundary | acquire succeeds but parse fails, partial artifact blessed, content-hash skip masks changed material, single-writer violation | ingest integration cases plus assumed realized workload identities/canaries/receipts from `1xc.14` | seeded corpus can still convert per-file errors to warnings or publish a sentinel independently of workload generation | thin content-addressed cache adapter over a realized canary through acquire→parse→write, with independent facts and fail-closed publication |
| Durable state | source/user DDL and migrations under `polylogue/storage/sqlite/`; archive writes in `archive_tiers/write.py` | partial transaction, destructive migration, backup mismatch, blob/reference crash window, user overlay loss | migration/backup tests, blob GC/integrity tests, write-path state machine | many narrow row/SQL checks overlap without proving recovery composition | failpoint/state-machine laws at transaction boundaries plus public/durable fact equality |
| Derived rebuilds | canonical index DDL, archive materializers, `DaemonConverger`, reset/rebuild operations | stale FTS/insights, replay omission/order drift, derived state incorporates user identity, incremental and rebuild disagree | FTS/insight rebuild tests, fast-forward probes, deterministic rebuild Beads | no one semantic corpus compares all public facts after incremental and source replay | incremental-versus-rebuild equality over planted public facts, content identities, FTS and insights |
| Query semantics | `polylogue/archive/query/expression.py`; lowering/execution modules; repository query reads | dropped predicate, shadow DTO divergence, wrong exact/native identity, count/page mismatch, unbounded irrelevant work | grammar suite, repository reads, cross-surface adapters, and assumed realized C-03 exact-session actions canary | partial shadow translations omit units/stages/projections/current origin; mock forwarding passes while selection drifts | extend C-03 to independent membership/count/partition/page and preview/apply laws through real surfaces plus its work bound |
| Daemon convergence | `polylogue/daemon/convergence.py:DaemonConverger`; `convergence_stages.py`; convergence-debt queries | pending treated as failure, debt lost across restart, duplicate retry, hot-file starvation, unscoped retry | focused stage tests and daemon integration cases | internal await/call-order tests do not prove durable state across restart | deterministic real-SQLite debt→restart→retry→quiescence scenario compared with uninterrupted execution |
| Public projections | repository/API facade, CLI renderers, daemon HTTP, MCP registry and tools | surfaces disagree on selection/status, internal provider vocabulary leaks, payload exactness loses evidence, facade forwards wrong authority | payload contracts, CLI/HTTP tests, selected cross-surface facts | current MCP and web reader are rewrite boundaries; local snapshots can agree with their own mocks | public fact algebra projected through stable surfaces; carry MCP/web obligations into rewrite-native suites |
| Verification tooling | `devtools/run_tests.py`, `pytest_supervisor.py`, `verify_runs.py`, live inventory/docs/layering checks, mutation/benchmark runners | testmon blind spots, partial seed accepted, self-authored catalog proves itself, stale receipts, verifier claims behavior without executing it | managed runner plus assumed realized complete seed receipts, real production-mutation/testmon proof, and repeated isolated/xdist witnesses from `b054.1.1.3`–`.5` | closure/scenario/manifest mirrors and temporal conductor pin declarations or retired paths | consume the shared receipts and executable proofs; subtract declaration loops; use generated cluster dossiers only for planning |

## Cross-cutting rules

- Identity, durability, security, compatibility, recovery, and diagnostics are
  separate obligations even when they traverse the same code.
- A real route plus an independent oracle is stronger than another adapter
  model that translates the same assumptions.
- MCP and the current web reader are rewrite boundaries: preserve their
  external obligations, do not renovate implementation-coupled tests.
- The realized `1xc.14.1` workload-profile branch owns workload semantics,
  scale tiers, C-03, identities, and shared receipts. Diet owns only missing
  cache/publication adaptation and independent behavioral laws.
- Every survivor-execution-grade dossier names its primary and secondary R-classes,
  preserves the concrete historical witness, and tests the broader invariant.
  A Bead ID by itself is evidence routing, not a test design.
- The first five dossiers concentrate on crash/rebuild/convergence/query/work
  and harness proof (R04–R08 and R15). Authority, identity, concurrency,
  evidence honesty, security, external lifecycle, configuration, temporal, and
  lineage classes still require decision-complete packets before holistic
  coverage can be claimed.
