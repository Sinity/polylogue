# 149. polylogue-yeq — Advanced verification lanes: metamorphic DSL, daemon chaos, API-contract walks

Priority/type/status: **P2 / task / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Beyond coverage-by-example, three method upgrades the suite lacks: (1) METAMORPHIC testing for the query engine — the DSL has algebraic laws (filter commutativity, pipeline-stage composition vs post-hoc filtering, unit-count consistency between find and aggregate paths) that hold for ALL queries, not just the ones we thought to write; hypothesis can generate queries from the grammar and assert the laws, catching lowering bugs example tests never will (the sessions-vs-observed-events pipeline inconsistency from the live probe is exactly this class). (2) CHAOS lane for the daemon — the design docs argue crash-safety (lease sweeps, WAL, trigger-drop atomicity, orphan cleanup) but no test kills the daemon mid-operation and asserts the invariants; the silent daemon deaths observed live make this concrete. (3) CONTRACT WALKS over HTTP/MCP — the bby.7 list-vs-detail break survived because nothing walks emitted payloads back into parameter positions systematically (that bead adds the sessions walk; generalize the method to every list->detail pair and every MCP tool result carrying refs).

## Existing design note

(1) Metamorphic lane: a hypothesis strategy over the registry grammar (bounded depth, seeded corpus) + ~8 laws as properties: predicate-order invariance, LIMIT monotonicity, group-by-count sums equal ungrouped count, unit-where result parity with equivalent find-mode filters, measure composition associativity once 9l5.7 lands. Laws ARE the spec — failures are either engine bugs or documented non-laws (record which, in the support matrix fnm.11). (2) Chaos lane (marked, serial, Linux-only): spawn polylogued against a scratch archive, inject SIGKILL at staged points (mid-ingest-batch, mid-FTS-rebuild, between lease-acquire and commit — the code has natural hook points via the stage events), restart, assert: no lost committed sessions, no orphan leases after sweep, FTS consistent or honestly non-ready, convergence resumes. Reuses the supervisor machinery from devtools verify. (3) Ref-walk contract lane: for each daemon list route and MCP list tool, walk every emitted ref/id into its detail routes; generated from the route/tool registries (declare-once). All three land as devtools lab lanes runnable independently; CI runs metamorphic + ref-walk per-PR (fast), chaos on schedule.

## Acceptance criteria

Metamorphic lane finds-or-proves: run against the current engine and either file real bugs or commit the laws as green properties; chaos lane demonstrates one seeded crash-recovery invariant per staged kill point; ref-walk lane covers 100% of list-emitting routes/tools and fails on a deliberately broken ref in a demonstration commit.

## Static mechanism / likely defect

Issue description localizes the mechanism: Beyond coverage-by-example, three method upgrades the suite lacks: (1) METAMORPHIC testing for the query engine — the DSL has algebraic laws (filter commutativity, pipeline-stage composition vs post-hoc filtering, unit-count consistency between find and aggregate paths) that hold for ALL queries, not just the ones we thought to write; hypothesis can generate queries from the grammar and assert the laws, catching lowering bugs example tests never will (the sessions-vs-observed-events pipeline inconsistency from the… Design direction: (1) Metamorphic lane: a hypothesis strategy over the registry grammar (bounded depth, seeded corpus) + ~8 laws as properties: predicate-order invariance, LIMIT monotonicity, group-by-count sums equal ungrouped count, unit-where result parity with equivalent find-mode filters, measure composition associativity once 9l5.7 lands. Laws ARE the spec — failures are either engine bugs or documented non-laws (record which, …

## Source anchors to inspect first

- `CONTRIBUTING.md:102` — Derived-tier schema changes require rebuild/blue-green planning.
- `AGENTS.md:168` — Agent guidance says schema mismatch should rebuild or blue-green-replace derived tiers.
- `polylogue/cli/commands/reset.py` — Current reset/rebuild commands are the operator path to replace derived tiers.
- `polylogue/daemon/convergence_stages.py` — Daemon convergence/readiness state should represent generation progress honestly.
- `polylogue/storage/sqlite/archive_tiers/index.py:296` — messages_fts table DDL lives in archive_tiers/index.py.
- `polylogue/storage/fts/sql.py:11` — messages_fts DDL copy also exists in storage/fts/sql.py.
- `polylogue/storage/fts/fts_lifecycle.py:292` — ensure_fts_triggers_sync owns runtime repair/recreation.
- `polylogue/storage/fts/fts_lifecycle.py:512` — Thread FTS rebuild logic is duplicated in lifecycle repair path.
- `polylogue/daemon/convergence_stages.py:988` — daemon convergence has another repair/readiness path for archive messages_fts.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. (1) Metamorphic lane: a hypothesis strategy over the registry grammar (bounded depth, seeded corpus) + ~8 laws as properties: predicate-order invariance, LIMIT monotonicity, group-by-count sums equal ungrouped count, unit-where result parity with equivalent find-mode filters, measure composition associativity once 9l5.7 lands.
2. Laws ARE the spec — failures are either engine bugs or documented non-laws (record which, in the support matrix fnm.11).
3. (2) Chaos lane (marked, serial, Linux-only): spawn polylogued against a scratch archive, inject SIGKILL at staged points (mid-ingest-batch, mid-FTS-rebuild, between lease-acquire and commit — the code has natural hook points via the stage events), restart, assert: no lost committed sessions, no orphan leases after sweep, FTS consistent or honestly non-ready, convergence resumes.
4. Reuses the supervisor machinery from devtools verify.
5. (3) Ref-walk contract lane: for each daemon list route and MCP list tool, walk every emitted ref/id into its detail routes
6. generated from the route/tool registries (declare-once).
7. All three land as devtools lab lanes runnable independently

## Tests to add

- Acceptance proof: Metamorphic lane finds-or-proves: run against the current engine and either file real bugs or commit the laws as green properties
- Acceptance proof: chaos lane demonstrates one seeded crash-recovery invariant per staged kill point
- Acceptance proof: ref-walk lane covers 100% of list-emitting routes/tools and fails on a deliberately broken ref in a demonstration commit.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
