---
created: 2026-07-16
purpose: Hierarchical map for executing test-suite strengthening and subtraction
status: active
project: polylogue
---

# Execution map

The area packets serve two purposes: improve behavioral coverage and measure
how much current test code a stronger law actually dominates. The current
8–13k savings band covers only already-mapped slices; it does not cap the
end-state opportunity. See [`03-savings-ledger.md`](03-savings-ledger.md).

## Portfolio

| Area | Current assessment | Packet | Next useful action |
| --- | --- | --- | --- |
| Retired conductor + verification catalogs | Confirmed dead adapter; adjudicated closed loops | [`areas/devtools-verification.md`](areas/devtools-verification.md) | Remove the temporal adapter independently; design the function-level catalog simplification |
| Query grammar/execution/actions | Large mixed cluster; composition failures escaped | [`areas/query-composition.md`](areas/query-composition.md) | Build the exact-selection micro archive and independent fact oracle |
| Status and thin facades | Repeated local snapshots/forwarding tests missed cross-surface disagreement | [`areas/status-and-facades.md`](areas/status-and-facades.md) | Define one seeded state-transition table and project it through real surfaces |
| Scale and generated corpora | `polylogue-1xc.14.1` is assumed to supply bounded profiles, correlated variants, tiers, canaries, and identity | [`areas/scale-and-corpus.md`](areas/scale-and-corpus.md) | Migrate weak scale/benchmark consumers onto realized workload IDs and exact canary laws; do not design another generator |
| Harness/cache/fixtures | Shared seed can bless partial builds; duplicate corpus paths; inert helpers | [`areas/harness-and-fixtures.md`](areas/harness-and-fixtures.md) | Adapt realized workload and receipt identities into fail-closed cache publication, immutable clones, and independent facts |
| MCP and web reader | Rewrite boundaries | [`areas/rewrite-boundaries.md`](areas/rewrite-boundaries.md) | Extract obligations only; design rewrite-native suites from public contracts |
| Implementation-coupled tests across all areas | Reproducible survey population | [`census/README.md`](census/README.md) | Use findings to enrich the owning area packet, not to run a blanket deletion sweep |
| Storage/durability/rebuild | 49.2k nonblank test LOC; strong anchors plus large overlap surface | [`areas/storage-durability.md`](areas/storage-durability.md) | Prove incremental-versus-rebuild equivalence before cluster subtraction |
| Source/provider normalization | 27.7k nonblank test LOC; 60 `@given` tests plus generated laws, real-route fixtures, curated provider witnesses, and some catalogs | [`areas/source-normalization.md`](areas/source-normalization.md) | Pilot provider-neutral fact blueprints on one family plus detector ambiguity; do not treat explicit wire witnesses as generic deletion candidates |
| Daemon/convergence/write authority | 27.0k nonblank test LOC; 181 mock-interaction candidates | [`areas/daemon-convergence.md`](areas/daemon-convergence.md) | Build deterministic debt→restart→retry→quiescence scenario |

The responsibility-level authority and proof map is
[`08-capability-map.md`](08-capability-map.md). Generated cluster evidence lives
under [`dossiers/`](dossiers/) and is refreshed with `census/dossier.py`; those
artifacts report missing evidence rather than issuing deletion verdicts.
The cross-cutting historical failure corpus is
[`12-bead-regression-class-map.md`](12-bead-regression-class-map.md). Area
ownership and regression class are orthogonal: every dossier must name both.

## Sequencing

1. Remove only unambiguous debris first: the temporal-conductor adapter and its
   direct tests/docs have no dependency on the new harness architecture.
2. Simplify false verification claims while retaining live inventory ratchets,
   architecture checks, executable lanes, and mutation/benchmark receipts.
3. Reconcile the merged `polylogue-1xc.14.1`, `polylogue-1xc.14`, and
   `polylogue-b054.1.1.3`–`.5` outcomes using
   [`10-realized-baseline.md`](10-realized-baseline.md). Remove planned helpers,
   identities, tiers, canaries, and receipts now supplied by those authorities.
4. Repair only the remaining corpus cache/publication and immutable-clone
   contract. Select a realized named workload/canary and attach independent
   expected facts; do not create a parallel semantic workload profile.
5. Prove one composition slice by extending the realized C-03 exact-session
   actions canary through exact membership, projections, pages, preview/apply,
   and bounded work across stable real routes.
6. Subtract dominated query tests only after the new slice kills the known
   dropped-filter/unbounded-work mutations.
7. Reuse realized active-growing and partial-convergence variants for the
   convergence restart law and later status/facade composition.
8. Migrate correctness, scale, and benchmark consumers onto shared workload
   identities and receipts; subtract obsolete seeders only after migration.
9. Use MCP/web rewrites as clean test-design boundaries.
10. Prepare the next risk-led packets from the Bead corpus: authority/replay
    fixed point and lineage first; then deterministic concurrency,
    identity/cardinality, evidence honesty, and security non-bypass.
11. Follow with provider normalization, temporal semantics, installed-runtime
    lifecycle, configuration/catalog truth, and stable cross-surface
    composition as their production decisions settle.
12. Audit storage/source/daemon clusters before forecasting them.

Implementation agents receive generated behavior-cluster dossiers, not raw
census rows. See [`07-agent-execution-playbook.md`](07-agent-execution-playbook.md).
The controlling portfolio and orchestration corrections are in
[`14-holistic-execution-audit.md`](14-holistic-execution-audit.md).
The executable prerequisite/hotspot/model routing for all 33 laws is in
[`15-law-execution-dag-and-model-routing.md`](15-law-execution-dag-and-model-routing.md).

## Shared-worktree waves

The execution shape is deliberately asymmetric:

1. Sol/Ultra performs the pre-dispatch reconciliation gate, regenerates
   dossiers, and adjudicates exact write sets against the merged upstream
   authorities.
2. One Terra integration job lands only still-missing corpus/cache integrity,
   without broad fixture subtraction in the same job.
   If the merged workload program already supplies a planned helper, Sol
   removes that helper from the job rather than duplicating it.
3. After cache integration, disjoint Terra/high query and convergence jobs may
   run beside one Terra/high certified verifier-removal job. Luna remains out
   of coding rotation until several later bounded jobs justify calibration.
4. Editing freezes when the survivor wave finishes. Sol or a separate
   Terra/high certifier uses a disposable worktree at the frozen survivor
   revision to run the historical witness, temporary production mutation, and
   deletion-obligation review; the certifier restores a clean tree and creates
   no merge.
5. Exact-file subtraction runs only after certification. Sol then reconciles
   actual diffs and receipts, repairs composition, and owns every broad check,
   commit, and PR.
6. Storage rebuild equivalence, one provider-normalization family, and
   status/facade composition wait for current-symbol execution-grade dossiers;
   architecture defaults come from [`architecture/`](architecture/00-index.md).

Native Ultra delegation and the external manifest runner are alternative
control planes for a wave. Do not recursively fan out the same jobs through
both. Use the manifest runner when exact Terra/Luna model attestation is
required; use native Ultra subagents for adjudication or bounded jobs when
generic delegated execution is sufficient.

Workers operate directly in one clean coordinator-owned checkout but never use git, Beads, broad
formatters, generated-surface sweeps, or broad tests. Overlapping write sets or
architecture-heavy decisions use isolated worktrees instead. Prefer
serialization over a second worktree when both jobs own one hotspot and
isolation would only create merge work.

## Required evidence per area

Every completed packet should leave:

- production paths reached by the surviving tests;
- independent oracle or externally observable invariant;
- historical defect or representative mutation killed;
- unique branches intentionally retained;
- old tests/helpers removed and why the replacement dominates them;
- focused command and broader affected-area command;
- gross removal, replacement addition, and net LOC in the savings ledger.
