---
created: 2026-07-16
purpose: Turn historical and active bug Beads into reusable behavioral test obligations
status: active-planning-map
project: polylogue
---

# Bead-derived regression classes

This map treats bug Beads as a design corpus for stronger tests. It does not
declare bugs covered, score subsystems, or require one test per Bead. The
unit of planning is a **failure class**: an invariant and state space broad
enough that the recorded bug is one witness among many possible violations.

The execution-level processing is in
[`13-bead-derived-test-laws.md`](13-bead-derived-test-laws.md): 33 concrete laws
with incident witnesses, invariants, dimensions, proof forms, retained seeds,
sensitivity mutations, and dossier homes. The class map below is navigation
and audit coverage, not the finished transformation.

The 2026-07-16 snapshot contains 181 explicit `issue_type=bug` Beads: 133
closed, 44 open, and 4 in progress. Closed bugs are especially valuable
because they provide historical pre-fix behavior and often a known production
mutation. Open bugs remain implementation obligations; listing one here does
not change its status or claim that a test already exists.

This audit is deliberately narrower than “every Bead that describes a
defect.” Features and tasks can also contain escaped-bug evidence. Cluster
dossiers must still search their owning Beads and git history.

## The TDD-shaped conversion rule

For every bug used by a cluster dossier:

1. Reproduce the concrete failure through the authoritative production route.
   Preserve the smallest deterministic witness, including its exact provider
   shape, interleaving, archive state, or workload scale.
2. State the invariant without implementation nouns. “The cursor helper calls
   `set` once” is not an invariant; “concurrent failure increments are never
   lost” is.
3. Identify the varying dimensions that made the failure possible: operation
   order, retry/crash point, duplicate cardinality, missing evidence, origin,
   surface, clock representation, archive size, or configuration layer.
4. Implement the strongest economical form: differential history, rule-based
   state machine, metamorphic relation, deterministic schedule, fault-point
   matrix, independent fact algebra, or bounded-work receipt.
5. Keep the historical witness as a named regression seed or Hypothesis
   example. Generalization must not erase a useful diagnostic reproduction.
6. Prove sensitivity against the historical parent, a temporary representative
   production mutation, or a focused mutation survivor.
7. Only then remove examples dominated by the class law. Retain distinct
   security, compatibility, recovery, and diagnostic witnesses.

A narrow example remains correct when the bug depends on a unique external
wire shape or error. The mistake is not having examples; it is mistaking one
example for proof of an invariant with a larger state space.

## Regression-class portfolio

| ID | Failure class and invariant | Preferred generalized proof | Present Diet home and remaining planning consequence |
| --- | --- | --- | --- |
| R01 | **Authority and replay fixed point.** Accepted evidence has one explainable authority; order, retry, and replay reach the same stable terminal state without resurrecting terminal work. | Generate equivalent raw/revision histories, permute acquisition and replacement order, run production reconciliation to quiescence, and compare accepted heads, receipts, debt, and a zero-work second pass. | Storage/convergence packets partially cover restart, but not the full authority lattice. Add an authority dossier after the realized raw-replay work lands. |
| R02 | **Identity and relational cardinality.** Stable logical entities neither collide nor fan out; duplicates, missing IDs, aliases, and generation changes preserve intended one-to-one/one-to-many semantics. | Independent relation facts over duplicate/missing/reordered identifiers; assert identity stability, partition conservation, and exact cardinality across rebuilds. | Query and storage packets cover portions. Make identity/cardinality an explicit obligation in query, lineage, embeddings, and assertion dossiers. |
| R03 | **Atomicity, lost updates, and snapshot isolation.** Each concurrent history is linearizable or reports a conflict; readers see one coherent snapshot. | Deterministic barriers at the production read/write seam, a small schedule matrix, and durable post-state facts; include rollback and stale-writer schedules. | Storage durability mentions failpoints but not a reusable concurrency lane. Prepare a Terra-owned deterministic-interleaving dossier. |
| R04 | **Crash, rollback, publication, and exact resume.** Every failpoint exposes either the valid old state or valid new state; retry resumes exactly and converges without leaks. | Process-level or transaction failpoint matrix with reopen, receipt, reservation/lease, sidecar, and retry checks. | Seeded-artifact integrity and storage equivalence cover parts. Extend both with shared failpoint and reopen obligations rather than local exception mocks. |
| R05 | **Incremental, targeted, and rebuild equivalence.** Full rebuild, refresh, async/sync, and targeted paths derive the same scoped public facts from the same authority. | Twin archives or one cloned archive, execute alternate production histories, then compare independent fact sets and untouched-scope sentinels. | One of the five prepared dossiers; add explicit targeted-scope and alternate-path dimensions. |
| R06 | **Freshness, debt, retry, and quiescence.** Work is neither falsely green nor permanently pending; freshness is monotonic and debt has a live feeder, bounded retry, and terminal explanation. | Rule-based stage/debt state machine across failures and restart, with freshness generations and a zero-work fixed-point receipt. | Convergence dossier is the main home; incorporate FTS, embeddings, insights, poisoned work, and per-unit receipts as variants, not separate mock-order tests. |
| R07 | **Query algebra, order, pagination, and work.** Selection, count, grouping, projection, page concatenation, and preview/apply agree; irrelevant archive growth does not change answers or exceed the stated work bound. | Independent planted fact algebra plus metamorphic query rewrites, page partitioning, duplicate relations, and VM/work receipts. | Exact query selection/work is prepared; broaden C-03 from one regression to these algebraic dimensions. |
| R08 | **Resource bounds, cancellation, progress, and cleanup.** Work stays bounded, can be interrupted, reports real progress, and releases processes, connections, locks, and memory. | Scale/selectivity tiers with resource receipts, cancellation/deadline injection, progress-event liveness, and cleanup assertions after every outcome. | Scale/work and verification dossiers cover pieces. Add query interruption and daemon catch-up envelopes before claiming the lane complete. |
| R09 | **Completeness, unknowns, provenance, and evidence honesty.** Missing, partial, approximate, unsupported, and terminal states never collapse into zero, ready, exact, or provider-authored facts. | Generate evidence-presence combinations and assert monotone truth projections: removing evidence cannot increase certainty or completeness; every quantitative/public claim carries its weakest provenance. | Status/facades and source packets need this as a shared fact algebra. It is not yet a decision-complete dossier. |
| R10 | **Detection, parsing, normalization, and material fidelity.** Every meaningful structural element survives detection and lowering, and ambiguous shapes are claimed only by the tightest valid parser. | Provider-neutral semantic facts plus curated real wire witnesses, ambiguity negatives, encoding/content-shape variants, and detector-order mutations. | Source-normalization packet is the home. Preserve explicit wire examples while adding one reusable blueprint per provider family. |
| R11 | **Cross-surface and protocol parity.** Stable surfaces expose the same selection, facts, typed errors, phase, and budget semantics while preserving surface-specific presentation. | Run one seeded state/query through repository, CLI, daemon HTTP, and stable adapters; compare a fact algebra rather than serialized snapshots. | Status/facades and query packets cover stable surfaces. MCP/web obligations remain rewrite inputs rather than renovation targets. |
| R12 | **Security, privacy, and destructive non-bypass.** Every alternate entry point enforces authentication, authorization, confirmation, excision, escaping, and sensitive-value suppression. | Entry-point matrix over adversarial payload strategies and bypass paths; temporarily remove one chokepoint or escaping rule to prove sensitivity. | No dedicated Diet packet exists. Prepare a security-boundary dossier; do not fold these obligations into generic query or parser tests. |
| R13 | **Installed runtime and capture lifecycle.** Pairing, restart, reconnect, queueing, stale contracts, exit attribution, and deployed-version/freshness claims work on the production-valid service profile. | Lifecycle scenario against production-valid services with stable identities, network/process fault injection, durable delivery receipts, and deployed artifact attestation. | Daemon packet covers convergence but not the external lifecycle. Add an installed-route dossier after service profiles become executable. |
| R14 | **Configuration, inventory, executable catalogs, and documentation truth.** Layered configuration composes predictably; one canonical inventory drives executable capabilities and user claims. | Metamorphic layer precedence/deep-merge tests, fresh-versus-upgraded inventory comparison, and invoke-every-advertised-operation checks through production dispatch. | Devtools subtraction must retain executable inventory ratchets. Add configuration composition and catalog-dispatch dossiers; forbid self-authored mirror catalogs. |
| R15 | **Harness selection, isolation, and nondeterministic execution.** The harness selects real dependents, attributes failures, isolates artifacts, and detects hangs under both isolated and parallel schedules. | Real production mutation/testmon selection, repeated isolated/xdist schedules, test-event liveness, exact receipt identity, and fresh-checkout runs. | The realized `b054.1.1` program is authoritative. Diet consumes its receipts and removes weaker declaration loops; it does not rebuild this machinery. |
| R16 | **Temporal representation and ordering.** Equivalent instants compare equally across epoch zero, missing fields, naive/aware values, timezone offsets, and relative clocks; timeless data remains explicitly representable. | Metamorphic instant representations, frozen-clock relative queries, ordering/page invariants, and null/timeless partitions through production SQL and surfaces. | Scattered source/query/insight tests exist, but there is no cross-route temporal dossier. Prepare one after central time predicates stabilize. |
| R17 | **Lineage composition and topology.** Equivalent lineage histories produce the same divergent tails, branch points, composed transcript, logical accounting, and completeness regardless of ingest/replacement order. | Rule-based lineage state machine with permuted parent/child arrival, replacement, compaction, dangling links, depth bounds, concurrent reads, and direct independent transcript facts. | Existing lineage Beads already specify strong ingredients. Add a dedicated architecture-heavy Sol/Terra dossier; do not bury lineage inside generic storage equivalence. |

## What this changes in the execution order

The first five dossiers remain useful and should land before projecting
savings, but they are not a holistic correctness program. They principally
exercise R04–R08 and R15. After them, dossier preparation should be ordered by
escaped-risk leverage rather than test LOC:

1. R01 authority/replay fixed point and R17 lineage composition;
2. R03 deterministic concurrency and R02 identity/cardinality;
3. R09 evidence honesty and R12 security non-bypass;
4. R10 one provider-normalization family and R16 temporal semantics;
5. R13 installed runtime lifecycle and R14 configuration/catalog truth;
6. R11 stable cross-surface composition after the underlying fact laws exist.

This is dossier order, not permission to ignore the priorities and dependency
graphs in Beads. Sol reconciles each packet against current source and the full
Bead thread immediately before dispatch.

## Primary mapping of the 181 explicit bug Beads

Each Bead appears once below as its primary regression-design home. A bug can
exercise secondary classes; cluster dossiers should record those locally. This
appendix is an audit trail for the analysis, not an allowlist or CI gate.

- **R01 Authority/replay fixed point:** `polylogue-hjpx`, `polylogue-25vy`, `polylogue-lkrc.3`, `polylogue-lkrc.2`, `polylogue-lkrc.1`, `polylogue-lkrc`, `polylogue-yla8.10`, `polylogue-yla8.9`, `polylogue-rgh2`, `polylogue-yla8.6`, `polylogue-yla8.5`, `polylogue-yla8.4`, `polylogue-yla8.2`, `polylogue-yla8.1`, `polylogue-yla8`, `polylogue-lkrc.4`, `polylogue-57rp`, `polylogue-t0dy`, `polylogue-0mu`.
- **R02 Identity/cardinality:** `polylogue-sjf6`, `polylogue-fmob`, `polylogue-xnkf`, `polylogue-tilk`, `polylogue-8k91`, `polylogue-tbe5`, `polylogue-f2qv.1`, `polylogue-85z0`.
- **R03 Atomicity/interleavings:** `polylogue-41ow`, `polylogue-hleq`, `polylogue-n2wy`, `polylogue-8jg9.4`, `polylogue-mpig`, `polylogue-y337`, `polylogue-qug2`, `polylogue-4ts.4`.
- **R04 Crash/rollback/resume:** `polylogue-7ufv`, `polylogue-rze2`, `polylogue-b08j`, `polylogue-8jg9.5`, `polylogue-kwlu`, `polylogue-b5l.1`, `polylogue-v7e0`, `polylogue-1xc.4`, `polylogue-1xc.1`, `polylogue-qs0a`, `polylogue-0puw`.
- **R05 Alternate-path equivalence:** `polylogue-61zb`, `polylogue-oucx`, `polylogue-y964`, `polylogue-1xc.2`, `polylogue-lyv4`, `polylogue-a7xr.2`.
- **R06 Freshness/debt/quiescence:** `polylogue-b5l.2`, `polylogue-wmsc`, `polylogue-1xc.12`, `polylogue-vwsv`, `polylogue-f2qv.5`, `polylogue-1xc.3`, `polylogue-x1uh`, `polylogue-1dk1`, `polylogue-n846`, `polylogue-1xc.11`, `polylogue-5vbs`.
- **R07 Query algebra/work:** `polylogue-z9gh.2`, `polylogue-1vv`, `polylogue-u0dm`, `polylogue-20d.4`.
- **R08 Resources/progress/cleanup:** `polylogue-z9gh.1`, `polylogue-cnaj`, `polylogue-w79`, `polylogue-20d.17`, `polylogue-rgbj`, `polylogue-09rn`, `polylogue-ng9m`, `polylogue-dlmv`, `polylogue-xy95`, `polylogue-3wb`, `polylogue-qhk`, `polylogue-35d`, `polylogue-s7ae.8`, `polylogue-zdeo`, `polylogue-1xc.6`, `polylogue-k8k`, `polylogue-a7xr.1`.
- **R09 Completeness/provenance:** `polylogue-7ry`, `polylogue-f2qv.6`, `polylogue-b2r9`, `polylogue-9e5.30`, `polylogue-cpf.5`, `polylogue-9e5.29`, `polylogue-07hj`, `polylogue-4iv`, `polylogue-4bu`, `polylogue-z9gh.6`, `polylogue-egm8`.
- **R10 Parse/normalize fidelity:** `polylogue-yla8.3`, `polylogue-z9gh.5`, `polylogue-j2zz`, `polylogue-ih67`, `polylogue-1frn`, `polylogue-t0p.1`, `polylogue-83u.1`, `polylogue-fs1.1`, `polylogue-segf`, `polylogue-kixp`, `polylogue-qda`, `polylogue-g99u`, `polylogue-a7xr.3`, `polylogue-tf0e`.
- **R11 Cross-surface parity:** `polylogue-rsad`, `polylogue-s7ae.7`, `polylogue-g9j6`, `polylogue-4pm`, `polylogue-bby.7`, `polylogue-6o9b`, `polylogue-f57q`, `polylogue-vh57`.
- **R12 Security/privacy/destructive boundaries:** `polylogue-layg`, `polylogue-jlme.2`, `polylogue-1xc.14.1.2`, `polylogue-5k5l.1`, `polylogue-6jjv`, `polylogue-2n39`, `polylogue-gnie`, `polylogue-kwsb.1`, `polylogue-jnj.5`, `polylogue-jn40`.
- **R13 Runtime/capture lifecycle:** `polylogue-jlme.6`, `polylogue-s2x7`, `polylogue-jlme.5`, `polylogue-jlme.3`, `polylogue-r4no`, `polylogue-7s57.1`, `polylogue-qvgt`, `polylogue-k2m`, `polylogue-enj7`, `polylogue-6rvt`, `polylogue-peo`, `polylogue-s8q`.
- **R14 Config/inventory/executable truth:** `polylogue-nkmy`, `polylogue-fd2s`, `polylogue-j9dt`, `polylogue-9itr`, `polylogue-71ey`, `polylogue-gxjh`, `polylogue-9e5.28`, `polylogue-cxlk`, `polylogue-nj80`, `polylogue-ihp0`, `polylogue-w379`, `polylogue-iyew`, `polylogue-rzve`, `polylogue-mhx.7`, `polylogue-1ty`, `polylogue-tsk`, `polylogue-l8ee`, `polylogue-vt0m`, `polylogue-gxly`, `polylogue-at44`.
- **R15 Harness selection/isolation:** `polylogue-b054.1.1.1`, `polylogue-88jp.1`, `polylogue-b054.1.1`, `polylogue-ra3w`, `polylogue-r3o3`, `polylogue-ooqh`, `polylogue-p5li`, `polylogue-27rb`, `polylogue-nu2h`.
- **R16 Temporal semantics:** `polylogue-cpf.6`, `polylogue-rvtu`, `polylogue-z29t`, `polylogue-2kvn`, `polylogue-2seq`, `polylogue-s5mm`, `polylogue-a7xr.6`.
- **R17 Lineage/topology:** `polylogue-866e`, `polylogue-9p0y`, `polylogue-4ts.3`, `polylogue-4ts.2`, `polylogue-5q2u`, `polylogue-4ts.6`.

## Interpretation limits

This program can substantially reduce runtime-discovered correctness bugs, but
it cannot make runtime observation obsolete. Provider format changes, browser
policies, kernel/filesystem behavior, packaging, and real archive scale remain
partly open systems. The right goal is that runtime canaries and dogfood find
new environmental facts, while known semantic classes are already guarded by
small, sensitive, production-route laws.
