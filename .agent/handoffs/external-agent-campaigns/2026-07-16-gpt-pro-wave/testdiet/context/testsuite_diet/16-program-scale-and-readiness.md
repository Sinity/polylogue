---
created: 2026-07-16
purpose: Quantify Testsuite Diet workload, execution shape, specification readiness, and remaining implementation decisions
status: active-readiness-assessment
project: polylogue
---

# Program scale and readiness

## Bottom line

The Testsuite Diet is a substantial multi-PR verification program, not one
large autonomous coding task. The opening wave is narrowly executable after
its upstream baseline merges; the complete 33-law portfolio is deliberately
only partially packetized.

Selecting Sol/Ultra is enough to coordinate the program, but not enough to make
stale or broad packets safe. Sol has now adjudicated the architecture defaults,
but must still reconcile source, translate each decision into exact dossiers,
and dispatch bounded Terra/high jobs. Agents should never receive “implement
the Diet” as one prompt.

## Planning and product scale

| Measure | Current scale | Meaning |
| --- | ---: | --- |
| Existing test suite | 941 Python files; 275,647 nonblank LOC | The Diet touches a very large verification surface; it does not propose rewriting it wholesale. |
| Diet Markdown | 33 files; 6,884 lines | Considerable analysis exists, but narrative volume is not execution readiness. |
| Top-level controlling analyses | 16 files; 5,253 lines | Architecture, proof-form audit, laws, orchestration, economics, and routing. |
| Area packets | 9 files; 729 lines | Subsystem ownership and preservation boundaries. |
| Prepared dossiers | 5 files; 554 lines plus generated evidence JSON | First representative clusters; all currently say `prepared-not-execution-grade`. |
| Bead-derived laws | 33 laws covering 181 explicit bug Beads | Behavioral design portfolio, not 33 mandatory tests or 33 PRs. |
| Opening manifest | 4 Terra/high jobs, 2 waves, 16 assigned files, 6 focused selectors | Concrete first implementation portfolio after reconciliation. |
| Orchestration implementation | 678-line runner, two receipt schemas, 10 passing runner tests | Operational support exists, with two pre-unattended hardening gaps. |

The historical near-term savings band is 8–13k net test LOC, only 3–5% of the
suite. It is prioritization context, not realized savings. The realized ledger
is empty. A one-third reduction (~92k lines) remains a hypothesis pending
representative cluster results by stratum.

## Workload horizons

### Horizon A — unblock and reconcile

Coordinator-only work:

1. merge and inspect `polylogue-1xc.14(.1)` and
   `polylogue-b054.1.1.3`-`.5` outcomes;
2. regenerate the five dossiers at the merged head;
3. remove Diet work already supplied upstream;
4. resolve exact symbols, workload/canary IDs, write sets, commands, and
   mutations;
5. create and content-validate the realized-baseline receipt;
6. make the runner's attested timeout/interrupt path safe for unattended use.

This is not clerical. Upstream work may substantially shrink or reshape the
foundation, devtools, and selection-proof jobs.

### Horizon B — opening representative portfolio

The prepared manifest contains:

1. one serialized corpus/cache/publication foundation job;
2. one query composition survivor job;
3. one convergence restart survivor job;
4. one tightly bounded certified temporal-adapter deletion job.

The shared-runner portion is four coding executions. The complete proof cycle
also needs approximately three to four independent certification/review
executions, one to four exact subtraction jobs depending on actual overlap,
and Sol integration. Thus the opening portfolio is roughly 8–12 bounded agent
executions plus coordinator reconciliation and publication—not merely the four
manifest rows.

The fifth representative cluster, incremental/rebuild storage equivalence,
follows after the corpus foundation and is serialized around the storage
hotspot. Once all five land, their realized economics support only
stratum-local projections.

### Horizon C — risk-led semantic portfolio

The remaining laws group naturally into roughly 12–18 cohesive implementation
clusters rather than 33 isolated jobs:

- authority/acquisition identity;
- lineage composition and snapshot consistency;
- linearizability, leases, publication, and resume;
- rebuild and freshness equivalence;
- query cardinality, algebra, cancellation, scaling, progress, and cleanup;
- convergence liveness;
- evidence/provenance/public projection;
- provider detection and semantic normalization;
- destructive boundaries, authentication, and output safety;
- configuration and temporal equivalence;
- executable inventory/tooling;
- capture and installed deployment lifecycle;
- residual testmon/xdist work only if upstream receipts leave gaps.

This 12–18 cluster figure is a planning inference from shared hotspots, not a
committed PR count. Some clusters may collapse after upstream work; provider
families, security, or installed-runtime work may split further. A typical
cluster has survivor implementation, independent certification, optional
subtraction, and coordinator integration. The full portfolio therefore likely
means several dozen focused agent executions across multiple PRs, not one
continuous swarm. No credible wall-time estimate exists until the first five
clusters record runtime, fixture builds, and coordinator overhead.

### Horizon D — rewrite-native obligations

MCP and the current web reader are rewrite boundaries. The Diet transfers
security, pagination, cancellation, typed-error, accessibility, and recovery
obligations into replacement designs. It does not forecast their implementation
or deletion workload yet. They are outside any credible current total.

## Orchestration shape

```text
Sol reconcile and adjudicate
        |
        v
Terra survivor implementation in exact disjoint files
        |
        v
editing freeze
        |
        v
different Terra certifier in disposable mutation worktree
        |
        v
Terra/Luna exact certified subtraction
        |
        v
Sol combined review, verification, PR, and realized ledger
```

Concurrency is intentionally modest:

- four workers only for read-only dossier/calibration work;
- up to three for ordinary exact-file-disjoint survivor work;
- one or two for SQLite, filesystem, or scale-sensitive work;
- one for a storage/architecture/generated-surface hotspot.

Focused tests retain the checkout lock, so parallel workers mainly overlap
source reading and editing; their test commands may queue. Broad verification
runs once at the coordinator boundary.

Shared checkout is preferred for disjoint, already-decided jobs. Worktrees are
used only for temporary mutation, overlapping files, or architecture/durability/
security changes where isolation pays for its merge cost. Jobs sharing one
hotspot are normally serialized rather than given competing worktrees.

## What each agent actually does

### Sol/Ultra coordinator

- read current source and full Bead evidence;
- resolve authoritative behavior and rewrite boundaries;
- regenerate/adjudicate dossiers and exact file ownership;
- select historical seeds, independent oracles, and production mutations;
- render and launch one control plane per wave;
- steer blockers rather than asking workers to invent decisions;
- inspect actual source/diffs/receipts, reconcile cross-cluster contracts;
- run combined focused and publish-boundary verification;
- own git, Beads, commits, PRs, merging, and realized economics.

### Terra/high survivor worker

- read only named packet/source/tests and immediate dependencies;
- edit exact assigned files in the shared checkout or assigned architecture
  worktree;
- implement one behavioral law through real production routes;
- use an independent fact/model/metamorphic/state oracle;
- run only named focused checks;
- report production dependencies, actual changes, proposed deletions,
  sensitivity state, risks, and blockers;
- leave deletion to the later certification boundary.

### Terra/high independent certifier

- start from the frozen survivor revision in a disposable worktree;
- reproduce the historical witness;
- make only the named temporary production mutation;
- prove the new law fails for the expected behavioral reason;
- review proposed deletions for unique security, compatibility, recovery, and
  diagnostic obligations;
- restore exact hashes and a clean tree; emit an attested certification
  receipt; create no commit or merge.

### Luna/high bounded worker

Luna is not in the first coding wave. After calibration proves an economic
advantage, Luna may execute certified exact-file deletion, mechanical
consolidation, localized fixtures, or dossier tooling. Semantic ambiguity
escalates to Terra; architecture/durability/rewrite conflicts escalate to Sol.

## Specification readiness

### Well specified now

- authority order between source, upstream receipts, laws, dossiers, manifests,
  and worker prose;
- 33 behavioral laws with witness, invariant, varied dimensions, proof form,
  retained seed, sensitivity mutation, and dossier home;
- survivor -> independent certification -> subtraction -> integration lifecycle;
- model roles, high reasoning requirement, concurrency profiles, worktree cost
  rule, stop/quarantine behavior, and verification ownership;
- first-wave missions, required reads, exact write/avoid files, acceptance, and
  focused commands;
- structured implementation and certification receipt shapes;
- anti-vacuity and deletion-dominance requirements;
- rewrite boundaries and savings-accounting rules;
- nine architecture decisions covering the four core and five contingent
  branches, with defaults, rejected alternatives, migration seams, proof
  obligations, and two explicit live-authority actions.

### Prepared but not executable yet

All five representative dossiers are pinned to old head
`21f78b4db2ba62ff44b5f16dfab96067bc249b4c`, declare themselves
`prepared-not-execution-grade`, lack realized sensitivity artifacts, and depend
on merged upstream receipts. Query and convergence additionally depend on the
corpus foundation. The opening manifest correctly fails because the
realized-baseline receipt does not exist.

### Still unresolved at execution level

1. **Exact post-merge source symbols and workloads.** The active schema work is
   still changing them; reconciliation must update every packet and assignment.
2. **Actual deletion sets.** Candidates exist, but almost none have independent
   sensitivity/dominance certification. This is intentional safety, not missing
   clerical detail.
3. **Later-law packets.** Source normalization is survey-ready; status/facades
   needs detailed inventory; security, authority, lineage, concurrency, and
   installed-runtime laws need current-symbol dossiers that implement the
   decisions under [`architecture/`](architecture/00-index.md).
4. **Architecture implementation decomposition.** Product semantics are now
   adjudicated. Sol still must choose coherent schema/PR slices, exact migration
   order, and current handler/symbol ownership from live source. Those are
   implementation-boundary decisions, not invitations to revisit semantics.
5. **Unattended runner safety.** Reconciliation-content validation and
   attested job-ID timeout/interruption are specified but not implemented.
6. **Certification operation.** The schema exists, but the direct attested
   disposable-worktree invocation has not yet been exercised end to end.
7. **Wall time, token cost, and PR count.** There is no honest estimate before
   baseline measurements and the first five realized clusters. The runner does
   not yet capture the telemetry needed for Luna economics.
8. **MCP/web rewrite workload.** Obligations are identified, but rewrite-native
   implementation scope and test economics are intentionally excluded.
9. **Durability of the plan itself.** `.agent/scratch/testsuite_diet` is
   ignored; it must be archived or promoted before this planning worktree is
   removed.

## Readiness verdict

The plan is strong enough to guide Sol and prevent speculative deletion, but
not ready for a single unattended “realize all of it” command.

- The opening portfolio is reasonably specified after upstream reconciliation
  and two runner hardenings.
- The first five clusters have useful packets but still need post-merge source
  resolution and real sensitivity proof.
- The full portfolio has strong behavioral and architecture design, but many
  later clusters remain Sol dossier/source-decomposition work before Terra
  implementation.
- A successful program should advance one certified cluster wave at a time,
  update realized economics, and revise the remaining portfolio from evidence.

The key anti-pattern is measuring readiness by document count. The useful unit
is a fresh dossier whose design decisions are closed enough that a worker can
either execute exact acceptance criteria or return one precise blocker.
