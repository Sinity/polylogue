---
created: 2026-07-16
purpose: Let implementation agents strengthen and shrink the suite by behavioral clusters
status: design-ready
project: polylogue
---

# Agent execution playbook

## Unit of work: a behavioral cluster

Never assign “clean up test file X” or “review the next 50 census hits.” Assign
one externally meaningful responsibility and every test/helper that currently
claims it. Examples:

- exact selection across parse → lower → SQL → repository/CLI/HTTP;
- ingest/reingest idempotency across source/index tiers;
- daemon convergence across interruption/restart;
- status truth across storage state and public projections;
- one provider's wire-shape normalization;
- one durable migration/recovery boundary.

This lets a survivor agent read production once and build one stronger oracle.
It proposes dominated local checks; independent sensitivity/dominance
certification and exact subtraction happen in later phases.

## Generate and adjudicate a cluster dossier before editing

A coordinator or script should join existing evidence into one disposable
packet:

- owning production paths and public entry points;
- test node IDs importing/reaching them (testmon plus per-test coverage
  contexts, with the known collection-import caveat);
- current test/helper LOC and runtime;
- mock/source/AST coupling census signals;
- unique and overlapping executed arcs;
- recent bug-fix commits and dogfood witnesses touching the area;
- the primary and secondary regression classes those witnesses instantiate,
  using [`12-bead-regression-class-map.md`](12-bead-regression-class-map.md);
- the closest processed test law from
  [`13-bead-derived-test-laws.md`](13-bead-derived-test-laws.md), including any
  source-driven correction or justified split made during adjudication;
- focused mutation survivors;
- fixtures and caches used;
- rewrite/deletion boundary status.

Do this by extending or composing the existing `devtools lab test-economics`,
`devtools workspace failure-context`, verify-run artifacts, and mutmut receipts.
Do not create a second test-economics database or committed dashboard. The
scratch coupling/infra censuses supply additional candidate signals.

Current warning: the 2026-07-16 economics run has no coverage JSON, and testmon
hub imports make `storage`, `daemon`, `mcp`, and `schemas` each appear to touch
all 13,486 recorded tests. Generate fresh per-test coverage contexts before
using those package totals. See
[`census/test-economics-observation-2026-07-16.md`](census/test-economics-observation-2026-07-16.md).

The dossier is generated evidence, not a new source of truth. Save it under the
area packet for execution, then regenerate after changes. Do not ask a weak
agent to reconstruct this map one test at a time.

Run:

```bash
python .agent/scratch/testsuite_diet/census/dossier.py render --cluster <id>
```

A packet may be called **survivor-execution-grade** only when the dossier
resolves:

- exact authoritative production symbols;
- exact test/helper files plus explicit files to avoid;
- independent behavioral, recovery, security, compatibility, and diagnostic
  obligations as applicable;
- proposed survivor tests and a dominance argument for every deletion group;
- a realized historical reproduction where available, plus the exact
  production mutation and expected failure reason for later independent
  certification (an already-realized focused mutation receipt is stronger but
  not required before the survivor test exists);
- the class invariant, varied dimensions, generalized proof form, and exact
  historical seed retained for every bug witness used;
- only the focused commands a worker is permitted to run;
- upstream prerequisites and rewrite boundaries.

Unavailable coverage contexts or mutation receipts remain visible as evidence
gaps. They do not become a synthetic score. A survivor-ready packet authorizes
only survivor implementation; no deletion is authorized until the independent
receipt makes the cluster **deletion-certified**.

## Work protocol

### 1. Establish obligations

Read the public behavior and historical failures first. For every current test,
record only obligations not already represented:

- observable success/error behavior;
- durable state or recovery;
- compatibility/protocol shape;
- security/privacy boundary;
- architecture authority boundary;
- useful diagnostic witness;
- no independent obligation (implementation echo).

Names, helper calls, mock call ordering, file placement, old-spelling absence,
and private return shapes are not obligations unless a real external contract
or architecture constraint makes them so.

### 2. Build the strongest law first

Before deleting anything, write or identify the smallest real-route test that
can dominate the cluster. Prefer, in order:

1. independent model/fact-set equality;
2. metamorphic relation;
3. state-machine invariant;
4. public protocol contract at a genuine boundary;
5. narrow example for a unique error/diagnostic branch.

Use the smallest realized named workload canary by default. Add an upstream
scale/selectivity tier only when it activates the failure or establishes a
work bound. Do not define a Diet-specific semantic profile.

Do not stop at rephrasing a Bead's acceptance example. Convert the example
through the TDD-shaped rule in
[`12-bead-regression-class-map.md`](12-bead-regression-class-map.md): preserve
the exact pre-fix witness, state the implementation-independent invariant,
vary the causal dimensions, and select the corresponding state-machine,
metamorphic, differential, fault-injection, or bounded-work proof. When no
honest generalization exists—typically a unique provider wire shape—retain the
narrow regression and say why.

The initial conversion has already been performed for the known bug corpus in
[`13-bead-derived-test-laws.md`](13-bead-derived-test-laws.md). A worker does
not restart that analysis from a class label. Sol selects and source-validates
the relevant law, resolves its exact symbols/files, and narrows it into the
cluster dossier. If current source disproves a law's mechanism, update the law
and record the correction rather than forcing the stale plan.

### 3. Prove sensitivity

Freeze survivor editing. Sol or a different Terra/high worker uses a disposable
worktree at the frozen survivor revision to run the historical reproduction and
make the named temporary representative production mutation: drop one filter,
reverse a comparison, skip a write, remove a retry, or break a public
projection. The new test must fail for the right reason. Use focused mutmut
survivors to explore neighboring gaps. Restore the mutation, prove the
worktree is clean, and retain an attested certification receipt; the survivor
author may not certify its own deletion authority.

### 4. Subtract by dominance

Survivor implementation and subtraction are separate by default. Freeze
editing after the survivor wave; Sol or an independent Terra/high certifier
runs the historical witness and representative mutation and reviews every
unique obligation. Only then dispatch an exact-file deletion/consolidation
wave. Keep small tests that retain a distinct security, compatibility, failure,
or diagnostic obligation. Remove shared infrastructure that has no remaining
production-facing consumer instead of adding a ceremonial consumer.

### 5. Measure the result

Record:

- old test/helper nonblank LOC removed;
- replacement LOC added;
- net LOC;
- old/new cluster runtime and fixture build count;
- production functions/arcs reached;
- mutation/historical witness killed;
- residual obligations and rewrite exclusions.

Update the savings ledger only with adjudicated numbers. Census population is
never booked as savings.

## Packet template

Each executable area packet should be short enough to hand directly to an
implementation agent:

```markdown
# Responsibility
Externally visible law and production routes.

# Owned scope
Production files, tests, helpers. Explicit rewrite boundaries and files to avoid.

# Current weakness
Concrete escaped defect, shadow oracle, redundant overlap, or work-bound gap.

# Regression class
Primary/secondary R-class, implementation-independent invariant, varied
dimensions, and exact historical witness retained.

# Stronger proof
Fixture/profile, independent oracle, state/metamorphic law, public routes.

# Sensitivity witness
Historical repro or representative mutation expected to fail.

# Deletion candidates
Named tests/helpers, with unique obligations that must be retained.

# Verification
Focused command, broader affected command, artifact receipts.

# Economics
Gross removed, replacement added, net, runtime before/after.
```

No completion checkbox says “coverage domain declared.” Completion is the
green behavior, sensitivity witness, deletion diff, and measured result.

## Sol, Terra, and Luna routing

- Sol owns portfolio boundaries, dossier adjudication, independent-oracle
  design, architectural decisions, prompt preparation, cross-cluster
  integration, and difficult diagnosis.
- Terra at high reasoning is the default implementation and independent
  semantic-certification worker for behavioral
  laws, cross-module state, concurrency, durability, and deletion judgment.
- Luna at high reasoning is limited to certified exact-file deletion, already-specified
  mechanical consolidation, dossier/census tooling, localized fixtures, and
  fully executable acceptance criteria.

Do not pay the six-run calibration cost for one small deletion. Before Luna
writes code, first identify several certified bounded jobs that can amortize
calibration. Then run three identical read-only calibration packets
through Terra and Luna: cluster classification, focused-test design, and an
adversarial review of a proposed deletion. Sol admits Luna only when none of
its outputs needs substantive correction and median wall time or token use is
at least 20% lower. Luna's first five coding jobs are probationary: at least
four must be accepted without escalation and none may have an escaped
correctness issue. Luna escalates semantic ambiguity or failed acceptance to
Terra; Terra escalates architecture, durability, rewrite-boundary, or
  cross-cluster conflicts to Sol.

When the coordinator is operator-selected Sol/Ultra (xhigh planning mode where
available), Ultra supplies proactive delegation and
synthesis, not authority to improvise missing dossier decisions. Use one fanout
control plane per wave:

- native Ultra subagents for read-only adjudication or bounded generic-agent
  jobs;
- the attested manifest runner when exact Terra/Luna model and high-effort receipts
  are required.

Do not have Ultra spawn a second copy of jobs already launched by the runner.

## Shared-worktree worker contract

The coordinator renders prompts with one byte-stable prefix and targeted
context. Every implementation job must:

1. work directly in the named shared checkout;
2. read only named context and immediate dependencies required to understand
   it;
3. edit only exact assigned files and ignore concurrent changes elsewhere;
4. avoid git, Beads, broad formatters, generated-surface sweeps, and broad
   tests;
5. implement the behavioral result; delete tests/helpers only in a certified
   deletion job that names the independent certification receipt;
6. run only named `devtools test` selectors and cheap local checks, through the
   existing checkout lock;
7. stop with a structured blocker when a design decision is unresolved;
8. return structured changed-files, behavior, production dependencies, check
   outputs, actual/proposed deletions, sensitivity state, residual risks, and
   recommended coordinator checks.

Completion prose is advisory. The coordinator inspects actual assigned-file
deltas and production routes. A non-blocked implementation receipt with no
assigned-file change is rejected.

The runner also requires a clean execution checkout, rejects same-wave
read/write dependencies, checks the whole wave's changed-path union, protects
the ignored Testsuite Diet control tree, and stops later waves after blockers
or invalid work. Sol never appears as a worker model in a manifest.

Mutation certification is not a shared-runner job. Its required successful
outcome is a clean final worktree with evidence of a temporary mutation, which
conflicts with the shared runner's correct requirement that a non-blocked
implementation job leave an assigned-file delta. Launch it directly through
the attested runtime in a disposable worktree and store its receipt beside the
survivor-wave artifacts.

When all jobs finish, editing freezes. Sol reviews the combined diff, repairs
composition issues, then runs combined focused tests and
`devtools verify --quick`. Harness/dependency waves additionally run
`devtools verify --all`; later leaf-only waves run ordinary testmon-selected
`devtools verify`. Only the coordinator stages, commits, pushes, or updates
Beads. Use isolated worktrees when assignments overlap or require unresolved
architecture decisions. Prefer serialization when both jobs share one hotspot
and a second worktree would only create branch/merge overhead. Certification
worktrees never merge; architecture implementation worktrees commit verified
chunks before cleanup.

## Choosing the next cluster

Prioritize direct evidence, not a synthetic quality score:

1. known escaped composition defect with a reproducible structural witness;
2. repeated failure class represented by several bug Beads but no generalized
   real-route law;
3. large test overlap around a central public route;
4. slow/shared fixture whose weak oracle dominates runtime;
5. implementation-coupled cluster in a stable area;
6. dead helper/verifier island with confirmed zero consumers.

Defer MCP and the current web reader to their rewrites. Do not spend effort
making tests elegant for code scheduled to disappear. Mine their externally
meaningful obligations and attach those to rewrite packets.

## Concrete first batches

### Batch A: seeded artifact integrity

Owned: only residual seeded fixture/cache code and focused tests after the
realized-baseline reconciliation. Extend shared workload identity, fail on
ingest errors, publish atomically, validate receipt/archive, and clone
immutably. Report unused fixtures whose consumer search remains empty; delete
them only in the certified subtraction wave. Do not rebuild profile inference,
tiers, C-03, or receipt accounting.

### Batch B: query composition slice

Owned: current cross-surface query cases, independent facts attached to the
realized C-03 workload, and exact-selection routes. Extend C-03 with public
expressions, counts, partitions, pages, and preview/apply agreement. Reuse its
global-first mutation/work receipt. Propose dominated per-layer and
mock-forwarding tests for later certified subtraction, retaining parser
diagnostics/security cases.

### Batch C: scale/work consolidation

Owned: benchmark seeder, scale fixtures/tests, and work counters. Migrate them
onto realized distribution/tail tiers and shared receipts, add exact result and
VM-step laws, then propose the second hand-built seeder and generic unused
growth budget for certification. Delete them only in the later exact-file
wave. Profile design remains upstream.

### Batch D: lifecycle state model

Owned: established repository/write-path Hypothesis state machines. Add
restart/convergence/overlay transitions and independent logical facts. Delete
the unused parallel `RepositoryLifecycleHarness` only after independent
consumer/dominance certification; do not wire tests to it merely to save it.

### Batch E: verification shrink

Owned: each verifier function and its consumers, one function at a time but one
coherent PR cluster. Retain checks derived from live inventory/behavior;
propose self-authorized declaration loops for independent certification, then
delete their tests/docs together in the exact subtraction wave.

## Review questions

- Does the oracle know anything the production implementation did not tell it?
- Does the test cross the boundary where the historical defect occurred?
- Would increasing fixture volume activate a new relation/state, or merely run
  the same branch more times?
- Is a mock standing in for an external process/network/clock, or for our own
  implementation?
- Can one property/state law replace a case matrix without losing diagnostics?
- Is new shared test infrastructure serving at least two real clusters now?
- Did the agent delete dominated code, or only add another test layer?
- Are MCP/web obligations being designed for the rewrites rather than ported?
