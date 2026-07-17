---
created: 2026-07-16
purpose: Index the top-down Polylogue test-suite diet research and execution packets
status: active
project: polylogue
---

# Polylogue test-suite diet

This directory turns the suite audit into an execution map. It is intentionally
scratch research, not a new product verification framework. Generated candidate
inventories locate review work; they are never automatic deletion verdicts or
CI policy.

## Read order

1. [`test-suite-composition-and-scale-2026-07-16.md`](test-suite-composition-and-scale-2026-07-16.md)
   — why the current suite misses composed failures and the target test
   architecture.
2. [`11-test-proof-form-audit.md`](11-test-proof-form-audit.md) — controlling
   correction to the earlier `example-heavy` shorthand, with a whole-suite
   proof-form inventory and narration of every high-risk responsibility.
3. [`02-execution-map.md`](02-execution-map.md) — subsystem hierarchy,
   sequencing, and packet status.
4. [`03-savings-ledger.md`](03-savings-ledger.md) — current LOC baseline,
   mapped near-term forecasts, explicit end-state size scenarios, overlap
   rules, and how realized savings are recorded. The 8–13k band is not an
   estimate of the eventual redesigned suite.
5. [`04-harness-architecture.md`](04-harness-architecture.md) — concrete
   corpus, cache, runner, layout, and work-bound redesign.
6. [`05-systematic-coverage.md`](05-systematic-coverage.md) — generated
   execution/responsibility/sensitivity/semantic evidence without coverage
   catalogs.
7. [`06-tooling-evaluation.md`](06-tooling-evaluation.md) — which existing
   tools to extend and which bounded pilots are justified.
8. [`07-agent-execution-playbook.md`](07-agent-execution-playbook.md) — how to
   prepare behavior-cluster dossiers and execute strengthening plus deletion.
9. [`08-capability-map.md`](08-capability-map.md) — behavioral-responsibility
   map from authoritative production routes to existing proof and known
   escapes. It routes planning; it is not a coverage declaration.
10. [`09-proposal-and-bead-audit.md`](09-proposal-and-bead-audit.md) — audit of
   prior suite proposals and related Beads against the anti-vacuity rules.
11. [`10-realized-baseline.md`](10-realized-baseline.md) — controlling
    reconciliation against the completed workload-profile, receipt, testmon
    mutation, and xdist-witness program.
12. [`areas/`](areas/) — bounded packets for an implementation agent. Each
   packet says what to preserve, what to remove or replace, and what proof must
   exist before deletion.
13. [`12-bead-regression-class-map.md`](12-bead-regression-class-map.md) — maps
    every explicit bug Bead to one primary reusable failure class and converts
    concrete regressions into state-machine, metamorphic, fault, differential,
    or bounded-work obligations.
14. [`13-bead-derived-test-laws.md`](13-bead-derived-test-laws.md) — the actual
    middle-layer processing: 33 execution-oriented laws with incident evidence,
    invariants, varied dimensions, generalized proof, retained seeds,
    sensitivity mutations, and dossier placement.
15. [`14-holistic-execution-audit.md`](14-holistic-execution-audit.md) —
    controlling adversarial review of authority order,
    survivor/certification/subtraction phases, shared-checkout safety, model
    routing, concurrency, projection limits, and useful preparation.
16. [`15-law-execution-dag-and-model-routing.md`](15-law-execution-dag-and-model-routing.md)
    — dependency, hotspot, substrate, wave, model, and independent-certification
    routing for all 33 laws, including the cost-aware worktree boundary.
17. [`16-program-scale-and-readiness.md`](16-program-scale-and-readiness.md) —
    quantified workload horizons, agent responsibilities, opening-wave size,
    specification readiness, and remaining implementation decisions.
18. [`17-unresolved-architecture-scope.md`](17-unresolved-architecture-scope.md)
    — resolution status for the four core and five contingent branches, the
    remaining live-authority actions, and the boundary between decided product
    semantics and current-source packet preparation.
19. [`architecture/00-index.md`](architecture/00-index.md) — Sol-adjudicated
    product contracts for authority/identity, lineage, concurrency/publication,
    destructive/auth boundaries, freshness, query execution, evidence/public
    algebra, configuration, and installed runtime. Each document includes the
    recommended choice, competitive alternatives, migration seams, and proof.
20. [`census/README.md`](census/README.md) — reproducible broad candidate query,
   evidence-only cluster dossier generator, and the first five prepared
   dossiers.
21. [`orchestration/README.md`](orchestration/README.md) — thin shared-worktree
   wave runner over the existing attested Sinnix agent launcher.

## Evidence levels

- **Confirmed**: direct source/consumer trace establishes dead code, a vacuous
  oracle, or domination by a stronger neighboring behavior test.
- **Adjudicated candidate**: concrete files and failure mode are known, but the
  replacement/dominance proof has not yet been implemented.
- **Survey lead**: a broad signal or sampled cluster worth inspecting; never a
  deletion count.
- **Rewrite boundary**: preserve behavioral obligations, but do not improve or
  port tests against an implementation already scheduled for replacement.

## Execution contract

A weaker implementation agent should work one area packet, not pick arbitrary
tests from the census:

1. Read the production route and every test in the named cluster.
2. Classify unique obligations: public behavior, durable state, security,
   architecture, diagnostics, or no independent obligation.
3. For each historical bug witness, name its regression class and varying
   dimensions using
   [`12-bead-regression-class-map.md`](12-bead-regression-class-map.md).
4. Select and source-validate the processed law in
   [`13-bead-derived-test-laws.md`](13-bead-derived-test-laws.md); resolve its
   exact symbols, files, prerequisites, and commands into the dossier.
5. Implement the smallest stronger real-route survivor law first when
   replacement is needed; return proposed deletions without taking deletion
   authority.
6. Freeze editing and have Sol or a separate Terra/high certifier demonstrate
   anti-vacuity in an isolated disposable worktree with the named production
   mutation or historical defect.
7. In a later exact-file job, delete only tests certified as dominated by that
   law; retain unique branches and useful diagnostics.
8. Update the area packet and savings ledger with gross removed LOC,
   replacement LOC, net LOC, exact verification, and residual risk.

An area packet is not survivor-execution-grade merely because it has a
plausible rewrite. Its generated dossier must resolve exact production symbols,
exact owned and avoided files, independent obligations, proposed survivor
tests, historical evidence, an exact planned sensitivity mutation, focused
commands, and named deletion candidates. That permits survivor implementation
only. Actual independent sensitivity plus dominance review is required before
the packet is deletion-certified. Missing evidence produces a blocker, not
permission to improvise.

## Assumed realized harness baseline

Execution assumes the full outcome of `polylogue-1xc.14.1`, its parent
`polylogue-1xc.14` receipt contract, and `polylogue-b054.1.1.3` through `.5`
has landed. The current files below are read-only upstream inputs, and Sol must
replace this snapshot list with the merged paths before dispatch:

- `polylogue/schemas/generation/workload_profiles.py`;
- `polylogue/schemas/generation/archive_workload_profile.py`;
- `polylogue/schemas/field_stats/distributions.py`;
- `polylogue/schemas/synthetic/runtime.py`;
- `tests/unit/core/test_schema_workload_profiles.py`.

The Diet foundation owns only the missing cache/publication adapter, archive
validation, immutable cloning, independent canary facts, and proven fixture
subtraction. It reuses upstream workload/profile/build/archive identity and
resource/cleanup receipts. It does not reimplement profile inference,
correlated generation, scale tiers, C-03, receipt accounting, real-testmon
mutation proof, or repeated xdist witnesses.

See [`10-realized-baseline.md`](10-realized-baseline.md) for the authority and
pre-dispatch reconciliation rules.

Do not create another catalog declaring that the work is covered. The durable
result is the production behavior test, the deletion diff, and its verification
receipt.
