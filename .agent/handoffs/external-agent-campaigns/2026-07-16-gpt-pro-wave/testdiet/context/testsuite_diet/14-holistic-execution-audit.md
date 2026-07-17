---
created: 2026-07-16
purpose: Adversarially reconcile the complete Testsuite Diet into one executable portfolio
status: controlling-execution-audit
project: polylogue
---

# Holistic execution and orchestration audit

## Verdict

The Diet has a strong test-design core: independent planted facts, real
production routes, historical witnesses, mutation sensitivity, workload
identity reuse, dominance-based deletion, and explicit rewrite boundaries.
The Bead-derived laws now cover the major escaped-failure mechanisms rather
than assuming the suite is merely example-heavy.

Its main weaknesses were execution ordering and authority management. The
first orchestration shape could delete tests in the same worker turn that
created their replacement, before an independent sensitivity/dominance check;
its supposedly blocked post-merge manifest actually validated on the active
unmerged branch; and the runner did not stop later waves or detect same-wave
read/write and out-of-assignment hazards. Those are controlling corrections,
not optional refinements.

## Survey

| Surface | Purpose | Concern | Finding |
| --- | --- | --- | --- |
| Composition synthesis, proof-form audit, capability map | Explain why and what to strengthen | medium | Rich evidence, but several generations of narrative can be mistaken for equal authority. |
| 33 Bead-derived laws | Convert incidents to generalized proofs | high | Strong portfolio input; exact current symbols and cluster economics are still required before dispatch. |
| Five generated dossiers | First representative implementation slices | high | Useful but incomplete evidence; JSON size is not readiness, and coverage/mutation gaps remain visible. |
| Area packets | Bound ownership and preservation | medium | Good subsystem routing; cross-cutting laws must be selected explicitly. |
| Workload/harness architecture | Shared semantic artifact and work receipts | high | High leverage and a hotspot dependency; it must not recreate upstream profiles. |
| Shared-worktree runner | Exact model/file/test dispatch | critical | Needed fail-stop, clean-checkout, read/write hazard, and assignment-union enforcement. |
| Model routing and calibration | Spend reasoning where it changes outcomes | high | Terra default is sound; immediate Luna calibration cost more than its one planned job and lacked telemetry. |
| Savings ledger | Prevent unsupported suite-size claims | medium | Honest today, but five heterogeneous clusters cannot support suite-wide extrapolation without strata. |
| Rewrite boundaries | Avoid polishing obsolete MCP/web code | low | Correct; urgent safety regressions remain exceptions and obligations need a rewrite handoff. |

## Authority ladder

When Diet artifacts disagree, use this order:

1. current production source, public behavior, and full Bead thread;
2. merged upstream workload/receipt/testmon outcomes and raw receipts;
3. this holistic audit and the realized-baseline gate;
4. Bead-derived test laws and capability map;
5. current area packet and freshly regenerated dossier;
6. rendered worker manifest and prompt;
7. worker receipt;
8. historical synthesis, census, forecasts, and generated dossier JSON.

Lower layers route work; they cannot overrule higher evidence. Dossiers and
manifests are disposable snapshots and must carry source/dossier hashes in the
pre-dispatch reconciliation receipt. This prevents the growing scratch tree
from becoming several competing plans.

## Revised cluster lifecycle

The portfolio now uses a four-stage proof boundary:

```text
survivor implementation
        │
        ▼
independent sensitivity + obligation/dominance review
        │
        ▼
exact deletion/consolidation wave
        │
        ▼
combined integration and publication
```

### 0. Reconcile the merged baseline

- Wait for the workload-profile and verification-proof program to merge.
- Use a coordinator-owned clean checkout; do not run shared editing while the
  schema branch or another operator change is dirty.
- Regenerate the five dossiers and resolve every planned/current symbol.
- Write `.local/testsuite-diet/reconciliation/realized-baseline.json` with the
  exact git head, upstream merge commits/receipts, workload/canary identities,
  refreshed dossier hashes, and clean-checkout evidence.
- The sample manifest names this receipt as a required read and therefore
  remains invalid until reconciliation is real.

The receipt is evidence routing, not a behavioral gate. Sol still reads the
source and receipts; file existence alone is insufficient.

### 1. Land the narrow foundation

One Terra/high job adapts the realized workload artifact into fail-closed
publication, validation, immutable base, and writable clone behavior. It does
not delete broad fixture/benchmark consumers in the same pass. Sol verifies
the cache mutation/interruption/corruption witnesses before downstream jobs
depend on the artifact.

### 2. Implement survivor laws in parallel

Dispatch only disjoint, decision-complete survivor work. The first useful
parallel group is query composition, convergence restart, and confirmed
devtools retirement. Storage equivalence waits for the foundation. Authority,
lineage, concurrency, security, and installed lifecycle require their new
law-derived dossiers.

Worker prompts may name likely dominated tests, but survivor jobs do not delete
them unless the historical sensitivity proof already exists independently.
They return an obligation matrix and proposed deletion set for certification.

### 3. Certify sensitivity and dominance

Editing freezes. Sol or an independent Terra/high certifier works from a
disposable worktree at the frozen survivor revision. It may make only the
packet's named temporary production mutation and must restore a clean tree
before returning. The certifier:

- runs the historical witness and one representative production mutation;
- checks the production route and independent oracle;
- reviews every proposed deletion for unique compatibility, security,
  recovery, diagnostic, and process obligations;
- records runtime, fixture builds, and failure diagnostics/shrinking quality;
- rejects a law that is flaky, opaque, or materially slower without justified
  coverage value.

Certification is batched across the survivor wave. This preserves verification
economy without letting self-authored green tests authorize their own cleanup.
The shared runner is not used for this lane: its implementation contract
correctly rejects a job that leaves no assigned-file delta, whereas a mutation
certifier must leave no delta. The direct attested job records the base commit,
mutation/revert hashes, model/effort, command/output, and final clean state.

### 4. Subtract and integrate

Exact-file deletion/consolidation jobs may run in parallel after certification.
This is the natural future Luna lane because semantics are already decided.
Sol then reviews the entire diff, runs combined focused checks, quick static/
generated verification, and the appropriate broad publish gate once.

Regenerate affected dossiers after each merged wave. A later manifest may not
use deletion candidates or economics from a pre-wave snapshot.

## Shared-worktree operating rules

Shared editing is an optimization for disjoint, already-decided work—not the
default for architecture. The runner now enforces:

- Terra/Luna only in worker manifests; Sol remains the coordinator;
- a clean git checkout at run start;
- no same-wave write/write or read/write collisions;
- exact assigned-file deltas per worker;
- no final wave changes outside the union of current assignments;
- no changes to the ignored `.agent/scratch/testsuite_diet` control tree;
- later waves are skipped after a blocked or invalid earlier wave;
- model and high effort are attested by the existing launcher.

Fail-stop is not rollback. If an invalid/blocked worker leaves edits, the
checkout is quarantined: do not resume a later wave and do not automatically
restore paths that may include concurrent work. Sol compares the wave baseline,
file deltas, receipts, logs, and source, then deliberately retains or reverts
only attributable worker changes.

Still use isolated worktrees for temporary production mutation, overlapping
files, authority/lineage/security architecture, production schema changes, or
a job needing git history. Prefer serialization when jobs share one hotspot
and isolation would only add a merge. Prompt boundaries and final-state checks
are not a filesystem sandbox.

Two hardenings are prerequisites for the first unattended real wave; the
remaining telemetry can follow when its first consumer exists:

1. validate the reconciliation receipt's git head, ancestor commits, and
   dossier hashes rather than checking existence only;
2. add a run-level job timeout that interrupts by attested job ID through
   `agent_job_control.sh`, records `timed-out`, and skips later waves—never use
   a raw PID kill that can orphan the scoped Codex process;
3. capture Codex JSONL events so session IDs, token use, and timing make
   calibration and resumption measurable;
4. validate and store `certification.schema.json` receipts beside worker
   receipts.

Do not turn these into a scheduler DSL. The first two close fail-stop and stale
input hazards; implement them before unattended execution. Implement the latter
two when their first real calibration/certification wave would otherwise need
manual, error-prone bookkeeping.

## Model and reasoning assignment

The current host is authoritative for private model names: Codex CLI 0.144.4
reports `multi_agent` stable; local configuration selects `gpt-5.6-sol`, high
reasoning, and xhigh for plan mode; the attested launcher can explicitly select
Terra/Luna and effort. Public OpenAI material supports parallel-agent and
worktree workflows, but does not document these private Sol/Terra/Luna names.
Do not infer their properties from older public model pages.

Official product evidence:

- [Codex is designed for multi-agent workflows](https://openai.com/codex/);
- [the Codex app uses separate threads and built-in worktrees for parallel agents](https://openai.com/index/introducing-the-codex-app/);
- [OpenAI recommends reviewing and validating agent work](https://openai.com/index/introducing-upgrades-to-codex/).

The practical routing is:

| Role/work | Model and effort | Why |
| --- | --- | --- |
| Portfolio coordinator, authority/rewrite boundaries, cross-cluster integration | operator-selected Sol in Ultra/xhigh planning mode where available | Needs global context, adjudication, and steering; it is not a manifest worker. |
| Stateful laws, query/storage/convergence, concurrency, security, cross-module implementation | Terra/high | Default implementation and independent semantic certification lane. |
| Exact deletion/consolidation after certification, localized fixtures, census tooling | Luna/high only after calibration and economic admission | Bounded semantics and exact files; no architectural inference. |
| Mechanical generated-surface repair after combined diff | coordinator or Luna/high with exact outputs | Avoid concurrent generator ownership. |
| High-risk independent review | separate Terra/high job, then Sol adjudication | Read-only for ordinary review; disposable mutation worktree for sensitivity certification. Independence matters more than a cheaper model. |

Selecting Ultra in the interactive Codex session is sufficient for Sol to act
as coordinator, but exact Terra/Luna selection is not guaranteed by generic
native subagent delegation. Use the attested runner (or its agent-control MCP
front end) when worker model identity matters. Native subagents remain useful
for bounded read-only analysis where exact model attestation is unnecessary.

### Luna admission is deferred, not assumed

The original plan spent six high-reasoning calibration calls to decide whether
Luna should perform one tiny deletion. That cannot save time. Keep Terra as the
only coding worker in the first wave. Calibrate Luna only after there is a
queue of several certified, disjoint deletion/tooling jobs large enough to
amortize calibration and probation.

Calibration needs a fixed rubric and actual telemetry:

- same three packets and exact source revision;
- correctness of obligation classification;
- no invented symbols/files or speculative deletion;
- sensitivity/dominance reasoning and blocker discipline;
- wall time and token use from captured runtime events;
- blinded Sol adjudication before seeing cost.

Admission still requires no substantive correction and at least 20% lower
median wall time or tokens. Four of the first five coding jobs must be accepted
without escalation and none may escape correctness.

## Concurrency profiles

`max_concurrency=4` is a ceiling, not a default promise:

| Wave shape | Suggested worker concurrency |
| --- | ---: |
| Read-only dossier/calibration packets | 4 |
| Disjoint test-law edits with focused tests serialized by checkout lock | 3–4 |
| Shared corpus/large SQLite or filesystem fault jobs | 1–2 |
| Generated surfaces or a shared hotspot | 1 |

Partition manifests when resource profiles differ. Do not add a resource DSL
to every job merely to avoid selecting an appropriate wave boundary.

## Work worth preparing now

1. **Execute the law-to-hotspot DAG:** use
   [`15-law-execution-dag-and-model-routing.md`](15-law-execution-dag-and-model-routing.md)
   to select the first decision-complete dossiers and invalidate routes when
   current source changes their prerequisites or write ownership.
2. **Certification recipe and receipt:** implement one disposable-worktree,
   direct-attested Terra/high certification job with final clean-state proof.
3. **Reusable primitives with two real consumers:** independent fact reader,
   deterministic transaction barriers, convergence event trace, SQLite work
   counter, temporal strategies, and evidence-lattice assertions. Do not build
   a generic framework before the second consumer is named.
4. **Certification packets:** preselect one historical seed and one mutation
   per first-wave law, including exact revert procedure and focused command.
5. **Baseline economics:** collect node duration, fixture builds, and per-test
   coverage contexts before edits, once per cluster.
6. **Failure promotion loop:** capture a failing generated sequence under
   `.local`, shrink it, privacy-review its structure, and promote the minimal
   case as an explicit seed linked to its Bead/class.
7. **Rewrite handoff ledger:** transfer MCP/web security, paging, cancellation,
   typed-error, accessibility, and recovery obligations without porting
   implementation-coupled tests.
8. **Freshness discipline:** regenerate a dossier immediately before its
   survivor wave and again before deletion.

## Savings and scheduling correction

The first five clusters are enough to begin learning, not enough to extrapolate
one replacement ratio across 275k test LOC. Projection requires examples in
distinct strata: parser/wire compatibility, pure property/algorithm, SQLite
state/recovery, cross-surface composition, process/integration, and
verification/tooling. Report low/base/high only within a stratum that has a
realized analog; leave other areas unforecast.

Schedule by risk reduction per coordinator bottleneck, not deletion LOC:

1. foundation integrity and exact query/convergence vertical slices;
2. authority/replay, lineage, deterministic concurrency, and security;
3. incremental/rebuild and freshness equivalence;
4. evidence/provenance and temporal laws;
5. provider-family normalization and installed lifecycle;
6. certified subtraction and rewrite-native migration.

## Residual limitations

- Real provider formats, browser policies, packaging, cross-device behavior,
  and host pressure remain partly open systems; deterministic laws complement
  rather than replace installed canaries and dogfood.
- Some production seams—failpoints, clocks, work counters, application
  factories—must be added only with real consumers.
- Current dossiers still lack fresh per-test coverage contexts and realized
  sensitivity receipts. They are prepared designs, not deletion authority.
- Scratch artifacts are ignored and tied to this long-lived planning worktree.
  Before changing or removing the worktree, archive or promote the controlling
  packets deliberately; otherwise the plan will not travel through git.
