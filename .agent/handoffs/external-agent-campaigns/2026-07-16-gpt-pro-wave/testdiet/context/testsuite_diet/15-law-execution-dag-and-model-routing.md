---
created: 2026-07-16
purpose: Turn the 33 Bead-derived test laws into dependency-aware waves, hotspot ownership, and model routes
status: controlling-execution-routing
project: polylogue
---

# Law execution DAG and model routing

## Why this exists

[`13-bead-derived-test-laws.md`](13-bead-derived-test-laws.md) specifies what
must be proved. [`14-holistic-execution-audit.md`](14-holistic-execution-audit.md)
specifies the proof lifecycle. This document removes the remaining scheduling
ambiguity: prerequisites, shared hotspots, execution substrate, model, and
certification route for every law.

It is not a coverage catalog or a claim that a law has landed. A law becomes
dispatchable only when a fresh dossier resolves its exact current symbols,
owned files, historical seed, production mutation, focused commands, and
deletion candidates against the reconciled git head.

## Critical path

```text
G0 merged upstream workload/receipt/testmon outcomes
 │
 ▼
R0 realized-baseline reconciliation + fresh dossier hashes
 │
 ▼
F1 fail-closed shared corpus/cache publication and immutable clones
 │
 ├───────────────┬────────────────────┐
 ▼               ▼                    ▼
S2 query L14     S2 convergence L12   S2 verifier L27/L31-L33 survivor
 │               │                    │
 └───────────────┴────────────────────┘
                     │ editing freeze
                     ▼
C2 independent isolated mutation/dominance certification
                     │
                     ▼
D2 exact certified subtraction
                     │
                     ▼
I2 Sol integration + publish gate + realized economics
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
S3 storage L11/L13       later risk-led law packets
```

`G0` is not satisfied by the current uncommitted schema checkout. `R0` must
record the merged identities and refresh every downstream file assignment.
No worker may infer a replacement profile or canary while waiting for it.

## Shared primitive and hotspot ownership

Create a primitive only when the table names at least two real consumers. The
first cluster owns the narrow implementation; the second proves it is actually
shared before any generalization.

| Hotspot or primitive | First consumers | Ownership and serialization rule |
| --- | --- | --- |
| Realized workload artifact, planted semantic facts, cache receipt, immutable clone | F1, L14, L12, later L11/L16 | F1 owns publication and clone semantics. Downstream waves read it; they do not edit its identity/profile modules concurrently. |
| Independent fact reader | L14 query membership, L20 public projections, L11 rebuild equivalence | Add only after L14 and one second dossier agree on the fact schema. It reads planted input facts, never production output. |
| Query expression/lowering/execution path | L04, L14, L15, L16 | One write owner per wave. Parser diagnostics can be separate only when no lowering/executor file overlaps. |
| Frozen clock and convergence event trace | L12, L13, L17, L28 | L12 may add the smallest trace seam. Clock strategy remains in existing test infrastructure. |
| SQLite work counter/progress trace | L15, L16, L17, L31 | One instrumentation owner. Counters must observe production work, not duplicate the algorithm in tests. |
| Storage writers, index DDL, rebuild orchestration | L01-L03, L07-L13, L23 | Architecture hotspot: serialize or use an isolated implementation worktree. Never place two shared-checkout jobs here concurrently. |
| Deterministic transaction barriers/failpoints | L06-L10, L23 | Add at a real transaction boundary and reuse; do not make sleeps or probabilistic races the oracle. |
| Evidence lattice/provenance assertions | L18, L19, L20 | L18 owns the smallest monotonicity vocabulary; L19/L20 consume only after source models agree. |
| Provider fact blueprint and detector ambiguity corpus | L21, L22, L25 | One provider family first; keep explicit wire witnesses that express compatibility. |
| Temporal strategies | L13, L22, L28 | Extend existing Hypothesis/frozen-clock infrastructure, not a Diet-only time model. |
| Capture/runtime lifecycle harness | L24, L29, L30 | Installed-process work; isolate from shared SQLite waves and preserve host/private-data boundaries. |
| Generated surfaces/inventories | L20, L25, L27 | Coordinator owns combined rendering after editing freezes. Workers do not concurrently render global outputs. |
| Testmon/xdist proof artifacts and receipts | L31-L33 | Consume the merged `b054.1.1.3`-`.5` evidence. Do not recreate a second selection or hang framework. |

## Per-law routing

All implementation and certification workers use high reasoning. Sol remains
the coordinating interactive model. Terra is the default worker. Luna appears
only in certified bounded deletion/tooling work after economic calibration.
Architecture workers consume the adjudicated defaults under
[`architecture/`](architecture/00-index.md); they do not choose among the
recorded alternatives unless current source disproves a stated invariant.

| Law | Prerequisite and hotspot | Implementation route | Independent proof route |
| --- | --- | --- | --- |
| L01 raw-authority fixed point | R0; [raw reconciler decision](architecture/01-evidence-authority-and-identity.md); storage authority/replay | Terra in an isolated architecture worktree; Sol prepares exact reconciler packet | Separate Terra/high mutation worktree; live apply waits for plan-digest authorization |
| L02 evidence-history equivalence | L01 reconciler; authority chooser | Same architecture branch as L01, serialized | Separate Terra/high history-permutation certification |
| L03 deterministic acquisition identity | R0; identity/hash and durable source | Terra isolated if production semantics change; shared survivor job only for test-only additions with disjoint files | Terra/high isolated content/metadata mutation |
| L04 relational cardinality conservation | F1; query/action joins | Terra shared survivor wave after L14 dossier resolves write ownership | Terra/high isolated dropped/distinct/join predicate mutation |
| L05 lineage arrival-order composition | R0; [lineage decision](architecture/02-lineage-composition-and-snapshots.md); writer/reader hotspot | Terra isolated implementation from exact packet | Separate Terra/high permutation and parent-replace mutation |
| L06 one lineage snapshot | L05 semantics; transaction barriers | Terra isolated concurrency implementation | Terra/high isolated interleaving certification with deterministic barriers |
| L07 linearizable or explicit-conflict writes | [concurrency decision](architecture/03-concurrent-writes-publication-and-resume.md); durable writer hotspot | Terra isolated; use atomic SQL/CAS/immediate transaction by declared invariant | Terra/high isolated lost-update mutation/interleaving |
| L08 adversarial lease/checkpoint schedules | L07 barriers; ops/source leases | Terra isolated or single-worker storage wave | Terra/high isolated lease-expiry/owner-loss mutation |
| L09 old-or-new publication | F1 for fixture cache; storage/file publication otherwise | F1 covers fixture artifact; production publication uses Terra isolated | Terra/high isolated interruption/corruption certification |
| L10 exact resume | L08/L09 seams; ingest/convergence cursors | Terra isolated state-machine job | Terra/high isolated checkpoint-skip/duplicate-effect mutation |
| L11 incremental/rebuild equivalence | F1 and merged workload IDs; storage/rebuild hotspot | Terra single-worker storage wave | Terra/high isolated omitted-rebuild-step mutation |
| L12 convergence debt liveness | F1 active-growing variant; convergence files | Terra shared survivor lane when disjoint from query | Terra/high isolated retry/debt-deletion mutation |
| L13 monotonic derived freshness | L11/L12; [freshness decision](architecture/05-derived-freshness.md); storage/convergence hotspot | Terra isolated or serialized immediately after L11 | Terra/high isolated stale-recipe/content mutation |
| L14 query algebra membership | F1 C-03 facts; query hotspot | Terra shared survivor lane | Terra/high isolated filter/projection/pagination mutation |
| L15 bounded and interruptible query work | L14; [query lifecycle decision](architecture/06-query-cancellation-and-bounds.md); cancellation/work counter | Terra isolated production-seam slice, then serialized survivor wave | Terra/high isolated cancellation and depth-limit mutation |
| L16 selected-work scaling | F1 scale tiers and L15 counter | Terra at concurrency 1-2 because it is resource-sensitive | Terra/high isolated work-amplification mutation plus measured bound |
| L17 truthful progress and cleanup | L12/L15 trace seams | Terra serialized with the owning lifecycle cluster | Terra/high isolated missing-cleanup/double-count mutation |
| L18 evidence monotonicity | Source models settled; [EvidenceValue decision](architecture/07-evidence-provenance-and-public-algebra.md) | Terra single owner; shared checkout if files are disjoint | Separate Terra/high evidence-removal mutation |
| L19 total/provenance conservation | L18 vocabulary; cost/quantitative models | Terra after L18 | Terra/high isolated dropped-source/stronger-provenance mutation |
| L20 one public fact algebra | L14/L18/L19 and surface rewrite status | Terra for stable CLI/API/daemon slices; MCP/web obligations transfer to rewrites | Terra/high surface-disagreement mutation; rewrite-native certifier for deferred surfaces |
| L21 tightest valid detector | One provider family packet; dispatch hotspot | Terra shared lane if source files are disjoint from other jobs | Terra/high detector-order/ambiguous-shape mutation |
| L22 semantic normalization | L21 corpus and temporal strategy | Terra per provider family, one family at a time | Terra/high authoredness/outcome/time-loss mutation |
| L23 uniform destructive/excision contract | [operation-gateway decision](architecture/04-destructive-and-authentication-boundaries.md); all write routes | Terra isolated architecture job; Sol resolves exact handlers and preview migrations | Terra/high isolated bypass-route mutation plus security review |
| L24 fail-closed authentication lifecycle | Same security decision; installed transport harness | Terra isolated installed-runtime job; receiver identity change remains explicit re-pair | Separate Terra/high fail-open/reconnect mutation |
| L25 output/schema injection safety | Provider corpus; generated surfaces serialized | Terra focused stable-surface job; coordinator renders afterward | Terra/high malicious-token mutation and schema/output parse check |
| L26 configuration composition/path coherence | [resolved-config decision](architecture/08-configuration-and-path-coherence.md); existing five-layer precedence | Terra shared lane if exact files are disjoint | Terra/high precedence/path-split mutation |
| L27 executable inventory authority | Live consumer trace; devtools hotspot | Terra survivor review first; Luna deletion only after certification and admission | Terra/high direct consumer/history proof; behavior mutation where authority remains |
| L28 equivalent-instant temporal behavior | Existing frozen clock/strategies | Terra focused temporal lane | Terra/high timezone/DST/boundary mutation |
| L29 recoverable capture delivery | [runtime decision](architecture/09-capture-delivery-and-deployed-status.md); installed capture runtime | Terra isolated installed-process job | Terra/high crash/restart/backpressure certification |
| L30 status bound to running artifact | Same runtime decision; L29 deployment harness | Terra installed/deployment lane, serialized with L29; absent optional host evidence stays `unknown` | Terra/high stale-binary/config mutation against host evidence |
| L31 real-mutation affected selection | Merged `b054.1.1.4` receipt | Consume upstream proof; only residual Terra tooling work | Independent receipt/source audit; do not duplicate harness |
| L32 isolated attributable harness artifacts | Merged `b054.1.1.3` receipt | Consume upstream; Terra only for proven residual | Independent cross-run/worker attribution audit |
| L33 isolated/xdist hang witness | Merged `b054.1.1.5` receipt | Consume upstream; Terra only for proven residual | Independent repeated-witness audit |

## Wave portfolio

### Wave 0 — reconcile, do not edit

Sol checks the merged upstream work, regenerates dossiers, resolves hashes and
symbols, creates the reconciliation receipt, and freezes exact write sets.
This gate also removes any Diet task already supplied upstream.

### Wave 1 — one foundation owner

One Terra/high job lands only residual F1 cache/publication/clone behavior and
survivor tests. Run at concurrency one. It returns proposed deletions but does
not remove consumers. Sol reviews and verifies the artifact before continuing.

### Wave 2 — first disjoint survivor portfolio

After file-set validation, run at most three Terra/high lanes:

1. L14 query composition and exact work facts;
2. L12 debt -> restart -> retry -> quiescence;
3. L27/L31-L33 devtools survivor/retirement review, limited to confirmed
   residual work after merged upstream proof receipts.

These lanes share the corpus as a read-only prerequisite. A read/write
collision moves the affected lane to a later wave rather than relying on
workers to avoid each other informally.

### Certification C2 — isolated and non-merging

The shared `wave.py` runner is intentionally not used for temporary production
mutation certification: its final-delta contract correctly requires an
implementation job to leave an assigned change, while a certifier must restore
the production tree exactly.

For each survivor law, the coordinator creates a disposable worktree at the
frozen survivor revision and launches a separate Terra/high attested job. The
certifier may make only the named temporary production mutation, runs only the
named witness, captures the expected failure, restores the file, and proves a
clean final tree. The receipt records base commit, prompt/model attestation,
mutation and revert hashes, command/output, failure reason, runtime, fixture
builds, and dominance review. Sol may certify directly, but the worker that
wrote the survivor test may not certify its own deletion authority.

Certification worktrees create no permanent branch, commit, or merge. Remove
them after the clean-state and artifact checks. Reuse the merged testmon
mutation harness when it fits rather than building a Diet-specific mutator.

### Deletion D2 — exact certified files only

Generate a new manifest from C2 receipts. Each mission names its certification
artifact and exact deletion/consolidation files. Terra/high remains the
default. Admit Luna/high only if a queue of several bounded jobs justifies the
calibration cost. No job expands its deletion set from nearby census hits.

### Integration I2 — one coordinator gate

Sol reads source, actual diff, receipts, and retained obligations; repairs
composition; runs combined focused tests and `devtools verify --quick`; then
runs `devtools verify --all` for the harness/dependency wave. Only after this
does the coordinator record realized economics and publish.

### Later waves

- Wave 3: L11 then L13, serialized around storage/rebuild ownership.
- Wave 4: L01/L02, L05/L06, L07-L10, and L23 as architecture-sized isolated
  branches after Sol translates the adjudicated decisions into current-symbol
  dossiers, usually one hotspot branch at a time rather than many worktrees.
- Wave 5: disjoint source/config/evidence/temporal/security slices (L18-L22,
  L25-L28) only where current file ownership proves parallelism.
- Wave 6: installed-runtime/deployment laws L24 and L29-L30, with real process
  boundaries and explicit host/privacy controls.
- Rewrite waves: L20 obligations for MCP/web land in their replacement design,
  except urgent security regressions that cannot safely wait.

## Worktree cost rule

Worktrees are a risk-control tool, not the default fanout mechanism. Use one
only when at least one condition holds:

- a temporary production mutation must be isolated and fully reverted;
- exact implementation write sets overlap;
- the job changes authority, durability, schema, security, or architecture and
  needs independent branch history;
- the shared coordinator checkout cannot be clean and frozen.

Prefer a shared wave when work is decision-complete, exact-file disjoint, and
the checkout test lock already serializes focused tests. Prefer serialization
in the existing implementation branch when two jobs share one hotspot and a
second worktree would merely create merge work. Count setup, branch sync,
conflict resolution, post-merge verification, and cleanup in the routing
decision—not just worker wall time.

Architecture implementation worktrees must commit verified logical chunks so
their changes survive worker cleanup. Certification worktrees are the opposite:
they must finish clean and never contribute a commit.

## Model economics and stop rules

- Sol/Ultra owns portfolio and architectural decisions, exact prompt
  preparation, steering, receipt adjudication, cross-lane repair, and publish.
- Terra/high owns every first-wave implementation and every independent
  semantic certifier.
- Luna/high is a later optimization for certified exact-file deletion,
  mechanical consolidation, localized fixtures, or dossier tooling. Calibrate
  only when several jobs can amortize six comparison calls and probation.
- Run four workers only for read-only packets. Use three for ordinary disjoint
  survivor edits, one or two for SQLite/filesystem/scale jobs, and one for a
  shared hotspot or generated surfaces.
- Stop a wave on a blocker, unassigned dependency, control-tree modification,
  receipt/model mismatch, unexpected dirty path, or failed named check. Later
  waves remain skipped until Sol reconciles the state.
- Treat an invalid shared checkout as quarantined, not rolled back. Sol uses
  the recorded baseline/deltas/logs to retain or revert attributable edits;
  workers and later waves do not continue in that checkout meanwhile.
- Do not retry a semantically incomplete packet with a stronger or more
  expensive model. Repair the dossier first.

## Preparation that repays later work

Prepare these in order, and only to the first real consumer:

1. exercise `orchestration/certification.schema.json` with one direct attested
   worktree certification job and retain the proven invocation as the recipe;
2. per-cluster baseline script that records exact node runtime, fixture builds,
   and source revision without creating another database;
3. reusable independent facts for C-03 plus one second consumer;
4. deterministic transaction barriers at the real boundaries specified by the
   concurrency decision, shared by the first two concurrency laws;
5. the query execution context plus SQLite work/progress counters specified by
   the query decision, shared by L15-L17;
6. a failure-promotion recipe: capture, shrink, privacy-review, retain the
   minimal structural seed, and link it to the Bead/law;
7. a rewrite handoff ledger for MCP/web security, paging, cancellation,
   typed-error, accessibility, and recovery obligations;
8. a post-wave dossier refresh that invalidates old deletion candidates and
   prevents stale economics from entering a later manifest.
9. before unattended fanout, wire reconciliation-content validation and an
   attested-job-ID timeout/interrupt path into the runner; a raw subprocess
   timeout is insufficient because it can leave the worker's systemd scope
   alive.

The acceleration criterion is concrete: prepare an artifact only when it
eliminates repeated source archaeology, repeated fixture construction, or
manual receipt reconciliation for at least two named laws.
