---
created: 2026-07-13
purpose: Standalone corrective proposal for restructuring the current Polylogue Beads design after adversarial review
status: proposal-only; historical reconciliation is recorded, corrective mutations are not yet applied
project: polylogue
baseline: origin/master cf30eea72 plus the pending session-reconciliation Beads snapshot (773 issues)
source_sessions:
  - claude-code-session:22155309-ec55-422c-84a6-2a81ed9e7aad
  - codex-session:019f59ac-a602-7ca0-872f-db5ba6a93070
---

# Corrective Beads proposal: a smaller kernel without flattening semantics

## Purpose and scope

This document proposes how to change Polylogue's current Beads design after
the 2026-07-13 design program, the independent Codex review, its adversarial
self-review, and the Claude reply assessing that review. It is deliberately
standalone: a reader should not need either chat transcript.

It does **not** apply the changes. The Beads snapshot used here already
contains the non-controversial historical reconciliation: missing lane notes,
execution-grade fields for the goal graph/worktree hygiene/D1-D3-D9 records,
the external-adoption activation note, and a durable verification-debt lane.
This document is the subsequent corrective pass.

The goal is not to minimize the count of Python types or Beads. The goal is to
minimize independently evolving semantics, durable stores, registries, daemon
loops, and UI concepts while preserving the distinctions on which Polylogue's
honesty depends.

## Executive conclusion

The strongest synthesis is:

> Aggressively unify identity machinery, provenance vocabulary, result
> contracts, authoring ergonomics, and surface plumbing. Conservatively retain
> typed domain definitions, durability placement, operational state, and
> authority boundaries.

The governing test is the review's four-axis rule:

1. **Identity:** what makes two instances the same?
2. **Lifecycle:** how are they created, changed, superseded, expired, and
   deleted?
3. **Authority:** who may assert, adopt, execute, or revoke them?
4. **Access shape:** which invariants and queries must their consumers rely on?

Polylogue needs one explicit veto in addition: **durability placement**. It is
technically part of lifecycle, but the split database architecture makes it too
load-bearing to leave implicit. Two concepts that belong in different tiers
must not be collapsed into one generic table merely because their JSON looks
similar.

If all four axes match and durability is compatible, unify the object. If only
serialization, refs, or receipts match, share a protocol. If only the UI flow
matches, share interaction vocabulary. This is the main correction to both the
original design program and the first independent review.

## Assessment of the Claude reply

### What I agree with completely

1. **Frame-exact replaces population-exact.** A query can enumerate every
   matching stored row while capture coverage and semantic measurement remain
   uncertain. Exactness must be exact over a named archive frame and named
   definitions, not an assertion that the archive contains the real-world
   population.

2. **A content hash is not a reproducibility receipt.** Query identity must
   bind a definition-protocol version before runtime persistence spreads.
   Executions must bind resolved time bounds, archive generations, runtime
   build, and model/classifier/ranker definitions. The present narrow use of
   `put_query()` makes this unusually cheap now.

3. **Provenance is sensitive data.** Query literals, result membership,
   annotation inputs, context deliveries, and execution configurations can
   repeat or amplify transcript secrets. Durable-by-default rigor would become
   durable-by-default disclosure. Persistence must be promotion-driven and
   excision-aware.

4. **Affinity, confidence, and membership are distinct.** A single scalar tag
   ladder cannot honestly represent all three. Algebra must fail closed across
   incomparable axes.

5. **Event ordering needs a declared contract.** Observed, checkpointed, and
   replay-verified order are different evidence grades. Typed unit kind,
   partition policy, lineage policy, horizon, and as-of receipt belong in every
   absence/sequence claim.

6. **`unresolved_inactive(H)` is better than abandonment.** It states the
   observation and horizon rather than pretending to know intent or finality.

7. **Context trust labels are insufficient without an execution-authority
   firewall.** Evidence may enter context as quoted evidence without becoming
   an instruction. Only an explicitly authorized policy assertion may cross
   that boundary.

8. **Judgment needs ties, incomparability, abstention, partial orders, and
   exploration quotas.** Forced total ordering manufactures information.

9. **The four-axis unification test is the best reusable artifact.** It should
   become doctrine in `polylogue-cpf` and a design check in
   `polylogue-o21`-style scaffolds.

10. **The first audit's diagnosis of `hg8n` was overconfident.** The durable
    observation was correct—no note existed in the current exported Bead—but
    the causal diagnosis was not. Evidence now shows the successful write was
    later eaten by the Beads auto-import race. This is exactly the difference
    between an observed state and an inferred cause that the result-contract
    work is meant to preserve.

11. **My first D8 control suggestion was confounded.** A causal evaluation
    needs matched-task, different-prompt pairs, not deliberately divergent
    baselines.

12. **Cap execution lanes, not design capture.** The large Beads graph is an
    encyclopedia/tech tree; the harmful WIP is simultaneous claims, branches,
    migrations, heavy verification, and merge pressure. The remedy is a narrow
    frontier view and bounded active lanes.

### Where I agree with qualifications

#### Generated curriculum

The reply is right that “defer until a smaller curriculum wins an A/B” is
circular if no curriculum arm exists. The corrected sequence is:

1. build a cheap, candidate-only v1 renderer from existing telemetry;
2. do not inject it automatically;
3. compare it against matched controls through the experiment substrate and
   context scheduler;
4. only then invest in adaptive generation, scheduling, or self-optimization.

The render may be cheap. Selecting safe teaching content, preventing leakage,
assigning arms, and measuring downstream effect are not cheap. Therefore the
right correction is “build the smallest measurable arm,” not “the curriculum
program is cheap.”

#### The two product wedges

Audit (“what supports this claim?”) and continuity (“have I resolved this
before?”) are excellent external-adoption wedges and useful gates for **new
platform bets**. They must not retroactively erase value that is already proven
for the operator: orchestration, cost honesty, forensics, capture repair, and
archive operations demonstrably supported the 2026-07-13 fleet.

A new investment should therefore name either:

- an external audit/continuity consumer; or
- an observed, recurring operator workflow with a receipt.

It need not pretend that all legitimate value is externally market-shaped.

#### Registry and object count

The reply is right that my proposed remedies also proliferated names. But
“number of types” is the wrong metric. A protocol type can reduce complexity;
a generic durable table can increase it. The count to minimize is independent
identity/lifecycle/authority implementations plus durable schemas, not type
declarations.

The loop registry should remain, because loops have operational state that a
recipe alone does not own. It should initially be a declare-once registry of
typed `ImprovementLoopSpec` records whose evidence lives in existing query,
metric, candidate, judgment, and artifact systems—not a new universal table,
daemon, and inbox for every loop.

### What I do not adopt

1. I do not treat “generation is a render” as evidence that generated teaching
   is operationally cheap. Only the candidate rendering step is cheap.

2. I do not use the two wedges as a universal deletion test. They prioritize
   external product proof and new platform work, not all operator-facing value.

3. I do not abolish typed domain definitions in pursuit of a lower object
   count. `ExperimentDefinition`, `PatternDefinition`, `Goal`, and
   `ImprovementLoopSpec` have different lifecycle and authority semantics even
   when they share definition hashes and receipts.

4. I do not create one generic receipt table, operation executor, relation
   object, or bundle compiler. Each would cross durability or authority
   boundaries that are already load-bearing in Polylogue.

## Target conceptual architecture

### Layer A: shared infrastructure

These mechanisms should be truly common:

- `ObjectRef` parsing and resolution;
- canonicalization and definition-protocol versioning;
- provenance/evidence edges;
- actor refs and execution-context refs;
- archive/evaluation-world references;
- enumeration/frame/measurement-authority vocabulary;
- common privacy and retention classifications;
- assertion/judgment authority;
- Selection × Projection × Render surface plumbing;
- plan/authorize/apply/receipt/reconcile envelope vocabulary.

Shared infrastructure does not imply a shared durable table.

### Layer B: typed definitions

Keep these as distinct semantic definitions, implemented over the common
definition protocol where useful:

- `QueryDefinition`
- `MetricDefinition`
- `ClassifierDefinition`
- `PatternDefinition`
- `CohortDefinition`
- `RankerDefinition`
- `AnnotationSchema`
- `ExperimentDefinition`
- `ImprovementLoopSpec`
- `Goal` / `Question`
- `ContextPolicy`

### Layer C: typed materializations and artifacts

Keep these distinct because their lifecycle/access semantics differ:

- `QueryResult` / `ResultSetSnapshot`
- `MatchSet`
- `RankingResult`
- `AnnotationBatch`
- `Finding`
- `ContextArtifact`
- `EvidenceReport`
- domain-specific execution/mutation/delivery receipts

### Concepts that should be protocols or fields, not standalone durable objects

| Proposed concept | Correct treatment |
|---|---|
| `EventOrderSpec` | Embedded protocol on pattern/event analyses; separately addressable only if multiple definitions actually reuse one instance |
| `ActorRef` | Stable ref vocabulary, not a registry containing every execution |
| `ExecutionContextRef` | Separate ref to prompt/tools/runtime/config; never folded into actor identity |
| `JudgeSpec` | Composition of actor ref, execution-context ref, role, rubric, and policy; no independent table until reuse/lifecycle proves it |
| `RelationManifest` | Protocol implemented by result sets, cohorts, match sets, and rankings; not a universal relation row |
| `EvaluationReceipt` | Common envelope plus typed domain receipts; no generic receipt table |
| Portable bundle | Artifact/export profile with a typed manifest; not an always-live product object or universal compiler |

## Exact corrective changes to current Beads

### 1. Make frame-exactness a common result contract

#### Amend `polylogue-rxdo.3`

Replace the single `exactness` field as the whole epistemic story with:

```text
enumeration: exact | capped | sampled | estimated
frame_ref:
  origins
  resolved interval
  archive/source generations
  capture coverage refs and degraded state
measurement_authority:
  structural | provider-reported | rule-derived | model-derived | judged
definition_refs:
  parser/classifier/metric/ranker versions
```

Keep compatibility projection to the current `exactness` field until all
surfaces move, but make the richer contract authoritative. Add AC proving an
enumeration-exact result can still render frame-incomplete and model-derived.

Add relationships from `rxdo.3` to:

- `polylogue-3uw` for frame/capture coverage;
- `polylogue-rxdo.9.8` for uncertainty rendering;
- `polylogue-bkzv` for common provenance glyphs.

#### Amend `polylogue-rxdo.9.8`

State explicitly:

- no sampling CI for an enumeration-exact census;
- frame coverage and classifier/judgment uncertainty remain reportable;
- bootstrap does not repair parser bias or missing capture;
- every interval names the uncertainty source it describes.

#### Amend and raise `polylogue-3uw` from P4 to P2

It is no longer a distant observability enhancement. It is a prerequisite for
honest frame declarations in every public analytic result. Its output should
be a coverage object/ref consumed by `rxdo.3` and measure rendering, not a new
analytics UI.

#### Amend `polylogue-9l5.7`

Replace “census” as sufficient validity metadata with the three-part result
contract. The measure registry must declare required enumeration, frame, and
measurement-authority conditions separately.

### 2. Fix definition identity before runtime wiring

#### Amend `polylogue-rxdo.2`

The durable note already records the decision. Move it into design and AC:

- canonical identity includes `definition_protocol_version`;
- that version covers language semantics, planner contract, field/operator
  definitions, canonicalization, tokenizer/collation, and relevant policy
  names;
- no reverse compilation from identity JSON;
- the planner-owned evaluator accepts the canonical executable form;
- macro expansion and relative time retain the current dynamic-definition /
  resolved-run split.

Raise `rxdo.2` from P2 to P1 until this correction lands. This is sequencing
urgency, not product priority: it becomes harder with every caller.

#### Amend `polylogue-rxdo.3`

Define the common evaluation-world envelope:

```text
definition refs
source/user/index/embeddings generations
runtime build
resolved temporal bounds
model/classifier/ranker refs
frame/degraded state
actor ref
execution-context ref
```

The envelope is common; query runs remain disposable ops-tier records, context
deliveries remain durable accountability records, experiments retain
assignment/exposure semantics, and remote mutations retain reconciliation
semantics.

### 3. Put privacy and retention into analysis provenance

#### Amend `polylogue-rxdo.2`

Change durable persistence policy to **persist on promotion**:

- saved, watched, cited, finding-dependent, experiment-dependent, and pinned
  definitions may be durable;
- ad-hoc definitions are not written to `user.db` merely because they ran;
- durable plans carry privacy class, retention policy, and excision linkage;
- result membership is durable only when its consumer requires snapshot
  identity.

#### Amend `polylogue-rxdo.3`

Committed ad-hoc runs should store privacy-safe structural telemetry by
default. A short-lived ops-tier payload may support `@last`, but must have an
explicit TTL/privacy class and be independently excisable. Preview/keystroke
queries remain non-persistent. Do not retain raw search literals in high-volume
telemetry by default.

#### Extend existing privacy owners rather than create another epic

- `polylogue-kwsb`: analysis provenance is part of the security/privacy
  coverage covenant.
- `polylogue-27m`: query definitions, ops query payloads, promoted result
  membership, findings, reports, and vectors are excision surfaces.
- `polylogue-303r.6`: backed-mode lifecycle covers those same artifacts and
  replicas.

Add blocking/validation edges from `rxdo.2`/`rxdo.3` to the relevant privacy
contract before broad runtime persistence. Do **not** create a second purge
vocabulary.

### 4. Replace the scalar tag ladder with three axes

#### Rewrite `polylogue-uh6c`

Retain namespaces and the rejection of a single-parent tree. Replace the
three-tier scalar model with:

```text
tagged(item, tag)            -> asserted membership, boolean/qualified
tag_affinity(item, prototype)-> embedding/model-derived similarity
tag_confidence(assertion)    -> uncertainty/calibration of a classification or judgment
```

Rules:

- affinity does not imply membership;
- confidence is not affinity;
- membership can remain informal forever;
- comparisons across axes fail closed unless an operation explicitly defines
  the conversion;
- tag prototypes are embedding definitions/resources, not the identity of the
  informal tag itself;
- the DSL exposes axis-specific predicates rather than `tag:x>0.7` with
  ambiguous semantics.

Add real AC and keep the bead P2. A seeded case must show high affinity without
membership, asserted membership with unknown affinity, and low-confidence
classification without changing either.

#### Amend `polylogue-dve1`

Formal ontology labels remain schema/version/batch-governed. Informal tags may
seed ontology proposals but are not silently migrated into ontology facts.

### 5. Make pattern/absence claims order-explicit

#### Amend `polylogue-avna`

Embed an `EventOrderSpec` protocol in `PatternDefinition` containing:

- partition and lineage composition policy;
- typed unit kinds;
- ordering source and tie policy;
- evidence grade: observed, checkpointed, replay-verified;
- horizon/as-of evaluation receipt;
- overlap and match policy.

Keep actions-only v1, land PACK-A/B, captures/measures, and SQL-vs-Python
metamorphic parity before mixed streams. Do not create a separate durable
EventOrder registry yet.

#### Amend `polylogue-7yk5`

Use `unresolved_inactive(H)` as the derived historical state. A goal may be
open, explicitly closed, explicitly blocked, or unresolved/inactive at a named
horizon. “Abandoned” remains a user-facing interpretation only when a measure
definition states the proxy and censoring policy.

#### Amend `polylogue-9l5.9`

Survival/abandonment measures depend on the goal graph's horizon-aware state
and must report censoring, frame, and closure authority. They may not treat the
last stored event as metaphysical finality.

### 6. Add the context execution-authority firewall

#### Amend `polylogue-37t.11`

The context scheduler must decide two independent questions:

1. may this content be disclosed to the recipient?
2. may this content act as an instruction?

Ordinary findings, memories, reports, recalled transcript, and generated
curriculum enter as quoted evidence. Only a distinct operator-authorized
policy assertion class may enter the executable instruction channel. Trust
labels and `inject:true` alone are insufficient.

#### Amend `polylogue-37t.15` and `polylogue-cpf.3`

The candidate chokepoint remains necessary but is not the firewall. Add a
fixture proving that even an adopted ordinary knowledge assertion cannot
silently become an instruction, while an authorized policy assertion can be
rendered into the instruction partition with an explicit receipt.

#### Amend `polylogue-xv1u`

Generated curricula emit candidate teaching artifacts only. They cannot set
their own execution authority or bypass the scheduler/judgment path.

### 7. Complete the judgment model without inventing `JudgeSpec`

#### Amend `polylogue-rxdo.9.11` through `.15`

Add:

- tie, incomparable, abstain, and insufficient-evidence verdicts;
- partial-order aggregation and visibility of disconnected components;
- exploration quotas so the active learner does not starve uncertain or
  minority regions;
- calibration by stable actor/model family and by exact execution context;
- receipts binding rubric, item ordering/blinding, execution context, and
  definition versions.

Represent judge identity using separate `ActorRef` and
`ExecutionContextRef`. Do not create a universal `JudgeSpec` table. A reusable
judge-routing policy may later become a definition if it gains independent
identity and lifecycle.

#### Keep `polylogue-stc` as a typed experiment definition

An experiment has preregistration, assignment, exposure, stopping, exclusion,
and outcome semantics that query recipes do not. It should implement the
common definition/receipt protocols but remain a distinct domain object.

Raise `stc` from P4 to P2 only after the frame/evaluation contracts land; it
is the mechanism needed to distinguish observational improvement claims from
causal ones.

### 8. Keep improvement loops, narrow their activation

#### Amend `polylogue-rxdo.11`

Retain `ImprovementLoopSpec` and a declare-once registry. Add the operational
state the first reductive review omitted:

- owner and authority;
- schedule/backoff and budget;
- last observation/proposal/judgment/artifact bump;
- starvation/failure/paused state;
- candidate queue and judge policy refs;
- metric and artifact version refs.

Do not activate thirteen bespoke daemon loops. Prove the abstraction with two
different pilots:

1. L1 recall relevance, sharing the implementation/evidence stream with
   `polylogue-37t.17`;
2. L2 classifier residue, exercising a different proposer/artifact type.

Only after both use the same operational contract without special casing
should more loops become active. The remaining loop designs stay durable as
horizon instances.

#### Raise `polylogue-o21` from P4 to P2

The registry count already grew. Declare-once derivation, scaffolding, and
actionable completeness errors should precede additional registry families.
Pilot on MCP tools as designed, then make classifier, marker, loop, and ranker
declarations consumers of the proven protocol rather than four new bespoke
registration systems.

### 9. Build the smallest measurable curriculum now

#### Narrow `polylogue-xv1u`, do not defer it wholesale

V1:

- reads existing usage/context receipts;
- renders a small candidate curriculum artifact;
- records source refs and exclusions;
- cannot inject itself;
- is used as one arm of a matched experiment;
- measures task outcomes, context use, correction recurrence, and operator
  judgment.

Defer adaptive topic selection, autonomous skill editing, continual
optimization, and automatic scheduling until v1 has a credible effect
receipt.

Use `polylogue-stc` for the experiment and `polylogue-37t.11` for delivery.
This resolves the circularity without granting the renderer self-authority.

### 10. Apply the two wedges only where they belong

#### Keep `polylogue-hg8n` P1

Its current repaired design is right: installation is substantially proven;
activation uses audit and continuity artifacts; one external person must run
the flow on their own archive.

#### Keep `polylogue-3tl.16` as a view

The claims ledger is a rendering over FINDING assertions, evidence ancestry,
judgments, and support state. No new claims table.

#### Keep internal operator value explicit

Add a short consumer classification to new platform Beads:

```text
consumer_proof: external-audit | external-continuity | observed-operator-flow
```

This is a design/readiness field or lint convention, not a new domain object.

### 11. Correct the flagship demo claims

#### Amend `polylogue-rxdo.10.2` (D3)

Rename the output from “the fix” to **prior observed recovery candidate**
unless evidence links the same target/state transition to the successful
change. Preserve precision@k, but label adjacency-only and captured-span modes.
The stronger “fix” claim requires target/state linkage or a judged receipt.

#### Amend `polylogue-rxdo.10.3` (D9)

Rename the primary measure to **repeated-context mass/cost**. It is observed
repetition, not automatically avoidable loss. A “context-loss tax” or savings
claim requires a counterfactual experiment comparing matched tasks/context
policies.

#### Amend `polylogue-212.6` (D8)

Keep the product demo—resume real abandoned work—as a descriptive proof. Add a
separate causal evaluation using matched task instances with different resume
brief/prompt treatments through `stc`. Do not use deliberately divergent
baselines as the control.

### 12. Consolidate the concepts that genuinely duplicate

#### MetricDefinition and MeasureSpec

- `polylogue-rxdo.9.1` owns canonical `MetricDefinition` identity/schema.
- `polylogue-9l5.7` owns statistical primitives, registry implementation,
  construct-validity enforcement, and registered measures.
- Remove `MeasureSpec` as a competing identity vocabulary; make 9l5.7 consume
  rxdo.9.1.

#### Recall relevance

`rxdo.11` L1 and `polylogue-37t.17` are one implementation and evidence
stream. The loop registry points to 37t.17; it does not materialize a parallel
read-access system.

#### Experiments

Context PROMPT_EVAL, curriculum A/B, routing experiments, harness comparisons,
and rigor mechanism J all use `polylogue-stc`, with treatment-specific
payloads.

#### Findings and claims

`polylogue-3tl.16` is a view over `polylogue-rxdo.4`; it does not own storage.

#### Judgment queue

`polylogue-37t.12` is the operator workflow over the comparative/calibrated
judgment lifecycle. No second judgment queue for K-O machinery.

#### Evidence basket

`polylogue-bby.15` remains a named mutable workspace pointer to versioned
selection/result snapshots plus annotations. It owns interaction and rendering,
not a parallel evidence collection model.

#### Inline markers

`polylogue-37t.2` owns authoring syntax. Marker kinds lower into goal,
assertion, event, and finding services; each marker is not its own domain
object.

#### Surface reads

Per-insight MCP tools that are named queries/projections should converge on
Selection × Projection × Render. Typed materialized insights remain when they
need incremental convergence, specialized indexes, or domain contracts.

#### Operations

Excision, provider mutation, maintenance, reset, and bulk judgment share
plan/authorize/apply/receipt/reconcile language and envelope components. They
retain separate actuators, capability gates, durability, and recovery rules.

### 13. Bound execution WIP without suppressing the tech tree

#### Amend `polylogue-ei94`

Add an active-lane budget to the conductor protocol:

- default maximum four code-writing/heavy-verification lanes on this host;
- read-only/lightweight analysis may exceed that only when it does not contend
  on the archive, database, migration slots, or generated surfaces;
- one durable migration per tier/window;
- no duplicate worker against the same branch/resource;
- a DONE process is not a completed AC;
- completion requires PR/receipt/Bead reconciliation.

The exact limit is configurable operational policy, not permanent product
semantics.

#### Amend `polylogue-2yax`

The frontier view should rank execution-ready clusters, expose file/migration/
resource collisions, and distinguish design-horizon Beads from claimable work.
It should predict the migration-slot collision the hand-built roster missed.

#### Keep design capture uncapped

Vision and mid-horizon Beads remain the durable encyclopedia. Enforce:

- only frontier Beads need complete implementation AC;
- only claimed/in-progress Beads consume WIP;
- active branches/PRs/agents are capped;
- priorities and delivery gates define the narrow executable view.

### 14. Persist the unification rule itself

#### Amend `polylogue-cpf`

Add the four-axis test plus durability veto to doctrine. Require design records
claiming “unified,” “generic,” or “first-class” to state:

- identity match;
- lifecycle match;
- authority match;
- access-shape match;
- durability compatibility;
- what remains typed/domain-specific.

#### Amend `polylogue-o21`

Generated scaffolds for new registries/definitions should ask these questions
and default to a shared protocol when the answers do not justify a new object.

## Cut / defer / combine / accelerate

### Cut

- “population-exact” language without a frame contract;
- one scalar tag value for membership, affinity, and confidence;
- a second durable claims ledger;
- a separate evidence-basket storage model;
- generic receipt table, generic operation executor, universal relation object,
  or universal bundle compiler;
- automatic curriculum injection;
- “fix” and “context-loss cost” claims unsupported by causal/state-linkage
  evidence;
- internal provider/origin compatibility aliases already rejected by the
  provider-origin decision.

### Defer

- activation of all thirteen improvement loops;
- mixed-stream pattern language before actions-only order/parity proof;
- autonomous ontology evolution beyond candidate generation;
- adaptive/self-optimizing curriculum after the minimal experiment arm;
- generalized portable evidence federation until one export/import profile
  proves closure, redaction, and excision;
- broad predictive/causal analytics tower until frame/evaluation/experiment
  contracts exist;
- reverse provider-web mutation beyond judged plan/apply receipts.

### Combine

- `MetricDefinition` and `MeasureSpec` identity;
- recall relevance L1 and 37t.17;
- experiment mechanism J, PROMPT_EVAL, curriculum A/B, and routing experiments;
- public claims ledger and FINDING/evidence views;
- judgment inbox and comparative/calibrated judgment lifecycle;
- web basket and promoted selection/result snapshots;
- marker registry and authoring syntax lowering;
- surface-specific ordinary reads into Selection × Projection × Render;
- plan/apply/receipt interaction vocabulary without combining actuators.

### Accelerate

1. query definition protocol version and planner evaluator;
2. frame-exact result contract plus capture coverage;
3. provenance privacy/persist-on-promotion/excision;
4. context execution-authority firewall;
5. PACK-A/B and order-explicit actions-only patterns;
6. audit flow: finding → evidence ancestry → basket/report → claims view;
7. continuity flow: D3 recovery candidates and D8 resume;
8. first external user;
9. declare-once ergonomics before more registries;
10. execution-lane cap and collision-aware frontier view.

## Proposed priority and dependency changes

| Bead | Current intent | Proposed change |
|---|---|---|
| `rxdo.2` | P2 query identity | Raise to P1 until protocol-version/evaluator/privacy contract lands |
| `rxdo.3` | P2 query runs/results | Keep P2; make frame/evaluation/privacy envelope authoritative |
| `3uw` | P4 capture completeness | Raise to P2; required by honest result frames |
| `27m` | P2 excision | Raise to P1 before broad query/result persistence |
| `37t.11` | P2 context scheduler | Keep P2; add execution-authority firewall |
| `avna.2` | P2 PACK-A/B | Keep P2; prerequisite for meaningful pattern product work |
| `uh6c` | P2 scalar tags | Keep P2 but rewrite to three axes before implementation |
| `o21` | P4 registry ergonomics | Raise to P2 before registry families multiply further |
| `stc` | P4 experiments | Raise to P2 after frame/evaluation contracts; causal-claim gate |
| `xv1u` | P2 generated curriculum | Keep P2, narrow to candidate-only experimental v1 |
| `rxdo.11` | P2 loop registry | Keep P2 spec; activate only L1/L2 pilots |
| `212.6` | P4 D8 | Raise to P2 after `tsk`; external continuity wedge |
| `hg8n` | P1 outside adoption | Keep P1; terminal proof remains one cold external user |
| `ei94` | P2 conductor | Keep P2; add active-lane/resource budget |
| `2yax` | P3 cluster tool | Raise to P2 after conductor protocol; collision-aware frontier |

Hard dependency direction:

```text
definition protocol + privacy + frame contract
    -> query execution/provenance wiring
    -> findings/reports and standing queries

PACK-A/B + EventOrderSpec
    -> pattern matches
    -> goal inactivity/survival claims

frame contract + evaluation receipts
    -> ExperimentDefinition
    -> causal improvement claims

context scheduler firewall + ExperimentDefinition
    -> minimal generated curriculum arm
    -> adaptive curriculum (later)

declare-once ergonomics
    -> additional classifier/marker/loop/ranker registries
```

## Delivery sequence

### Phase 0: finish and reconcile the current lanes

Finish hot-daemon, query-language, provider-origin, embeddings-hygiene,
fast-forward, and EQP work with honest per-AC outcomes. Do not mix the design
rewrite into their existing PRs unless the changed contract is directly in
their scope. Record blockers and residuals in the owning Beads.

### Phase 1: correctness boundary before more runtime wiring

One cohesive design/contract batch:

- `rxdo.2` protocol version/evaluator/persistence policy;
- `rxdo.3` frame/evaluation/privacy envelope;
- `3uw` coverage ref;
- privacy/excision edges;
- renderer vocabulary in `bkzv`.

Proof: a query that is enumeration-exact, frame-incomplete, model-derived,
privacy-restricted, and reproducible at one evaluation world renders honestly
and is excisable.

### Phase 2: prove the two external wedges

Audit slice:

```text
query/result -> finding -> ancestry check -> basket/report -> claims view
```

Continuity slice:

```text
fresh error -> prior recovery candidates with precision@k
abandoned session -> evidence-cited resume brief -> actual continuation
```

Then run `hg8n` with a cold external user. This is product validation, not the
only source of operator value.

### Phase 3: semantic analysis contracts

- PACK-A/B;
- actions-only order-explicit pattern v1;
- goal/question graph with `unresolved_inactive(H)`;
- tag-axis split;
- judgment verdict/partial-order/calibration extensions.

### Phase 4: governed adaptation

- ExperimentDefinition substrate;
- two improvement-loop pilots;
- candidate-only curriculum v1 experiment;
- promote additional loops only from receipts.

### Phase 5: horizon expansion

Only after the preceding receipts:

- mixed-stream patterns;
- broader D1-D9 atlas;
- predictive/process-mining analytics;
- autonomous ontology evolution;
- federation/portable bundles;
- reverse provider organization.

## Required falsification receipts

Every major design claim above has a failure test:

| Claim | Falsification |
|---|---|
| Frame contract adds honesty | Seed exact stored rows with a missing origin and classifier disagreement; output must not collapse to one “exact” badge |
| Protocol-versioned hash is meaningful | Change operator/field semantics without changing query text; identity must change or the contract fails |
| Persist-on-promotion protects privacy | Run ad-hoc secret-bearing query; no durable user-tier plan/member copy may remain, and ops TTL/excision must remove the temporary payload |
| Tag axes are distinct | High-affinity/unasserted and asserted/low-affinity cases must remain representable |
| Event order is honest | Same timestamp/tie under observed vs replay-verified order must produce distinct grade or ambiguity, not silent sequence |
| Injection firewall works | Adopted knowledge cannot become executable instruction without policy authority |
| Partial-order judgment works | Incomparable items must not receive a fabricated total rank |
| Loop registry is reusable | L1 and L2 run without per-loop scheduler/state machinery forks |
| Curriculum v1 earns expansion | Matched experiment shows useful effect without unacceptable correction/leakage cost |
| Lane cap helps | Compare collision/rework/merge latency before and after; lower concurrency must not merely reduce throughput |

## Provisional operator decisions

These choices are provisionally ratified from Fable's recommendations. They
should be applied in the corrective Beads pass, but remain revisable when
implementation evidence or the named falsification receipts contradict them.

1. **`@last` retention:** 48-hour TTL, one slot per `(workspace, surface)`,
   independently excisable. It is a same-day convenience rather than durable
   history.
2. **Instruction authority:** use a distinct `policy` assertion kind rather
   than an authority boolean on arbitrary assertions. Executability still
   requires explicit adoption, scope, authority provenance, validation, and
   revocation; the kind is necessary but does not authorize itself.
3. **Lane budget:** make admission adaptive by contention class. Allow one
   migration-touching writer per tier/window, one live archive writer, and
   bounded generated-surface writers only where their outputs do not overlap.
   Use four heavy lanes as the conductor's default backstop; lightweight
   read-only work may exceed it but remains subject to memory and I/O limits.
4. **First external activation:** run D3 prior-recovery candidates before D8
   actual resume. D3 has the smaller cold-start dependency surface; D8 remains
   the stronger subsequent continuity proof.
5. **ExperimentDefinition placement:** begin as a versioned typed assertion
   payload. Promote it to a dedicated durable schema only after two materially
   different consumers stabilize the shared lifecycle.
6. **Definition protocol versions:** use one language-wide version plus
   referenced component versions. The coarse version supplies cheap
   comparability gating; component refs supply precise invalidation and
   reproduction.

## Final shape

Polylogue should become a local evidence ledger and continuity system with a
small shared semantic kernel, typed domain definitions, honest evaluation
worlds, and governed promotion from observation to instruction or action.

It should refuse to become:

- a warehouse of permanently retained search literals and intermediate
  provenance;
- a universal schema that erases domain lifecycle and authority;
- an autonomous self-modifier whose own outputs grant themselves authority;
- a causal-claim machine built on observational correlations;
- a proliferation of parallel ledgers, baskets, queues, registries, and
  receipts that differ only in name;
- or a design encyclopedia whose entire contents are simultaneously treated
  as executable WIP.

The backlog may remain large. The semantic kernel, active frontier, and number
of independently evolving mechanisms should become smaller.
