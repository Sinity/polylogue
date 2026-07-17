---
created: 2026-07-16
updated: 2026-07-16
purpose: Track architecture resolution and the remaining operator versus implementation decisions for the Testsuite Diet
status: architecture-adjudicated
project: polylogue
---

# Architecture scope and resolution

## Verdict

The previously unresolved product contracts are now adjudicated in
[`architecture/`](architecture/00-index.md). Four core branches and five
contingent branches have recommended defaults, competing alternatives,
migration seams, and proof obligations. They are no longer reasons for Terra to
invent semantics or ask broad operator questions.

This does not make their implementation packets executable automatically.
Current symbols, write sets, migrations, focused commands, historical seeds,
and exact mutation witnesses must still be resolved at the merged baseline.
Architecture is decided; source reconciliation and dossier preparation remain.

## Resolved core branches

| Branch | Laws | Adjudicated default | Implementation consequence |
| --- | --- | --- | --- |
| Evidence authority and identity | L01-L03 | One proof-driven raw reconciler; byte evidence outranks metadata; conflicts block; identities remain separated | Consolidate chooser/repair routes around a plan/apply/receipt/postflight protocol |
| Lineage composition and snapshots | L05-L06 | Canonical divergent tails plus typed edges; one deferred read snapshot; incomplete lineage is explicit and readable | Centralize normalization and return `LineageCompleteness` through readers/surfaces |
| Concurrent writes, leases, publication, resume | L07-L10 | Atomic SQL/CAS/immediate transactions by invariant; durable reservations and sagas across files | Add generation predicates, terminal reservation proof, deterministic failpoints, and restart receipts |
| Destructive and authentication boundaries | L23-L24 | Executable `OperationSpec` gateway, preview-bound confirmation, role/capability enforcement, stable receiver pairing | Route adapters and maintenance through one executor; complete packaged pairing proof |

## Resolved contingent branches

| Branch | Laws | Adjudicated default | Implementation consequence |
| --- | --- | --- | --- |
| Derived freshness | L11-L13 | Exact source plus recipe `DerivationKey` and active generation; domain-specific ledgers | Embedding predicate first; FTS identity ledger in a batched index generation; insight adoption after |
| Query cancellation and bounds | L14-L17 | One execution context, weighted admission, dedicated interruptible reader, lossless page/spool lifecycle | Land a production cancellation/resource seam before cancellation and scaling survivor laws |
| Evidence/provenance/public algebra | L18-L20, L25 | Adopt `EvidenceValue` independent axes and one canonical fact per domain | Dogfood three canaries, then migrate domain slices; MCP/web consume at rewrite boundary |
| Configuration/path coherence | L26-L28 | Existing five-layer loader is sole authority; resolve once and inject one archive/path identity | Convert composition roots and legacy path/config calls without changing precedence |
| Capture/deployed status | L29-L30 | Production service graph, durable capture checkpoints, per-component snapshots, evidence-bound runtime/termination identity | Extract graph, split status cache, bind receipts, add optional host reconciliation |

## Remaining operator actions

Only two product actions require explicit human authority:

1. **Live raw repair actuation.** Review and authorize the immutable digest of a
   specific read-only reconciler plan after preflight. This is permission to
   mutate live archive evidence, not a choice among authority algorithms.
2. **Receiver trust replacement.** Explicitly re-pair when the observed receiver
   identity differs from the trusted identity. Endpoint failover for the same
   identity is automatic and bounded; identity change is not.

Optional access to local journal/cgroup/kernel evidence improves termination
diagnosis. Denying it leaves cause `unknown` and does not block runtime or the
deterministic test program.

Numeric query budgets, admission weights, status deadlines, and reconciliation
inspection ages are selected from workload evidence and ship with conservative
defaults. They are configuration tuning, not unresolved semantics.

## Still unresolved, but not architecture

- merged identities and source symbols after workload-profile/harness work;
- exact PR boundaries and schema-bump batching;
- exact owned/avoided files and focused commands per cluster;
- realized sensitivity and deletion-dominance evidence;
- whether current partial implementations already satisfy a slice;
- runner timeout/interruption hardening and direct certification exercise;
- realized LOC/runtime/fixture/model economics;
- rewrite-native MCP/web implementation sizing;
- promotion of this ignored scratch corpus before worktree removal.

These are reconciliation, execution preparation, implementation, measurement,
or publication work. They do not authorize workers to revisit the decisions.

## Implementation packet rule

An architecture-sized dossier must cite one decision document and translate it
to current source:

1. exact production symbols and current mechanism delta;
2. exact legal states/outcomes and public projection;
3. exact files owned/avoided and migration boundary;
4. historical seed plus independent state/metamorphic/fault oracle;
5. deterministic production failpoint or mutation;
6. focused commands and expected resource envelope;
7. rollback/restart/postflight contract;
8. named residual obligations and rewrite handoffs.

If current source contradicts a decision invariant, the worker returns a
blocker with evidence. If it merely makes an alternative easier, the default
stands.

## Risk and sequencing

Risk order remains:

1. evidence authority, because a wrong winner silently rewrites truth;
2. concurrent durable writes/publication, because plausible success can lose
   operator state or effects;
3. destructive/authentication boundaries, because one bypass defeats surface
   parity;
4. lineage composition, because mixed snapshots fabricate history;
5. query execution and installed runtime, because leaks and hangs appear only
   under composition/load;
6. derived freshness and evidence/config projections, which are broad but can
   migrate domain by domain.

The opening corpus/query-membership/convergence/devtools portfolio remains
independent of most architecture implementation. Architecture branches should
land as isolated or serialized hotspot waves before test subtraction in their
own areas.

## Scale

- Nine adjudicated design packets cover the four core and five contingent
  branches.
- They likely translate to roughly 9–14 production slices because the raw
  reconciler, operation gateway, query lifecycle, and installed runtime may
  need staged migrations, while some config/evidence/freshness work can batch.
- Each slice needs survivor implementation, independent mutation/fault
  certification, and Sol integration; deletion is optional and later.
- This is still a multi-PR program. The architecture set reduces decision
  latency; it does not turn the portfolio into one safe autonomous run.
