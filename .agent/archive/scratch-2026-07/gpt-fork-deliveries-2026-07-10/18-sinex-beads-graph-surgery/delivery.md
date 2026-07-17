<!-- This branch has no captured turn-79-style pivot user message (unlike sibling branches); it diverges directly from the shared turn-45 'external legibility' ask into Sinex Beads graph surgery work, referencing an attached spec at /mnt/data/10-beads-program-surgery.md (visible in tool output) without a separately transcribed user turn introducing it. -->

## Turn 116 (role=assistant)

# Sinex Beads graph surgery completed

I adjudicated the packaged Sinex graph under the attached standing contract, including the requirement that this remain a static, independently applicable proposal rather than a mutation of live Beads. fileciteturn0file0

The complete package is here:

**[Download the Sinex Beads graph-surgery package](sandbox:/mnt/data/sinex-beads-graph-surgery.zip)**  
[SHA-256 checksum](sandbox:/mnt/data/sinex-beads-graph-surgery.zip.sha256): `68af3f3594b2946c0ef4b536d33001c658ffabfdbad7ad87343c23c6ae8913a5`

The source repository remains clean. No `bd update`, `bd close`, dependency mutation, commit, or other repository-state change was performed.

## Result

The supplied snapshot contains **326 issues and 15 durable memories**, rather than the approximate 640-item estimate in the mission. The source checkout is `feature/fix/cloud-harness-bootstrap` at commit `5e788c31a6a8382ec12e8681d5e69f568b71c092`.

The final delta contains **79 independently applicable operations**:

| Operation | Count |
|---|---:|
| Updates | 51 |
| Relabels | 11 |
| Dependency additions | 9 |
| Merges | 6 |
| Dependency removals | 2 |
| Direct closes | 0 |

Confidence is deliberately conservative:

| Confidence | Count |
|---|---:|
| Proven against source and snapshot | 15 |
| Likely | 8 |
| Proposal/design judgment | 56 |

The machine-applicable proposal is:

**[beads-delta.jsonl](sandbox:/mnt/data/sinex-beads-surgery/beads-delta.jsonl)**

Every line has the requested `op`, `id`, `fields`, `rationale`, `evidence`, and `confidence` fields. The operations were checked for independence: no merge source is targeted elsewhere, no merge append overlaps another survivor update, no update fields overlap, and no edge operations conflict.

## Why there are no direct close operations

Several open Beads have materially stale problem statements, but static evidence does not establish their complete acceptance contracts:

- `sinex-d4qg`: the dedicated API-pool mechanism and capacity accounting have landed; realistic concurrent production proof remains.
- `sinex-5ai`: four of the original five convergence gaps appear addressed; startup COPY verification remains.
- `sinex-v7od`: test-binary aggregation has landed; inventory, performance measurement, and broad failure triage remain.
- `sinex-lb8`: the prime-removal portion has landed; defragmentation, granularity, and naming work remain.
- `sinex-9wx`: the original bootstrap race is fixed, but its remaining parallel proof-seal control belongs in `sinex-9es`.

Closing those would overstate static source inspection and, for test/integration work, violate the mission’s calibration. They are narrowed, updated, or merged instead.

## Six sibling merges

The proposed merges are:

1. `sinex-lv8.1 → sinex-k6s`  
   The entity/relation query-unit split is fully subsumed by the richer evidence-graph query design.

2. `sinex-9wx → sinex-9es`  
   The original race is fixed; the residual fresh-checkout concurrency control belongs to the proof-substrate seal.

3. `sinex-pya → sinex-hlv`  
   The retired conductor corpus is historical source material for the current Beads/devloop ingestion path, not a second live work-state authority.

4. `sinex-pi1 → sinex-ej9`  
   Both own the README, concepts, positioning, proof links, and first-run legibility path. The richer public-legibility Bead survives.

5. `sinex-sn6s → sinex-r6d.8`  
   Source criticality and reconstructibility are an operator-facing projection of the existing `SourceRecoveryPolicy` design.

6. `sinex-r6d.13 → sinex-5smc`  
   Both describe the same deterministic confirmation-publication failure. The exact two-versus-three-failure control moves into the richer durability decision.

## Frontier specification repair

Sinex does not contain Polylogue-style `horizon:frontier|mid|vision` or A→N delivery labels. Introducing those labels would corrupt the repository’s native planning model.

I therefore used Sinex’s documented equivalent:

> `status=open`, no unresolved `blocks` dependency, and priority ≤2.

That produces a current execution frontier of **101 issues**.

Of those, **50 had blank design fields**. Every one is covered by a design update or merge. The proposed designs include:

- a settled decision or explicit options frame;
- concrete files, symbols, or subsystems;
- pitfalls and invariants;
- interacting Beads;
- observable verification rather than “tests pass.”

The two active non-epics without structured acceptance criteria—`sinex-d4qg` and `sinex-pasb`—receive field-level AC repairs.

The all-accepted simulation changes execution-grade coverage as follows:

| Metric | Before | After |
|---|---:|---:|
| Structured AC coverage | 237/239, **99.16%** | 233/233, **100%** |
| Structured design coverage | 138/239, **57.74%** | 183/233, **78.54%** |

The raw AC count falls only because six duplicate execution items are merged out of the active denominator; both pre-existing AC gaps are repaired.

## Label and horizon-equivalent corrections

The graph had **12 active violations** of its native invariant requiring exactly one `wave:N` and one `area:*` label.

Eleven receive additive relabel proposals. The twelfth, `sinex-sn6s`, disappears through its merge into `sinex-r6d.8`.

The all-accepted simulation has **zero wave/area violations**.

No priority was changed, no foreign horizon labels were introduced, and no wave was treated as a strict release gate. This follows the repository’s statement that waves represent genuinely parallel lanes, although hard dependencies still may not point to higher-wave prerequisites.

## Edge audit

All **212 existing hard edges** were inspected. The complete audit also records the nine proposed additions, producing 221 rows.

Important proposed hard additions include:

- `sinex-h3g` blocked by the shared durable cursor frontier and revision policy;
- `sinex-2pl` and `sinex-lcj` blocked by `sinex-32p2`’s expectation IR;
- `sinex-08z` blocked by the identity/revision policy it reports;
- `sinex-nbi.5` blocked by the common civil-time policy;
- `sinex-cem.13` blocked by the occurrence-identity regime matrix;
- the full Polylogue historical backfill blocked by the pacing/capacity prerequisite.

Two hard dependencies are proposed for removal where they serialize independent proof or product work rather than establish an implementation prerequisite.

A separate parent-child edge attaches the surviving confirmation-durability issue to the substrate-honesty program without affecting readiness.

The simulated graph has no hard-dependency cycle and no wave inversion.

## Graph-health effect

| Metric | Before | After all accepted |
|---|---:|---:|
| Open issues | 247 | 241 |
| Active issues | 252 | 246 |
| Ready open issues | 142 | 132 |
| Blocked open issues | 105 | 109 |
| Hard edges | 212 | 218 |
| Unlinked active non-epics | 21 | 19 |
| Wave/area violations | 12 | 0 |

The ready count falls by ten because five duplicate ready items are merged and five false-ready items receive real prerequisites. It is corrective suppression, not a loss of executable work.

Nineteen active non-epics remain unlinked after simulation. They are identified in the report, but I did not fabricate parent or provenance edges merely to improve the orphan metric.

## Cluster map

The simulated 132-item ready set is partitioned exactly once across 14 file-footprint clusters, including:

- admission, settlement, receipts, and durability;
- schema and database lifecycle;
- temporal and interval semantics;
- source runtimes and material handling;
- operations and performance;
- xtask, tests, and proof infrastructure;
- query, recall, and evidence packs;
- authority, judgment, and agent context;
- Polylogue and external-producer work;
- new source families;
- legibility, demonstrations, and research;
- architectural cleanup;
- derivation and semantic graph;
- privacy and lifecycle.

Each cluster includes a suggested claim order and shared-hotspot warnings. In particular, the map warns against concurrent ownership of event persistence/schema surfaces, shared cursor and interval-revision machinery, xtask affected-test logic, generated source catalogs, context/authority DTOs, and destructive privacy paths.

## Validation

The package passed 20 final static checks:

- all 79 operations validate under the included JSON Schema;
- all 79 pass the stronger independence validator;
- all 265 evidence references—179 unique coordinates—resolve to existing, nonblank snapshot or source lines;
- all IDs, labels, existing edge removals, and new edge targets match the snapshot;
- all 326 issues appear exactly once in the full audit;
- all 101 frontier issues appear in the frontier audit;
- all 221 existing/proposed hard edges appear in the edge audit;
- all 132 simulated-ready issues appear exactly once in the cluster map;
- all 79 operations appear in the confidence-sorted review table;
- no cycle or wave inversion exists before or after simulation;
- the same-snapshot drift preflight classifies all 79 operations as `applicable`;
- the copied Beads export and archive hashes match the recorded provenance;
- every manifest checksum passes;
- the source checkout remains clean.

No Rust, Nix, PostgreSQL, NATS, production, security, or integration acceptance was executed. Those limitations are explicit in `NON-CLAIMS.md`.

## Primary artifacts

- [Human review table](sandbox:/mnt/data/sinex-beads-surgery/REVIEW-TABLE.md)
- [Review table as CSV](sandbox:/mnt/data/sinex-beads-surgery/review-table.csv)
- [Graph-health report](sandbox:/mnt/data/sinex-beads-surgery/GRAPH-HEALTH.md)
- [Machine-readable graph health](sandbox:/mnt/data/sinex-beads-surgery/graph-health.json)
- [Ready-work cluster map](sandbox:/mnt/data/sinex-beads-surgery/CLUSTER-MAP.md)
- [Ready clusters as CSV](sandbox:/mnt/data/sinex-beads-surgery/ready-clusters.csv)
- [Full edge audit](sandbox:/mnt/data/sinex-beads-surgery/EDGE-AUDIT.csv)
- [Complete 326-issue audit](sandbox:/mnt/data/sinex-beads-surgery/ISSUE-AUDIT.csv)
- [Frontier audit](sandbox:/mnt/data/sinex-beads-surgery/FRONTIER-AUDIT.csv)
- [Full-graph findings](sandbox:/mnt/data/sinex-beads-surgery/FULL-GRAPH-FINDINGS.md)
- [Evidence index](sandbox:/mnt/data/sinex-beads-surgery/EVIDENCE-INDEX.md)
- [NON-CLAIMS](sandbox:/mnt/data/sinex-beads-surgery/NON-CLAIMS.md)
- [Live-applier runbook](sandbox:/mnt/data/sinex-beads-surgery/APPLIER-RUNBOOK.md)
- [Read-only live-drift preflight](sandbox:/mnt/data/sinex-beads-surgery/preflight_live_delta.py)
- [Delta JSON Schema](sandbox:/mnt/data/sinex-beads-surgery/delta-operation.schema.json)
- [Final validation receipt](sandbox:/mnt/data/sinex-beads-surgery/FINAL-VALIDATION.md)
- [Package manifest](sandbox:/mnt/data/sinex-beads-surgery/MANIFEST.sha256)


---

