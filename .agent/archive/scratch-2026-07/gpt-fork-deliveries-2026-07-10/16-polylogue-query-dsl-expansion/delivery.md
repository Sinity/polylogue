## Turn 119 (role=assistant)

# Polylogue Query DSL expansion package delivered

The complete implementation and research package is here:

**[Download the full Polylogue Query DSL expansion kit](sandbox:/mnt/data/Polylogue-query-DSL-expansion-kit.zip)**  
[SHA-256 checksum](sandbox:/mnt/data/Polylogue-query-DSL-expansion-kit.zip.sha256)  
[Machine-readable release receipt](sandbox:/mnt/data/Polylogue-query-DSL-release-receipt.json)

Two smaller cuts are also available:

- **[Patch, proof, vectors, and reviewer bundle](sandbox:/mnt/data/Polylogue-query-DSL-review-bundle.zip)**
- **[Sixteen-agent swarm prompt pack](sandbox:/mnt/data/Polylogue-query-DSL-swarm-prompts.zip)**

The fastest entry points are:

- [Start here](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/00-START-HERE.md)
- [Artifact map](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/ARTIFACT-MAP.md)
- [Executive assessment](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/01-executive-assessment.md)
- [PDF report](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/Polylogue-query-DSL-expansion.pdf)
- [HTML report](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/query-dsl-expansion-report.html)
- [Validation ledger](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/VALIDATION.md)
- [Non-claims](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/NON-CLAIMS.md)

This package is based on exact Polylogue commit:

```text
f6c1da997bea64bc6cd9670d9cbb8f7e7439ec51
```

The principal apply-ready artifact is:

**[Monolithic exact-base patch](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/patches/polylogue-query-dsl-expansion.patch)**

A reviewable ordered series and concern-separated patches are under [patches](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/patches/README.md).

---

# What was implemented

I selected three constructs with the highest immediate analytical value and the lowest risk of corrupting the language’s existing semantics.

## 1. Constrained action sequences

The existing bare sequence arrow retains its historical meaning: ordered, but not necessarily adjacent.

The expansion adds two stronger constraints:

```text
->[next]
```

This means the next normalized action in the same sequence domain. Intervening actions make the sequence fail.

```text
->[within:5m]
```

This means that the matched endpoints must occur in order and within the stated elapsed-time bound.

The semantics deliberately distinguish several cases:

- an ordered non-contiguous sequence;
- an immediately adjacent sequence;
- a bounded elapsed-time sequence;
- a bounded sequence with a real intervening action;
- missing timestamps;
- equal timestamps;
- cross-session candidates;
- partial structural evidence.

Missing time is not treated as zero elapsed time. A `within` predicate requires usable endpoint timestamps. This prevents unknown timing from being converted into a confident match.

The existing unconstrained arrow was not redefined. That compatibility decision is important: introducing a stronger operator should not silently alter archived queries.

The complete contract is in [grammar and semantics](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/design/grammar-and-semantics.md).

## 2. Typed numerical aggregates and nearest-rank percentiles

The implementation adds numerical aggregation only where the actual SQLite index has real numerical columns and the query unit has a defensible interpretation.

The supported design covers quantities such as:

- message word counts;
- durations;
- fresh input tokens;
- cache-read tokens;
- reasoning tokens;
- ordinary output tokens;
- other explicitly indexed numerical fields.

Percentiles use an explicit **nearest-rank** definition:

```text
rank = ceil(percentile × number_of_non_null_values)
```

The sorted value at that one-based rank is returned.

The contract specifies:

- the handling of null values;
- empty input;
- unsupported fields;
- invalid percentile bounds;
- grouping behavior;
- stable aggregate names;
- physical versus logical counting caveats;
- why percentiles cannot be silently approximated by averages.

This is especially relevant to AI work. Mean latency, token use, or tool duration can conceal a damaging tail. A p90 or p95 view can reveal pathological sessions that an average makes invisible.

The implementation does not pretend that SQLite has a universal percentile primitive. The lowering and verification explicitly define the selected algorithm. See [lowering and SQL shapes](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/design/lowering-and-sql-shapes.md).

## 3. Parent-session grouping

The third construct supplies the first useful cross-unit analytical bridge: grouping child observations by their owning physical session through `session.id`.

This enables analyses such as:

- actions per physical session;
- failed actions per session;
- token distributions grouped by session;
- message statistics grouped by session;
- tail behavior across sessions.

Its scope is intentionally narrow.

It does **not** claim to be:

- a general relational join language;
- logical-lineage grouping;
- root-session composition;
- shared-prefix deduplication;
- arbitrary cross-unit navigation;
- a replacement for Polylogue’s topology-aware readers.

That narrowness is a feature. A broad `join` operator introduced prematurely would need cardinality rules, ref identity, privacy propagation, projection freshness, null behavior, ordering, and cost controls. `session.id` delivers substantial value without disguising those unsolved contracts.

---

# Why these constructs won

The candidates were evaluated against Beads evidence, the present grammar, the actual SQLite DDL, compatibility risk, and whether each feature could be demonstrated with an independent oracle.

| Candidate | Decision | Reason |
|---|---|---|
| Constrained action sequences | Implemented | Directly supports edit→test→edit and recovery-pattern analysis; extends existing sequence semantics cleanly |
| Numerical percentiles | Implemented | High analytical value for latency, token, duration, and tail-pathology work; independently specifiable |
| Parent-session grouping | Implemented | Valuable cross-unit aggregation with bounded cardinality and a real schema relationship |
| Named-query set algebra | Deferred | Requires typed result-set identity, ranking semantics, duplicate policy, revision provenance, and mixed-unit rules |
| Saved-query invocation | Deferred | Requires immutable query revisions, parameter typing, recursion policy, authority, and reproducibility |
| General joins | Deferred | Too easy to introduce unbounded scans, cardinality explosions, and topology ambiguity |

The deferred designs are not vague backlog notes. They are implementation-grade follow-on packets:

- [Named-query set algebra](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/follow-ons/named-query-set-algebra.md)
- [Saved-query invocation](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/follow-ons/saved-query-invocation.md)
- [Generic cross-unit joins](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/follow-ons/generic-cross-unit-joins.md)

The broader sequencing is in the [query-language roadmap](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/04-query-language-roadmap.md).

---

# The proof system

The work does not rely on one self-confirming test layer.

The proof design uses several independent mechanisms.

## Production compatibility tests

The existing parser, lowering, search, and pipeline suite was rerun after the expansion:

```text
388 tests passed
1 pre-existing test skipped
```

Focused Ruff and strict MyPy checks also passed.

The implementation preserves:

- hand-parsed pipeline stages outside the Lark expression grammar;
- the strict command floor for unquoted bare words;
- existing bare-arrow semantics;
- actual-schema-only lowering;
- explicit parse and usage errors rather than silent widening.

## Executable production vectors

The package includes machine-readable vectors extracted from and mapped to concrete production tests:

- [Production vectors](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/vectors/production-vectors.json)
- [Negative production vectors](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/vectors/production-negative-vectors.json)
- [Production vector receipt](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/proofs/production-vector-receipt.json)

The vectors record:

```text
query
expected grammar/lowering behavior
expected SQL-shape properties
expected result behavior
owning production test
```

## Independent reference corpus and oracle

A second corpus was authored independently of the production implementation.

It contains deliberately distinguishable cases for:

- non-contiguous ordered matches;
- immediate adjacency;
- elapsed-time bounds;
- missing timestamps;
- equal or boundary timestamps;
- percentile values with known nearest-rank answers;
- grouped action and failure counts.

Artifacts:

- [Reference corpus](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/vectors/reference-corpus.json)
- [Reference vectors](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/vectors/reference-vectors.json)
- [Reference oracle receipt](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/proofs/reference-oracle-receipt.json)
- [Human-readable vector table](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/proofs/sql-shape-and-result-vectors.md)

The production implementation and reference oracle do not share the same lowering logic.

## Two anti-vacuity controls

The campaign proves that its test mechanisms can fail.

The first control deliberately corrupts a reference-vector expectation. The harness rejects it:

- [Reference anti-vacuity receipt](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/proofs/anti-vacuity-receipt.json)

The second mutates a real production assertion from equality to inequality. The corresponding production test fails:

- [Production mutation receipt](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/proofs/production-anti-vacuity-mutation.json)
- [Applied mutation](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/proofs/production-anti-vacuity.patch)

This is materially stronger than reporting only green tests. It establishes that at least two independent proof paths notice a deliberately false result.

## Tokenization proof

New syntax involving colons and bracketed sequence constraints was checked against the LALR tokenization hazards identified in the mission.

The proof covers:

- terminal priority relative to `FIELD_CLAUSE`;
- `within:` tokenization;
- invalid duration forms;
- invalid constraint names;
- old syntax remaining old syntax;
- strict handling of unsignalled bare terms.

See [tokenization proof](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/proofs/tokenization-proof.md).

## SQL-shape and bounded scale probe

The package includes a synthetic SQLite probe over:

```text
100 sessions
100,000 normalized actions
```

It exercises reference plan shapes for:

- immediate adjacency;
- five-minute bounded sequence matching;
- a high-percentile numerical aggregate;
- action grouping by session.

Artifact:

- [Synthetic SQL probe](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/proofs/synthetic-sql-probe.json)

This is a bounded query-plan and complexity probe, not a production-scale service-level claim. That limitation is explicit in [performance and complexity](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/design/performance-and-complexity.md).

## Machine-readable campaign receipt

The aggregate proof result is available at:

- [Campaign receipt](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/proofs/campaign-receipt.json)
- [Proof packet schema](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/proofs/proof-packet.schema.json)
- [Claim-to-proof matrix](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/02-proof-matrix.md)

---

# An important remaining evidence gap

I have not converted the vector campaign into a claim that every vector was proven through one final public CLI invocation against Polylogue’s seeded deterministic demo archive.

The current proof is strong at four layers:

1. production parser and runtime tests;
2. vectors mapped to concrete production tests;
3. an independent reference corpus and oracle;
4. anti-vacuity and mutation controls.

But a true public-path adapter should independently:

- create the deterministic demo archive;
- invoke the public CLI or service boundary;
- execute each positive and negative vector;
- capture the emitted SQL or bounded plan description;
- compare public results to the declared fixture oracle;
- emit one portable receipt.

The attempted black-box status is preserved rather than hidden:

- [Black-box attempt receipt](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/proofs/demo-blackbox/receipt.json)

The first item in the rapid follow-on program, `QDSL-01`, closes precisely this gap. It is not acceptable to rename production test-node execution as a public end-to-end demo.

---

# Demo portfolio reconsidered from first principles

The demo portfolio is in:

**[Construct-valid Query DSL demo portfolio](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/03-demo-portfolio.md)**

Each demo is designed around:

- a consequential question;
- independently planted ground truth;
- a credible simpler baseline;
- positive and negative controls;
- a refusal or incomplete-evidence control;
- one explicit falsifier;
- exact source and result refs;
- machine and human artifacts;
- a non-claims section.

## 1. Thrash Loop Under Oath

Question:

> Which sessions contain an actual edit→test→edit loop, and which only contain those action kinds in an unrelated order?

The corpus includes:

- a real adjacent loop;
- an ordered but non-adjacent loop;
- an edit and test separated by unrelated work;
- a loop outside the allowed time window;
- a loop with missing timestamps;
- prose containing the words “edit” and “test” but no structural actions.

It compares:

- keyword search;
- unconstrained sequence;
- `next`;
- `within`.

The demo succeeds only when the constrained operators distinguish the planted cases.

## 2. The Tail the Average Hid

Question:

> Which sessions have pathological latency or token tails that the mean conceals?

The fixture is designed so two cohorts have similar averages but materially different upper tails. The demo compares mean, median, p90, and p95 against independently computed nearest-rank values.

This demonstrates why numerical percentile support is not merely syntax decoration.

## 3. The Parent Trap

Question:

> Are failures concentrated in a small number of physical sessions?

The demo groups action outcomes by `session.id` and includes a copied-lineage scenario. It explicitly shows the difference between:

- physical owning session;
- logical lineage root;
- copied prefix;
- fresh subagent.

It must not silently claim that physical grouping is logical deduplication.

## 4. Missing Time Is Not Zero Time

Question:

> Should an event with unknown timing satisfy a bounded sequence predicate?

The answer is no.

This demo is a refusal proof: incomplete temporal evidence yields a caveat or non-match rather than a fabricated duration.

## 5. Query Semantics Diff

Question:

> Would this saved or historical query mean the same thing under a future language version?

The prototype records syntax, normalized representation, semantics version, expected plan properties, and result-set identity. It becomes the groundwork for query migration and revisioned saved queries.

## 6. A Federated Moment with Sinex

Question:

> Given a Polylogue sequence match, what machine evidence surrounds it?

Polylogue returns typed session/action refs and a half-open time interval. Sinex expands that interval into terminal, Git, browser, filesystem, or source-health evidence. The systems exchange refs, intervals, frontiers, privacy state, and caveats—not free-form query strings.

## 7. Saved Cohort Time Machine

Question:

> Can a named cohort be rerun later without silently inheriting a new query meaning?

This is the proving ground for immutable saved-query revisions, typed parameters, set algebra, and explicit semantics migration.

---

# Sinex interoperability

I inspected the supplied newer Sinex snapshot rather than treating interoperation as a generic future note.

Artifacts:

- [Sinex query-surface inventory](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/analysis/sinex-query-surface-inventory.md)
- [Polylogue–Sinex query interop design](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/design/sinex-query-interop.md)
- [Typed query-handoff JSON Schema](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/design/query-handoff.schema.json)
- [Polylogue-to-Sinex example](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/design/query-handoff.polylogue-to-sinex.example.json)
- [Sinex-to-Polylogue example](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/design/query-handoff.sinex-to-polylogue.example.json)

The recommendation is **not** to merge the two query grammars.

Polylogue’s language owns transcript-domain concepts such as:

- messages and blocks;
- normalized actions;
- tool sequences;
- logical session lineage;
- usage lanes;
- assertions;
- context deliveries.

Sinex’s query system owns broad evidence domains, source-material state, temporal quality, cross-source intervals, operations, and projection frontiers.

The interoperable object should be a typed result envelope containing:

```text
source system
result identity
query semantics version
terminal unit
namespaced refs
half-open time intervals
clock quality
ordering contract
source frontier
projection frontier
privacy state
inaccessible-result count
caveats
```

This also fits the maximal backend direction established earlier:

- Sinex can store provider-native transcript artifacts, normalized material, and durable Polylogue-domain history.
- Polylogue still owns the AI-work ontology and the meaning of its query language.
- PostgreSQL and Sinex events can be the durable substrate.
- Polylogue SQLite can remain a standalone authority, offline replica, local projection, and search accelerator.
- Query results can cross the boundary as typed refs and intervals without flattening either ontology.

Future typed set algebra should operate over compatible ref sets. It should not pretend that a set of Polylogue messages and a set of Sinex source materials can be intersected without an explicit relationship or projection.

---

# Beads-aligned rapid execution plan

The execution program is here:

- [Rapid Beads execution plan](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/beads/rapid-execution-plan.md)
- [Existing Beads alignment](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/beads/existing-beads-alignment.csv)
- [Proposed machine-readable issues](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/beads/proposed-issues.json)
- [Current `polylogue-fnm` evidence](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/beads/polylogue-fnm-evidence.md)

The proposed frontier is:

1. Build a true public deterministic-demo vector adapter.
2. Add positive, negative, refusal, and boundary controls to the public corpus.
3. Prove sequence plan behavior on long, heterogeneous sessions.
4. Add property tests for percentile monotonicity and nearest-rank behavior.
5. Prove physical-session grouping versus copied lineage and fresh subagents.
6. Expose a bounded query-explain surface.
7. Stabilize parse, usage, unsupported-field, and execution-error categories.
8. Ship Thrash Loop Under Oath as the flagship public demo.
9. Add immutable, revisioned saved-query definitions.
10. Add typed set algebra over compatible terminal-unit ref sets.
11. Add declared cross-unit relationships rather than generic joins.
12. Land the Polylogue–Sinex ref-set and interval handoff.

The sequencing is intentional:

```text
public black-box vectors
        ↓
semantic and property proofs
        ↓
bounded explainability
        ↓
flagship demo
        ↓
revisioned saved queries
        ↓
typed set algebra
        ↓
controlled cross-unit relationships
        ↓
Polylogue–Sinex federation
```

Saved queries precede named-query set composition because a named set needs an immutable query revision and parameter contract before it can be reproducible.

---

# Single-machine frontier-agent swarm

The complete procedure is in:

- [Swarm runbook](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/swarm/runbook.md)
- [Machine-readable work packages](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/swarm/work-packages.json)
- [Initial coordination board](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/swarm/board.tsv)
- [Worktree bootstrap helper](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/scripts/bootstrap-worktrees.sh)
- [Full verification driver](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/scripts/verify-all.sh)
- [Heavy-operation lock](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/scripts/with-heavy-lock.sh)

The main coordination rule is that only one worker owns the grammar and lowering hotspot at a time. Nearly every new construct wants to touch `expression.py`; allowing several agents to edit it concurrently creates expensive semantic conflicts even when Git can merge the text.

Other work proceeds in parallel across isolated worktrees:

```text
integration
grammar-and-lowering
demo-vectors
sequence-properties
percentile-properties
performance
grouping-and-lineage
errors-and-explain
docs-and-demo
set-algebra-design
saved-query-design
Sinex-interop
adversarial-review
```

The swarm uses four distinct authority roles:

**Grammar captain.** Sole owner of parser, AST, and lowering decisions during each integration window.

**Proof captain.** Owns independent oracles, anti-vacuity controls, and claim-to-proof mapping. This should not be the same agent that implemented the construct.

**Beads captain.** Sole writer of Beads state during the campaign. Workers submit proposed changes in handoffs rather than racing on the planning database.

**Integration captain.** May cherry-pick, copy clean file sets, apply patches, squash, or discard branch history. Semantic correctness and proof artifacts matter more than tidy intermediate Git history.

Heavy operations are serialized:

- broad test suite;
- MyPy;
- large SQLite probes;
- deterministic archive generation;
- media rendering;
- final package verification.

Documentation, fixture design, property-test design, review, and interop work can continue concurrently.

---

# Sixteen fork-ready prompts

The standalone package is:

**[Download all sixteen prompts](sandbox:/mnt/data/Polylogue-query-DSL-swarm-prompts.zip)**

The missions are:

1. Grammar and LALR tokenization audit.
2. Direct public production-vector adapter.
3. Deterministic independent ground-truth corpus.
4. Sequence semantics and property audit.
5. Percentile property and boundary suite.
6. Physical session versus logical-lineage proof.
7. Sequence SQL-plan analysis.
8. Percentile SQL-lowering and scale analysis.
9. Negative syntax, strict command floor, and error taxonomy.
10. Bounded query-explain surface.
11. Large-session performance and complexity proof.
12. Typed named-query set algebra design.
13. Revisioned saved-query invocation design.
14. Thrash Loop Under Oath public demo.
15. Polylogue–Sinex query federation and backend compatibility.
16. Adversarial integration and release review.

Each prompt includes:

- owned files;
- relevant Beads;
- forbidden scope expansion;
- expected semantics;
- independent proof obligations;
- positive and negative controls;
- exact deliverables;
- handoff format;
- integration dependencies.

They are intended to be run as parallel forks of this conversation while one integration captain maintains the authoritative worktree.

---

# Applying the implementation

The patch is tied to the exact base commit to avoid ambiguous application:

```bash
git worktree add --detach /tmp/polylogue-query-dsl \
  f6c1da997bea64bc6cd9670d9cbb8f7e7439ec51

git -C /tmp/polylogue-query-dsl apply \
  polylogue-query-dsl-expansion.patch

git -C /tmp/polylogue-query-dsl diff --check
```

The package contains a complete verification driver and detailed reviewer guide:

- [Reviewer guide](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/review/reviewer-guide.md)
- [Adversarial checklist](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/review/adversarial-checklist.md)
- [Release-decision template](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/review/release-decision-template.md)
- [Worker handoff template](sandbox:/mnt/data/polylogue-query-dsl-expansion-pack/review/WORKER-HANDOFF.md)

The package manifests and archives were checksum-verified, the monolithic patch was checked against a clean exact-base worktree, and the ZIP archives passed integrity checks.

---

