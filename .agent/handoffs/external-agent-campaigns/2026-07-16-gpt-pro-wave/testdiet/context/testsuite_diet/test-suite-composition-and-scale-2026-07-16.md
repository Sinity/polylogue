---
created: 2026-07-16
purpose: Explain why the extensive suite missed dogfood failures and define a schema-driven, real-route test portfolio that catches the class
status: synthesized
project: polylogue
---

# Test-suite composition and scale synthesis

## Realized-baseline reconciliation

This synthesis predates the implementation of `polylogue-1xc.14.1` and the
residual `polylogue-b054.1.1.3`–`.5` proof work. Execution now assumes those
outcomes are realized: bounded privacy-safe package/archive profiles,
correlated variants, named scale/selectivity tiers, C-03 and related canaries,
shared workload/resource/cleanup receipts, real production-mutation/testmon
proof, and repeated isolated/xdist lifecycle witnesses. Recommendations below
to create those mechanisms are historical rationale. Diet work consumes them
under [`10-realized-baseline.md`](10-realized-baseline.md) and must not build
parallel authorities.

## Question

How can Polylogue have an extensive test suite while the dogfood failures in
`.agent/scratch/dogfood/` survive, and what fundamental change would make this
class of failure difficult to ship?

## Proof-form correction

The suite is not globally `unit/example-heavy`. A whole-tree audit found 210
`@given` tests, two rule-based state machines, 713 parametrized test functions,
and extensive real SQLite, filesystem, async, HTTP, subprocess, and fault
testing under `tests/unit/`. Directory placement is not proof form. See
[`11-test-proof-form-audit.md`](11-test-proof-form-audit.md) for the controlling
measurements and all-high narration.

This document's references to repeated examples apply only to named clusters
whose bodies were inspected. They do not authorize suite-wide example removal.

The scale question is part of the answer: the live archive activates execution
shapes that tiny fixtures cannot activate. It is not the whole answer. A large
fixture with a weak oracle, incoherent generated actions, or a mocked surface
would still pass.

## Verdict

The dominant escape pattern is **composition risk**, not insufficient test
count:

1. component behavior is proved independently;
2. schemas prove that individual provider records are accepted;
3. surface tests often prove their adapters or mocks;
4. scale tests prove that a database can be built and a bounded result is
   returned;
5. concentrated clusters preserve helper names, call graphs, module layout,
   catalog membership, historical absence, or weak shapes instead of product
   truth, while other large clusters already use generative, stateful, or
   real-route proof;
6. failures occur where selection, semantics, relational work, and projection
   cross those boundaries.

The 36.7 GB archive is an activation condition and amplifier. For the exact
session action case, direct indexed evidence takes 0.062 ms and a semantically
equivalent query with the session bound inside both window CTEs takes 0.271 ms,
while the current view exceeds a two-second progress budget. This proves that
the archive is not inherently too large for the requested answer. The faulty
composition ranks an unbounded result population before the outer predicate can
constrain it.

Therefore the fundamental unit of verification should become:

> a semantic law executed through the production composition route over a
> schema-generated workload shape, checked by an independent oracle and a
> bounded-work receipt.

Test volume, line coverage, file-presence matrices, and one-off endpoint tests
are supporting evidence, not closure.

## What the current schema system already provides

The provider schema system is explicitly intended to be observation-derived.
Generation annotates schemas with:

- field/document frequency;
- safe structural values;
- numeric ranges;
- array-length ranges;
- semantic formats and multiline shape;
- foreign-key references;
- time deltas;
- mutual exclusions;
- string-length relationships.

`SyntheticCorpus` consumes these annotations with a deterministic random seed.
This should remain the source of production-like provider distributions. Do not
create a second manually maintained table of “real distributions” beside it.

The appropriate chain is:

```
privacy-safe archive observations
    -> regenerated/versioned provider schemas and relation annotations
    -> deterministic CorpusSpec / workload scenario
    -> real provider artifacts
    -> acquire -> parse -> materialize -> index
    -> immutable archive build + manifest
    -> production-route semantic and work-bound tests
```

Schema regeneration is a separate upstream project, but the test architecture
must make schema freshness and provenance visible in the fixture digest and
manifest.

## Where the current generation promise is lost

### 1. Marginals are not joint distributions

Frequency, min/max range, and array min/max are useful but insufficient for
facts such as:

- a tool result refers to a preceding tool use in the same session;
- duplicate tool IDs occur within a session and pair by transcript occurrence;
- results may be missing, delayed, or arrive after intervening messages;
- a modern Codex `functions.exec` envelope contains the actual nested tool/path;
- branch width/depth correlates with transcript replay and session size;
- active sessions grow while convergence is incomplete;
- provider/version determines which structural variants co-occur.

Those are relational and lifecycle distributions. They need learned relation
annotations or scenario constraints layered on the schemas, not hand-written
wire payloads that bypass the schemas.

An empirical generation probe found that default Codex and Claude Code
generation emitted many tool uses and results but no within-session matched
tool IDs. The `tool-heavy` style creates direct paired calls, but neither style
emitted the modern nested `functions.exec` envelope implicated by the path
failure. Schema-valid output was therefore not semantically representative of
the relation under test.

### 2. CorpusSpec controls only coarse archive shape

`CorpusSpec` presently provides provider, artifact count, a uniform
messages-per-session range, seed, style, package version, and profile metadata.
It does not express a learned session-size distribution, provider mix,
tool/result lag distribution, lineage topology, selectivity, active growth, or
large payload tail.

The schema runtime also samples numeric ranges uniformly and clamps generated
array lengths to at most five. Min/max preservation is not distribution
preservation, and the clamp deliberately erases observed tails.

Required extension: a versioned workload/corpus profile should refer to schema
packages and specify archive-level and cross-record relations. It should carry
histograms/quantiles or another privacy-safe sampler for session size and
payload length, not only extrema. Provider wire schemas remain authoritative
for record shape.

### 3. The benchmark fixture does not use the claimed source

`tests/benchmarks/conftest.py` says it uses `SyntheticCorpus`, but directly
constructs storage records from a hard-coded historical provider/session
distribution and word pools. Its content-block generator emits tool uses but
not tool results. It cannot activate the unbounded `ranked_results` defect even
at its large tier.

This is shadow fixture infrastructure: it resembles production by prose and
some marginals but bypasses provider artifacts, parsers, schema versions, and
the decisive relation.

### 4. Current scale assertions are vacuous for correctness

The six tier smoke tests assert either `0 <= search_hits <= limit` or that a
session list is nonempty and bounded. Search can return zero for every query and
pass. Large-fixture construction dominates runtime while the oracle proves
almost nothing about membership, completeness, cross-surface agreement, or
upstream work.

These tests are prime replacement/deletion candidates once a stronger
schema-driven scale law suite owns fixture lifecycle plus semantic assertions.

### 5. Extensive catalogs prove presence, not behavior

The closure matrix verifier checks that target files and representative test
files exist and that declared rows are populated. It cannot establish that a
test reaches the production dependency, has an independent oracle, activates a
failure shape, or kills a relevant mutation. This creates a high-confidence
looking inventory without a corresponding falsification claim.

## The replacement test architecture

### A. One schema-driven workload model, several intentional profiles

Do not build one universal 36 GB fixture. Generate a small portfolio from the
same versioned schema packages:

1. **Micro semantic corpus**: minimal exact cases for branch behavior and fast
   failure localization.
2. **Distribution corpus**: archive/provider/session/message/block shapes drawn
   from regenerated schema and corpus profiles.
3. **Tail corpus**: p95/max-like session, payload, lineage, and action-result
   populations.
4. **Adversarial relation corpus**: duplicate/missing/late results, repeated IDs,
   nested tool envelopes, wide/deep lineage, stale derived state, active growth,
   and high/low selectivity.
5. **Human-language corpus**: small and separate, used only where natural prose
   materially affects tokenization, semantic ranking, summaries, or UX.

Most correctness and relational-performance tests should use deterministic,
controlled tokens. LLM-realistic prose adds noise without helping identity,
count, pagination, lineage, action pairing, or predicate-pushdown proofs.

The “adversarial” profiles are still schema-driven: they select valid
low-frequency variants and relation states deliberately instead of hoping a
random seed samples them. A representative distribution fixture and a coverage
fixture serve different purposes.

### B. Generate through the real route

Build provider wire artifacts from `SyntheticCorpus`, then run the real
acquire/parse/materialize/index path. Tests should consume the resulting
five-tier archive when their semantics cross tiers. Direct storage-record
builders remain appropriate only for storage-local invariants.

Each built fixture publishes a manifest containing:

- workload-spec bytes and seed;
- schema package/version/content hashes and observation window;
- generator and materializer source/version hashes;
- tier DDL/user versions;
- expected counts and construct coverage;
- relation invariants (paired/missing/late/duplicate action counts, topology
  width/depth, selectivity buckets, payload quantiles);
- database hashes/sizes and `quick_check` results.

A seed alone is not reproducibility: the same seed under changed schemas or
generator code is a different corpus.

### C. Combine compositional laws with explicit witnesses

Run a generated set of canonical selections/projections through repository,
Python facade, CLI, and HTTP. Add the rewritten MCP surface when it exists;
do not build this law layer around the current MCP implementation. Compare
semantic facts, not rendering bytes.

Retain explicit examples for historical defects, provider compatibility,
security attacks, durability boundaries, public representations, and useful
diagnostics. Laws replace examples only when route, oracle, branch, mutation,
and diagnostic dominance has been demonstrated.
Important laws include:

- exact identity is preserved at every surface;
- equivalent DSL and structured filters select the same members;
- adding an irrelevant corpus changes neither result nor ordering;
- page concatenation equals the unpaged logical population exactly once;
- `LIMIT n` is a prefix/monotonic restriction and bounds upstream work;
- grouped counts sum to the matching-grain population;
- list-emitted refs resolve to the same detail object;
- direct cheapest-correct evidence agrees with the rich route;
- missing/unknown/stale/error states retain their distinct types;
- cancellation stops server work and continuation progresses.

The existing cross-surface agreement test is the right seed architecture, but
its case space currently covers only provider, text, and minimum-message
filters. Extend that harness rather than creating another parallel surface
framework.

### D. Treat work shape as part of correctness

Wall-clock thresholds alone are noisy and machine-specific. For selective
queries, collect:

- SQLite progress/VM work or rows visited;
- SQL statement count;
- `EXPLAIN QUERY PLAN` scans and temporary B-trees;
- bytes returned and temporary/read/write I/O;
- process/cgroup CPU, RSS/PSS, anonymous/cache/swap;
- cancellation latency and cleanup state.

Then express metamorphic growth laws. Example: appending 100,000 irrelevant
tool-result rows must not materially increase the work of an exact-session
action query. This catches the current defect without requiring realistic
prose or a full 36 GB fixture. The result cardinality alone cannot catch it.

Every new class-level harness needs anti-vacuity controls: remove predicate
pushdown, drop an authority input, corrupt continuation state, or widen a
projection and show the test fails.

## Test subtraction: a dominance rule

Adding the stronger suite without deleting weaker duplicates would worsen the
economics. Define a test as a proof claim with five dimensions:

1. production dependency/route executed;
2. independent oracle or semantic law;
3. data and lifecycle shapes activated;
4. relevant mutations/failure modes killed;
5. cadence and runtime cost.

Test A dominates test B only when A:

- executes the same production dependency or a stricter enclosing real route;
- asserts B's invariant or a strictly stronger one;
- includes B's relevant input/failure shape;
- kills every mutation/failure B uniquely kills;
- runs in the same or an earlier gate with acceptable diagnosis/cost.

Then B should be deleted or collapsed into A's generated case table. Preserve
small unit tests that uniquely cover combinatorial branches, give materially
better diagnosis, or run far earlier than the enclosing route. Avoid replacing
the suite with one opaque mega-E2E test.

Likely first subtraction candidates:

- replace all six scale-tier smoke tests with semantic/growth laws;
- consolidate mocked CLI query-branch examples once the production
  cross-surface query-law harness contains those branches and mutations;
- demote/remove presence-only closure claims once the verification-risk record
  proves production dependencies and anti-vacuity evidence;
- collapse repeated format/surface snapshots into parametrized semantic
  contracts while retaining one rendering golden per genuinely distinct
  format.

This is the practical meaning of “more powerful tests incidentally cover the
smaller behavior”: incidental execution is not sufficient; the stronger oracle
must actually make the smaller defect observable.

## Deterministic build, Btrfs, and concurrency

Use a content-addressed cross-run cache on the NVMe scratch volume, e.g.
`/realm/tmp/polylogue-fixtures/<fixture-digest>/`, rather than committing DBs or
rebuilding per pytest session.

Build protocol:

1. acquire a per-digest lock;
2. build under a temporary directory;
3. close writers and checkpoint/truncate WAL;
4. run SQLite `quick_check` plus manifest invariant census;
5. fsync/publish the manifest last and atomically rename the directory;
6. validate the digest and manifest on every reuse.

Do not use an unvalidated `.build.done` sentinel. A killed builder must leave a
non-publishable temp directory, never a cache hit.

Concurrency model:

- immutable/read-only tests share the checkpointed base fixture;
- mutating tests receive a private per-worker/test clone;
- never share writable WAL/SHM files between workers;
- serialize the heavyweight fixture build and large workload census;
- avoid multiple heavy readers against the same archive at once.

On Btrfs, a reflink clone is attractive because unchanged extents are shared,
but source and target must have compatible NOCOW/checksum status. The current
pytest scratch root is marked NOCOW; NOCOW also disables data checksums and
compression for new files. Benchmark reflink versus ordinary copy for the
actual write-heavy clone pattern rather than assuming reflink always wins.
Reflink only a closed/checkpointed archive with no live WAL dependency. Use
SQLite's online backup API (or another SQLite-consistent snapshot mechanism)
when the source is live.

## Gate placement

- PR/default affected gate: micro corpus, selected distribution slices,
  cross-surface semantic laws, mutations for changed high-risk capabilities.
- Lab/pre-merge for query/harness changes: medium/tail corpus, growth laws,
  resource receipts.
- Nightly/post-merge: full provider/version matrix, larger quantiles,
  adversarial scenario coverage, bounded performance census.
- Periodic schema-refresh lane: regenerate candidate schema packages from
  privacy-safe observations, compare distribution/construct drift, and require
  explicit promotion. The test fixture cache naturally invalidates when the
  promoted package hashes change.

Testmon can choose which laws are affected, but affected selection is not the
proof model. Harness/schema/generator changes must invalidate the relevant
fixture digest and select the portfolio that depends on it.

## Existing durable owners

No new tracking item is needed for the core concept:

- `polylogue-yeq.3` already owns query laws, cross-surface parity, p50/p95/max
  and pathological fixtures, workload receipts, and anti-vacuity mutations.
- `polylogue-88jp` owns the verification-risk record based on production-route
  dependencies, mutation evidence, runtime, selection, flake, and escape risk.
- `polylogue-ixqt` owns dominance/equivalence-based test consolidation for the
  mechanical surface suite.

Their implementation should explicitly reuse the schema-regeneration and
`SyntheticCorpus` substrate described here, rather than introducing a separate
live-distribution fixture system. A separate follow-up is justified only if no
current schema owner covers learned joint/lifecycle annotations and archive-
level corpus profiles.

## Recommended implementation sequence

1. Add a generator conformance census that proves each promoted schema/profile
   reproduces declared marginals, quantiles, relations, and construct coverage.
   Include paired/late/missing/duplicate tool outcomes and nested exec envelopes.
2. Make a content-addressed real-pipeline fixture builder with validated
   manifests and safe immutable/private-clone concurrency.
3. Expand the existing cross-surface agreement harness into query laws and add
   direct-cheapest-correct oracles plus work receipts.
4. Encode the dogfood anchors as named workload shapes, not one-off regression
   examples.
5. Run mutation controls.
6. Apply the dominance rule and delete/collapse the scale smokes and redundant
   mock/surface tests they subsume.
7. Use the risk model to choose the next composition seams, rather than growing
   raw test count.

## Outcome

The correct goal is not a bigger test suite. It is a smaller number of tests
with a larger semantic radius and explicit falsification power, backed by
schema-derived production-like workloads. Scale becomes one dimension of every
important law, not an isolated nightly fixture whose only assertion is that it
returned at most the requested limit. The same laws should leave private names,
call graphs, batching, caching, and module placement free to change unless one
of those structures is an explicit public, durable, security, or architecture
contract.

## Expressiveness and semantic compression

### Current source shape

A read-only AST/LOC survey, refreshed against the worktree on 2026-07-16,
found approximately:

- 275,647 nonblank Python LOC under `tests/`;
- 251,462 of those LOC under `tests/unit/`;
- 10,874 `test_*` functions;
- 210 `@given` test functions across the tree, not only the 61 test functions
  in the dedicated `tests/property/` package;
- 54 Hypothesis-importing files spanning 18,608 nonblank LOC, including 15,348
  LOC of test modules and 2,276 LOC of shared strategies;
- two substantive Hypothesis state machines (write/lineage and basic archive
  save/delete lifecycle), despite many other temporal protocols;
- 713 parametrized test functions, with at least 2,117 statically visible
  literal case rows;
- 3,320 filesystem-backed and 990 async test functions, plus explicit
  concurrency and subprocess suites (overlapping counts, not quality
  verdicts).

Test and product Python are now comparable in nonblank LOC (about 274k versus
257k). Exact duplicate AST bodies are rare. Narration found concentrated
semantic repetition in query lowering, small surface helpers, status
projections, discovery catalogs, and mocked forwarding, but it also found
large stateful, durable, and real-route clusters. `test_query_expression.py`
contains 328 mixed-strength tests; `test_facade_contracts.py` contains 104
mostly real-archive async tests plus smaller discovery/signature/empty-shape
families. These are cluster-audit leads, not evidence that the whole suite is
example-heavy or safely compressible.

### Oracle taxonomy

The suite needs to label what kind of evidence makes an assertion true:

1. **Raw/generated-fact oracle**: expected normalized facts are emitted by the
   source generator before the production parser runs.
2. **Reference-model oracle**: a deliberately small in-memory evaluator or
   lifecycle model computes expected membership/state without sharing the
   production SQL/lowerer.
3. **Metamorphic oracle**: a transformation declares equality, monotonicity,
   or another relation without needing an absolute golden answer.
4. **Differential oracle**: independent implementations/routes produce the
   same logical state (full vs incremental rebuild, direct primitive vs rich
   query, backup vs restored archive).
5. **Parity oracle**: public surfaces agree semantically.
6. **Representation oracle**: a golden/snapshot fixes an intentionally public
   byte/text layout.
7. **Fault oracle**: after controlled failure, named durability/recovery
   invariants hold.
8. **Resource oracle**: bounded work, cancellation, progress, and cleanup meet
   an envelope.

These are not interchangeable. Parity cannot prove truth when every surface
shares the same bad lowerer. A snapshot cannot prove the right population was
selected. “Does not crash” cannot prove preservation. Every high-authority
capability should have at least one truth-bearing oracle (1–4), with parity and
representation added where relevant.

### Reusable executable laws

Use ordinary pytest modules, Hypothesis strategies, and small typed helpers.
Polylogue does not need a new testing framework, obligation language, or
universal case registry.

A helper is justified only when its caller executes a production route and its
assertion can be falsified by a named production mutation. The useful reusable
laws remain:

- selection/identity/cardinality;
- order/pagination/continuation;
- null/unknown/error/freshness preservation;
- round-trip/losslessness;
- idempotence/commutativity/monotonicity;
- list-to-detail/ref closure;
- authorization/dry-run/confirmation/effect isolation;
- rebuild/incremental equivalence;
- cancellation/progress/cleanup;
- work-growth bounds.

They should appear as a handful of normal test runners near the subsystem that
owns them, not as a central compiler. Parametrized node IDs should retain the
law, surface, scenario, and seed so testmon and failure diagnosis stay granular.
Do not combine the suite into one giant generated test selected for every
change.

### Independent oracles and capability coverage

Current test helpers sometimes mirror product declarations manually:

- `ArchiveQueryCase` repeats a selected subset of query fields;
- its surface adapters manually translate those fields into SQL, query specs,
  CLI flags, HTTP parameters, and MCP arguments;
- filter strategies hard-code a vocabulary including `provider` and a provider
  token list while public vocabulary is moving to `origin`;
- `EXPECTED_TOOL_NAMES` duplicates the registered MCP tool inventory.

Mirrors detect some additions by requiring two edits, but they can also keep an
obsolete concept mutually green or omit the same field from every adapter.

Reusable runners should separate:

- **subject discovery**, derived automatically from production registries,
  grammar/IR types, OpenAPI, the rewritten MCP protocol, schemas, and runtime
  declarations;
- **executable law assignment**, where every discovered selection predicate is
  dispatched into membership, structured/DSL equivalence, count, pagination,
  and surface-parity tests; collection fails if no runner owns it;
- **expected truth**, built from generator manifests, raw evidence, simple
  reference models, or transformations—not from production execution;
- **exemptions**, explicit, justified, expiring, and visible when a newly
  discovered capability has no law class.

This removes long copied name lists without letting production declare itself
correct.

### Interaction coverage rather than point coverage

Most escaped bugs require combinations:

```
provider/version × data shape × query unit/grain × predicate × projection
× surface × lifecycle/freshness state × pagination × failure × scale
```

Neither one example per field nor the full Cartesian product is sensible. The
first implementation should use tools already present:

- Hypothesis for constrained values, shrinking, and transition sequences;
- explicit `@example` witnesses for historical incidents and architecture
  hazards;
- small `pytest.parametrize` decision tables for finite public enums and
  surface/action combinations;
- quantile/pathological schema profiles for scale dimensions.

Do not add a covering-array dependency or build a pairwise generator until an
actual matrix is large enough that explicit high-risk tuples plus Hypothesis
are demonstrably insufficient. The important forced combinations are already
known: selection × aggregate × surface, raw construct × parser × normalized
fact × projection, and lifecycle state × retry/restart × public freshness.

### Model-based temporal and concurrency testing

Many Polylogue contracts are protocols, not pure functions:

- acquire/revision/cursor authority;
- materialization/convergence debt and retries;
- generation promotion/rollback;
- assertion create/supersede/suppress/delete;
- query paging/cancellation/resume;
- daemon worker handoff and restart;
- backup/restore and blob GC leases;
- lineage resolution as parents arrive, change, or disappear.

The two existing state machines demonstrate the right pattern: a small model,
production commands, and invariants after every transition. Expand this style
risk-first. Include invalid reorderings, duplicates, partial failure, restart,
and retry.

For concurrency, random xdist and sleeps are weak. Add deterministic barriers
or scheduler/fault points at transaction and handoff seams, then explore
selected interleavings against a linearizable/serializable reference model.
Use virtual clocks. A few explicit concurrency models can replace dozens of
timing-sensitive example tests.

Do not introduce a formal-methods tool initially. Expand the existing
`RuleBasedStateMachine` tests first. A separate formal model is warranted only
if a bounded Python model cannot express or explore a demonstrated high-risk
protocol.

### Domain-semantic mutation and historical escapes

Conventional mutation testing is useful but dominated by local expression
changes. The important escaped failures require semantic defect operators:

- drop an identity field while adapting a surface;
- place a selective predicate outside a global join/window;
- convert unknown/unavailable to zero/false;
- apply `LIMIT` after global materialization;
- broaden exact-session scope to origin/archive scope;
- lose a provider envelope or nested tool operation;
- report success before convergence/durable commit;
- reuse a stale continuation or authority head;
- ignore cancellation or retain work after client disconnect.

Maintain an **escape corpus** of minimized historical defect patches/operators,
including dogfood incidents. A candidate suite revision must kill these defects,
not merely preserve line coverage. This is more realistic than asking generic
mutmut alone whether the suite is strong.

For each rewritten cluster, compare the old and replacement tests against the
same focused mutants and historical witnesses. A per-test kill matrix may
eventually help larger deletion decisions, but it is not a prerequisite for
the first replacements.

### Proof by construction

Some test volume is compensation for production duplication. The strongest
way to reduce it is to eliminate representable drift:

- use one typed query IR through every surface rather than separately mapping
  field arguments;
- generate CLI/MCP/HTTP adapter schemas from shared typed contracts where the
  public contracts genuinely coincide;
- use exhaustive enum/match checking and closed discriminated unions;
- enforce storage facts with STRICT tables, CHECKs, FKs, and transactional
  invariants;
- generate sync/async variants or converge them behind one semantic core;
- remove dead/unwired surfaces instead of raising their coverage.

Then test the compiler/constraint once plus a small set of black-box routes,
rather than asserting every copied adapter spelling. Illegal states made
unrepresentable need fewer regression examples.

Generated code does not remove the need for an external semantic oracle; it
removes a whole class of adapter drift and frees tests to focus on behavior.

### Continuous real-archive discovery

No finite synthetic corpus captures every provider drift or real correlation.
Add privacy-local, read-only continuous checks:

- raw → normalized negative-space/construct coverage;
- contradiction invariants across materialized views;
- sampled reference-query versus optimized-query shadow comparison;
- schema/profile distribution drift;
- live query/resource receipt outliers;
- derived freshness/authority reconciliation.

These are not part of the ordinary test suite and are not substitutes for
pre-merge tests. They are local discovery campaigns that feed minimized,
privacy-safe witnesses or schema/generator improvements back into ordinary
tests. Keeping that boundary explicit prevents a live-data dashboard from being
mistaken for executable regression coverage.

### Change and compatibility differentials

The suite should treat time/version as a dimension:

- old raw corpus under new parser/materializer;
- previous promoted schema package versus candidate package;
- previous release reader against current durable tiers where supported;
- full rebuild versus incremental state after the same event history;
- backup/restore before and after schema evolution;
- old continuation/cursor/public payload decoding under declared compatibility.

Golden files capture a few representations; version differentials capture the
semantic population and durability promises across change.

### Harness integrity and failure receipts

A sophisticated generator/harness becomes software that can be wrong. Keep it
small, typed, and independently tested. Every powerful harness needs seeded
negative controls demonstrating that it notices:

- an omitted product capability;
- a wrong oracle population;
- a dropped adapter field;
- an unexecuted production dependency;
- a resource collector omitting child work;
- a fixture cache with mismatched schema/generator digest.

Generated failures must emit a replayable minimal witness: seed, shrunk input,
schema/workload digest, canonical plan, surface invocation, oracle/actual diff,
production trace, and resource receipt. Expressiveness without diagnosis will
cause engineers to add local example tests again.

### Gate semantics

A strong test that runs only after merge is an incident detector, not a merge
gate. An observed tmpfs run reportedly reduces the broad suite to under two
minutes, making broad pull-request execution practical while the suite is made
more expressive.

The exact CI shape still needs a retained receipt because coverage
instrumentation, Python-version matrices, xdist, and GitHub runner tmpfs size
can differ from the local run. The decision should be based on elapsed time,
peak RSS/PSS, tmpfs high-water, worker count, and selected-test count—not on the
old comment in `.github/workflows/ci.yml`.

Gate placement should follow proof value:

- PR: the full non-integration suite on tmpfs when the host receipt proves it
  fits, plus type/static constraints; retain focused affected selection as the
  local inner loop rather than as the only merge evidence;
- pre-merge/lab: medium interaction array, state-machine sequences, scale/work
  growth for query/storage changes;
- post-merge/nightly: broad provider/version matrix, long fuzz/mutation,
  concurrency schedules, tail scale;
- local continuous archive: privacy-sensitive contradiction and drift checks.

Testmon is an execution selector, not evidence that the selected basis is
complete. Harness, schema, grammar, registry, and generator changes need
explicit dependency edges to every compiled obligation they affect.

## A practical shrinking procedure

Use an auditable manual procedure before adding any central evidence database
or optimizer:

1. Remove tests that reach no production behavior (unless they validate the
   harness itself).
2. Remove representation/string tests for non-public implementation details.
3. Collapse same-law examples into generators/decision tables.
4. Compute strict dominance using the earlier five-part rule.
5. For one candidate cluster, run old-only and replacement-only tests against
   the same historical defect patch or semantic mutation. Delete only when the
   replacement catches it and reaches the same relevant production route.
6. Retain a small diagnostic basis where a broader test detects but cannot
   localize a high-risk fault cheaply.
7. Replay the historical escape corpus and compare mutation score, dependency
   reach, interaction coverage, runtime, and flake behavior before deleting the
   old tests.

Do not add a set-cover solver until the evidence exists at per-test granularity
and manual dominance decisions have become a measured bottleneck. LOC is not
an optimization constraint by itself, but this process should remove
large amounts of source because thousands of manually encoded cases become data
rows or generated combinations. Executed case count may rise for high-value
laws while total runtime falls through focused interaction cases, fixture
reuse, and deletion of low-yield examples.

## Migration rule for future bugs

A fixed defect should normally add exactly one of:

- a new general law;
- a missing generator/scenario dimension;
- a semantic mutation operator;
- a minimized witness consumed by an existing generic runner.

It should not automatically add another bespoke test function. Once a witness
is subsumed by a generator/law and the relevant mutation remains killed, it can
remain as compact historical data or be deleted according to portfolio policy.

## Best first vertical slice

Use query selection/execution as the proving ground because it has:

- large manually enumerated test files;
- an existing query grammar/IR and cross-surface harness;
- known dogfood escapes;
- cheap direct/reference oracles on small archives;
- scale-sensitive semantic mutants;
- multiple public surfaces.

Build a discriminating archive with a plain-fact manifest, a small independent
evaluator over those facts, real-surface selection/paging/count/bounded-work
laws, and focused historical defect mutations. Compare it with the current
query tests; delete only the strictly dominated cluster. This demonstrates
whether dramatic LOC reduction is real before attempting a suite-wide rewrite.

## Tmpfs and gate economics

### Broad pull-request gate

Treat the reported under-two-minute tmpfs run as the current design input. It
does **not** solve the escaped-defect problem, but it removes a false constraint:
the suite no longer has to be semantically compressed merely to make a broad PR
gate possible.

With the `b054.1.1.3`–`.5` outcomes treated as realized, the Diet gate work is
to consume and compare their receipts, not repeat their seed/mutation/hang
campaign. Any remaining full-gate experiment should:

1. run the full non-integration suite on tmpfs on the primary Python version;
2. retain a receipt containing selected count, elapsed time, coverage mode,
   worker count, peak process-tree RSS/PSS, tmpfs high-water, and cleanup state;
3. repeat with the actual CI coverage command, not only bare pytest;
4. decide whether all three Python versions run the full suite or whether the
   secondary versions run a smaller compatibility lane;
5. delete the stale 45–50 minute CI rationale once the replacement receipt is
   reproducible on the runner that will enforce it.

Testmon remains valuable for the local edit/test loop. It should not be the only
pre-merge evidence if the full suite is now cheap.

### Tmpfs should be the ordinary substrate, not a semantic assumption

Use tmpfs for ordinary unit/property/contract tests and for reusable generated
archives that fit the measured memory envelope. Keep explicit disk-backed lanes
for behavior whose subject is the filesystem or durable scale itself:

- SQLite locking/WAL/fsync and crash/reopen behavior;
- backup/restore, blob GC, permissions, atomic replacement, and free-space
  handling;
- truly huge tail/stretch datasets whose resident footprint threatens the
  runner;
- Btrfs-specific CoW/NOCOW/reflink behavior.

A tmpfs pass is not evidence about durable-write behavior. Conversely, a
correctness test that only needs SQLite semantics should not pay SATA/NVMe
latency by default.

For concurrency:

- share a large fixture only through read-only/immutable connections;
- give every mutating test or xdist worker its own archive root;
- use micro fixtures for state-machine writes rather than cloning a multi-GB
  archive per example;
- serialize tests that mutate process-global MCP/runtime service state;
- keep semantic concurrency tests deterministic with barriers/fault points;
  xdist parallel execution is not a concurrency oracle.

For disk exceptions, benchmark the two existing reasonable shapes instead of
declaring one universally correct: a NOCOW ordinary copy for write-heavy SQLite
versus a reflinked immutable template for read-mostly tests. Do not mix Btrfs
policy into the default tmpfs path.

## Reuse the wheels already installed

| Existing substrate | Use it for | Do not build beside it |
| --- | --- | --- |
| pytest + parametrization | finite public enum/action/surface decision tables | a new case-execution framework |
| Hypothesis | query values/ASTs, correlated records, shrinking, state sequences, historical `@example`s | a home-grown property generator or shrinker |
| `hypothesis-jsonschema` + `SyntheticCorpus` | provider wire shape and schema-derived values | a second “production-like” wire generator |
| existing write-path `RuleBasedStateMachine` tests | lineage/write protocol expansion | a parallel lifecycle DSL |
| mutmut + `devtools/mutation_scenario_catalog.py` | ordinary mutants and focused campaign ownership | a second conventional mutation engine |
| Atheris targets | invalid bytes, decoder/parser crashes, pathological syntax | using fuzzing as a semantic membership oracle |
| pytest-testmon + `b054.1.1.4` mutation receipt | fast affected selection and real production-dependency evidence | treating selection as proof of suite completeness or wrapping it in another mocked proof |
| coverage.py | executable-line/branch reach and dead-code evidence | interpreting coverage percentage as behavioral closure |
| pytest-benchmark + SQLite progress handlers | latency distributions and bounded-work receipts | wall-clock-only assertions for relational work |
| pytest-randomly | ambient order/global-state leakage | pretending random order explores transaction interleavings |
| planned web-reader rewrite + existing Playwright lane | DOM behavior, security, interaction, accessibility | improving the current inline-reader source-spelling suite or adding another browser harness |
| planned Schemathesis work | OpenAPI request/response fuzzing | a custom HTTP schema fuzzer |

One conditional addition remains plausible: a pairwise/covering-array tool if a
real finite matrix becomes too large to review explicitly. It is not justified
yet. Likewise, symbolic execution, a SAT/set-cover optimizer, or a formal-model
tool should not enter the first phase.

## Actual suite audit: where volume is being spent

The current Python test tree contains about **276k nonblank lines** and 10,874
source test functions. That census is only a denominator, not an argument that
any given test is weak. The useful question is what kind of falsification each
cluster can perform.

The largest immediately suspicious concentration is not ordinary product tests
but verification of the repository's own declarations. `tests/unit/devtools/`
alone is about 21k nonblank lines. Much of it is legitimate harness behavior,
but several clusters form a closed loop:

1. a catalog or YAML file declares that a path, command, campaign, or test owns
   something;
2. a verifier checks that the declaration is well formed or that the named
   path/string exists;
3. a test recreates the declaration and asserts that the verifier accepts it;
4. generated documentation reports the declaration as coverage or readiness.

No step in that loop has to execute the claimed behavior. This is precisely the
kind of volume that can coexist with the dogfood failures.

| Current mechanism | What it actually proves | Decision |
| --- | --- | --- |
| `test_surface_storage_boundary.py` | production AST imports respect a real layering invariant | **Keep.** It is static, but the invariant is architectural and independently checked. Node count can be compressed without changing its meaning. |
| closure matrix + `test-coverage-domains.yaml` | target/test paths exist and prose fields are populated | **Delete.** A path-to-test assertion is not behavioral ownership. |
| scenario coverage/projection + quality registry + artifact-graph coverage sections | authored target lists agree with other authored target lists | **Delete the coverage claims and their exact-list tests.** If a small navigation index remains useful, call it a declared target index, not evidence. |
| validation-lane catalog tests | mostly exact command arrays, labels, and dry-run strings | **Retain generic executor laws; delete per-lane spelling pins.** Prove that every live lane resolves, expands without cycles/duplicates, invokes collectable commands, and propagates failures. |
| generated-surface tests | some renderer/cache invariants plus a large owner-path inventory | **Keep the generic renderer/cache laws; delete owner subset pins unless a changed dependency is shown to invalidate the cache in a behavior test.** |
| code-ref resolver | ordinary symbols resolve; nested MCP functions fall back to regex source spelling | **Keep real symbol resolution only while catalogs consume it. Delete regex existence as proof and delete the resolver with catalogs that disappear.** |
| public-claims and release-readiness gates | headings, required fields, evidence-path existence, command/CI substrings, retired issue spellings | **Delete.** A command written in prose was not run, and an evidence path does not bind content, commit, result, or freshness. |
| affordance-usage report | real action aggregation, then arbitrary `kill`/`keep`/`promote` thresholds over a test-authored surface list | **Keep raw observed usage. Remove automatic product judgments and source the surface inventory from production if a report remains.** |
| test-economics report | useful coverage/git/testmon measurements, then median-split quadrant labels | **Keep or reuse the raw collectors; remove the quadrant classifier, its tests, and the stale generated verdict document.** The current report itself records counterexamples to its labels. |
| manifest verifier | Pydantic shape, path existence, mirrored catalog lists, claimed command substrings, and some real workflow checks | **Split by observation.** Keep checks that compare a live public inventory or artifact receipt to reality; delete hand-maintained coverage/readiness mirrors and tests that only validate their schema. |

The manifest split matters. `docs-coverage-baseline.yaml` is at least consumed
as a ratchet against a live surface inventory. Mutation run artifacts and their
freshness/kill-rate receipts are also real observations. In contrast:

- `test-coverage-domains.yaml` maps modules to tests by assertion;
- `test-quality-coverage.yaml` still claims 3,963 tests and an April baseline
  while the source census is now 10,874 functions;
- `scenario-coverage.yaml` describes its own scenario abstractions;
- `oracle-quality.yaml` and `evidence-freshness.yaml` label themselves
  transitional taxonomies rather than sources of authority;
- `distribution-coverage.yaml`, `docs-media-coverage.yaml`, and much of
  `security-privacy-coverage.yaml` duplicate facts better obtained from real
  build/test execution or serve as prose documentation.

The last group may remain as ordinary documentation where useful, but a shape
validator must not turn it into verification. Security controls in particular
need adversarial behavioral tests; a row naming a control and a test path does
not strengthen them.

### Two different “dev loop” systems

Two similarly named systems have different ownership and lifecycle status.

`devtools/dev_loop.py` (3,831 lines) is live branch-local development tooling,
not the retired task conductor. It is registered as `devtools workspace
dev-loop`, documented as the daemon/web/browser-capture preflight and proof
ladder, and implements process-tree cleanup, isolated ports, daemon startup,
receiver polling, browser-extension smokes, copied-profile proofs, artifact
inspection, and TUI/browser plans. It contains no conductor/frontier/wait-ahead
logic. `docs/dev-loop.md`, the browser-extension smoke scripts, daemon
`/api/dev-loop` metadata, web-shell development chip, visual tape, and
`test_dev_loop.py` belong to that live tool. Keep and audit them on their actual
behavior; do not delete or migrate them because of the name.

`devtools/devloop_temporal.py` (255 lines) is the retired task-conductor
adapter. Its defaults are `.agent/conductor-devloop/OPERATING-LOG.md` and
`EVENTS.jsonl`, and its 142-line test explicitly pins those paths and formats.
That module, test, `workspace temporal-devloop` command entry, and generated
documentation are a definite deletion cluster.

`.agent/tools/conductor_compact.py` also targets the retired conductor log and
has no live source/test/docs references found outside itself, so it is a
separate cleanup candidate rather than part of the branch-local dev-loop count.
Likewise, frontier/conductor/wait-ahead terminology must be audited by its own
consumers; it cannot be condemned from the `dev-loop` name alone.

Two test tails remain independently weak:

- `test_claim_guard.py` ports classification logic out of the archived shell
  script and compares production to that replica. The product readiness
  invariant may deserve a direct truth-table test, but the conductor parity
  oracle should go.
- `test_envelope.py` asserts that the spelling `conductor-devloop` is absent.
  That is a refactoring fossil and should go.

### Evidence-backed volume estimate

The current audit supports three different claims, which should not be blended:

1. **High-confidence deletion/rewrite retirement: about 2.5–4.5k nonblank test
   lines.** This includes the 118-nonblank-line temporal-conductor tests,
   closed-loop coverage and readiness tests, most exact validation-lane pins,
   the six vacuous scale smokes, direct implementation fossils, and UI
   implementation tests retired with the web-reader rewrite. Most are cheap,
   so this primarily reduces change amplification, node count, and misleading
   confidence. Deleting the temporal adapter has negligible recorded runtime
   effect; branch-local dev-loop test runtime is outside this deletion cluster.
2. **Sampled composition-cluster consolidation: another about 5–8k lines if
   dominance is demonstrated.** Query tests are about 8.7k nonblank lines, the
   three status clusters about 4.9k, and facade contracts about 4.9k. Replacing
   repeated forwarding, empty-shape, and one-field cases with a few independent
   real-route laws plausibly removes 25–45% inside those clusters while
   retaining unique parser, durability, protocol, and diagnostic branches.
   MCP is excluded from this estimate because its planned rewrite is a clean
   test-design boundary; the old suite should not be incrementally consolidated
   or ported test by test, and the rewrite's net test volume is not yet known.
3. **No suite-wide percentage beyond that is presently credible.** The audited
   near-term total is roughly 8–13k lines, or 3–5% of the current suite—not a
   claimed 20–30% collapse. Larger reductions may emerge from storage/source
   audits, but should be earned cluster by cluster through mutation and route
   dominance evidence.

The production-code estimate must be smaller and more carefully partitioned.
The confirmed dead temporal adapter is 255 lines. Five live verification files
span 1,842 lines—`verify_closure_matrix.py` (180), `scenario_coverage.py`
(151), `render_quality_reference.py` (492), `verify_manifests.py` (794), and
`verify_docs_coverage.py` (225)—but 1,842 is an audit surface, not a deletion
total. A function-level trace gives this disposition:

- `verify_closure_matrix.py` is entirely a check that authored domain, target,
  representative-test, gate, and known-gap rows have acceptable shapes and
  existing paths. It never executes or observes the representative behavior.
- `scenario_coverage.py` joins authored scenario target declarations to the
  authored artifact/operation graph. Its production consumers add that map to
  `devtools lab graph` and the generated quality reference, and `--strict`
  fails when a graph item lacks a declaration. This is useful navigation at
  most; calling completeness behavioral coverage is the closed loop.
- `render_quality_reference.py` is mostly ordinary generated navigation over
  executable lane, mutation, and benchmark registries. Keep that renderer and
  its generic rendering laws. Remove the runtime-coverage scorecard, the claim
  that scenario projections establish coverage, and closure-matrix guidance;
  do not count the whole 492-line file as deletion.
- `verify_manifests.py` mixes several species. Coverage-gap schemas,
  path-existence checks, implemented/test-location consistency,
  `test-coverage-domains` checks, campaign catalog mirrors, and YAML claims
  matched to workflow command substrings do not establish behavior and should
  disappear with their mirrors. Mutation/benchmark artifact existence,
  non-emptiness, and freshness are real observations; preserve that receipt
  check in the campaign machinery rather than requiring a duplicate YAML row.
  Pydantic validation remains useful only for manifests retained for an
  independently useful purpose.
- `verify_docs_coverage.py` derives CLI, MCP, config, and stable-route
  inventories from production, compares them with the documentation corpus,
  and allows only explicitly baselined debt. It is a real public-surface
  documentation ratchet and remains.

This supports a moderate-confidence production shrink lead of roughly 1–2k
lines including the 255 confirmed dead lines, not a 1,842-line adjudicated
deletion. The exact count should be
made from the surviving function-level design, including the affected command
entries, manifest models, graph output, generated documentation, and tests—not
by summing these five files.

### Representative narration of the large product clusters

The line counts alone would be a poor basis for deletion. Reading the tests
shows mixed-strength files rather than uniformly weak suites:

- `TestLowererFieldMapping` in `test_query_expression.py` contains roughly 50
  one-input/one-field examples such as `repo:polylogue ->
  spec.repo_names`. These are useful diagnostics but can be one reviewed
  decision table. The same file also executes boolean, lineage, sequence, and
  terminal queries against real archives; those are substantially stronger.
  Four tests additionally assert that dropped projection tables remain absent,
  which is an implementation-history fossil rather than query behavior.
- `test_query_exec_laws.py` has 66 source tests, about 43 of which patch or mock
  collaborators. Many construct a `MagicMock` environment, replace the archive
  executor, call the CLI dispatcher, and assert the forwarded arguments. This
  can protect a routing fork, but it cannot catch a missing predicate or
  unbounded relational plan inside the executor. A compact routing decision
  table plus real-route selection laws should dominate much of it.
- The current MCP files are similarly mixed, but they are not a consolidation
  project because MCP itself is planned for a substantial rewrite. Extract the
  externally meaningful obligations from the few tests that build a real
  archive; do not port their implementation, fixtures, FastMCP registration
  assumptions, source parsing, or mock-forwarding structure. The new MCP suite
  starts from the rewritten public protocol and production route.
- `test_facade_contracts.py` has a strong lower half proving which split tier
  owns raw evidence, derived insight rows, and user mutations. Its discovery
  guard only proves that every async method was assigned to a test-authored set,
  and its typed-signature sweep largely repeats strict type checking. The large
  empty-archive family proves shapes but not nonempty semantics. Keep the tier
  authority and corruption/absence tests; compress discovery, signature, and
  repeated empty-shape coverage.
- CLI status, daemon status, and CLI-command status total about 4.9k nonblank
  lines and use mocks/patches in 79 of 202 source tests. Their separate payload
  examples did not prevent the dogfood truth from diverging across surfaces.
  A shared status transition model, one projection implementation, and
  cross-surface laws over the same seeded state should replace repeated local
  snapshots—not merely add a fourth suite.

This is the intended subtraction method: preserve tests with a unique route,
oracle, failure branch, durability boundary, security adversary, or diagnostic
value; replace repeated adapter examples only after a stronger law is shown to
fail under the relevant implementation mutation.

### Implementation-binding and low-value test pathology

Identifier spelling is only the most conspicuous form. The broader pathology is
a test whose oracle is an internal arrangement that the product is free to
change, or an assertion so weak/self-referential that it creates maintenance
work without excluding an important wrong behavior.

A coarse AST survey provides candidate signals, not deletion counts. The first
ad-hoc pass found 172 source/text/AST functions (53 with literal membership),
479 mock-interaction functions covering about 12.3k body lines (303 also
patching), and 416 functions with only coarsely weak assertion shapes. That
query was not saved, so those figures are sampling evidence rather than a
reproducible baseline.

The saved, deliberately broader v1 query at
`census/test_coupling_census.py` records its predicates, per-test findings, and
input hashes. It currently finds 265 source/text/AST functions (81 with literal
membership), 562 mock-interaction functions covering 14,043 body-span lines
(355 also patching), and 503 coarse-assertion-only functions. The methods are
not comparable growth measurements; v1 exists so later audits are regenerable.
Catalog/registry/surface-matrix naming remains only a survey hint: it does not
say whether a registry is a runtime authority or a test-authored mirror.

Many hits are legitimate. Reading a generated output file, asserting a true
external side effect, testing a null result, or enforcing a deliberate import
boundary is not bad. These signals locate review work; they must never become a
new static quality gate or an agent-generated denylist.

#### Recurrent low-value forms

| Form | Why it is weak | Better owner |
| --- | --- | --- |
| source contains `helper_x(` / imports module Y | correct behavior can survive a rename, inline, extraction, or different algorithm; the named helper can also be called while doing the wrong thing | seed discriminating facts and assert result, side effect, or work bound |
| source does not contain old helper/string/name | fossilizes the refactoring diff and guesses at every way the old bug might return | historical input that failed before, plus an independent semantic oracle |
| `hasattr(module, "invented_bad_api") is False` | the absence of one imagined identifier says nothing about unsupported behavior | omit it; if a public operation must be rejected, invoke the public protocol and assert the typed rejection |
| private attribute/class/name assertions | pins caching, decomposition, or implementation language | assert generated distribution, state transition, or public protocol behavior |
| exact mocked collaborator call/arguments | often proves that a wrapper forwards the values the test itself supplied; it cannot see downstream loss, translation, or persistence errors | run through the wrapper into the real in-process substrate; mock only the true external boundary |
| exact internal call order/count | breaks under batching/caching while allowing semantically wrong work | assert transaction/fault outcome, idempotency, emitted external effect, and bounded work |
| every method appears in a test-authored set | adding a row satisfies the test without testing the method | enumerate the production public protocol and execute a generic contract, or delete the closure claim |
| `hasattr`/`isinstance`/annotation checks already guaranteed by types | duplicates mypy/Pydantic/import behavior without using the object | perform the smallest meaningful operation; rely on strict typing for static shape |
| module/file existence and topology pins | preserves the present directory layout | keep only centrally owned layering/dependency rules; delete per-refactor file lists |
| snapshots of incidental private output | converts harmless wording/order/layout changes into approval churn | snapshot only a declared public representation; otherwise assert semantic fields/relations |
| copied reference implementation | two copies can agree and be wrong, and every change must update both | use a deliberately simpler independent model, raw fact manifest, metamorphic law, or differential route |
| permissive smoke assertion | `0 <= hits <= limit`, non-null, or “did not raise” admits the important broken result | plant exact truth and assert membership/completeness or a named invariant |

#### Direct examples already present

- `test_session_profile_stale_predicate_has_exactly_one_definition` explicitly
  fails if correct SQL is implemented inline “even if its NULL-branch semantics
  happened to be correct.” Three neighboring real-database tests already prove
  convergence/repair agreement and idempotence. Delete the source-spelling test.
- `TestSharedCardinalityPath` inspects Click callback source to require
  `verb_cardinality`, `check_cardinality`, and `query_spec()` while forbidding
  three old spellings. The file already has behavioral cardinality regressions;
  those should own the bug.
- `test_archive_session_list_route_uses_bounded_sql_helper` asserts four private
  call spellings. Replace it with exact results plus a SQLite progress/work
  receipt and cancellation test. Calling the named helper is not evidence that
  the query inside it is bounded.
- `test_attachment_identity_query_does_not_extract_native_ids_from_json`
  forbids eight SQL substrings. Poison JSON metadata while keeping typed columns
  correct, then assert identity search follows the typed authority and remains
  within a work envelope. That catches semantically equivalent regressions too.
- `TestSemanticGeneratorActivation` reads `_semantic_gen` and `_role_cycle`.
  It should assert the generated provider role/semantic distribution; the
  generator may be stateless or differently decomposed.
- `test_storage_twins.py` checks files exist, a comment says “10 known,” source
  contains `save_` or `async def`, a class is named `SQLiteArchiveMixin`, and
  code mentions `self.queries`. These assertions are both binding and weak.
  Run the same save/read/fault laws against both backends and delete the file-
  topology/source prose suite.
- `test_model_runtime_does_not_import_classifier` and
  `test_no_cross_context_pooling_function_is_exposed` assert the absence of
  particular attributes. The former is immediately followed by a behavioral
  authority test; the negative attribute assertion adds nothing. The latter
  should disappear unless an actual public request for cross-context pooling is
  rejected by contract.
- `test_email_smtplib_classes_are_present_for_documentation` proves properties
  of Python's standard library, not Polylogue. Delete it.
- the static SQLite connection test forbids the exact text
  `with sqlite3.connect(` even though the same file already captures real
  connections and asserts they close. Keep the runtime resource invariant;
  delete the textual companion.

The 679-line interpolated-SQL guard has a serious security purpose, but its
trust model illustrates a subtler form of implementation binding: variable
names such as `where_clause`, function qualnames, syntax hashes, and occurrence
indices authorize dynamic SQL. A safe name does not establish safe provenance.
Move the invariant into the production boundary: dynamic identifiers/fragments
must come through one small closed construction path, while user values remain
bound parameters. Then retain one architectural scan for bypasses and
adversarial property tests at query inputs. Do not grow the identifier-name and
syntax-fingerprint allowlists.

Mock-interaction tests need the same discrimination. Verifying that an SMTP
client was called once, a transaction rollback happened, a process group was
killed, or a retry did not fabricate another external request can be the
behavior. Verifying hundreds of facade/repository/MCP/CLI forwarders by replacing
their entire substrate and asserting `assert_awaited_once_with` usually pins the
call graph. `test_repository_insight_runtime.py` and
`test_sync_surface_runtime.py` each enumerate long lists of same-named methods
and their exact arguments. Prefer one generic adapter contract if the adapter
is intentionally public, then exercise nontrivial translation/policy through
the real substrate. If a wrapper has no policy beyond forwarding, deleting the
wrapper may be better than testing it exhaustively.

#### What may legitimately bind names or structure

The rule is not “never inspect source” or “never pin a name.” Binding is earned
when the bound thing is itself the contract:

- documented CLI commands/options, HTTP paths, serialized field names, MCP
  names explicitly promised for compatibility, and package exports declared as
  public API;
- durable SQLite tables/columns and migration behavior needed to read existing
  archives;
- security/privacy/resource boundaries where the forbidden dependency or
  effect is the invariant;
- centrally owned architecture rules such as “pipeline cannot import CLI” and
  “surfaces cannot reach storage internals.” These should be few, AST-based,
  and expressed once, not repeated as helper/file/source substrings;
- exact emitted bytes/text only when a consumer or documented format depends on
  them;
- interaction with a true external boundary when occurrence/order is
  semantically meaningful.

Everything else must survive behavior-preserving renames, extraction, inlining,
batching, cache changes, and module moves.

#### Review and deletion test

For each suspect test, answer in order:

1. What independently meaningful production mistake does it make impossible?
2. Is the asserted name/shape documented or consumed outside this repository?
3. Would a behavior-preserving refactor fail it? If yes, why is that coupling
   required?
4. Can the implementation be importantly wrong while the assertion remains
   green?
5. Is the same obligation already owned by types, a stronger behavioral test,
   a central architecture rule, or a rewrite boundary?
6. What production mutation should make it fail?

Delete immediately when there is no independent obligation, or when a stronger
neighbor already owns it. Replace only when the obligation is real but the
oracle is coupled. Keep unchanged when the representation/dependency itself is
the public, durable, security, or architecture contract.

#### Preventing agent-generated recurrence

The repository instructions already prohibit tests that memorialize a renamed
identifier, deleted spelling, moved module, removed list entry, import path, or
other refactoring diff. The missing ingredient is not another policy file or
lint. It is applying that rule during test design and review:

1. State the user-visible behavior, durable invariant, security boundary, or
   architecture rule before writing the test. “The implementation uses X” is
   not an obligation.
2. Name the production mutation that must fail. Renaming a helper is not an
   acceptable anti-vacuity mutation unless the helper name is public API.
3. Mock only beyond the boundary being tested. A surface test that replaces the
   entire product operation needs either a separate real-route law or deletion.
4. A bug-fix test starts from the failing input/state and expected result, not
   from the textual diff that repaired it.
5. A refactor normally adds no test. If existing coverage was genuinely
   missing, the new test must remain valuable had the refactor been implemented
   differently.
6. A stronger composition law carries a deletion list. Do not leave dominated
   helper, forwarding, source-spelling, and shape tests behind “for extra
   coverage.”
7. Rewrite projects derive tests from the new public contract and historical
   behavioral obligations, not from a one-to-one port of old tests. This applies
   to both MCP and the web reader.

The review artifact can be a short anti-vacuity note in the PR or test-cluster
rewrite receipt. Turning these questions into another permanent registry would
repeat the problem.

The candidate census does not justify deleting 12k mock-interaction lines or
all 172 source-reading tests. The high-confidence direct source/reflection
fossils appear to contribute hundreds of additional lines, while real-route
replacement could dominate several thousand forwarding-test lines. These
overlap heavily with the query/status/facade estimates above and with the MCP
and web rewrite retirements, so they should not be added again to the suite-wide
total.

## Concrete keep / strengthen / replace / delete map

### 1. Query grammar, lowering, execution, and terminal actions

Current relevant surfaces:

- `tests/unit/cli/test_query_expression.py` is about 6.5k lines and mixes lexer,
  AST, lowerer, execution, rejection, registry, CLI, MCP, and parity tests;
- `tests/unit/cli/test_query_exec_laws.py` is another large route suite, much of
  it built around mocked collaborators;
- `tests/infra/query_cases.py` repeats a partial selection model;
- `tests/infra/surfaces.py` manually translates that partial model into SQL,
  `SessionQuerySpec`, CLI flags, MCP arguments, and HTTP parameters;
- `tests/infra/strategies/filters.py` contains a stale `provider` vocabulary and
  hand-copied provider tokens.

The manual translation harness omitted exact session identity and could not
express query units/stages/projections. This is not merely missing one test; it
is the same parallel-adapter pattern as F-003/F-009.

Keep:

- a compact set of grammar examples for whitespace, quoting, escaping,
  precedence, and error messages;
- unique execution tests for vector, lineage, FTS, and terminal-unit plans;
- explicit historical examples for ambiguous refs and unsupported semantics;
- public representation snapshots only where the exact payload is a contract.

Strengthen with ordinary tests, beginning from the realized C-03 workload and
receipt:

1. Independent expected facts attached to the smallest realized named canary,
   containing a hit and miss
   for each selection dimension, an unambiguous native/canonical ID pair, an
   intentionally ambiguous native suffix, tool/failure rows, lineage, dates,
   tags, repo/path, and planted FTS terms.
2. A small independent evaluator over that manifest. It computes expected IDs,
   counts, and partitions without importing the production lowerer or SQL.
3. Hypothesis-generated valid query expressions plus `@example` dogfood
   witnesses.
4. Laws over the real CLI/API/HTTP routes, and the rewritten MCP route when it
   exists: selected IDs match truth; count equals membership; groups/facets partition the selected set; page
   concatenation equals unpaged order; explain preserves the effective base
   selection; dry-run/apply affect exactly the read selection.
5. A work-bound law for exact/selective queries after appending irrelevant
   sessions.

Replace/delete candidates after the new laws demonstrate dominance:

- one-test-per-field lowerer examples that assert only a dataclass attribute;
- repeated mocked route tests that prove argument forwarding but never execute
  the real selector;
- `ArchiveQueryCase` fields and surface adapters that duplicate production
  translation without an independent truth oracle;
- stale hard-coded filter/provider strategies.

Do not delete the whole query suite. Parser diagnostics and unique execution
branches have localization value that a broad property test will not replace.

### 2. Scale tiers and benchmark fixture

`tests/benchmarks/conftest.py` manually constructs normalized storage records
from hard-coded provider/session marginals. It bypasses provider schemas and
parsers and does not create the tool-use/result relations implicated by the
dogfood failures.

The duplication is wider than that one file:

- `tests/infra/scale_fixtures.py` imports the benchmark module's private
  `_seed_realistic_db` rather than the schema corpus path;
- `devtools/large_archive_generator.py` already knows how to use
  `SyntheticCorpus` and inferred corpus scenarios;
- `tests/infra/corpus_fixtures.py` already runs `SyntheticCorpus` wire
  artifacts through `parse_sources_archive`, but few tests use it;
- `devtools/scale_regression_probe.py` uses tiny mocked/global-patched shapes,
  so it is useful for particular regressions but is not a general large-archive
  proof.

There are also three incompatible meanings of `small`/`medium`/`large`: the
scale fixtures use 100/1,000/10,000 sessions, the large-archive generator uses
100/500/2,000 plus a 10,000-session stretch tier, and cached benchmark fixtures
use 1k/5k/10k/50k messages. The `scale-slow` validation lane then selects the
whole `tests/unit/storage/` directory, contrary to the repository's own
affected-selection policy.

Migrate these consumers onto the realized versioned workload-profile authority
from `polylogue-1xc.14.1`. Its profiles are real because they drive provider
artifact generation; Diet must not introduce another registry or coverage
catalog. Tests select the upstream named size/selectivity tier and structural
canary rather than relying on `large` to imply correctness coverage.

All six tests in `tests/benchmarks/test_scale_tiers.py` are direct replacement
candidates. `0 <= hits <= limit` passes when search always returns zero; a
nonempty bounded list proves little beyond fixture existence.

Replace them with schema-generated fixtures and laws whose truth is planted in
the fixture manifest:

- DB cardinalities and relation counts equal the manifest;
- a planted FTS term returns exactly its known sessions;
- exact-ID and high-selectivity queries return exact known memberships;
- count equals the unpaged membership and grouped counts partition it;
- pagination concatenates to the same stable order;
- adding irrelevant sessions does not change the result;
- progress/row-work stays within the declared growth envelope;
- cancellation interrupts and releases work.

Fixture size and law are separate dimensions. Most semantic laws can run on a
small discriminator corpus. The distribution/tail corpus is for plan shape,
work growth, memory, payload bounds, and correlations that require scale.

### 3. API facade contracts

`tests/unit/api/test_facade_contracts.py` is about 5.4k lines. It contains both
valuable tier-authority behavior and many repeated signature/empty-archive/
return-shape checks.

Keep:

- durable-tier routing and authority tests;
- corruption versus absence distinctions;
- bounded-reference and privacy failures;
- full production build-order tests for insight/usage projections.

Consolidate candidates:

- mechanically repeated “typed result on empty archive” cases into a normal
  parametrized runner driven by actual public method signatures;
- repeated list/detail/ref closure checks into semantic families;
- surface copies of behavior already proven through the same facade unless the
  surface adapter itself is under test.

The large tier-routing section should not be deleted merely for being long. It
protects the five-tier durability boundary. Compression is appropriate only
where setup and assertions are genuinely identical and a mutation of the tier
selection is still caught.

### 4. MCP contracts

Do not improve or mechanically consolidate the current MCP tests. The planned
MCP rewrite is the test boundary, just as the web-reader rewrite is.

The old suite contains `EXPECTED_TOOL_NAMES`, runtime discovery/classification
tables, source parsing of registration modules, FastMCP-private manager access,
and many mock-forwarding tests. These describe the current implementation and
should retire with it. Do not create a migration matrix saying which old test
maps to which new test.

Design the new suite from the rewritten public protocol and a small
schema-generated archive. It should prove:

- public discovery and schemas are internally coherent and usable by a real
  client;
- read/admin authorization is enforced at the actual boundary;
- exact query, context, user-state, and maintenance semantics agree with the
  underlying production operations;
- envelopes are bounded, typed, path-redacted, and distinguish missing,
  unavailable, degraded, and empty;
- mutations are idempotent and preserve durable user state;
- cancellation, deadlines, and server restart behavior are truthful.

Generate generic protocol cases from the new protocol's own runtime schema,
then keep compact bespoke semantic tests. Pin exact tool/resource/prompt names
only if the rewrite explicitly declares compatibility with an external client;
otherwise names are allowed to change with the new design. Anti-vacuity means
removing or corrupting the underlying production operation must fail the test;
a mocked method receiving the same arguments is insufficient.

### 5. Web reader and visual behavior

Do not improve the present inline reader's UI tests. The planned rewrite is the
test boundary.

Separate the current file by responsibility first. Most of
`test_web_reader.py` exercises valuable loopback HTTP/API contracts: auth,
privacy, typed empty/degraded states, bounded queries, cancellation, marks,
saved views, and response envelopes. Those substrate tests survive, probably
under a name that no longer implies browser interaction. The rewrite does not
need to re-prove them through a browser.

Retire with the old UI implementation:

- `test_web_shell_reader.py` (410 nonblank lines of CSS/JS/export/source
  spellings);
- the 394-line visual smoke that extracts private functions from the shipped
  source and emulates a DOM by hand;
- static regex/definition-count/dead-implementation assertions in the XSS
  file;
- the JavaScript substring section in `test_web_reader.py`, including private
  function names, assignment snippets, and `AbortController` spelling.

Write rewrite-native tests from public behavior, not as replacements for each
old assertion. Use the existing Playwright project for browser journeys and
the rewrite's ordinary component/unit runner, if it has one. The inherited
behavioral obligations are:

- route state distinguishes loading, empty, no-match, degraded, timed-out, and
  unavailable outcomes;
- a newer navigation cancels or obsoletes the previous request;
- deep links, selection, saved-view create/conflict/delete, retry, and disabled
  actions work from the user's perspective;
- keyboard and accessibility behavior is tested only for interactions the
  rewrite intentionally retains;
- adversarial stored content cannot create script, handler, URL, or attribute
  execution in a real DOM;
- authenticated API calls and daemon-unavailable recovery are exercised
  through the shipped page.

Prefer accessible roles, labels, URL state, visible text, network outcomes,
and stable product-level test IDs. A private function rename or a different
component decomposition must not require a test edit. The Node test that
executes the current escape helper captures a real security obligation, but
the rewrite should prove inert DOM behavior rather than port that helper or its
source scanner.

### 6. Provider/source tests

`tests/unit/sources/test_source_laws.py` is not generally weak: it already has
generated payloads, streaming, encoding, zip, raw-capture, and fault behavior.
Do not rewrite it wholesale.

The missing layer is relational semantic generation. Extend the existing
schema/corpus path to emit and record:

- matched/missing/late/duplicate tool-use/result relations;
- modern nested `functions.exec` operations and paths;
- provider-version-correlated constructs;
- lineage references and replay shapes;
- cumulative/disjoint usage events and explicit absence of monetary evidence.

Keep one curated raw witness per important provider/version construct. Delete
hand-written parser examples only where the schema-generated witness runs
through detection → parse → materialize → query and proves the same semantic
fact.

### 7. Storage, lineage, convergence, and restart protocols

The existing `tests/property/test_write_path_state_machine.py` is a model of
the desired direction: it writes through production code and checks lineage,
physical storage, FTS, and link invariants after transitions. Expand it rather
than creating a new lifecycle abstraction.

Add focused state machines for:

- raw revision/cursor authority, exclusion, retry, and restart;
- convergence debt, bounded partial progress, retry, and freshness projection;
- assertion supersede/suppress/delete lifecycle;
- backup/restore and index generation promotion.

The small `test_repository_state_machine.py` may become redundant once its
save/delete/query laws are present in a stronger real write-path machine. That
is a concrete deletion candidate, but only after comparing production routes
and mutations; its simple diagnostics may still justify retaining a reduced
version.

### 8. Usage authority and freshness truth

These dogfood failures need causal, not surface-by-surface, tests.

For usage, seed one Codex session with cumulative provider events, disjoint
model lanes, zero message-token fields, and no provider-reported USD. Run the
real materialization order. Then assert that profile, cost insight, portfolio,
postmortem, thread, and summary projections agree on token lanes and preserve
the independent price state as unavailable/estimated rather than exact zero.
The oracle is the source-event manifest, not another materialized table.

For freshness, drive source growth → acquire → parse → exclude/retry →
converge through a state model and assert the named-source public projection at
every step. `excluded`, `stale`, `refreshing`, `timed_out`, and `unavailable`
must remain distinct. A tiny fixture is enough for state truth; a distribution
fixture is needed separately to prove the status route remains bounded.

## Example implementation shapes

These sketches intentionally use normal pytest/Hypothesis patterns rather than
proposing a new framework.

### Exact selection survives every terminal action

```python
@pytest.mark.parametrize("ref_form", ["canonical", "native", "unique-prefix"])
def test_exact_selection_is_invariant_across_actions(
    discriminating_archive,
    cli_runner,
    ref_form,
):
    target = discriminating_archive.target
    ref = target.ref(ref_form)

    assert cli_read_ids(cli_runner, f"id:{ref}") == {target.session_id}
    assert cli_count(cli_runner, f"id:{ref}") == 1
    assert cli_facet_total(cli_runner, f"id:{ref}") == 1
    assert cli_delete_preview_ids(cli_runner, f"id:{ref}") == {target.session_id}
```

The fixture manifest owns `target.session_id`; none of the helpers compute
expected membership through `SessionQuerySpec`. Restoring either the dropped
aggregate `session_id` or unresolved residual `startswith` defects must make
this test fail.

### Query membership against an independent evaluator

```python
@given(case=query_expression_cases())
@example(case=historical_exact_id_count_case())
def test_query_membership_matches_manifest(discriminating_archive, case):
    expected = evaluate_manifest(case.predicate, discriminating_archive.facts)
    observed = execute_real_archive_query(discriminating_archive.root, case.expression)

    assert observed.session_ids == expected
    assert observed.total == len(expected)
```

`query_expression_cases` may reuse production enum values for valid spelling,
but `evaluate_manifest` must implement semantics independently over plain facts.
If both expected and observed call the production lowerer, the property is
vacuous.

### Irrelevant data must not change truth or explode work

```python
def test_exact_action_query_is_archive_size_insensitive(schema_archive_factory):
    small = schema_archive_factory(profile="action-discriminator", irrelevant=100)
    large = schema_archive_factory(profile="action-discriminator", irrelevant=100_000)

    small_result, small_work = run_action_query_with_progress_receipt(small)
    large_result, large_work = run_action_query_with_progress_receipt(large)

    assert large_result == small_result == small.manifest.expected_actions
    assert large_work.vm_steps <= small_work.vm_steps * 4 + FIXED_SETUP_ALLOWANCE
```

The multiplier is calibrated from query-plan evidence and retained receipts,
not guessed. This catches an exact predicate moved outside a global window even
when wall-clock time happens to be fast on one machine.

### Provider usage authority survives materialization

```python
def test_provider_usage_is_authoritative_across_public_projections(codex_usage_archive):
    expected = codex_usage_archive.manifest.usage_snapshot
    rebuild_session_insights(codex_usage_archive.root)

    projections = read_all_usage_consumers(codex_usage_archive.root, expected.session_id)
    for projection in projections:
        assert projection.token_lanes == expected.token_lanes
        assert projection.usage_provenance == "origin_reported"
        assert projection.price_state == "unavailable"
        assert projection.cost_usd is None
```

The important activation conditions are zero message-token fields, non-zero
provider-event lanes, and no exact USD. A generic two-message fixture misses
the authority conflict regardless of archive size.

## Deterministic generated fixture build

The generated archive cache should be a build artifact, not committed data.
Extend the upstream workload/profile/build/archive identity rather than minting
a second identity. Its key additionally covers at least:

- source and index schema versions;
- promoted provider-schema package digests;
- generator code/version;
- the shared workload profile and seed identity;
- SQLite/runtime features that affect the materialized representation.

Build into a temporary sibling, validate independent fact counts plus SQLite
`quick_check`, extend the shared `1xc.14`/`b054` receipt last, then atomically
publish the directory. A bare `.build.done` created before validation is
unsafe—the documented SIGKILL failure mode already demonstrates that.

The artifact should contain:

- the split archive files;
- the privacy-safe generation manifest and exact planted truth;
- distribution/relationship summaries;
- schema/profile/seed digest;
- build elapsed time and peak memory;
- validation receipt.

Read-only tests share the artifact. Mutating semantic tests use micro archives;
only a small number of explicit disk/scale tests copy a large artifact. Natural
language realism is unnecessary except for tokenizer, semantic retrieval,
summary, and UI tests; those receive a separate small language corpus.

## Periodic detection of misguided tests

Do not create another quality registry, score, quadrant, or declarative
coverage map. `devtools/test_economics_report.py` is itself an example of the
failure mode: its raw coverage/git/testmon collection is useful, but the
median-split labels turn relative package medians into claims such as
“under-tested substrate” and “over-tested mechanical surface.” Its generated
document records that foundational `paths` is a counterexample and that the
testmon graph was stale. Remove the classifier, classifier tests, and generated
verdict document instead of extending them.

Use the raw sources directly in a disposable or plainly evidential candidate
report: testmon dependencies/durations, pytest receipts, coverage contexts,
mutation artifacts, and AST/source-inspection heuristics. It must not become a
gate or deletion command.

Per-test signals available now or cheaply derivable:

- production files in the testmon dependency fingerprint;
- last duration and selection fan-out;
- whether the test only depends on test helpers/mocks;
- whether its assertions are entirely mock call counts/arguments against a
  replaced in-process substrate;
- branch/line reach when coverage contexts are available;
- source-inspection/private-spelling patterns;
- private reflection, file/module topology, exact internal order/count, and
  snapshots with no declared public-format owner;
- tautological assertion shapes such as `0 <= result <= limit`, unconditional
  `is not None`, or only `hasattr`;
- skip/xfail/flaky history from pytest receipts;
- ownership by an existing mutmut campaign and surviving mutations at that
  subsystem boundary.

Flag, do not condemn:

1. no production dependency;
2. a production module is entirely replaced by mocks before the asserted path;
3. only private source text/spelling is asserted;
4. a behavior-preserving rename, extraction, inlining, batching, or module move
   fails the test without changing a declared public/architecture contract;
5. the oracle permits every meaningful wrong answer;
6. the test is a semantic duplicate with no unique route, witness, mutation,
   or diagnostic value;
7. high runtime/fan-out with no unique behavioral evidence;
8. stale vocabulary or an obsolete public path.

The report itself needs negative controls. It must flag the current vacuous
scale assertions and representative JS source-spelling tests, while not
flagging a real loopback HTTP XSS test or the write-path state machine. It must
also notice a deliberately severed production call in a small fixture test.
If it merely inventories names or rewards declarations, it has failed.

Run the query periodically and before/after a test-cluster rewrite. Human
review decides whether a flagged test is a public compatibility pin, cheap
diagnostic, harness test, or genuine deletion candidate. Persist the actual
receipts for a rewrite, not a permanent self-rating taxonomy.

## First executable phase

1. Remove the retired temporal-conductor adapter, its test, command/catalog
   entry, and generated documentation, plus the archived-parity and
   negative-spelling test tails. Preserve the branch-local daemon/web/browser
   dev-loop and audit its tests normally against the behavior they protect.
2. Delete the closed-loop coverage/readiness mechanisms identified above and
   reduce validation-lane tests to generic execution laws. Preserve real
   architecture, build, docs-ratchet, and mutation-receipt checks.
3. Delete direct implementation fossils already dominated by neighboring
   behavior: exact-one-definition/source-import/helper-call checks, archived
   name absence, private generator attributes, stdlib existence, and storage-
   twin file/comment/mixin spellings. Keep an item only when review identifies
   an independently meaningful architecture or public compatibility contract.
4. Re-enable a measured full-suite PR experiment on tmpfs and retain the
   resource/runtime receipt.
5. Replace the six scale smoke tests with manifest-truth and bounded-work laws;
   route their fixture through `SyntheticCorpus` after adding the necessary
   relational profile.
6. Build the exact-selection query slice: canonical/native/prefix identity,
   count/facets/read/delete-preview, independent manifest oracle, and one
   irrelevant-data work-growth test.
7. Demonstrate anti-vacuity by restoring the known dropped-field/residual-filter
   defects locally and recording that the new slice fails.
8. Compare the new slice with the relevant old query cluster, then delete only
   dominated tests/helpers. Use a one-off evidence report, not a new registry,
   to record production dependency, duration, mutation, and oracle evidence.
9. Repeat subsystem by subsystem: usage authority, freshness/convergence, and
   provider relations. Design the MCP and web-reader public-behavior suites with
   their rewrites; do not spend phases strengthening either current suite.

This phase is intentionally not a suite-wide framework project. Its success
criterion is concrete: stronger historical-defect detection, production-route
reach, and bounded-work evidence with fewer hand-maintained test lines in the
first rewritten cluster.
