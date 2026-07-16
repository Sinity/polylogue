---
created: 2026-07-16
purpose: Define systematic test coverage without authored coverage claims
status: design-ready
project: polylogue
---

# Systematic coverage without coverage theater

Proof-form interpretation is controlled by
[`11-test-proof-form-audit.md`](11-test-proof-form-audit.md). In particular,
`tests/unit/` is not synonymous with isolated example testing: Hypothesis,
stateful, real SQLite, filesystem, HTTP, subprocess, and fault tests live beside
their owning modules. The views below nominate evidence gaps and dominated
clusters; they do not authorize deletion by directory, decorator, or mock
count.

## What “systematic” means

Systematic coverage is not a catalog naming every subsystem. It is a repeatable
way to expose distinct classes of missing evidence. Four generated views are
needed because none subsumes the others:

| View | Question answered | Evidence | What it cannot prove |
| --- | --- | --- | --- |
| execution | Which statements/branches/arcs ran? | coverage.py branch data, changed-line report | the assertion would notice a defect |
| responsibility | Which tests uniquely or redundantly exercise each arc? | per-test dynamic coverage contexts | semantic independence or good oracle |
| sensitivity | Which plausible implementation changes survive? | focused mutation results | unmutated integrations, scale, real protocols |
| semantics/composition | Which externally meaningful laws hold? | independent models, metamorphic laws, real routes, historical witnesses | exhaustive implementation paths |

The views are computed from runs and code. They are diagnostic inputs to an
area audit, not a permanently authored matrix of green boxes.

## Derive the scope from the running system

The way to avoid blind spots without a hand-maintained coverage catalog is to
inventory live surfaces mechanically, then ask what runtime evidence reaches
them:

- Click's actual command/parameter tree;
- generated OpenAPI operations and response schemas;
- registered insight/materializer descriptors;
- canonical SQLite tables/views/triggers and durable migration steps;
- provider detectors/parsers and schema artifacts;
- convergence stages and external process/network boundaries;
- production files/functions/arcs from coverage and code topology.

This inventory establishes *what exists*, not that it is correct. Join it to
per-test execution contexts, mutations, historical fixes, and witnesses. Items
with no evidence become audit leads; items with much execution but no killed
mutations become oracle-strength leads. Because the inventory is regenerated
from executable surfaces, deleting or adding a surface changes it without
editing a parallel truth file.

Do not collapse these dimensions into a quality score. A stable foundational
module can have low fix density and high fan-out without being “over-tested,”
while a security boundary may deserve expensive redundant evidence.

## Execution and responsibility

Keep branch coverage, but stop treating one aggregate `fail_under = 82` as a
useful description of risk. Produce two reports:

- aggregate line/branch trend for broad regressions;
- changed-line coverage for review focus, initially informational.

For periodic area audits, run pytest-cov with per-test contexts. Coverage.py
stores dynamic contexts and pytest-cov can label executed lines with the test
node ID and phase using `--cov-context=test`. Query the resulting database to
derive, for each production arc:

- tests that reach it;
- arcs reached by only one test;
- tests whose executed arc set is almost entirely contained by stronger tests;
- large groups that all execute the same narrow helper while no test crosses
  the real boundary.

This is the missing top-down subtraction aid. It turns “these 60 tests look
duplicative” into an inspectable overlap map. Run it as an audit/campaign mode,
not on every PR: context data and instrumentation add overhead, and arc overlap
still says nothing about oracle strength.

Official references: [coverage.py measurement contexts](https://coverage.readthedocs.io/en/7.12.0/contexts.html)
and [pytest-cov dynamic contexts](https://pytest-cov.readthedocs.io/en/latest/contexts.html).

## Sensitivity through mutation

Use the existing mutmut runner as a cluster acceptance tool:

1. select changed/owned production functions, not a hand-maintained coverage
   campaign claim;
2. use covered-line mutation to avoid spending the campaign on unreachable
   lines;
3. triage survivors into equivalent, invalid, unique missing obligation, or
   weak/redundant oracle;
4. turn legitimate survivors into the worklist for the cluster;
5. archive the raw receipt and summarized survivor IDs with the area packet.

The important result is not a global mutation percentage. It is that the new
composition law kills the historical failure and its neighboring plausible
mutations, after which dominated local interaction tests can be removed.
Mutmut already supports limiting mutations to covered lines via
`mutate_only_covered_lines`; use it in focused campaigns before considering
additional machinery. See the [mutmut documentation](https://mutmut.readthedocs.io/en/latest/).

## Semantic generators

Systematic semantic coverage should come from executable algebras, not case
lists copied from production:

### Selection algebra

For each query predicate/projection/operator, generate small archives with
planted truth and combine predicates through AND/OR/NOT, ranges, grouping, and
projection. Compare all routes to a simple independent set/fact model. Include
explicit examples for historical failures so randomized shrinking does not
replace permanent regressions.

### State transitions

Extend the two existing Hypothesis state-machine suites rather than introduce a
third harness. Model a small logical archive independently and execute commands
such as acquire, ingest, reingest, tag, correct, delete, converge, interrupt,
restart, and repair. Check invariants after every step and terminal equivalence
after quiescence. Hypothesis rule-based state machines are designed around
rules, preconditions, bundles, and invariants; that is enough machinery here.
See the [Hypothesis stateful testing guide](https://hypothesis.readthedocs.io/en/latest/stateful.html).

### Metamorphic laws

Prefer transformations whose expected effect is simpler than the full output:

- reorder independent inputs → identical archive facts;
- reingest identical bytes → no semantic change;
- add irrelevant data → same selected results and bounded extra work;
- restart between stages → same converged result;
- split/bundle equivalent provider artifacts → same normalized sessions;
- rebuild derived tiers → same public read facts;
- apply a user overlay → content hash and reimport decision remain unchanged;
- parent prefix replay → one logical prefix after lineage composition.

These laws cover families of cases with less LOC than enumerated examples.

### Fault boundaries

Inject faults at existing durable boundaries, not at arbitrary private helper
calls: blob committed before row, durable migration after backup, transaction
rollback, process interruption, hot-file append, source disappearance, parser
failure, convergence retry, and daemon restart. Verify externally observable
state and recovery, not exact internal call order.

### Differential paths

Where Polylogue has two legitimate ways to reach the same semantics, compare
them through public facts:

- incremental materialization versus full derived-tier rebuild;
- streaming versus ordinary parser;
- bundled versus split provider artifacts;
- direct in-process facade versus CLI/HTTP transport projection;
- uninterrupted execution versus restart/retry.

This is especially expressive because neither path has to be encoded as the
test's expected implementation. Differences nominate a defect; planted facts
or a minimized witness decide which path is wrong.

### Concurrency

Archive scale also means interleavings, not only row count. Use deterministic
barriers and failpoints at transaction, lease, queue, watcher, and restart
boundaries. Enumerate a small set of meaningful schedules and assert valid
durable/public states; do not try to create coverage through random sleeps.
Promote any discovered schedule to a named regression sequence.

### Shrinking large failures

Every distribution/tail profile needs a way to emit its relational blueprint
and replay a subset. When a large run fails, minimize sessions/relations while
preserving the failure before adding it to the default suite. Keep the original
large seed as a campaign receipt, but make the permanent correctness test the
smallest activation witness unless volume itself is causal.

## API schema testing

Pilot Schemathesis only for stable OpenAPI-backed daemon routes. It can generate
OpenAPI-conformant cases and check status/schema/5xx behavior, and its stateful
mode can traverse declared or inferred operation links. That is valuable for
protocol robustness but does not establish query correctness by itself.

A successful pilot must:

- run against a real daemon over the smallest realized named workload canary;
- cover the stable query endpoints, not the web-reader rewrite surface;
- attach a custom check that compares successful query responses with planted
  fact truth where possible;
- discover defects or replace a meaningful amount of handwritten protocol
  permutation code;
- stay deterministic/reproducible enough for the PR gate.

Reject or demote it if most findings are duplicate schema noise, stateful links
cannot express the real workflow, or the semantic oracle still requires a
parallel hand-written client. See the [Schemathesis documentation](https://schemathesis.readthedocs.io/en/stable/)
and [stateful testing guide](https://schemathesis.readthedocs.io/en/latest/guides/stateful-testing/).

## Historical failures as witnesses

For each dogfood incident, preserve the smallest privacy-safe structural
witness, not necessarily original prose or a full archive. Classify which
activation dimensions mattered:

- relation shape;
- operation sequence/state;
- query combination;
- archive cardinality/selectivity;
- process/restart boundary;
- filesystem/concurrency behavior;
- exact provider wire oddity.

Promote that witness into the relevant generator as an explicit example or
profile feature. The incident list is a prioritization input, not a manually
maintained claim that every incident class is “covered.” A representative
production mutation or replay proves that the witness remains sensitive.

## Deleting tests safely

A stronger test dominates an old test only when all of these are true:

1. it reaches the same externally meaningful responsibility through at least
   as real a route;
2. its oracle would fail for the defect the old test could detect;
3. unique branches/diagnostics in the old test have been enumerated;
4. a representative production mutation makes the stronger test fail;
5. deleting the old test does not remove the only evidence for a distinct
   security, compatibility, error, or process boundary.

Coverage overlap nominates deletions. It never authorizes them. Conversely, a
mock-heavy test may stay if the mock represents a genuine remote boundary and
the asserted protocol is public. The target is fewer obligations proved more
powerfully, not fewer test functions as an end in itself.
