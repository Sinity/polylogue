---
created: 2026-07-16
purpose: Concrete target architecture for a faster, stronger, smaller Polylogue test harness
status: design-ready
project: polylogue
---

# Harness architecture

## Diagnosis

The main harness problem is not a missing framework. Polylogue already has
pytest, Hypothesis, mutation testing, schema-derived provider generation,
real-pipeline fixtures, SQLite instrumentation helpers, coverage, xdist, and a
managed runner. The problem is that these capabilities do not yet compose into
one trustworthy path:

- the shared seeded archive can publish `.build.done` after individual source
  failures were converted to warnings;
- its reuse key is the checkout path, not the inputs that determine the
  archive;
- the scale generator claims `SyntheticCorpus` provenance but inserts
  `SessionRecord`/`MessageRecord` objects directly with a second hand-authored
  distribution;
- the scale tiers mostly prove bounded/nonempty output, not planted truth or a
  work bound;
- cross-surface tests reach real adapters, but translate a small shadow query
  model into each surface independently;
- useful infrastructure has accumulated without consumers: the 124-line
  `RepositoryLifecycleHarness` is unused, while the 136-line generic growth
  budget is tested only by its own 62-line test;
- test selection is treated as part of correctness even though testmon cannot
  observe collection-time-only imports in 95 covered production files.

This calls for consolidation and subtraction, not another registry.

## Realized upstream substrate

This design no longer owns workload inference or workload execution identity.
Assume `polylogue-1xc.14.1` has landed bounded mergeable distribution sketches,
package/archive workload profiles, correlated variants, named scale and
selectivity tiers, structural canaries beginning with C-03, and deterministic
profile identities. Assume `polylogue-1xc.14` and the `b054` proof children
have landed shared resource/cleanup receipts, complete seed identity, real
testmon mutation proof, and repeated isolated/xdist lifecycle evidence.

The harness work below is an adapter and consolidation layer over that
substrate. Any component already supplied upstream is deleted from this plan
during the pre-dispatch reconciliation gate.

## Target shape

```text
versioned workload profiles + correlated provider generators
          │
          ▼
content-addressed canary build ───────► shared receipt + planted facts
          │                                  │
          ├── immutable read-only base       │ independent oracle
          ├── reflink/copy writable clone    │
          └── real acquire→parse→write route │
                                             ▼
      unit laws     composition laws     state machines     scale/work laws
          │                │                    │                 │
          └────────────────┴────────────────────┴─────────────────┘
                                   │
                   full PR suite on tmpfs + receipts
```

There are only two durable authorities in this design:

1. executable production routes;
2. facts planted before execution and interpreted without calling the code
   under test.

Everything else is disposable run machinery. In particular, no YAML file
declares that a domain is covered.

## Corpus artifact, not fixture soup

Replace the current shared seeded DB and benchmark-only seeder with one thin
artifact publication adapter over the realized workload generator. It selects
a versioned workload ID/canary and produces the split archive plus the shared
workload/seed receipt. Its cache key extends, rather than renames, upstream
profile/build/archive identity with canonical DDL and parser/materializer
identity. The receipt or attached fact manifest should record:

- cache key and every input digest;
- upstream workload, profile, build, archive, scale-tier, and seed identity;
- expected sessions/messages/blocks/actions by origin and relation class;
- planted session IDs for each semantic witness;
- observed row counts, SQLite `quick_check`, schema versions, file sizes, and
  content hashes after the build;
- build command/version and completion timestamp for diagnosis, not identity.

Build into a fresh sibling directory, fail on any ingest error, validate the
manifest against the finished archive, fsync/close it, then rename the whole
directory into place. The completion marker is the valid manifest itself, not
an independently touched sentinel. Readers validate the key and cheap archive
checks before reuse. A killed or partial build is therefore never cache-valid.

Use the upstream named profile/tier/canary portfolio rather than defining a
second combinatorial fixture API. The table below describes consumer purposes,
not a new schema:

| Profile | Purpose | Typical size |
| --- | --- | ---: |
| semantic canary | precise planted relation(s) and independently known IDs | tens of sessions |
| distribution | provider/session/message/block distributions | thousands of sessions |
| tail | long sessions, large blocks, sparse/late relations | bounded multi-GB |
| adversarial | duplicates, missing peers, lineage, malformed/partial inputs | small to medium |
| prose | ranking/rendering cases where token content matters | tiny and curated |

The smallest named semantic canary should be the default composition fixture.
Volume tiers add activation conditions; they do not replace precise expected
sets or mint a separate workload identity.
The distribution data can be synthetic and ugly: correctness needs relational
and statistical shape, not LLM-quality prose.

Distribution inference, privacy classification, correlations, loss accounting,
and rotating/fixed seed policy belong to `polylogue-1xc.14.1`. Diet consumers
verify and reuse their provenance; they do not infer or serialize another
summary. The profile is executable input, never a coverage declaration.

### Read/write ownership

- Read-only tests open the immutable base archive.
- Mutating tests receive a private clone. Try filesystem reflink when the
  selected storage supports it, otherwise ordinary copy; record the chosen
  method in the run receipt.
- Do not share a writable SQLite database between tests or xdist workers.
- tmpfs is the default candidate for ordinary runs after measuring peak RSS and
  concurrency. Very large and I/O-realism lanes explicitly select NVMe.
- Never mix a persistent cache artifact and per-run pytest basetemp lifecycle.
  Give them separate roots and cleanup policies.

## Composition tests should use public protocols

The existing `ArchiveQueryCase` approach has the right intent but creates a
second partial query language. Replace its filter fields with public query
expressions (or a typed AST produced by the production parser) and keep only
the independent planted-fact oracle. A test should look conceptually like:

```python
expected = corpus.facts.session_ids_matching(
    origin="codex-session", text_token="needle", min_messages=7
)
assert await run_repository_query(expression) == expected
assert run_cli_query(expression).session_ids == expected
assert await run_http_query(expression) == expected
```

`facts.session_ids_matching` must operate on the manifest's source facts, not
SQL, `SessionFilter`, the query evaluator, or surface response builders. The
same expression can then exercise parsing, lowering, storage selection, and
output projection without implementing each surface's semantics in test code.

For MCP and the web reader, apply this pattern only to their rewrites. The
current implementations are obligation-mining sources, not migration targets.

## Correctness under growth

Elapsed time is too noisy to be the only oracle and result cardinality says
nothing about unbounded work. Add counters at the boundary relevant to each
failure:

- SQLite VM steps through `set_progress_handler`;
- rows/pages/bytes read where a production adapter exposes them;
- parser bytes and records consumed;
- peak memory for explicitly memory-sensitive routes;
- number of queued/retried convergence items.

Then express metamorphic work laws beside the vertical slice:

- appending irrelevant sessions does not change the selected ID set;
- work grows with the relevant/selective partition, not total archive size;
- adding an index-preserving irrelevant tail stays within a measured step
  envelope;
- replay/restart changes neither facts nor terminal state;
- duplicating a source artifact preserves idempotent cardinality.

Large failures must be reducible. The generator should preserve a provider-
neutral relational blueprint so sessions/relations can be delta-minimized while
replaying the same route. Otherwise a multi-GB failing seed becomes an incident
artifact rather than a maintainable regression test.

## Testability changes in production

Some stronger tests require small production seams, but add them only with a
real consumer:

- structured stage/transaction receipts rather than parsing log prose;
- injected clock/executor at temporal boundaries already modeled as services;
- explicit failpoints/barriers compiled or enabled only for tests;
- query/work counters exposed by the storage adapter;
- a real in-process HTTP application factory for protocol tests.

These seams report real behavior. Do not create an alternate test-only
implementation, generic event DSL, or mirror repository.

Delete `tests/infra/growth_budgets.py` and its self-test unless a real product
test first demonstrates that this generic class materially improves the
assertion. A direct assertion over an actual VM-step observation is clearer
than keeping an abstraction alive for possible future use.

## Test layout

Do not reorganize 274k test lines in one mechanical move. Apply this taxonomy
to new tests and migrate a cluster only while strengthening it:

- `tests/unit/`: one algorithm/module responsibility, real values, narrow
  collaborators only at genuine process/network/time boundaries;
- `tests/composition/`: in-process multi-module routes over real SQLite and
  public protocols, included in every PR run;
- `tests/property/`: independently computed models, metamorphic relations, and
  stateful sequences;
- `tests/integration/`: subprocess, daemon, socket, browser, restart, and fault
  boundaries;
- `tests/benchmarks/`: calibrated latency/throughput reporting, not semantic
  correctness;
- `tests/fuzz/`: crash/security exploration with promoted minimal regressions.

The folder is descriptive, not the gate. Marker expressions remain only where
execution isolation or resource cost actually differs.

## Runner changes

If repeated tmpfs measurements confirm the reported sub-two-minute full run:

1. make the full non-integration suite a single-version PR correctness gate;
2. keep testmon as an optional local edit-loop accelerator;
3. keep multi-Python full coverage post-merge or scheduled if its marginal
   compatibility value justifies the cost;
4. retain the isolated `load_sensitive`/TUI lane;
5. benchmark xdist's default scheduling against `--dist worksteal` using the
   same selection and fixture cache before changing it;
6. publish selection, duration, peak RSS, corpus cache hit/build, clone method,
   and failure artifacts from the managed runner.

This removes the testmon dependency graph from the merge correctness argument.
It remains useful for speed and failure context.

## Implementation sequence

The execution boundary is survivor-first. Land and verify the stronger
artifact/law before deleting its consumers. A separate certification pass runs
the historical witness and representative mutation, adjudicates unique
obligations, and only then authorizes an exact subtraction wave. This keeps one
self-authored green test from certifying its own dominance.

1. Record repeatable full-run disk/tmpfs/worker-count measurements and peak
   memory. Change no policy from a single lucky run.
2. Reconcile the merged workload, receipt, mutation, and repetition contracts;
   remove duplicated planned machinery.
3. Fix only the residual seeded-artifact validity: fail closed, extend shared
   identity, publish atomically, validate the archive, and clone immutably.
4. Attach independent planted facts to one realized canary through the real
   provider/pipeline route; do not define another workload profile.
5. Extend the C-03 query composition canary to public expressions plus
   the independent fact oracle; prove it kills representative mutations.
6. Reuse C-03's work-bound mutant/receipt, add the missing VM-step law, and
   remove the inert generic budget.
7. Move the full non-integration suite onto PRs when its measured envelope is
   stable.
8. Migrate scale/benchmark seeders onto upstream workload IDs and shared
   receipts, then delete dominated fixture/test paths.
9. Extend the established Hypothesis state machines for lifecycle and faults;
   delete the unused parallel lifecycle harness.

Each step produces behavior evidence and can land independently. None requires
the whole suite redesign to exist first.
