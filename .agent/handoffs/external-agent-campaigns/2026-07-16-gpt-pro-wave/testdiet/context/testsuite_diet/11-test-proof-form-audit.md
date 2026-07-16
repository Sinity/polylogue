---
created: 2026-07-16
purpose: Correct the suite-wide example-heavy claim by auditing actual proof forms across every high-risk responsibility
status: adjudicated
project: polylogue
---

# Test proof-form audit

## Controlling correction

The suite must not be described as globally or overwhelmingly
`unit/example-heavy`. That conclusion came from treating `tests/unit/` and
`tests/property/` as proof-form classifications. They are not. Polylogue
keeps property tests beside their owning modules, and many tests under
`tests/unit/` execute real SQLite archives, async facades, HTTP servers,
filesystem workflows, concurrency seams, and production writers.

The defensible description is:

> Polylogue is explicit-witness-rich and predominantly component-owned, with
> substantial generative, stateful, real-storage, real-protocol, fault, and
> resource testing. Enumerated examples and declaration-bound tests are
> concentrated in particular clusters; they are not the suite-wide proof
> form.

This correction controls every Diet packet. A test is not a deletion candidate
because it is an explicit example or lives under `tests/unit/`. Deletion still
requires route, oracle, obligation, sensitivity, and diagnostic dominance.

## Method and limits

The audit combined:

1. an AST inventory of every Python test function and decorator;
2. whole-tree searches for Hypothesis, parametrization, iteration, async,
   filesystem, concurrency, subprocess, SQLite, archive builders, mocks, and
   interaction assertions;
3. a stratified read of the largest files and the files most likely to be
   misclassified as plain examples;
4. cross-reference reads of shared scenarios, surface adapters, Hypothesis
   strategies, state machines, and real daemon tests;
5. narration by behavioral responsibility rather than directory name.

The structural signals overlap. A filesystem test may also be parametrized,
async, and mocked. A patch may be a configuration seam or spy around a real
route. A direct example may be a uniquely valuable historical or security
witness. These counts are evidence about composition, not scores or automatic
verdicts.

Current worktree measurements:

| Signal | Observed amount | Interpretation |
| --- | ---: | --- |
| Python under `tests/` | 941 files, 275,647 nonblank LOC, 10,874 `test_*` functions | suite scale only |
| Under `tests/unit/` | 251,462 nonblank LOC, 10,527 test functions | ownership placement, not proof form |
| Hypothesis imports | 54 files, 18,608 nonblank LOC | mixed tests, strategies, and configuration |
| Hypothesis test modules | 15,348 nonblank LOC | 13,455 under `tests/unit/`, 1,893 under `tests/property/` |
| Shared Hypothesis strategies | 2,276 nonblank LOC | provider, message, storage, filter, pipeline, schema, and adversarial generation |
| `@given` tests | 210 | default profile is 100 examples; CI profile is normally 30 |
| Rule-based state machines | 2 | write/lineage and archive lifecycle |
| Parametrized test functions | 713 | 464 literal decorators expose at least 2,117 case rows; dynamic tables add more |
| Tests containing iteration | 789 | often execute a family of cases or surfaces inside one node |
| Filesystem-backed tests | 3,320 functions | includes real archive roots and durable artifacts |
| Async tests | 990 | facade, daemon, MCP, and storage coverage |
| Explicit concurrency constructs | 78 tests | distinct from xdist execution |
| Explicit subprocess calls | 51 tests | process/server coverage exists but remains narrow |
| Storage-route consumers | 332 files | union mentioning `SessionBuilder`, `ArchiveStore`, `sqlite3.connect`, or `write_parsed` |

The AST inventory found 1,899 test functions with a mock/patch signal, but
manual narration rejects using that number as a weakness count. For example,
`tests/unit/mcp/test_tool_contracts.py` patches archive-root configuration and
spies on query payload construction while still seeding a real archive and
executing the registered MCP tool. Conversely, some unmocked tests assert only
an authored registry tuple. Mock counts and lack of mocks are both inadequate
quality proxies.

## All-high narration

### Provider normalization and source parsing

This family is strongly mixed rather than example-only.
`tests/unit/sources/test_source_laws.py` uses generated payloads and laws for
detection, dispatch, encoding, streaming, bundles, archives, and failure
behavior. Provider-focused files such as
`tests/unit/sources/test_parsers_codex.py` retain explicit wire witnesses for
timestamps, compaction, nested usage, lineage, provenance suppression, and
historical shapes. `tests/unit/sources/test_models.py` contains many finite
field/type combinations, much of them parametrized. Schema-conformant
crashlessness and semantic round trips add generative breadth.

Explicit examples are appropriate where one provider version has a unique wire
quirk or the exact diagnostic matters. The remaining concern is whether a
schema-generated artifact traverses detection, lowering, parsing,
materialization, and public projection while an independent fact manifest
checks preservation. After workload profiles land, unknown future producer
drift and content-sensitive semantics remain outside observed distributions.

### Ingestion and materialization

`test_ingestion_chaos.py`, `test_resilience.py`, and
`test_roundtrip_hydration_laws.py` combine Hypothesis, malformed inputs,
round-trip laws, real writes, and injected failure. `test_ingest_batch.py`
exercises concrete append, replacement, raw-link, topology, and index behavior.
Mocks are common around orchestration boundaries, but this family is not merely
forwarding examples.

The residual weakness is sequence composition: acquire, parse, exclusion,
retry, hot-file growth, convergence, and restart are not yet explored by one
independent transition model. Individual stage examples can all be correct
while the composed lifecycle loses or duplicates work.

### Durable storage, lineage, blobs, and migrations

This is the clearest counterexample to the original claim.
`tests/property/test_write_path_state_machine.py` drives the production SQLite
writer and composed reader through parent/child arrival, replay, replacement,
append, variants, deletion, and stale writes. It checks logical transcripts,
physical tail storage, link state, FTS equality, and variant uniqueness after
transitions. `test_repository_state_machine.py` supplies another lifecycle
machine.

The surrounding suites use real files and databases for lineage normalization,
raw retention, repair receipts, blob integrity, durable migrations, backup
authorization, WAL/sidecar tamper, and targeted insight rebuilds. Many explicit
tests encode distinct crash or authorization boundaries and should not be
collapsed merely because their setup is verbose.

Residual risk is process-level crash consistency and schedule exploration:
abrupt termination between filesystem and SQLite durability points, ENOSPC,
permission changes, filesystem-specific rename/fsync behavior, and concurrent
user/daemon writes.

### Query grammar, semantics, execution, and surfaces

This area contains the strongest genuine explicit-example concentration.
`test_query_expression.py` has 328 test functions and many
one-syntax/one-AST or one-field/one-lowering assertions. It also contains real
archive executions for assertion, message, run, event, lineage, and terminal
units, so treating the entire file as low-value would be false.

The property backbone includes filter monotonicity, idempotence, commutativity,
RRF laws, cursor pagination, and cross-surface agreement.
`tests/unit/test_cross_surface_agreement.py` executes repository, facade, CLI,
MCP, and daemon HTTP adapters over planted scenarios and performs a
schema-generated parse/hydrate round trip. Its current query case space is
small, but the proof form is strong.

The real weakness is uneven semantic radius. Explicit lexer and diagnostic
examples are valuable. Repeated one-field lowering and mocked routing examples
become deletion candidates only when a generated query algebra with an
independent membership oracle reaches the same branches and preserves useful
diagnostics. C-03 and planned query laws strengthen exact selection and work
bounds; residual risk is the full DSL across grains, pagination, lineage,
FTS/vector retrieval, null/unknown states, and concurrent writes.

### Derived insights, analytics, and retrieval

`test_semantic_facts.py` uses hand-built but semantically rich sessions to
distinguish authored/runtime material, structured tool outcomes, terminal
state, provenance, timestamps, actions, and cost evidence.
`test_session_insight_refresh.py` exercises real incremental, targeted,
bounded, and full rebuild behavior. These are not trivial examples.

There are example-heavy pockets: `test_registry.py` is largely one
accessor/formatting condition per test, and several transform suites use one
large curated session as their truth source. The danger is not verbosity alone;
expected values can share production assumptions. Rebuild differential testing
addresses state equivalence, but analytics still need independent known-answer
datasets, conservation laws, provenance completeness, and versioned relevance
judgments.

### Daemon convergence and lifecycle

Direct Hypothesis use is sparse, but this family is scenario- and fault-heavy.
Unit suites exercise real ops/index files, locks, watcher cursors, retry debt,
bounded stages, cancellation, and HTTP routes. Integration launches real daemon
processes, delivers SIGTERM, holds SQLite locks, observes lifecycle forensics,
and waits for actual convergence. Web-reader tests frequently start a real
local server and call its HTTP routes.

Mocks remain dense in stage and supervisor tests, and explicit scenarios cover
selected schedules rather than the schedule space. The planned
debt-restart-retry-quiescence dossier is a strong survivor. Residual gaps are
deterministic interleavings, repeated restarts/failures, disconnect
propagation, and long-horizon leak/backlog behavior.

### CLI, Python API, MCP, HTTP, and browser projections

The surfaces are uneven:

- `test_facade_contracts.py` is mostly real async behavior over temporary
  split archives. Its lower sections prove tier ownership,
  corruption/absence distinctions, bounded refs, query units, mutations, and
  insight reads. The method-discovery and signature sweeps are static checks,
  and broad empty-archive matrices have a smaller semantic radius.
- MCP has catalog and forwarding tests, but the main tool-contract suite also
  seeds real archives and invokes registered tools. Raw patch counts overstate
  its isolation. The rewrite remains the consolidation boundary.
- CLI contains true micro-example clusters such as conversion/render helpers
  in `test_archive_query.py`, alongside destructive-confirmation and query
  behavior.
- Current web-reader tests mix real HTTP behavior, response-shape examples,
  source/DOM implementation checks, smoke budgets, and security cases. The
  rewrite should inherit product obligations rather than port files.

The residual concern is protocol composition: auth, selection, pagination,
cancellation, budgets, degradation, and errors must agree through actual
transports while sharing independent truth.

### Schema inference and workload generation

This area is property-rich. Core schema tests use Hypothesis for dynamic/static
keys, structural fingerprints, nested required fields, variant preservation,
privacy suppression, field distributions, and generated semantic relations.
`SyntheticCorpus` appears throughout source, cross-surface, schema, and
scenario tests.

Bounded correlated workload profiles remove important marginal and tail blind
spots. They still cannot generate unknown future producer changes or safely
retain sensitive content semantics. Maintain a separate small curated
semantic/adversarial corpus for Unicode, tokenizer, malformed textual
envelopes, locale/timezone, ranking, and human-language behavior. It is not a
replacement workload-profile authority.

### Verification tooling and harness trust

This family is mixed. `test_pytest_supervisor.py` uses subprocesses, process
groups, timeouts, stalled children, and cleanup deadlines. `test_verify.py`
exercises selection, seed receipts, progress, containment, and failure
reporting. These are behavioral harness tests.

`test_validation_lanes.py`, closure matrices, coverage declarations, and
parts of manifest testing contain genuine declaration-heavy pockets: exact
authored tuples, tags, path targets, and command strings can be made green by
editing a parallel declaration. They are navigation/configuration evidence,
not behavioral coverage. Verifier subtraction should preserve receipts and
process behavior while removing self-certifying claims.

Residual systemic risk is selective-test unsoundness. Real mutation and xdist
witnesses prove important paths, but testmon collection-only imports, dynamic
loading, generated code, external artifacts, and stale dependency data still
require periodic full verification and explicit unsafe-selection handling.

### Deployment, recovery, and external environment

Integration and deployment-smoke tests exist, including actual daemon
processes and operator-facing schema workflows. Some deployment-smoke unit
tests replace command and HTTP execution to prove reporting; they do not prove
an installed artifact. The schema workflow uses a real temporary registry but
pre-populates it rather than running corpus inference.

The suite has deployment examples, but not an environment matrix. Residual
failures can hide in built wheels, Nix/systemd permissions, filesystem
differences, optional SQLite extensions, old archive upgrade chains, missing
credentials, and upstream browser/provider changes.

### Security and privacy

Security is not example-only. Hypothesis drives attachment/path adversaries,
SQL-injection boundaries, query values, and privacy guards. HTTP auth uses
finite decision matrices. Migration and repair suites exercise forged,
aliased, tampered, and stale receipts. Some AST/source scans enforce genuine
secret-flow or architecture boundaries.

Explicit attack strings should remain as historical witnesses even when
generators cover their class. Residual gaps are complexity attacks,
decompression bombs, symlink/hardlink races, prompt/context injection,
cross-surface secret leakage, and authorization races under concurrent state.

### Scale, performance, and long-horizon behavior

This is not mainly an example-count problem; it is an oracle and route problem.
The benchmark fixture manually constructs normalized storage records from
hard-coded distributions despite claiming `SyntheticCorpus` in its prose. It
bypasses provider parsing and omits decisive relationships. Several scale
assertions accept any bounded/nonempty answer and do not prove membership.

Workload profiles, resource receipts, scale tiers, and C-03 improve this area.
Residual risk is duration: per-run boundedness does not prove days of repeated
capture, convergence, retry, rebuild, or client activity without FD/thread/
process leaks, retained roots, WAL growth, fragmentation, or debt accumulation.

## Where `example-heavy` is accurate

The phrase is useful only when attached to a named cluster:

| Cluster | Actual shape | Diet implication |
| --- | --- | --- |
| query lexer/AST/one-field lowering | many explicit syntax-to-IR examples | retain grammar boundaries and diagnostics; consolidate only dominated field mappings |
| small CLI conversion/render helpers | one-input/one-output cases | decision tables may improve density; formatting contracts are not query truth |
| insight registry accessors | finite micro-examples for defaults and formatting | low risk; compression is optional unless maintenance cost is demonstrated |
| provider wire models/parsers | curated finite historical shapes | valuable compatibility witnesses; generated round trips dominate only duplicated field cases |
| validation-lane/closure declarations | exact authored metadata, paths, tags, commands | navigation/config proof only; never behavioral completeness |
| API/MCP discovery catalogs | operations assigned to test-authored sets | useful change detector, not proof that each operation works |
| current MCP/web forwarding and layout | real routes, mocks, catalogs, source/DOM pins | extract obligations at rewrite boundary; neither bulk-port nor bulk-delete |

An explicit example is strong when it preserves a unique compatibility input,
historical regression, security attack, durability boundary, diagnostic, or
public representation. A property is weak when its oracle is self-referential
or merely asserts that arbitrary input does not crash. Proof form and strength
are separate dimensions.

## Corrected Diet rules

1. Do not target example tests as a portfolio category.
2. Do not infer isolation from `tests/unit/`; inspect the production route.
3. Do not infer weakness from a mock count; identify what remains real and
   whether the mock is an external seam, a spy, or a replacement substrate.
4. Retain explicit historical, compatibility, security, durability, and
   diagnostic witnesses even when a broader property exists.
5. A strong survivor combines the proof forms the risk warrants: independent
   truth or metamorphic law, real route, historical example, fault/restart
   sequence, and work/cleanup receipt.
6. Compression is authorized only after coverage-context overlap nominates a
   cluster and a representative production mutation demonstrates dominance.
7. Prefer decision tables for finite mappings, Hypothesis for values,
   combinations, shrinking, and transitions, and separate examples when their
   diagnostics or historical identity matter.
8. Describe the suite-wide residual as composition, interleaving, environment,
   and long-horizon risk—not a general shortage of property testing.

## Revised post-plan residual priorities

Assuming the planned workload, query-law, rebuild, convergence, rewrite,
restore, receipt, mutation, and xdist work lands, the highest residual test
investments are:

1. extend state machines beyond storage into acquire/cursor/convergence,
   assertion lifecycle, backup/restore, and cancellation protocols;
2. execute generated provider artifacts through the complete acquire-to-public
   route with independent planted facts;
3. explore deterministic daemon/user/query interleavings and process crash
   points rather than relying on random scheduling;
4. add installed-artifact and historical-archive matrices for packaging,
   system service, SQLite feature, migration, and restore behavior;
5. add accelerated soak tests for resource retention and debt/WAL growth;
6. retain curated semantic/adversarial content for behavior privacy-safe
   profiles cannot learn;
7. maintain independent analytics/retrieval truth and relevance judgments;
8. periodically run full verification because selective dependency tracking
   cannot prove its own completeness.

These priorities strengthen missing proof dimensions. They do not justify a
larger suite by default: each survivor should replace dominated local tests
where—and only where—the evidence is real.
