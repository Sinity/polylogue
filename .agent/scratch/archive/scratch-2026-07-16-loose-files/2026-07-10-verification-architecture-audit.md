---
created: 2026-07-10
purpose: Read-only audit of Polylogue's verification architecture against systematic gap discovery, non-vacuous Web/runtime proof, and responsive evidence retention.
status: complete-read-only-audit
project: polylogue
---

# Verification architecture audit

## Scope and evidence boundary

This audit read the test and verification substrate, current strategy and Web
audit notes, cached verification evidence, and the relevant Beads. It did not
edit product code or Beads and did not run tests, builds, mutation campaigns, or
live archive operations.

The main sources were:

- `TESTING.md`, `devtools/verify.py`, `devtools/run_tests.py`,
  `devtools/verify_runs.py`, and `devtools/pytest_progress_plugin.py`;
- `devtools/verify_closure_matrix.py`, `devtools/verify_manifests.py`,
  `devtools/evidence_dashboard.py`, and `devtools/failure_context.py`;
- `polylogue/scenarios/`, `tests/infra/`, validation/mutation/benchmark
  catalogs, and the runtime artifact graph;
- `polylogue/product/workflows.py`, the Web/daemon tests, `tests/visual/`, the
  SLO catalog, and reader benchmarks;
- Beads `f2qv*`, `v6vy`, `hjwr`, `1ilk`, `37km`, `fs1.1`, `9e5.19-.22`,
  `csg7`, `k6fm`, `kj22`, `yeq`, `stzx`, `3utv`, and `bby.11`;
- `.agent/scratch/2026-07-10-broad-project-strategy-and-verifiability.md` and
  `.agent/scratch/2026-07-10-webui-rewrite-and-proof-audit.md`.

## Verdict

The operator's goal is feasible in a bounded form, and the codebase already has
most of the difficult mechanics. The suite can systematically expose missing
evidence relative to declared workflows, runtime paths, origin contracts,
states, and budgets. It cannot discover undeclared product requirements or
decide whether a UI is good. Those still require design judgment, exploratory
use, and cold-reader sampling.

The current problem is not insufficient test volume. It is that Polylogue has
several inventories of coverage but no single, falsifiable relationship between
a product claim and the exact proof that authorizes it. Structural registries
are repeatedly described as coverage or closure even when they only prove that
names and files exist. Agents then inherit a green vocabulary that is stronger
than the underlying evidence.

A compact proof-gap compiler is worthwhile only if it is a projection over
existing authorities. It must not become another YAML matrix, a second test
runner, a witness database in the product, or a generic verification framework.
The smallest sound design is:

1. product registries declare what exists and what surfaces claim support;
2. test/devtools registries bind those declarations to exact executable proofs;
3. one compiler computes required cells minus bound proofs and emits gaps;
4. the existing verify-run ledger records what actually executed and whether it
   passed.

`WorkflowProofSpec` is a reasonable name for the workflow adapter, but it is too
narrow as the global abstraction. Storage convergence and origin compatibility
are not query-action workflows. The global type should be a small
`ProofObligation`/`ProofBinding` relation whose subject is one of an existing
workflow ref, runtime-path ref, or origin-contract ref.

## What already exists and should be kept

### Strong execution mechanics

The managed pytest harness is substantially better than the average repository:

- `devtools verify` and `devtools test` own subprocess lifetime, timeouts,
  process-group termination, and checkout-local serialization.
- Pytest emits selection counts and nodeids, per-test start/finish/phase events,
  slow phase summaries, stdout/stderr, resource samples, and a classified
  postmortem.
- Resource sampling includes process-tree RSS/PSS/CPU, host memory and swap,
  PSI, `/dev/shm`, basetemp size, and top processes.
- Progress detection reads test events as well as output bytes, so xdist chatter
  does not falsely prove worker progress.
- Runs are isolated under `.cache/verify/runs/<run-id>/`; the current run has an
  atomic pointer.
- Testmon, frozen clocks, Hypothesis, scale tiers, xdist controls, seeded
  archives, and cleanup of managed basetemps are already present.

This is the right substrate for responsiveness and evidence. It should be
repaired and indexed, not replaced.

### Strong semantic test abstractions

The highest-leverage reusable pieces are also already present:

- `tests/infra/archive_scenarios.py` provides typed session/message/block
  fixtures rather than raw dictionary blobs.
- `tests/infra/surfaces.py` projects the same archive through SQL records,
  hydrated records, repository/facade, CLI, MCP, and HTTP adapters.
- `tests/infra/oracles.py` compares semantic facts instead of snapshots of
  implementation detail.
- schema-driven Hypothesis strategies and protected parser/security/integration
  suites exercise broad classes of input.
- the runtime artifact graph plus scenario projections can enumerate artifacts,
  operations, maintenance targets, and declared paths.
- mutation and benchmark campaigns are typed executable catalogs, not only
  prose.
- `polylogue/product/workflows.py` already declares nine operator-facing
  query/action workflows and their claimed surfaces.

The proof compiler should reuse these types. It should not introduce another
fixture DSL or another runner.

### Existing evidence consumers

The repository has useful beginnings for compounding feedback:

- `test_economics_report.py` joins real coverage, fix history, testmon duration,
  and dependency fan-out.
- `failure_context.py` joins a failing nodeid to testmon dependencies, recent
  changes, fixtures, and witnesses.
- `task_history.py` tracks command latency/resource summaries and supports
  budget checks.
- `evidence_dashboard.py` reports cached pytest, coverage, benchmarks, static
  gates, witnesses, and mutation artifacts.

The missing step is consistent run identity and a durable compact outcome
index. The data is collected but only partially retained and consumed.

## Why agents get false green

### 1. The closure matrix proves file existence, not behavioral closure

`devtools/verify_closure_matrix.py:1-17` calls the matrix an executable
backstop. Its implementation at `:46-103` only checks that target paths and the
file portion of representative nodeids exist, that rows have legal gate names,
and that absent rows carry prose gaps. It does not:

- collect the claimed nodeid;
- prove that the default or named lane runs it;
- prove that the test imports or executes the target;
- inspect branch/statement coverage;
- verify an oracle or mutation kill;
- relate a user journey to a surface and state.

The broad rows make this particularly misleading. The entire daemon package is
represented by convergence tests, and the entire devtools package by four
representative files. There is no Web interaction row. A `required` row can be
green while the first-party authenticated Web UI cannot load.

This command should be called `verify coverage-index` or have its claims
narrowed. It can remain a useful inventory lint, but it must not be the product
closure authority.

### 2. Coverage authority is duplicated and contradictory

There are at least these overlapping sources:

- deprecated `test-coverage-domains.yaml`;
- `test-closure-matrix.yaml`;
- `scenario-coverage.yaml`;
- `campaign-coverage.yaml`;
- `test-quality-coverage.yaml`;
- `api-parity.yaml`;
- the product workflow registry;
- validation, mutation, benchmark, and runtime projection catalogs;
- CI workflow definitions;
- the evidence dashboard's hard-coded gate list.

The contradiction is observable now. `docs/plans/test-quality-coverage.yaml`
claims a 90 percent coverage floor and says it is enforced
(`:15-26`), while `pyproject.toml:158-168` sets `fail_under = 82`. The manifest
also reports 3,963 tests measured in March (`:89-95`), while the current tree
contains 4,956 explicit test functions before parametrization. `verify
manifests` validates the shapes and the existence of the referenced config; it
does not compare these numeric claims with the config or artifacts.

The deprecated qualitative catalog is still path-validated. This retains a
second stale place to describe coverage even after the closure matrix was
declared its successor.

### 3. Scenario coverage measures authored metadata, not observed execution

The generated quality reference reports 110 scenario projections and 21
covered runtime paths. `build_runtime_scenario_coverage()` considers a runtime
artifact or operation covered when an authored projection references it. That
is useful navigation, and the generated doc honestly says it is navigation.
It is not evidence that the command executed, asserted a meaningful invariant,
or survived a seeded defect.

`tests/unit/devtools/test_scenario_coverage.py` then locks the current inventory
and uncovered-name tuples. This catches registry drift, not behavioral
vacuity. A projection can be green because its metadata names an operation.

### 4. Workflow surface claims do not compile into surface proofs

The product registry requires nine workflow IDs
(`polylogue/product/workflows.py:88-100`) and declares surfaces such as CLI,
daemon, MCP, Web, completion, and docs. For example, the read-messages workflow
claims Web support at `:103-118`, and browser-capture status claims daemon/Web
at `:232-246`.

The executable golden paths begin at `:320` and are all CLI commands. They
cover only six of the nine workflow IDs. There is no rule requiring a Web
claim to have a browser proof, a daemon claim to have an HTTP proof, or a
completion claim to have a completion proof. This is the most natural existing
authority to extend with compiled proof obligations.

### 5. The Web proof lane does not execute a browser

`tests/visual` uses `urllib` and `HTMLParser`. Its assertions mostly verify IDs,
class names, function names, route strings, and envelope fields. It never runs
JavaScript, performs layout, clicks, focuses, scrolls, reconnects, or observes
responsive behavior.

Consequently it could not catch either verified structural failure:

- token-auth deployments serve a shell whose JavaScript never supplies the
  required authorization;
- healthy split-archive read routes bypass the bounded executor claimed by the
  closed concurrency work.

The current CI calls this job `demo-visual-verify`
(`.github/workflows/ci.yml:78-92`). The job is useful deterministic HTTP/DOM
smoke, but its name and downstream Bead wording encourage agents to treat it as
UI proof.

### 6. The PR gate omits the full behavioral suite

The full coverage/test job is explicitly disabled for pull requests
(`.github/workflows/ci.yml:30-55`). PRs receive lint/typecheck/Nix,
browserless demo visual smoke, and distribution checks. Local `devtools verify`
is therefore the behavioral merge evidence, but its testmon selector has a
verified stale-dependency question (`csg7`) and a history of xdist stalls.

This is not an argument to restore a 45-minute PR suite. It is an argument that
the small mandatory proof cells must run directly per PR instead of hoping
testmon or post-merge coverage happens to select them.

### 7. The SLO labels do not match the measured boundary

`docs/plans/slo-catalog.yaml` calls its rows reader and facets endpoints.
`tests/benchmarks/test_reader_api.py:39-65` and `:100-113` call repository
methods directly; no HTTP server, routing, serialization, auth, thread
admission, or response bytes are measured. Context and cost are placeholders.

The declared targets may be reasonable starting targets, but they are not
endpoint evidence. The p95 is also estimated as mean plus 1.645 standard
deviations under a near-normal assumption, rather than measured from raw
latencies. Product Web budgets must be re-baselined at the actual HTTP/browser
boundary.

### 8. Mutation authority is mostly dormant and can be badly scoped

There are 19 active mutation campaigns but no local run artifacts under
`.local/mutation-campaigns/` in this checkout. Freshness is soft by default.
The `daemon-http` campaign mutates the entire 3,827-line `http.py` but runs only
`tests/unit/daemon/test_daemon_http.py`, which contains five auth/origin helper
tests. It excludes the 1,016-line daemon contract suite and 3,182-line Web
reader suite.

Running that campaign would not establish HTTP-handler anti-vacuity. The
campaign scope itself needs the same proof-binding discipline as ordinary
tests.

### 9. Harness telemetry is rich but not yet a trustworthy history

The current cache contains:

- 4,046 run directories and about 1.7 GiB under `.cache/verify`;
- 3,402 compact history rows in a 6.8 MiB JSONL file;
- 23 MiB of testmon state and a 9.7 MiB coverage JSON report.

There is no run-artifact retention policy. Full per-test duration history is
not retained; only the top 20 slow phases per run survive, while testmon stores
the latest duration. Focused runs pass `git_head=None` from `run_tests.py`, and
all 2,542 focused-run manifests sampled from the cache lack a commit identity.
This is the reason `9e5.20` could not compute same-commit flakiness.

`evidence_dashboard.py:221-230` hard-codes only eight static gate names and
omits current gates such as closure-matrix, CI workflow, doc-command,
test-infra-currency, and clock hygiene. Its fallback searches history for the
last appearance of a gate, which can combine evidence from different commits
and tiers into one apparently current dashboard.

### 10. Tracker closure sometimes certifies investigation as implementation

This is a verification-system bug, not merely a Beads hygiene issue:

- `9e5.20` is closed after proving same-commit flakiness was not computable,
  although its AC required a ledger, quarantine marker, CI warning behavior,
  and a seeded random-fail demonstration.
- `9e5.21` is closed after an AST audit, although its AC required converting
  three tests and committing a rerunnable scan.
- `9e5.22` is closed after reading coverage, although its AC required active
  per-package floors and a deliberate failing demonstration.
- `fs1.1` is closed while its design explicitly says the Hermes wiring is
  speculative and unexercised, its acceptance criteria are empty, and real
  payload/outcome extraction is missing.

The likely root is that the parent `9e5` says every child is a read-only audit,
while those child designs and ACs describe implementation. The correct repair
is to make the audit/result and implementation/residual separate Beads, not to
pretend the implementation landed.

## Why the rejected cloud storage scenario happened

The bad cloud result is not evidence that cloud execution is intrinsically
unusable, and it was not caused by ChatGPT Instant versus Pro. It was a Codex
Cloud task with no exposed/pinned model identity.

The work contract made the failure likely:

- `9e5.19` asks for a "storage-layer correctness scenario family" without
  binding exact production paths or tests;
- it is incorrectly parented under an epic whose children are required to be
  read-only audits;
- its design still names the removed blob-lease mechanism;
- the scenario/manifest gates accept named projections and existing test paths;
- the cloud sandbox cannot use the live private archive, so a self-contained
  synthetic implementation is an easy local optimum.

The model then created a toy `SyntheticArchive` and proved its own DDL and
algorithms. A stronger model might have resisted, but the acceptance boundary
did not forbid reimplementation or require real production call paths. The
corrected packet's exact existing nodeids and explicit ban on alternate DDL,
hashing, FTS, GC, and lineage logic are the right cloud pattern.

Cloud lanes should receive one of these shapes:

- a narrow production diff with exact files and executable AC;
- a test adapter over exact existing production nodeids;
- a read-only audit that returns evidence and no code.

They should not receive an underspecified request to create a new "scenario
family" over a complex substrate.

## Smallest sound architecture

### Ownership boundaries

Keep the architecture clean:

- **Product declarations:** `polylogue/product/workflows.py`, runtime artifact
  graph, origin contracts, route/tool registries. These say what the product
  claims and what identity is authoritative.
- **Proof bindings:** devtools/test code. These say which real execution and
  oracle prove a declaration. They must not become product runtime semantics.
- **Execution:** existing pytest, Playwright/Vitest, devtools commands, and lab
  runners.
- **Evidence:** existing `.cache/verify/runs` ledger plus a compact derived
  index. No new product SQLite tier.
- **Task ownership:** Beads. The proof compiler reports a Bead ref or an
  unowned gap; it does not create a second queue.

Polylogue may later ingest verification-run artifacts as a generic external
origin if that becomes independently useful across projects. The proof
architecture must not depend on that, and no Polylogue storage schema should be
warped for its own test harness.

### Types

The minimum new relation is conceptually:

```text
ProofSubjectRef = WorkflowRef | RuntimePathRef | OriginContractRef

ProofRequirement:
  requirement_id
  subject_ref
  surface
  state_profile
  required_proof_kind
  cadence
  owner_bead

ProofBinding:
  proof_id
  requirement_id
  fixture_authority
  execution_spec
  assertion/oracle ref
  exact test nodeid or command
  evidence artifact kinds
  latency/resource budgets
  anti_vacuity_operator_ids

ProofResult:
  proof_id, run_id, git_head, fixture_fingerprint
  collected, executed, outcome, duration/resources
  evidence artifact paths and digests
```

Use existing `ExecutionSpec`, `AssertionSpec`, workflow IDs, runtime path IDs,
and origin names. Do not create a second command vocabulary.

### Compiler behavior

The compiler does four operations:

1. derive mandatory requirements from existing authorities and a small policy
   table;
2. validate that every binding references a real requirement, exact runner,
   oracle, owner, and evidence kind;
3. compute missing, duplicate, stale, and orphan bindings;
4. emit JSON plus concise text and fail only for ratcheted mandatory cells.

It does not execute tests itself. The runner records proof IDs in the existing
verify-run manifest. This separation keeps the static compiler fast and makes
"declared but not run" visible in evidence.

Do not add another authored `docs/plans/*coverage*.yaml`. Generate any human
matrix from the registries and proof bindings. Retire or freeze the deprecated
qualitative coverage catalog once the generated projection is useful.

### Mandatory policies, not a Cartesian product

The first compiler must not generate every workflow x surface x state x origin
x proof-kind combination. Start with semantic policies:

- each declared runtime surface has one canonical positive proof or an owned
  explicit gap;
- `web` requires a real browser journey, never HTML source inspection;
- a mutating workflow requires zero/many confirmation and idempotency evidence;
- a degraded-capable path requires one explicit degraded-state proof;
- an authoritative origin requires one observed-shape compatibility proof;
- a derived rebuild path requires full-versus-incremental equivalence;
- exact nodeids must collect and the named cadence/lane must include them.

Cap mandatory default cells at 40. Optional/nightly pairwise coverage can use a
greedy set cover capped at 96 cases later. Overflow remains an explicit gap; it
must not silently spawn hundreds of tests.

## Initial three authority cells

These are deliberately chosen to catch three different false-green classes.

### Cell 1: ready Web read journey

```text
id: workflow.find-read.web.ready
subject: workflow:find-then-read-messages
surface: web
state: demo-ready
fixture: deterministic demo archive
proof: Playwright Chromium journey
owner: 1ilk + bby.11, using 3utv contract spine
```

Journey: boot the real daemon UI, search a unique fixture term, open the emitted
session ref, assert title/messages/tool outcome/ref, navigate back, and assert
query, focus, and list state are restored. Retain trace, route ledger, and one
key screenshot.

Seeded defect: alter the emitted list ID so the detail route 404s. The journey
must fail. Removing the proof binding must make the compiler report the cell as
uncovered.

### Cell 2: authenticated first-party Web journey

```text
id: workflow.find-read.web.token-auth
subject: workflow:find-then-read-messages
surface: web
state: api-token-required
fixture: deterministic demo archive + configured token
proof: Playwright Chromium journey plus API/SSE request log
owner: 1ilk + bby.11/3utv residual auth acceptance
```

The first-party shell must acquire/use credentials through the chosen product
contract, load sessions, open detail, and connect/recover the live channel
without leaking the token into URLs, DOM, screenshots, or logs.

Seeded defect: remove credential injection from generated fetch/SSE transport.
The browser journey must fail on 401 rather than leaving a superficially valid
shell.

### Cell 3: derived rebuild equivalence

```text
id: runtime.source-index.incremental-full-equivalence
subject: runtime-path:raw-archive-ingest-loop + session-insight-repair-loop
surface: daemon/ops
state: reingest-update-delete-and-rebuild
fixture: real production seed/import path over demo corpus
proof: ordered logical table differential
owner: hjwr; supersedes the substantive part of 9e5.19
```

Path A performs a clean derived rebuild. Path B incrementally ingests,
converges, re-ingests one session, and deletes one session. Auto-census every
derived table; each must be diffed or explicitly classified volatile. Compare
A to a second A run for determinism and A to B for equivalence.

Seeded defects: omit provider-usage refresh or corrupt FTS repair. The
differential must identify the table and row mismatch. A registry-only check
cannot satisfy this cell.

### Next cell, not in the MVP three

After correcting `fs1.1`, add an observed Hermes state.db compatibility cell:
authoritative observed schema -> acquire/detect/parse/store -> tool outcome and
usage lanes -> read/continue. Removing `tool_result_is_error` or exit-code
extraction must fail. This is more valuable than another synthetic parser
fixture, but it should not delay the three current production failures.

## Anti-vacuity contract

Anti-vacuity has two distinct layers and both are required.

### Compiler mutations

Unit-test the compiler with in-memory registry variants:

1. remove a required workflow surface - orphan bindings must be reported;
2. remove a proof binding - the required cell must become missing;
3. point a binding at a missing/renamed exact nodeid - collection validation
   must fail;
4. remove the proof from its claimed CI/devtools lane - cadence validation must
   fail;
5. duplicate a binding with contradictory fixture authority - ambiguity must
   fail, not count twice as stronger coverage;
6. present a stale result from a different git head/fixture fingerprint - it
   must render stale, not green.

### Object-level mutations

Each mandatory proof carries at least one demonstrated defect operator:

- Web ref walk: corrupt emitted ID;
- Web auth: remove authorization transport;
- rebuild differential: skip one derived refresh or FTS repair;
- Hermes later: strip structural tool outcome fields;
- mutating workflow later: bypass confirmation/idempotency;
- browser state later: allow an old response to overwrite a new query.

The compiler only proves obligation coverage. These object mutations prove the
bound oracle is capable of detecting the defect class.

Do not require a full mutmut campaign for every cell. A small named fixture
fault or dependency-injected mutation is adequate when it changes the same
observable behavior and cannot agree with itself.

## Telemetry schema, storage, and retention

### Repair the current run ledger

Every tier must record:

- run ID, schema version, git head, dirty flag, worktree/root fingerprint;
- command/tier, Python/platform, start/end/duration/status/exit/diagnosis;
- selected/deselected/collected/executed counts and exact proof IDs;
- fixture/archive schema and content fingerprints where relevant;
- process-tree peak RSS/PSS/count, CPU, PSI maxima, temp bytes, output bytes;
- artifact paths plus content digests;
- for each proof: collected, executed, passed/failed/skipped, duration, evidence
  artifacts, and anti-vacuity operator version.

`k6fm` is the immediate identity repair. The current focused runner explicitly
passes no git head, which blocks same-commit flake analysis.

### Keep immutable artifacts, add a rebuildable index

Use the existing per-run directories as truth. Add one small rebuildable local
SQLite index under `.cache/verify/` for queries over runs, test outcomes, proof
results, and resources. It is a devtools cache, not a Polylogue archive tier.
It can be regenerated from retained `run.json` and compact outcome files.

Persist all node outcomes, but compact them:

- successful phase rows need nodeid, outcome, phase duration, and run/proof ID;
- long representations and raw stdout belong only to failures;
- compress finished event/outcome JSONL after the current-run pointer moves;
- keep a per-commit/node outcome index for same-commit flake detection;
- keep full raw resource samples only while investigating or benchmarking;
  retain peak/quantile summaries long-term.

### Retention

The current 1.7 GiB/4,046-directory cache has no policy. A reasonable first
policy is:

- never prune running runs;
- retain compact run/test/proof summaries for one year;
- retain raw failed-run logs/events/resources for 90 days and at least the last
  20 failures;
- retain raw successful focused/quick runs for 14 days and at least the last 20
  per tier;
- retain full/lab/mutation/benchmark raw runs for 90 days and at least the last
  10 per campaign;
- prune oldest successful raw payload first when the raw cache exceeds 2 GiB;
- never prune committed baselines or an artifact referenced by an open Bead;
- emit a prune receipt with bytes/runs removed and summaries retained.

This hoards the information needed for learning without hoarding duplicate
stdout and two-second resource samples indefinitely.

### Make the data pay rent

Initial consumers should be concrete:

- same-commit flakiness ledger and quarantine proposals;
- per-proof latency/RSS trends and budget recalibration;
- selection fan-out/staleness detection by comparing testmon with real
  execution coverage;
- proof-gap age and last-green commit;
- suggested focused verification for a changed subject;
- failure context showing the exact proof claim invalidated.

Do not build a general analytics UI before these five queries work.

## Latency and responsiveness budgets

These values come from the existing 3,402-row verify history and current run
artifacts; they are not invented Web-product targets.

| Surface | Samples | Current p50 | Current p95 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `verify closure-matrix` | 1,398 | 0.27 s | 0.32 s | Static inventory lint is cheap. |
| `verify manifests` | 1,398 | 1.47 s | 2.24 s | Manifest import/validation sets the static-compiler envelope. |
| quick verify | 3,073 | 16.86 s | 44.14 s | Appropriate upper bound for per-PR static feedback. |
| testmon verify | 307 | 94.69 s | 2,319.16 s | Tail is operationally unacceptable; current selector cannot be treated as a bounded gate. |
| latest focused run | 64 tests | 18.59 s | n/a | Peak tree RSS 137.6 MiB. |
| full verify history | 6 | 775.48 s | 954.83 s | All six recorded rows failed; not a green performance baseline. |

Budgets:

- proof-gap compilation: p95 <= 0.5 s, hard ceiling 1 s on this workstation;
- manifests plus proof-gap compilation: p95 <= 3 s, hard ceiling 5 s;
- quick/static verify: retain p95 <= 45 s;
- one mandatory per-PR workflow proof lane: provisional wall p95 <= 45 s
  after at least 20 runs, with a hard 120 s timeout and 60 s no-progress limit;
- default affected-test verify: remediation SLO p95 <= 300 s. Current p95 is
  2,319 s, so `csg7` and the xdist/D-state issue must be resolved before this
  lane is described as rapid or reliable;
- heartbeat for browser/proof lanes: 10 s with current step/state/artifact
  path; resource sampling can remain 2 s locally and be summarized after run.

There is no evidence-derived HTTP byte, first-paint, or browser interaction
budget yet. Do not carry the proposed 300 ms/64 KiB numbers as established
facts. The first 20 real route/browser runs should establish distributions,
then set budgets with fixture size, machine class, and route boundary named.

## Bead reconciliation

Do the Bead surgery before parallel implementation so agents share one
authority.

### Load-bearing now

1. **`1ilk`: amend and elevate.** It owns the real browser/component proof
   stack and is a binding rider of ratified `bby.11`. Add the ready and
   token-auth cells, failure states, a11y/keyboard assertions, real trace and
   screenshot artifacts, and explicit seeded regressions. Its current one-e2e
   AC is too weak for the verified auth/interaction failures.
2. **`3utv`: keep as the route contract spine.** The RouteSpec census is the
   generator for ref walks, OpenAPI, auth matrices, and the typed client. Do not
   build broad `stzx` or ref-walk inventories before it.
3. **`hjwr`: keep and make the storage authority.** Its auto-census,
   full-vs-incremental differential, and seeded divergence are exactly the
   non-vacuous storage proof needed.
4. **`csg7`: move ahead of reliance on default testmon.** Root-cause whether a
   fresh seed repairs the missing dependency edges and quantify the blast
   radius.
5. **`k6fm`: reconcile current master/merged PR state and close only after a
   focused run proves git identity.** The evidence cache still contains no
   focused git heads.
6. **`f2qv.2`: finish the already-active invariant work.** It is a concrete
   semantic trust-floor proof, not part of the meta-system. Land the real
   disjoint-lane invariant and live scratch reconciliation.

### Rewrite or supersede

1. **`9e5.19`: supersede its substantive scope with `hjwr`, or narrow it to a
   thin scenario adapter over real production nodeids.** Remove the stale blob
   lease premise. It cannot remain an implementation task under a read-only
   audit epic.
2. **`9e5.20-.22`: preserve their audit results, correct titles/AC/closure
   narrative, and link implementation residuals.** Do not claim the ledger,
   conversions, or per-module floors landed. `k6fm`, `kp4q`, `fgmk`, and
   `n4hb` are partial residuals; create only the missing residuals still wanted.
3. **`fs1.1`: reopen or create an explicit completion successor.** A closed
   authoritative-origin importer cannot remain speculative, unexercised, and
   AC-free. Require observed state.db schema, real fixture, keystone tool
   outcomes, usage/cost fidelity, and one provider-agnostic resume/continue
   proof.
4. **`37km`: amend its proof wording.** Browserless `tests/visual` can assert
   markup structure but cannot satisfy before/after screenshot, collapsed
   interaction, responsive layout, or readability claims. Land after `1ilk`'s
   first harness slice and reuse it.
5. **`yeq`: split the three programs.** Ref-walk belongs with `3utv`/`1ilk` and
   can be early. Metamorphic DSL and daemon chaos remain later independent lab
   Beads. One three-lane Bead encourages partial closure.

### Defer or drop from the near-term verification campaign

- `stzx` until enough routes derive from `3utv`; otherwise it fuzzes a
  descriptive OpenAPI layer that is not runtime authority.
- broad pairwise proof generation and a generalized artifact compatibility
  framework;
- `20d.16` three-scale performance family until the actual Web/HTTP boundary
  has one measured vertical;
- mutation campaign expansion beyond the three named defect operators;
- `f2qv.3-.5` if the week slips, after `f2qv.2` is safely landed;
- `v6vy`, which is useful MCP cleanup but unrelated to verification authority;
- mock-depth conversion and global per-module floors until the three mandatory
  proof cells are real.

`kj22` is an independent cheap correctness repair and can be pipelined, but it
must not displace the three authority cells.

## Execution order and drop order

1. Correct Bead claims and ownership (`1ilk`, `9e5.19-.22`, `fs1.1`, `hjwr`,
   `37km`, `yeq`).
2. Land the static proof requirement/binding compiler with the three mandatory
   cells declared and intentionally red where behavior is absent. Demonstrate
   compiler anti-vacuity in unit tests.
3. Repair run identity/retention (`k6fm`) and testmon authority (`csg7`) so
   execution evidence is attributable.
4. Stabilize Web auth/direct-read contracts and land `1ilk`'s two Playwright
   cells with seeded defects.
5. Land `hjwr` as the real storage differential and retire/supersede `9e5.19`.
6. Add the Hermes observed-origin cell only after `fs1.1` has an honest live
   completion contract.
7. Then expand through RouteSpec-generated ref walks, state-machine/component
   tests, and measured performance.

If schedule slips, cut in this order:

1. all pairwise/general proof generation;
2. Schemathesis and broad mutation campaigns;
3. metamorphic DSL and daemon chaos;
4. performance-family breadth and historical dashboard UI;
5. Hermes proof execution if no observed fixture is available.

Do not cut:

- correction of misleading closure claims;
- compiler anti-vacuity;
- the ready and token-auth browser journeys;
- the full-vs-incremental storage differential;
- git/fixture identity on verification evidence.

## Hard criticisms and limits

1. The repo currently has more verification vocabulary than verification
   authority. Adding another matrix without deleting or demoting an old one
   will make this worse.
2. `polylogue/verification/__init__.py` claims "anti-vacuity" while the package
   contains only manifest models. That claim should be removed or made real.
3. A green 82 percent aggregate coverage floor, 110 scenario projections, 33
   closure domains, or 19 mutation campaign declarations says little about
   whether the current Web UI works.
4. Tests cannot make product exploration obsolete. They can make manual use a
   final design/acceptance check rather than the first discovery mechanism.
5. Aesthetics and workflow desirability need human judgment. Automated browser
   evidence can catch broken layout, focus, responsiveness, state, a11y, and
   interaction, but not decide whether the experience is compelling.
6. Agent-written tests are not independent evidence by default. A test and a
   toy implementation written together can agree perfectly. Bind tests to real
   product paths, independent fixture authority, and seeded defect operators.
7. The proof compiler should live in devtools/tests. Making it a new Polylogue
   product subsystem or storage tier would be exactly the random appendage the
   architecture should avoid.

## Bottom line

The dream is directionally right: a compact obligation compiler plus real
browser/storage/origin proof can make gaps visible and turn each bug into a
compounding regression asset. The misframing would be believing that more test
registries or more generated cases create that outcome.

Polylogue needs fewer authorities, stronger bindings, and three real red/green
vertical proofs. Start with Web ready, Web token-auth, and derived rebuild
equivalence. Everything else should earn its place by reusing those mechanics
or by exposing a genuinely different defect class.
