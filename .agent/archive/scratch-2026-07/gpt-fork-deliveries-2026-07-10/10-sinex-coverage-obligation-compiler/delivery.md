## Turn 23 (role=assistant)

# Sinex coverage-obligation compiler review

This report executes the attached specification against the supplied Sinex chisel package as the complete repository, Beads export, and scratch-note record. fileciteturn0file0

## Verified evidence baseline

The scratch record reports the following read-only counts at **2026-07-10 06:57**:

```text
invocations                 20,268
test_results               418,743
test_proof_units               970
test_execution_manifests    14,403
test_dependency_edges        2,743
stage_timings               36,417
coverage_regions                 0
impact_audit_runs                0
trace_events                     0
```

Evidence: `sinex-scratchpad.xml:36953-36958`.

The package does not contain `.sinex/state/xtask-history.db`, so those counts cannot be independently re-queried. The source does confirm that the history database is explicitly an evidence ledger rather than a disposable cache: `xtask/src/history/db/schema.rs:100-105 — HistoryDb::init_schema`.

| Evidence surface | Schema | Production writer found | Finding |
|---|---|---|---|
| Invocations | `xtask/src/history/db/schema.rs:108-168 — invocations` | `xtask/src/history/db/invocations.rs:38-57 — HistoryDb::start_invocation`; completion at `60-98` | Live writer |
| Test results | `xtask/src/history/db/schema.rs:170-187 — test_results` | `xtask/src/history/db/test_results.rs:3-24 — HistoryDb::record_test_result` | Live writer |
| Stage timings | `xtask/src/history/db/schema.rs:213-235 — stage_timings` | `xtask/src/history/db/invocations.rs:154-190 — HistoryDb::record_stage_timing` | Live writer |
| Proof evidence | `xtask/src/history/db/schema.rs:252-266 — proof_evidence` | `xtask/src/history/db/invocations.rs:788-851 — HistoryDb::record_proof_evidence` | Live writer; row count unknown |
| Test proof units | `xtask/src/history/db/schema.rs:268-282 — test_proof_units` | `xtask/src/history/db/invocations.rs:881-938 — HistoryDb::record_test_proof_unit`; called after a successful test invocation at `xtask/src/commands/test.rs:2002-2087 — TestCommand::execute` | Live writer |
| Dependency edges | `xtask/src/history/db/schema.rs:284-295 — test_dependency_edges` | `xtask/src/history/db/impact.rs:535-568 — insert_test_dependency_edge` | Live artifact importer |
| Execution manifests | `xtask/src/history/db/schema.rs:311-326 — test_execution_manifests` | Test macro emits artifacts through `xtask/src/sandbox/context.rs:1506-1545 — persist_test_execution_manifest`; importer at `xtask/src/history/db/impact.rs:456-497` | Live artifact importer |
| Coverage regions | `xtask/src/history/db/schema.rs:297-309 — coverage_regions`; unique identity index at `784-792` | `xtask impact seed-coverage` runs one exact test under LLVM coverage and imports rows: `xtask/src/commands/impact.rs:138-247 — seed_coverage`; importer at `xtask/src/history/db/impact.rs:498-531` | Writer exists, but is manual and one-exact-test-at-a-time; zero rows in the recorded store |
| Impact runs and decisions | `xtask/src/history/db/schema.rs:328-345` | `xtask/src/history/db/impact.rs:61-112 — record_impact_plan` | Live writer; row counts unknown |
| Impact audit runs | `xtask/src/history/db/schema.rs:347-358 — impact_audit_runs` | `xtask/src/commands/impact.rs:249-339 — audit`; DB writer at `xtask/src/history/db/impact.rs:114-153` | Writer exists, but the current audit runs one broadened command on the current tree, records the requested rather than actual sample size, and turns any command failure into a count of one |
| Trace events | `xtask/src/history/db/schema.rs:369-378 — trace_events` | Bounded, lossy writer at `xtask/src/history/tracing_layer.rs:248-356 — writer_loop/flush_batch`; installed after invocation creation at `xtask/src/lib.rs:499-517` | Writer exists. Current source contains a per-layer filter fix for the default-verbosity suppression path, but still discards `try_init()`’s result, so installation failure remains unobservable: `xtask/src/lib.rs:951-981 — init_tracing` |

The conclusion is narrower than “empty scaffolding”: the three empty tables have real writers, but no evidence in the package demonstrates that those paths have successfully populated the audited database. The first gap report must therefore work without LLVM regions or trace rows. This matches the package scratch conclusion at `sinex-scratchpad.xml:35938-35965` and `36961-36974`.

The existing mechanisms should be reused: `#[sinex_test]`, isolated DB/NATS contexts, VM categories, impact planning, proof reuse, fault hooks, waits/timeouts, flaky tracking, and history analysis are all recorded as existing strengths in `sinex-scratchpad.xml:35923-35930`. The `Sandbox` already exposes isolated production APIs rather than wrapping them in a separate test abstraction: `xtask/src/sandbox/context.rs:1-13 — Sandbox`.

---

# 1. Obligation model

## 1.1 Types and identity rules

```rust
#[repr(transparent)]
pub struct ObligationId(Box<str>);

#[repr(transparent)]
pub struct ClaimId(Box<str>);

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum ProofLevel {
    Static,
    Unit,
    Component,
    Pipeline,
    Process,
    Vm,
}

pub enum RegistryKind {
    Automaton,
    Source,
    Schema,
    Rpc,
    JetStream,
    TestModule,
}

pub enum ObligationFamily {
    Registration,
    InputBoundary,
    IncrementalBehavior,
    BoundedRebuild,
    LiveVsRebuildDifferential,
    ReplayRemint,
    CrashSettlement,
    OrderingBatchingRestart,
    RevisionInvalidation,
    ConsumerContract,
    LaneAuthorityIsolation,
    ResourceBudget,
    TestReachability,
}

pub struct CoverageObligation {
    pub id: ObligationId,
    pub area: Box<str>,
    pub registry_kind: RegistryKind,
    pub subject_key: Box<str>,
    pub family: ObligationFamily,
    pub variant_key: Option<Box<str>>,
    pub surface_fingerprint: Digest,
    pub minimum_level: ProofLevel,
    pub mechanism_contract: MechanismContract,
    pub activation: ObligationActivation,
    pub source: SourceLocator,
}

pub struct CoverageClaim {
    pub id: ClaimId,
    pub obligation_id: ObligationId,
    pub declared_level: ProofLevel,
    pub test: TestLocator,
    pub declaration_fingerprint: Digest,
}

pub enum CoverageGapKind {
    Missing,
    Stale,
    Waived,
    Overstated,
    Unreachable,
    FalsePassing,
}

pub struct CoverageGap {
    pub kind: CoverageGapKind,
    pub obligation_id: ObligationId,
    pub claim_id: Option<ClaimId>,
    pub blocking: bool,
    pub reason_code: GapReasonCode,
    pub expected: serde_json::Value,
    pub observed: serde_json::Value,
    pub evidence_refs: Vec<HistoryEvidenceRef>,
}
```

`ObligationId` uses this canonical shape:

```text
sxob1/<registry-kind>/<escaped-stable-subject>/<family>[/<variant>]
```

Examples:

```text
sxob1/automaton/canonicalizer/live-vs-rebuild-differential
sxob1/source/terminal.atuin-history/input-boundary
sxob1/schema/core.events/revision-invalidation
sxob1/rpc/events.query/consumer-contract
sxob1/jetstream/activity/confirmed-events/main-consumer/crash-settlement
sxob1/test-module/crate%2Fsinexd%2Ftests%2Fsources%2Fmessaging_parser_test.rs/test-reachability
```

The ID never contains a test name, function name, file line, content hash, Git commit, or runtime function address. Those are evidence attributes and may change. Registry descriptor changes update `surface_fingerprint` while preserving the obligation ID; this is what makes existing evidence become **stale** rather than creating a new unrelated obligation.

`ClaimId` similarly uses an explicit, owner-chosen stable ID:

```text
sxcl1/<area>/<semantic-claim-slug>
```

A valid waiver is represented as a reported, nonblocking `Waived` gap. It must name an owner, reason, expiry, exact obligation ID, and exact current surface fingerprint. Expiry or a surface change invalidates the waiver and produces a blocking stale/missing result.

The six classes have these exact meanings:

- **Missing:** no reachable, current, successful claim and no valid waiver. A failed test does not introduce a seventh class; it yields `Missing` with reason `CLAIM_FAILED`.
- **Stale:** the claim was executed against a different obligation surface, declaration, source content, build graph, toolchain/substrate seal, or evidence policy.
- **Waived:** a current, unexpired, fingerprint-bound waiver suppresses blocking but remains visible.
- **Overstated:** the declared proof level exceeds the mechanism actually observed.
- **Unreachable:** the claimed test or a test-bearing module is outside the current Cargo/module/nextest graph.
- **False-passing:** a green result cannot discharge the mechanism contract—for example zero tests selected, a fault hook was never hit, no archive/replay operation occurred, or no negative-control mutation was observed.

For one candidate claim, evaluation precedence is: valid waiver, unreachable, stale, overstated, false-passing, missing. This prevents a single dead test from producing six redundant diagnostics.

## 1.2 Exact derivation sources

### Automata

The authoritative catalog is `AUTOMATA`, whose 16 entries are:

```text
canonicalizer
analytics
attention-stream
interval-lift
health
session
hourly
daily
entity-extractor
entity-resolver
relation-extractor
entity-enricher
tag-applier
embedding-producer
document-parser
instruction-reconciler
```

Evidence: `crate/sinexd/src/automata/registry.rs:46-146 — AutomatonSpec/AUTOMATA`.

Its existing contract probe exposes continuous, historical, loop-ownership, and checkpoint-ownership properties: `crate/sinexd/src/automata/registry.rs:29-54 — AutomatonRuntimeContract/AutomatonSpec`. Deployment selection is already resolved through `SINEX_AUTOMATA_ENABLED`: `crate/sinexd/src/automata/registry.rs:164-195 — parse_enabled`.

The exporter must add one stable `runtime_type_key` or typed descriptor callback to `AutomatonSpec`; it must not fingerprint the `run` function pointer. The fingerprint is the normalized tuple of name, runtime type key, and contract bits. All 16 receive registration and input/consumer obligations; historical/checkpoint/rebuild families activate from the contract flags.

### Sources

Source obligations are derived from the cross-product and consistency checks among:

1. `SourceContract`: ID, namespace, event types, privacy tier, horizons, retention, occurrence identity, and access scope. `crate/sinex-primitives/src/source_contracts/contract.rs:7-33 — SourceContract/all_source_contracts`.
2. `SourceRuntimeBinding`: implementation and adapter identity, privacy context, resource profile, capabilities, checkpoint family, runtime shape, material lifecycle, transport semantics, and build impact. Proposed bindings are explicitly non-authoritative. `crate/sinex-primitives/src/source_contracts/runtime.rs:11-52 — SourceRuntimeBinding`; inventory at `293-297 — source_runtime_bindings`.
3. Registered parser manifest and privacy metadata. `crate/sinexd/src/sources/dispatch.rs:94-142 — ParserRegistryEntry/parser_inventory_records`.
4. Registered source factory. The deployed binding validator already rejects a contract with no source factory: `crate/sinexd/src/sources/bindings.rs:129-163 — validate_bindings`.
5. The actual `SourceBindingsManifest`, including enabled instance indices, runtime configuration, arguments, and environment overlays. `crate/sinexd/src/sources/bindings.rs:57-111 — SourceBinding/SourceBindingsManifest`.

A compile-time contract or runtime binding marked `proposed` is catalogued but does not create a blocking deployed obligation. A live manifest row creates instance-specific lane, checkpoint, restart, resource, and authority obligations. If the manifest is unavailable, deployment status is `Unknown`; it is not silently treated as deployed or absent.

The current binding runner serializes only the first-poll environment mutation window and documents a residual after-first-`await` hazard: `crate/sinexd/src/sources/bindings.rs:24-54 — BINDING_ENV_LOCK`. That surface should activate a concurrency/isolation obligation for bindings with `extra_env`.

### Schema definitions

`crate/sinex-schema/src/defs/` is explicitly the canonical schema source for tables, columns, indexes, and constraints: `crate/sinex-schema/src/defs/mod.rs:1-6`. `ALL_TABLES` and `all_tables()` provide table-level identity and broad metadata: `crate/sinex-schema/src/defs/mod.rs:83-94,458-460 — TableMeta/all_tables`.

`TableMeta` alone is insufficient for staleness because it omits columns, indexes, constraints, and trigger definitions. The compiler should add a normalized exporter over the existing SeaQuery definitions in this canonical module. It must serialize sorted table, column, type, nullability, default, key, index, constraint, trigger, hypertable, and cleanup-policy facts. It must not create a second hand-maintained schema registry.

Every table receives registration and boundary obligations. Revision/invalidation, replay/remint, bounded rebuild, lane isolation, and resource obligations activate from actual schema features such as derived-plane ownership, triggers, archive/replacement relationships, hypertables, and cleanup protection.

### API handler surface

The typed authority already exists as `RpcMethod<Req, Resp>`, carrying method name, role, domain, stability, mutability, and request/response Rust types: `crate/sinex-primitives/src/rpc/mod.rs:87-148 — RpcMethod/RpcMethodInfo`.

The runtime registry currently discards most of that metadata and stores only a handler plus required role: `crate/sinexd/src/api/rpc_registry.rs:121-147 — HandlerFn/RegistryEntry/RpcRegistry`. Registration owns the JSON decode/encode boundary at `crate/sinexd/src/api/rpc_registry.rs:157-187 — pool_typed_rpc`, while dispatch enforces lookup and authorization at `477-508 — RpcRegistry::dispatch`. `list_all_methods()` currently returns only names and roles: `540-561`.

Change `RegistryEntry` to retain `RpcMethodInfo`, and export the built registry from `build_registry_impl()`, whose registration chain starts at `crate/sinexd/src/api/rpc_registry.rs:619-655`. The compiler then derives from the same runtime registration calls rather than grepping source.

Each registered method gets registration, request boundary, response/consumer contract, and role/authority obligations. Mutating methods additionally receive settlement, revision/invalidation, and retry/idempotency obligations. Batch, replay, or stream-oriented descriptors activate their respective metamorphic and replay families.

### NATS stream and consumer shapes

The current source has authoritative but scattered literal shapes:

- Source-material stream: WorkQueue retention, file storage, 72-hour age, byte ceiling, and 512 KiB message ceiling. `crate/sinexd/src/runtime/acquisition_manager.rs:331-371 — AcquisitionManager::ensure_streams_once`.
- Material assembler duplicates that stream shape and creates a DeliverAll durable consumer with bounded pending work and an explicit backpressure gate. `crate/sinexd/src/event_engine/material_assembler/pipeline.rs:130-162 — bootstrap_streams`; `222-276 — spawn_material_consumer`.
- Pull-consumer contract: stream, durable, filters, delivery policy, ack wait, max pending, max delivery, and initial-replay rejection. `crate/sinexd/src/runtime/stream/kernel.rs:10-65 — PullConsumerSpec`; creation, replay refusal, reconciliation, and snapshot at `68-179 — ensure_pull_consumer`; explicit-ack config at `181-198 — pull_consumer_config`.
- Main event consumer constructs a DeliverAll consumer and configures ack/retry limits: `crate/sinexd/src/event_engine/jetstream_consumer/run_loop.rs:24-45 — JetStreamConsumer::run_with_ready_signal`.
- Confirmed events, DLQ, processing-failure, and invalidation stream shapes are created at `crate/sinexd/src/event_engine/jetstream_consumer/bootstrap.rs:27-131 — bootstrap_streams`. The confirmed-events stream is explicitly a bounded delivery bus rather than an archive.
- Externally managed mode currently verifies stream presence, not complete shape, for source material and event-engine streams: `crate/sinexd/src/runtime/acquisition_manager.rs:338-351`; `crate/sinexd/src/event_engine/jetstream_consumer/bootstrap.rs:137-175`.

Extract those existing literals into serializable `StreamBootstrapSpec` constructors and continue using `PullConsumerSpec`. Runtime bootstrap and the obligation exporter must call the same constructors. This avoids creating a parallel topology registry.

Expected topology comes from those constructors; observed topology comes from the substrate seal’s live NATS digest. Presence-only external verification cannot discharge a consumer-contract or resource-budget obligation.

### Test module reachability

The compiler also creates a synthetic subject for every Rust test-bearing file. A candidate is any `tests/**/*.rs` file containing `#[test]`, `#[tokio::test]`, `#[sinex_test]`, `#[rstest]`, or a known test-generating macro. Its `TestReachability` obligation is discharged only when the module is reachable from a Cargo test target and its generated tests are present in the current nextest list.

This is independent of whether the file contains a claim. It therefore permanently catches claimless orphan files.

## 1.3 Obligation family matrix

`R` means always derived for a registered subject. `C` means derived when the actual descriptor/topology has the relevant capability. `—` means the family is not native to that registry kind.

| Registry kind | Registration | Input boundary | Incremental behavior | Bounded rebuild | Live/rebuild differential | Replay remint |
|---|---:|---:|---:|---:|---:|---:|
| Automaton | R | R | C | C | C | C |
| Source | R | R | C | C | C | C |
| Schema | R | R | C | C | C | C |
| RPC | R | R | C | C | C | C |
| JetStream | R | R | R | C | — | — |

| Registry kind | Crash settlement | Ordering/batching/restart | Revision/invalidation | Consumer contract | Lane/authority isolation | Resource budget |
|---|---:|---:|---:|---:|---:|---:|
| Automaton | C | C | C | R | C | C |
| Source | C | C | C | R | R | R |
| Schema | C | C | C | R | C | C |
| RPC | C | C | C | R | R | C |
| JetStream | R | R | C | R | R | R |

Activation rules are descriptor-driven:

- Automaton historical/checkpoint flags activate rebuild, differential, replay, and settlement families.
- Source checkpoint family, runtime shape, material lifecycle, transport semantics, occurrence identity, privacy tier, access scope, and resource profile activate the corresponding families.
- Schema archive/replacement, derived-plane, trigger, hypertable, lane, and cleanup characteristics activate behavioral families.
- RPC mutability, domain, stability, role, and method-specific batch/replay/stream descriptors activate operational families.
- JetStream retention, discard, delivery, ack, pending, retry, replay, subject, namespace, and invalidation shape directly activate its families.

A missing descriptor field does not manufacture a speculative obligation. It produces a catalog diagnostic where the registry cannot express a capability that the runtime demonstrably has.

---

# 2. Proof-claim ID mechanics

## 2.1 `#[sinex_test]` annotation

The macro currently accepts only timeout, trace, serial, and scope, and deliberately rejects unknown fields: `xtask/macros/src/lib.rs:18-34 — SinexTestConfig/KNOWN_SINEX_TEST_ATTRS`; parser at `91-203 — parse_sinex_test_attrs_tokens`.

Extend it with repeatable `claim(...)` clauses:

```rust
#[sinex_test(
    timeout = 60,
    claim(
        id = "sxcl1/source/atuin/parser-input-v1",
        obligation = "sxob1/source/terminal.atuin-history/input-boundary",
        level = "component"
    ),
    claim(
        id = "sxcl1/source/atuin/occurrence-repeat-v1",
        obligation = "sxob1/source/terminal.atuin-history/incremental-behavior",
        level = "component"
    )
)]
async fn atuin_parser_contract() -> TestResult<()> {
    // ...
}
```

A test can carry multiple claims, but each `ClaimId` names exactly one obligation. This supports property, state-machine, differential, and fault scenarios without forcing one test per branch.

There should be no central claim-name registry. The explicit claim ID is the stable key. Renaming a function, moving its module, or consolidating it under the `v7od` roots changes `TestLocator` but not `ClaimId`. Moving a claim to a replacement test is allowed; duplicate live definitions of one claim ID are a catalog error.

For generated source cases, extend `production_path_case_test!` to pass claim clauses through to the generated `#[sinex_test(...)]`. The current macro generates that attribute centrally at `crate/sinexd/tests/sources/production_path.rs:145-155 — production_path_case_test`, so aggregation does not require editing generated function bodies.

Every expansion path must emit claims alongside the existing execution manifest. The macro currently emits that manifest for property tests, rstest cases, context tests, and simple tests at:

- `xtask/macros/src/lib.rs:557-562`
- `xtask/macros/src/lib.rs:1080-1084`
- `xtask/macros/src/lib.rs:1124-1136`
- `xtask/macros/src/lib.rs:1204-1214`

The current artifact stores the bare function name, package, module, source, binary, PID, attempt, planner version, and content hash: `xtask/src/sandbox/context.rs:106-118 — TestExecutionManifestArtifact`; persistence at `1506-1545`. Add claim IDs and current test-target identity to the same artifact.

## 2.2 Reachability and rename survival

The compiler resolves a claim through three independent identities:

1. **Stable semantic identity:** explicit `ClaimId`.
2. **Static source identity:** package, test target, module path, file, function, declaration fingerprint.
3. **Current executable identity:** exact current nextest test ID and executable hash.

A rename changes layers 2 and 3 but not layer 1. A source-body change changes the declaration/content fingerprint and requires fresh evidence. A module move under the `v7od` aggregate roots does not invalidate the claim unless its mechanism or body changes.

Reachability is computed from:

- Cargo metadata test targets;
- recursively resolved `mod` and `#[path]` edges;
- cfg evaluation for the current xtask environment;
- current `cargo nextest list` output invoked internally by xtask;
- a repository scan for test-bearing files.

Static module reachability detects orphan files even when there is no claim to look up. Current nextest discovery detects renamed, filtered-out, ignored, or zero-test identities. A historical name is never accepted merely because it exists in `test_results`.

## 2.3 Proof level is detected, not trusted

The declared level is a claim. The observed level comes from structured mechanism evidence emitted at production choke points.

```rust
pub struct MechanismWitness {
    pub boundary_ids: Vec<BoundaryId>,
    pub executable_hashes: Vec<Digest>,
    pub process_instances: Vec<ProcessWitness>,
    pub db_transactions: Vec<DbWitness>,
    pub nats_deliveries: Vec<NatsWitness>,
    pub lineage: Vec<LineageWitness>,
    pub replay_operations: Vec<ReplayWitness>,
    pub fault_activations: Vec<FaultWitness>,
    pub overlap_intervals: Vec<IntervalWitness>,
    pub lane_authorities: Vec<LaneAuthorityWitness>,
    pub vm_scenarios: Vec<VmWitness>,
}
```

Witness constructors must be sealed inside the production boundary or harness component they attest. A test cannot promote itself by writing `"process": true` into arbitrary JSON. Witness artifacts use the existing test-artifact import path and are materialized into the history DB; `trace_events` is unsuitable because its channel is intentionally droppable and its installation is not fail-closed.

Observed levels are:

| Level | Required observed mechanism |
|---|---|
| Static | Catalog/schema/source analysis only; no executed test |
| Unit | Direct pure function or method invocation; no production substrate boundary |
| Component | One production component, adapter, repository, parser, DB, or NATS component exercised in-process with a real isolated dependency where relevant |
| Pipeline | At least two registered production stages connected through real DB/NATS boundaries, with one correlated material/event/operation lineage |
| Process | A distinct production executable PID and executable hash, real IPC/NATS/DB boundary, startup/readiness and shutdown or crash lifecycle |
| VM | An exported xtask VM scenario plus guest/service/process witnesses from inside the VM |

The rank is only the envelope. Every obligation also has a `MechanismContract`. Examples:

- A source input-boundary pipeline claim requires source factory or adapter activity, captured material identity, material publication/consumption, parser/admission activity, and persisted or published event lineage.
- Replay remint requires an actual archive/replay operation, the same occurrence identity, a fresh interpretation identity, and proof that the superseded interpretation is no longer live. Parsing the same bytes twice cannot satisfy it.
- Isolation requires overlapping execution intervals or a barrier, distinct lane/authority identities, a perturbation on one side, and an asserted absence of cross-lane mutation. Sequential calls cannot satisfy it.
- Privacy requires a secret canary to cross the admission/redaction boundary, followed by inspection of persisted, published, and diagnostic outputs. Privacy declarations alone cannot satisfy it.
- Crash settlement requires an activated kill/fault hook, process or consumer interruption, restart/rebind evidence, and a checked settlement frontier. A green test with `fault_hook_hits == 0` is false-passing.
- Process claims require a PID distinct from the test runner. An in-process Tokio task remains pipeline at most.
- VM claims require the exported VM category/scenario and guest evidence; invoking a Nix helper on the host does not qualify.

An `Overstated` result is emitted when `declared_level > observed_level`. A `FalsePassing` result is emitted when the envelope is sufficient but the obligation-specific predicate or falsifier is absent.

This distinction catches the current source-test problem mechanically. The shared harness calls parser dispatch directly for “initial ingestion,” performs two independent parser dispatches for “replay,” executes “isolation” sequentially, and checks privacy metadata rather than admission redaction. Evidence: `crate/sinexd/tests/sources/production_path/obligations/initial_ingestion.rs:12-17,48-65`; `replay.rs:13-18,35-45`; `isolation.rs:1-12,35-62`; `privacy.rs:11-15,51-74`.

---

# 3. Gap compiler and `xtask test gaps`

## 3.1 Command placement and data flow

Add:

```rust
TestSubcommand::Gaps(GapsArgs)

pub struct GapsArgs {
    #[arg(long)]
    pub area: Option<String>,

    #[arg(long)]
    pub json: bool,
}
```

The current `TestSubcommand` contains bench, fuzz, coverage, mutants, and VM only: `xtask/src/commands/test.rs:296-320 — TestSubcommand`; dispatch is centralized at `1602-1621 — TestCommand::execute`.

The supported frontend is:

```text
xtask test gaps
xtask test gaps --area sources
xtask test gaps --json
```

The flow is:

```text
compiled registries/deployment descriptors
        ↓
normalized subject catalog
        ↓
derived obligations + surface fingerprints
        ↓
source claim scan + Cargo module graph + current nextest list
        ↓
current claims and reachability
        ↓
history DB: results, manifests, proof units, seals, regions, audits
        ↓
mechanism evaluation
        ↓
CoverageGap report
```

Because `xtask` cannot directly depend on `sinexd` without creating a dependency cycle, `xtask test gaps` should invoke a hidden, offline `sinexd registry-export --format json` path internally. That path must stop before DB/NATS/runtime initialization and export automata, sources, schema, RPC, and typed NATS descriptors. It is not a user-facing cargo command, and no evidence is stored outside the existing history DB.

The command performs these steps:

1. Open the same configured history database read/write.
2. Invoke the offline registry exporter.
3. Normalize and sort descriptors; derive obligation IDs, activation, mechanism contracts, and fingerprints.
4. Scan test-bearing files and resolve Cargo/module reachability.
5. obtain the current exact nextest list and map current claims to executable identities.
6. Persist the catalog snapshot and claim resolution.
7. Join the latest matching test result, execution manifest, proof unit, substrate seal, coverage regions, and impact audit.
8. Evaluate observed proof level and obligation-specific mechanism predicates.
9. Persist claim evaluation and render the deterministic report.

`--area X` filters obligations first and then retains only claims, modules, and evidence connected to those obligations. JSON has a versioned top-level schema and is sorted by gap class, obligation ID, and claim ID.

Exit status:

```text
0  no blocking gaps; valid waivers may be present
1  one or more blocking gaps
2  catalog/export/history corruption or command failure
```

There is no `coverage_gaps` table. Gaps are derived from current catalog plus current evidence, preventing stale cached diagnoses from becoming another source of truth.

## 3.2 Exact new history tables

All tables remain in `xtask-history.db`.

```sql
CREATE TABLE coverage_catalog_runs (
    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    invocation_id              INTEGER REFERENCES invocations(id),
    catalog_version            TEXT NOT NULL,
    tree_fingerprint           TEXT NOT NULL,
    registry_fingerprint       TEXT NOT NULL,
    nextest_list_fingerprint   TEXT NOT NULL,
    area_filter                TEXT,
    status                     TEXT NOT NULL,
    diagnostics_json           TEXT NOT NULL,
    started_at                 TEXT NOT NULL,
    finished_at                TEXT
);
```

```sql
CREATE TABLE coverage_obligations (
    catalog_run_id             INTEGER NOT NULL REFERENCES coverage_catalog_runs(id),
    obligation_id              TEXT NOT NULL,
    area                       TEXT NOT NULL,
    registry_kind              TEXT NOT NULL,
    subject_key                TEXT NOT NULL,
    family                     TEXT NOT NULL,
    variant_key                TEXT,
    active                     INTEGER NOT NULL,
    minimum_level              TEXT NOT NULL,
    surface_fingerprint        TEXT NOT NULL,
    mechanism_contract_json    TEXT NOT NULL,
    source_file                TEXT NOT NULL,
    source_line                INTEGER,
    metadata_json              TEXT NOT NULL,
    PRIMARY KEY (catalog_run_id, obligation_id)
);
```

```sql
CREATE TABLE coverage_claims (
    catalog_run_id             INTEGER NOT NULL REFERENCES coverage_catalog_runs(id),
    claim_id                   TEXT NOT NULL,
    obligation_id              TEXT NOT NULL,
    declared_level             TEXT NOT NULL,
    package                    TEXT NOT NULL,
    test_target                TEXT,
    nextest_test_id            TEXT,
    module_path                TEXT NOT NULL,
    source_file                TEXT NOT NULL,
    source_line                INTEGER NOT NULL,
    declaration_fingerprint    TEXT NOT NULL,
    reachable                  INTEGER NOT NULL,
    reachability_reason        TEXT,
    PRIMARY KEY (catalog_run_id, claim_id)
);
```

```sql
CREATE TABLE coverage_claim_results (
    invocation_id                  INTEGER NOT NULL REFERENCES invocations(id),
    catalog_run_id                 INTEGER NOT NULL REFERENCES coverage_catalog_runs(id),
    claim_id                       TEXT NOT NULL,
    obligation_id                  TEXT NOT NULL,
    test_result_id                 INTEGER REFERENCES test_results(id),
    execution_manifest_id          INTEGER REFERENCES test_execution_manifests(id),
    proof_unit_id                  INTEGER REFERENCES test_proof_units(id),
    obligation_surface_fingerprint TEXT NOT NULL,
    substrate_seal_fingerprint     TEXT,
    observed_level                 TEXT NOT NULL,
    mechanism_evidence_json        TEXT NOT NULL,
    falsifier_status               TEXT NOT NULL,
    evaluation_status              TEXT NOT NULL,
    evaluated_at                   TEXT NOT NULL,
    PRIMARY KEY (invocation_id, claim_id)
);
```

```sql
CREATE TABLE coverage_waivers (
    catalog_run_id          INTEGER NOT NULL REFERENCES coverage_catalog_runs(id),
    waiver_id               TEXT NOT NULL,
    obligation_id           TEXT NOT NULL,
    owner                   TEXT NOT NULL,
    reason                  TEXT NOT NULL,
    issued_at               TEXT NOT NULL,
    expires_at              TEXT NOT NULL,
    surface_fingerprint     TEXT NOT NULL,
    source_file             TEXT NOT NULL,
    source_line             INTEGER NOT NULL,
    PRIMARY KEY (catalog_run_id, waiver_id)
);
```

Waiver declarations may live in a small checked-in `xtask/config/coverage-waivers.toml` or typed source declaration. That is review policy, not an evidence store; each catalog run materializes it into SQLite.

## 3.3 Exact changes to existing tables

`test_results`:

- Add `nextest_test_id TEXT`.
- Add `test_target TEXT`.
- Move uniqueness to `(invocation_id, nextest_test_id, attempt)`, retaining `test_name` as display text.

`test_execution_manifests`:

- Add `test_target TEXT`.
- Add `nextest_test_id TEXT`.
- Add `executable_hash TEXT`.
- Add `process_role TEXT`.
- Add `claim_ids_json TEXT NOT NULL DEFAULT '[]'`.
- Make artifact filenames use the exact test/claim identity rather than only the sanitized bare function name.

This addresses the current artifact’s dependence on `stringify!(fn_name)` and file/module metadata: `xtask/src/sandbox/context.rs:1517-1535 — persist_test_execution_manifest`.

`coverage_regions`:

- Add `catalog_run_id INTEGER`.
- Add `claim_id TEXT`.
- Add `obligation_id TEXT`.
- Add `surface_fingerprint TEXT`.
- Add `coverage_tool_fingerprint TEXT`.
- Change its identity index to include `(invocation_id, claim_id, file_path, line_start, line_end, region_hash)`.

`impact_audit_runs`:

- Add `catalog_run_id INTEGER`.
- Add `planner_version TEXT`.
- Add `audit_kind TEXT`, with values such as `current_tree_oracle` and `historical_replay`.
- Add `tree_fingerprint TEXT`.
- Add `nextest_list_fingerprint TEXT`.
- Add `selected_tests_json TEXT`.
- Add `oracle_tests_json TEXT`.
- Add `false_negatives_json TEXT`.
- Add `obligation_ids_json TEXT`.
- Record the actual sampled decision count, not the requested `--sample-skips` value.

`proof_evidence` and `test_proof_units` do not require a parallel proof table. The substrate seal from `9es` is folded into their `input_fingerprint`/manifest and copied into `coverage_claim_results`.

`trace_events` is not used as required proof evidence because its documented architecture permits dropping events: `xtask/src/history/tracing_layer.rs:3-13`.

## 3.4 Populating `coverage_regions`

The existing `xtask impact seed-coverage` remains the writer but gains obligation-aware sampling. Its current exact-test LLVM implementation is reusable: `xtask/src/commands/impact.rs:138-247 — seed_coverage`; region extraction at `360-423 — coverage_regions_from_llvm_json`.

The sampling policy is deterministic and time-budgeted rather than test-count-driven:

1. Select only reachable claims whose current obligation and declaration fingerprints are known.
2. Prioritize:
   - changed, newly introduced, or stale obligation surfaces;
   - replay, privacy, settlement, authority, and resource obligations;
   - claims selected by the current impact plan;
   - obligations never region-seeded;
   - oldest matching region evidence.
3. For an obligation with multiple claims, choose the cheapest recent successful claim capable of the required mechanism.
4. Rotate equal-priority obligations by a hash of tree fingerprint and obligation ID.
5. Stop at a predicted wall-time budget, for example ten minutes in a local/pre-merge sampling lane and sixty minutes in the nightly lane.
6. Run each selected exact nextest ID under `cargo llvm-cov nextest --json` internally through xtask.
7. Attach each imported region to claim ID, obligation ID, current surface fingerprint, and LLVM/toolchain fingerprint.

One covered region can support impact mapping and source reachability, but it does not prove replay, privacy, settlement, or differential semantics. Unseeded region evidence is reported as `unknown`, not as a missing behavioral claim.

## 3.5 Populating `impact_audit_runs`

The current audit is not a completeness oracle. It samples decisions, runs a broadened current-tree command, and represents any command failure as one false negative: `xtask/src/commands/impact.rs:249-312 — audit`. It may also return `no_sample` as success when no sampled decisions exist: `272-292`.

The upgraded audit should:

1. Freeze tree fingerprint, catalog fingerprint, planner version, and current nextest-list fingerprint.
2. Resolve the exact impact-selected test IDs.
3. Resolve an immutable oracle set—affected package closure or workspace/full set according to risk class.
4. Run selected and oracle sets on the same tree and substrate seal.
5. Compare exact current test identities and outcomes.
6. Record a false negative for every oracle failure absent from the selected set, not merely for command failure.
7. Treat a selected filter that resolves to zero current tests as false-passing and invalidate reuse.
8. Optionally replay historical diffs where the relevant Git object and substrate are available; record `historical_replay` separately. Absence of historical material is `unknown`, not success.

An impact-mode claim is current only when its latest matching `9nl` completeness/liveness audit has no false negatives. This gives `sinex-9nl` a concrete DB output rather than a separate bookkeeping mechanism.

---

# 4. Anti-vacuity harness

The classifier, history migrations, registry fixture exporter, CLI renderer, JSON renderer, and exit status must be exercised together against temporary SQLite databases. Unit-testing only a `classify()` function is insufficient.

Each fixture must first produce the expected red or waived report and then be mutated to prove that the diagnosis clears.

| Fixture | Deliberate evidence state | Required report | Clearing mutation |
|---|---|---|---|
| Missing | Active obligation, no claim and no waiver | One blocking `Missing`, reason `NO_REACHABLE_CURRENT_CLAIM`; exit 1 | Insert a reachable, current, successful claim result; expected count becomes zero |
| Valid waiver | Active obligation, no claim, unexpired waiver with matching surface fingerprint | One nonblocking `Waived`, reason `VALID_FINGERPRINT_BOUND_WAIVER`; exit 0 | Expire the waiver or change the obligation fingerprint; result becomes blocking stale/missing and exit 1 |
| Stale claim | Claim result records surface fingerprint A; current catalog emits B | One blocking `Stale`, reason `OBLIGATION_SURFACE_CHANGED`, with expected B and observed A; exit 1 | Execute and insert evidence against B; stale result disappears |
| Overstated level | Claim declares `Process`; witnesses show one same-process adapter/parser component, no child PID or IPC boundary | One blocking `Overstated`, reason `DECLARED_LEVEL_EXCEEDS_OBSERVED`, declared `process`, observed `component`; exit 1 | Add an actual child executable, distinct PID/hash, readiness and IPC witnesses; result clears |
| Unreachable module | A file contains a claim/test, but no Cargo root or recursive `mod`/`#[path]` edge reaches it and current nextest does not list it | One blocking `Unreachable`, reason `MODULE_NOT_IN_BUILD_GRAPH`; exit 1 | Add the module edge and current nextest identity; result clears |
| False-passing fault claim | Test result is green but the required fault hook has zero activations and no settlement frontier | One blocking `FalsePassing`, reason `FALSIFIER_NOT_OBSERVED`; exit 1 | Activate the fault, record interruption/recovery and settlement witnesses; result clears |

For every fixture, assert:

- exact human summary counts;
- exact JSON `schema_version`, class, reason code, IDs, declared/observed levels, and blocking flag;
- exact exit status;
- no duplicate secondary diagnosis for the same claim;
- the red-to-green or waiver-to-expired transition.

The reachability fixture must use a real temporary Rust module tree and the production module resolver. The false-passing fixture must use a real fault-hook counter. The stale fixture must change a normalized registry descriptor, not manually set a Boolean called `stale`.

---

# 5. False-confidence inventory

## 5.1 Shared mechanism behind the broad cases

`ProductionPathCase::new` assigns all four named obligations—initial ingestion, replay, isolation, and privacy—to every default case: `crate/sinexd/tests/sources/production_path.rs:41-52,86-112 — ALL_OBLIGATIONS/ProductionPathCase::new`.

The actual shared mechanism is:

- initial ingestion: parser registration plus direct `default_parser_dispatch`;
- replay: two direct dispatches with two newly generated material IDs, comparing event types;
- isolation: bad subject dispatch followed by two sequential peer dispatches;
- privacy: clean parser dispatch plus parser privacy-context declaration.

Evidence: `crate/sinexd/tests/sources/production_path/obligations/initial_ingestion.rs:42-90`; `replay.rs:29-82`; `isolation.rs:24-70`; `privacy.rs:32-75`.

These are useful component-level parser contract checks. They may discharge registration, parser input acceptance, deterministic repeatability, sequential registry recovery, and metadata-declaration claims. They must not discharge pipeline ingestion, archive/replay remint, concurrent isolation, or runtime redaction.

## 5.2 Thirty-three default `ProductionPathCase` overclaims

| File | Tests |
|---|---|
| `crate/sinexd/tests/sources/production_path/ai_session_test.rs:60-77` | `ai_session_claude_obligations`; `ai_session_chatgpt_obligations` |
| `crate/sinexd/tests/sources/production_path/browser_test.rs:19-51` | `browser_history_qutebrowser_initial_ingestion`; `browser_history_chromium_initial_ingestion`; `browser_history_jsonl_dump_initial_ingestion` |
| `crate/sinexd/tests/sources/production_path/desktop_test.rs:26-69,226` | `desktop_activitywatch_window_obligations`; `desktop_activitywatch_afk_obligations`; `desktop_activitywatch_web_obligations`; `desktop_clipboard_obligations` |
| `crate/sinexd/tests/sources/production_path/document_test.rs:15-54` | `document_staging_obligations`; `docs_library_index_directory_entry_obligations` |
| `crate/sinexd/tests/sources/production_path/email_test.rs:29-89` | `email_rfc822_drop_obligations`; `email_maildir_entry_obligations`; `email_mbox_slice_obligations` |
| `crate/sinexd/tests/sources/production_path/export_parsers_test.rs:48-83` | `raindrop_bookmarks_obligations`; `spotify_extended_history_obligations`; `hledger_journal_obligations`; `facebook_messenger_thread_obligations` |
| `crate/sinexd/tests/sources/production_path/health_exports_test.rs:6-14` | `sleep_merged_summary_obligations` |
| `crate/sinexd/tests/sources/production_path/media_test.rs:76-104` | `media_audio_transcript_obligations`; `media_screen_ocr_obligations` |
| `crate/sinexd/tests/sources/production_path/path_sensitive_test.rs:15-52` | `knowledgebase_vault_obligations`; `git_commit_history_obligations` |
| `crate/sinexd/tests/sources/production_path/social_exports_test.rs:17-52` | `reddit_gdpr_comments_obligations`; `reddit_gdpr_posts_obligations`; `wykop_entries_obligations`; `wykop_entry_comments_obligations` |
| `crate/sinexd/tests/sources/production_path/terminal_test.rs:31-149` | `terminal_atuin_history_obligations`; `terminal_bash_history_obligations`; `terminal_zsh_history_plain_obligations`; `terminal_zsh_history_extended_obligations`; `terminal_text_history_obligations`; `terminal_fish_history_obligations` |

**Claimed:** all four broad production-path mechanisms.

**Actual:** component parser registration/dispatch; repeated parsing; sequential peer health; privacy metadata.

**Action:** immediately annotate these at `Component` level against honest registration, input-boundary, repeatability, and metadata obligations. Rename the three browser `_initial_ingestion` tests to `_parser_dispatch` or `_parser_contract`. Leave the true pipeline/replay/isolation/redaction obligations missing until shared higher-level scenarios exist. Strengthen one generated scenario per mechanism family rather than reproducing a full pipeline test per source.

## 5.3 Additional named overclaims

| Test and evidence | Claimed mechanism | Actual mechanism | Demotion or strengthening |
|---|---|---|---|
| `desktop_window_manager_obligations`, `crate/sinexd/tests/sources/production_path/desktop_test.rs:294-321` | Broad ingestion, replay, isolation, privacy obligations | Unix-socket adapter/parser fixture; “replay” is a second freshly built fixture parse; shared sequential isolation; privacy metadata | Keep adapter/parser claim at `Component`. Rename to `desktop_window_manager_adapter_parser_contract`; add separate archive/replay, concurrent-lane, and admission-redaction scenarios |
| `test_system_journald_initial_ingestion`, `test_system_systemd_initial_ingestion`, `test_system_dbus_initial_ingestion`, `test_desktop_notification_initial_ingestion`, `test_desktop_notification_action_initial_ingestion`, `test_desktop_notification_closed_initial_ingestion`, `test_system_udev_initial_ingestion`; `crate/sinexd/tests/sources/production_path/system.rs:45-151` | Initial ingestion | Calls the same direct parser-dispatch helper | Rename to `_parser_dispatch` and claim component input boundaries. Strengthen through real journal/DBus/file-drop adapter, material stream, parser, admission, and persistence lineage |
| `weechat_message_canary`, `crate/sinexd/tests/sources/production_path/obligations/initial_ingestion_canary.rs:8-24` | Documented as “Wave A end-to-end integration” | Calls `initial_ingestion::run`, which directly dispatches the parser | Demote and rename to parser-dispatch canary, or replace with the existing binary/NATS/DB pattern before claiming pipeline/process |
| `terminal_asciinema_session_json_ingestion`, `crate/sinexd/tests/sources/production_path/terminal_test.rs:337-395` | Ingestion evidence | Direct construction of `AsciinemaParser` and direct `parse_record` | Rename to `terminal_asciinema_session_json_parser_contract`; claim `Unit` or `Component`, not ingestion pipeline |
| `messenger_replay_produces_identical_occurrence_keys`, `crate/sinexd/tests/sources/messaging_parser_test.rs:385-412` | Replay | Calls the parser twice on the same bytes and compares occurrence keys | Rename to `messenger_reparse_preserves_occurrence_keys`; a replay-remint claim requires archive/replay control and fresh/superseded interpretation evidence. This file is also statically unreachable |
| `rfc822_drop_replay_preserves_occurrence_identity`, `crate/sinexd/tests/sources/email_mailbox_parser_test.rs:173-210` | Replay and archive/replay relationship | Constructs two parser instances and parses the same staged bytes twice | Rename to `rfc822_drop_reparse_preserves_occurrence_identity`; keep as deterministic identity evidence and add archive/replay differential separately |

The resulting inventory is **45 overclaiming names**: 33 default broad cases plus 12 additional names.

## 5.4 Current unreachable test modules

The `v7od` package comments say the test tree has been consolidated into aggregated roots: `crate/sinexd/Cargo.toml:148-153`. The current roots are `tests/api.rs`, `tests/automata.rs`, `tests/event_engine.rs`, `tests/runtime.rs`, and `tests/sources.rs`.

The static module graph exposes ten test-bearing files with no inbound root/module edge:

```text
crate/sinexd/tests/event_engine/wal_integrity_test.rs
crate/sinexd/tests/event_engine/test_utils_integration/configuration_test.rs
crate/sinexd/tests/event_engine/test_utils_integration/e2e_source_to_db_test.rs
crate/sinexd/tests/event_engine/test_utils_integration/event_processing_integration_test.rs
crate/sinexd/tests/event_engine/test_utils_integration/stream_processing_test.rs
crate/sinexd/tests/sources/ai_session_parser_test.rs
crate/sinexd/tests/sources/bookmark_parser_test.rs
crate/sinexd/tests/sources/finance_parser_test.rs
crate/sinexd/tests/sources/messaging_parser_test.rs
crate/sinexd/tests/sources/music_parser_test.rs
```

The event-engine root lists its modules at `crate/sinexd/tests/event_engine.rs:13-38` and omits the first five. The sources root lists its modules at `crate/sinexd/tests/sources.rs:19-30` and omits the latter five. Test attributes are present in the omitted files, including:

- `wal_integrity_test.rs:12`
- `configuration_test.rs:788`
- `e2e_source_to_db_test.rs:129`
- `event_processing_integration_test.rs:15`
- `stream_processing_test.rs:31`
- `ai_session_parser_test.rs:63`
- `bookmark_parser_test.rs:82`
- `finance_parser_test.rs:93`
- `messaging_parser_test.rs:155`
- `music_parser_test.rs:64`

`sinex-usgn` independently documents the five event-engine files as genuinely dead and requires nextest-list verification after repair: `sinex-beads-export.jsonl:227 — sinex-usgn`. The five source-parser files are additional static findings in the supplied working tree.

The compiler should therefore seed ten blocking `Unreachable` gaps unless current dynamic discovery disproves the static result.

---

# 6. Rollout sequence

## Slice 1: truthful catalog and reachability

Implement only what is necessary to produce a real, mostly-red report:

1. Add stable obligation/claim IDs and the catalog tables.
2. Add the offline registry exporter for the five existing registry surfaces.
3. Derive and activate:
   - registration obligations for every registered subject;
   - source component input-boundary obligations;
   - synthetic test-module reachability obligations.
4. Extend `#[sinex_test]` with explicit claims and emit claim IDs in execution manifests.
5. Build the Cargo/module/nextest reachability compiler.
6. Add the six anti-vacuity fixtures.
7. Run `xtask test gaps --json` without requiring coverage regions, impact audits, trace events, a live DB, or live NATS.
8. Record region, live deployment, and substrate evidence as `unknown` where unavailable.

Annotate a small representative set of existing tests at their honest component level. Do not bulk-assign claims from test names. The expected first result is red because most obligations have no stable claim and at least ten test modules are statically unreachable.

## Slice 2: boundaries and consumers

Activate:

- source factory/adapter/parser input boundaries;
- RPC JSON/auth request and response contracts;
- schema repository/read-write boundaries;
- JetStream stream and durable-consumer contracts;
- automaton confirmed-event consumer registration.

This slice should also move NATS literal shapes into shared typed constructors and retain `RpcMethodInfo` in the runtime registry.

## Slice 3: incremental, rebuild, and invalidation

Activate:

- incremental behavior;
- bounded rebuild;
- revision/invalidation;
- checkpoint and restart behavior.

Use property/state-machine scenarios, not one test per registry branch. `sinex-pdq5` supplies interval ordering, duplicate, reordering, restart, replay-redelivery, and invariant scenarios: `sinex-beads-export.jsonl:105 — sinex-pdq5`.

## Slice 4: differential and replay remint

Activate live-vs-rebuild differential and replay-remint obligations after the mechanism witnesses exist. `sinex-dtw5` becomes the shared occurrence-keyed whole-derived-plane claim producer instead of inventing a separate result format: `sinex-beads-export.jsonl:92 — sinex-dtw5`.

Parser-repeatability tests remain component evidence and do not upgrade automatically.

## Slice 5: settlement, metamorphics, isolation, and budgets

Activate:

- crash settlement;
- ordering/batching/restart metamorphics;
- lane/authority isolation;
- resource budgets.

`sinex-r6d.9` supplies kill-window process/settlement claims: `sinex-beads-export.jsonl:47 — sinex-r6d.9`.

`sinex-uz9d` supplies error-return NATS/DB fault schedules and must emit fault-hit plus settlement witnesses: `sinex-beads-export.jsonl:90 — sinex-uz9d`.

A test that passes without activating its selected schedule is `FalsePassing`.

## Slice 6: confidence in selection and reuse

Add:

- obligation-aware bounded LLVM sampling;
- selected-vs-oracle impact audits;
- current-test liveness/completeness tokens;
- substrate seal validation;
- proof-reuse linkage.

Only after the anti-vacuity suite proves sensitivity should `sinex-pke` consume the blocking gap summary as a gate.

## No derivation-kernel dependency

No `DerivationDefinition` symbol was found in the supplied working tree. The coverage compiler should attach directly to `AutomatonSpec`, source contracts/bindings, schema defs, `RpcMethod` registrations, and typed NATS topology now.

A future `DerivationDefinition` may implement the same obligation-subject export interface if it later becomes a real authoritative registry. It must not block this work or become a prerequisite merely to host coverage metadata.

---

# 7. Bead surgery sketch

Create one umbrella bead only:

**Title:** `Coverage-obligation compiler and truthful xtask test gaps`

**Acceptance criteria:**

```text
xtask test gaps --json derives a versioned catalog from the existing
automaton, source, schema, RPC, NATS, and test-module surfaces; resolves
stable claim IDs against the current build/nextest graph; reports missing,
stale, waived, overstated, unreachable, and false-passing classes from the
existing history DB; includes red-to-green anti-vacuity fixtures for every
class; and attaches the first mostly-red report. LLVM regions and historical
impact evidence may be unknown in the first report.
```

Do not create one bead per obligation family.

Reshape the existing beads as follows:

- **`sinex-9es` — proof substrate seal.** It owns the normalized environment, rustc/cargo/nextest versions, executable/tool identity, live schema digest, NATS topology digest, and preflight identity that enter claim freshness and proof reuse. A changed or missing seal makes pipeline/process/VM evidence stale; it must never become an independent evidence ledger. Its current defect statement explicitly names omitted DB/NATS/env/toolchain inputs and reuse before preflight: `sinex-beads-export.jsonl:134 — sinex-9es`.

- **`sinex-9nl` — impact evidence soundness.** It owns current nextest liveness, package completeness tokens, zero-selection failure, selected-vs-oracle audit, and risk-class fallback. Its output populates the strengthened `impact_audit_runs`; the gap compiler consumes it. The bead already identifies historical dead names and zero-test proofs as a false-skip class: `sinex-beads-export.jsonl:133 — sinex-9nl`.

- **`sinex-pke` — gate wiring.** It does not implement another classifier. After anti-vacuity is proven, it consumes `xtask test gaps --json` and blocks only on blocking classes for activated obligation families. Its purpose is already to connect disconnected verification machinery to enforced gates: `sinex-beads-export.jsonl:309 — sinex-pke`.

- **`sinex-uz9d` — fault evidence producer.** Every deterministic fault schedule declares claims against crash-settlement and consumer-contract obligations and emits a sealed fault-activation witness plus the resulting settlement frontier. A schedule with zero activations becomes false-passing. `sinex-beads-export.jsonl:90 — sinex-uz9d`.

The other beads attach without surgery:

- `sinex-dtw5`: live/rebuild differential claims.
- `sinex-r6d.9`: crash-window settlement claims.
- `sinex-pdq5`: ordering/batching/restart property claims.
- `sinex-v7od`: aggregation changes test locators, not claim IDs; its target is no more than five binaries. `sinex-beads-export.jsonl:83 — sinex-v7od`.
- `sinex-usgn`: seeds the first reachability gaps.
- `sinex-jdp`: remains a product capture-losslessness measure, not the denominator for test-proof coverage. Its missing-material, missing-event, unobserved-binding, bridge, and stale-source classes may themselves have testing obligations, but the live capture metric is not proof evidence. `sinex-beads-export.jsonl:34 — sinex-jdp`.

# Claims I could not verify from the package

1. The actual history database was not included. The approximately 1.9 GB size and row counts are verified only as statements in `sinex-scratchpad.xml:35932-35936,36953-36958`, not by a direct SQLite query.

2. Counts for `proof_evidence`, `impact_runs`, and `impact_decisions` are unknown. Their schema and writers exist, but the scratch count block does not enumerate them.

3. The package environment did not provide a usable Cargo/Rust toolchain for this review. I could not compile the current tree, execute `cargo nextest list`, or run `xtask`.

4. The ten orphan files are a static module-graph result. `sinex-usgn` independently confirms the five event-engine files; dynamic nextest absence for the five additional source-parser files is unknown until the current tree compiles and lists successfully.

5. Whether all pre-consolidation tests remain compiled under the current `sinex-v7od` working tree is unknown. The aggregate roots are present, but a complete before/after test identity comparison was unavailable.

6. The exact compiled counts of source contracts, live source runtime bindings, schema tables, RPC methods, streams, and consumers are unknown because no registry exporter currently exists and the tree could not be built. The count of 16 automata is directly verified from `AUTOMATA`.

7. The active `SourceBindingsManifest`, `SINEX_AUTOMATA_ENABLED` value, live schema digest, and externally managed NATS topology were not present. Which registered subjects are actually deployed is therefore unknown.

8. The current source appears to fix the default-verbosity trace suppression path, but whether that change has produced any `trace_events` rows is unknown. `init_tracing()` still ignores subscriber installation failure.

9. No successful invocation of `xtask impact seed-coverage` or `xtask impact audit` is evidenced in the package. Their writers exist, but the audited store was recorded with zero corresponding rows.

10. Existing test executions do not emit the structured mechanism witnesses required to classify unit/component/pipeline/process/VM levels. The proof-level assignments in the false-confidence inventory are source-mechanism classifications, not measurements imported from history.

11. Historical selected-vs-full impact false-negative rates are unavailable. The current audit command does not compute them.

12. The DB/NATS/environment/toolchain substrate seal required by `sinex-9es` is not implemented in the supplied source, so no existing proof unit can be verified as fresh against those dimensions.

---

