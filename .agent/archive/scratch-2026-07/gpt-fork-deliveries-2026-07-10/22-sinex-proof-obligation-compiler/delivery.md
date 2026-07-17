## Turn 122 (role=assistant)

# Sinex Proof Obligation Compiler delivered

I treated the attached mission as an implementation contract and completed the working compiler, executed assessment, anti-vacuity campaign, patch reconstruction, validation, planning program, and parallel follow-up package.

**[Download the complete ZIP package](sandbox:/mnt/data/Sinex-proof-obligation-compiler-kit.zip)**  
[ZIP checksum](sandbox:/mnt/data/Sinex-proof-obligation-compiler-kit.zip.sha256)  
[Download the deterministic tar.gz package](sandbox:/mnt/data/Sinex-proof-obligation-compiler-kit.tar.gz)  
[tar.gz checksum](sandbox:/mnt/data/Sinex-proof-obligation-compiler-kit.tar.gz.sha256)  
[Package statistics](sandbox:/mnt/data/Sinex-proof-obligation-compiler-kit-stats.json)

The best entry points are:

[Start here](sandbox:/mnt/data/proof-obligation-compiler-kit/00-START-HERE.md)  
[Interactive package index](sandbox:/mnt/data/proof-obligation-compiler-kit/index.html)  
[Executive assessment](sandbox:/mnt/data/proof-obligation-compiler-kit/01-executive-assessment.md)  
[Executed proof dashboard](sandbox:/mnt/data/proof-obligation-compiler-kit/reports/dashboard.html)  
[Validation ledger](sandbox:/mnt/data/proof-obligation-compiler-kit/validation/VALIDATION.md)

## What was implemented

The supplied Sinex snapshot was on `feature/fix/cloud-harness-bootstrap` at `5e788c31`, with a pre-existing dirty working tree and a Rust-first `xtask` developer surface. I preserved that boundary rather than quietly absorbing the unrelated branch delta.

The implementation introduces:

- a portable proof engine behind `python3 -m devtools lab proof-obligations`;
- a bridge through `cargo xtask verify obligations`;
- typed claims, obligations, witnesses, probes, executions, and verdicts;
- explicit proof-strength ceilings;
- separate required and actually reached environments;
- baseline-first mutation execution;
- a known-vacuous control;
- meaningful killed and surviving weakening probes;
- one isolated scratch tree per run rather than copying the repository for every mutation;
- before, changed, and restored target hashes;
- distinct killed, survived, baseline-failed, and infrastructure-error outcomes;
- deterministic structured and rendered reports;
- focused tests and patch-reconstruction verification.

The compiler is deliberately unable to promote a portable structural predicate into native Rust or live-service proof merely because the stronger environment was unavailable.

## Principal patches

There are two patch boundaries because the original snapshot was dirty.

**[Compiler-only patch relative to the supplied snapshot](sandbox:/mnt/data/proof-obligation-compiler-kit/patches/01-compiler-only-against-supplied-snapshot.patch)**

This contains only the proof-obligation compiler delta. It should be applied only to a tree that already matches the supplied dirty snapshot.

**[Complete final state relative to clean commit `5e788c31`](sandbox:/mnt/data/proof-obligation-compiler-kit/patches/00-full-state-from-5e788c31.patch)**

This contains both the pre-existing cloud-harness changes and the compiler, allowing the complete state to be reconstructed from an ordinary clean Git checkout.

Both patches were independently subjected to `git apply --check`, applied in reconstructed worktrees, and checked with `git diff --check`. The receipts are in the [patch-application report](sandbox:/mnt/data/proof-obligation-compiler-kit/validation/patch-application.txt).

Review-oriented slices are under [patches/by-layer](sandbox:/mnt/data/proof-obligation-compiler-kit/patches/by-layer/), and the final changed files are also available as a [source overlay](sandbox:/mnt/data/proof-obligation-compiler-kit/source-overlay/README.md).

## What the first full compile found

The first complete inventory contained **81 non-control obligations**:

- 70 behavior-proven;
- six capability-only;
- five mandatory obligations still red before probing.

The MCP adapter discovered all 68 exposed tools. It located direct fixture-backed behavioral witnesses for 64 and retained four as catalog-only rather than claiming behavioral coverage from registration alone.

The first three meaningful repository findings were:

### Closure evidence can be declarative rather than executable

A closure artifact can contain a command-shaped value and a satisfied state without proving that the command exists, ran successfully, ran at the relevant revision, or was relevant to the acceptance criterion.

The compiler now treats command declaration, resolution, execution, revision binding, and criterion relevance as separate evidence states.

### MCP catalogs and dispatch can drift

A catalog declaration can be weakened to advertise a different read-only RPC while a narrower dispatch test remains green. Tool presence, schema presence, authority classification, advertised target, and actual dispatch behavior therefore need distinct obligations.

### Demo repeatability is not demo construct validity

The focused demo test establishes random-number-generator repeatability. It does not establish that all five claimed event families are generated, persisted, independently queryable, correctly related, and exposed through the public product path.

The full remediation analysis is in [Real findings and remediation](sandbox:/mnt/data/proof-obligation-compiler-kit/05-real-findings-and-remediation.md).

## The evidence model

The compiler is not a test lister or a coverage-percentage generator. It compiles this chain:

```text
claim
  → falsifiable obligation
  → candidate witness
  → witness mechanism and environment
  → baseline receipt
  → bounded weakening probe
  → killed, survived, or invalid result
  → strength-limited verdict and caveats
```

A killed mutation shows that the named witness detects that particular weakening. It does not establish universal correctness.

A surviving meaningful mutation exposes a witness blind spot. It does not by itself establish a current production bug.

A surviving known-vacuous mutation is required as a positive control for the mutation infrastructure. A runner that kills every mutation—including irrelevant ones—is suspicious rather than impressive.

The complete model is documented in [Proof-obligation semantics](sandbox:/mnt/data/proof-obligation-compiler-kit/02-proof-obligation-semantics.md). Machine contracts are included for the [proof IR](sandbox:/mnt/data/proof-obligation-compiler-kit/schemas/proof-obligation-ir.schema.json), [run packet](sandbox:/mnt/data/proof-obligation-compiler-kit/schemas/proof-run-packet.schema.json), and [mutation operator](sandbox:/mnt/data/proof-obligation-compiler-kit/schemas/mutation-operator.schema.json).

A concrete three-finding obligation manifest is available at [real-sinex-obligations.json](sandbox:/mnt/data/proof-obligation-compiler-kit/examples/real-sinex-obligations.json).

## Executed artifacts

The compiler’s own output is preserved under `run/compiler-authored/`. The easiest rendered views are:

- [Executed companion assessment](sandbox:/mnt/data/proof-obligation-compiler-kit/reports/executed-assessment.md)
- [Static dashboard](sandbox:/mnt/data/proof-obligation-compiler-kit/reports/dashboard.html)
- [Flattened obligation inventory](sandbox:/mnt/data/proof-obligation-compiler-kit/reports/obligations.csv)
- [Flattened mutation inventory](sandbox:/mnt/data/proof-obligation-compiler-kit/reports/mutation-probes.csv)
- [Implementation source audit](sandbox:/mnt/data/proof-obligation-compiler-kit/reports/source-audit.md)
- [Literal mission coverage map](sandbox:/mnt/data/proof-obligation-compiler-kit/reports/mission-coverage.md)
- [Artifact manifest](sandbox:/mnt/data/proof-obligation-compiler-kit/manifest.json)

The compiler was run twice against an unchanged tree. Expected run-specific fields were normalized, and the resulting structured JSON reports compared equal. The exact native Cargo/`xtask` status is retained separately in [snapshot metadata](sandbox:/mnt/data/proof-obligation-compiler-kit/snapshot/metadata.json) and its raw log; it is not inferred from the existence of the bridge.

## Beads execution program

The package includes an 18-item draft Beads program covering:

- proof IR and grade invariants;
- deterministic compiler receipts;
- mutation safety;
- native Cargo witnesses;
- secure Chisel and Beads command receipts;
- MCP catalog/dispatch parity;
- demo construct compilation;
- public-claim compilation;
- PostgreSQL, NATS, and live-local environments;
- Beads closure integration;
- CI and incremental obligation selection;
- property and chaos testing;
- domain-owned mutation operators;
- performance and caching;
- CLI, HTML, JSON, and MCP proof views;
- replay, settlement, coverage, and deletion obligations;
- Sinex–Polylogue backend proof;
- independent false-green review.

The human plan is [06-beads-program.md](sandbox:/mnt/data/proof-obligation-compiler-kit/06-beads-program.md). The machine-readable draft is [proposed-issues.json](sandbox:/mnt/data/proof-obligation-compiler-kit/beads/proposed-issues.json), and existing related Sinex issues found in the snapshot are separated in [existing-related.json](sandbox:/mnt/data/proof-obligation-compiler-kit/beads/existing-related.json). Only one Beads captain should reconcile and mutate the real issue graph.

## Frontier-agent swarm

The [single-machine swarm plan](sandbox:/mnt/data/proof-obligation-compiler-kit/07-frontier-agent-swarm.md) defines independent lanes for:

- IR and schemas;
- mutation safety;
- native Cargo execution;
- closure receipts;
- MCP parity;
- demo construct validity;
- report surfaces;
- incremental CI;
- live-service proof environments;
- Sinex–Polylogue proof;
- two red teams;
- cold-reader replication;
- final integration.

It includes resource policy, file ownership, handoff contracts, integration checkpoints, and explicit heavy-build and live-environment semaphores.

Executable helpers include:

- [Worktree bootstrapper](sandbox:/mnt/data/proof-obligation-compiler-kit/scripts/bootstrap-worktrees.sh)
- [Shared swarm board](sandbox:/mnt/data/proof-obligation-compiler-kit/scripts/swarm-board.py)
- [Heavy-operation lock](sandbox:/mnt/data/proof-obligation-compiler-kit/scripts/with-heavy-lock.sh)
- [Host resource planner](sandbox:/mnt/data/proof-obligation-compiler-kit/scripts/resource-plan.sh)
- [Complete patch application](sandbox:/mnt/data/proof-obligation-compiler-kit/scripts/apply-full-patch.sh)
- [Compiler reproduction](sandbox:/mnt/data/proof-obligation-compiler-kit/scripts/reproduce-run.sh)
- [Canonical run comparison](sandbox:/mnt/data/proof-obligation-compiler-kit/scripts/canonical-compare.py)

## Twenty-four forkable prompts

The package contains **24 substantial parallel missions**, not merely short task descriptions.

The first 16 cover implementation and integration: native Cargo execution, proof IR, mutation chaos, closure receipts, MCP parity, demo compilation, reporting, Beads integration, CI, live-service harnesses, lifecycle obligations, the Polylogue backend proof, performance, security, documentation, and final integration.

The remaining eight cover independent red-team and research work: false-green attacks, scratch escape, grade inflation, formal strength composition, a public proof fixture corpus, LLM/model-effect obligations, Chisel self-application, and independent replication.

Use the [fork-prompt index](sandbox:/mnt/data/proof-obligation-compiler-kit/08-fork-prompt-index.md); individual prompts are under [prompts](sandbox:/mnt/data/proof-obligation-compiler-kit/prompts/).

## Sinex–Polylogue proof program

I also compiled the maximal backend architecture into a concrete obligation program in [Proof obligations for the maximal Sinex–Polylogue architecture](sandbox:/mnt/data/proof-obligation-compiler-kit/09-sinex-polylogue-proof-obligations.md).

It covers:

- provider-native and normalized transcript material authority;
- stable Polylogue identity versus replay-specific Sinex event identity;
- complete transcript revision settlement;
- PostgreSQL and SQLite projection reconstruction;
- physical versus logical session lineage;
- structural tool outcomes;
- durable assertions and judgments;
- context selection and actual delivery;
- deletion across materials, history, projections, search, vectors, and replicas;
- privacy capabilities separate from storage;
- offline outbox reconciliation.

Its flagship proof is **Rebuild Under Oath**: destroy all rebuildable Polylogue SQLite tiers, reconstruct them solely from Sinex-held material and durable history, and compare the result against an independently declared parity oracle. Merely emitting `session_indexed` metadata cannot satisfy that obligation.

## Validation

The package validator checked:

- all JSON and CSV artifacts;
- local Markdown links;
- shell and Python helper syntax;
- all 24 fork prompts;
- both patch boundaries;
- snapshot identity;
- repeated-run determinism;
- focused Python tests;
- presence of killed, survived, known-vacuous, baseline, and capability-only run states;
- generated-artifact path scrubbing;
- all supplied JSON Schema examples;
- final file checksums;
- ZIP and tar archive integrity.

See the [complete validation report](sandbox:/mnt/data/proof-obligation-compiler-kit/validation/VALIDATION.md) and [machine-readable validation result](sandbox:/mnt/data/proof-obligation-compiler-kit/validation/report.json).

The most important limitation remains explicit: portable/static counterexamples are valuable source-level evidence, but they are not substitutes for native Rust, PostgreSQL, NATS, cloud-harness, or deployed-service execution. The package preserves those environment boundaries instead of hiding them behind a green aggregate.

---

