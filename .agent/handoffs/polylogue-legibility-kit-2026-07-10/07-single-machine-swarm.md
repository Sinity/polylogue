# Single-machine swarm playbook

## Purpose

Run a large frontier-agent effort across Polylogue and Sinex on one machine without turning concurrency into corrupt state, false test passes, or unmergeable overlapping patches.

The machine should be treated as a small build farm with one human/operator and one integration coordinator. Worktrees provide code isolation; explicit runtime roots provide state isolation; Beads provide task authority; proof packets provide completion evidence.

The core doctrine is:

> **Parallelize interpretation, fixtures, isolated implementation, review, and narrow verification. Serialize shared-state runtime work, full builds, integration, and public claims.**

## 1. Swarm topology

Use four agent classes.

### 1.1 Coordinator

One agent owns the integration branches, merge queue, Beads status, and public candidate. It does not disappear into feature implementation.

Responsibilities:

- freeze and record base revisions;
- create and prune worktrees;
- maintain file ownership;
- maintain a dependency-aware merge queue;
- route blockers and design questions;
- run integration verification;
- refuse premature Bead closure;
- maintain the claims ledger;
- produce the final candidate packet.

### 1.2 Implementers

At most eight concurrent code-writing lanes across both repositories, and fewer when Sinex Rust builds are active.

Implementers own nonoverlapping file footprints and a bounded deliverable. They commit usable checkpoints and leave machine-readable verification receipts.

### 1.3 Analysts and reviewers

These agents do not need worktrees unless they generate patches. They can run in many parallel chat forks and produce:

- architecture specifications;
- fixture truth tables;
- construct-validity audits;
- public-copy reviews;
- security/privacy reviews;
- test plans;
- API schemas;
- Beads decomposition;
- cold-reader reports;
- adversarial counterexamples.

This is the best use of high parallelism. Sixteen reasoning agents can improve one implementation lane without sixteen writers colliding.

### 1.4 Proof runners

One or two agents run expensive checks, deterministic demos, browser recordings, and cold-reader evaluations. They do not edit feature code while testing it.

This separation prevents a common failure mode in which the same agent quietly changes the test, fixture, and implementation until its preferred story passes.

---

# 2. Repository and worktree layout

Use a single top-level workspace:

```text
/work/legibility/
  polylogue/                  integration checkout
  sinex/                      integration checkout
  wt/
    polylogue/
      docs/
      site/
      scenario/
      receipts/
      renderer/
      lineage-demo/
      readiness/
      launch/
    sinex/
      docs/
      missing-source/
      import-twice/
      replay-demo/
      interop/
      proof/
  state/
    polylogue/<lane>/
    sinex/<checkout-hash>/
  artifacts/
    incoming/<lane>/
    integrated/
    rejected/
  control/
    ownership.yaml
    queue.yaml
    baseline.json
    resource-policy.yaml
    claims.yaml
```

Do not place independent Git clones in every lane. Linked worktrees share the object database and make branch provenance visible.

## 2.1 Polylogue worktrees

```bash
P=/work/legibility/polylogue
PW=/work/legibility/wt/polylogue

git -C "$P" switch -c presentable/integration
for lane in docs site scenario receipts renderer lineage-demo readiness launch; do
  git -C "$P" worktree add -b "presentable/$lane" "$PW/$lane" master
done
```

Every runtime-capable lane receives a unique archive root and ports through:

- `POLYLOGUE_ARCHIVE_ROOT`;
- `POLYLOGUE_API_PORT`;
- `POLYLOGUE_BROWSER_CAPTURE_PORT`.

Prefer the repository’s branch-local `devtools dev-loop` machinery.

## 2.2 Sinex worktrees

```bash
S=/work/legibility/sinex
SW=/work/legibility/wt/sinex

git -C "$S" switch -c legibility/integration
for lane in docs missing-source import-twice replay-demo interop proof; do
  git -C "$S" worktree add -b "legibility/$lane" "$SW/$lane" master
done
```

Sinex’s checkout-local dev infrastructure is already designed to isolate PostgreSQL, NATS, JetStream state, and SQLx validation by checkout hash. Do not point several branches at one mutable dev database to save RAM.

When a worktree inherits the parent shell, verify that `xtask` corrects `CARGO_TARGET_DIR` to the active checkout. A suspicious subsecond Rust check is a failure signal, not a triumph.

---

# 3. Control files

## 3.1 `baseline.json`

```json
{
  "created_at": "2026-07-10T00:00:00Z",
  "polylogue": {"base": "f6c1da99", "dirty": false},
  "sinex": {"base": "b70a08d9", "dirty": false},
  "beads_exports": {
    "polylogue": ".beads/issues.jsonl",
    "sinex": ".beads/issues.jsonl"
  }
}
```

The actual run should regenerate this file from Git, not copy the example.

## 3.2 `ownership.yaml`

Every path has one writer. A reviewer may comment but not patch without a transfer.

```yaml
polylogue:
  README.md: docs
  docs/site/**: site
  devtools/pages_*.py: site
  polylogue/demo/**: scenario
  polylogue/rendering/**: renderer
sinex:
  README.md: docs
  docs/**: docs
  demo/incident-1432/**: missing-source
  crate/sinexd/src/sources/source_contracts/polylogue.rs: interop
```

Use directory ownership sparingly. Broad globs create accidental monopolies; list exact files for hot modules.

## 3.3 `queue.yaml`

```yaml
ready:
  - lane: polylogue-scenario
    depends_on: []
  - lane: polylogue-site
    depends_on: []
blocked:
  - lane: polylogue-receipts
    depends_on: [polylogue-scenario, polylogue-renderer]
in_review: []
merged: []
```

The coordinator updates the queue after every handoff. Agents do not infer readiness from an old chat message.

## 3.4 Verification receipts

Each lane writes a JSON receipt:

```json
{
  "base": "...",
  "head": "...",
  "commands": [
    {"argv": ["pytest", "-q", "tests/unit/rendering"], "exit_code": 0}
  ],
  "generated": ["docs/examples/visual-tapes/the-receipts.gif"],
  "not_run": ["full test suite"],
  "known_failures": []
}
```

A prose “tests pass” in a handoff is insufficient—the projects exist partly to make that lesson visible.

---

# 4. Delegation strategy

## 4.1 Delegate by epistemic role, not only code module

For each flagship demo, use at least three independent roles:

1. **Scenario author** declares source fixtures and expected ground truth.
2. **Product implementer** makes the product derive and render the result.
3. **Verifier/red team** tries to falsify the claim and checks controls.

The scenario author should not implement the primary classifier. The implementer should not be the only judge of the result.

## 4.2 Use two-stage design reviews

Before code:

- a construct-validity agent reviews the primary claim, oracle, controls, and generalization boundary;
- a product agent reviews whether the demo lands reusable capability rather than one-off glue.

After code:

- a security/privacy agent audits generated artifacts;
- a cold-reader agent assesses external comprehension without the implementation chat.

## 4.3 Assign one agent to destroy every headline

A dedicated adversarial agent receives the README, claims ledger, demo packets, and recordings. Its job is to find:

- denominator changes;
- circular oracles;
- private-field observations presented as benchmarks;
- missing negative controls;
- stale generated artifacts;
- present-tense roadmap language;
- inaccessible install commands;
- source gaps rendered as absence;
- semantic differences between CLI and web;
- claims that cannot be resolved from the packet.

This agent should have no incentive to preserve launch copy.

## 4.4 Keep interop design separate from current-state copy

One lane owns maximal Sinex–Polylogue design. Another owns current public README wording. This prevents an ambitious target architecture from leaking into present-tense product claims.

---

# 5. Resource governance

## 5.1 Machine-wide policy

Create `/work/legibility/control/resource-policy.yaml`:

```yaml
limits:
  concurrent_polylogue_pytest: 4
  concurrent_sinex_cargo_builds: 2
  concurrent_sinex_pipeline_scopes: 6
  concurrent_browser_recordings: 1
  concurrent_full_repo_checks: 1
  concurrent_local_sinex_stacks: 2
priorities:
  interactive_agents: high
  targeted_tests: normal
  full_checks: low
  recordings: low
```

The exact numbers should be adjusted to host RAM and cores. The important part is making the limits explicit.

## 5.2 Compile scheduling

Sinex Rust compilation is the dominant I/O load. Batch Sinex code lanes into alternating compile windows:

```text
Window A: interop + docs agents edit; one Sinex lane compiles.
Window B: second Sinex lane compiles; first lane reviews or writes tests.
Window C: integration compile/check only.
```

Do not run three full `xtask check --lint` jobs because three branches are “nearly done.” Merge them or run package-scoped checks first.

## 5.3 Runtime scheduling

- Polylogue deterministic SQLite demos can run concurrently with unique roots.
- Browser-capture and visual recording run serially to avoid port, focus, and timing interference.
- Sinex checkout-local stacks are isolated but memory-heavy; keep only the stacks needed by active integration tests.
- Stop abandoned stacks and inspect `xtask infra status --all-checkouts` before assuming the host is slow.
- Never test destructive replay or purge against a shared/live deployment.

## 5.4 Full-suite ownership

Only the coordinator/proof runner executes the final repository-wide suites. Implementers run narrow and subsystem checks.

This avoids duplicating expensive work and reduces the chance that all agents block simultaneously on the same host.

---

# 6. Merge mechanics for brute speed

## 6.1 Prefer nonoverlapping branches

The fastest merge is no conflict. File ownership and early API contracts matter more than elegant branch graphs.

## 6.2 Integrate contracts before consumers

Merge shared fixture schemas, result envelopes, and renderer descriptors before demo consumers. Consumers then rebase once against stable contracts.

## 6.3 Use path imports when history is not worth the conflict

For documentation or generated artifacts:

```bash
git checkout legibility/docs -- README.md docs/
git commit -m "docs: import external-legibility surface" \
  -m "Source branch: legibility/docs @ <sha>"
```

This trades Git aesthetics for deterministic integration while preserving provenance in the commit message.

## 6.4 Squash only at publication boundary

During the swarm, keep checkpoint commits because they let the coordinator bisect, cherry-pick, or drop a bad slice. Squash after the candidate is verified, if the repository prefers it.

## 6.5 Conflict triage

When a merge conflicts:

1. classify the conflict as generated, copy, contract, or semantic;
2. regenerate generated files rather than hand-merging them;
3. let the owner of the underlying contract resolve semantic conflicts;
4. rerun the smallest relevant proof immediately;
5. preserve both branches until the integrated result is verified.

---

# 7. Cross-project coordination

## 7.1 Keep two integration branches

Do not make one repository’s branch depend on an uncommitted path in the other.

Use explicit versioned artifacts for cross-project work:

- JSON Schema;
- protobuf/JSON event examples;
- material/revision manifests;
- ref grammar;
- rebuild parity reports;
- demo corpus bundle.

Each repo consumes a copied, versioned fixture or a pinned sibling checkout. Record the digest.

## 7.2 Contract-first interop lane

The interop lane should produce, before implementation:

1. authority matrix;
2. stable identity and revision vocabulary;
3. raw and normalized material descriptors;
4. admitted event examples;
5. bundle settlement protocol;
6. projection frontier semantics;
7. deletion and replay behavior;
8. standalone versus backed-mode matrix;
9. one end-to-end rebuild acceptance script.

Only after both projects accept these artifacts should code lanes modify producers and consumers.

## 7.3 Shared Incident bundle

The shared Evidence Lab corpus should be exported as a content-addressed bundle with:

- provider-native fixtures;
- Polylogue-normalized segments;
- Sinex source-material manifests;
- Beads task fixture;
- expected domain objects;
- expected cross-source evidence;
- privacy manifest;
- license/source statement.

Both repositories verify the same bundle digest. This prevents similar-looking but semantically divergent demos.

---

# 8. Parallel chat-fork intake

The 16 prompts in `08-fork-prompts.md` should not all be told to “implement everything.” They should produce complementary artifacts.

Recommended intake pattern:

```text
Forks 01–08   implementation-ready patches/specs for isolated lanes
Forks 09–12   Sinex and interop design/proof artifacts
Forks 13–15   adversarial review, construct validity, and cold-reader audits
Fork 16       coordinator synthesis over the returned artifacts
```

For each returned fork:

1. store its artifact bundle under `artifacts/incoming/<prompt-id>/`;
2. record base revision and stated verification;
3. run a static file-overlap check;
4. assign it `accept`, `partial`, `supersede`, or `reject`;
5. extract reusable specs before merging code;
6. never trust a fork’s self-reported tests without the receipt or rerun.

A simple overlap check:

```bash
for d in artifacts/incoming/*; do
  find "$d" -type f -name '*.patch' -print0 |
    xargs -0 -r grep '^+++ b/'
done | sort | uniq -c | sort -nr
```

High-overlap patches go to the coordinator, not directly into the merge queue.

---

# 9. Daily/iteration cadence without meetings

The swarm can coordinate entirely through artifacts.

## Start of iteration

- coordinator updates queue and ownership;
- agents pull/rebase from the recorded integration head;
- each lane writes its intended file set and verification plan;
- proof runner publishes current green baseline.

## Mid-iteration checkpoint

- each agent commits a compilable or inspectable checkpoint;
- status file lists blockers and any requested contract change;
- coordinator merges low-risk foundational work;
- red team reviews newly stable packets/copy.

## End of iteration

- agents finalize handoffs and verification receipts;
- coordinator integrates in dependency order;
- generated surfaces are rebuilt once;
- proof runner executes the integration gate;
- Beads receive evidence-backed comments/status;
- abandoned worktrees are preserved until the candidate is accepted, then pruned.

No synchronous meeting is necessary if status files are specific and current.

---

# 10. Failure modes and responses

## Two agents edited the same core file

Stop both. Choose one contract owner. Have the other agent export tests or a design note, then rebase onto the chosen implementation.

## A branch “passes” instantly in Sinex

Assume wrong checkout artifacts. Inspect `CARGO_TARGET_DIR` and run through `xtask` from the worktree.

## A demo passes but no independent oracle exists

Do not merge it as a proof. Keep the product capability, create an oracle/control task, and mark the public claim unsupported.

## A recording differs from the packet

Regenerate the recording from the packet commands. Never edit the recording narrative independently.

## A public copy agent outruns implementation

Downgrade the claim to aspirational or capability language. Do not pressure implementation to satisfy prewritten marketing text.

## The machine is saturated

Pause full checks and extra Sinex stacks before pausing reasoning/review agents. Use targeted tests, inspect host pressure, and serialize compile-heavy lanes.

## Integration becomes chaotic

Freeze new branches. Merge only contract and fixture foundations. Produce one green integration checkpoint. Resume from that point rather than carrying an unbounded merge queue.

---

# 11. Completion condition

The swarm is complete when the integrated candidate—not the individual branches—has:

- validated demo packets;
- reproducible recordings;
- no overlapping claim contradictions;
- supported install receipts;
- cold-reader comprehension;
- adversarial claim audit;
- current Beads evidence;
- a clean rebuild from declared inputs where claimed;
- a documented list of intentionally deferred work.

High agent utilization is not a success metric. The metric is a smaller, sharper, better-proven public product surface.
