## Turn 119 (role=assistant)

# Context/memory loop implementation and experiment package

I treated the attached mission as an implementation brief rather than a design-only request. The work now present in the Polylogue working tree covers the highest-value executable slice, a substantially hardened three-arm experiment harness, and a PR-sized design decomposition for the rest of `polylogue-37t`. fileciteturn0file0

The working tree is:

```text
/mnt/data/context-memory-work/polylogue-memory
branch: context-memory-loop
base: f6c1da997bea64bc6cd9670d9cbb8f7e7439ec51
```

The current work is **not yet committed or assembled into the final ZIP, patch series, deck, visual cockpit, worktree swarm kit, or 24 parallel prompts** I intended to include. The implementation and research artifacts themselves are present and usable, but the final packaging pass was interrupted before completion. The last addition—a published `protocol.schema.json` enforced by the harness—was written but not rerun through the final validation suite.

## What is implemented

The implemented product slice is:

```text
agent-authored assertion
        ↓ forced to candidate + inject:false
candidate review queue with inline evidence
        ↓
explicit accept / reject / defer / supersede judgment
        ↓
separate injection authorization
        ↓
bounded ContextImage compilation
        ↓
exact delivery snapshot persisted in user.db
        ↓
byte-equivalent retrieval and idempotent retry
```

This establishes several important invariants.

**An automated writer cannot self-promote a memory.** Non-user writers are coerced to `status=candidate` with:

```json
{
  "inject": false,
  "promotion_required": true
}
```

This overrides a hostile or careless caller that requests active or injectable state.

**Acceptance and injection are separate decisions.** A reviewer can accept a statement as durable user-owned knowledge while leaving it ineligible for future context. Injection requires a separate explicit `--inject` decision.

**Candidate judgments are append-preserving.** An exact retry is idempotent. A different decision, or a changed injection decision on retry, fails closed rather than silently rewriting review history.

**Bulk review has partial-success semantics.** Each candidate is processed under a SQLite savepoint. A malformed or conflicting candidate cannot roll back successful reviews of unrelated candidates.

**Compilation is not delivery.** A pure `ContextImage` becomes delivery evidence only when an explicit delivery operation persists it.

**The full compiled context image is preserved.** The delivery record contains the exact segments, evidence and assertion refs, omissions, caveats, policies, and metadata—not merely a list of refs.

**Delivery identity includes the recipient and actor.** Delivering identical context to two different sessions or by two different actors no longer collapses into one receipt.

**Same-ref drift is refused.** Reusing a context-snapshot ref with different context bytes, recipient, actor, boundary, run, inheritance mode, omissions, caveats, or refs produces an error.

**Delivery evidence is stored in `user.db`.** It is treated as irreplaceable audit state, not as a disposable analytical projection.

## Core implementation artifacts

The lifecycle transition authority is here:

[Assertion lifecycle state machine](sandbox:/mnt/data/context-memory-work/polylogue-memory/polylogue/core/assertion_lifecycle.py)

The candidate-review storage implementation is here:

[Assertion review and promotion storage](sandbox:/mnt/data/context-memory-work/polylogue-memory/polylogue/storage/sqlite/archive_tiers/user_write.py)

The exact context-delivery store is here:

[Context delivery persistence](sandbox:/mnt/data/context-memory-work/polylogue-memory/polylogue/storage/sqlite/archive_tiers/context_delivery_write.py)

The corresponding user-tier migration is:

[User schema v5 context-delivery migration](sandbox:/mnt/data/context-memory-work/polylogue-memory/polylogue/storage/sqlite/migrations/user/005_context_deliveries.sql)

The operator-facing CLI is:

[Candidate review CLI](sandbox:/mnt/data/context-memory-work/polylogue-memory/polylogue/cli/commands/judge.py)

It adds commands with this shape:

```bash
polylogue judge queue

polylogue judge accept assertion:<id>
polylogue judge accept assertion:<id> --inject

polylogue judge reject assertion:<id> \
  --reason "Contradicted by the structural test result"

polylogue judge defer assertion:<id>

polylogue judge supersede assertion:<id> \
  --kind decision \
  --body "Corrected reviewed statement" \
  --inject
```

The API implementation includes candidate queues, evidence resolution, single and bulk judgment, exact context-delivery persistence, retrieval, listing, and compile-then-record operations:

[Polylogue archive API changes](sandbox:/mnt/data/context-memory-work/polylogue-memory/polylogue/api/archive.py)

The new shared surface contracts are:

[Review and delivery payloads](sandbox:/mnt/data/context-memory-work/polylogue-memory/polylogue/surfaces/payloads.py)

## MCP authority boundary

The MCP role vocabulary now distinguishes:

```text
read < write < review < admin
```

Ordinary `write` capability is not enough to judge memory candidates. Candidate promotion is registered only for `review` or `admin`.

The implemented division is:

- `read`: list candidate claims and inspect review/evidence state;
- `write`: create ordinary mutable user state and record context delivery;
- `review`: apply candidate judgments;
- `admin`: maintenance and all lower capabilities.

That boundary is implemented in:

- [MCP support and role hierarchy](sandbox:/mnt/data/context-memory-work/polylogue-memory/polylogue/mcp/server_support.py)
- [MCP read tools](sandbox:/mnt/data/context-memory-work/polylogue-memory/polylogue/mcp/server_tools.py)
- [MCP mutation and review tools](sandbox:/mnt/data/context-memory-work/polylogue-memory/polylogue/mcp/server_mutation_tools.py)
- [MCP CLI role selection](sandbox:/mnt/data/context-memory-work/polylogue-memory/polylogue/mcp/cli.py)

One limitation remains explicit: `actor_ref` is durable audit metadata. It is not authentication. A deployment still has to bind an authenticated principal and capability to the MCP process or transport.

# Executed deterministic proof

The executable proof script is:

[Context/memory-loop proof generator](sandbox:/mnt/data/context-memory-work/polylogue-memory/devtools/context_memory_loop_proof.py)

The latest generated report is:

[Deterministic memory-loop proof report](sandbox:/mnt/data/context-memory-loop-proof-final/report.md)

The proof uses Polylogue’s private-data-free deterministic demo corpus and establishes this path:

1. An automated author proposes an assertion while requesting injection.
2. The storage boundary overrides it to candidate and non-injectable.
3. The review queue returns the candidate with resolved evidence.
4. A human reviewer accepts it.
5. Injection is authorized separately.
6. Context compilation includes an intentional unresolved ref.
7. The compiler records that omission rather than silently dropping it.
8. The complete compiled image is persisted.
9. The persisted image is read back exactly.
10. An exact retry is idempotent.

The latest proof generated:

```text
context snapshot: context-snapshot:457b8c95b2d86092
image SHA-256:    9bce3a04c3105e9f9c06b1440ccafdde6832c435da6f07fd3072900225b4c6b1
```

Because the implementation had not yet been committed, that proof’s repository-revision field still identifies the base commit rather than a final patch-series commit. It should be regenerated after committing.

# Resume Under Oath

The experiment has been developed as a runnable pre-registered protocol, not a narrative proposal.

Start here:

[Resume Under Oath README](sandbox:/mnt/data/context-memory-work/polylogue-memory/experiments/resume-under-oath/README.md)

The full pre-registration is:

[Pre-registration](sandbox:/mnt/data/context-memory-work/polylogue-memory/experiments/resume-under-oath/PREREGISTRATION.md)

The frozen machine-readable protocol is:

[protocol.json](sandbox:/mnt/data/context-memory-work/polylogue-memory/experiments/resume-under-oath/protocol.json)

The runnable implementation is:

[Experiment preparation, execution, audit, scoring, and aggregation](sandbox:/mnt/data/context-memory-work/polylogue-memory/devtools/resume_under_oath.py)

The operating instructions are:

[Confirmatory runbook](sandbox:/mnt/data/context-memory-work/polylogue-memory/experiments/resume-under-oath/RUNBOOK.md)

The checkpoint-construction protocol is:

[Checkpoint authoring standard](sandbox:/mnt/data/context-memory-work/polylogue-memory/experiments/resume-under-oath/CHECKPOINT-AUTHORING.md)

The experiment’s explicit exclusions and non-claims are:

[Resume Under Oath non-claims](sandbox:/mnt/data/context-memory-work/polylogue-memory/experiments/resume-under-oath/NON-CLAIMS.md)

## Experimental arms

Every checkpoint has three conditions:

1. **Raw reference:** exact refs and the frozen evidence set, without synthesis.
2. **Generated summary:** the same evidence plus an unreviewed generated handoff, including deliberately planted stale-memory traps.
3. **Reviewed context:** the same evidence plus reviewed assertions, validity information, caveats, and explicit exclusions.

All arms retain access to the same frozen evidence. The experimental manipulation is the memory packet, not evidence availability.

## Outcome definition

The scorer consumes structured JSON. It does not grade writing quality or ask another model whether the response “looks good.”

The frozen rubric is:

| Construct | Points |
|---|---:|
| Factual reconstruction | 60 |
| Exact evidence grounding | 15 |
| Rejection of stale-memory traps | 15 |
| Correct next action and support | 10 |
| Each accepted critical stale claim | −20 |

Scores are clamped to 0–100.

Unknown rubric IDs, duplicate IDs, fake evidence refs, extra fields, invalid schema versions, missing output, malformed output, and timed-out executions fail closed.

A missing or unscorable execution receives zero under intention-to-treat and is conservatively counted as a critical safety failure. This prevents runner attrition from making an arm look safer.

## Confirmatory population

The intended confirmatory design is:

```text
40 independently selected frozen checkpoints
× 3 arms
× 3 independent executions per checkpoint-arm
= 360 task executions per fixed agent configuration
```

The three executions are averaged inside each checkpoint-arm. The checkpoint remains the independent analysis unit. The design does not pretend that 40 checkpoints with three executions are 120 independent checkpoints.

The power calculation assumes:

```text
minimum practical mean difference: 5 points
paired standard deviation:         10 points
two-sided alpha:                    0.05
power:                              0.80
closed-form requirement:           32 checkpoints
target with 20% defect buffer:      40 checkpoints
```

That calculation concerns superiority against zero for the score endpoint. It is not an independent power claim for the safety non-inferiority constraint.

## Decision rule

A result supports only a bounded uplift statement when all of these hold:

1. Reviewed-minus-raw point estimate is at least five points.
2. The checkpoint-paired score interval’s lower 95% bound is above zero.
3. The checkpoint-paired safety-harm interval’s upper 95% bound is at most five percentage points.

The practical uplift threshold is falsified when the score interval’s upper 95% bound is below five.

The safety constraint is falsified when the paired safety-harm interval’s lower 95% bound exceeds five percentage points.

Everything else is inconclusive.

The generated-summary arm and score subcomponents are descriptive rather than additional confirmatory tests.

## Blinding and leakage control

Preparation produces separate trees:

```text
prepared/
  runner/
    runner-manifest.json
    tasks/<opaque-id>/
      prompt.md
      evidence.json
      task.json

  evaluator/
    sealed task/arm map
    ground-truth hashes
    audit state
    run metadata
```

Task and condition labels are keyed HMACs. The key itself is not stored; its digest and the evaluator-side assignment map are retained.

The static audit checks file hashes, manifest parity, known evaluator-token leaks, prompt drift, and runner-tree vocabulary. It does **not** provide operating-system isolation. A confirmatory run must use a container, VM, remote execution adapter, or equivalent capability boundary that exposes only the task input and one writable output path.

## Frozen run identity

A confirmatory run must record:

- exact model and provider;
- frozen agent configuration;
- runner revision;
- packet-generator revision;
- review-protocol revision;
- tool policy;
- sampling policy;
- budget policy;
- environment or container fingerprint;
- checkpoint-sampling-manifest digest.

Runner receipts record timeout state, return code, duration, and output digest for every task.

The run-metadata contract is:

[Run metadata schema](sandbox:/mnt/data/context-memory-work/polylogue-memory/experiments/resume-under-oath/run-metadata.schema.json)

A final hardening change also added:

[Protocol schema](sandbox:/mnt/data/context-memory-work/polylogue-memory/experiments/resume-under-oath/protocol.schema.json)

That protocol schema is now called by `validate_protocol`, but this final change was not rerun through the test suite before tool execution ended.

## Harness fixtures

Six committed checkpoints test constructs and harness behavior:

1. structural failed test versus a later success claim;
2. Beads state at the frozen frontier versus a later ready event;
3. repository state anchored to a particular checkpoint;
4. information lost during compaction;
5. disjoint token-accounting lanes;
6. an accepted and previously delivered assertion that has become stale.

The checkpoint tree is here:

[Resume Under Oath checkpoint fixtures](sandbox:/mnt/data/context-memory-work/polylogue-memory/experiments/resume-under-oath/checkpoints)

These fixtures prepare 54 tasks and are explicitly marked synthetic/harness-only. The executable confirmatory gate refuses to treat them as outcome evidence.

# Full-loop design pack

The design index is:

[Context/memory-loop design pack](sandbox:/mnt/data/context-memory-work/polylogue-memory/docs/design/context-memory-loop/README.md)

The durable lifecycle contract is:

[Lifecycle contract](sandbox:/mnt/data/context-memory-work/polylogue-memory/docs/design/context-memory-loop/lifecycle.md)

The target context allocation and inclusion/exclusion ledger is:

[Scheduler ledger](sandbox:/mnt/data/context-memory-work/polylogue-memory/docs/design/context-memory-loop/scheduler-ledger.md)

The planned structural effect-observation model is:

[Effect observation](sandbox:/mnt/data/context-memory-work/polylogue-memory/docs/design/context-memory-loop/effect-observation.md)

Validity anchors and staleness are covered by:

[Staleness and validity](sandbox:/mnt/data/context-memory-work/polylogue-memory/docs/design/context-memory-loop/staleness.md)

The adversarial analysis is:

[Memory-loop trust model](sandbox:/mnt/data/context-memory-work/polylogue-memory/docs/design/context-memory-loop/trust-model.md)

The Sinex relationship is:

[Maximal Sinex memory backend](sandbox:/mnt/data/context-memory-work/polylogue-memory/docs/design/context-memory-loop/sinex-backend.md)

The design-level non-claims are:

[Context/memory-loop non-claims](sandbox:/mnt/data/context-memory-work/polylogue-memory/docs/design/context-memory-loop/NON-CLAIMS.md)

## Planned machine contracts

The pack includes draft JSON Schemas for:

- [Context selection ledger](sandbox:/mnt/data/context-memory-work/polylogue-memory/docs/design/context-memory-loop/schemas/context-selection-ledger.schema.json)
- [Assertion validity anchors](sandbox:/mnt/data/context-memory-work/polylogue-memory/docs/design/context-memory-loop/schemas/assertion-validity.schema.json)
- [Effect observations](sandbox:/mnt/data/context-memory-work/polylogue-memory/docs/design/context-memory-loop/schemas/assertion-effect-observation.schema.json)

These are designs, not implemented runtime contracts.

# PR-sized continuation plan

The work is decomposed into 12 packets:

[Human-readable packet plan](sandbox:/mnt/data/context-memory-work/polylogue-memory/docs/design/context-memory-loop/pr-packets.md)

[Machine-readable packet dependency graph](sandbox:/mnt/data/context-memory-work/polylogue-memory/docs/design/context-memory-loop/pr-packets.json)

The sequence is:

| Packet | Scope | State |
|---|---|---|
| M00 | Review and exact-delivery spine | Implemented in working tree |
| M01 | `ContextSource` protocol and complete allocation ledger | Planned |
| M02 | Move SessionStart sources onto the scheduler | Planned |
| M03 | Validity anchors and evaluators | Planned |
| M04 | Recursive grounding and compatibility checks | Planned |
| M05 | Structural effect observations | Planned |
| M06 | Declared agent candidate-capture protocol | Planned |
| M07 | Reboot and failure-loop context composition | Planned |
| M08 | Prompt-evaluation and setup-improvement writers | Planned |
| M09 | Maximal Sinex persistence | Planned |
| M10 | Confirmatory checkpoint factory | Planned |
| M11 | Confirmatory execution and audit | Planned |

The strongest next implementation sequence is:

```text
M01 scheduler ledger
    ↓
M03 validity anchors
    ↓
M04 recursive grounding
    ↓
M02 SessionStart integration
    ↓
M05 effect observation
```

This order makes selection and staleness explainable before introducing downstream effect labels.

# Maximal Sinex fit

The design does not use the older “Sinex receives only transcript metadata” doctrine.

The intended authority model is:

```text
Sinex
  durable provider-native transcript material
  normalized transcript material
  attachments
  assertion and judgment history
  context images and delivery records
  validity and effect observations
  model-effect records
  lifecycle, retention, and deletion state

Polylogue
  AI-work ontology
  provider normalization
  logical-session composition
  assertion and context policy
  context scheduling
  memory-loop product behavior
  CLI, MCP, web, and analytical UX

SQLite
  standalone Polylogue authority when Sinex is absent
  local/offline projection in Sinex-backed mode
  local FTS/vector acceleration
  UI state
  durable offline outbox
```

The important backend rule is:

> In Sinex-backed mode, no irreplaceable transcript, review, context-delivery, judgment, or memory-policy state should silently exist only in SQLite.

At the same time, Sinex must not flatten Polylogue objects into generic event sludge. Stable Polylogue session, message, assertion, judgment, and delivery identities must remain domain objects whose durable material and history are stored through Sinex.

# Public claims ledger

A small claim ledger now separates implementation, experiment, design, and aspiration:

[Public claims ledger](sandbox:/mnt/data/context-memory-work/polylogue-memory/docs/public-claims.yaml)

The current statuses are:

- candidate review and exact delivery: implemented and deterministically demonstrated;
- memory uplift: aspirational experimental claim;
- complete scheduler allocation ledger: designed, not implemented;
- effect observation: designed, not implemented;
- Sinex memory backend: architectural direction, not implemented.

# Validation performed

The following validation was completed before the final protocol-schema change:

- lifecycle and experiment harness tests: 19 passing after run-metadata schema enforcement;
- broader focused non-MCP set: 320 passing;
- MCP per-tool discovery, authorization, argument, and contract tests: 177 passing;
- MCP contract-evidence tests: 26 passing;
- MCP envelope-contract tests: 35 passing;
- MCP server-surface tests: 78 passing;
- MCP discovery and CLI tests: 72 passing;
- MCP mutation idempotency subset: 11 passing;
- Ruff and strict MyPy over changed implementation paths passed before the final small edits;
- deterministic demo-memory round trip passed;
- fixture preparation produced 54 tasks;
- fixture leakage audit passed;
- fixture confirmatory gate refused execution as designed;
- fixed power calculation produced 32 required and 40 targeted checkpoints.

A complete repository-wide test run was not completed. One attempt to iterate every MCP test file stalled in an existing context-preamble test after partial progress; the affected new MCP surfaces were then validated through focused suites instead.

The final `protocol.schema.json` addition and the last documentation wording correction still need one focused rerun of:

```bash
cd /mnt/data/context-memory-work/polylogue-memory

.venv/bin/python -m pytest \
  tests/unit/devtools/test_resume_under_oath.py -q

.venv/bin/ruff check \
  devtools/resume_under_oath.py \
  tests/unit/devtools/test_resume_under_oath.py

.venv/bin/mypy \
  devtools/resume_under_oath.py

PYTHONPATH=. .venv/bin/python -m devtools.resume_under_oath \
  --root experiments/resume-under-oath validate

git diff --check
```

# Explicitly unfinished packaging work

The implementation and design artifacts are present, but the following requested presentation layer was not completed before execution time ended:

- three-commit patch series;
- combined apply-ready patch;
- final ZIP and SHA-256 checksum;
- regenerated proof after final commits;
- executive PDF and editable PowerPoint;
- lifecycle, trust, experiment, and Sinex SVG diagrams;
- static Memory Loop Cockpit mockup;
- 24 standalone fork prompts;
- worktree bootstrap and tmux-launch scripts;
- swarm board and resource-lock helpers;
- cold-reader, handoff, interface-decision, and proof-report templates;
- one integrated final validation ledger.

The code has not been committed, so it should not yet be treated as a polished merge-ready series. The most useful immediate state is nevertheless substantially beyond design: the core lifecycle, review, delivery receipt, proof path, experiment harness, fixtures, scoring, preregistration, schemas, and continuation architecture all exist in the working tree.

---

