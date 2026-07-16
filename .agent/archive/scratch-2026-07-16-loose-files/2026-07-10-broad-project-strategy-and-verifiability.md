---
created: "2026-07-10T06:18:16+02:00"
purpose: "Comprehensive synthesis of the 2026-07-10 Polylogue codebase/backlog/test/demo/coordination audit, including live defects, strategic verdict, Fable review adjudication, and a revised execution approach."
status: "active"
project: "polylogue"
---

# Polylogue: Broad Project Strategy and Verifiability Synthesis

This note captures the current investigation so another frontier agent can start
from evidence rather than repeat the survey. It is a scratch synthesis, not a
replacement for Beads. Any work that survives into execution must be encoded in
existing or new Beads before implementation.

Grounding date: 2026-07-10. Current checkout: `master`, canonical source index
schema v29. Deployed archive: `/home/sinity/.local/share/polylogue/index.db`,
schema v24. The checkout is dirty with unrelated user/agent work; none of it was
modified for this note.

## Executive Conclusions

1. Polylogue is not a waste of time. Its strongest differentiated asset is a
   local, cross-provider, longitudinal evidence substrate that preserves raw
   provenance, normalized tool outcomes, costs, lineage, user judgment, and
   queryable session history. Most agent runtimes do not solve this across
   vendors or across years of work.
2. Polylogue should not become another execution runtime or durable workflow
   engine. Hermes, Codex, Claude, Kitty/worktrees, and similar systems should
   continue to execute. Polylogue can be the evidence, evaluation, context, and
   coordination plane around them.
3. Polylogue is currently too timid in one respect: it mostly observes after
   the fact. The compounding opportunity is a closed loop: compile bounded
   evidence-backed context/planned-work views, record what agents actually ran,
   verify outcomes, measure cost and failures, then improve the next context or
   dispatch decision.
4. It is simultaneously too broad in another respect: many surfaces, tools,
   reports, and ambitious Web UI plans exist before the canonical operator
   journeys are demonstrated against real-shaped data. This creates impressive
   structural coverage with weak product confidence.
5. The user's testing intuition is directionally right, with one limit. Tests
   cannot reveal truly undeclared product requirements. They can systematically
   reveal missing evidence relative to declared workflows, surfaces, source
   variants, states, invariants, and performance budgets. Polylogue already has
   most ingredients for such a compiler, but they are disconnected.
6. The earlier one-week proposal was oversubscribed by roughly 2x to 3x. The
   refined campaign must have a hard drop order and a day-two checkpoint. The
   first work to drop is v2 scaffolding, session-to-commit demonstration work,
   broad state-machine/differential expansions, and the context-pack experiment;
   reliability and proof authority remain.
7. The execution-planning idea must be reconciled into `polylogue-s7ae`, not
   introduced as an independent `ExecutionManifest` product. Otherwise it is a
   successor to the retired bespoke conductor, with the same drift risk.
8. No live archive rebuild should occur during the campaign. Canonical code is
   v29 and the deployed archive is v24. Use read-only v24 analysis and scratch
   v29 archives until blue-green rebuild support exists or an explicit rebuild
   window is chosen.

## Product And Architecture Model

Polylogue has four practical rings:

- source acquisition and parsing under `polylogue/sources/` and
  `polylogue/pipeline/`;
- the durable/rebuildable archive substrate under `polylogue/storage/`;
- derived models under `polylogue/insights/`;
- leaf surfaces under CLI, MCP, API, daemon HTTP/Web UI, rendering, demos, and
  verification/devtools.

The central architectural rule is correct: meaning belongs in source/storage or
insights, then surfaces project it. A new demo is allowed to be a one-off
application, but the facts it needs should be composable product fields rather
than report-private SQL.

### Data identity and evidence

- Session, message, and block ids are generated from origin/native identity and
  transcript position; they are not redundantly stored Python ids.
- The normalized tree is sessions -> messages -> blocks.
- `messages.material_origin` distinguishes authored user text from runtime
  protocol/context/tool results, which is essential for honest word/token/cost
  accounting.
- `blocks.tool_result_is_error` and `tool_result_exit_code` are the structural
  outcome keystones. `NULL` means unknown, not success.
- `actions` is a derived view pairing tool use and tool result evidence.
- Content hashes exclude user metadata, so assertions/tags do not cause source
  re-import.
- `session_links` carries unresolved/resolved/quarantined topology and lineage.
  Prefix-sharing reads compose parent prefix plus child tail.
- Block content hashes and typed resolver states landed in schema v25, but the
  deployed archive remains v24 and therefore does not yet contain them.

### Five SQLite tiers

- `source.db`: durable raw evidence, artifacts, blobs, hook events, sidecars.
- `index.db`: rebuildable normalized tree, FTS, topology, costs, and insights.
- `embeddings.db`: rebuildable vector data and catch-up state.
- `user.db`: durable assertions/settings and operator judgment.
- `ops.db`: disposable ingest/convergence/telemetry state.

Durable source/user schema changes use explicit additive migrations. Derived
index/embedding schemas rebuild rather than migrate in place. This separation is
sound. The weak point is operational: without `polylogue-b5l`, derived rebuilds
degrade the live archive for 20-40 minutes.

### Canonical versus deployed schema

The current canonical constant is `INDEX_SCHEMA_VERSION = 29`. The deployed
archive reports `PRAGMA user_version = 24`.

The post-v24 source changes are real, already merged changes rather than a
hallucinated roadmap:

- v24 -> v25: block content-hash citation anchors (`697470661`, PR #2588);
- v25 -> v26: transcript-rank action pairing (`7b5a5aa05`, PR #2597);
- v26 -> v28: delegation/profile/view additions (`09dc46c26`, PR #2607; the
  commit deliberately jumps the constant by two);
- v28 -> v29: trigram index for affordance usage (`f5c35e702`, PR #2622).

Implication: any new live artifact must name both code commit and archive schema.
The earlier phrase "regenerate against v29" was source-grounded but
operationally incomplete. It must not imply rebuilding the canonical archive
during a multi-lane week. Aggregate analysis can continue read-only against v24
when its required columns exist. New v29-only proofs should use seeded or
sampled scratch archives.

## Beads And Delivery State

Current census from the audit:

- 612 total Beads;
- 171 closed, 440 open, 1 in progress;
- 104 blocked and 336 dependency-ready at survey time;
- 93 active Beads have no `delivery:*` assignment;
- no open P0 items; 13 active P1s; most work is P2-P4;
- the only in-progress item is `polylogue-f2qv.2`.

The delivery board is a useful safety topology, not a truthful scheduler:

- `R0-normalize` is hard-coded as a gate but has zero members;
- unlabeled work is counted as a warning and excluded from all gate totals;
- the first non-empty incomplete gate therefore appears as A-trust-floor;
- A currently reports 40/60 closed (67%), one WIP, 18 ready, one blocked;
- later gates contain large ready populations even though the linear board
  presents them as future work.

The full A-N ordering still encodes valid dependency concerns. It should not be
silently replaced by a parallel campaign. A bounded campaign should instead
declare which exact gate contracts its artifact depends on. The board remains
the long-range safety topology; campaign membership is a separate selection
view, not a second authority.

Do not make R0 globally fail merely because 93 Beads lack labels. That would
incentivize a rushed labeling sweep and recreate the same vacuous-green failure
as the closure matrix. Better policy:

- unlabeled items render as `unclassified`, never disappear;
- only campaign candidates must meet a strict execution-grade schema before
  dispatch;
- background normalization is sampled for quality before it becomes blocking;
- a proposed hard gate needs a reviewed sample with zero critical
  misclassification and explicit `unknown` handling.

## Current Frontier Tooling Is Not Policy-Grade

`devtools workspace frontier --json` currently:

- reads only the first 40 ready Beads;
- groups by the first `area:*` label or prose keywords;
- classifies proof cost, runtime risk, schema lane, and subagent suitability by
  substring checks;
- checks `audit|classify|investigate|research|plan|read-only` before checking
  implementation verbs.

Consequently `f2qv.2-.5` and many implementation tasks are labeled
`read-only-audit`. The tool is a useful radar, not a scheduler or dispatcher.
Closed `polylogue-qra` honestly delivered its literal acceptance criteria; it
should remain closed. Any improvement is a specific successor under the
coordination program.

The current `polylogue agents status --format json` result was about 55.5 KiB:
roughly 16.4 KiB of subagent exchanges, 7.6 KiB of activity, and 7.1 KiB of
proof refs. Count bounds do not make this an agent dispatch packet. Historical
evidence, launcher metadata, and full exchange payloads need explicit detail
views; default/self/current/planned-work projections must be much smaller.

A follow-up measurement after this text was drafted made the problem more
specific. With the current dirty repo and active archive:

- `status`: 64,378 bytes;
- `self`: 51,393 bytes;
- `work-item`: 52,745 bytes;
- `conflicts`: 63,258 bytes;
- `handoff`: 52,519 bytes.

`project_coordination_envelope` only removes a few peer/resource/handoff arrays
per view. It retains session trees, archive activity, subagent exchanges, proof
refs, and context-flow refs in almost every projection. In the measured `self`
view, subagent exchanges alone used 16,449 serialized bytes, activity 7,998,
proofs 7,582, and context refs 6,551. The view names therefore overpromise
semantic projection. Repairing this is a prerequisite to adding `planned_work`:
each existing view should have an explicit allowlist of sections and a separate
ref-only/detail policy, not merely a different `view` tag on the same envelope.

## Coordination: Reconcile With s7ae

Fable's highest-value correction is that a new execution compiler risks
becoming conductor-2.0. This is correct unless the design is changed.

### What already exists

`polylogue-s7ae` is the coordination epic. Its declared ontology already
includes:

- source-agnostic WorkItemRef with Beads/GitHub/git/inferred adapters;
- sessions, worktrees, branches, activity/resource episodes;
- messages/advisories, context flow, proof and handoff refs;
- CLI/MCP/Web projections over one typed envelope;
- a live two-agent separate-worktree proof.

Relevant children:

- `s7ae.1` (closed): typed envelope and bounded agent-grade CLI/MCP views;
- `s7ae.4` (closed): archive session-tree/activity/proof/context evidence;
- `s7ae.2` (open): pre-deploy MCP and hook batch;
- `s7ae.3` (open): scoped coordination messages and scheduler-mediated
  advisories;
- `37t.11` (open): one context scheduler and injection ledger;
- `s7ae.5` (open): live two-agent worktree/message/context/handoff proof.

Fable described a strict `37t.12 -> 37t.11 -> s7ae.3 -> s7ae.5` spine. That is
not quite accurate. `37t.12` is the candidate-assertion judgment queue, a trust
prerequisite for promoted operator context, but not the execution-plan data
model. It is relevant to trusted context, not the direct home of planned work.

### Stale/conflicting s7ae details

- `s7ae.1` claimed bounded output, but the current default status payload is
  51-64 KiB even for supposedly narrow views. A successor or reopened
  acceptance gap is warranted.
- `s7ae.5` still requires a handoff packet under
  `.agent/conductor-devloop/*.md`, even though that packet was retired on
  2026-07-08. Its AC must point to a ContextImage/coordination handoff artifact
  and durable Beads refs instead.
- Current subagent exchange inference sometimes records launcher metadata rather
  than the actual dispatch/final exchange. This weakens operational use.

### Structurally safe replacement for ExecutionManifest

Do not add a new durable scheduler vocabulary. Add a bounded `planned_work`
projection to the existing coordination envelope:

- source of truth remains Beads plus git/worktree/live-resource evidence;
- the projection is read-only and generated on demand;
- no active-loop markdown, packet directory, mirrored queue, lifecycle state
  machine, or new `devloop-*` script exists;
- output is stdout/JSON or disposable `.local/` evidence, not committed current
  state;
- each planned-work row carries existing WorkItemRef, exact dependency refs,
  explicit ownership/avoidance paths when authored, proof contract, resource
  lock, source commit, freshness, provenance, and confidence;
- missing structured information remains `unknown`; prose inference cannot
  authorize dispatch;
- dispatch remains an external adapter concern (Codex/Claude/Hermes/Kitty);
- execution events feed the existing coordination/evidence model and later the
  `rii.1` write leg.

Implementation should first make current views honest:

- `self`: repo/self/current work-item plus top-level freshness/caveats only;
- `work-item`: work item, dependency/gate summary, proof contract, refs;
- `conflicts`: peers, overlaps, resource episodes, paths, confidence;
- `handoff`: bounded context/handoff/proof refs, no historical exchange bodies;
- `status`: compact counts and refs by default, with `--detail evidence` for the
  current full envelope;
- `planned-work`: explicit campaign/work-item rows and resource locks, never
  inherited historical arrays by accident.

This extends `s7ae.1/.4`, supports `s7ae.5`, and consumes `37t.11` only for the
context-injection leg. It does not replace Beads, the context scheduler, or the
agent runtime.

### Why this is not the retired conductor

The retired packet was a second state system: active-loop markdown, operation
logs, mode transitions, demo radar, generated sidecars, helper scripts, and
handoff packets. Its 2026-07-08 retirement explicitly says Beads now owns the
loop and the packet must not be resurrected.

The surviving useful ideas are proof ladders, evidence-first direction,
serial-heavy/parallel-light execution, and artifact closure. Those already live
in `CLAUDE.md`, `.agent/CONVENTIONS.md`, Beads, and current devtools. A safe
planned-work projection merely reads those authorities and expires with its
input cursor.

## Test And Verification Substrate

Polylogue's test infrastructure is sophisticated:

- roughly 13k collected tasks across unit/property/integration/benchmark/fuzz
  families;
- `devtools test` provides isolated temp roots, serialization, live progress,
  timeouts, and retained run artifacts;
- testmon affected selection, xdist overrides, frozen clocks, Hypothesis
  profiles, schema-driven corpus generation, and shared archive builders exist;
- verify run directories retain selection, per-test events, resource samples,
  summaries, logs, and postmortems;
- quality registries contain dozens of validation lanes, mutation campaigns,
  benchmarks, and scenario projections.

The weakness is not test quantity. It is evidence authority and obligation
linkage.

### Closure matrix false confidence

`devtools verify closure-matrix` reports `clean (33 domains)`, but the verifier
only checks:

- target paths exist;
- representative test paths exist;
- required/optional rows name a test;
- absent rows have prose gaps;
- gate names and domain ids are valid.

It does not collect nodeids, execute tests, verify lane inclusion, inspect
coverage, or prove the representative test exercises the target. It contains no
Hermes parser row and no Web UI interaction row. Its name overstates its proof.

### Workflow declarations are not executable across surfaces

`polylogue/product/workflows.py` declares nine required query/action workflows
and surfaces such as CLI, daemon, MCP, Web, docs, and completion. The executable
golden paths are all CLI commands. Three required workflows have no golden path.
Declaring `web` therefore creates no browser obligation.

### Browserless visual evidence

The existing visual lane explicitly records `evidence_kind: browserless-dom`.
Many tests assert literal CSS classes, JavaScript function names, selectors, or
source strings. These can prove that markup/code was emitted, but not that a
user can click, scroll, focus, expand, navigate, recover from errors, or read a
long transcript.

### Test telemetry is retained but weakly used

Historical run data includes thousands of per-test events and resource samples.
Observed peak RSS reached about 10.4 GiB. Focused-test runs omit git head/dirty
(`polylogue-k6fm`), blocking same-commit flake analysis. Current task-history
consumers use only a small aggregate subset. Default verify/testmon has recorded
long stalls and D-state xdist workers; this needs an owning Bead rather than
remaining only a memory note.

### Fuzz collection gap

`polylogue-kj22` records that 418 pytest-mode fuzz tests exist and pass when the
pattern is overridden, but normal pytest collection excludes `fuzz_*.py`.
Documentation saying they run on every commit is false.

## Proposed Verification Model

The right improvement is a coverage-obligation compiler built over existing
types, not a new parallel test framework.

### WorkflowProofSpec

A proof specification should reference existing workflow and scenario
registries and add these dimensions:

- workflow id;
- surface: CLI/API/MCP/daemon/Web/completion;
- state profile: empty/single/many/large/long-tool/degraded/stale/malformed;
- fixture authority: synthetic/observed-schema/sanitized-real/live-canary;
- proof kind: contract/journey/property/differential/chaos/performance;
- execution and assertion specs;
- lane, runtime/RSS/request budgets, and evidence artifacts.

Mandatory rules should be small and semantic:

- every declared workflow/surface has one executable positive proof or an open
  gap Bead;
- mutating workflows prove zero/many/confirmation/idempotency behavior;
- a Web surface requires a real browser journey;
- authoritative sources require observed-schema compatibility;
- degraded-capable reads require an explicit degraded-state proof;
- a claimed nodeid must collect and its lane must actually execute it.

The compiler must demonstrate anti-vacuity: deleting the Hermes observed-schema
proof or a browser journey in a fixture makes the gap report red.

### Combinatorial budget

Do not generate a Cartesian matrix. Initial PR/default behavior:

- mandatory obligations only;
- at most one canonical positive per declared surface plus required safety
  negatives;
- global default generated-case budget of 40;
- optional/nightly greedy pairwise set cover capped at 96 cases;
- overflow remains visible as uncovered pairs, not silently generated test
  spam;
- every case records estimated and observed runtime/RSS so future selection is
  evidence-driven.

Pairwise generation is useful only after mandatory cells work. It is not part of
the first MVP if it delays real browser and observed-source proof.

### ArtifactCompatibilityProfile

Origin compatibility needs a typed contract:

- artifact kind and version probe;
- required tables/columns/shape;
- optional fields by known version/family;
- structural fingerprint and fidelity fields;
- supported versions and unknown-version policy;
- privacy-safe fixture-generation recipe.

For SQLite sources, persist only schema version, table/column/type/nullability
fingerprints, counts, and compatibility diagnostics. Real live canaries remain
read-only. Unknown/newer variants should stay identified as the correct origin
and fail as `unsupported_schema`, not disappear into provider detection failure.

### Verification event use

Extend existing verify-run metadata rather than create a new database writer:

- git head/dirty, branch, Bead/work-item refs, agent-session ref;
- proof ids and selected nodeids;
- normalized failure signature;
- runtime/RSS/IO samples;
- trace/screenshot/log refs.

Emit a bounded `VerificationRunEvent` through the existing live substrate/write
leg. Retain full logs/traces primarily for failures, keep passing raw samples for
a bounded window, and downsample long-term summaries. Derived uses should include
same-commit flakiness, recurring failures, runtime trends, stale source schemas,
and proof-gap prioritization.

## Live Defects And Structural Gaps Found

### 1. Hermes state.db rejects the current supported product

Installed Hermes is current `v0.18.2 (2026.7.7.2)`, upstream `1a477697`.
Its live `/home/sinity/.hermes/state.db` is about 85 MiB, schema version 11,
with 91 sessions and 5,375 messages.

Polylogue's `hermes_state.py` requires a superset of all known session/message
columns before recognizing the database. Current Hermes v11 lacks session
columns such as cwd/git fields, rewind/archive fields and message columns such
as active/compacted/observed/platform_message_id. Therefore:

```
looks_like_state_db_path(...) == False
parse_state_db(...) -> ValueError: ... is not a Hermes state.db file
```

Meanwhile five focused synthetic Hermes state tests pass and the closure matrix
is clean. The fixture invents a complete later/superset schema (and labels it
version 17), so it never tests a real supported artifact shape.

`polylogue-fs1.1` is closed even though its core capability is false for the
current product. It should be reopened or explicitly superseded, with v11 and
superset compatibility fixtures generated from structural evidence.

Correct repair shape:

- require only identity/content columns actually necessary to parse;
- read optional columns through a safe accessor or dynamically projected NULL;
- apply `active` filtering only when that column exists;
- record source schema version/fingerprint/fidelity;
- treat real content as private and commit only structural/sanitized fixtures.

### 2. Web UI is structurally present but operationally poor

A private-browser probe of the deployed reader showed:

- about 17,091 conversations and a roughly 24.7 GiB index;
- initial stale/API-timeout messaging even while stale data rendered;
- unsupported read views shown as `(pending HTTP route)` controls;
- a selected 66-message Codex session returned a 681,498-byte detail payload;
- raw tool results render at full weight without progressive disclosure;
- titles often fall back to native ids;
- `/api/status` is roughly 68 KiB and is polled;
- a 100-row list response is roughly 107 KiB;
- the detail click itself was about 92 ms, so transfer/DOM/readability rather
  than only server latency is the problem.

The daemon already has paginated `/api/sessions/{id}/messages`, but the Web UI
loads `/api/sessions/{id}`, which embeds every message. This is a leverage point:
make session metadata and message preview/full retrieval separate, then reuse the
same API in v2.

Budgets should be ratified from baseline evidence, not asserted as timeless
truth. Initial campaign targets can be relative:

- reduce initial detail transfer by at least 80% from 681 KiB;
- reduce brief health/status polling by at least 75% from 68 KiB;
- no pre-expansion response above 256 KiB;
- measure warm first-meaningful-render at desktop/mobile and set the blocking
  SLO only after the baseline fixture and machine are recorded.

The correct sequence is Playwright against the current UI, then API/reader
repairs, then v2. A rewrite before black-box journeys only transfers unknown
failures.

### 3. Session-to-commit archaeology is not currently credible

`polylogue/insights/session_commit.py` has three production-breaking issues:

- `_parse_git_log_blocks` splits on a marker whose placement precedes
  `--name-only` files; real git output gives the first commit zero files and can
  interpret later filenames as commit hashes;
- `extract_referenced_files` reads `block["input"]`, while production domain
  blocks expose `tool_input`;
- `persist_session_commits` is a no-op.

Live `session_commits` rows are all Codex `parser-git-meta` checkpoint/base
observations, not proof that a session authored a commit. `polylogue-212.2`
therefore has a false premise when it says all reads exist and a PR can be
resolved to its authoring session.

Do not make archaeology the flagship. Repair semantics first:

- use NUL/record-separated git output;
- consume production block envelopes;
- distinguish `checkpoint_observed` from `authored` evidence;
- persist and reverse-query typed edges;
- add a positive real-repository fixture;
- resolve the conflict with `polylogue-1a9`, which currently proposes deleting
  the no-op stubs.

The first public receipt benchmark does not need commit attribution and should
not wait on it.

### 4. Derived usage views do not uniformly self-heal

`polylogue-f2qv.5` records that provider-usage projections are written during
ingest but absent from convergence refresh. `polylogue-hjwr` proposes the right
general proof: full rebuild versus incremental convergence must agree, with an
auto-census that prevents new derived tables silently escaping the differential.

This remains high leverage, but implementation is not required in the first
operator-proof week unless core work is ahead of schedule. The active
`f2qv.2` implementation appears to be the uncommitted canonical-checkout work
Fable referenced; resolve and land that owner's work rather than duplicate it.

### 5. Frontier and coordination projections overstate their utility

The frontier classifier and 55 KiB coordination envelope are observational
tools, not dispatch contracts. They should degrade to compact ref-first views,
carry confidence, and never turn prose heuristics into ownership or execution
authority.

## Temporal Integrity Adjudication

Fable warned that the Receipts alignment rule depends on `cpf.5`, described as
a verified live temporal-laundering bug. This feedback is stale in its strongest
form:

- `polylogue-cpf.5` is now closed;
- `classify_aggregate_hwm_source` propagates the weakest contributor;
- focused temporal taxonomy tests currently pass: 64/64;
- the source has an AST audit for unjustified leaf classifier call shapes.

There is still a residual design weakness: `classify_profile_hwm_source` treats
any non-null `updated_at` as provider time and the AST audit verifies attribute
shape, not the parser's true provenance. That matters for cross-session
freshness claims.

The Receipts benchmark should avoid this substrate entirely for its core label:

- align claims, code-changing actions, verifier commands, and outcomes by
  `(session_id, message.position, variant_index, block.position)`;
- never use wall clock or `updated_at` to decide the within-session validity
  window;
- exclude cross-session temporal claims from v1;
- use explicit session refs for controlled specimens;
- treat timestamps only as displayed provenance with source class, not ordering
  authority.

Temporal integrity remains a trust-floor concern, but it is not a blocker for a
position-based claim/outcome benchmark.

## Demo And External-Proof Assessment

### Existing claim-vs-evidence artifact

The mature demo under `.agent/demos/claim-vs-evidence/` analyzes structured
tool failures and the next assistant turn. Its current private aggregate:

- 42,033 structured failures in the frame;
- 5,000 inspected in an origin-stratified bounded sample;
- 420 acknowledged next turn;
- 1,205 silent-proceed;
- 3,375 ambiguous;
- 24.1% silent lower bound in the selected frame;
- 50 labeled marker-calibration rows;
- reported marker precision 100%, recall 84.2%.

It has a public-safe aggregate wrapper, deterministic synthetic reproduction,
and cold-reader contract. Its limitations:

- it measures failure acknowledgment, not whether completion claims are true;
- row selection is deterministic/stratified but not a probability sample of the
  whole archive, so rates should not be generalized as population prevalence;
- 50 labels calibrate one acknowledgment marker, not claim support;
- the packet is based on live index v24 and is now stale relative to source;
- thin model cells lack explicit minimum-N refusal (`polylogue-jph5`);
- the demo registry validator checks packet shape, not command execution or ref
  resolution.

### Receipts benchmark relationship

The proposed `Receipts: Claim-to-Evidence Benchmark` extends, not supersedes,
the shipped campaign:

- existing artifact: after a structural failure, did the next assistant turn
  acknowledge/recover/silently proceed?
- Receipts: when an assistant explicitly claims tests/build/lint/typecheck
  passed, does ordered structural evidence support, contradict, or fail to
  establish that claim?

The shared infrastructure is extraction, action outcomes, packet format,
privacy discipline, and cold-reader proof. The measures and gold labels are
different. Keep both names and non-claims explicit.

`polylogue-3tl.16` should ledger Receipts as `capability` while the benchmark is
being built and `proven` only after protocol/label/audit thresholds pass.
`polylogue-3tl.3` multi-model leaderboard remains blocked on the base relation
and minimum-N policy.

### Rigorous v1 benchmark protocol

Scope:

- claim types: `test_passed`, `build_passed`, `lint_passed`,
  `typecheck_passed`;
- outcomes: `supported`, `contradicted`, `unknown`;
- 120 eligible final claims, seeded random selection within origin, 60 Claude
  Code and 60 Codex;
- no broad `fixed`, `done`, authored-commit, or merged-PR claims in v1.

Validity rule:

- freeze extractor/alignment code before labels are revealed;
- identify the last code-changing action before the claim using transcript
  positions;
- find the latest matching verifier command after that action and before the
  claim;
- success supports, structural failure contradicts;
- missing outcome, mixed command, ambiguous command family, or another later
  edit yields `unknown`.

The deployed v24 `actions` view uses a plain `(session_id, tool_id)` left join.
Canonical v26 fixed a verified cross-product bug by ranking repeated tool uses
and results in transcript order. Therefore the benchmark must not trust the v24
view directly. Its read-only extractor should embed the canonical ranked-use /
ranked-result CTE against v24 tables, or else build the selected sample in a
scratch v29 archive. The first option is cheaper and avoids a live rebuild, but
the extractor must carry the canonical query as versioned benchmark logic and
prove parity against a seeded repeated-tool-id fixture.

Blinding:

- each labeler receives a deidentified ClaimPacket: claim text, normalized
  command family, ordered action/outcome evidence, and necessary code-change
  markers;
- hide origin/model/agent identity, current classifier output, other labeler's
  answer, and sample-group labels;
- randomize packet order independently;
- use two different frontier model families, not two instances of one model.

Human audit:

- operator adjudicates every disagreement;
- operator audits 24 agreement cases, stratified up to eight per agreed class;
- if more than two audited agreement cases are wrong, or any systematic rubric
  failure appears, invalidate the automated-label shortcut and require a full
  human pass before publication;
- retain the audit seed and packet ids.

Statistics:

- report confusion matrix, precision/recall/F1 per class, macro-F1, abstention
  coverage, inter-rater agreement, and Wilson/bootstrap intervals as applicable;
- expected worst-case 95% binomial interval at n=120 is about +/-9 percentage
  points overall; at n=60 per origin it is about +/-13 points;
- say this prominently in the cold-reader packet;
- refuse subgroup claims below n=20;
- do not infer archive prevalence from the origin-balanced benchmark.

Privacy:

- tracked packet contains only operator-reviewed sanitized snippets, normalized
  commands/outcomes, aggregate metrics, and stable public refs;
- raw ref maps, full transcript previews, home paths, and private content remain
  ignored locally;
- packet validation scans for home/realm paths, emails, credentials, and
  unreviewed raw payloads.

### Context-pack uplift

The n=5 pilot is hopeful but not publishable:

- pack arm won 4/5; means 30.2/40 versus 22.8/40;
- packs were hand-written, not production-generated;
- all checkpoints came from one continuous devloop;
- one blind was compromised;
- effort budgets were soft;
- pair 3 showed a stale pack causing a confidently false status claim.

The most defensible interpretation is not "packs contain otherwise unavailable
information." Raw-live access is a superset. The plausible value is prepaid
synthesis: equal or better reconstruction with fewer tokens/tools/time.

`polylogue-x35k` correctly extends ContextImage with per-claim freshness and
verifier queries. `polylogue-e5b5` proposes cheap state-fact and stale-claim
micro-evals. These are better next steps than immediately spending on n=12-20.
They should be dropped from the first week if reliability/proof work slips.
`polylogue-57bg` remains blocked until production packs and micro-evals are not
obviously harmful.

## External Positioning

The public README already has a coherent category: Polylogue as a system of
record for AI work, closer to git for work around code than a chat viewer. The
constraint is demonstrated use, not another positioning rewrite.

The strongest external wedge is:

- cross-provider structural evidence, not self-reported assistant prose;
- local/private longitudinal history;
- evidence-linked context and verification;
- exportable, citable agent trajectories and outcomes.

Hermes already provides its own execution, memory, skill-learning, delegation,
and dashboard surfaces. Atropos provides trajectory/evaluation/training loops.
Polylogue should not compete with those. A targeted Hermes bridge and eventual
Atropos export are valuable because they turn one runtime's local state into a
cross-runtime evidence corpus, not because Polylogue should run Hermes.

The immediate externally legible artifact should remain cross-provider Receipts.
Hermes compatibility is an important reliability proof and targeted appendix,
not the headline until a safe, rich corpus exists.

## Resource And Coordination Model

Host:

- i7-13700K, 24 hardware threads;
- 32 GiB RAM;
- RTX 3080;
- `/realm` NVMe has ample space;
- observed test peak around 10.4 GiB RSS;
- severe IO pressure can occur during Borg compaction even when memory is fine.

Execution discipline:

- one coordinator plus at most three worktree-isolated workers;
- worktrees under `/realm/tmp/worktrees`;
- exactly one `heavy-db`/full-suite/live-archive diagnostic at a time;
- light source/test/browser work may proceed beside one heavy lane;
- pause heavy work during Borg or sustained IO full pressure;
- canonical archive access is read-only unless one explicit coordinator-owned
  write/rebuild lock exists;
- the coordinator owns Beads mutations to avoid shared-Dolt/write conflicts;
- workers commit each logical chunk before worktree cleanup.

Do not permanently lower build/test parallelism because of transient pressure.
Use current runtime evidence and one-shot containment.

## Fable Feedback: Point-By-Point Adjudication

### 1. Parallel meta-system and missing s7ae

Verdict: correct and strategy-changing.

The execution compiler should become a compact `planned_work` projection in the
existing s7ae envelope. `s7ae.5` should use it in the live proof. Beads remains
the authority. No new state machine, queue, active-loop packet, or scheduler is
added. Fix stale s7ae ACs that still name the conductor.

### 2. Oversubscribed by 2x-3x

Verdict: correct.

The earlier plan mixed campaign-critical reliability, broad trust-floor work,
future verification laws, context experiments, a benchmark, and Web UI
foundation. A 4/6 exit gate did not prevent mid-week thrash. The revised plan
has explicit tiers and a day-two stop/go decision below.

### 3. v29 ungrounded/operationally scary

Verdict: half right.

v29 is canonical source truth; v24 is deployed archive truth. The issue was not
hallucination but an unstated rebuild assumption. No live rebuild belongs in the
campaign. Report v24 where live evidence is used; use scratch v29 archives for
new-schema proof. `b5l` is not required to write parser/UI/test changes, but it
is a prerequisite for treating future live rebuilds as routine.

### 4. Benchmark epistemics

Verdict: correct.

Blinding, correlated labeler error, audit size/invalidation, expected interval
width, and relation to sru/3tl.16 are now explicit above. The artifact must call
itself a first benchmark, not a precise population estimate.

### 5. Temporal trust floor

Verdict: concern valid, cited current state stale.

`cpf.5` is fixed and 64 focused tests pass. Residual leaf-provenance weakness
exists. The benchmark avoids timestamp authority by aligning within-session
positions. Cross-session temporal claims stay out of v1.

### Smaller points

- Web budgets become measured relative targets before hard SLOs.
- R0 does not become a rushed completion gate; campaign-entry quality is the
  immediate gate.
- pairwise generation gets explicit case budgets and remains behind mandatory
  evidence cells.
- `f2qv.2` must be resolved with its current owner and existing uncommitted work,
  not reassigned.
- Bead surgery must happen before any future worktree dispatch.

## Revised One-Week Campaign

This is a future execution shape, not yet written into Beads.

### Tier 0: hard preflight, no workers yet

- Reconcile every new deliverable with existing Beads, especially s7ae.
- Amend `s7ae.5` away from conductor packet paths.
- Record the compact planned-work successor under s7ae/qra, not a new epic
  ontology.
- Reopen/supersede `fs1.1` with current Hermes evidence.
- Rewrite `1ilk` to cover the current UI first.
- Add the WorkflowProofSpec/gap-compiler bead.
- Add/extend the Receipts base-relation bead under the existing claim-vs-
  evidence/demo/claims-ledger program.
- Record explicit file ownership, proof commands, drop tier, and resource class
  on campaign members.
- Resolve the existing `f2qv.2` owner and dirty-checkout implementation before
  spawning overlapping work.

No worktree spawns until this tracker state is reviewable and graph lint passes.

### Tier 1: load-bearing deliverables

1. Hermes current-state compatibility plus observed-schema structural fixture.
2. Playwright harness against the current Web UI, with red canonical journeys.
3. WorkflowProofSpec MVP and gap report covering declared surface and observed
   source authority; no pairwise generator yet.
4. Receipts extractor/alignment dry run, then the 120-case protocol if the dry
   run is stable.
5. Compact `planned_work` s7ae projection used for at least one real dispatch,
   capped at 8 KiB and prohibited from authorizing unknown fields.

### Tier 2: fix what Tier 1 exposes

- Web metadata/message separation, previews/full drilldown, unsupported-control
  removal, and progressive tool disclosure.
- Hermes parser fidelity diagnostics and live read-only canary.
- proof-compiler anti-vacuity fixtures.
- benchmark privacy scan, label audit, metrics, cold-reader packet, claims-ledger
  entry.

### Tier 3: cut first when schedule slips

1. v2 Preact/Vite scaffold: do not start this week.
2. session-to-commit repair and D1 PR specimen: keep as execution-grade Beads,
   not campaign requirements.
3. `hjwr` and `g9f2` implementation: retain as next verification-law work.
4. `x35k`/`e5b5`: execute only if Tier 1 is green by the checkpoint.
5. broad usage/cost cluster beyond landing the already-active `f2qv.2` work.
6. pairwise proof generation, compact coordination telemetry write leg, and
   visual polish beyond structural reader usability.

### Day-two checkpoint

By the end of day two require:

- Hermes real v11 reproduction is encoded and the synthetic structural fixture
  reaches the parser;
- at least three current-UI Playwright journeys are red for the expected
  reasons, with traces/request sizes;
- `verify gaps` reports missing Web and Hermes authority cells;
- Receipts produces a 12-case blinded-format pilot without private leakage;
- planned-work projection produces one <=8 KiB candidate dispatch packet from
  Beads/s7ae.

If fewer than three of these five are true, cancel full benchmark labeling and
all Tier 3 work. Finish reliability, proof gaps, and tracker truth only.

### Lane allocation

- Coordinator: Beads/s7ae reconciliation, planned-work projection, integration,
  benchmark protocol/privacy/claims ledger.
- Worker A: WorkflowProofSpec/gap compiler and proof telemetry metadata.
- Worker B: Playwright current UI and reader/API repairs.
- Worker C: Hermes compatibility, then benchmark extraction only after Hermes
  lands.

This shape avoids file hotspots and preserves one free path for integration.

### Success/falsification gate

Core week success requires four of five:

1. current Hermes v11 is recognized and parsed through the supported path with
   privacy-safe regression proof;
2. canonical browser journeys pass on seeded-scale data and one read-only live
   canary;
3. the gap compiler detects deliberately removed Web/Hermes proofs;
4. Receipts produces either a valid benchmark packet or an honest negative
   methodological result;
5. one agent run is dispatched from a compact s7ae planned-work projection and
   its proof refs return to the coordination envelope.

If fewer than four land, pause new surfaces and narrow active work to archive,
source compatibility, evidence contracts, and operator journeys. This does not
mean deleting the long-range backlog. It means the project has not yet earned
further surface expansion.

## Immediate Next Actions

1. Review this note against the user's intent and Fable feedback.
2. Convert only the revised, non-duplicative work into Beads, with s7ae mapping
   and drop tiers explicit.
3. Correct stale Bead premises (`s7ae.5` conductor path, `fs1.1` closure,
   `1ilk` rewrite dependency, `212.2` authoring-link claim).
4. Resolve the active `f2qv.2` checkout/owner before any branch creation.
5. Start with red evidence: Hermes current schema, browser journeys, and proof
   gap report. Do not begin with framework scaffolding.

## Cross-Agent Refinement Update

The ongoing Fable/Codex exchange is distilled in
`.agent/scratch/2026-07-10-agent-dialogue.md`; full sessions remain available
through Polylogue refs. This is evidence, not a second planning authority.

New conclusions from that exchange:

- Name the recurring testing pathology **fixture/matrix vacuity**: a verifier
  proves the shape of its own registry/fixture/shadow contract rather than the
  product reality it claims. Behavioral/evidence coverage checkers need a
  failure witness: remove or corrupt the claimed proof and the checker must turn
  red. Cheap syntax/policy lints need an ordinary negative fixture, not a
  mutation framework.
- Run a read-only Receipts eligibility census before committing to 120 labels.
  Preserve origin balance: use `2 * min(60, eligible_claude, eligible_codex)`
  only when both origins have at least 40 eligible claims. Below that floor,
  publish the census and a methodological pilot/negative result rather than an
  origin-comparative score. Balance both labeler families across both claim
  origins and report labeler-family by origin agreement.
- A planned-work freshness contract needs composite content epochs, not mtimes:
  exact git HEAD plus normalized dirty-state hash; a hash of selected Bead rows
  and dependency states; projection-policy version/hash; and short-TTL live
  resource evidence. Any durable epoch mismatch invalidates the projection.
  Unknown ownership/proof/resource fields cannot authorize dispatch.
- Closed `s7ae.1` should remain closed. Create a successor for semantic
  projection and byte budgets, then make it a prerequisite of corrected
  `s7ae.5`. The old row-count AC was literally delivered; the 51-64 KiB defect
  is a newly measured requirement.
- Do not claim the affordance review selected 59 MCP tools and 34 CLI commands
  for retirement. Those were machine-generated `kill` candidates. The completed
  manual review retained 56/59 MCP tools and all 34 CLI commands; it produced
  one exact-duplicate retirement (`v6vy`) and two consolidation candidates
  (`moyt`). Treating the machine label as authority would reproduce the vacuity
  pathology. Proof obligations may exclude a surface only after an explicit
  reviewed `retiring` lifecycle decision names its replacement and proof.
- Dogfood campaign accounting by cohorting exact session/proof refs and
  reporting archived token/cost equivalence per landed verified outcome. Do not
  label this direct Codex/Claude subscription quota consumption unless the
  provider exposes that measurement.
- Preserve the invented Hermes superset fixture as an unknown-future-schema
  case: recognize the origin, report `unsupported_schema`. Add a distinct
  observed-v11 structural fixture for the currently supported path.

Cloud implementation should start only after the cloud-bootstrap Gate 0 is
merged (or equivalent environment bounds are supplied manually), then with
already execution-grade, private-data-free Beads: `k6fm` focused-run git
identity, `kj22` fuzz discovery, and `v6vy` duplicate MCP retirement.
Coordination allowlists and
WorkflowProofSpec become cloud lanes only after their successor Beads/specs are
adjudicated. Receipts corpus work, real Hermes fingerprinting, live Web canary,
archive migration/rebuild, and `s7ae.5` remain local.

GPT-5.6 Pro prompts and cloud packets live under `/realm/inbox/gpt-pro-sol/`.
The prompt set now includes a strategy-falsification task that may recommend
continue, narrow, pivot, or stop; the other design prompts should not smuggle
project continuation in as a premise.

The external `/realm/inbox/download/quota-burst-operating-plan-2026-07-10.md`
is useful as an operating-pattern source, not an execution board. Retain its
single Beads writer, immutable task snapshot, durable branch/outcome artifact,
wave review, environment-vs-code failure classification, and quota-as-dogfood
ideas. Reject or refresh its stale board: PRs 2626/2627 are merged, `b0b.1` is
closed, and `f2qv.2` is active locally. Its suggested six-to-eight first-wave
branches exceed unmeasured review/cloud capacity. Start with `k6fm`, `kj22`,
and `v6vy`; review the first return before expanding.

The quota plan found a real runway defect: `.claude/setup.sh` calls nonexistent
`devtools render-all --check` and hides the error; cloud settings omit bounded
pytest workers and a `/tmp` base-temp root. A cloud executor packet documents
the fix, but it must not launch until a coordinator-created Bead owns the
tracked change. Testmon seeding belongs behind one measured Claude and one
measured Codex environment run, not automatic rollout. Task-intent/outcome
documents may be ephemeral/ref-first PR artifacts; they must not become a new
committed mission database.

Operational update: the deployed durable `source.db` was migrated from schema
v2 to v3 on 2026-07-10 after a verified backup. `source.db` and `user.db` pass
`PRAGMA quick_check`; the daemon now reports only the expected derived-index
v24-versus-v29 mismatch. The verified backup manifest is under
`/realm/data/captures/polylogue/backups/2026-07-10-source-v2-pre-v3/`.

## Verification Performed During This Audit

- `devtools test tests/unit/sources/test_parsers_local_agent.py -k hermes_state_db`
  -> 5 synthetic tests passed while the real v11 DB was rejected.
- direct real Hermes probe -> `looks_like=False`, parser `ValueError`.
- `devtools verify closure-matrix` -> clean, 33 domains, demonstrating the
  matrix's limited authority.
- `devtools test tests/unit/insights/test_temporal_source_taxonomy.py` -> 64
  passed.
- private Chrome probe of `http://127.0.0.1:8766/` -> UI/load/request evidence
  summarized above.
- `devtools workspace frontier --json` -> 40-ready truncation and
  misclassification evidence.
- `python3 .agent/tools/delivery-gate-status.py --fresh --json` -> empty R0,
  A=40/60 closed, 93 unlabeled active Beads.
- live archive `PRAGMA user_version` -> 24; canonical source constant -> 29.
- real-git narration/probe of `session_commit.py` -> delimiter and block-shape
  defects summarized above.

## Open Questions Worth Further Investigation

- Can current v24 action evidence be sampled directly by a new benchmark tool
  without opening the archive through version-enforcing current APIs? The
  existing claim-vs-evidence devtool uses a raw read-only SQLite connection and
  suggests yes, but this needs a 12-case pilot using canonical rank-pairing
  semantics rather than the v24 `actions` view.
- Which minimal fields define Hermes state compatibility across v11-current and
  future reconciled schemas? Verify against upstream migration/reconciliation
  code rather than freeze one snapshot.
- What is the smallest browser fixture that reproduces 681 KiB detail and tool
  output pressure without importing private transcripts?
- Can s7ae default/self/planned-work views remain below 8 KiB while retaining
  useful conflicts, proof, and freshness? Current named views are 51-63 KiB and
  retain irrelevant arrays, so projection policy must be repaired before the
  target is judged.
- Does the current action view's rank-pairing v26 fix materially change v24
  claim/outcome labels? If so, the benchmark must either backport the pairing
  logic in its read query or use a scratch v29 sample and disclose the choice.
- How often do both frontier labelers make the same error on claim packets? The
  24-case human agreement audit is the first measurement, not a guarantee.
- Which current tests provide the highest defect yield per second/RSS, and can
  verify history drive proof selection without trusting stale testmon graphs?
## Cloud execution control proved 2026-07-10

The stable Codex Cloud control surface is the authenticated local CLI, not GUI
automation:

```text
codex cloud exec --env <environment-id> --branch master '<prompt>'
codex cloud list --json
codex cloud status <task-id>
codex cloud diff <task-id>
codex cloud apply <task-id>
```

Environment `polylogue` (`6a507da0a2108191a8070e2d5a90d240`) targets
`Sinity/polylogue`, runs `bash .claude/setup.sh` for both setup and maintenance,
enables post-setup caching and internet, and carries the five bounded synthetic
variables from `docs/cloud-agents.md`. PR #2631 / `c68585b8b` is the minimum
safe base. First canary: `polylogue-k6fm`, task
`task_e_6a507f7577c08320a27a5205c0c9522b`.

Operational rule: one canary first; inspect setup, actual diff, verification,
and publication durability before opening `kj22` or `v6vy`. Browser control is
unnecessary for coding lanes and is actively inferior on this host because the
private visible profile is shared with another agent. Claude Code Web is a
fallback after its account session limit resets.

## Strategy falsification and browser actuator update (2026-07-10)

GPT-5.6 Pro's first falsification memo recommends **NARROW**, not continuation
of the broad roadmap. It preserves acquisition/raw evidence, normalized
sessions/messages/blocks/actions, lineage, exact/structured search, stable refs,
fidelity, portable exports, and a compact read surface. It proposes freezing
most coordination/web/evaluation expansion behind a 30-day comparative gate:
Polylogue versus direct files/ripgrep/small SQLite, receipts, fidelity,
two-runtime coordination, and export/restore reproduction. The memo is saved at
`/realm/inbox/gpt-pro-sol/results/06-strategy-falsification.txt`.

This is serious evidence against the current breadth, but not yet decisive.
The source package omitted this note and the shared agent dialogue, and the
memo sometimes treats no adoption telemetry as evidence to freeze rather than
as a reason for a bounded value experiment. Both missing notes are now attached
to the same GPT conversation. The follow-up asks it to retain or amend the
verdict explicitly, reconcile `s7ae`/`1hj` with the conductor graveyard, and
produce a strict seven-day drop order. Until then, preserve the stop-gate idea
but do not perform broad Bead surgery from the first memo alone.

The existing Polylogue browser extension reverse channel is real and already
implements receiver queue -> extension poll -> addressed provider tab ->
composer fill/optional submit -> ack. A live no-submit ChatGPT canary passed
with both guards enabled against an isolated loopback receiver/spool. This
changes the execution assessment: Polylogue already owns a narrow actuator
substrate, not merely observation. It does **not** yet satisfy the desired cloud
control contract: `BrowserPostCommand` lacks attachments and model selection,
and duplicate tabs sharing a conversation id are ambiguous. Existing Bead
`polylogue-ptx` owns un-gating plus attachments; extend it with deterministic
tab selection/model evidence rather than create another control vocabulary.
