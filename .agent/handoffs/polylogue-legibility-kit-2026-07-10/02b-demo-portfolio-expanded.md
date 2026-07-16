# Demo portfolio v2: impressive because it is falsifiable

## Executive recommendation

Do not launch either project with a feature tour.

Launch with a sequence of short demonstrations in which the audience can state the expected answer before the command runs, inspect the structural oracle afterward, and see at least one negative or missing-evidence control.

The public portfolio should have four layers:

1. **A 30-second category proof** — one scene that immediately distinguishes the product from its nearest wrong category.
2. **A three-minute thesis proof** — one coherent incident with several evidence surfaces.
3. **A ten-minute trust suite** — replay, lineage, coverage, authority, and anti-demo.
4. **Research demonstrations** — preregistered comparisons with measured uncertainty, kept separate from product capability demos.

The shared deterministic corpus should be **Incident 14:32**, described in `08a-joint-public-story.md`. Every demo below should use that corpus unless the construct inherently requires a live or private deployment.

## Why the existing portfolio needs reshaping

The current Beads contain many strong ideas, but the list mixes four different things:

- capability demonstrations;
- system verification;
- scientific experiments;
- stagecraft or launch moments.

Those should not share one standard of evidence.

A capability demo asks whether a particular operation exists and behaves correctly on a declared fixture.

A system verification asks whether a safety or durability invariant survives fault injection or scale.

An experiment asks whether one intervention improves an outcome relative to a baseline.

Stagecraft makes a capability memorable but does not strengthen the claim.

The redesign below retains most Beads but changes sequencing, names, controls, and claim boundaries.

---

# 1. Demo doctrine

## 1.1 Every demo has one primary construct

A demo may contain many visible features, but only one claim should determine success.

Bad:

> Polylogue proves memory, audit, cost, lineage, and productivity in one five-minute session.

Good:

> Given one assistant success claim and one paired tool result, Polylogue classifies the structural outcome from the result rather than the prose.

Secondary features can enrich the scene, but the primary claim must have one oracle.

## 1.2 Declare the oracle before running

Every demo packet should define:

```yaml
claim: What the demo is allowed to establish
oracle: The field, invariant, or independently computed result that decides success
falsifier: The observation that would make the demo fail
controls: Positive, negative, and missing-evidence cases
scope: What the result does not generalize to
```

The UI should expose this information in a collapsible “How this is judged” panel.

## 1.3 Use structural ground truth where available

Preferred oracles, in order:

1. externally authoritative source state;
2. provider- or runtime-structured outcome;
3. exact material bytes or record anchors;
4. independently implemented baseline query;
5. human adjudication under a written rubric;
6. text heuristic, clearly labeled.

A demo about test failure should use exit status or structured `is_error`, not the presence of “failed” in text.

## 1.4 Include negative and missing-evidence controls

A positive result alone proves little when the fixture was hand-picked.

Each public demo should include at least one of:

- the same phrase with a successful structural result;
- an assistant claim with no supporting tool result;
- a source outage that forces `coverage_gap`;
- an ambiguous duplicate that routes to adjudication;
- a stale assertion that remains noninjectable;
- a query that correctly returns `not_supported`.

## 1.5 Distinguish deterministic fixture proof from field evidence

Every page should have two lanes:

- **Reproduce publicly** — deterministic synthetic or redistributable corpus;
- **Observed in deployment** — private-archive aggregate or operational proof with exact date, corpus, method, and caveat.

Never imply that a deterministic fixture estimates prevalence. Never imply that a private personal archive is representative of all users or models.

## 1.6 Preserve the baseline arm

The relevant baseline is not “nothing.” It is the strongest realistic alternative:

- grep or provider search for transcript questions;
- raw source command for Sinex recall;
- physical transcript count for logical-lineage accounting;
- live refs alone for context reconstruction;
- first-import counts for idempotence;
- old parser semantics for replay;
- query without source-health metadata for coverage.

## 1.7 A demo packet is a product artifact

A demo is complete only when it emits:

- `packet.yaml` or `packet.json`;
- human report;
- terminal transcript;
- exact command list;
- input manifest and content hashes;
- expected and observed result;
- refs to evidence objects;
- caveats and unsupported claims;
- environment and software revision;
- optional recording generated from the same script.

The recording is an alternate rendering of the packet, not a separate hand-edited truth.

## 1.8 An impressive demo should survive interruption

The demo runner should support:

```text
prepare → inspect → run step N → verify → render → clean
```

A presenter should be able to jump directly to any step, rerun verification, and open the packet after a failed live rendering.

---

# 2. The shared Evidence Lab corpus: Incident 14:32

## 2.1 Narrative

A developer asks an agent to repair a flaky clock-sensitive test in a small repository. The first agent edits the wrong fixture, receives a nonzero test result, but says the issue is resolved. A fork or continuation copies the prior transcript. Context compaction omits the failed experiment. A second agent receives a bounded brief, observes repository and ambient-machine evidence, repairs the correct fixture, and verifies the result.

The corpus also includes one capture outage and one parser semantics revision.

## 2.2 Required objects

### AI-work material

- one browser/design conversation;
- two coding-agent physical sessions;
- one copied-prefix continuation or fork;
- one fresh subagent;
- one compaction boundary and generated summary;
- user, assistant, tool-use, tool-result, runtime-context, and protocol material;
- one attachment with retained bytes;
- provider usage with cache lanes and reasoning output;
- one assistant success claim contradicted by a structural result;
- one later successful structural result;
- one candidate assertion, one accepted assertion, and one stale/rejected assertion;
- one context image and delivery snapshot.

### Ambient Sinex material

- shell command outside the transcript;
- Git status, diff, and commit evidence;
- filesystem observations;
- two browser visits;
- desktop focus interval;
- one deliberately missing source interval;
- source material snapshot and occurrence payload;
- parser v1 and v2 interpretation of one record;
- one ambiguous cross-source duplicate;
- one reflection event that must not appear in ordinary activity recall.

### Task intent

- one Bead with acceptance criteria;
- one dependency transition;
- one premature close attempt or stale status observation;
- one final evidence-backed completion.

## 2.3 Fixture generators and independent oracles

The fixture should be generated from a declarative scenario file, not copied from product output.

Independent oracle files should include:

- expected physical sessions and message ordinals;
- expected logical session composition;
- expected structural tool outcomes;
- expected usage-lane totals;
- expected source coverage intervals;
- expected parser semantic diff;
- expected material content hashes;
- expected accepted/rejected assertion state;
- expected context-delivery manifest;
- expected Agent Work Packet legs.

The product reads source fixtures; the verifier reads the independent oracle. Generating both from the same reducer would create circular proof.

## 2.4 Public-safe design

Use invented code, domains, paths, people, model names where licensing or privacy requires it. Preserve realistic provider structure and failure modes. The corpus should contain no copied private conversation text, real secret, personal hostname, or absolute user path.

A secret-control demo should use unmistakably synthetic tokens such as:

```text
SINEX_DEMO_SECRET_DO_NOT_USE_7YQ9
```

and assert both suppression in a declared public view and preservation in the restricted evidence lane where that is the claimed policy.

---

# 3. Polylogue public demos

## P1. The Receipts

**Beads:** `polylogue-212.2`, implementation follow-up `polylogue-xyel`; supports `polylogue-3tl.15`.

**Category role:** 30-second launch hero.

### Claim

Polylogue can compare an assistant claim with the provider-structured tool outcome supporting or contradicting it, and resolve both to exact archive evidence.

### Scene

Split screen:

```text
CLAIM                                      OBSERVED
“All tests pass; the fix is complete.”    pytest · exit 1 · 2 failed
message:<ref>                              tool-result:<ref>
```

Selecting either side opens the normalized block and source-material anchor. A follow-up badge says `not acknowledged in next assistant turn` only if the declared marker or structural rule supports it.

### Oracle

The paired tool-result block has `exit_code != 0` or `is_error = true`.

### Controls

- positive failure case;
- successful test command containing the word “failed” in captured output or fixture name, which must not be classified as failure merely by text;
- unsupported claim with no paired tool result, which must render `insufficient_evidence`;
- malformed tool pairing, which must render a caveat rather than guess.

### Baseline

A grep command finds the words `tests pass` and `pytest`; it does not pair the call/result, classify the structural outcome, or resolve logical lineage.

### What it proves

A concrete category distinction from chat viewers and text search.

### What it does not prove

Prevalence of agent overclaiming, general model reliability, or causal intent.

### Presentation

This should be the first GIF, homepage card, and social clip. It should complete in under 30 seconds and use no aggregate statistics.

---

## P2. Count It Once

**Beads:** should be added under `polylogue-212`; implementation uses `polylogue-4ts` lineage work and existing logical/physical views.

**Category role:** second launch hero; uniquely agent-specific.

### Claim

Polylogue preserves copied transcript evidence while distinguishing physical storage/usage from logical unique work.

### Scene

A parent session contains 100 message units. A fork physically contains the copied 100-unit prefix and a 20-unit unique tail.

The screen first shows the naïve result:

```text
physical transcript total: 220 units
```

Then the logical lineage view:

```text
unique logical work: 120 units
copied replay:       100 units
```

A topology graph highlights the shared prefix and unique tail. Clicking the replay amount resolves to the edge and message ranges.

### Oracle

An independently declared prefix identity manifest and expected logical composition.

### Controls

- a fresh subagent with similar text but no copied-prefix edge must remain distinct;
- a continuation with a copied prefix must deduplicate logically;
- a compaction summary remains a real new message and is not removed as a duplicate;
- a near-match prefix with one changed source record must follow declared identity policy rather than fuzzy text matching.

### Baseline

Sum physical sessions or provider-reported totals without lineage composition.

### What it proves

Why transcript files and run traces are insufficient for longitudinal agent accounting.

### What it does not prove

That every provider’s lineage can always be reconstructed or that provider billing should be reduced by logical replay.

### Presentation

Use a simple animated ribbon: copied prefix folds into one shared lane; the raw artifacts remain visible underneath. Avoid a generic node graph as the only visualization.

---

## P3. Context Autopsy

**Beads:** context compiler and snapshot work under `polylogue-37t`; supports compaction program `polylogue-gjg`.

**Category role:** strongest AI-memory differentiation.

### Claim

Polylogue can show exactly which evidence and reviewed assertions were delivered to an agent, what was omitted under budget, and which material came from generated context rather than a human.

### Scene

The second agent makes a wrong assumption. The operator opens the delivery snapshot:

```text
included
  accepted lesson: clock fixture is process-scoped
  latest successful command
  relevant file diff

omitted
  failed experiment from session A — budget
  browser source — unavailable

excluded by policy
  stale candidate assertion — inject:false
```

The UI then compares the snapshot with the complete evidence available at the delivery time.

### Oracle

The immutable context-image manifest and actual delivery record, not a reconstruction from the final prompt text alone.

### Controls

- a stale candidate exists but must not appear;
- an accepted assertion appears with evidence refs;
- one source is unavailable and is reported as a gap;
- a segment selected by the compiler but redacted before delivery must be distinguished from delivered content.

### Baseline

A provider transcript showing a generated summary without selection or omission provenance.

### What it proves

Context can be audited as an input to agent behavior.

### What it does not prove

That the omitted evidence caused the failure. Causal claims require a controlled experiment.

---

## P4. Resume Under Oath

**Beads:** reframe `polylogue-212.6`; experiment substrate should align with Sinex `sinex-cem.12` and memory program `polylogue-37t`.

**Category role:** research headline after launch, not launch blocker.

### Claim

A declared Polylogue resume packet improves a preregistered reconstruction or continuation outcome relative to a strong raw-reference baseline on the same checkpoint.

### Why the current demo idea needs strengthening

“Find abandoned work, generate a brief, and continue it” demonstrates workflow, but the presenter already knows the intended answer and can choose a favorable session. It does not establish that the packet helped.

### Protocol

- sample checkpoints using a rule fixed before scoring;
- pair two fresh agent instances per checkpoint;
- arm A receives bounded live raw refs and equal tool access;
- arm B receives the same access plus the generated packet;
- hide arm identity from scorers where feasible;
- use identical budgets and stopping rules;
- score factual reconstruction, first useful action, unsupported claims, and time/tool calls;
- preserve all prompts, outputs, and evidence;
- include stale-packet negative controls.

### Oracle

Human or rule-based scoring under a preregistered rubric, with paired analysis and uncertainty intervals.

### Falsifier

The packet arm does not improve the preregistered primary outcome or introduces more unsupported state.

### Public claim discipline

A pilot with a few correlated checkpoints is hypothesis-generating. Do not headline a general uplift until the production pipeline, sampling, and independent scoring are credible.

---

## P5. Cost Truth, Not “Cost by Outcome” Yet

**Beads:** `polylogue-f2qv.2`–`.5`, `polylogue-5hf`, and a narrowed precursor to `polylogue-212.3`.

**Category role:** trust-suite demo.

### Claim

Polylogue keeps provider input, fresh input, cache reads/writes, output, reasoning, API-list estimates, and subscription-credit views separate and explains every calculation.

### Scene

A crafted Codex event reports inclusive input and cached input. The old additive interpretation double bills the cache. The current normalization shows:

```text
provider input = fresh input + cache read
catalog estimate uses disjoint lanes
subscription credit remains a separate view
```

Then show physical versus logical lineage cost without claiming either is an invoice.

### Oracle

Crafted provider payload plus independent arithmetic fixture.

### Controls

- Claude-style exclusive input lane, which must not be subtracted twice;
- missing pricing catalog row, which renders unpriced coverage;
- subscription plan with no defensible currency conversion, which remains credits rather than dollars;
- reasoning output tracked separately.

### Why not headline cost by outcome now

Outcome-conditioned cost adds another uncertain join: what counts as abandoned, failed, or successful work? Until outcome evidence and coverage are strong, the honest demo is accounting semantics, not productivity economics.

`polylogue-212.3` should later become a measured analytic demonstration with explicit denominator coverage rather than launch copy.

---

## P6. The Honest Refusal

**Bead:** `polylogue-212.8` closed.

**Category role:** mandatory trust-suite companion.

### Claim

When the archive lacks required modalities or support, the demo surface returns `not_supported` with exact missing evidence instead of manufacturing a result.

### Scene

Ask for minute-by-minute operator reconstruction during an interval where the deterministic corpus has transcript and Git evidence but no browser or desktop source.

Expected result:

```text
verdict: not_supported
missing:
  desktop focus source
  browser continuity for 14:31–14:36
  required evidence ref for causal attribution
available:
  agent messages
  tool result
  Git commit
```

### Oracle

The scenario’s source-coverage manifest.

### Requirement

Publish this beside successful demos, not in a limitations appendix.

---

## P7. Forensic Question Set

**Bead:** reframe `polylogue-212.1`.

**Category role:** ten-minute analyst demonstration.

### Current problem

“Take one completed multi-hour session and ask questions a tracer cannot answer” is compelling but weakly bounded. A hand-selected private session and retrospective questions allow presenter degrees of freedom.

### Better design

Use Incident 14:32 and a fixed question set:

1. At which exact evidence object did the first false assumption enter?
2. Which file changed most before the first failing test?
3. Which prior attempt shared the same structural failure signature?
4. Which claim had no supporting tool result?
5. Which evidence did the second agent receive?
6. Which physical transcript content was copied rather than new?
7. What remains unknown because of the source outage?

For each question, ship:

- expected answer or adjudication rubric;
- query and result refs;
- independently checked evidence path;
- a baseline answer from provider search or grep;
- a declared failure condition.

The private-archive variant can demonstrate scale but not alter the question set after seeing the data.

---

## P8. The Session That Watched Itself

**Bead:** `polylogue-212.5`.

**Category role:** launch stagecraft after the deterministic proof suite is reliable.

### Claim

A live supported capture path can make a newly emitted session observation queryable within the declared freshness contract.

### Strengthening

The demo should not simply query itself mid-session. It should display four frontiers:

```text
provider/source observation
raw material durable
normalized projection current
search/read surface current
```

The latency claim is the difference among recorded timestamps/frontiers, not a stopwatch held by the presenter.

### Controls

- pause the projection worker and show a loud `search_not_current` state;
- resume it and show convergence;
- query an unsupported provider path and show the capability boundary.

---

## P9. Behavioral Archaeology

**Bead:** `polylogue-212.4` closed.

**Category role:** supporting long-horizon story, not first contact.

The existing idea should be presented as a method for longitudinal self-analysis, not as a performance leaderboard or diagnosis of model quality. Every trend must expose capture coverage, provider mix, schema changes, and denominator shifts.

Use it after the audience understands the evidence model.

---

# 4. Sinex public demos

## S1. The Missing Source

**Bead:** strengthen `sinex-cem.2` with `sinex-jdp` and `sinex-60r`.

**Category role:** 30-second Sinex launch hero.

### Claim

Sinex distinguishes “no observed activity” from “the source was unable to provide evidence.”

### Scene

The same interval is queried twice.

Without source-health metadata, a timeline appears empty.

With Sinex:

```text
14:31–14:36 browser evidence unavailable
reason: checkpoint stalled after material 0042
last confirmed occurrence: 14:30:58
result completeness: degraded
```

A terminal event inside the interval still appears, proving that the query itself was not empty.

### Oracle

A deliberate source fault plus independent source sequence/checkpoint manifest.

### Controls

- true quiet interval with healthy source, which should report complete/empty;
- source process alive but emitting semantically empty records, which should report quality failure rather than healthy coverage;
- missing runtime binding;
- material acquired but parser behind.

### What it proves

Sinex is an evidence system, not merely a row collector.

### What it does not prove

That every possible outage is automatically detectable.

---

## S2. The System Changes Its Mind Honestly

**Bead:** `sinex-cem.14`; combine conceptually with `sinex-cem.3` but keep two scales.

**Category role:** strongest conceptual Sinex demo.

### Claim

Sinex can preserve source material and an earlier interpretation, admit a corrected interpretation, rebuild downstream state, and explain the semantic difference.

### Scene

Parser semantics v1 reads a growing focus interval as 20 seconds. A later snapshot or revised record shows 140 seconds.

The UI presents:

```text
current interpretation   140 s   semantics v2
prior interpretation      20 s   semantics v1   archived
source evidence           exact material refs
affected products         attention span · project attribution
```

A semantic-diff view shows which downstream outputs changed and which did not.

### Oracle

Known source record revisions and independent expected reducer output.

### Controls

- replay with identical semantics should not create a semantic change;
- unrelated downstream product remains unchanged;
- missing replay evidence produces a gap rather than a corrected value;
- stable domain occurrence remains the same while interpretation event ID changes.

### Scale variants

- **S2a:** one interval revision, fast and visual;
- **S2b:** parser bug fixed over months of history (`sinex-cem.3`), a deeper operational proof after replay robustness lands.

---

## S3. Import It Twice

**Bead:** `sinex-cem.13`.

**Category role:** trust-suite hero.

### Claim

Re-importing the same occurrence material produces zero duplicate current occurrences, while replay under changed semantics intentionally produces a new interpretation.

### Scene

```text
first import
  137 occurrences admitted

second import, identical bytes
  0 new current occurrences
  137 duplicates suppressed

replay under semantics v2
  137 new interpretations
  137 stable occurrence identities
  3 current projections changed
```

### Oracle

Independent occurrence manifest plus current-row counts and interpretation lineage.

### Controls

- export grows with one new record;
- same content arrives from another source with ambiguous identity and becomes an adjudication candidate;
- hash collision fixture or malformed identity must fail closed;
- parser semantics change without source change.

### Value

This one demo teaches the most subtle Sinex idea: occurrence idempotence and interpretation non-idempotence are both correct.

---

## S4. Disclosure Control, Not “Retroactive Privacy”

**Bead:** rename/reframe `sinex-cem.1`.

### Why the current name is risky

“Retroactive privacy” can be heard as secure deletion. The stated demo preserves originals under operator authority and changes an interpretation or public view. That is valuable, but it is disclosure control, redaction, or derived-view policy—not erasure.

### Claim

A newly applied privacy policy can suppress a planted secret from declared public/query/model-input views without reingesting source material, while restricted evidence remains available under the stated authority model.

### Controls

- exact planted secret is suppressed in the public view;
- neighboring harmless token remains visible;
- restricted raw-material resolution still requires explicit capability;
- cache, report, vector, and context-pack behavior is checked;
- a separate future deletion demo proves physical excision when that lifecycle exists.

### Falsifier

The token appears in any surface claimed to be covered, or the system describes retention as deletion.

---

## S5. Reconstruct the Moment, Not Tuesday Yet

**Bead:** narrow precursor to `sinex-cem.8`; build on closed recall work.

### Current problem

A full-day reconstruction is visually impressive but invites completeness theater. A dozen sources with uneven capture can produce a beautiful but epistemically weak timeline.

### Better first demo

Reconstruct one 12-minute incident from a declared set of sources:

- terminal;
- Git;
- filesystem;
- browser;
- desktop focus;
- Polylogue transcript.

Show a source matrix above the timeline:

| Source | Coverage | Evidence strength |
|---|---|---|
| terminal | complete | direct structured history |
| Git | snapshot-backed | repository evidence |
| browser | gap 14:31–14:36 | incomplete |
| desktop | complete | source snapshot |
| Polylogue | complete revision | provider + normalized material |

Every timeline item resolves to bytes or source events.

Only after this bounded demo is trustworthy should `reconstruct my Tuesday` scale the same contract to a full day.

---

## S6. Kill It Mid-Thought

**Bead:** `sinex-cem.15`.

**Category role:** engineering proof, not landing-page hero.

### Claim

At declared killpoints, every input is either durably admitted, terminally settled, or eligible for replay after restart; no source progress silently advances past lost evidence.

### Protocol

- deterministic input sequence;
- sampled killpoint catalog;
- SIGKILL rather than graceful stop;
- restart and settle;
- reconcile source records, checkpoints, transport state, event rows, and derived outputs;
- repeat enough times to cover each killpoint class;
- emit counterexample packet on first mismatch.

### Oracle

Independent input manifest and receipt barrier, not final aggregate count alone.

### What it proves

A bounded crash-safety contract for tested killpoints.

### What it does not prove

Absence of all possible data-loss bugs.

---

## S7. Run Twice, Pay Once

**Bead:** `sinex-cem.4`.

**Category role:** model-plane trust demo after the worker lands.

### Claim

A repeated deterministic model effect with the same complete recipe identity reuses the recorded result and does not incur another provider call or budget debit.

### Recipe identity must include

- input content digest;
- canonicalization version;
- chunk selector and chunking version;
- provider and model revision;
- dimensions and task/input type;
- normalization policy;
- prompt or instruction version where applicable.

### Controls

Change exactly one recipe component and prove that reuse does not occur. Include a failed or partial prior effect and prove it is not treated as a valid cache hit.

---

## S8. Retrieval Earns Its Keep

**Beads:** `sinex-cem.5`, `.6`, `.12`.

**Category role:** research result, not launch hero.

### Better protocol

- create query/evidence judgments before running competing retrieval lanes;
- define lexical, vector, and hybrid recipes completely;
- use the same candidate corpus and privacy filter;
- report hit@k, reciprocal rank, abstention, latency, and model-effect cost;
- analyze by query type rather than one aggregate number;
- include queries where lexical should win and where semantic retrieval might help;
- publish negative results and kill pgvector for the product path if it does not pay rent.

The lane duel with adjudicated promotion should run only after this simpler evaluation substrate works.

---

## S9. Blinded Resumption Duel

**Bead:** `sinex-cem.7`.

This should be a joint Sinex–Polylogue experiment, not a Sinex-only demo. Polylogue owns AI-work context semantics; Sinex supplies ambient evidence and experimental provenance.

Use the protocol described under P4. The intervention arms can later compare:

- raw refs;
- Polylogue-only context;
- Polylogue plus Sinex ambient evidence;
- stale packet negative control.

Do not start with all four arms. Establish a credible two-arm protocol first.

---

## S10. Closed-Loop Actuation

**Bead:** `sinex-cem.9`.

Defer from first public launch.

Actuation expands the threat model and is not necessary to establish Sinex’s category. When built, the demo should use an operator-declared controller, a bounded reversible action, an independently observed effect, and a dead-man/timeout policy. The model proposes or selects within policy; it does not register an arbitrary persistent loop.

---

## S11. Devloop as Source

**Bead:** `sinex-cem.10`.

Use as dogfood evidence and recurring content rather than a primary product claim. A strong version asks a fixed question about a Sinex development incident and resolves the answer across Beads, Git, shell, Polylogue, and Sinex operations.

---

# 5. Joint demos

## J1. The World Around the Claim

**Launch role:** three-minute flagship.

This is the combined demonstration described in `08a-joint-public-story.md`.

Primary construct: a claim can be audited against both transcript-structured outcomes and independently captured machine evidence while preserving source gaps.

The impressive visual is a vertical evidence stack:

```text
intent       Bead and acceptance criteria
context      exact packet delivered to the agent
claim        assistant message
transcript   tool call/result and lineage
machine      terminal, files, Git, browser, focus
outcome      verification and commit
memory       proposed, accepted, stale assertions
coverage     what was not observed
```

The user can collapse every layer except the claim and outcome, then expand to receipts.

## J2. Agent Work Packet

Primary construct: a bounded work object can join independent domain refs without pretending they are one intrinsic event.

Packet legs:

- task intent;
- agent sessions and subagents;
- repository/worktree/branch;
- files and commits;
- commands and test results;
- context deliveries;
- source materials;
- verified outcomes;
- accepted lessons;
- missing legs and coverage.

The packet should be useful even when incomplete. Missing CI must render as a gap, not “CI did not run,” unless authoritative evidence supports that statement.

## J3. Stale Memory Trap

Primary construct: reviewed memory is useful only when freshness and source state are part of the contract.

A once-correct assertion references commit A. Commit B invalidates the relevant file. Sinex observes the change; Polylogue marks the assertion stale or routes it for review; the context compiler excludes or labels it.

Controls:

- unrelated file change does not stale the assertion;
- same assertion without a scope/evidence dependency cannot be automatically declared stale and is routed for judgment;
- stale packet arm demonstrates the actual failure mode.

## J4. Compaction Autopsy

Primary construct: distinguish what existed in full history, what the compaction summary retained, what later context retrieval restored, and what the downstream agent received.

This is an analysis demo, not a causal claim. A later paired experiment can test whether regrounding changes performance.

## J5. Rebuild the Product

Primary construct: in Sinex-backed mode, every durable Polylogue evidence and judgment object can reconstruct Polylogue’s rebuildable SQLite state.

Protocol:

1. ingest Incident 14:32 into Sinex;
2. use Polylogue normally;
3. record manifest and public refs;
4. delete all rebuildable Polylogue SQLite tiers;
5. rebuild from Sinex;
6. compare sessions, messages, blocks, topology, usage, assertions, context deliveries, FTS documents, and refs;
7. classify every difference as defect, intentionally local state, unsupported legacy material, or nondeterministic projection;
8. rerun flagship demos.

This is the decisive proof of the maximal backend architecture.

---

# 6. Portfolio scoring

Score each candidate before implementation. Suggested 1–5 scale:

- **category separation** — does it demonstrate something the nearest competitor cannot naturally claim?
- **oracle strength** — is success decided by structural or independent evidence?
- **control quality** — are positive, negative, and missing-evidence controls present?
- **reproducibility** — can a stranger run it without private data?
- **visual immediacy** — can the primary claim be understood in 30 seconds?
- **implementation leverage** — does it land reusable capability rather than one-off script logic?
- **scope honesty** — can limitations be stated without undermining the actual claim?
- **operational reliability** — can it run repeatedly on one machine under a bounded reset contract?

Recommended first-wave scores:

| Demo | Category | Oracle | Controls | Repro | Visual | Leverage | Recommendation |
|---|---:|---:|---:|---:|---:|---:|---|
| P1 The Receipts | 5 | 5 | 5 | 5 | 5 | 5 | Launch hero |
| P2 Count It Once | 5 | 5 | 5 | 5 | 5 | 5 | Launch hero |
| P6 Honest Refusal | 5 | 5 | 5 | 5 | 4 | 5 | Launch trust card |
| S1 Missing Source | 5 | 5 | 5 | 5 | 5 | 5 | Sinex launch hero |
| S2 Changes Mind Honestly | 5 | 5 | 5 | 5 | 5 | 5 | Sinex thesis hero |
| S3 Import It Twice | 5 | 5 | 5 | 5 | 4 | 5 | Trust suite |
| J1 World Around Claim | 5 | 5 | 5 | 5 | 5 | 5 | Joint flagship |
| P5 Cost Truth | 4 | 5 | 5 | 5 | 4 | 5 | Trust suite |
| P3 Context Autopsy | 5 | 5 | 5 | 5 | 4 | 5 | Second wave |
| J5 Rebuild Product | 5 | 5 | 4 | 5 | 3 | 5 | Architecture proof |
| S6 Crash-No-Loss | 4 | 5 | 5 | 4 | 4 | 5 | Engineering proof |
| P4/J resumption duel | 5 | 3–4 | 5 | 3 | 4 | 5 | Research after substrate |
| S8 retrieval benchmark | 3 | 4 | 5 | 4 | 2 | 4 | Internal/research |
| Cost by outcome | 4 | 2 today | 2 | 3 | 5 | 4 | Defer |
| Model leaderboard | 2 | 2 | 2 | 3 | 5 | 2 | Do not prioritize |
| Full-day Tuesday | 4 | 3 | 3 | 2 | 5 | 4 | Scale after bounded moment |
| Actuation | 3 | 4 | 5 | 3 | 5 | 3 | Later, safety-heavy |

---

# 7. Launch sequence

## Wave 0: evidence infrastructure

Before public recordings:

- demo packet v2 schema and validator;
- scenario manifest and independent oracle;
- public claims ledger;
- reproducible reset and one-command runner;
- renderers for evidence refs, caveats, unsupported states, and source coverage;
- deterministic visual capture from the same run output.

## Wave 1: category proof

Ship together:

- Polylogue P1 The Receipts;
- Polylogue P2 Count It Once;
- Polylogue P6 Honest Refusal;
- Sinex S1 The Missing Source;
- Sinex S3 Import It Twice;
- rewritten READMEs and docs landing pages.

These require no productivity-uplift claim.

## Wave 2: conceptual depth

- Sinex S2 Changes Mind Honestly;
- Polylogue P3 Context Autopsy;
- Joint J1 World Around the Claim;
- bounded moment reconstruction;
- public finding page for claim-vs-evidence field evidence.

## Wave 3: architecture and durability

- Joint J5 Rebuild the Product;
- Sinex S6 Kill It Mid-Thought;
- disclosure-control demo;
- shared model-effect ledger.

## Wave 4: measured benefit

- preregistered resumption duel;
- retrieval hit@k evaluation;
- outcome-conditioned cost only after outcome coverage is demonstrated;
- multi-agent coordination proof.

---

# 8. Recording and presentation grammar

Every recording should use the same visual grammar:

- upper-left: question or claim;
- upper-right: declared oracle and current verdict;
- center: product surface;
- bottom drawer: refs, material anchors, caveats, and reproduction command;
- consistent badges: `observed`, `derived`, `reviewed`, `candidate`, `missing`, `degraded`, `not_supported`;
- no unlabelled animation or counter whose origin cannot be inspected;
- no scrolling through hundreds of rows to create an impression of scale;
- one click from every headline number to a bounded evidence set.

Use real command output. Avoid simulated terminal typing when a fixed-speed replay of the actual transcript is more reliable and honest.

## The memorable ending

The combined three-minute demo should end by asking a tempting unsupported question. The product should refuse it and show exactly which source was missing.

The audience should leave with this impression:

> The system is powerful because it can reconstruct a great deal, and trustworthy because it knows when it cannot.
