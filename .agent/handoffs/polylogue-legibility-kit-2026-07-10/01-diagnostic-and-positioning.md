# External legibility diagnosis and positioning

## Executive conclusion

Polylogue and Sinex do not primarily suffer from a lack of capability. They suffer from a mismatch between the sophistication of the substrate and the simplicity of the story available to an outsider.

Polylogue is much closer to externally presentable. Its README already names a category, ships a deterministic one-command demo, has public-safe visual tapes, and maintains a proof-artifact map. The remaining problem is that its strongest story is still fragmented: the README says “system of record,” the generated site says “Your AI memory,” the most compelling findings live under `.agent/demos`, the reader does not yet render tool activity with enough semantic force, and the launch path still asks a stranger to infer value from a construct inventory.

Sinex has the opposite shape. Its conceptual model is stronger than its public framing, but the README leads with implementation architecture and an illustrative SQL query. A stranger can mistake it for a local activity logger, an observability pipeline, or a PostgreSQL hobby project before encountering its genuinely distinctive ideas: source material versus interpretation, three clocks, replayable belief, explicit coverage error, authority-gated model outputs, and cross-source evidence. Its current deterministic demo is a smoke test rather than a thesis demonstration, and its best recall evidence is split between tracked agent artifacts and a shell script that still demonstrates mostly one source family.

The public category pair should be stable:

> **Polylogue is the local flight recorder and system of record for AI work.**

> **Sinex is the local evidence substrate for digital life and agent work.**

The joint story is equally simple:

> **Polylogue explains AI work; Sinex preserves the wider evidentiary world in which that work happened. In Sinex-backed mode, Sinex stores the durable transcript evidence and Polylogue remains the AI-work domain kernel and product.**

Everything else should support these three statements.

---

## 1. What an outsider must understand

A stranger should be able to answer the following after one screen, one minute, and ten minutes.

### After one screen

For Polylogue:

- What category is this?
- Which concrete problem does it solve?
- Why is it more than grep or a chat export viewer?
- What can I run immediately without exposing private data?

For Sinex:

- What category is this?
- Why is it more than an activity logger or event bus?
- What is unusual about its evidence and replay model?
- What can I run immediately?

### After one minute

For Polylogue:

- A transcript contains claims; Polylogue resolves those claims against structured tool evidence.
- Provider forks, continuations, and compactions make naïve counts wrong.
- Memory is reviewed and evidence-linked rather than silently injected.
- The archive is local-first and cross-provider.

For Sinex:

- Source bytes, interpreted events, current projections, and generated artifacts are different layers.
- Occurrence time, interpretation time, and persistence time are different clocks.
- A parser fix can reinterpret history without deleting the previous reading.
- A missing source must render as a coverage gap, not an empty result.

### After ten minutes

A reader should be able to run a demo, inspect the exact evidence packet, understand the architecture, see current limitations, and distinguish shipped capability from roadmap.

Both repositories currently require too much prior context to reach that point.

---

## 2. Polylogue: current external-legibility assessment

### What already works

Polylogue has several unusually strong public assets:

1. The README opens with a named category rather than “a tool for managing chats.”
2. The one-command deterministic tour is real product code, not a shell-only showcase.
3. The demo corpus has an explicit construct audit generated from fixtures and verification logic.
4. The proof-artifact page distinguishes facts, capabilities, and promises.
5. Public-safe GIFs are regenerated from committed tapes.
6. The project openly preserves negative findings and performance failures.
7. The query-first command shape is distinctive and legible.
8. The archive’s trust model is genuinely differentiated: structured tool outcomes, raw evidence, explicit caveats, and rebuildable projections.

These are better foundations than most pre-1.0 projects possess.

### What still obscures the project

#### 2.1 The generated site contradicts the README category

The README says “system of record for AI work.” The generated site title and footer say “Your AI memory.” “AI memory” places Polylogue in the wrong comparison set: vector memories, retrieval injectors, personal assistants, and vendor memory features. It also obscures audit, cost, lineage, and evidence.

The site should use the same category as the README. Recommended wording:

> **The local flight recorder for AI work**

Subhead:

> Search every provider. Read tool activity as work, not chat. Audit claims against outcomes. Resume from reviewed evidence.

#### 2.2 The README contains too much meta-instruction

“Skim this in seven minutes,” “the short version,” and repeated explanations of how to read the documentation make the text feel persuaded rather than demonstrated. The repository already has a Bead for a de-meta and de-persuasion pass (`polylogue-3tl.12`). It should be executed.

The README should show value in this order:

1. Category and one-sentence promise.
2. A visible tool-aware screenshot or recording.
3. Five questions the product answers.
4. One command.
5. One evidence chain.
6. Trust model.
7. Architecture and status.

#### 2.3 The current demo proves breadth before value

“Eleven sessions, 43 messages, 87 blocks, five origins” proves fixture coverage. It does not make a user care.

The current one-command tour should remain as the substrate proof, but the public landing page should narrate one compact incident:

- an agent claims tests pass;
- a structural tool result says otherwise;
- a fork duplicates a large prefix;
- a later session finally succeeds;
- Polylogue shows the claim, the evidence, the lineage, the corrected cost, and the resume point.

The construct audit belongs behind that story, not in front of it.

#### 2.4 The strongest finding is not presented as a first-class public page

The 24.1% silent-proceed lower-bound finding is compelling because it demonstrates a question ordinary chat viewers cannot answer. It is currently described in proof documentation and tracked packets, but there is no durable public finding page with:

- exact claim;
- corpus and sampling frame;
- method;
- structural oracle;
- calibration;
- limitations;
- public reproduction on synthetic data;
- stable URL.

This is the core of `polylogue-3tl.4` and `polylogue-3tl.16`. A static v1 should ship immediately rather than waiting for every eventual first-class finding object and evidence-basket feature.

#### 2.5 The transcript reader does not yet visually cash the ontology

Polylogue’s ontology understands tool calls, tool results, edits, shell exits, subagents, compaction, and evidence refs. The visual reader still risks looking like another transcript list until `polylogue-ap7`, `polylogue-37km`, and related renderer work land.

The reader must make the difference visible without explanation:

- shell command card with exit status and folded output;
- file edit as a diff;
- file read as a path and range;
- search as query plus result count;
- subagent dispatch as linked child work;
- compaction as a context boundary;
- structural failure highlighted independently of assistant prose;
- evidence and derivation badges with consistent vocabulary.

This is the single highest-value product change for external perception.

#### 2.6 “Why not grep?” is described but not demonstrated

The anti-grep argument should be one proof card, not a generic comparison table.

A good card uses the same fixture for both arms:

| Question | Grep | Polylogue |
|---|---|---|
| Find the word `pytest` | Yes | Yes |
| Pair tool call with structural result | Manual and provider-specific | Native |
| Determine exit-code failure | Manual parsing | Typed predicate |
| Avoid double-counting copied fork prefixes | No | Logical lineage view |
| Separate human text from injected protocol context | No | `material_origin` |
| Resolve a derived claim to source evidence | No | Stable refs |

The point is not that grep is bad. The point is that lexical matching and domain interpretation are different operations.

#### 2.7 Installation status is internally honest but externally awkward

The README’s first demo command assumes an installed `polylogue`, while installation documentation says PyPI is not yet a supported channel. The most honest first command today is the Nix path or a source-checkout command.

Until a tagged release and install matrix exist, use:

```bash
nix run github:Sinity/polylogue -- demo tour
```

or an explicit source-checkout path. Once `polylogue-3tl.7` and the first tag land, switch to `uvx polylogue demo tour`.

#### 2.8 Public claims are not yet mechanically governed

The project already understands that “every metric resolves to evidence” must apply to its own marketing. The claims ledger should be shipped as a simple YAML file now, even before it becomes a first-class finding projection.

Every quantitative, comparative, or benefit claim should be one of:

- `proven` — a particular evidence packet supports it;
- `capability` — code exists, but no measured outcome is claimed;
- `aspirational` — roadmap only;
- `retired` — no longer true.

#### 2.9 The roadmap is too large to communicate

The Beads graph is sophisticated, but a stranger does not need the entire graph. The public status page should name four near-term programs:

1. **Trust floor** — accounting, readiness, degraded modes, storage correctness.
2. **Read the work** — semantic transcript rendering and evidence drilldown.
3. **Prove the value** — receipts, lineage, cost, resume experiments.
4. **Sinex-backed future** — durable cross-domain evidence and ambient context.

The full Beads graph remains the authority; the public page is a projection.

---

## 3. Sinex: current external-legibility assessment

### What is genuinely distinctive

Sinex has several concepts that deserve much more prominence:

1. Source material is preserved separately from event interpretation.
2. Occurrence, interpretation, and persistence have separate clocks.
3. Events are append-only interpretations; replay can generate a new reading without destroying the old one.
4. Current state is a projection, not authority.
5. Missing coverage is a first-class result.
6. Model confidence does not grant authority; proposals require judgment or policy promotion.
7. High-volume capture, replay, lifecycle, and cross-source derivation exist in a live large deployment.
8. Activity evidence and system self-observation are separate lanes.
9. The project is architected to let new parsers and models reinterpret retained material.

Those ideas place Sinex in a category that ordinary activity loggers do not occupy.

### What the README currently gets wrong

#### 3.1 It leads with the implementation rather than the payoff

“Rust services, NATS JetStream, PostgreSQL” appears before the project’s conceptual advantage. That is useful for operators, but it causes category capture by infrastructure.

The first screen should instead say:

> Sinex records the evidence your tools leave behind—commands, files, browser activity, Git, system state, exports, and agent work—and turns it into a replayable history without erasing provenance or uncertainty.

Then:

> Think Nix for personal data: source material is content-addressed, interpretations are versioned computations, and current views can be rebuilt.

#### 3.2 The example SQL is not the product surface

The opening SQL query communicates an imagined relational payoff but not the actual `sinexctl` experience. It also invites the reaction “this is just a database.”

The first example should use current product commands, such as bounded context recall, source coverage, and evidence resolution.

#### 3.3 The README points to retired GitHub issues

The repository has explicitly retired GitHub Issues as the planning authority, yet the documentation table still links numerous conceptual topics to issue numbers. This is both stale and confusing.

Replace those links with:

- stable design docs;
- a generated roadmap/status page derived from Beads;
- Bead IDs where a roadmap reference is necessary.

#### 3.4 The deterministic demo is a smoke test, not a demonstration of value

`sinexctl ops verify --demo` proves that queries return expected lower bounds. It does not show why Sinex should exist.

The first public deterministic demo should be a bounded “moment” scenario with at least four independent source families and one deliberate coverage gap. The operator should ask:

> What happened around the failed build at 14:32?

The answer should show:

- browser research;
- shell commands;
- file changes;
- Git activity;
- an agent session;
- a source outage strip;
- exact evidence refs.

#### 3.5 The current recall demo overstates its category breadth

The tracked recall demo is honest that its headline uses one source plus derived structure. That honesty is good, but the demo should not remain the primary public proof of a cross-source system.

Keep it as a depth proof. Replace it as the hero with a truly multi-source fixture and a live field packet.

#### 3.6 “Retroactive privacy with originals preserved” needs semantic correction

The current demo idea is strong but the name is dangerous. Redacting derived views while retaining original sensitive bytes is not erasure and should not be described as privacy deletion.

Split the concepts:

- **Retroactive disclosure control** — views and exports change, originals remain under restricted operator authority.
- **Forgetting/excision** — raw material and all derived copies are physically removed according to a lifecycle contract.

The first can ship earlier. The second is the real deletion proof.

#### 3.7 The strongest architecture is invisible because it is temporal

Sinex’s best demonstration is not a dashboard screenshot. It is a before/after timeline:

1. Parser v1 interprets retained material incorrectly.
2. A fix is deployed.
3. Replay emits a new interpretation.
4. The current projection changes.
5. The old interpretation remains inspectable.
6. Derived descendants are invalidated and rebuilt.

This “the system changes its mind honestly” demo should become a headline artifact.

#### 3.8 Proof artifacts need a public shelf

Sinex has strong private or agent-oriented packets—production restore, recall, coverage audits—but no concise public claims map equivalent to Polylogue’s proof-artifact page.

Create a public page with three classes:

- deterministic public proof;
- private-archive field evidence with scrubbed aggregates;
- planned experiment.

The page should state what each artifact does **not** prove.

#### 3.9 The project needs a plain-language vocabulary bridge

The concepts page should translate internal language:

| Internal term | Public explanation |
|---|---|
| `ts_orig` | When the occurrence happened in the source world |
| `ts_coided` | When Sinex coined this interpretation |
| material anchor | Exact location in source evidence |
| automaton | Replayable derived computation |
| reducer | Rule that rebuilds current state from event history |
| settlement | Proof that an emitted item reached a terminal durable outcome |
| ClaimSupport | Structured support and uncertainty, not model confidence |
| reflection lane | Telemetry about Sinex itself, separate from captured activity |

This table will do more for public comprehension than another architecture diagram.

---

## 4. The joint positioning

The projects should not be marketed as competitors or as two versions of the same system.

### Polylogue’s center of gravity

- AI-session capture and provider normalization;
- tool-aware transcripts;
- logical session lineage;
- token and cost accounting;
- evidence-backed postmortems;
- reviewed assertions and context policy;
- agent memory and coordination UX.

### Sinex’s center of gravity

- source-material registry and content-addressed storage;
- temporal and provenance semantics;
- replay and interpretation history;
- cross-source evidence;
- source coverage and lifecycle;
- model-effect accounting;
- broad operator and agent context.

### The public combined sentence

> Polylogue tells you what happened in AI work. Sinex tells you what happened around it—and preserves the evidence well enough to reinterpret both later.

### The maximal backend story

The long-term architecture should be explicit but status-labelled:

- Sinex stores provider-native transcript artifacts, normalized transcript segments, attachments, durable transcript-domain events, judgments, context deliveries, and model effects.
- Polylogue defines the AI-work ontology, parses providers, composes logical conversations, renders transcripts, governs memory, and exposes the product.
- PostgreSQL hosts shared durable projections in backed mode.
- Polylogue SQLite remains a local edge projection, offline store, and standalone authority when Sinex is absent.

This is not a metadata bridge. It is a domain backend relationship.

---

## 5. What to put above the fold

### Polylogue

**Headline**

> The local flight recorder for AI work.

**Subhead**

> Search ChatGPT, Claude, Codex, Gemini, and coding-agent histories in one local archive. Read tool activity as work, audit claims against structural outcomes, understand cost and lineage, and hand the next agent reviewed context.

**Primary action**

```bash
nix run github:Sinity/polylogue -- demo tour
```

**Visual**

A 20–30 second recording of a single failure-to-evidence drilldown. Avoid a fast montage of unrelated commands.

**Proof strip**

- Private-data-free deterministic corpus.
- Structured tool outcomes, not prose classification.
- Physical and logical session views.
- Raw evidence and caveats.

### Sinex

**Headline**

> The local evidence substrate for digital life and agent work.

**Subhead**

> Preserve source material from your machine, interpret it as typed events, replay those interpretations when code or models improve, and query a cross-source history that says when evidence is missing.

**Primary action**

```bash
sinexctl ops verify --demo
```

This should be replaced by the new moment demo as soon as it exists.

**Visual**

A timeline with five source-family tracks, one deliberate outage gap, and click-through evidence anchors.

**Proof strip**

- Source bytes separate from interpretations.
- Three explicit clocks.
- Append-only correction and replay.
- Coverage gaps are data.

---

## 6. Public claims discipline

Both projects should share the same public claim grammar.

### Proven

A particular artifact supports a bounded statement. The artifact identifies its corpus, method, oracle, caveats, and regeneration command.

### Capability

The system contains a working implementation, but no general outcome is claimed. Example: “Polylogue can compile a bounded context image.” This is not the same as “Polylogue improves agent performance.”

### Field evidence

A result measured on a private deployment. It may demonstrate scale or reveal a failure mode, but it is not a population benchmark.

### Experimental

A preregistered comparative result with declared arms, metrics, and stopping rules. Small pilots remain labelled pilots.

### Aspirational

A Beads-owned future direction.

### Retired

A statement that was once true but no longer describes the product.

No public page should use “proven” as a synonym for “tests exist.”

---

## 7. Visual language

The projects need a shared visual vocabulary for epistemic state.

Recommended badges:

- **Observed** — direct material or structural source evidence.
- **Derived** — reproducible computation over cited evidence.
- **Reviewed** — accepted by an operator or declared authority.
- **Candidate** — proposed, not accepted.
- **Missing** — expected evidence absent or source unavailable.
- **Stale** — projection or assertion is behind its relevant frontier.
- **Unsupported** — requested conclusion cannot be established.

Polylogue can use these in tool cards, findings, and context packs. Sinex can use them in timelines, source coverage, and derivation cards. Shared vocabulary reinforces the relationship without merging the products.

---

## 8. Immediate highest-leverage changes

### Polylogue

1. Change the generated site category and footer.
2. Merge `polylogue-ap7` / transcript-reading visual work before recording new media.
3. Ship a static public claims ledger and findings page.
4. Reframe the one-command demo around a single incident story.
5. Publish the “why not grep?” proof card.
6. Run an install matrix and cut a first tag.
7. Record one slow, comprehensible evidence drilldown—not another command montage.

### Sinex

1. Replace the README with payoff-first copy and remove retired GitHub issue links.
2. Add a concepts page and public proof-artifact map.
3. Build a deterministic multi-source moment demo with a deliberate coverage gap.
4. Promote “the system changes its mind honestly” and “self-diagnosing outage” as headline demos.
5. Reframe the privacy demo into disclosure control versus actual forgetting.
6. Publish a Beads-derived roadmap projection.
7. Add a public page explaining the Polylogue backend direction.

---

## 9. The launch order

Polylogue should launch first because it already has:

- a comprehensible user problem;
- a runnable deterministic corpus;
- public-safe media;
- strong proof packets;
- a lightweight standalone install story;
- a differentiated agent relevance story.

Sinex should not wait for every source or every derived product. It should become externally legible in parallel, then make its first major public push around a paired demonstration:

1. a source goes silent and the system diagnoses the gap;
2. a parser is fixed and months of retained evidence are reinterpreted without erasing the prior reading.

The combined launch moment should follow a real end-to-end Sinex-backed Polylogue proof:

- transcript bytes stored in Sinex;
- stable Polylogue refs survive replay;
- SQLite is rebuilt from Sinex;
- a Polylogue session view includes ambient Sinex context;
- an Agent Work Packet links intent, agent work, machine evidence, verification, and outcome.

That is the point at which the joint architecture becomes visibly more than an integration diagram.
