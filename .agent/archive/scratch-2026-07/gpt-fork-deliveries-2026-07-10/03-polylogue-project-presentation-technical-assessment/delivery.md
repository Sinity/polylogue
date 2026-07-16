## Turn 20 (role=assistant)

# Polylogue: project presentation and technical assessment

I inspected the supplied snapshot generated on July 10, 2026: the working tree, bundled Git history, source and tests, architecture and design documents, proof artifacts, and the complete Beads export. I treated Beads as the planning authority; GitHub issues and pull requests are not used below as the roadmap. I did not execute the entire test suite, so the maturity assessment is based on repository structure, committed verification artifacts, test inventory, and recorded live-archive runs rather than a fresh runtime certification.

## Executive thesis

**Polylogue turns AI conversations and agent executions into a local, cross-provider, evidence-addressable system of record for work.**

That description is more precise than “chat archive,” “agent memory,” or “LLM observability.”

A chat archive stores text. Polylogue tries to store and reconstruct the work represented by that text:

- prompts and model responses;
- tool calls and structural tool results;
- shell exits, failures, file operations, and observed events;
- forks, continuations, subagents, and compactions;
- token usage, cache lanes, model attribution, and cost estimates;
- attachments and original source artifacts;
- user annotations, corrections, judgments, and handoff state;
- the context supplied to a later agent;
- links from all derived interpretations back to inspectable evidence.

The repository’s own analogy—“closer to Git for the work around your code”—is a good one. Git records source states and changes. Polylogue attempts to record the surrounding AI-mediated process: what was requested, what the agent attempted, what tools actually returned, where it failed, how much context was replayed, what conclusions were later reviewed, and what another agent should receive when work resumes.

The deeper product is therefore not a transcript browser. It is an **epistemic and operational substrate for AI work**: a place where evidence, interpretations, memory, and agent coordination can coexist without being confused with one another.

---

# 1. The problem Polylogue is solving

AI-assisted work currently falls through several gaps.

First, the work is fragmented by vendor and runtime. A ChatGPT export, a Claude Code JSONL session, a Codex rollout, a Gemini CLI history, an Antigravity trajectory, and a browser chat may all describe the same project, but they use different schemas, identifiers, tool representations, usage conventions, and lineage models.

Second, an LLM’s context window is not durable memory. Even when a vendor preserves a conversation, that does not provide portable, queryable, project-level memory. Context compaction can remove details. A new session may inherit only a summary. A different agent or vendor normally starts without the prior work.

Third, an agent transcript mixes claims and evidence. An assistant can say “the tests now pass,” but the reliable evidence is the structured tool result, process exit code, output, and possibly the resulting commit. Treating prose as proof is a recurring source of false confidence.

Fourth, agent logs routinely double-count work. Forks, resumes, and subagents may contain copied prefixes. Compaction may replace a large history with a generated summary. Naively summing every physical transcript exaggerates message counts, tokens, cost, and activity.

Fifth, most memory systems optimize retrieval before establishing trust. They embed text, retrieve a similar summary, and inject it into a later prompt. That can be useful, but an incorrect or stale summary can become self-reinforcing machine folklore. Polylogue’s design instead asks:

1. What is the source evidence?
2. What interpretation was derived from it?
3. Who authored or approved that interpretation?
4. Is it stale?
5. Is it eligible for context injection?
6. What exactly was supplied to the next agent?

That ordering is the project’s strongest conceptual contribution.

---

# 2. The central concepts

## Evidence is stronger than plausible prose

Polylogue assigns more authority to provider structure than to conversational wording.

For example, a tool call with `is_error=true` or a nonzero exit code is a failure even when the assistant proceeds as though it succeeded. Conversely, the word “error” appearing in prose is not enough to establish a failed operation.

This principle drives several parts of the system:

- provider parsers retain structured tool-use and tool-result blocks;
- actions are derived from those blocks;
- failures are determined from structural outcomes;
- public refs identify exact sessions, messages, blocks, runs, events, assertions, and context snapshots;
- analytics carry caveats when the underlying evidence is incomplete;
- unsupported inferences are represented as unavailable, candidate, or unknown rather than quietly guessed.

The product is willing to return less information in order to avoid manufacturing certainty. That is unusual in AI tooling, where a polished narrative often wins over an incomplete but honest result.

## Source evidence and user judgment are durable; projections are disposable

The project’s principal storage rule is:

> Preserve acquired evidence and irreplaceable user-authored state. Rebuild search indexes, analytics, embeddings, and operational telemetry.

This produces a clean distinction between four classes of information:

1. **What the external system emitted.** Raw files, hook events, exports, telemetry, and acquired bytes.
2. **What Polylogue normalized from it.** Sessions, messages, blocks, actions, topology, and usage rows.
3. **What Polylogue inferred or materialized.** Profiles, work events, phases, threads, costs, search indexes, vectors, and aggregates.
4. **What a human or reviewed process asserted.** Marks, corrections, judgments, lessons, caveats, handoffs, and approved memory.

This resembles an evidence ledger with CQRS-style projections. It is not pure event sourcing, but it has the same useful property: read models can be thrown away and reconstructed without redefining historical truth.

## Origin is not the same thing as provider

The code distinguishes the public `Origin` of archived material from the legacy/runtime `Provider`.

That matters because “OpenAI” or “Anthropic” is not enough to describe the semantics of a record. A ChatGPT export and a Codex coding-agent rollout have different capture guarantees, lineage conventions, tool structures, and usage accounting even when they come from the same company.

Current origin vocabulary includes:

- `chatgpt-export`
- `claude-ai-export`
- `claude-code-session`
- `codex-session`
- `gemini-cli-session`
- `aistudio-drive`
- `hermes-session`
- `antigravity-session`
- `grok-export`
- `unknown-export`

Grok exists in the vocabulary but is not a complete current parser path. This is an example of the project distinguishing a reserved identity from an implemented evidence contract.

The `Source` abstraction also carries family, runtime root, and originating lab. The goal is to describe where evidence came from without collapsing capture format, vendor, runtime, and product into one overloaded field.

## Role is not the same thing as authoredness

Polylogue separately models:

- `Role`: user, assistant, system, tool;
- `MessageType`: ordinary message, summary, tool use, tool result, thinking, context, protocol;
- `MaterialOrigin`: human-authored, assistant-authored, operator command, runtime protocol, runtime context, tool result, generated context pack, generated analysis pack, or unknown.

This is a sophisticated and important distinction.

A provider may encode injected runtime context as a user-role message. A compaction summary might appear in a conversational role while actually being generated context. An operator command and a human’s natural-language request may share the same envelope role but have very different meanings.

For agent safety, analytics, training-data preparation, and memory injection, “who occupied the API role?” is not enough. Polylogue tries to answer “what kind of material was this, and who or what produced it?”

## A physical session is not necessarily a logical unit of work

Providers represent continuation in several incompatible ways:

- a new session can copy the entire previous prefix;
- a fork can include the parent transcript followed by a divergent tail;
- a subagent can start with a fresh context;
- a compaction process can replace prior history with a summary;
- a compaction helper agent can contain almost a complete copy of the parent merely to produce that summary.

Polylogue models these as lineage edges and composition semantics rather than treating every file as independent work.

The intended invariant is:

- messages are stored once where they originated;
- inherited prefixes are represented by edges;
- copy-bearing child sessions store only their unique tail in the derived index;
- fresh subagents remain distinct;
- compaction summaries remain real messages at explicit context boundaries;
- composed reads reconstruct what a logical session contains;
- aggregate counts do not charge copied prefixes repeatedly;
- raw provider artifacts remain preserved even when the normalized index deduplicates their content.

This distinction is essential for accurate agent analytics. In one tracked live-archive proof artifact, the physical session corpus accounted for approximately 399.9 billion tokens, while the logical high-water view accounted for approximately 292.9 billion. Roughly 107 billion tokens were attributable to replayed lineage rather than distinct work. That is a single private archive, not a universal rate, but it demonstrates why ordinary transcript counting is badly misleading for long-running agents.

## Memory is a reviewed assertion, not merely retrieved text

User-owned state is consolidated around an `assertions` model. Assertion kinds include marks, annotations, corrections, suppressions, tags, metadata, notes, decisions, caveats, lessons, blockers, handoffs, judgments, prompt evaluations, transform candidates, and pathologies.

Assertions have lifecycle states such as:

- candidate;
- active;
- accepted;
- rejected;
- deferred;
- superseded;
- inactive;
- deleted.

They also have an explicit context policy, whose default is effectively `inject: false`.

That default is fundamental. An agent-generated observation does not automatically become future instruction. The planned memory loop is:

```text
source evidence
    ↓
agent or system proposes a candidate claim
    ↓
operator/user reviews or judges it
    ↓
accepted claim becomes durable
    ↓
context scheduler considers it under budget, trust, scope, and freshness rules
    ↓
compiled context cites the claim and its evidence
```

This prevents an agent from writing a hallucination into memory and then treating its own earlier hallucination as authoritative evidence.

## Context itself is evidence

The implemented context compiler takes a declarative `ContextSpec` containing seed queries or refs, requested views, terminal query units, token and message budgets, assertion policy, candidate policy, and redaction policy.

It emits a `ContextImage` with:

- bounded segments;
- object refs;
- evidence refs;
- assertion refs;
- token estimates;
- omissions and omission reasons;
- caveats;
- lossiness descriptions;
- the selection strategy and redaction policy.

At an actual delivery boundary, Polylogue can construct a `ContextSnapshotRecord`. That records which segments and evidence refs were delivered, under what boundary and inheritance mode.

This means a later analysis can ask not merely “what did the agent output?” but also:

> What information did the agent receive, what was omitted, and what was the provenance of that information?

That is important for debugging, evaluation, compaction analysis, reproducibility, and assigning responsibility for failures.

## Readiness is not binary

The daemon and operational model distinguish states such as:

- database openable;
- raw evidence materialized;
- derived projections converged;
- FTS current;
- search ready;
- embeddings caught up;
- performance claims measurable.

The active Bead `polylogue-avg` aims to expose these as a “what may be claimed” vocabulary. A schema-version match might mean the archive is openable, but not that every source row has been materialized or that search results are complete.

This is another unusual strength. The project treats operational readiness as an epistemic state, not merely an HTTP health check.

---

# 3. Architecture

The repository describes four architectural rings:

1. **Archive substrate** — acquisition, detection, parsing, normalization, persistence, and query.
2. **Derived read models** — profiles, work events, phases, threads, costs, FTS, and vectors.
3. **Surfaces** — CLI, MCP, Python API, daemon HTTP/web reader, TUI/dashboard, and renderers.
4. **Verification and maintenance** — schemas, deterministic demos, readiness checks, devtools, property tests, experiments, and proof artifacts.

An operational view looks like this:

```text
Chat exports      Coding-agent files      Hooks      Browser capture      OTLP
      \                    |                 |               |              /
       └────────── acquire / classify / shape-detect / validate ──────────┘
                                      |
                         source.db + SHA-256 blob store
                         durable acquisition evidence
                                      |
                    provider parser → normalized archive model
                                      |
                                  index.db
          sessions / messages / blocks / actions / topology / FTS / insights
                           |                         |
                    embeddings.db                user.db
                 optional vector projection    reviewed assertions
                           \                         /
                            └──── shared query layer ────┐
                                                       |
                  CLI | MCP | Python API | daemon/web | TUI | OTel export
                                                       |
                           humans, agents, experiments, reports
                                      |
                                   ops.db
                   convergence debt, cursors, attempts, telemetry
```

## Acquisition and provider detection

Provider files are detected by structure rather than by filenames alone. Files are first classified into an artifact taxonomy such as:

- session document;
- session record stream;
- subagent stream;
- sidecar metadata;
- session index;
- bridge pointer;
- supplementary metadata;
- unknown artifact.

Provider-specific parsers then normalize those records into shared models.

A provider integration is therefore not considered complete merely because one parser can read one sample. The project expects a complete path to include detection, artifact classification, raw acquisition, parser behavior, normalization, schemas, fixtures, query/read coverage, import explanation, caveats, and tests.

Capture paths include local files, exported archives, Google Drive acquisition for AI Studio/Gemini material, coding-agent hooks, a browser extension and local receiver, Antigravity language-server export with artifact fallback, and OTLP ingestion.

Repeated ingestion is idempotent by content hash. The session content hash is SHA-256 over an NFC-normalized payload including title, timestamps, messages, and attachments while excluding editable user metadata such as tags. Reimporting the same evidence should coalesce rather than create another history.

## The five SQLite tiers

The current code has independently versioned tiers:

| Tier | Current code version | Responsibility | Durability |
|---|---:|---|---|
| `source.db` | 3 | Raw sessions, artifacts, hook events, sidecars, OTLP source evidence, blob references, acquisition and validation state | Archival evidence |
| `index.db` | 29 | Normalized sessions, messages, blocks, actions, topology, attachments, FTS, costs, profiles, work events, phases, threads, runs and other read models | Rebuildable |
| `embeddings.db` | 1 | SQLite-vec vectors, metadata, status and catch-up state | Rebuildable but potentially costly |
| `user.db` | 4 | Assertions, reviewed notes, corrections, judgments, saved views, recall packs and settings | Irreplaceable |
| `ops.db` | 1 | Cursors, attempts, convergence debt, daemon stages, catch-up runs and local telemetry | Disposable |

The blob store is content-addressed by SHA-256 and prefix-sharded across 256 directories. One documented production archive contained roughly 24,000 blobs and 42 GB of stored material.

The conceptual model is stronger than the current prose documentation in one respect: `docs/schema.md` is visibly stale. Its version table still says source v1, index v11, and user v3, while the code is at v3, v29, and v4. It also speaks broadly about a globally fresh-only model, whereas the current implementation has additive migrations for durable source and user tiers and rebuild behavior for derived tiers. The DDL under `polylogue/storage/sqlite/archive_tiers/` is correctly identified as the authority, but the stale table is a current documentation defect.

## Normalized archive model

The central normalized structure is approximately:

```text
Origin / Source
   └── Session
         ├── Messages
         │     ├── role
         │     ├── material origin
         │     ├── message type
         │     ├── model and token lanes
         │     └── Content blocks
         │             ├── text
         │             ├── thinking/reasoning
         │             ├── tool use
         │             ├── tool result
         │             ├── attachments
         │             └── protocol/context material
         ├── Derived actions
         ├── Attachments
         ├── Events and runs
         ├── Usage and cost evidence
         └── Topology edges
                ├── continuation
                ├── fork/prefix sharing
                ├── fresh subagent
                └── compaction/context boundary
```

Content blocks are first-class. “Action” is a normalized view over tool-related blocks rather than an independent source of truth. Tool-use and tool-result pairing is session-scoped because provider-local tool IDs are not globally unique.

Cross-session relationships live in `session_links`. A logical session root is materialized in the session profile, allowing physical and logical counts to coexist.

## Derived insights

The read-model layer derives structures useful to both people and agents:

- session profiles;
- work events;
- workflow phases;
- latency profiles;
- thread membership;
- session and model usage;
- cost summaries and rollups;
- tool-use distributions;
- resume candidates;
- terminal states;
- pathologies and postmortem bundles;
- correlations with files and Git activity;
- archive coverage and convergence debt.

Not all derived constructs have the same evidentiary strength. The project maintains an “insight rigor” discipline: structural outcomes are stronger than text classifiers, and every analysis should disclose evidence type, confidence, caveats, and unsupported cases.

## Search and query model

The CLI is query-first:

```bash
polylogue find "pytest" then read --view messages
polylogue find 'repo:polylogue since:7d' then analyze --facets
polylogue find 'repo:polylogue' then read --view correlation
```

The query language is more than FTS syntax. It supports:

- explicit Boolean composition;
- typed fields and ranges;
- structural `exists` predicates;
- action sequences;
- session pipelines;
- grouping, aggregation, and counts;
- lineage-aware queries;
- lexical, semantic, and hybrid retrieval;
- terminal units such as messages, blocks, actions, files, assertions, runs, observed events, and context snapshots.

The retrieval lanes are deliberately explicit:

- **dialogue** — lexical FTS over conversational material;
- **actions** — lexical search over action/tool blocks;
- **semantic** — vector-only similarity;
- **hybrid** — reciprocal-rank fusion of lexical and vector results.

Lexical search remains the default. Embeddings are optional rather than foundational, which means the archive remains useful without an external embedding key and avoids presenting vector similarity as evidence.

Public refs such as `session:…`, `message:…`, `block:…`, and `assertion:…` resolve through a typed resolver. A malformed or missing ref is not silently widened into a broad text search. That fail-closed behavior is particularly valuable when an agent is chaining tool outputs.

## Daemon and convergence

`polylogued` watches source locations, ingests changes, materializes normalized rows, refreshes FTS and insights, catches up embeddings when enabled, exposes local HTTP surfaces, and records operational debt.

The project’s “automagic invariants” doctrine says that maintainable derived state should heal automatically. Manual CLI commands are intended for diagnostics or break-glass operation, not routine synchronization.

FTS freshness is a hard invariant: stale search should block or report unavailability instead of returning an apparently complete but partial result set.

Recent master work shows this operational layer being actively hardened. The latest commits in the snapshot:

- bounded daemon HTTP archive-query concurrency and added per-request timeout behavior;
- normalized ChatGPT recipient-addressed tool calls into tool-use blocks rather than raw JSON text;
- corrected a work-event text matcher to use word boundaries;
- routed reset mutations through the shared mutation contract;
- repaired three query-path bugs found in production smoke testing.

This is the pattern of a system in intensive dogfooding and trust hardening rather than a static prototype.

## Surfaces

The primary surfaces are:

**CLI.** The most complete human/operator interface, including query, read, analysis, import, maintenance, context, and demo commands.

**MCP server.** The test-enforced contract currently contains 96 tools, four fixed resources, one resource template, and 12 prompts. Tools cover search, typed query units, ref resolution, context compilation, session topology, annotations and judgments, coordination, costs, usage, postmortems, pathologies, readiness, and maintenance.

**Python API.** An async facade over the same archive substrate rather than a separate implementation.

**Daemon HTTP and web reader.** Local read/query APIs, browserless evidence routes, SSE, metrics, and a web shell.

**Browser capture.** A Manifest V3 extension and dedicated local receiver. Capture is opt-in per site, uses a token distinct from the daemon token, prefers provider-native payloads, and falls back to DOM-derived material where necessary.

**TUI/dashboard and renderers.** Present, but the semantic transcript experience remains a current work area. `polylogue-ap7` explicitly targets tool-aware cards for edits, shell commands, file reads, searches, subagents, web fetches, and MCP tools across terminal and web backends.

**OpenTelemetry projection.** Inbound OTLP data can be correlated locally. Outbound projection maps runs and actions to spans and messages/events/context snapshots to log-like records while retaining Polylogue refs as the canonical navigation mechanism. Tool output and absolute local paths are omitted by default.

## Security model

The present security model is local and single-operator oriented:

- loopback binding by default;
- bearer-token authentication;
- explicit insecure override for remote binding;
- Origin allowlisting on mutating browser-accessible routes;
- a distinct browser-capture token;
- no permissive CORS preflight behavior;
- bounded request bodies and local route contracts.

Raw archive content is not generally redacted at rest. It can contain source code, secrets, personal conversations, tool output, and file paths. The project delegates disk encryption to the operating system.

Explicit forgetting and evidence excision are not finished. The Beads security program `polylogue-kwsb` owns that work. This means the local-first architecture reduces third-party exposure, but the archive itself is a highly sensitive asset and currently lacks a complete selective-erasure story.

---

# 4. A concrete end-to-end workflow

Consider a coding agent asked to repair a flaky test.

1. The agent session starts. A hook records the session boundary and can inject a bounded context preamble containing relevant lineage, recent related sessions, project state, and approved assertions.

2. The user prompt is captured before provider transformations such as paste expansion obscure its original boundary.

3. The agent searches files, edits code, and runs tests. Pre-tool and post-tool events preserve tool identity, arguments, outcomes, failures, permission decisions, and timing that might not survive in the provider’s ordinary transcript file.

4. The source artifact and hook records enter `source.db`; large bytes are placed in the content-addressed blob store.

5. A provider parser normalizes messages, content blocks, roles, material origins, token lanes, tool calls, and relationships. The normalized data enters `index.db`.

6. FTS and insight projections are refreshed. The session receives a profile, work-event sequence, terminal state, latency information, cost evidence, and lineage membership.

7. A later query can ask for sessions touching the affected file, a sequence of edit followed by failed test followed by successful test, or sessions that ended with an unacknowledged structural failure.

8. A postmortem or user review can record a candidate lesson such as “this test flakes when the clock fixture is shared.” That lesson is not automatically injected.

9. The operator accepts, rejects, edits, or defers it. An accepted assertion receives evidence refs and an explicit context policy.

10. During a later continuation, the context compiler selects that assertion and related source evidence under a token budget. It records omissions and caveats.

11. When the context is delivered, a context snapshot records what the new agent actually received.

That workflow illustrates the project’s intended closed loop:

```text
capture → normalize → inspect → judge → remember → deliver → measure
```

Most AI memory systems concentrate on the “remember → deliver” portion. Polylogue’s value is that it builds the evidence and review machinery before trusting the memory.

---

# 5. Why this matters for agents and LLM systems

## It externalizes state without pretending the model owns that state

An LLM session is temporary computational context. It is not a stable database, and a summary inside that context is not independently trustworthy.

Polylogue makes state external, inspectable, and provider-independent. The model can query it, but the archive does not disappear when the model, vendor, or context window changes.

This is especially relevant to coding agents, where meaningful work may span dozens of sessions, branches, worktrees, subagents, and compactions. A new agent needs more than conversational similarity; it needs project state, failed approaches, live blockers, relevant files, prior decisions, and evidence that those statements remain current.

## It provides a reliability layer above agent self-reporting

Agents are often evaluated by final text or task success. Polylogue enables a richer reliability analysis:

- Did a tool structurally fail?
- Did the agent acknowledge the failure?
- Did it continue silently?
- Did a later action repair the problem?
- Did it claim success before evidence existed?
- What context had been supplied at the time?
- Was a remembered claim accepted, stale, or merely a candidate?

A tracked claim-vs-evidence experiment examined an origin-stratified sample of 5,000 structured failures from a frame of 42,033. It found a 24.1% lower bound for “silent proceed on the next turn,” with a large ambiguous remainder. The acknowledgment marker’s calibration reported 100% precision and 84.2% recall on 50 labeled rows. Those are private-archive and method-specific findings, not general claims about all agents, but they demonstrate the kind of question the substrate can answer.

## It treats memory quality as an experimental question

The project has already run paired context-reconstruction experiments instead of simply declaring that summaries improve agents.

The current five-pair pilot compared:

- live raw-reference access;
- the same live access plus a bounded handoff summary.

The handoff arm won four of five pairs, with mean scores of 30.2/40 versus 22.8/40. The losing handoff case is more important than the headline: the packet asserted that two work items were already closed at a checkpoint where they were not. The raw-reference arm, forced to derive state from live evidence, avoided that falsehood.

The project correctly labels this as non-publishable: five correlated checkpoints, partially compromised blinding, weaker process isolation, and hand-written rather than production-generated packs. The result supports continuing the experiment, not claiming general uplift.

This is exactly the right design pressure for agent memory. Summarization helps bounded cognition, but stale synthesis creates a new class of confident error. Polylogue’s planned freshness metadata, judgment queue, citation safety, context ledger, and production-pipeline experiment all follow from that counterexample.

## It can make context provenance measurable

Agent frameworks commonly record outputs and tool calls. Fewer systems make the supplied context a first-class, queryable object.

Because Polylogue records context snapshots, it can support analyses such as:

- failures correlated with omitted evidence;
- performance before and after compaction;
- whether a stale assertion influenced an action;
- whether two agents received materially different briefs;
- how much context was inherited, summarized, or newly retrieved;
- whether a handoff pack improved reconstruction within a fixed effort budget.

This is a basis for serious agent evaluation rather than retrospective storytelling.

## It supports cross-provider agent continuity

A useful project memory should not disappear when work moves from Claude Code to Codex or from a browser chat to a local agent. Polylogue normalizes those origins into one archive while preserving their distinct evidence semantics.

That means a Codex agent could retrieve a reviewed decision produced during a Claude Code session, while still resolving the claim to the original Claude evidence. The archive is cross-provider; the evidence remains origin-specific.

## It creates a substrate for multi-agent coordination

The current repository already contains coordination envelopes, blackboard/assertion surfaces, session topology, context compilation, and an MCP `agent_coordination` tool. The larger coordination program is not complete.

Bead `polylogue-s7ae` describes a coordination substrate that joins:

- agent sessions and logical session trees;
- repositories, worktrees, and branches;
- optional work items;
- activity and resource episodes;
- coordination messages and advisories;
- context-flow refs;
- proof and outcome refs;
- freshness and confidence.

The intended product is not merely an agent chatroom. It is a durable mission-control view in which delegation, evidence, context transfer, and outcomes can be reconstructed after the agents have stopped.

## It is useful for evaluation and research

The normalized distinctions among human-authored text, assistant text, runtime protocol, tool results, generated context, and generated analysis make the archive a potentially valuable evaluation corpus.

Possible research uses include:

- tool-use reliability;
- failure recovery;
- context-compaction loss;
- memory staleness;
- cost versus outcome;
- delegation quality;
- model or provider migration;
- behavioral changes over time;
- which agent affordances are actually used;
- correlations between tool sequence and successful outcomes.

This does not automatically make the archive suitable training data. Privacy, provider terms, source-code ownership, and selection bias remain serious constraints. Its strongest immediate research value is in private evaluation and behavioral forensics.

---

# 6. How Polylogue fits the current agent standards stack

The surrounding standards now divide into relatively clear layers.

The Model Context Protocol standardizes how an LLM application connects to external context and capabilities. MCP servers can expose resources, prompts, and tools, and Polylogue currently exposes all three. In this role, Polylogue is an **agent-to-evidence and agent-to-memory server**: an agent can search the archive, resolve refs, compile context, inspect topology, record candidate state, or request a postmortem through a standard interface. citeturn836405view0

A2A 1.0 addresses a different layer: communication, discovery, delegation, and result exchange between independent agents. Its official documentation explicitly describes MCP as complementary agent-to-tool communication and A2A as agent-to-agent communication. Polylogue is not currently an A2A transport. Its prospective role is orthogonal: preserving the evidence, context flow, decisions, and outcomes around A2A exchanges, and possibly exporting or ingesting those relationships later. citeturn836405view1

OpenTelemetry semantic conventions provide common naming for telemetry across traces, metrics, logs, profiles, and resources; generative-AI conventions now have their own dedicated workstream. Polylogue already receives OTLP and can project archive evidence into an OTel-shaped view. The distinction it makes is important: telemetry is an interchange and observability view, while the Polylogue archive remains the canonical source of longitudinal evidence. citeturn836405view2turn511787search34

Framework tracing, such as the OpenAI Agents SDK trace model, records generations, tool calls, handoffs, guardrails, and custom events within agent workflows. Polylogue overlaps with that observability function but has a broader intended temporal and provenance scope: multiple vendors, post-hoc source files, browser chats, raw artifact preservation, reviewed assertions, logical session lineage, context snapshots, and long-term local analysis. Framework tracing answers “what happened in this run?” Polylogue aims to answer “what happened across all of this project’s AI-mediated work, what evidence supports the interpretation, and what should another agent be allowed to remember?” citeturn836405view3

A useful landscape comparison is:

| Category | What it primarily owns | Where Polylogue differs |
|---|---|---|
| Chat export viewer | Rendering conversations | Normalizes work structure, tools, costs, lineage, evidence and user state |
| Agent tracing dashboard | Live run spans and debugging | Cross-provider, longitudinal, source-preserving, local and retrospective |
| Vector memory/RAG store | Similarity retrieval | Embeddings are optional projections; review, provenance and exact refs come first |
| Personal knowledge base | Human-authored notes and links | Automatically captures machine work and structural outcomes |
| Agent framework/orchestrator | Deciding and executing actions | Does not aim to be the execution engine; records and informs one |
| Git | Code history and diffs | Covers prompts, attempts, failures, tool evidence and context around code history |
| Data warehouse | Aggregate analysis | Domain-specific local evidence model with raw-byte resolution |
| A2A transport | Inter-agent communication | Could archive and evaluate coordination but is not the transport itself |

The best category description is therefore the one the project has converged on internally:

> **A local flight recorder and system of record for AI work.**

---

# 7. Present implementation state

## Repository maturity

The package reports version `0.1.0`, Python 3.11+, MIT licensing, and Beta status. The bundled history contains no release tags. The product posture is therefore pre-stable even though the implementation is already large.

The snapshot’s attribution statistics include:

- 441 core-and-storage files, with 98,589 code lines;
- 134 CLI/MCP/operations files, with 35,032 code lines;
- 59 daemon files, with 26,428 code lines;
- 74 pipeline/product/readiness files, with 19,287 code lines;
- 33 API/surface files, with 12,360 code lines;
- 26 archive-query files, with 9,172 code lines;
- 74 rendering/site files, with 18,020 code lines;
- 809 test-and-QA files, with 223,681 code lines;
- 132 documentation files containing 35,700 total lines.

The main product buckets alone contain roughly 219,000 code lines before counting devtools and test code. This is not a small experimental script.

The project enforces strict MyPy over production code, tests, and devtools; Ruff; Hypothesis; snapshot and terminal tests; mutation testing; optional fuzzing; machine-contract markers; scale tiers; chaos tests; and load-sensitive test lanes.

The current coverage configuration records approximately 83% aggregate coverage after a major archive split, with line coverage around 86.3% and branch coverage around 73%, below a historical 90% floor. The configured failure threshold is temporarily 82%. That is solid but also a declared regression caused by architectural migration, not a finished quality state.

## Implemented versus incomplete areas

| Area | Present now | Material remaining |
|---|---|---|
| Multi-origin capture | Local provider files, exports, hooks, browser receiver, Drive and Antigravity paths, OTLP | Full completeness and drift handling for every origin; Grok path |
| Raw evidence preservation | Source tier, blobs, artifact taxonomy, import explanation | Attachment acquisition/reacquisition consistency and explicit excision |
| Normalized archive | Sessions, messages, blocks, actions, topology, usage | Complete Provider→Origin transition and more provider parity |
| Search/query | FTS, structural DSL, terminal units, refs, lexical/semantic/hybrid lanes | Grammar/surface unification, performance and parity consolidation |
| Analytics | Profiles, phases, work events, costs, usage, latency, postmortems, pathologies | Provenance-first analysis objects, outcome validation and faster large-archive queries |
| Context | Context compiler, context images, preambles, snapshots, resume candidates | Complete judgment queue, scheduler, freshness loop and production uplift proof |
| Agent interface | 96 MCP tools plus resources and prompts | Surface simplification, stronger role/trust contracts and lower cognitive load |
| Coordination | Envelope/rendering primitives, blackboard and topology surfaces | Full multi-agent mission control, activity/resource episodes and live proof |
| Web/UI | Local reader, HTTP APIs, TUI/dashboard | Semantic transcript cards and full evidence cockpit |
| Embeddings | Voyage `voyage-4`, 1024 dimensions, SQLite-vec, backfill and hybrid search | Provider-neutral local/cloud abstraction and measured retrieval quality |
| Interop | MCP, OTLP receive, OTel projection, JSON/YAML/CLI contracts | Wider export/federation, OTel GenAI normalization and other knowledge-system bridges |
| Security | Loopback, tokens, origin controls, local storage | Selective forgetting, hard deletion, broader audit and team/federation security |

## Deterministic demo state

The private-data-free demo currently contains:

- 11 sessions;
- 43 indexed messages;
- 87 blocks;
- five origins;
- acquired attachments;
- browser-capture coalescing;
- lineage links;
- subagent runs;
- terminal-state examples;
- synthetic embeddings;
- user overlays;
- context snapshots and observed events.

`polylogue demo tour` exercises the canonical import, query, read, and analysis paths and emits a report, transcript, tape, and optional recording. The demo is designed to prove constructs, not scale.

## Live dogfooding

A tracked July 5 forensics artifact ran against a private archive containing:

- 16,816 physical sessions;
- 4,364,655 messages;
- 665,890 blocks;
- eight origin rows;
- approximately 399.9 billion physically accounted tokens;
- approximately 292.9 billion logical high-water tokens;
- an approximately 107.0 billion-token replay gap.

It also produced distinct stored-cost, API-list-equivalent, and logical-list-equivalent estimates. The report explicitly says these are not provider invoices.

The same proof artifact records an important failure: a cost-rollup analysis exceeded a 120-second timeout. That finding now feeds the performance backlog. This is a positive sign for the project’s honesty—the performance defect is preserved in the demonstration instead of edited out—but it also confirms that interactive analytics at multi-million-message scale remain incomplete.

## Current uncommitted work

The snapshot is marked dirty with `polylogue-f2qv.2` in progress.

The current changes correct Codex per-message token accounting. Codex reports `input_tokens` inclusive of cached input, while downstream pricing treats fresh input and cache-read input as separate additive lanes. Storing the inclusive number therefore double-bills the cached portion.

The patch subtracts cached tokens at parse time and adds:

- a parser-level disjointness regression;
- a pricing-consequence test using a realistic heavily cached ratio;
- a Claude control case, because Claude’s native input lane is already exclusive of cache;
- cost-model documentation.

The complete Bead also targets separating reasoning output from ordinary completion output. The working patch visibly handles the cached-input part; the Bead remains in progress rather than prematurely closed.

---

# 8. The Beads planning system

The roadmap is not maintained in design-plan Markdown or inferred from GitHub.

`docs/design/README.md` states the authority rule directly: when a design document and its owning Bead disagree, **the Bead wins**. Plans are mined into Beads and then removed rather than retained as competing sources of truth.

The snapshot contains:

- 612 Beads issues;
- 440 open;
- one in progress;
- 171 closed;
- 336 reported ready;
- 104 blocked;
- 34 durable operational memories;
- 880 dependency edges.

Issue types include 34 epics, 144 features, 76 bugs, 340 tasks, 14 chores, and four decisions.

## Why the Beads graph is more credible than a flat backlog

The project models its plan as a technology tree:

- `horizon:frontier` means executable now and requires implementation-grade design and acceptance criteria;
- `horizon:mid` means prerequisites are still landing and some detail may remain provisional;
- `horizon:vision` records a far target, its purpose, and what it enables without inventing false precision.

Hard `blocks` edges are reserved for actual execution ordering. Soft relationships use related links. Parent-child edges express epic membership.

A well-formed execution Bead contains:

- the problem and supporting evidence;
- relevant code locations;
- a proposed algorithm or design;
- pitfalls;
- checkable acceptance criteria;
- verification commands or proof artifacts;
- dependency and delivery-gate labels.

The operational doctrine also says to batch Beads by overlapping file footprint, not blindly execute one issue at a time. This reflects an agent-heavy development process where parallel work and shared-module collisions are real concerns.

## The current frontier

The current delivery frontier is **A — trust floor**.

That choice says something important about project priorities. The next objective is not autonomous memory, a larger web dashboard, or more embedding providers. It is making the evidence and read contracts reliable enough that those higher-level features cannot amplify wrong data.

The delivery-gate graph is:

| Gate | Theme | Closed / total |
|---|---|---:|
| A | Trust floor | 40 / 60, with one in progress |
| B | Storage, rebuild and bytes | 1 / 23 |
| C | Read and evidence contract | 1 / 61 |
| D | Agent context and coordination | 0 / 48 |
| E | Content variants and preferences | 0 / 12 |
| F | Lineage and compaction | 2 / 13 |
| G | Live operation and performance | 0 / 25 |
| H | Web evidence cockpit | 0 / 18 |
| I | Analytics and experiments | 1 / 30 |
| J | Embeddings and retrieval | 0 / 11 |
| K | Interop, origin and export | 1 / 31 |
| L | External legibility and demos | 7 / 30 |
| M | Substrate consolidation | 1 / 34 |
| N | Longer-horizon vision | 0 / 6 |

These counts are delivery-labeled Beads, not a percentage of existing code. A gate with zero closed items can still have substantial substrate already implemented; it means the gate’s defined proof and product contract have not been completed.

## Active P1 work

The current P1 frontier divides into three clusters.

### Usage and cost honesty

`polylogue-f2qv.2` is the in-progress Codex disjoint-lane normalizer.

`polylogue-f2qv.3` will expose API-list-equivalent cost and subscription-credit consumption as separate views. A subscription user’s economic cost is not the same as API list price, and the UI should not conflate them.

`polylogue-f2qv.4` will establish one pricing catalog source and remove competing lookup behavior.

`polylogue-f2qv.5` will version-gate the provider-usage projection so it self-heals like other read models.

`polylogue-5hf` owns the broader honest cross-provider token ledger.

`polylogue-xy95` addresses the unbounded/full provider-usage diagnostic that can hang on the live archive.

### Evidence and trust contracts

`polylogue-cpf.2` adds writer-class and layering checks so mutation ownership is explicit.

`polylogue-cpf.3` establishes a trust-class fixture for injected content. Operator instructions, system material, and quoted/untrusted material must not be treated as equivalent.

`polylogue-cpf.4` audits “robust but silent” paths. A degraded, timed-out, truncated, unsupported, or incomplete result must carry one bounded signal rather than masquerade as ordinary success.

`polylogue-avg` promotes the project’s claim-guard vocabulary into product readiness surfaces.

### Storage, identity and reading

`polylogue-9e5.19` creates a storage-layer correctness scenario family.

`polylogue-9e5.8` completes the migration from overloaded provider identity to public origin identity.

`polylogue-ap7` implements semantic transcript rendering across CLI and web. A shell command should look like a command with folded output and structural exit status; an edit should look like a diff; a subagent dispatch should link to its child; unknown tools should retain the generic fallback.

That prioritization is coherent: first make the accounting, storage, readiness, trust, and rendering contracts honest; then build richer memory and coordination over them.

---

# 9. Major planned programs, especially for agents

## `polylogue-37t`: the judged context and memory loop

This is the central agent-memory program.

The substrate already exists in pieces: assertions, candidates, context policies, context compilation, preambles, hooks, resume candidates, and context snapshots. The epic turns those pieces into a complete product loop:

- agents declare structured candidate claims;
- operators receive a judgment queue;
- accepted claims become active;
- rejected and superseded claims remain historically legible;
- context scheduling applies trust, budget, freshness, and scope;
- SessionStart hooks receive a bounded evidence-backed preamble;
- recursive citation and injection safety are enforced;
- utility analytics measure whether the memory helped;
- production experiments compare memory-assisted and raw-reference arms.

A closed safety chokepoint already enforces the doctrine that non-user authors create candidates with injection disabled. The remaining work makes that doctrine usable at scale.

## `polylogue-s7ae`: multi-agent coordination substrate

This program treats agent coordination as an evidence problem.

It plans a durable envelope connecting session trees, repositories, worktrees, branches, work items, activity episodes, resources, coordination messages, advisories, context flows, proof, outcomes, freshness, and confidence.

Planned surfaces include:

- one-command hook installation;
- provenance-carrying pull-request or change artifacts;
- bounded MCP and CLI coordination views;
- a web mission-control view;
- blackboard communication;
- workflow catalogs and cookbooks;
- a live two-agent proof.

The model is intentionally independent of Beads. Beads manages Polylogue’s own roadmap; the product coordination substrate is meant to represent work regardless of the external task system.

## `polylogue-rii`: live substrate intake

Current ingestion is a mixture of file watching, hooks, browser capture, and OTLP. This program makes in-loop agent work events a first-class live input rather than waiting for complete post-hoc artifacts.

That is necessary for:

- real-time context injection;
- live mission control;
- permission and tool audits;
- timely blackboard updates;
- accurate continuation state;
- observing work that never reaches a final transcript file.

## `polylogue-4ts` and `polylogue-gjg`: lineage and compaction truth

`4ts` owns “store once, count once, compose correctly” for shared prefixes and logical sessions.

`gjg` extends that into compaction lifecycle:

- pre-compaction snapshots;
- loss forensics;
- effective-context reconstruction;
- deterministic regrounding after compaction;
- comparison between full history, summaries, and retrieved context.

Together these projects enable trustworthy cost accounting and experiments about what models actually knew after context compression.

## `polylogue-1vpm`: work-graph units

Messages and sessions are insufficient as the only analysis units. This program introduces higher-level work objects such as:

- delegations;
- episodes;
- artifacts;
- turn pairs;
- correction edges.

This would allow queries about “a delegated task and its result” rather than forcing every analysis to rediscover that unit from raw messages.

## `polylogue-rxdo`: provenance-carrying analytical objects

Queries, cohorts, findings, result sets, and analyses are currently often transient command outputs. This program promotes them into first-class objects with inputs, transformations, evidence refs, caveats, and reproducible outputs.

That is important for both research and agents. A future agent should be able to consume a finding without losing the query and evidence that produced it.

## `polylogue-mhx`: provider-general embeddings and measured retrieval

The current embedding implementation is deliberately narrow: Voyage `voyage-4`, 1024 dimensions, SQLite-vec, opt-in backfill, and explicit semantic or hybrid retrieval.

The planned program adds:

- provider abstraction, including local and cloud options;
- explicit target-selection policy;
- lifecycle handling for model/dimension changes;
- bounded vector work;
- measured lexical/vector/hybrid retrieval quality;
- semantic recall inside context compilation;
- clustering, novelty and related analyses;
- refusal to mix incompatible vector spaces.

The important point is that embeddings remain a projection. They are not allowed to become archive authority.

## `polylogue-l4kf`: interop, origins and export

The interop program plans:

- broader origin specifications;
- OpenTelemetry GenAI normalization;
- additional providers including Grok;
- research-export formats;
- HPI/Promnesia-style integration;
- Git notes and SARIF;
- Obsidian export;
- federation and citable exchange.

This is where Polylogue could become an interchange layer rather than only a personal archive.

## Product and architecture programs

`polylogue-bby` develops the web reader into a workbench and evidence cockpit.

`polylogue-a7xr` owns substrate consolidation: reducing duplicated read/write paths, oversized modules, surface-specific logic, and architectural drift.

`polylogue-3tl` makes the project understandable and citable by outsiders.

`polylogue-212` builds a portfolio of construct-valid demos where every displayed result resolves to product primitives and structural evidence rather than hidden demo scripts.

---

# 10. What has already been learned through the Beads campaigns

The project’s highest-priority campaigns have not merely shipped features; they have produced negative and qualified evidence.

The closed claim-vs-evidence campaign established the structural failure methodology.

The closed agent-forensics campaign regenerated a large longitudinal archive report and exposed the distinction among physical, logical, provider-reported, and catalog-priced usage.

The first handoff uplift campaign produced a negative result: a stale packet scored worse than raw references. Rather than reinterpret the result as success, the campaign closed with staleness as its central finding.

The successor pilot then found directional benefit from fresh synthesis, but retained the one stale/ahead counterexample and refused to publish a general uplift claim.

This experimental discipline is part of the project’s value. Many “agent memory” systems are justified by intuition. Polylogue is building an environment in which memory, retrieval, summaries, and coordination can be falsified.

---

# 11. Current weaknesses and risks

## The surface area is extremely broad

The project includes ingestion, schemas, five database tiers, a query language, analytics, CLI, 96 MCP tools, a daemon, web APIs, a browser extension, TUI, rendering, telemetry, context compilation, coordination, demos, and a large developer toolchain.

That breadth creates a risk that every subsystem becomes almost complete but the product remains hard to understand or operate. The `M-substrate-consolidation` gate exists for a reason. Large storage modules and parallel sync/async or surface-specific paths are visible architectural debt.

The MCP catalog itself illustrates the tension. A broad tool set exposes enormous capability, but agents need a small, predictable decision surface. The standing MCP doctrine says an agent should generally resolve a decision in no more than three calls and two seconds. Ninety-six tools can work only if discovery, grouping, chaining, schemas, and defaults are excellent.

## Provider normalization is permanently expensive

Provider formats drift. Tool-call representations, usage fields, compaction encodings, browser payloads, and filesystem locations are not stable public contracts in every case.

Polylogue’s evidence-honest approach makes this harder rather than easier: it cannot simply parse “something plausible” and discard unsupported fields. Every origin requires long-term schema and fixture maintenance.

This is likely the project’s principal ongoing maintenance cost.

## Large-archive interactivity is not solved

The live archive proves that the architecture can hold millions of messages, but not that every analysis is interactive. The 120-second cost-rollup timeout and provider-usage full-diagnostic hang are concrete counterexamples.

Several future capabilities—web mission control, live context compilation, coordination views, and research exploration—depend on predictable bounded queries. Performance is therefore a product requirement, not merely optimization.

## Documentation drift is already visible

The stale schema-version table is the clearest example. Some standing design documents describe older architecture or use issue references that are no longer planning authority.

The repository explicitly says code and Beads win, which limits the damage, but an outsider still has to know which document classes are trustworthy.

## Attachment preservation is uneven

The deterministic demo proves attachment-byte acquisition, but lineage design work documents real-archive cases where attachment rows were metadata-only or carried synthetic hashes rather than acquired content. Attachment acquisition, re-acquisition, lineage preservation, and extraction remain active work.

The blob store is real and large; that does not imply that every historical attachment is already recoverable.

## Memory staleness remains the central semantic risk

The project’s own pilot demonstrated the failure mode: a well-written context pack can make an agent confidently wrong when the pack is stale or ahead of the target checkpoint.

Freshness cannot be a timestamp decoration. It must be tied to the relevant source state, scope, supersession, and live verification. The planned scheduler and judgment loop address this, but the complete safety contract is not yet delivered.

## Security and deletion need completion

A local archive containing millions of messages and raw tool output is exceptionally sensitive. Loopback binding and bearer tokens protect network access, but they do not solve:

- secrets preserved in raw evidence;
- selective deletion;
- retention policies;
- evidence references after deletion;
- encrypted storage;
- team access control;
- federation trust.

`polylogue-kwsb` is strategically important, not peripheral housekeeping.

## Product maturity trails implementation maturity

The codebase looks like a substantial internal product, but the packaging says `0.1.0` Beta and there are no tags in the supplied history. Installation, onboarding, contract stability, migration guarantees, and external documentation are therefore not yet at the level suggested by the code volume.

---

# 12. The project’s real value

Polylogue’s defensible value is not its transcript UI, vector search, or number of MCP tools. Those can be reproduced.

The harder asset is the **semantic normalization and evidence model**:

- separating source origin from vendor;
- separating role from material authoredness;
- representing tool calls and results structurally;
- preserving raw evidence while rebuilding read models;
- modeling fork, replay, subagent, and compaction lineage;
- keeping token lanes disjoint across provider conventions;
- distinguishing physical from logical usage;
- making context delivery and omission inspectable;
- requiring review before agent-authored claims become memory;
- resolving every important object through stable public refs;
- making degraded or unsupported states visible.

That combination turns heterogeneous AI activity into something that can be queried and reasoned about without flattening away the very distinctions needed for trust.

For an individual developer, the value is continuity: recovering dead work, finding prior attempts, avoiding repeated mistakes, and moving between agents without losing project history.

For a team, the prospective value is shared provenance: handoffs, reviewed decisions, postmortems, cost and outcome analysis, and evidence-backed coordination.

For an agent builder, the value is a framework-independent memory and evaluation layer.

For a researcher, the value is longitudinal behavioral evidence with structural outcomes rather than transcript-only labels.

For a security- or governance-conscious operator, the value is local control and auditable derivation, although retention and excision need more work before that story is complete.

---

# 13. Strategic assessment

Polylogue is already beyond the “chat archive” phase. It has a real ingestion substrate, a sophisticated data model, multiple operational surfaces, extensive tests, deterministic demos, and evidence from a multi-million-message live archive.

It is not yet a finished autonomous agent-memory or multi-agent platform. The pieces exist, but the project is correctly refusing to declare the loop complete before the trust floor, lineage, context safety, performance, and judgment contracts are proven.

The sequencing is the right one:

```text
evidence correctness
    → storage and read contracts
        → lineage and context truth
            → judged memory
                → coordination and automation
                    → externally defensible claims
```

Reversing that sequence would produce an impressive but unreliable agent dashboard. Following it could produce something more durable: infrastructure that lets agents work across time, vendors, and collaborators without treating their own prose as truth.

The project’s strongest description is therefore:

> **Polylogue is a local flight recorder, evidence ledger, and memory substrate for AI-mediated work. It captures heterogeneous agent activity, normalizes it without erasing provenance, lets humans and agents query and audit what happened, and is building a reviewed path from historical evidence into future agent context.**

Its broader significance is that increasingly capable models do not remove the need for systems of record. They increase it. As agents execute more tools, inherit more context, delegate more work, and operate for longer periods, the critical engineering problems shift from “can the model produce an answer?” toward:

- what did it actually do;
- what evidence supports the claim;
- what context influenced it;
- what should persist;
- who approved that persistence;
- how do several agents share state without amplifying mistakes;
- how can the resulting history be reproduced and audited?

Polylogue is a serious attempt to build that missing layer.

## Repository orientation

The most useful reading sequence is:

1. `README.md` for the product thesis and four verbs.
2. `docs/architecture.md` for the four rings and data flow.
3. `docs/internals.md` for invariants, schema behavior, FTS, blobs, WAL, and convergence.
4. `docs/search.md` for the DSL, terminal units, refs, retrieval lanes, and OTel projection.
5. `docs/proof-artifacts.md` for the claim-to-evidence map.
6. `docs/security.md` for the current threat model.
7. `docs/design/README.md` for the rule that Beads, not plans, own future direction.
8. `polylogue/storage/sqlite/archive_tiers/` for actual database authority.
9. `polylogue/core/enums.py` for the normalized vocabularies.
10. `polylogue/context/` and `polylogue/coordination/` for the emerging agent-facing substrate.
11. `tests/infra/mcp.py` for the exact current MCP contract.
12. The Beads export, especially the A-trust-floor items and epics `37t`, `s7ae`, `4ts`, `gjg`, `rxdo`, `mhx`, `l4kf`, `a7xr`, `3tl`, and `212`, for the real plan.

---

