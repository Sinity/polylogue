<!-- This is a SHARED ROOT prefix, not a single conversation's delivery. 11 ChatGPT 'Branch ·' forks (6a506bcf [polylogue-overview turn only], 6a5112e7, 6a5112f3, 6a5112f5, 6a5112f9, 6a5112fd, 6a5113e8, 6a511407, 6a51140b, 6a511416, 6a511425) all descend from one root conversation that asked, in sequence: (1) explain the Polylogue project at length, (2) do the same for Sinex + interop, (3) improve both projects' external legibility. Each branch then diverged at a later turn ('Now, just like you did ... but even more ambitiously, here, against this prompt') by attaching a DIFFERENT execution-mission spec, producing a distinct final implementation delivery captured in that branch's own subdir. The three turns below are BYTE-IDENTICAL (sha256-verified) across all 11 branches that reach that point, so they are captured ONCE here instead of being duplicated ~11x (~175K chars each duplication avoided). Each branch subdir's STATUS.md / inline-artifacts.md references this folder instead of re-embedding these reports. -->

## Turn 23 (role=assistant)

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

## Turn 44 (role=assistant)

# Sinex: project presentation, present state, and the maximal Sinex–Polylogue architecture

I inspected the Sinex snapshot generated on July 10, 2026: the working tree, bundled Git history, architecture and source-contract documentation, proof artifacts, current uncommitted changes, and the complete Beads export. I also compared it against the supplied Polylogue snapshot and its Beads graph.

The snapshot is `master` at `b70a08d9`, version `0.4.2`, with an active dirty working tree. I did not execute the full test suite or deploy the services, so runtime conclusions below are based on committed tests, production proof packets, Beads closure evidence, and repository contracts rather than a fresh certification run. The primary source summaries are the [Sinex snapshot overview](sandbox:/mnt/data/sinex-src/sinex/sinex-overview.md), [Sinex Beads summary](sandbox:/mnt/data/sinex-src/sinex/sinex-beads.md), [Sinex Beads export](sandbox:/mnt/data/sinex-src/sinex/sinex-beads-export.jsonl), and [repository README](sandbox:/mnt/data/sinex-inspect/README.md).

## Executive thesis

**Sinex is a local, replayable evidence substrate for a person’s and machine’s digital activity.**

It captures heterogeneous source material, interprets it into typed events, preserves provenance and temporal uncertainty, derives higher-order objects through replayable computations, and exposes the result to operators and agents without silently treating interpretation as truth.

“Activity logger” is too weak. “Personal data warehouse” misses the temporal and evidentiary semantics. “Observability stack” misses source material, human activity, adjudication, and longitudinal memory. “Knowledge graph” gives the derived layer too much authority. “Agent memory” describes only one future consumer.

A more accurate description is:

> **Sinex is a local evidence operating system and exocortex substrate: a durable record of what was observed, when it apparently happened, how it was interpreted, what remains missing, and which interpretations have actually been accepted.**

Under the maximal Sinex–Polylogue design, the relationship should be:

```text
Beads       = intent and task authority
Sinex       = durable evidence, event, lifecycle, and model-effect backend
Polylogue   = AI-work ontology, transcript semantics, memory product, and UX
LLMs/agents = actors, analysts, and proposal generators
SQLite      = Polylogue’s local projection, offline replica, and acceleration layer
```

This is a stronger and more coherent architecture than the current metadata-only bridge.

Sinex should ultimately store the transcripts. More precisely, it should store:

1. the provider-native transcript artifacts and attachment bytes;
2. immutable normalized Polylogue transcript material;
3. durable transcript-domain observations and their revisions;
4. session lineage, usage, context-delivery, assertion, and judgment history;
5. derived cross-domain relations such as Agent Work Packets;
6. model effects, embeddings, and analysis products where enabled.

Polylogue should remain responsible for defining what those things mean. Sinex becoming the backend does **not** mean dissolving Polylogue into generic Sinex events.

The correct division is:

> **Sinex owns durable persistence and evidentiary lifecycle. Polylogue owns AI-work semantics and product behavior.**

---

# 1. The problem Sinex is solving

Most of a person’s actual work is distributed across systems that do not share an ontology or clock:

- terminal histories;
- shell exit results;
- Git commits, branches, and repository state;
- browser visits and research;
- active windows and desktop focus;
- filesystem activity;
- application and system logs;
- documents and exports;
- task systems;
- model calls and agent sessions;
- health, finance, or other personal records;
- manually written notes and declarations.

Even when all of this can be collected, ordinary collection systems have several serious weaknesses.

They confuse **absence of evidence** with **evidence of absence**. If a browser source silently stopped working, a query may appear to show no browsing. If a capture cursor advanced before an event became durable, the missing event becomes invisible. If a parser skipped half an export, downstream analytics may confidently describe the partial result as complete.

They flatten different notions of time. An event may have happened at 14:00, been captured at 14:07, interpreted again during replay three months later, and persisted during a restore. A database with one timestamp cannot represent that honestly.

They overwrite prior interpretations. When a parser changes, a conventional ETL pipeline updates a row and destroys the history of how that source was previously understood.

They treat inferred state as fact. A model labels an interval as “deep work,” an entity extractor decides that two names refer to one person, or a summarizer generates a project status—and the result enters the database without an authority boundary.

They optimize retrieval without first securing provenance. A vector store can find similar text, but it normally cannot explain whether the text was raw evidence, an old model summary, an operator-approved decision, or a stale context pack.

Sinex is designed around those failures. Its central question is not merely:

> What data can be collected?

It is:

> What can the system honestly claim from the material it holds, under which interpretation, with what temporal quality, provenance, coverage, and authority?

That makes Sinex especially relevant to AI agents. Agents consume context and produce interpretations at high volume. Without an evidence substrate, their outputs can recursively become their own supposed ground truth.

---

# 2. The defining concepts

## Source material is not an event

Sinex draws a hard boundary between acquired material and interpreted observations.

Source material includes:

- an exported archive;
- a terminal database;
- a browser history database;
- a session JSONL file;
- a document;
- a Git repository snapshot;
- a streamed capture segment;
- a screenshot or attachment;
- an immutable acquisition snapshot.

A typed event is an interpretation anchored to that material:

- a command was executed;
- a page was visited;
- a message was emitted;
- a tool call returned;
- a file changed;
- a task was closed;
- an agent session compacted.

This matters because the original bytes and the interpretation have different authority. A parser can be wrong. A schema can improve. A source can be partially corrupt. By preserving the material, Sinex can later reinterpret it without pretending the original interpretation was the source.

The project’s knowledge-boundary doctrine lays out the layers explicitly:

```text
source material
    ↓
material interpretation
    ↓
typed events and domain objects
    ↓
derived graph and cross-source products
    ↓
workspace artifacts such as context packs and reports
```

Context packs, reports, and semantic summaries are products over evidence. They are not automatically canonical facts. This doctrine is documented in [knowledge boundaries](sandbox:/mnt/data/sinex-inspect/crate/sinex-primitives/docs/knowledge_boundaries.md).

For Polylogue, the corresponding mapping is straightforward:

| Sinex layer | Polylogue material |
|---|---|
| Source material | Provider exports, rollout JSONL, browser payloads, hook logs, attachments |
| Material interpretation | Normalized sessions, messages, blocks, tool calls, usage observations |
| Domain projections | Current transcript view, logical session topology, cost and usage views |
| Derived graph/products | Work episodes, postmortems, Agent Work Packets, context candidates |
| Workspace artifacts | Context images, handoff packs, reports, saved analyses |

## One occurrence has several times

Sinex treats time as a semantic problem rather than a single database column.

The principal clocks are:

- `ts_orig`: when the observed occurrence happened in its source domain;
- `ts_coided`: when Sinex coined or minted this interpretation;
- `ts_persisted`: when the interpretation was written to the database.

A replay should normally preserve `ts_orig`, while generating a new event identity and new `ts_coided` and `ts_persisted` values.

For a Polylogue message:

- `ts_orig` is the provider or runtime message timestamp;
- `ts_coided` is when the current Polylogue parser interpretation entered Sinex;
- `ts_persisted` is when the event engine committed it.

For a provider export, import time must not replace the conversation’s actual occurrence time. Sinex’s `polylogue-4j2` program explicitly identifies this as a settled requirement.

This separation supports questions such as:

- How late did a source arrive?
- Was a result reconstructed after the fact?
- Did a model act on evidence that existed at the time?
- Was a session event imported months later?
- Did a replay change interpretation without changing occurrence time?

Transcript ordering also needs an additional domain-local ordering key. Provider message sequence, block ordinal, tool-call pairing, and causal order cannot be reconstructed reliably from timestamps alone.

## Material provenance and derived provenance are different

Sinex enforces a fundamental provenance distinction.

A material interpretation points to source material, including an anchor or offset. A derived event instead points to one or more source events.

Conceptually:

```text
Material interpretation:
    source_material_id + anchor
    no source_event_ids

Derived interpretation:
    source_event_ids
    no source_material_id
```

This is enforced in the event model, serialization, and database constraints.

For Polylogue:

- a parsed message is a material interpretation of a provider artifact;
- a tool failure pathology is derived from one or more message/tool-result events;
- a work episode is derived from transcript, terminal, focus, Git, and task events;
- a context pack is a workspace artifact over selected evidence.

This distinction prevents an important mistake: a Polylogue fork edge, continuation edge, or parent–subagent relationship is **not** Sinex derivation provenance. It is a domain relationship. Encoding a session fork as `source_event_ids` would make replay, deletion, and derivation tracing semantically wrong.

## Event identity is interpretation identity

Sinex’s UUIDv7 event ID identifies a particular interpretation instance. It is useful for ordering and exact traceability, but it is intentionally not stable across replays.

When a parser or derivation is rerun:

1. the old interpretation can be archived;
2. the material or source events remain;
3. a new interpretation is emitted;
4. the new event receives a new ID and coining time.

This is the right behavior for evidentiary replay. It is not sufficient for stable Polylogue object identity.

A Polylogue message needs at least four distinct identities:

| Identity | Meaning |
|---|---|
| Domain object ID | Stable identity of the logical message, session, assertion, or context delivery |
| Domain revision ID | A particular content or semantic revision of that object |
| Sinex event ID | A particular admitted interpretation produced during one replay epoch |
| Material occurrence | Exact source bytes or record anchor supporting the interpretation |

Conflating these would break citations after replay.

A durable Polylogue ref such as `polylogue:message/<id>` must not resolve solely through a Sinex event UUID. It should resolve through a stable domain identity ledger, which can then enumerate current and historical interpretations and their source evidence.

## Replay is intentionally non-idempotent at one level and idempotent at another

The Bead `sinex-908` describes three different de-duplication regimes:

1. **Interpretations are deliberately non-idempotent across replay.** A new parser run creates a new interpretation.
2. **Occurrences should be idempotent.** Re-reading the same source occurrence should not invent another occurrence.
3. **Ambiguous cross-source duplicates require adjudication.** Two sources may describe the same real-world occurrence without enough evidence for automatic fusion.

This is an unusually important distinction.

For Polylogue:

- replaying a provider artifact with parser semantics version 5 should produce a new interpretation epoch;
- the underlying provider message remains the same logical occurrence;
- a copied prefix in a fork is not automatically another logical message;
- two exports containing the same conversation may need reconciliation;
- two similar messages are not fused merely because their text hashes match.

Sinex’s current `equivalence_key` machinery is part of this, but the Polylogue domain requires a stronger stable identity and revision layer than a single equivalence key can provide.

## Corrections are new records

`core.events` is append-only. Corrections, supersession, archival, replay, and deletion lifecycle are represented explicitly rather than by silently rewriting historical events.

This supports:

- auditability;
- parser-version comparison;
- reproducible derivations;
- rollback of promoted interpretations;
- tracing which state existed at a past point;
- explaining why a current projection changed.

The current-state view remains disposable. It is reconstructed through domain reducers or other projections.

## Projection is not authority

Sinex separates durable events from current-state projections.

A domain reducer declares:

- the domain and semantics version;
- accepted input event types;
- object-key policy;
- event ordering;
- settlement and conflict policy;
- output shape;
- teardown and trace behavior.

The shared reducer substrate is only partially implemented today—the task domain provides the first concrete reducer specification—but the doctrine is clear: current objects remain projections over canonical events. See [domain reducers](sandbox:/mnt/data/sinex-inspect/crate/sinex-primitives/docs/domain_reducers.md).

For Polylogue, a corresponding reducer family would produce:

- current sessions;
- current message revisions;
- content blocks;
- current session topology;
- current assertion state;
- current context policies;
- current usage and cost rows;
- complete transcript revision manifests.

The transcript page that a user reads is a projection. The raw provider artifacts and admitted transcript-domain history remain the recoverable substrate.

## Confidence is not authority

Sinex is developing a proposal, judgment, and finalizer model.

An AI model, heuristic, or extractor may propose:

- an entity merge;
- a project association;
- a work-episode title;
- an assertion;
- a lesson;
- a task-like candidate;
- a semantic relation.

That proposal can carry evidence and confidence, but confidence does not promote it into canonical state. Promotion requires a separate judgment or policy authority.

The planned `ClaimSupport` vector goes beyond a scalar confidence number. It is intended to record factors such as source coverage, temporal quality, supporting evidence, adjudication state, and evidence counts.

This is nearly a direct match for Polylogue’s assertion model:

```text
source evidence
    ↓
candidate assertion
    ↓
operator or policy judgment
    ↓
accepted / rejected / deferred / superseded
    ↓
context eligibility
```

Polylogue’s default `inject: false` policy for agent-generated state is exactly aligned with Sinex’s “confidence is not authority” doctrine.

## Coverage error is a first-class result

Sinex does not want a query to silently interpret a broken source as an empty source.

The in-progress Bead `sinex-jdp` promotes capture losslessness into a standing coverage measure. In its first live proof, 58 of 64 source entries had at least one coverage error, producing a source-level rate of 9,062 basis points. That number does **not** mean that 90.62% of all events were lost. It means 90.62% of the evaluated source entries had some identified coverage problem, such as missing materials, missing events, unobserved runtime bindings, stale source state, or continuity gaps.

That distinction is valuable because it prevents a false quantitative interpretation of the metric itself.

For Polylogue, a query result should similarly be able to say:

- this origin was never imported;
- this provider export is stale;
- this session material is partial;
- attachments were referenced but not acquired;
- the local SQLite projection is behind the Sinex event frontier;
- semantic search is unavailable or stale;
- the result covers physical sessions but not composed logical sessions.

## Self-observation belongs in a separate lane

Sinex distinguishes activity evidence from telemetry about Sinex itself.

- `core.events` holds activity and evidence.
- `reflection.events` holds system self-observation, health, processing behavior, and related telemetry.

The in-progress `sinex-x9i` program is completing this separation so that logs about capture do not drown the captured activity.

For a combined system:

- provider transcripts and agent actions are activity evidence;
- Polylogue daemon health, indexing latency, and parser diagnostics are reflection;
- model-call effects are explicit model effects;
- query and context-delivery telemetry may be reflection, while the fact that context was delivered is an activity-domain observation.

This prevents an agent query for “what happened during the session?” from returning pages of ingestion metrics unless explicitly requested.

---

# 3. Sinex architecture

The current deployed architecture is built around Rust services, PostgreSQL, NATS JetStream, and NixOS/systemd. The repository’s central overview is in the [README](sandbox:/mnt/data/sinex-inspect/README.md).

A simplified flow is:

```text
Files / databases / streams / exports / external producers
                             |
                    source contracts
                             |
          register and stage source material
                             |
              input-shape adapter + parser
                             |
          material-provenance EventIntent records
                             |
                   NATS JetStream raw lane
                             |
                       event engine
      validate → temporal resolution → privacy → persist
                             |
              confirmed publication / settlement
                             |
               PostgreSQL canonical event plane
                             |
          automata / reducers / projections / search
                             |
       API | sinexctl | MCP | TUI | context consumers
```

## Source contracts and staged material

A source integration declares substantially more than a parser name. Its contract includes:

- source ID and event vocabulary;
- source-material shape;
- runtime and adapter type;
- occurrence-identity policy;
- temporal horizons;
- privacy tier;
- checkpoint family;
- resource profile;
- retention policy;
- expected coverage behavior.

Source material enters a registry that acts as a manifest or birth certificate. It records identifiers, material backend, status, timing information, metadata, size, parse counts, coverage expectations, privacy class, and lifecycle state.

Material status is not just “present” or “absent.” It can represent sensing, completion, cancellation, partial recovery, or failure.

This gives Sinex a place to represent a partially written or interrupted transcript import without pretending it is complete.

## Evidence lanes

Sinex distinguishes several evidence roles:

- occurrence or acquisition payload;
- immutable epistemic snapshot;
- storage substrate;
- derived semantic material.

For a volatile stream, the material that was captured may be the only available evidence, and missed stream segments must remain visible as gaps. For a static export, the same file can be both the acquisition payload and the replayable epistemic snapshot.

For Polylogue, this supports several related but nonidentical materials:

1. original provider archive;
2. live hook/acquisition stream;
3. immutable normalized transcript segment;
4. attachment bytes;
5. rendered or exported transcript artifact;
6. context image delivered to an agent.

Material links can connect these without pretending they are interchangeable.

## Blob storage

Sinex stores blob metadata in PostgreSQL and the bytes in a content store. The current implementation uses BLAKE3-oriented content addressing and supports several backend labels, including local content storage and legacy or larger-material paths.

Polylogue’s current blob system uses SHA-256.

This is a manageable mismatch. The combined content descriptor should retain multiple digests:

```text
storage_digest       = BLAKE3 used by Sinex CAS
polylogue_digest     = SHA-256 used by Polylogue identity/content contracts
provider_digest      = original digest when supplied
canonicalizer        = version of any normalization applied before hashing
size
media type
```

A digest is a content descriptor, not a session or message identity.

## Event transport and admission

NATS JetStream is the runtime transport, not the ultimate evidence store.

The event engine performs:

- envelope and schema validation;
- timestamp plausibility checks;
- occurrence/equivalence handling;
- material foreign-key readiness checks;
- temporal-ledger resolution;
- privacy processing;
- durable PostgreSQL insertion;
- confirmed publication after persistence.

Large material bytes should not be copied into every NATS event. For Polylogue, the correct pattern is:

```text
stage transcript bytes in Sinex material/CAS
publish compact event carrying typed fields + material anchor
```

The existing Sinex guardrail against publishing raw private text in generic durable NATS payloads remains correct. What is wrong is using that transport guardrail to conclude that Sinex should not store transcript bytes at all.

## PostgreSQL organization

The system separates several concerns:

- `core`: canonical events, blobs, manifests, entities, relations, tags, embeddings, tombstones, operations, runs, and model effects;
- `raw`: source-material registry and temporal ledger;
- `reflection`: system self-observation;
- `audit`: archived event interpretations;
- `sinex_schemas`: schema and validation state;
- telemetry and derived views.

The system uses PostgreSQL 18 with TimescaleDB, pgvector, and pg_jsonschema.

Schema evolution follows declarative convergence rather than a conventional linear migration chain. The desired schema is applied and checked for drift, while data transformations remain explicit backfills. Recent commits harden function-body drift detection and query-plan correctness in lifecycle cascades.

For Polylogue, a Sinex-backed `polylogue` schema should follow that convergence model, while Polylogue’s local SQLite schemas can retain their own migration and rebuild strategy. The semantic parity contract should be at the domain-result level, not “the SQLite and PostgreSQL tables must be byte-for-byte identical.”

## Automata and derived computation

Sinex has a shared automata runtime supporting several computational shapes:

- transducers;
- windowed computations;
- scope reconcilers.

Registered automata currently cover sessionization and aggregation, attention streams, entity and relation processing, command canonicalization, health aggregation, document parsing, tag application, instruction reconciliation, and an embedding receipt path.

Derived events return to the same evidence pipeline with:

- source-event lineage;
- semantics version;
- temporal policy;
- automaton identity;
- replay metadata.

High-fan-in derivations can use explicit scopes rather than storing enormous parent arrays.

The planned derivation-control plane, `sinex-0vx`, adds product classes, epochs, experimental and shadow lanes, semantic diffs, and promotion through judgment. This is directly relevant to Polylogue analyses: a new failure detector, context-ranking strategy, or session-composition algorithm could run as a shadow semantics version before being promoted.

## API, CLI, query, and refs

`sinexctl query` uses a typed descriptor grammar rather than exposing SQL directly:

```text
<unit> where <typed predicates> order by <field> limit <n>
```

Current query units cover events, source drivers, source materials, debt, operations, and runtime health.

Responses use a shared `ViewEnvelope`, carrying:

- schema and view identity;
- generation time;
- source surface;
- query echo;
- payload;
- caveats;
- privacy state;
- freshness.

Sinex also has namespaced public object refs, though the current parser recognizes more kinds than the resolver can actually resolve. Bead `sinex-bm1` explicitly owns making that resolution matrix honest.

A Polylogue backend should reuse this envelope vocabulary. Polylogue already models caveats, omissions, lossiness, evidence refs, and readiness. A common outer response shape would make the systems feel related without forcing them to share one query language.

## MCP

The current Sinex MCP server exposes 68 read-only tools over local stdio. It is intentionally an evidence-access surface, not an actuator. It omits raw blob retrieval, direct material staging, event publication, deletion, proposal finalization, replay mutation, and other write-shaped behavior. See the [MCP contract](sandbox:/mnt/data/sinex-inspect/crate/sinexctl/docs/mcp_readonly_server.md).

This is a sound initial authority boundary.

Polylogue currently has a much larger domain-specific MCP catalog of 96 tools. Together, that would expose 164 tools before adding future interop operations. That is too much cognitive surface for most agents. The integration therefore needs a composed agent-facing layer, discussed below, rather than simply attaching both full tool catalogs to every model.

## Deployment and security

The canonical deployment is NixOS/systemd. Current controls include:

- TLS-only API RPC;
- stronger policy for non-loopback binding;
- bearer-token authentication;
- rate limiting;
- structured request auditing;
- managed NATS TLS and subject-level authorization;
- systemd hardening;
- a trusted single-user local-host threat model;
- full-disk encryption as the intended storage baseline.

Current generic MCP behavior is deliberately redacted and read-only.

Sinex does not currently make redaction a complete security boundary. Raw local evidence can contain secrets, personal text, code, and sensitive application state. A transcript-complete Polylogue integration would increase both the value and the blast radius of the Sinex store. It therefore needs stronger per-domain capabilities, physical deletion lifecycle, and explicit access to raw text.

---

# 4. A concrete Sinex workflow

Consider a build failure during an AI-assisted coding session.

At 14:02, a coding agent runs a test command. The terminal source captures the command and exit code. Polylogue captures the agent’s tool call and tool result. The browser source records research visits. Filesystem and Git sources record edits and commits. Window-focus evidence indicates which repository or editor was active. Beads records the work item that was claimed.

Sinex can preserve:

```text
terminal command event
    ↳ material anchor in shell history or capture stream

Polylogue tool-call observation
    ↳ material anchor in provider transcript

Polylogue tool-result observation
    ↳ structural exit/failure evidence

browser visit
    ↳ browser-history material

file changes and Git commit
    ↳ filesystem/Git material

Beads task events
    ↳ Beads event material
```

A derived work episode can then relate these observations without fusing them into one false intrinsic object.

A later query could ask:

> What was I researching when this build failed?

Or:

> The agent said the tests passed. What structural test result and machine activity support that claim?

Or:

> Resume this Bead. Show the prior agent sessions, the current branch and commits, failed verification attempts, accepted lessons, and evidence gaps.

Sinex provides the heterogeneous evidence and coverage state. Polylogue provides the semantics of messages, tools, logical sessions, compaction, agent context, and reviewed memory. Beads supplies the task intent.

That three-way relationship is much more powerful than transcript search alone.

---

# 5. Present implementation state

## Size and maturity

The package version is `0.4.2`, Rust edition 2024, MIT licensed.

The snapshot contains 320 Beads, with:

- 242 open;
- five in progress;
- 73 closed;
- 101 blocked;
- 639 dependency edges;
- 15 epics;
- 100 exported ready IDs in the snapshot overview.

GitHub Issues have been retired as the planning substrate. The zero open GitHub issue count in the snapshot does not mean there is no outstanding work; it means Beads are authoritative.

The major code-attribution buckets include approximately:

| Area | Files | Code lines |
|---|---:|---:|
| `sinexd` runtime | 189 | 56,343 |
| `sinexd` API | 63 | 28,326 |
| Event engine | 53 | 18,514 |
| Database layer | 51 | 20,523 |
| Shared primitives | 161 | 35,938 |
| CLI | 79 | 35,983 |
| Developer tooling | 200 | 85,882 |
| Schema/macros | 34 | 14,430 |
| Test suite | 834 | 186,127 |

The test suite also contains 380 split Rust test files under source trees and 364 files with inline test modules. These attribution buckets overlap in purpose and should not be treated as a clean count of unique product logic, but they clearly show that Sinex is a large system rather than an experimental collector. The detailed report is in [Sinex tokei statistics](sandbox:/mnt/data/sinex-src/sinex/sinex-tokei-stats.md).

The project remains pre-stable. There are no conventional release tags indicating a supported public release line. Operationally it is closer to a substantial, intensely dogfooded personal platform than a packaged general-user product.

## Live deployment evidence

The closed production-restore campaign `sinex-mhk` recorded:

- a Timescale chunk estimate of 80,872,976 events;
- exactly 34,782 source-material registry rows;
- exactly zero persisted DLQ rows at the proof point;
- a healthy serving runtime;
- a five-minute live recall query returning 12 events from five sources in roughly 140 milliseconds.

The campaign also records a real loss event during recovery: 11,631 raw-stream messages and 836 DLQ messages were intentionally purged to establish a fresh real-time boundary after the consumer became pinned behind poisoned backlog. Some of that material may have been recoverable.

That caveat matters. It means the proof demonstrates a functioning large live deployment, but not a perfect lossless recovery path. The current `sinex-r6d` and `sinex-n23` programs exist partly to ensure that manual stream purging does not become the normal recovery mechanism.

A broader 30-minute recall path was later hardened to return in approximately 9.1 seconds with an explicit timeout caveat rather than hanging. This shows that narrow queries can be interactive, while broader recall still requires ongoing performance and projection work.

## Recall proof

The closed campaign `sinex-9j9` built a multi-source recall demonstration with:

- filesystem, Git, shell, and browser evidence;
- a raw Atuin/Git baseline arm;
- resolvable source evidence;
- a one-command regeneration path;
- a cold-reader gate;
- explicit capture caveats.

This is important because it does not merely show a rendered timeline. It compares the product output with lower-level source queries and requires a fresh reader to explain what the artifact proves and what it does not prove.

## Current coverage honesty

The in-progress `sinex-jdp` work makes source incompleteness visible rather than treating a silent source as an empty source.

Its first proof found:

- 64 evaluated sources;
- 58 sources with some coverage error;
- missing-material, missing-event, continuity, runtime-binding, and bridge-observation problems.

That is not a flattering result, but it is exactly the kind of result an evidence system should surface.

## Current source-quality work

The in-progress `sinex-60r` work addresses sources that existed operationally but were unusable semantically:

- approximately 2.4 million ActivityWatch events with empty application/title fields;
- an old Hyprland snapshot path producing approximately 582,000 duplicate-timestamp artifacts.

A source being “green” while emitting unusable content is a false readiness claim. This Bead is about making usefulness, not merely process liveness, part of the source contract.

## Current uncommitted work

The dirty working tree corresponds to `sinex-v7od`, which consolidates many individual Rust test binaries into aggregated harnesses such as:

- `tests/api.rs`;
- `tests/event_engine.rs`;
- `tests/runtime.rs`;
- `tests/sources.rs`.

The change removes approximately 357 lines of repetitive `Cargo.toml` test declarations. The objective is to reduce relinking and memory pressure while preserving nextest’s per-test-function process isolation.

This is in progress rather than completed.

## Recent development emphasis

Recent commits focus on:

- retiring GitHub Issues in favor of Beads;
- correcting test-filter build scoping;
- proving a cascade integrity check redundant;
- separating API database-pool sizing;
- detecting schema function-body drift;
- retaining a GIN index in large lifecycle cascades;
- preventing oversized replay responses;
- integrating child settlement into the runtime;
- reconnect and event-type coverage for a desktop source.

This is a project hardening replay, schema convergence, query plans, settlement, and operational correctness—not one spending most of its effort on superficial UI additions.

---

# 6. The Beads roadmap

The important planning graph is the [Sinex Beads export](sandbox:/mnt/data/sinex-src/sinex/sinex-beads-export.jsonl).

## Current in-progress work

The five active Beads are:

- `sinex-jdp`: standing capture-losslessness and coverage-error measurement;
- `sinex-x9i`: separation of activity evidence from self-observation telemetry;
- `sinex-60r`: repair of semantically unusable attention sources;
- `sinex-v7od`: consolidation of test binaries;
- `sinex-dkq`: audit and pruning of stale agent worktrees.

The first three indicate the current product priority: making evidence and source-health claims honest before building more interpretation on top.

## Principal programs

### `sinex-r6d`: substrate honesty

This program is explicitly framed as “no silent data loss, no false claims.”

It owns loss windows and false readiness across:

- in-memory handoff before durable transport;
- source checkpoint advancement;
- recovery spool limits;
- health reporters;
- replay and invalidation;
- settlement;
- transport acknowledgment;
- partial source material.

The target is not merely a healthy process. It is a system able to explain where evidence could be lost and prevent downstream surfaces from silently presenting an incomplete result as complete.

### `sinex-n23`: robustness under load

This program targets sustained backlog behavior, bounded memory and disk use, paced catch-up, capacity warnings, and recovery without surgical manual intervention.

A transcript backfill containing millions of messages and very large sessions depends on this program. The full Polylogue archive should become one of its primary load proofs.

### `sinex-a4w`: the derived plane pays rent

This program addresses a common failure of ambitious data platforms: collecting and modeling everything without delivering useful recurring products.

Its requirement is that derived state answer real operator questions rather than merely exist as schema.

### `sinex-a4w.3`: cross-source composites

This is where Sinex’s distinctive payoff appears.

Planned composites include:

- attention streams;
- project attribution;
- work episodes;
- change episodes;
- Agent Work Packets;
- other cross-source temporal joins.

The project is correctly avoiding an intrinsic `session_commit` object. A conversation, a commit, a test run, and a Bead remain separate domain objects. Their relationship is derived from time, repository, branch, file, session, and task evidence.

### `sinex-7z0`: operator and agent UX

This program includes:

- a daily operating surface;
- an adjudication or judgment seat;
- context-budget arbitration;
- agent arrival and orientation;
- agent-triggered persistent sessions;
- operator-visible evidence and caveats.

Agent-triggered activity is intentionally constrained: handlers are operator-declared rather than self-registering autonomous loops.

### `sinex-cem`: demo portfolio

The demonstrations are intended as falsifiable product proofs, including:

- a self-diagnosing capture outage;
- retroactive privacy behavior;
- deterministic model-effect accounting;
- semantic retrieval evaluation;
- blinded resumption comparisons;
- self-hosting work reconstruction.

This is a healthy roadmap choice. It forces the infrastructure to answer whether a feature actually improves recall or agent work rather than assuming that more machinery implies more value.

### `sinex-rkv`: source breadth and quality

The source program expands coverage while requiring source contracts, privacy policy, evidence quality, and usable outputs.

### `sinex-nbi`: architecture and hygiene

This program addresses accumulated complexity, ownership boundaries, large modules, test organization, and architectural drift.

### `sinex-4j2`: Polylogue on Sinex

This is the dedicated interop program. Its epic contains the correct broad premise:

- Polylogue remains the AI-work product and ontology authority;
- Sinex becomes the generalized evidence substrate;
- raw materials, normalized observations, derived correlations, and Polylogue UX remain distinct layers;
- Phase B is proven by rebuilding Polylogue SQLite from Sinex-held materials and events;
- generic ontology collapse is rejected.

All nine children remain open. The implementation has not reached that target.

Several child notes are internally inconsistent with the epic, particularly a later metadata-only note in `sinex-4j2.1`. The epic’s settled design and the maximal architecture described here should supersede that note.

---

# 7. Sinex’s relevance to agents and LLMs

## It gives agents a world outside their own transcript

An agent transcript says what the agent saw, requested, claimed, and sometimes what its tools returned. It does not necessarily describe what actually happened on the machine.

Sinex adds:

- terminal history outside the transcript;
- Git state and changes;
- browser research;
- active-window context;
- filesystem observations;
- source outages;
- system state;
- task intent;
- other agents’ activity;
- actual deployment and verification evidence.

This enables the system to distinguish:

```text
agent claim
agent-visible tool result
machine-observed effect
repository outcome
operator judgment
```

That is considerably more useful than treating the assistant’s final prose as the account of record.

## It can provide durable, source-aware agent context

Sinex already has context-pack and orientation surfaces, and `sinex-7z0.6` plans a common context-budget arbiter.

Polylogue has a more developed AI-session context compiler, including evidence refs, assertion policy, omissions, caveats, and delivery snapshots.

The natural combination is:

1. Sinex supplies broad candidate evidence and coverage state.
2. Polylogue applies AI-work-specific selection, trust, lineage, and injection rules.
3. One shared budget policy prevents independent over-selection.
4. The final delivered context is stored as an evidence-bearing context snapshot.
5. Later analysis can determine what the agent actually received.

This supports rigorous questions about context:

- Was the critical prior failure omitted?
- Did the pack contain a stale assertion?
- Was a source unavailable?
- Did compaction remove necessary state?
- Did the agent receive a raw excerpt, an accepted lesson, or an unreviewed candidate?
- Which evidence was dropped under budget?

## It makes model effects accountable

The planned embedding worker, `sinex-5v6`, routes work through a local model gateway and records a model-effect ledger.

A model effect can include:

- provider and model;
- prompt or input identity;
- output identity;
- cost or budget usage;
- policy decision;
- replay and cache behavior;
- resulting projection or proposal.

This makes model-derived outputs reproducible and budgetable.

The first embedding run is gated by:

- `sinex-2k2`, a static cost fuse;
- `sinex-h2x`, which prevents agent-session and reflection content from being embedded before ownership and privacy are settled.

That quarantine should not be interpreted as “Sinex must not store transcripts.” Storage, embedding, and agent access are three different policies.

## It creates a safe promotion boundary

Agents and models can generate candidates without receiving authority to rewrite durable state.

This supports:

- extracted claims;
- suggested relations;
- proposed episode labels;
- suggested context policies;
- possible task or blocker observations;
- candidate lessons.

The operator or a declared policy judges those candidates. The accepted result becomes a new authoritative event, with the proposal and evidence still traceable.

## It can trigger agents from events

`sinex-7z0.9` plans persistent agent sessions triggered by events.

Examples might include:

- a capture source becomes stale;
- a build fails;
- a deployment changes state;
- a new provider export arrives;
- a Bead becomes unblocked;
- a long-running operation finishes;
- a privacy tripwire fires.

The safe shape is not “agents can register arbitrary loops.” The operator declares handlers and budgets. Agent outputs remain proposals or evidence unless separately authorized.

## It can evaluate whether agent memory works

Planned demonstrations include:

- a blinded resumption duel;
- semantic retrieval hit@k evaluation;
- deterministic model-effect accounting;
- work-packet reconstruction.

Combined with Polylogue’s existing memory experiments, this creates an environment in which context packs, embeddings, summaries, and reviewed memory can be tested rather than justified by intuition.

---

# 8. Fit in the broader landscape

## Compared with activity loggers

Activity loggers typically answer what application was active, which command ran, or which URL was visited.

Sinex adds:

- source material;
- temporal-quality distinctions;
- replay;
- explicit gaps;
- typed provenance;
- cross-source derivation;
- authority and judgment;
- lifecycle and deletion;
- agent-facing evidence contracts.

The current ActivityWatch problem—millions of records with empty meaningful fields—illustrates the difference between capturing rows and capturing usable evidence.

## Compared with observability stacks

Conventional observability focuses on services, traces, metrics, logs, and incidents.

Sinex includes some of that territory, but its scope is personal and cross-domain:

- human actions;
- local desktop activity;
- documents;
- browser history;
- tasks;
- agent sessions;
- source exports;
- long-lived personal context;
- operator judgments.

OpenTelemetry-shaped data can be one source or projection. It is not the whole ontology.

## Compared with event-sourced applications

An event-sourced application normally owns the events for one domain and emits them directly from its business operations.

Sinex often begins with external evidence that was not designed for event sourcing. It must acquire source material, reconstruct occurrence identity, represent clock quality, and distinguish evidence from interpretation.

That makes it closer to an evidentiary event platform than a conventional application event store.

## Compared with a data lake

A data lake stores many source bytes and supports large analysis.

Sinex adds:

- typed occurrence semantics;
- replay-aware interpretations;
- public evidence refs;
- domain reducers;
- explicit source coverage;
- operator-visible caveats;
- adjudication;
- local operational control.

A data lake can preserve an export. Sinex is trying to explain what claims may responsibly be made from it.

## Compared with personal knowledge management

A PKM system generally begins with notes and links.

Sinex begins with observed evidence and typed records. Notes can be source material, declarations, or workspace artifacts, but they are not forced to become the primary database.

The distinction is valuable for agents: a note saying “tests pass” and a structured process result showing exit code zero are not the same evidence.

## Compared with vector memory and RAG

Vector memory normally starts with chunks and similarity.

Sinex makes embeddings an optional, recorded model effect over an already identifiable evidence corpus. Similarity does not become authority, and retrieval quality is supposed to be measured.

## Compared with agent tracing

Agent tracing records one framework’s runs, tool calls, and handoffs.

Sinex plus Polylogue would cover:

- multiple providers and runtimes;
- imported historical artifacts;
- provider-native material;
- ambient machine activity;
- Git and task state;
- logical transcript lineage;
- context delivery;
- reviewed memory;
- cross-session and cross-agent work;
- source coverage and gaps.

That is a longitudinal evidence and memory layer rather than only run debugging.

---

# 9. The existing Polylogue bridge

The current implementation is intentionally thin.

The Sinex source contract `integration.polylogue` defines an external producer. The Polylogue daemon is expected to publish an `integration.polylogue.session_indexed` event directly to the admitted NATS stream. The current payload includes:

- session ID;
- origin;
- title;
- tags;
- content hash;
- creation and update times;
- message count;
- optional cost;
- optional primary model.

It explicitly excludes transcript text and describes occurrence identity as `(content_hash, session_id)`. The source contract is in [the current Polylogue bridge module](sandbox:/mnt/data/sinex-inspect/crate/sinexd/src/sources/source_contracts/polylogue.rs).

The corresponding integration-authority document classifies Polylogue as a `FederatedCanonicalMirror`, with Polylogue retaining normalized archive authority and Sinex receiving metadata or source-backed signals. See [integration authority](sandbox:/mnt/data/sinex-inspect/crate/sinexd/docs/sources/integration_authority.md).

On the Polylogue side, the closed decision `polylogue-6mv` states that Sinex must not ingest raw agent transcripts and should receive only privacy-preserving derived events. The open `polylogue-fs1.9` is designed around that boundary. Both are present in the [Polylogue Beads export](sandbox:/mnt/data/polylogue-src/polylogue/polylogue-beads-export.jsonl).

Under the requested maximal architecture, that doctrine is wrong.

The corrective interpretation is:

- raw private text should not be sprayed into generic NATS event payloads;
- generic Sinex MCP tools should not return raw private text by default;
- embedding should not begin before policy and cost ownership are settled;
- **but Sinex should canonically store the transcript material and durable transcript-domain history.**

Privacy should be enforced through storage encryption, privacy classification, capabilities, redaction, retention, and access paths—not by forcing the transcript backend to remain split across two unrelated authorities.

`polylogue-6mv` therefore needs to be superseded by a new Beads decision.

`polylogue-fs1.9` remains useful only after re-scoping it. Its low-volume events can serve as:

- notification signals;
- indexing wakeups;
- correlation summaries;
- health and lag indicators;
- low-privilege redacted feeds.

They should no longer define the storage boundary.

The existing Sinex epic `sinex-4j2` is much closer to the correct target. Its main design explicitly admits messages and tool calls as full-grain observations, stores raw material in Sinex, and defines Phase B as Sinex-primary evidence with a Polylogue SQLite rebuild proof. A later child note saying “structured metadata, not transcripts” contradicts the epic’s settled design and should be treated as stale.

---

# 10. The correct maximal architecture

“Sinex is the backend” can mean three different things:

1. **Byte authority:** where the raw and normalized content is durably retained.
2. **Domain-history authority:** where durable session, message, assertion, and lifecycle observations live.
3. **Read execution:** which database serves transcript pages, search, and analytics.

The correct maximal answer is:

| Dimension | Ultimate owner |
|---|---|
| Raw and normalized bytes | Sinex |
| Durable transcript-domain history | Sinex, under Polylogue-defined semantics |
| Current shared server-side projections | Sinex/PostgreSQL |
| Product-specific query and composition logic | Polylogue |
| Local/offline projections and indexes | Polylogue SQLite |
| AI-work ontology | Polylogue |
| Cross-domain evidence ontology | Sinex |
| Task intent and dependency state | Beads |
| User judgment authority | Sinex-backed durable records, exposed through Polylogue |
| UI and agent-work experience | Primarily Polylogue |

The architecture should look like this:

```text
Provider exports / runtime files / browser payloads / hooks / attachments
                                  |
                    Polylogue acquisition + normalization
                                  |
            ┌─────────────────────┴─────────────────────┐
            │                                           │
            ▼                                           ▼
   Sinex source-material registry              Typed transcript observations
   + content-addressed storage                 via admitted producer protocol
            │                                           │
            └─────────────────────┬─────────────────────┘
                                  ▼
                   Sinex canonical evidence plane
         raw materials + event history + judgments + effects
                                  |
               ┌──────────────────┴───────────────────┐
               ▼                                      ▼
     Polylogue-domain projections             Cross-source derivations
     sessions/messages/blocks/FTS              moments/work episodes/
     topology/usage/assertions                 Agent Work Packets
               │                                      │
               └──────────────────┬───────────────────┘
                                  ▼
                   Polylogue service and product
              CLI | web | MCP | context | analytics
                                  |
                     local SQLite edge replica
             offline operation / cache / local search / UI state
```

## Sinex should store both raw and normalized transcripts

Keeping only provider-native files would make Sinex a byte archive but leave every consumer dependent on reparsing provider-specific formats.

Keeping only normalized transcripts would lose the ability to audit parser behavior and reinterpret the original provider material.

The maximal design therefore preserves both:

### Provider-native evidence

Examples:

- ChatGPT export files;
- Claude Code JSONL;
- Codex rollouts;
- Gemini CLI histories;
- browser network or DOM capture;
- hook-event records;
- original attachments;
- provider usage exports.

### Polylogue-normalized material

Examples:

- stable normalized message records;
- content-block records;
- tool calls and results;
- context and protocol blocks;
- attachment references;
- usage lanes;
- provider identity fields;
- material-authorship classifications.

Normalized material should be written in immutable, bounded segments with explicit semantics and canonicalization versions.

The event records then point to exact normalized records or source anchors, while the normalized material links back to the provider-native evidence.

## Polylogue remains the domain kernel

Sinex must not decide:

- what constitutes a logical conversation;
- how shared prefixes are composed;
- whether a session is a fork, continuation, or subagent;
- how provider blocks become messages and actions;
- how compaction boundaries work;
- how physical and logical token accounting differ;
- which assertion kinds exist;
- how AI context is compiled;
- how transcript cards render.

Those remain Polylogue semantics.

Sinex stores and serves those semantics through versioned domain events and projections. A database backend does not need to own the meaning of every domain object it stores.

## PostgreSQL projections and SQLite are complementary

The current `sinex-4j2.3` Bead says Polylogue SQLite remains the right store for Polylogue read models. That is a sound transitional and standalone design, but too restrictive as the ultimate architecture.

A fully backed deployment should eventually have PostgreSQL projections such as:

```text
polylogue.sessions_current
polylogue.session_revisions
polylogue.messages
polylogue.message_revisions
polylogue.blocks
polylogue.tool_invocations
polylogue.tool_results
polylogue.session_edges
polylogue.attachments
polylogue.usage_observations
polylogue.assertions_current
polylogue.assertion_history
polylogue.context_deliveries
polylogue.projection_watermarks
polylogue.search_documents
```

These remain rebuildable projections over Sinex material and event history. They do not replace the event spine.

Polylogue SQLite remains valuable for:

- standalone operation without Sinex;
- offline use;
- local FTS acceleration;
- downloaded material cache;
- local vector cache;
- local UI state;
- projection watermarks;
- a durable offline-write outbox;
- development and deterministic tests;
- fast transcript rendering without network round trips.

The important rule is not “SQLite disappears.” It is:

> **No irreplaceable durable state should silently exist only in SQLite when Polylogue is operating in Sinex-backed mode.**

---

# 11. The authority matrix

A maximal integration needs an explicit per-data-class authority contract.

| Data class | Sinex-backed authority | Polylogue SQLite role |
|---|---|---|
| Provider-native artifacts | Sinex material/CAS | Local spool or cache |
| Original attachments | Sinex material/CAS | Download cache |
| Normalized transcript segments | Sinex material/CAS | Local projection source/cache |
| Session/message/block history | Sinex events and domain history | Rebuildable local index |
| Current transcript view | Rebuildable PostgreSQL projection | Rebuildable edge projection |
| Logical session topology | Polylogue-defined, durably stored in Sinex | Local composed-read index |
| Usage and cost observations | Sinex durable events | Local analytic cache |
| Search indexes | Rebuildable projection | Local FTS/vector acceleration |
| Candidate assertions | Sinex durable proposal history | Offline replica/outbox |
| Accepted/rejected judgments | Sinex authority records | Local replica |
| Context policy | Sinex-backed durable state | Local replica |
| Context image bytes | Sinex artifact/material | Local cache |
| Context-delivery occurrence | Sinex durable event | Local audit projection |
| UI layout and panel state | Local SQLite | Authoritative locally |
| Last scroll position/read cursor | Usually local; optionally synchronized | Local authoritative state |
| Local query history | Local SQLite | Local authoritative state |
| Daemon telemetry | Sinex reflection or local ops | Local diagnostic cache |
| Pending offline writes | Local durable outbox until accepted | Temporary local authority |
| Beads task state | Beads | Reference/cache only |

The key change from the current `sinex-4j2.3` design is user-authored state.

Marks, analysis notes, corrections, lessons, handoffs, judgments, and context policy are irreplaceable. Merely emitting an event saying “the user annotated this session” is not enough to rebuild the annotation’s content or state.

In backed mode, the durable assertion lifecycle should therefore be stored in Sinex. Local SQLite can remain the offline editing and read replica.

Ephemeral presentation state can remain local.

---

# 12. How transcript ingestion should work

A robust transcript ingestion protocol should operate at the level of a complete transcript revision, not as a loose sequence of unrelated events.

## Step 1: Stage provider-native material

Polylogue acquires or observes:

- a provider file;
- a browser payload;
- a hook stream;
- an attachment;
- a usage export.

It stages the bytes in Sinex and receives a stable material ID.

The material record carries:

- source identifier;
- provider origin;
- content digests;
- size;
- MIME or record format;
- acquisition interval;
- privacy class;
- replayability;
- completeness and coverage state.

## Step 2: Create immutable normalized segments

Polylogue parses the provider artifact and emits bounded normalized segments.

A segment might contain:

- a fixed range of messages;
- one append-only acquisition interval;
- one attachment manifest;
- one usage batch.

The segment is immutable once sealed.

This is safer than one mutable per-session JSONL file. A growing JSONL works only while the source is truly append-only. Many exports are regenerated, reordered, or rewritten. Re-serialization can shift every byte offset and invalidate anchors.

The better pattern is:

```text
stable session identity
    └── revision manifest
          ├── raw material refs
          ├── normalized segment 1
          ├── normalized segment 2
          ├── normalized segment 3
          ├── attachment refs
          ├── parser semantics version
          ├── expected object counts
          └── aggregate content digest
```

## Step 3: Emit stable domain observations

Events should include a full vocabulary such as:

- `agent.session.started`;
- `agent.session.resumed`;
- `agent.session.compacted`;
- `agent.session.ended`;
- `agent.message.emitted`;
- `agent.tool_call.started`;
- `agent.tool_call.returned`;
- `agent.subagent.started`;
- `agent.subagent.stopped`;
- `agent.attachment.referenced`;
- `polylogue.export.imported`;
- `polylogue.usage.observed`;
- assertion and judgment lifecycle events;
- context-delivery events.

Each carries:

- stable Polylogue object ID;
- object revision;
- session-local sequence;
- provider-native IDs where available;
- semantics version;
- source-domain timestamp and quality;
- material ID and exact record anchor;
- compact typed metadata;
- privacy classification;
- revision/batch identity.

Bulk message or tool output text should remain in material, not be duplicated into every transport payload.

## Step 4: Admit a revision bundle

A transcript revision can contain thousands of related records. Partial visibility would produce broken session pages and incorrect counts.

The ingress protocol therefore needs:

- `ingest_batch_id`;
- revision manifest;
- expected event and material counts;
- content digests;
- child settlement;
- a finalization or commit frontier;
- failure and retry state.

A projection should expose either:

- the previous complete revision; or
- the new complete revision.

It should not silently expose a half-imported session as complete.

Sinex’s current settlement and commit-frontier work provides the right building blocks, but Polylogue needs a domain-level revision protocol over them.

## Step 5: Build domain projections

A Polylogue-owned reducer or projection worker reconstructs:

- current session state;
- message and block ordering;
- tool-call/result pairing;
- topology;
- usage;
- logical composition;
- assertion state;
- context-delivery history.

The implementation can remain outside the Rust runtime, but it should use a registered, scoped projection contract rather than ad hoc direct PostgreSQL writes.

The existing external-producer boundary is a strong starting point because it avoids porting Polylogue’s provider parsers to Rust.

## Step 6: Update the SQLite edge replica

Polylogue consumes the confirmed Sinex frontier and updates its SQLite indexes.

Every local result should be able to report:

- Sinex material frontier;
- admitted event frontier;
- server projection frontier;
- local SQLite frontier;
- FTS frontier;
- embedding frontier;
- known coverage gaps.

This combines Sinex’s source continuity with Polylogue’s multi-tier readiness model.

## Step 7: Support full rebuild

The decisive Phase B proof remains:

```bash
polylogue rebuild --from-sinex
```

The command should be able to drop all rebuildable Polylogue SQLite state and reconstruct it from:

- Sinex materials;
- transcript-domain events;
- assertion and judgment history;
- context deliveries;
- lifecycle and deletion state;
- model effects where relevant.

The parity report should distinguish:

- true defects;
- intentionally local UI state;
- unsupported legacy material;
- semantically changed parser versions;
- unavailable embeddings;
- stale or deleted content.

---

# 13. The main impedance mismatches

The projects fit strongly, but there are real mismatches. They are not reasons to avoid the architecture; they are the contracts that must be designed correctly.

## 1. Interpretation identity versus stable domain identity

**Sinex:** an event UUID identifies one interpretation and changes on replay.

**Polylogue:** session, message, block, assertion, and context refs must survive replay.

**Resolution:** maintain separate stable object IDs, revision IDs, interpretation event IDs, and material anchors.

The current `sinex-4j2.7` assumption that stable cross-resolution can be computed without a new identity table is too optimistic. Re-export, provider drift, resegmentation, and collision handling require an explicit identity and alias registry.

## 2. A session is a mutable aggregate, while events are immutable occurrences

A Polylogue session can grow, be resumed, receive attachments, be compacted, or be re-exported.

The current `(content_hash, session_id)` occurrence identity is insufficient:

- the session ID is the stable object;
- the content hash identifies a content revision;
- changing the content hash should not imply a different logical session;
- a reparse can change normalized content without changing the underlying provider occurrence.

**Resolution:** stable session identity plus explicit revision manifests and lifecycle events.

## 3. Material anchors can move

The `sinex-4j2.1` design currently discusses line or byte anchors into a session JSONL.

That works for immutable or append-only material. It fails when a file is regenerated or normalized differently.

**Resolution:** immutable sealed segments, record IDs, and versioned manifests. Material offsets remain exact evidence coordinates, while stable message identity survives resegmentation.

## 4. Sinex provenance is not Polylogue lineage

Sinex `source_event_ids` means “this interpretation was derived from these events.”

Polylogue relationships such as:

- continuation;
- fork;
- shared prefix;
- subagent;
- compaction;
- context inheritance;

are domain topology.

**Resolution:** store them as typed Polylogue relationships and projection tables, not derivation parents.

This is also essential for deletion. Deleting one fork must not cascade through a shared-prefix edge as though the parent message were a derivation dependency.

## 5. Physical evidence and logical conversation composition differ

Sinex should preserve every physical provider artifact and observation.

Polylogue must calculate:

- composed logical sessions;
- copied prefixes;
- unique tails;
- physical versus logical token usage;
- compaction boundaries;
- replayed lineage.

The current Polylogue live archive demonstrates why this matters: approximately 399.9 billion physically accounted tokens versus approximately 292.9 billion logical high-water tokens, a replay gap around 107 billion.

**Resolution:** preserve physical evidence in Sinex; keep Polylogue-owned logical composition and accounting projections. Generic event counts must never be presented as unique work counts.

## 6. Event granularity can explode

The current Polylogue archive contains millions of messages. Adding every internal row as a separate Sinex event could create unnecessary event and index pressure.

At the same time, excluding messages would make the integration too shallow.

**Resolution:** every stable normalized observation is admissible, but not every UI implementation row needs an event.

First-class events should include:

- messages;
- tool calls and results;
- session lifecycle;
- compaction;
- subagents;
- attachments;
- usage;
- assertions and judgments;
- context delivery.

Some subordinate content-block structure can live in the normalized material and domain projection unless it has independent lifecycle or forensic significance.

“All observations are admissible” does not mean “blindly map every SQLite row to `core.events`.”

## 7. Transcript revisions need transaction-like completeness

Sinex transport and derivation are asynchronous. A session revision contains many dependent observations.

**Resolution:** revision manifests, settlement, expected counts, digests, and projection frontiers. A transcript view should know whether a revision is complete, partial, failed, or superseded.

## 8. Timestamp ordering is not enough

Provider messages can have equal, missing, rounded, or inconsistent timestamps. Tool blocks have strict local order. Fork topology imposes causal relationships.

**Resolution:** preserve `ts_orig` and quality, but add explicit:

- session sequence;
- provider ordinal;
- block ordinal;
- tool-call correlation;
- causal predecessor;
- revision ordering.

`ts_coided` must never become transcript order.

## 9. Query models differ

Sinex’s current typed query-unit grammar is relatively generic and event oriented.

Polylogue’s query model includes:

- nested transcript views;
- action sequences;
- session pipelines;
- logical lineage composition;
- structural failure queries;
- text and semantic retrieval;
- costs and token lanes;
- assertion and context views.

**Resolution:** add a Polylogue domain-query service or descriptor family over Sinex-backed projections. Do not force transcript pages to scan generic `core.events`.

The two query languages can remain distinct while sharing refs, intervals, envelopes, caveats, and privacy semantics.

## 10. Schema evolution differs

Sinex uses declarative PostgreSQL convergence and explicit backfills.

Polylogue uses independently versioned SQLite tiers and rebuildable indexes.

**Resolution:** version separately:

- source-material format;
- transcript observation schema;
- Polylogue domain semantics;
- PostgreSQL projection schema;
- SQLite projection schema;
- search and embedding recipes.

Parity should compare domain behavior, not raw DDL.

## 11. Assertions overlap but are not identical

Polylogue has rich assertion kinds, lifecycle, scope, evidence refs, staleness, and context-injection policy.

Sinex has a broader planned proposal/judgment/finalizer substrate and `ClaimSupport`.

**Resolution:** Polylogue retains the assertion domain model. Sinex stores the durable lifecycle and generic authority metadata.

A combined assertion might include:

```text
polylogue assertion id
kind and scope
author and authoredness
candidate/accepted/rejected/superseded state
evidence refs
ClaimSupport
judgment ref
validity/freshness
context injection policy
supersession history
```

`ClaimSupport` augments rather than replaces Polylogue’s evidence and policy fields.

## 12. Context packs overlap

Sinex has generic context packs. Polylogue has context images and delivery snapshots.

Two independent context stores would create ambiguity over what was actually given to the agent.

**Resolution:**

- Sinex supplies broad evidence candidates, source gaps, and ambient context;
- Polylogue owns AI-work context compilation;
- one shared artifact contract stores the final pack;
- one context-delivery event records exactly what crossed the model boundary;
- generated pack text remains a workspace artifact, while the delivery occurrence is durable evidence.

## 13. Embedding ownership

The `sinex-4j2.8` Bead has ratified a shared model-effect ledger rather than permanent double embedding.

The proposed key `(provider, model, content-hash)` is not sufficient. It can incorrectly reuse vectors produced with different preprocessing.

The key should include at least:

```text
input content digest
canonicalization version
chunk selector and chunking version
provider
model
model revision
dimensions
task or input type
normalization policy
```

The vector remains a projection. The effect ledger records how it was produced.

## 14. Content-addressing algorithms differ

Polylogue uses SHA-256. Sinex uses BLAKE3-oriented CAS metadata.

**Resolution:** retain a multihash descriptor. Do not rehash away either project’s existing contract, and do not make either digest the domain identity.

## 15. Privacy is currently enforced partly by non-storage

The metadata-only bridge reduced privacy risk by leaving raw text elsewhere. The maximal backend centralizes more sensitive evidence.

**Resolution:**

- raw-text capability distinct from metadata search;
- privacy classes for transcript, tool output, reasoning, attachment, and context material;
- scoped service tokens;
- generic MCP redacted by default;
- Polylogue service authorized to retrieve transcript content;
- encrypted host storage;
- explicit export and rendering redaction;
- model-input policy;
- physical deletion and cache invalidation.

The generic Sinex MCP should remain unable to fetch arbitrary transcript bytes by default.

## 16. Append-only audit and forgetting can conflict

A transcript deletion must propagate through:

- raw artifacts;
- normalized segments;
- event interpretations;
- PostgreSQL projections;
- SQLite replicas;
- FTS;
- embeddings;
- context packs;
- reports;
- model-effect inputs.

Logical session topology may share evidence across several sessions.

**Resolution:** domain-aware tombstones, material reference accounting, physical purge operations, derived-cascade invalidation, and explicit proof that caches and model projections have been removed.

A domain relationship such as shared prefix must not be mistaken for derivation lineage during cascade.

## 17. Deployment weights differ

Sinex requires PostgreSQL, NATS, NixOS/systemd in its canonical deployment. Polylogue is comparatively lightweight and SQLite-native.

**Resolution:** retain two explicit profiles:

- **standalone Polylogue:** SQLite is authoritative;
- **Sinex-backed Polylogue:** Sinex is durable authority and SQLite is a local replica/projection.

Do not create silent dual-master behavior.

Offline writes in backed mode should enter a durable SQLite outbox, later accepted into Sinex under revision and conflict rules.

## 18. Agent tool surface is too large

Sinex exposes 68 MCP tools and Polylogue 96.

Attaching all 164 to every agent would impair discovery and decision-making.

**Resolution:** create a small composed workflow surface, likely with operations resembling:

- orient;
- search;
- resolve/show;
- compile context;
- inspect work packet;
- propose assertion;
- inspect readiness;
- judge through an explicit operator surface.

Polylogue should generally be the front door for AI-work questions. It can delegate ambient-machine and cross-source retrieval to Sinex internally. Full low-level tool catalogs remain available for specialized analysis.

## 19. Product surfaces overlap

Both projects have CLI, MCP, TUI, web, context, and evidence ambitions.

**Resolution:**

- Polylogue owns transcript and agent-work UX;
- Sinex owns generic evidence, source health, operations, and cross-domain workbench UX;
- each deep-links to the other through stable refs;
- shared card and envelope descriptors can reduce duplication;
- neither product should become a thin skin over the other.

---

# 14. What Sinex gives Polylogue

## A durable, scalable canonical backend

Polylogue’s current SQLite tiering is thoughtfully designed, but its live archive is already measured in millions of messages and tens of gigabytes of material.

Sinex provides:

- PostgreSQL and Timescale-scale persistence;
- a content-addressed material plane;
- replay and archive lifecycle;
- source-material verification;
- paced historical import;
- settlement and operations;
- durable cross-device potential;
- a single retention authority.

## Ambient reality

Polylogue knows what the AI conversation says happened.

Sinex can add:

- terminal commands outside provider-visible tools;
- actual Git history;
- browser research;
- filesystem changes;
- active project/window context;
- deployment state;
- source gaps;
- task lifecycle.

This makes claim-versus-evidence analysis substantially stronger.

## Capture completeness and caveats

A Polylogue query can inherit Sinex source continuity:

> There are no browser events in this interval because the browser source was stale.

That is much more useful than an empty browser section.

## Cross-source temporal joins

Sinex’s moment-search design can turn transcript hits into bounded intervals and gather heterogeneous evidence from those intervals.

Polylogue alone would need to ingest every machine source to provide that capability.

## Replay and reinterpretation

A provider parser update can:

- preserve the raw artifact;
- archive the old interpretation;
- produce a new semantics version;
- compare old and new projections;
- retain stable domain refs.

## Model-effect accounting

Embeddings, summarizations, classifiers, and context analyses can use one recorded effect ledger and one budget policy.

## Unified retention and deletion

A single evidence lifecycle can coordinate transcript materials, derived events, vectors, context artifacts, and local replica invalidation.

## Agent and work-episode infrastructure

Sinex can join Polylogue sessions with:

- Beads;
- Git;
- CI;
- terminal verification;
- focus intervals;
- other agents;
- deployment effects.

That gives Polylogue an Agent Work Packet without forcing it to become a universal machine-activity database.

---

# 15. What Polylogue gives Sinex

## A mature AI-work ontology

Sinex currently has a broad event ontology but lacks Polylogue’s detailed understanding of:

- providers and origins;
- human versus assistant versus protocol material;
- messages and content blocks;
- structured tool calls and results;
- sessions, forks, continuations, and subagents;
- compaction and context boundaries;
- physical versus logical transcript composition;
- token and cost lanes;
- assertion and memory policy.

Trying to recreate that generically inside Sinex would be wasteful and would weaken both projects.

## A semantically rich high-value source

Agent sessions are unusually valuable events because they contain:

- intent;
- plans;
- attempted actions;
- structural outcomes;
- explanations;
- failures;
- reflection;
- delegated work;
- context transitions;
- explicit model usage.

This source can become one of the strongest consumers of Sinex’s evidence model.

## A demanding proof of the substrate

Polylogue exercises difficult Sinex properties:

- millions of related events;
- huge individual sessions;
- append and rewrite behavior;
- stable identity across re-export;
- exact material anchors;
- shared-prefix topology;
- replay;
- privacy-heavy content;
- partial import;
- logical versus physical accounting;
- context provenance;
- user judgment.

If Sinex can support Polylogue correctly, many of its general substrate claims become substantially more credible.

## A developed agent-facing product

Polylogue’s transcript reader, context compiler, AI-work query language, postmortems, and MCP surfaces give agents a meaningful way to consume Sinex-backed evidence.

Sinex’s current challenge is making the derived plane “pay rent.” Polylogue is a natural flagship consumer.

## Context and memory discipline

Polylogue already distinguishes:

- evidence;
- candidate assertions;
- reviewed assertions;
- context policy;
- omissions;
- context delivery.

This can supply a mature domain implementation for Sinex’s more general proposal and judgment architecture.

## Provider drift handling

Polylogue’s provider-specific parser suite and fixture discipline prevent Sinex from needing to absorb fast-changing vendor formats into its Rust core.

The current external-producer approach should be deepened rather than replaced.

---

# 16. The emergent combined product

The combination is more than “Polylogue stored in Postgres.”

It becomes a local evidence and memory plane for human–agent work.

## Evidence-backed resumption

A new agent can receive:

- current Bead intent and blockers;
- relevant prior sessions;
- actual branch and repository state;
- failed and successful commands;
- accepted lessons;
- stale or superseded assertions;
- relevant browser research;
- source coverage caveats;
- exact refs to all supporting evidence.

The delivered context snapshot records what the agent received.

## Claim versus machine effect

A query can compare:

```text
assistant says: “tests pass”
Polylogue tool result: exit code 0
Sinex terminal history: matching command observed
Git state: expected files changed
CI state: check passed
```

Or identify where those layers disagree.

## Compaction forensics

Polylogue knows where compaction occurred and which summary replaced prior context.

Sinex can add:

- surrounding terminal and Git activity;
- context-delivery evidence;
- source availability;
- later outcomes.

The combined system can test whether a failure followed from lost context rather than merely noting that compaction happened.

## Stale-memory detection

An accepted assertion may have been valid against commit A but stale after commits B and C.

Sinex can observe the relevant repository change. Polylogue knows which assertion references the old evidence. A derivation can mark it for review rather than silently injecting it.

## Agent Work Packets

The planned `sinex-a4w.3.9` object joins:

- Polylogue agent sessions and subagents;
- Bead intent;
- repository and branch;
- commits and pull requests;
- CI/checks;
- test and development-tool runs;
- tool spans;
- source materials;
- outcomes;
- optional reviewed lessons.

Every leg remains a ref to its original domain. Missing legs are explicit gaps.

This is a strong answer to “what did this agent actually accomplish, and what evidence supports that?”

## Cross-agent coordination

Polylogue can model:

- agent session topology;
- delegated work;
- context transfer;
- agent coordination messages;
- handoffs.

Sinex can persist those observations and correlate them with:

- actual resource activity;
- worktrees and branches;
- files and commands;
- task state;
- outcomes.

Beads remains the task authority.

## Longitudinal model evaluation

The system can evaluate:

- cost versus verified outcome;
- context size versus task completion;
- retrieval strategy versus resumption quality;
- compaction versus failure rate;
- model or provider changes;
- silent continuation after tool failure;
- time spent versus produced artifacts;
- repeated failed approaches.

## Event-triggered agent operations

An operator can declare that certain evidence patterns should open or notify an agent session:

- a capture source has become stale;
- a deployment is unhealthy;
- an accepted assertion may be stale;
- a Bead has become ready;
- a repeated failure pattern appears;
- a historical import finishes;
- a privacy policy detects forbidden content.

The agent’s response remains an evidence-bearing proposal rather than an unreviewed mutation.

---

# 17. The composed agent interface

The combined system should not expose agents to two unrelated universes of tools.

A preferable interaction is:

```text
polylogue.orient
    explains AI-work refs, evidence classes, trust, and backend readiness

polylogue.search
    searches transcript-domain material and can request ambient Sinex expansion

polylogue.show
    resolves session/message/tool/assertion/work-packet/Sinex evidence refs

polylogue.context
    compiles bounded AI-work context using Polylogue policy and Sinex evidence

polylogue.work_packet
    returns intent, sessions, machine activity, changes, verification, outcomes

polylogue.propose
    records a candidate assertion or handoff; never silently promotes it

sinex.source_health / sinex.operations
    remain available for infrastructure specialists
```

Internally, a query such as:

> What commands did I run while discussing replay corruption?

can become:

1. Polylogue text or semantic search finds message refs.
2. Polylogue maps those refs to session intervals.
3. Sinex moment search gathers terminal, browser, Git, and filesystem evidence.
4. Polylogue renders the result as an agent-work view.
5. Every result retains native refs and caveats.
6. The context compiler can select from the result without flattening provenance.

The front door should be domain oriented. The backing evidence remains federated.

---

# 18. Correcting the Beads roadmap

The current `sinex-4j2` epic contains most of the correct architecture, but several decisions need strengthening for the maximal target.

## Decision correction

A superseding Polylogue Bead should replace `polylogue-6mv` with this authority rule:

> Sinex is the canonical durable evidence backend for raw and normalized AI-session material in Sinex-backed deployments. Polylogue retains ontology, normalization semantics, composition, product behavior, and standalone capability. Raw text is protected by capability and material policy, not excluded from Sinex storage.

`polylogue-fs1.9` should become an optional notification and redacted-correlation lane rather than the storage boundary.

The later metadata-only note in `sinex-4j2.1` should be removed or explicitly superseded.

## Phase 1: cross-project identity and authority contract

This phase should define:

- stable Polylogue object IDs;
- revision IDs;
- Sinex interpretation event IDs;
- material occurrence refs;
- alias and reconciliation behavior;
- source and normalized material roles;
- lifecycle ownership;
- standalone versus backed profile.

Relevant existing Beads include:

- `sinex-4j2.7`;
- `sinex-bm1`;
- `sinex-05a`;
- Polylogue’s ref and origin work.

The “no new table” assumption in `sinex-4j2.7` should be removed. Stable cross-replay identity deserves an explicit ledger.

## Phase 2: full material storage

Implement:

- provider-native material staging;
- normalized immutable segment material;
- attachment storage;
- multi-digest content descriptors;
- material links;
- revision manifests;
- coverage accounting.

This is part of `sinex-4j2.1` and `sinex-4j2.6`, with pacing and robustness supplied by `sinex-n23` and `sinex-r6d`.

## Phase 3: full observation vocabulary

Land full-grain session, message, tool, subagent, compaction, attachment, usage, assertion, and context-delivery observations.

Every event must have:

- stable domain identity;
- material evidence;
- source-domain time;
- explicit semantics version;
- declared volume class;
- named consumer.

The construct-validity gate should remain: no event vocabulary lands without a read or product operation that uses it.

## Phase 4: bundle settlement and projection completeness

Add:

- revision bundle manifests;
- expected counts;
- commit frontier;
- retry semantics;
- partial-state visibility;
- projection watermarks;
- confirmed ingest acknowledgments.

This should reuse Sinex settlement rather than inventing a parallel transaction protocol.

## Phase 5: PostgreSQL Polylogue projections and SQLite rebuild

Build Polylogue-domain projections in Sinex/PostgreSQL, while preserving SQLite as the edge read model.

The proof is:

- drop SQLite rebuildable tiers;
- reconstruct from Sinex;
- compare physical sessions, messages, blocks, topology, usage, assertions, and context deliveries;
- explain every difference;
- prove standalone mode separately.

This strengthens `sinex-4j2.3`.

## Phase 6: durable user state

Move backed-mode authority for:

- marks;
- corrections;
- notes;
- lessons;
- handoffs;
- judgments;
- context policy;
- saved durable analyses;

into Sinex events and material.

Retain only genuinely local presentation state in SQLite.

This should align Polylogue assertions with:

- `sinex-7z0.4`;
- `sinex-8cr`;
- `sinex-0vx`;
- the proposal/judgment/finalizer contract.

## Phase 7: common retrieval and model-effect ledger

Complete:

- shared embedding recipe identity;
- effect reuse;
- privacy allowlists;
- retrieval-quality evaluation;
- lexical/vector/hybrid recipe recording;
- query-result provenance.

Relevant Beads include:

- `sinex-4j2.8`;
- `sinex-5v6`;
- `sinex-h2x`;
- `sinex-cem.5`;
- `sinex-cem.4`.

## Phase 8: reverse context and cross-source products

Deliver:

- interval context for Polylogue session views;
- moment search;
- work episodes;
- project attribution;
- Agent Work Packets;
- Beads intent linkage;
- Git and CI verification;
- context delivery from the combined evidence plane.

Relevant Beads include:

- `sinex-4j2.2`;
- `sinex-4j2.5`;
- `sinex-a4w.3.4`;
- `sinex-a4w.3.9`;
- `sinex-hlv`;
- Polylogue’s coordination and context programs.

## Phase 9: lifecycle, security, and offline proof

Prove:

- raw-content capability separation;
- selective forgetting;
- shared-prefix-safe deletion;
- vector and cache purge;
- offline writes and reconnect;
- no silent dual master;
- bounded historical backfill;
- interactive SLOs;
- coverage-error reporting;
- degraded-mode honesty.

Relevant Sinex programs include:

- `sinex-r6d`;
- `sinex-n23`;
- `sinex-jdp`;
- `sinex-1wm`;
- `sinex-cem`.

## Phase 10: agent and operator proof

Run falsifiable demonstrations:

- blinded resumption with and without combined context;
- stale-memory negative controls;
- claim-versus-machine-effect analysis;
- compaction-loss reconstruction;
- one full Agent Work Packet;
- one deletion and rebuild proof;
- one interrupted historical import and resume;
- one offline Polylogue write reconciled into Sinex;
- one provider reparse preserving stable refs.

The integration is not complete when schemas exist. It is complete when the combined product answers questions that neither project can answer independently.

---

# 19. What should remain in Polylogue SQLite

The maximal backend does not require eliminating SQLite.

SQLite remains an excellent edge store for a single-user local application.

The durable categories that can legitimately remain SQLite-only are narrow:

- window geometry;
- selected tabs and filters;
- scroll position;
- local query history;
- temporary analysis scratch state;
- local cache eviction metadata;
- downloaded blob cache;
- projection watermarks;
- pending offline outbox entries before Sinex acceptance;
- test fixtures and deterministic demo state.

The following should **not** remain SQLite-only in Sinex-backed mode:

- transcript bytes;
- attachment bytes;
- normalized message history;
- stable object identity;
- durable topology;
- accepted corrections;
- analysis notes intended for later use;
- handoffs;
- reviewed lessons;
- assertion judgments;
- context policy;
- delivered context history;
- retention and deletion state.

A useful rule is:

> Local state may be private, ephemeral, or reconstructable. Shared meaning, durable evidence, and user judgment belong in Sinex.

---

# 20. Risks

## Dual authority

The most dangerous failure is allowing both SQLite and Sinex to accept authoritative updates to the same object without a conflict protocol.

The profile must declare one durable authority per data class.

## Identity drift

Using content hashes, event IDs, or material offsets as stable domain identity would produce broken refs after reparse or re-export.

The identity ledger is not optional.

## Partial transcript visibility

Asynchronous event ingestion can expose a half-complete revision unless bundle settlement is explicit.

## Generic-event sludge

The wrong implementation is to put every Polylogue row into arbitrary JSON events and make the UI query `core.events` directly.

That would lose performance, domain constraints, composition semantics, and product clarity.

## Privacy blast radius

A unified backend centralizes the most sensitive local content. Capability control, deletion, model-input policy, and host security must become stronger than they are in the metadata-only bridge.

## Query and tool sprawl

Two rich products can easily create duplicate search, context, MCP, and rendering surfaces. The agent-facing workflows must be consolidated even while domain ownership remains separate.

## Backfill pressure

The live Polylogue archive already contains several million messages, and some sessions are very large. The import must be segmented, paced, resumable, and reconciled. Sinex has demonstrated a larger event order of magnitude, but not this exact content and projection workload.

## Replay and deletion interaction

Replayed interpretations, shared session prefixes, cached context packs, vectors, and reports create a complicated excision graph. Domain lineage must remain separate from derivation lineage.

## Operational weight

Polylogue’s current ease of running against local SQLite is valuable. Requiring full Sinex infrastructure for every user would shrink its applicability. Standalone operation must remain a first-class profile rather than a degraded afterthought.

---

# 21. Strategic assessment

The strategic fit is very strong.

The projects have complementary strengths:

```text
Sinex:
    broad evidence
    temporal and provenance semantics
    source coverage
    replay
    lifecycle
    operational durability
    cross-source joins

Polylogue:
    rich AI-session ontology
    provider normalization
    transcript composition
    tool semantics
    cost accounting
    reviewed memory
    context compilation
    agent-work UX
```

There is no fundamental conceptual mismatch.

The central technical mismatch is that Sinex events identify replay-specific interpretations, while Polylogue needs stable, richly structured domain objects. That is solved by a Polylogue domain identity and projection layer over Sinex—not by limiting the bridge to metadata.

The other significant mismatches—mutable session aggregates, material anchors, asynchronous settlement, query shape, user-state authority, hash algorithms, privacy, and offline operation—are all tractable if explicitly modeled.

The wrong architecture is:

```text
Polylogue keeps all real transcripts in SQLite
    ↓
Sinex receives a few metadata events
    ↓
both projects duplicate partial analytics
```

That preserves independence but leaves the most valuable evidence split, prevents true rebuild, weakens deletion and model-effect accounting, and reduces Sinex to a correlation sidecar.

Another wrong architecture is:

```text
all Polylogue concepts flattened into generic Sinex JSON events
    ↓
Polylogue becomes a thin generic-event viewer
```

That destroys Polylogue’s domain value and creates the “event-browser sludge” that `sinex-4j2` correctly rejects.

The correct architecture is:

```text
Sinex stores the raw and normalized evidence,
durable domain history, judgments, lifecycle, and effects.

Polylogue defines the AI-work domain,
builds the transcript and memory projections,
and presents the product.

SQLite remains the local edge store.
```

Under that design, Polylogue becomes one of the strongest possible domain products on top of Sinex, while Sinex becomes the backend that lets Polylogue escape provider silos and transcript-only reality.

The combined system’s strongest description is:

> **A local evidence, memory, and coordination plane for human–agent work. Beads records intended work. Polylogue records the cognitive and conversational work of humans and agents. Sinex records the wider evidentiary world in which that work occurred. Together they can reconstruct not merely what an agent said, but what it knew, what it attempted, what the machine did, what changed, what was verified, what remains uncertain, and what another agent should be allowed to remember.**

That is substantially more valuable than either a universal activity log or a transcript archive in isolation.

## Repository orientation

For understanding Sinex itself, the most useful reading sequence is:

1. [README](sandbox:/mnt/data/sinex-inspect/README.md) for the product shape and deployed architecture.
2. `crate/sinexd/docs/sources/staged_source_parser_substrate.md` for acquisition and parser design.
3. `crate/sinexd/docs/sources/evidence_lanes.md` for source material and epistemic evidence.
4. `crate/sinex-db/docs/schema/event-taxonomy.md` for event storage.
5. [Knowledge boundaries](sandbox:/mnt/data/sinex-inspect/crate/sinex-primitives/docs/knowledge_boundaries.md) for the material/event/projection/artifact hierarchy.
6. [Domain reducers](sandbox:/mnt/data/sinex-inspect/crate/sinex-primitives/docs/domain_reducers.md) for current-state projection semantics.
7. [Integration authority](sandbox:/mnt/data/sinex-inspect/crate/sinexd/docs/sources/integration_authority.md) for current sibling-system policy.
8. [MCP contract](sandbox:/mnt/data/sinex-inspect/crate/sinexctl/docs/mcp_readonly_server.md) for the agent evidence boundary.
9. [Current Polylogue source contract](sandbox:/mnt/data/sinex-inspect/crate/sinexd/src/sources/source_contracts/polylogue.rs) for the implemented metadata-thin bridge.
10. [Sinex Beads export](sandbox:/mnt/data/sinex-src/sinex/sinex-beads-export.jsonl), especially `sinex-4j2`, `r6d`, `n23`, `a4w`, `a4w.3`, `7z0`, `cem`, `8cr`, `0vx`, `5v6`, `908`, and `jdp`, for the real plan.
11. [Polylogue Beads export](sandbox:/mnt/data/polylogue-src/polylogue/polylogue-beads-export.jsonl), especially `polylogue-6mv` and `polylogue-fs1.9`, for the decisions that need to be superseded under the maximal architecture.

---

## Turn 78 (role=assistant)

# External-legibility package delivered

The complete 219-file package is here:

**[Download the Polylogue + Sinex external-legibility kit](sandbox:/mnt/data/Polylogue-Sinex-external-legibility-kit.zip)**  
[SHA-256 checksum](sandbox:/mnt/data/Polylogue-Sinex-external-legibility-kit.zip.sha256): `f436edb1814f39c1cb94749066d07c7a285f27a4c014986ad8c91a875cf9aad5`

Start with [00-START-HERE.md](sandbox:/mnt/data/legibility-kit/00-START-HERE.md), or use the [artifact map](sandbox:/mnt/data/legibility-kit/16-artifact-map.md).

This is not only an analysis package. It contains apply-ready repository patches, an executed Polylogue proof packet, a redesigned demo system, machine-readable claims and demo contracts, a Beads launch cut, a single-machine swarm plan, 32 forkable agent prompts, visual prototypes, and an executive deck.

## The most important strategic change

The projects should stop trying to explain themselves through inventories of infrastructure.

Their first-contact descriptions should be:

> **Polylogue is the local flight recorder and system of record for AI work.**

> **Sinex is the local, replayable evidence substrate for digital life and agent work.**

The joint story is:

> **Polylogue explains AI work. Sinex preserves the wider evidentiary world in which that work happened.**

In the maximal architecture, Sinex stores provider-native artifacts, normalized transcripts, attachments, durable Polylogue-domain history, judgments, context deliveries, model effects, and lifecycle state. Polylogue remains responsible for provider normalization, AI-work ontology, transcript composition, query semantics, context and memory policy, and the user-facing product. SQLite remains valuable as Polylogue’s standalone authority, offline replica, local search index, cache, and UI-state store.

That direction is developed in the [joint architecture one-pager](sandbox:/mnt/data/legibility-kit/08-joint-architecture-one-pager.md) and [full joint public story](sandbox:/mnt/data/legibility-kit/08a-joint-public-story.md).

# Concrete repository patches

## Polylogue

[Download the Polylogue patch](sandbox:/mnt/data/legibility-kit/patches/polylogue-external-legibility.patch)

It is based on commit `f6c1da997bea64bc6cd9670d9cbb8f7e7439ec51`.

The patch changes 30 repository paths. It:

- rewrites the README and package description around the flight-recorder category;
- improves the generated site’s landing hierarchy, navigation, proof cards, and project framing;
- adds demos, findings, public-claims, and Sinex-interop documentation;
- adds a machine-readable public claims ledger;
- states the transcript-complete maximal Sinex direction;
- changes the actual deterministic tour implementation, not merely its documentation;
- regenerates the tour transcript, report, recording recipe, command outputs, and GIF;
- adds or updates focused tests for the changed surfaces.

The tour no longer opens with a large fixture-audit dump. Its evidence order is now:

1. show one exact failed tool-result receipt;
2. aggregate failures using structural `is_error` fields rather than keyword matching;
3. demonstrate composed lineage while retaining original refs;
4. only then zoom out to archive-wide facets.

Its report explicitly separates what it proves from scale, provider-completeness, memory-uplift, deletion, invoice, and Sinex-backend claims.

## Sinex

[Download the Sinex patch](sandbox:/mnt/data/legibility-kit/patches/sinex-external-legibility.patch)

It is based on commit `b70a08d9e51cecc9e76b92f955a88183e99842cf`.

The patch changes nine repository paths. It:

- replaces infrastructure-first README framing with payoff-first framing;
- introduces a short documentation map and concepts/product explanation;
- gives agents and AI systems a clear place in the story;
- adds a rigorous demo portfolio, proof-artifact index, and public claims ledger;
- correctly characterizes the current demonstration as a PostgreSQL/API operational smoke proof rather than end-to-end acquisition proof;
- distinguishes the current metadata-thin Polylogue event from the ultimate backend boundary;
- states that transcript bytes belong in Sinex’s protected material plane rather than generic NATS payloads;
- records the target in which Sinex stores complete transcript evidence and Polylogue retains domain authority.

This patch changes architecture documentation and source-contract comments, not Sinex runtime behavior.

Both patches were tested by creating clean detached worktrees at their exact base commits, running `git apply --check`, applying the patches, and running `git diff --check`.

# Executed Polylogue proof artifact

The actual regenerated proof packet is under [polylogue-demo-tour](sandbox:/mnt/data/legibility-kit/polylogue-demo-tour/report.md).

Useful individual artifacts:

- [Human-readable evidence transcript](sandbox:/mnt/data/legibility-kit/polylogue-demo-tour/transcript.txt)
- [Machine-readable report](sandbox:/mnt/data/legibility-kit/polylogue-demo-tour/report.json)
- [Rendered tour GIF](sandbox:/mnt/data/legibility-kit/polylogue-demo-tour/demo-tour.gif)
- [First-frame preview](sandbox:/mnt/data/legibility-kit/polylogue-demo-tour/demo-tour-first-frame.png)
- [VHS recording recipe](sandbox:/mnt/data/legibility-kit/polylogue-demo-tour/recording.tape)

The generated private-data-free archive contains 11 sessions, 43 messages, five assertions, five origins, structured failures, attachments, copied lineage, subagents, compaction, context snapshots, observed events, and synthetic embeddings.

The verifier found **30 of 30 declared constructs**, no semantic problems, and no path leaks. The first evidence result appeared in 2.576 seconds and the complete four-step tour finished in 6.849 seconds.

# Full demo reconsideration

The most comprehensive artifact is [Demo portfolio v2: impressive because it is falsifiable](sandbox:/mnt/data/legibility-kit/02b-demo-portfolio-expanded.md).

The central change is to treat every demo as a bounded experiment with:

- one primary construct;
- a claim stated before execution;
- a declared independent oracle;
- negative and missing-evidence controls;
- a baseline arm;
- an explicit falsifier;
- resolvable receipts;
- a machine-readable packet;
- a human presentation;
- a non-claims section;
- interruption and regeneration behavior.

The accompanying contracts are:

- [Machine-readable demo portfolio](sandbox:/mnt/data/legibility-kit/11-demo-portfolio.yaml)
- [Demo Packet v2 JSON Schema](sandbox:/mnt/data/legibility-kit/10-demo-packet-v2.schema.json)
- [Validating Demo Packet example](sandbox:/mnt/data/legibility-kit/10-demo-packet-v2-example.yaml)

## The shared proof world: Incident 14:32

Rather than maintain a series of unrelated synthetic fixtures, the proposed portfolio uses one public-safe incident world.

It contains:

- a structured test failure;
- an assistant success claim that conflicts with that structural result;
- a later verified repair;
- a copied-prefix fork;
- a fresh-subagent control;
- a compaction summary that omits the failed attempt;
- disjoint input, cache, reasoning, and output usage lanes;
- an acquired attachment;
- candidate, accepted, rejected, superseded, and stale assertions;
- a compiled context image and delivery record;
- terminal, filesystem, Git, browser, desktop-focus, and Beads evidence;
- a deliberate source outage;
- parser semantics versions one and two;
- an ambiguous cross-material duplicate.

This lets many demonstrations reuse the same memorable event without contaminating their individual construct validity.

## Recommended public arc

The first launch should be a small public arc, not the entire internal demo catalog.

**Polylogue: The Receipts** asks whether the assistant’s account agrees with structural tool evidence. Its anti-grep control includes prose containing words such as “error” without a failed operation, and a genuine structured failure whose output does not depend on the word.

**Polylogue: Count It Once** shows that every physical artifact can remain inspectable while copied lineage is counted once in logical usage and activity views. A fresh-subagent control demonstrates that not every related session is deduplicated.

**Sinex: The Missing Source** contrasts a truly empty interval with an interval whose source was unavailable. The product must refuse to turn capture failure into a confident absence claim.

**Sinex: The System Changes Its Mind Honestly** parses the same material under two semantics versions, preserves both interpretations, and exposes which interpretation is currently promoted.

**Joint: The World Around the Claim** starts from an agent claim and reconstructs its transcript evidence, terminal activity, Git effects, Beads intent, verification outcome, and source-coverage caveats.

**Resume Under Oath** is the stronger memory demonstration. It compares raw-reference, generated-summary, and reviewed-context arms against a frozen checkpoint. It includes stale-memory traps and scores reconstruction against independently declared ground truth rather than model eloquence.

Longer-horizon demonstrations include Context Autopsy, Compaction Autopsy, Agent Work Packets, full Sinex-backed rebuild, crash receipt barriers, import-twice occurrence identity, disclosure control, retrieval evaluation, and blinded resumption.

# Rapid Polylogue presentability program

The primary execution artifact is [Polylogue presentability plan: a frontier-agent execution program](sandbox:/mnt/data/legibility-kit/03b-polylogue-agent-execution-expanded.md).

The plan deliberately excludes most of the long-term product vision from the first public cut. The launch-blocking wedge is:

1. truthful readiness, timeout, partial-result, and degraded-result behavior;
2. one stable category sentence and public claims ledger;
3. a narrow provider-neutral semantic renderer for shell, edit, and lineage evidence;
4. The Receipts;
5. Count It Once;
6. findings and proof pages;
7. clean-install and recording receipts;
8. cold-reader and adversarial-claims gates.

The proposed Beads subset is in [12-beads-launch-cut.csv](sandbox:/mnt/data/legibility-kit/12-beads-launch-cut.csv). It contains 28 entries across Polylogue, Sinex, and joint work.

Important Polylogue launch items include:

- `polylogue-0hqs` and `polylogue-bby.1` for bounded, truthful web behavior;
- `polylogue-3tl.12`, `.15`, `.16`, and `.8` for concise framing, anti-grep proof, public claims, and landing consistency;
- `polylogue-ap7` for the narrow semantic-renderer slice;
- `polylogue-212.2` for The Receipts;
- existing `polylogue-212.8` as the honesty companion proof;
- `polylogue-3tl.4` for a bounded findings shelf;
- `polylogue-3tl.7`, `.9`, and `.10` for installation, ownership, drift, and launch packaging;
- closed `polylogue-3tl.5` as the basis for regenerated, slower media.

The proposed merge order is deliberately dependency-oriented:

```text
scenario and oracle substrate
→ readiness/result-state vocabulary
→ shared semantic renderer
→ The Receipts
→ Count It Once
→ narrative grounded in actual commands
→ site routes and landing page
→ install proof and recordings
→ final integration and generated-surface checks
```

Polished public copy is not allowed to merge before the corresponding executable command and proof packet exist.

# Single-machine swarm design

The detailed coordination procedure is in the [single-machine frontier-agent swarm runbook](sandbox:/mnt/data/legibility-kit/05-single-machine-swarm-runbook.md). A machine-readable lane map is in [13-worktree-lanes.csv](sandbox:/mnt/data/legibility-kit/13-worktree-lanes.csv), and a bootstrap script is included at [scripts/bootstrap-worktrees.sh](sandbox:/mnt/data/legibility-kit/scripts/bootstrap-worktrees.sh).

The operating model favors throughput over attractive branch history.

One coordinator owns the integration worktree and public claims. Implementation agents own bounded file surfaces. Agents communicate through checked-in or shared handoff artifacts rather than relying on conversational memory. Every handoff contains:

- base commit;
- owned files;
- changed files;
- contracts introduced or consumed;
- exact tests run;
- generated artifacts;
- known failures;
- expected conflicts;
- recommended merge action.

The coordinator may cherry-pick useful commits, apply a patch extracted from a noisy branch, copy a clean file set, or replace conflicted work with the smaller implementation. Agents are not expected to preserve elegant branch histories.

The practical concurrency limit is resource-based:

- up to four Python-heavy implementation lanes concurrently;
- one browser or daemon visual lane at a time;
- one full validation lane at a time;
- documentation, review, claims, and fixture-design work may continue during tests;
- Sinex Rust builds should not compete with Polylogue full-suite runs unless the machine has substantial spare memory;
- branch-local archives, ports, databases, logs, and generated outputs remain isolated;
- full verification is run by the integration coordinator, not independently by every lane.

# Sixteen primary parallel fork prompts

The complete suite is in [08-fork-prompts.md](sandbox:/mnt/data/legibility-kit/08-fork-prompts.md), with individual files under [fork-prompts](sandbox:/mnt/data/legibility-kit/fork-prompts/).

The prompts are substantial, self-contained missions:

1. Build the shared Incident 14:32 corpus.
2. Implement Polylogue The Receipts.
3. Implement Polylogue Count It Once.
4. Land the first semantic transcript-renderer slice.
5. Make Polylogue degrade loudly.
6. Build claims and findings surfaces.
7. Rewrite Polylogue’s public surface.
8. Produce install, media, and launch artifacts.
9. Make Sinex externally legible.
10. Implement Sinex The Missing Source.
11. Implement Import It Twice and replay contrast.
12. Specify and patch the maximal Sinex–Polylogue contract.
13. Prototype stable identity and Sinex-backed rebuild.
14. Build the joint World Around the Claim demo.
15. Adversarially audit every public claim and demonstration.
16. Integrate the entire swarm.

Each prompt specifies scope, relevant Beads, design constraints, construct-validity requirements, controls, expected artifacts, verification, and handoff rules. They are written to be used directly as forks of this conversation.

A second set of 16 narrower, repository-owned tasks is in [14-alternate-worktree-prompts.md](sandbox:/mnt/data/legibility-kit/14-alternate-worktree-prompts.md). These are useful as a second wave or where strict file ownership is more important than shared synthesis.

# Visual and presentation artifacts

The executive presentation is available in both formats:

- [Polylogue + Sinex external-legibility deck — PDF](sandbox:/mnt/data/legibility-kit/Polylogue-Sinex-external-legibility.pdf)
- [Editable PowerPoint](sandbox:/mnt/data/legibility-kit/Polylogue-Sinex-external-legibility.pptx)

It contains 13 slides covering positioning, the joint architecture, demo doctrine, recommended public portfolio, launch cut, swarm design, fork missions, and prepared patches.

Static product targets:

- [Polylogue landing-page prototype](sandbox:/mnt/data/legibility-kit/mockups/polylogue-home.png)
- [Sinex landing-page prototype](sandbox:/mnt/data/legibility-kit/mockups/sinex-home.png)
- [Joint Resume This Bead / Agent Work Packet prototype](sandbox:/mnt/data/legibility-kit/mockups/resume-this-bead.png)

The source HTML for all three mockups is included in the package. A rendered preview of the patched Polylogue site is under `previews/polylogue-site/`.

# Public-claims discipline

The joint claims inventory is in [09-public-claims-ledger.yaml](sandbox:/mnt/data/legibility-kit/09-public-claims-ledger.yaml).

Every claim has:

- project;
- status;
- evidence class;
- precise internal wording;
- bounded publication wording;
- evidence locations;
- scope;
- caveats;
- owning Beads.

This separates:

- currently implemented capabilities;
- deterministic-fixture proofs;
- private field observations;
- architecture decisions;
- plans;
- hypotheses requiring experiments.

It prevents a common launch failure in sophisticated systems: a valid but narrow field result being turned into a universal capability claim.

# Validation status

The full ledger is [15-validation-report.md](sandbox:/mnt/data/legibility-kit/15-validation-report.md).

Verified:

```text
Polylogue Ruff lint                         passed
Polylogue format check                     passed
Polylogue strict MyPy                      passed
Polylogue focused tests                    41 passed
Polylogue documentation rendering          passed
Polylogue command verification             75 documents, no stale commands
Polylogue rendered-site local links        passed
Polylogue deterministic tour               passed
Declared tour constructs                   30/30
Patch application at exact bases           both passed
Package-relative Markdown links            67 passed
Machine-readable YAML/JSON/CSV             passed
Demo Packet example/schema validation      passed
Public artifact path/credential scrub      passed
Deck                                        13 slides / 13 PDF pages
PDF preflight                              passed
ZIP integrity                              passed
```

The Sinex runtime was not compiled or deployed because Rust, Cargo, Nix, PostgreSQL, and NATS were unavailable in this execution environment. Its patch received static documentation, claims, link, planning-policy, and diff validation.

The work does not claim implementation of the complete semantic renderer, Incident 14:32 corpus, Sinex-backed Polylogue data plane, full rebuild, selective physical deletion, or general agent-performance uplift. Those are represented as bounded execution and experiment programs rather than as completed capabilities.

---

