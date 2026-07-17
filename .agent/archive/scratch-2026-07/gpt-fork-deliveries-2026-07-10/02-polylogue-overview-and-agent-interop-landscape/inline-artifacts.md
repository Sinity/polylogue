## Turn 79 (role=assistant) — earlier full project-overview delivery answering the initial 'explain the project at length' ask (different wording/structure than the 62,564-char report used elsewhere; this one begins '## The project, in one sentence')

## The project, in one sentence

Polylogue is a local, evidence-preserving system of record for AI work: it ingests sessions from ChatGPT, Claude, Claude Code, Codex, Gemini, Antigravity, Hermes, browser captures, and coding-agent runtimes; normalizes them into a durable archive; then lets humans and agents search, analyze, audit, resume, and cite what actually happened.

The important distinction is that Polylogue is not trying to be “another chat viewer.” It is closer to a flight recorder, ledger, debugger, memory substrate, and forensic workbench for LLM-assisted work. Its unit of value is not the transcript as text. Its unit of value is evidence: source bytes, parsed session topology, messages, blocks, tool calls, tool outcomes, files, costs, claims, user judgments, context snapshots, continuation packs, and reproducible proof artifacts.

## Why it exists

The motivating problem is that AI work has become real work, but most of its evidence is scattered across vendor silos. A coding agent edits files, runs tests, fails commands, retries, summarizes, compacts, forks, resumes, hands work to subagents, spends tokens, and leaves behind claims like “done” or “fixed.” But the evidence for those claims often lives in provider-specific logs, hidden local state, web-app history, tool output, chat exports, or ephemeral cloud sandboxes.

Polylogue’s answer is: archive the work itself, locally, in a shape that can be queried and audited later.

That makes it valuable in several different modes. For a developer, it is a way to find “that session where the agent debugged the SQLite locking issue.” For a project maintainer, it is a way to ask which tools are failing, which models are burning tokens, which tasks were abandoned, and which claims lack evidence. For future agents, it is a way to retrieve grounded continuation context instead of relying on stale summaries or model-side memory. For research, it is a way to turn messy agent traces into a corpus of structured behavior.

The README’s framing is accurate: Polylogue is “the system of record for AI work.” That phrase is doing real architectural work. The project is trying to preserve the provenance of AI-assisted labor in the same way git preserves code changes, except the thing being preserved is the surrounding work: prompts, tool calls, file edits, failed tests, costs, context packs, summaries, decisions, and resume points.

## The core product model

The product is organized around four verbs: search, analyze, audit, and remember.

Search means a query-first interface over the archive. The CLI shape is deliberately “find query then action”: `polylogue find "pytest" then read`, `polylogue find 'repo:polylogue since:7d' then analyze --facets`, `polylogue find id:... then continue`. The same query semantics are intended to flow through the CLI, Python API, MCP server, and daemon/web surfaces.

Analyze means aggregating work evidence: origins, models, token lanes, cost rollups, actions, tool families, failure follow-up, work phases, session topology, usage timelines, and eventually richer outcome-grounded analytics.

Audit means refusing to treat assistant prose as ground truth. A command that failed is a failed command because a structured tool result carried `is_error` or a non-zero exit code, not because the assistant later said “there was an error.” A cost is priced under a named basis, not silently merged with another cost basis. A search index is ready only if its freshness invariant holds. An unavailable field is reported as unavailable, not guessed.

Remember means turning reviewed evidence into context that can be handed to later agents. This is the most strategically important but also the most honestly unfinished part: the project has context bundles and continuation reports, but the Beads state explicitly treats “memory improves future agents” as a promise until measured uplift experiments support it.

## Conceptual vocabulary

Polylogue is careful about vocabulary because the wrong word causes downstream correctness bugs.

“Origin” is the public source identity: `claude-code-session`, `chatgpt-export`, `codex-session`, `aistudio-drive`, `gemini-cli-session`, `hermes-session`, `antigravity-session`, and so on. “Provider” is reserved for the raw provider-wire boundary, such as a parser family or schema package. This distinction matters because Claude Code, Claude.ai export, Anthropic-the-lab, and a filesystem source root are related but not interchangeable.

A “session” is the top-level archived unit. A session has messages, metadata, working directories, repository hints, topology links, events, attachments, and derived profiles.

A “message” is not just a role plus text. Messages carry role, message type, material origin, content blocks, attachments, model and token evidence, and structural flags. The project explicitly separates provider role from authoredness. For example, a provider may encode a runtime context artifact as a “user” message, but Polylogue should not count it as human-authored prose.

A “block” is a first-class row inside a message. Text, thinking, tool use, tool result, image, code, and document content live as structured blocks. This is essential because agents do not just chat; they act. Tool calls and tool results need their own evidence grain.

An “action” is a normalized view over tool-use/tool-result evidence. The project treats actions as queryable units: shell commands, file edits, reads, errors, paths touched, command text, outputs, and sequences such as “file edit → shell test → failure → file edit.”

An “assertion” is durable user-authored or reviewed state. Tags, notes, corrections, candidate claims, judgments, saved views, blackboard notes, recall packs, and metadata overlays live in the user tier. This matters because user judgment is irreplaceable, whereas most indexes and projections can be rebuilt.

A “context pack” or continuation report is a generated object for a successor agent. The project’s design discipline is that context should be evidence-backed, scoped, and lossiness-aware. The handoff-pack experiment found that a stale pack can be worse than a raw reference, which is exactly the kind of negative result this project is designed to surface.

## Architecture

The architecture is arranged in four rings.

The first ring is the archive substrate. It owns acquisition, provider detection, parsing, normalization, content hashing, storage, and query. Provider detection is shape-based, not filename-based. Raw source material is normalized into a canonical session/message/block model, with content hashes over normalized payloads so re-ingesting the same source is idempotent.

The second ring is derived read models. These are materialized insights over the archive: session profiles, work events, phases, threads, topology, usage rollups, cost rollups, search indexes, and other query acceleration structures. They are useful, but they are not the durable truth.

The third ring is surfaces. The CLI, MCP server, Python API, daemon HTTP reader, dashboard/TUI, web shell, renderers, and browser workbench are leaf adapters over the same substrate. A surface should not invent its own semantics. New semantics are supposed to land in substrate or insights first, then surface adapters expose them.

The fourth ring is verification and maintenance. This includes schemas, deterministic demo fixtures, devtools, smoke tests, visual tests, generated docs, topology/layering checks, benchmark lanes, fuzz/property tests, and proof artifacts. The project treats generated surfaces as testable contracts rather than documentation afterthoughts.

The storage architecture is one of the project’s strongest ideas. Polylogue does not put everything in one SQLite file. It splits the archive by durability class:

`source.db` stores raw acquisition evidence and source artifacts. This is durable and used to rebuild downstream state.

`index.db` stores parsed sessions, messages, blocks, FTS indexes, topology, actions, attachments, and materialized read models. This is rebuildable.

`embeddings.db` stores opt-in vector rows and embedding catch-up state. This is rebuildable but expensive, because re-embedding may call an external embedding provider.

`user.db` stores irreplaceable human-authored assertions and overlays. This is the tier to protect.

`ops.db` stores daemon telemetry, convergence debt, ingest attempts, health, and local operational state. This is disposable.

Large binary content lives in a SHA-256 content-addressed blob store with prefix sharding. That supports deduplication, raw evidence preservation, attachment acquisition, and eventual “forget on purpose”/excision semantics.

## Query and retrieval

The query language is much more ambitious than a keyword search box. It supports compact session queries, explicit Boolean session predicates, terminal unit queries over messages/actions/blocks/files/assertions/runs/observed-events/context-snapshots, and pipeline-like operations such as sorting, limiting, grouping, and counting.

The project has lexical FTS as a hard invariant, not a cache that may or may not be fresh. Semantic search exists as an optional lane. Hybrid retrieval uses reciprocal-rank style fusion between lexical and vector lanes. Action retrieval lets the operator search the work, not just the dialogue: commands, tools, paths, outputs, and action sequences.

This is where Polylogue starts to look less like a transcript archive and more like a domain-specific database for agentic work. You can ask not only “where did someone mention SQLite?” but “which sessions in this repo had a file edit followed by a failing shell command?” or “which assistant messages followed structured tool failures silently?”

## Surfaces

The CLI is the primary product surface. Its shape is query-first: `find QUERY then read|select|analyze|mark|delete|continue`. This matters because it makes search a reusable selection primitive rather than a standalone command. A query result can be read, analyzed, marked, deleted with guards, or turned into continuation context.

The MCP server is the agent-facing bridge. It currently exposes search/list/session/stats resources and agent-friendly tools such as `get_postmortem_bundle` and `get_pathologies`. The project’s MCP design is aligned with the broader MCP ecosystem: MCP defines hosts, clients, and servers, and lets servers expose resources, prompts, and tools to AI applications via a standardized protocol. Polylogue’s role in that ecosystem is to be a local evidence and memory server for agents, not just a tool that returns chat text. citeturn232026view0

The Python API is the library surface for integrations and scripts. The daemon handles live ingestion, convergence stages, status, HTTP reader behavior, metrics, and browser-capture receiver paths. The web/dashboard surfaces are present but still less strategically mature than the CLI/archive substrate; Beads has an explicit web-workbench program to turn the reader into an “evidence cockpit.”

The browser extension is an important acquisition path. For ChatGPT and Claude.ai, it prefers provider-native page/app payloads when available and falls back to DOM extraction only as compatibility evidence. Captures are written to a local spool and then ingested like source artifacts. That is a strong design choice: browser capture should coalesce with normal provider imports rather than creating duplicate sessions just because evidence arrived by extension.

## Present state of the snapshot

The uploaded snapshot was generated on 2026-07-08 from `master` at commit `60ec563a`, with the snapshot marked dirty but with no branch delta against `origin/master`. In practical terms: the branch itself appears aligned with master, while the capture process saw some local dirty state.

The project is substantial. The snapshot reports 7 configured source slices, 11 XML snapshots, 35 artifacts, 7,824 all-ref git commits, and a large code/test surface. The tokei attribution shows roughly 111k lines in core-and-storage, 40k in CLI/MCP/operations, 30k in daemon, 22k in pipeline/product/readiness, 20k in rendering/site, 44k in devtools/packaging/Nix, and about 268k lines in tests and QA. There are 802 files under `tests/` in the extracted working tree, with unit, property, integration, benchmark, fuzz, visual, storage, CLI, daemon, MCP, browser-capture, and security lanes.

The public package version is still `0.1.0`, so the codebase is far more mature internally than the semver suggests. It looks like a pre-1.0 project with serious architecture and verification, not a small experimental script.

Provider support is broad. Accepted/importer rows exist in code for Claude Code JSONL, Codex session JSONL, ChatGPT Takeout JSON, Claude.ai export JSON, AI Studio/Gemini Drive-like exports, Antigravity language-server export, Gemini CLI local-agent documents, and Hermes state DB. Browser capture is represented as a proposed live receiver package that maps captured sessions onto provider-specific origins.

The deterministic demo corpus is a major present-state asset. It has 11 sessions, 43 indexed messages, 87 blocks, 11 session profiles, 12 run rows, 37 observed-event rows, 12 context-snapshot rows, five origins, acquired attachments, browser-capture coalescing, lineage links, subagent runs, terminal-state examples, synthetic embeddings, and user overlays. That lets someone run meaningful examples without exposing private archives.

The project also has proof artifacts rather than only feature claims. The claim-vs-evidence artifact inspects structured tool failures and classifies whether the next assistant turn acknowledges or silently proceeds. The cost-accounting example proves a real bug class: Codex token fields overlap, so cached input must be decomposed into disjoint lanes or costs inflate badly. The agent-forensics artifacts summarize usage, costs, token lanes, and model evolution. The handoff-pack two-arm experiment is especially healthy from an epistemic standpoint: it produced a negative/diagnostic result, not a marketing claim.

One local caveat I found: `docs/schema.md` contains a stale tier-version table relative to the current code. The code reports `SOURCE_SCHEMA_VERSION = 2`, `INDEX_SCHEMA_VERSION = 24`, `USER_SCHEMA_VERSION = 4`, while `docs/schema.md` still lists older values. That fits the project’s own Beads direction: doc-vs-code drift and column-honesty audits are active trust-floor concerns.

## Beads state and planning model

You asked for plans via Beads, not GitHub. The Beads export is the right source here.

The Beads snapshot contains 529 total issues, 408 open, 2 in progress, 119 closed, 831 dependency edges, and 34 memories. The ready/blocked signals are slightly split across counters: the snapshot summary reports 297 ready issues, while the exported ready-ID list has 100 IDs; blocked is around 111/113 depending on which counter is used. The important operational reading is that the backlog is large, dependency-rich, and actively sequenced.

The Beads doctrine is also clear: Beads are the durable backlog and directive substrate. GitHub issues are external references, not the plan. Current design direction lives in Beads issue fields, especially `description`, `design`, `acceptance_criteria`, and sometimes `notes`. Planning documents are intentionally mined into Beads and then purged or demoted.

The current sequencing is encoded through delivery labels. The frontier is `delivery:A-trust-floor`, followed by `B-storage-rebuild-bytes`, `C-read-evidence-contract`, `D-agent-context-coordination`, then later gates for variants/preferences, lineage/compaction, live performance, web cockpit, analytics experiments, embeddings/retrieval, interop/origin export, external legibility, substrate consolidation, and horizon work.

There is an in-progress Bead, `polylogue-8e1b`, whose purpose is to reconcile the numeric priority field with the delivery-gate order. That is a useful warning: sorting by raw priority alone can mislead. The delivery gate labels currently tell the truer sequencing story.

The other in-progress Bead is `polylogue-83u.6`, an attachment acquisition census by origin and byte volume. That sits in the storage/blob evidence lane and is meant to quantify acquired, unfetched, and unavailable attachment bytes on the live archive.

The closed P0 campaign history is important. Three proof campaigns have closed:

`polylogue-sru`, the claim-vs-evidence report campaign, closed with a bounded sample-frame report, marker calibration, reproduction package, cold-reader gate, and productized follow-up classifications.

`polylogue-tf2`, the agent-forensics regeneration and all-provider repricing campaign, closed with a usage headline that separates physical-session tokens, logical high-water tokens, stored/provider-priced USD, catalog API-equivalent USD, and provenance lanes.

`polylogue-jxe`, the handoff-pack uplift experiment, closed honestly as a diagnostic/negative result: raw reference scored 8/10, handoff pack scored 5/10 because the pack was stale. Follow-up product work now targets freshness and successor links.

That closed campaign set explains the present strategy. The project has already proven that it can produce serious artifacts; now the plan is to harden the trust floor so those artifacts are safe to quote.

## Current Beads frontier: A-trust-floor

The A-trust-floor lane is the immediate center of gravity. It contains 49 open items, most of the P1 work, and many of the most important correctness tasks.

The biggest A-trust-floor epic is `polylogue-f2qv`: provider usage and cost honesty. Its point is that cost accounting is not a reporting nicety; it is a correctness surface. The epic owns disjoint token lanes, one pricing source of truth, dual cost views, stale-row hygiene, and regression guards for previous double-count classes. This is high-value because cost numbers are exactly the kind of numbers people quote publicly, and wrong ones destroy credibility.

Another core trust-floor epic is `polylogue-kwsb`: security and privacy. Its purpose is to make sure the archive can forget on purpose and does not leak secrets. It owns excision, reset mutation safety, blob excision, right-to-forget semantics, localhost receiver security, MCP destructive-path contracts, and dry-run/confirmation behavior. For a local archive of all AI work, this is existential.

`polylogue-cpf` is the doctrine epic: time semantics, writer classes, finding provenance, degraded modes, non-goals, and injected-context trust. The project has learned enough from real failures that those lessons need to become written doctrine plus cheap lints. This is the sort of work that makes future agent contributions safer: it turns tribal rules into executable constraints.

`polylogue-9e5` is an audit lane. It is intentionally read-only: assertion adoption, column honesty, race audits, hash-boundary census, daemon loop/starvation maps, provider-to-origin completion, heuristic accuracy, resume/context efficacy, test-suite economics, schema-inference ROI, doc-vs-code drift, facade decomposition, dead-code sweeps, zero-use MCP/CLI surfaces, affordance usage, and similar evidence artifacts. Its role is to replace “I think this surface matters” with data.

There are also P1 trust-floor bugs/tasks like `polylogue-xnkf`, where duplicate tool IDs can cause one logical action to fan out into many rows; `polylogue-jnj.5`, routing reset operations through mutation contracts; `polylogue-f2qv.5`, version-gating provider-usage projections so stale projections self-heal; and `polylogue-cpf.4`, enforcing “degrade loudly” rather than silently soft-failing.

The pattern is coherent: before pushing bigger product stories, the project is tightening truthfulness, safety, pricing, evidence provenance, and failure visibility.

## Downstream Beads programs

After the trust floor, the storage and evidence lanes matter most.

`delivery:B-storage-rebuild-bytes` includes attachment/blob acquisition work, byte accounting, rebuild correctness, and archive-readiness repair. This is where the project makes “raw evidence preserved” more literal: not just metadata about attachments, but acquired bytes, blob refs, GC safety, rebuild behavior, and proof of what is missing.

`delivery:C-read-evidence-contract` is about making read surfaces contractual. The large epic `polylogue-t46` says contracts should own surfaces: query DSL, terminal units, refs, read-view profiles, action/route contracts, generated docs/schemas, and behavior should replace hand-written parallel dispatch. This is key for agent use because agents need stable contracts, not fragile CLI prose.

`delivery:D-agent-context-coordination` contains the context/memory and coordination substrate work. `polylogue-37t` is the judged-memory loop: agents declare structured claims, the operator judges them, active claims compile into preambles, and sessions can reboot into evidence packs. `polylogue-s7ae` is broader: evidence-backed multi-agent work without tracker lock-in. This is where Polylogue starts becoming not just a memory database but an operational substrate for concurrent agents.

`delivery:F-lineage-compaction` owns the hard truth that forks, resumes, compactions, shared prefixes, and logical sessions must be modeled correctly or every aggregate becomes suspect. `polylogue-4ts` says shared content must be stored once, counted once, and composed correctly. This is fundamental for agent traces because modern agent sessions often fork, compact, summarize, and resume.

`delivery:G-live-performance` addresses startup latency, cold CLI costs, slow help commands, query spikes, ingest catch-up speed, and daemon fast paths. A system of record is only useful if it is cheap enough to use while working.

`delivery:H-web-cockpit` evolves the web reader into an evidence cockpit. Current Beads describe the MK2 shell as solid but under-expressive: the DSL, aggregates, live capture, and long-session structure are hidden. The plan is to surface those at the point of use.

`delivery:I-analytics-experiments` is the “so what?” layer. It includes outcome-grounded analytics, measure registries, uncertainty, sample frames, temporal trends, process mining, experiments, predictive surfaces, and derived work-graph units such as delegation, episodes, artifact edges, entity mentions, world effects, verification runs, projects, topic clusters, and cross-origin threads.

`delivery:J-embeddings-retrieval` aims to move beyond the current hardcoded/limited embedding story. The present system supports optional semantic search, but the Beads plan is more provider-general, lifecycle-honest, and cost-justified.

`delivery:K-interop-origin-export` includes broader origin support and two-way citable export. Hermes bridge work is important here: Hermes acts, Polylogue remembers/explains, and external agent/runtime evidence should flow into canonical archive units and out as citable artifacts.

`delivery:L-external-legibility` is about making the project understandable and runnable by a stranger. The project already has deep value; the Beads diagnosis is that category anchoring and explanation lag behind capability.

`delivery:M-substrate-consolidation` is internal debt: kill sync/async storage twins, split god modules, eliminate duplicate schema declarations, and reduce correctness hazards caused by duplicated logic.

## Fit into the broader AI-agent landscape

The broader agent ecosystem is converging on a few layers: agent orchestration, tool/context protocols, agent-to-agent communication, memory/persistence, observability/evaluation, and governance/safety. Polylogue sits mostly underneath and beside those layers.

MCP solves agent-to-tool and agent-to-context integration. The official MCP spec describes a host/client/server model where servers expose resources, prompts, and tools, with explicit attention to user consent, data privacy, and control. Polylogue’s MCP server fits naturally here: it can expose archived sessions, stats, postmortem bundles, and pathologies to an agent as local tools/resources. But Polylogue’s distinctive value is not that it “has MCP”; it is that the MCP surface is backed by a cross-vendor evidence archive. citeturn232026view0

A2A solves a different problem: agent-to-agent communication and delegation across vendors/frameworks. The official A2A docs describe it as an open protocol for communication and collaboration between opaque agentic applications; Google’s launch post framed it around agents securely exchanging information, coordinating actions, advertising capabilities via Agent Cards, and managing tasks/artifacts. Polylogue is not an A2A replacement. It is the evidence layer that can make A2A-style collaboration auditable after the fact: which agent delegated what, which artifact came back, what failed, what was accepted, and what should be cited. citeturn232026view3 citeturn172674view0

Agent SDKs and orchestration frameworks run the work. OpenAI’s Agents SDK, for example, provides primitives for agents, handoffs, guardrails, tool loops, sessions, tracing, and approval flows. LangGraph positions itself around durable execution, persistence, streaming, human-in-the-loop, and memory. Polylogue’s role is different: it is not primarily the runtime loop. It is the archive and forensic substrate that can ingest work from many loops, including ones Polylogue did not control. citeturn232026view1 citeturn172674view1 citeturn172674view2

Memory systems inside agent frameworks are usually scoped to an application or graph. LangGraph’s persistence docs distinguish thread-scoped checkpointers for short-term memory from stores for long-term cross-thread memory, useful for resuming, recovering from failure, and remembering facts across interactions. Polylogue’s memory angle is broader and more archival: it wants to preserve the actual multi-provider work history, then compile reviewed slices of that history into continuation context. That makes it less like an in-app memory store and more like a local institutional memory for a developer or project. citeturn232026view2

Human-in-the-loop systems pause or gate actions before execution. LangChain’s HITL middleware, for instance, can interrupt tool calls such as SQL writes or file deletion and wait for approval/edit/reject/respond decisions, saving graph state so execution can resume. Polylogue complements that by preserving what decisions were requested, what actions happened, what the tool returned, and whether later claims matched the evidence. citeturn172674view3

Observability platforms such as LangSmith and Phoenix focus on traces, evaluations, production monitoring, datasets, experiments, and prompt iteration. OpenTelemetry provides vendor-neutral APIs and SDKs for traces, metrics, and logs. Polylogue overlaps with observability, but its wedge is different: it is local-first, provider-export-friendly, and evidence-preserving even when the original agent runtime was not instrumented for your observability backend. That is strategically important because a lot of personal and coding-agent work happens in vendor tools, CLIs, browser sessions, and local state, not in a single instrumented production app. citeturn232026view4 citeturn232026view5 citeturn232026view6

## Why Polylogue is especially relevant to agents and LLMs

Agents create more evidence than humans can manually track. A serious coding session can contain hundreds or thousands of messages, file reads, edits, shell commands, tool failures, summaries, branches, compactions, and context injections. Without a system like Polylogue, most of that evidence is either invisible, non-portable, or too expensive to inspect.

Agents also fail in ways that ordinary logs do not capture well. They can proceed after a tool failure, summarize stale state, re-run the same broken command, lose context after compaction, inflate cost estimates by double-counting provider token lanes, confuse provider role with authoredness, or claim a fix without running the relevant test. Polylogue’s design is unusually focused on these failure modes.

For future agents, Polylogue becomes a memory substrate with guardrails. A successor agent can ask for context, but the context can be built from cited sessions, actions, files, observed events, and judged assertions. That is better than dumping raw transcripts, and safer than letting an LLM summarize arbitrary history into unreviewed “memory.”

For multi-agent work, Polylogue can become the blackboard and audit layer. Beads already point toward coordination objects: work items, agent activity, resource episodes, handoffs, messages, context injection, and proof evidence. This matters because multi-agent systems do not only need communication; they need accountability. Who did what? From which context? With which evidence? Did another agent overwrite it? Which task tracker state was current? Which claim was accepted by a human?

For AI research and evaluation, Polylogue can turn lived agent work into datasets. The claim-vs-evidence campaign is a good example: it uses real structured failures, calibrates markers, reports caveats, and produces a reproducible demo method. That is a stronger evaluation posture than anecdotal “agents often ignore errors.”

For cost governance, Polylogue addresses a real blind spot. Agent systems resend context, use cache lanes, switch models, perform retries, and sometimes report overlapping token fields. The project’s cost doctrine recognizes that provider-reported usage, catalog API-equivalent cost, subscription-credit views, cache reads/writes, reasoning tokens, and stored billing truth are different axes. That separation is exactly what agent-heavy workflows need.

## What makes the project valuable

The main value is not any one surface. The value is the combination of local-first evidence, cross-provider normalization, queryable work units, durable user judgment, and proof-oriented development.

Local-first matters because AI work is sensitive. The archive can contain private code, business plans, credentials accidentally pasted into chats, unreleased product strategy, legal/medical/financial notes, and raw browser captures. Keeping the archive under local XDG data paths, with no upload by default, is a strong default.

Cross-provider support matters because no serious user lives in one model or one tool. A real AI workflow may include ChatGPT, Claude.ai, Claude Code, Codex, Gemini, browser chats, local CLIs, and future agent runtimes. Vendor-native memory cannot solve that.

Evidence preservation matters because summaries lie, models confabulate, and “done” is cheap. Polylogue keeps raw source bytes and derived claims side by side, so derived artifacts can be audited or rebuilt.

The split-tier storage model matters because it encodes data ethics into architecture. Raw source and user assertions are durable. Indexes, insights, embeddings, and daemon telemetry are rebuildable. That is the right separation for a personal archive.

The Beads-first planning model matters because it is itself an agent-coordination experiment. The backlog is not a loose pile of tickets; it is a typed, dependency-rich directive graph with design, acceptance criteria, memories, delivery gates, and execution packets. That fits Polylogue’s subject matter: building software with many agents requires a better substrate than a flat issue list.

## Main risks and gaps

The biggest product risk is legibility. The project is deep, but a stranger can still struggle to understand the category. “Chat export viewer” is too small. “Agent observability” is partly right but too cloud/runtime-oriented. “Second brain” is too vague. “System of record for AI work” is the best current phrase, and the external-legibility Beads lane exists because the value needs packaging.

The second risk is overbreadth. The project touches ingest, storage, query languages, web UI, MCP, browser extensions, cost accounting, embeddings, analytics, agent memory, coordination, demos, docs, devtools, and release engineering. The Beads delivery-gate structure is the right antidote, but the backlog is large enough that prioritization drift is a real risk. The in-progress priority-field reconciliation Bead exists for a reason.

The third risk is stale derived state. The architecture says derived tiers are rebuildable, but large archives make rebuilds expensive. The project has already hit readiness, FTS, session-insight, usage-rollup, and topology performance problems. The “automagic invariants” memory is correct: conditions Polylogue can maintain automatically should not become routine operator chores.

The fourth risk is memory claims. The project should keep saying “context packs are evidence-backed and promising” rather than “agents get better” until the freshness-fixed uplift experiment proves it. The negative handoff-pack pilot is a strength precisely because it prevents premature marketing.

The fifth risk is documentation drift. I found at least one concrete stale schema-version table. The Beads backlog already contains doc-vs-code and column-honesty audits, so this is known class debt, not a surprise.

The sixth risk is security and excision. A local archive of all AI work needs serious delete, redaction, blob excision, receiver auth, localhost threat modeling, MCP role boundaries, and dry-run contracts. The Beads plan treats this as A-trust-floor work, which is the correct priority.

## Strategic positioning

The best way to position Polylogue is:

Polylogue is the local evidence ledger and memory substrate for AI-assisted work.

It is upstream of analytics because it preserves the evidence. It is adjacent to observability because it can ingest traces and infer work units, but it is not limited to instrumented apps. It is adjacent to agent frameworks because it can feed and audit agents, but it does not need to own the agent loop. It is adjacent to MCP/A2A because it gives agents and other systems a trustworthy archive to consult, but it is not merely a protocol server. It is adjacent to personal knowledge management because it remembers, but its memory is structured around evidence, refs, assertions, and reproducible work, not freeform notes.

The deepest value is that Polylogue treats AI work as something that deserves provenance. That is the right instinct for the next phase of agents. As agents do more work, the limiting factor will not only be model capability. It will be whether we can remember what they did, verify what they claimed, resume safely, coordinate multiple workers, and learn from failures without trusting plausible prose. Polylogue is aimed exactly at that gap.

---

## Turn 102 (role=assistant) — rename brainstorm delivery answering 'brainstorm potential project renames' (name-collision analysis vs polylogue.page product + candidate names)

My take: **Polylogue is a beautiful name, but I would rename before the project becomes more public.** It is semantically right, but commercially and search-wise weaker than it looked at first.

“Polylogue” already has a real dictionary/academic meaning around many-party speech or multiparty conversation, which is why it feels so natural for a cross-provider, multi-agent archive. YourDictionary defines it as “a speech delivered by several persons,” and Cambridge’s argumentation book uses it as an alternative framing for “multiparty conversation.” citeturn625736view5 citeturn690627view0

That semantic fit is also the problem. The name points at **conversation**, while the project’s strongest value is **evidence**: provenance, traces, tool calls, failures, costs, context handoffs, source bytes, and auditability. “Polylogue” sounds like a multi-agent chat/document product. Your project is closer to an AI-work flight recorder.

The practical collision is nontrivial. There is an active **Polylogue** at polylogue.page that describes itself as a collaborative writing platform where AI agents join a workspace, read documents, comment, edit, and can be @mentioned. That is adjacent enough to create confusion in AI-agent/productivity/search contexts. citeturn625736view0 There are also other public uses: a PolyLogue AI iOS app for an “AI Friend Group,” a Godot dialogue manager, a Pure Data synthesizer, and an arXiv/ML project using “polylogue” as a term for latent persona dynamics in LLM reasoning. citeturn690627view3 citeturn625736view2 citeturn690627view4 citeturn625736view3

So the issue is not “someone somewhere used the word.” That alone would be fine. The real issue is that the strongest existing collision is in **AI agents + documents/workspaces**, which is close enough to your future surface area: browser capture, MCP, context packs, agent memory, evidence cockpit, and cross-tool archives.

I would rate the current-name risk like this: **low** for private/local use, **medium** for open-source package identity, **high** for SEO and user comprehension, and **medium-to-high** if you ever want a public product/site/cloud/service around it. I did not do legal clearance; a proper pass would use USPTO/WIPO and expanded mark searches, not just exact web search. The USPTO explicitly recommends starting with exact wording but then expanding the search beyond exact matches. citeturn718076search11

The best move is probably: **rename the public project, keep “polylogue” as an internal concept or historical codename.** For example: “Polylogue” can remain the name of the normalized many-party conversation graph, but the product should be named after what it actually guarantees.

The naming target should be something like:

**A local evidence ledger for AI work.**

Or:

**The flight recorder for agents.**

Or:

**Searchable provenance for LLM-assisted work.**

Those phrases are much closer to the project’s durable value than “multi-party conversation.”

My strongest rename directions are these.

**TraceBraid**

This is my favorite poetic-but-accurate option. “Trace” says runtime evidence, tool calls, actions, logs, provenance. “Braid” says many sources woven together: ChatGPT, Claude, Codex, Gemini, browser captures, coding agents, files, costs, handoffs. It also fits lineage, forks, compaction, shared prefixes, and multi-agent continuation.

It has a better center of gravity than Polylogue because it does not suggest “another chat app.” It suggests many execution threads woven into one inspectable record.

Possible tagline: **TraceBraid — local evidence for AI work.**

CLI could be `tracebraid`, with short aliases like `tb` or preserving `plog` temporarily.

**WorkBraid**

Softer and broader than TraceBraid. It says the object being preserved is not “chat” but “work.” That is strategically correct. Agents are not just conversational partners; they are workers producing attempts, claims, commands, files, failures, and artifacts.

Possible tagline: **WorkBraid — the local record of AI-assisted work.**

The downside is that “work” is generic and can sound like productivity tracking. It is less technical than TraceBraid.

**SourceBraid**

This emphasizes raw source preservation and provenance. It is good if you want the project to feel archival and correctness-oriented. It also hints at code-source plus evidence-source.

Possible tagline: **SourceBraid — preserve and query the sources behind AI work.**

The downside is that it may sound like source-code tooling only.

**EvidenceBraid**

Very accurate. It says exactly what the system does: weave evidence together. It fits the project’s claim-vs-evidence, cost honesty, action audit, attachment bytes, and right-to-forget work.

Possible tagline: **EvidenceBraid — an evidence graph for agentic work.**

The downside is that it sounds legal/forensic, maybe heavier than the project needs.

**Tracearium**

Distinctive, slightly scholarly, and archive-flavored. “An -arium for traces.” It has some of Polylogue’s intellectual feel but points toward traces instead of conversation.

Possible tagline: **Tracearium — an archive of AI work traces.**

The downside is spelling/pronunciation. It is less instantly clear than TraceBraid.

**Runfolio**

A folio of agent runs. Short, pleasant, and product-ish. It fits sessions, artifacts, reports, costs, and postmortems.

Possible tagline: **Runfolio — collect, search, and audit AI runs.**

The downside is that “run” may over-index on execution traces and underplay human chat exports.

**Attestory**

Attest + story. I like the concept: the project records the story of AI work, but with attestable evidence. It also has a nice implication that assistant claims need support.

Possible tagline: **Attestory — evidence-backed history for AI work.**

The downside is that it may be too cute, and there are document-attestation businesses using similar forms. citeturn578890search1

**Claimtrail**

Good if you want to emphasize one of the project’s sharpest insights: assistant claims are not truth; they must be checked against tool results and source evidence.

Possible tagline: **Claimtrail — follow AI claims back to evidence.**

The downside is that it names a subproblem, not the whole system.

**Factrail**

This is conceptually good but practically weaker because Factrail is already used for an AI facts/causal-impact product. citeturn213560search2

**Prooftrail**

Strong conceptually: proof artifacts, reproducible postmortems, evidence trails. It is generic enough to need clearance, but the positioning is right.

Possible tagline: **Prooftrail — reproducible records for agentic work.**

**Agent Ledger**

Extremely clear as a descriptor, but I would not use it as the main brand. “Agent Ledger” already appears in real estate/financial products and agent-infrastructure discussions/projects. citeturn329748search7 citeturn329748search15 Use it as a tagline/category phrase instead: “an agent ledger,” not necessarily “Agent Ledger.”

**AI Work Ledger**

This is probably the best plain-English category label. It is not a poetic brand, but it explains the product instantly.

Possible full positioning: **TraceBraid, an AI work ledger.**

That combination is stronger than either alone.

I would avoid these names even though they are tempting:

**Tracewright** is excellent semantically, but spot-checking shows existing software/PyPI usage around traces and self-healing/replay-driven evals. citeturn329748search0 citeturn329748search4

**LedgerLoom** sounds perfect, but it is already on PyPI and used by accounting/practice-management products. citeturn329748search1 citeturn329748search5

**Worktrace** is too collision-heavy, including automatic work-log/workflow-observation products in adjacent productivity/AI territory. citeturn329748search2 citeturn329748search6

**TraceLoom** is taken by a testing/tracing product. citeturn408165search9

**Agentarium** is already an AI agent orchestration framework. citeturn217827search0

**Evidentia** is already evidence-management genealogy software. citeturn408165search0

**ProofRail** is already used for an agentic control/audit service, which is quite close conceptually. citeturn213560search1

**Chorus, Agora, Conclave, Council, Symposium, Forum** are all attractive because they preserve the many-voices idea, but they repeat Polylogue’s central weakness: they foreground conversation, not evidence. They also have very high generic collision risk.

**ChatVault, PromptVault, Conversation Archive, LLM Archive** are too small. Your project is already beyond chat transcripts. It archives work: actions, tool results, costs, files, topology, context packs, and claims.

**AgentOps, AgentTrace, AgentMemory** are descriptive but crowded and trend-dependent. “Agent” is useful in the tagline, but maybe not ideal as the brand root unless you want to bind the project tightly to the agent wave.

My actual shortlist would be:

1. **TraceBraid** — best balance of distinctiveness, meaning, and future scope.
2. **WorkBraid** — friendlier, broader, less technical.
3. **EvidenceBraid** — most accurate, more forensic.
4. **Tracearium** — most like Polylogue aesthetically, but better aligned.
5. **Runfolio** — product-ish and compact, but slightly narrower.
6. **Claimtrail** — sharp for trust/audit positioning, but maybe too narrow.
7. **Prooftrail** — strong if you want the project to feel like reproducible audit infrastructure.
8. **SourceBraid** — best if raw evidence/source preservation is the headline.

If I had to choose one: **TraceBraid**.

The full public line would be:

**TraceBraid is a local evidence ledger for AI work. It preserves conversations, tool calls, files, failures, costs, context, and handoffs across agents and LLM providers.**

That says what Polylogue currently implies only indirectly. It also fits the Beads direction: trust floor, storage/rebuild bytes, read evidence contracts, agent context coordination, lineage, and analytics. “TraceBraid” can grow into those plans without sounding like a chat viewer or collaborative writing app.

A clean migration path would be: rename the project to **TraceBraid**, keep `polylogue` as a deprecated package/CLI alias for a few releases, keep `plog` if you like it, and use “polylogue” internally for the normalized multiparty conversation object. That preserves the nice idea without letting the public brand collide with adjacent AI-agent products.

---

## Turn 146 (role=assistant) — agent-interop positioning delivery answering the A2A/MCP quoted-context ask, before the user asked to broaden the research (turn 147) leading to the final delivery

Yes. The nuance should be corrected: Polylogue is not merely “after-the-fact evidence” for A2A-style work. It already has a real coordination direction, and the snapshot contains a working first slice of it.

The current Polylogue coordination plan is best understood as **local, evidence-backed coordination**, not yet **standardized agent-to-agent interoperability**. A2A is the latter.

Polylogue should not try to replace A2A. It should probably **adopt A2A at the external boundary**, while keeping Polylogue’s own coordination substrate underneath as the local evidence, memory, safety, and audit layer.

## What Polylogue already has

The Beads plan definitely includes coordination and communication.

The key epic is `polylogue-s7ae`: **“Agent coordination substrate: evidence-backed multi-agent work without tracker lock-in.”** Its description says Polylogue should make concurrent agent work operational, not merely visible. The planned substrate covers sessions, topology, repos/worktrees, work items, activity/resource episodes, context injection, messages, handoffs, and proof evidence. Beads is treated as an adapter, not as the only possible task tracker.

That is not an A2A clone. It is a local operating picture for agents working in and around the same project.

The code snapshot already contains a `polylogue/coordination` package. The typed payload is an `AgentCoordinationPayload` with repo state, current self identity, current work item, peer agents, resource episodes, overlap warnings, handoff refs, archive state, session trees, activity episodes, subagent exchanges, proof refs, context-flow refs, Beads hooks/gates, advisories, limits, and provenance. That payload is exposed through the CLI, MCP tool `agent_coordination`, MCP prompt `agent_coordination_brief`, daemon route `/api/agents/coordination`, and web coordination panel.

So: yes, Polylogue already has a coordination substrate. What it does not currently have, at least in the snapshot I inspected, is an explicit A2A adapter or A2A Beads plan.

## What A2A is, relative to that

A2A is a standard protocol for agents to discover each other, communicate, delegate tasks, exchange context/results, and coordinate across frameworks/vendors. The official docs describe it as an open standard for communication and collaboration between AI agents, especially when agents are built on different platforms or by different vendors. citeturn338987view0

Its core model is not “archive” or “memory.” Its core model is **interop**: Agent Cards, Messages, Parts, Tasks, Artifacts, streaming, push notifications, and task lifecycle. The official spec summarizes these as the main A2A concepts: Agent Card for identity/capabilities/auth, Message for a communication turn, Task as the stateful unit of work, Part as the content container, Artifact as an output, plus streaming and push updates. citeturn338987view2

That maps very naturally to Polylogue, but it does not replace it.

A2A says: “How does agent A ask agent B to do something, follow progress, and receive outputs?”

Polylogue says: “What actually happened, where is the evidence, what should future agents remember, what claims are grounded, what context was injected, what costs were incurred, what failed, and how do we resume safely?”

Those are different layers.

The A2A docs explicitly distinguish A2A from MCP: MCP is for tools/resources inside an agent; A2A is for communication and coordination between agents. citeturn524688view0 Polylogue currently uses MCP mostly to expose its local archive and coordination facts to agents. An A2A integration would expose Polylogue itself, or Polylogue-mediated agents, as participants in inter-agent task exchange.

## The key design distinction

Polylogue’s coordination substrate should remain **protocol-neutral internal truth**.

A2A should be an **adapter**, **source**, and **export surface**.

That distinction matters. A2A deliberately supports opaque agents: agents can collaborate based on declared capabilities and exchanged information without sharing internal memory, tools, or implementation details. citeturn312060view2 Polylogue’s strongest value is almost the opposite: when you control the local side, it preserves the internals that A2A does not require or expose: tool calls, failures, costs, context snapshots, subagent runs, pathologies, evidence refs, and user judgments.

So Polylogue should not remodel itself around A2A as the primary ontology. It should preserve richer local truth, then project to/from A2A where useful.

A good slogan would be:

**A2A is the wire protocol. Polylogue is the black box recorder and local coordination memory.**

## Where A2A fits directly

Polylogue could relate to A2A in three direct ways.

First, Polylogue can be an **A2A server**. It could publish an Agent Card for a “Polylogue Archivist” or “Polylogue Evidence Auditor” agent. That server would expose skills like search archive, produce context pack, audit claim against evidence, generate postmortem bundle, list active coordination state, retrieve handoff, and explain current work item. A2A Agent Cards are specifically meant to describe identity, endpoint, capabilities, authentication requirements, and skills, so this is a natural fit. citeturn140962view1

Second, Polylogue can be an **A2A client/proxy**. When a local Claude/Codex/Gemini/Hermes agent wants to delegate to another A2A-capable agent, Polylogue can send the A2A request or at least wrap/observe it, then archive the outgoing message, selected Agent Card hash, task ID, status updates, artifacts, and final result. A2A supports request/response, polling, SSE streaming, and push notification patterns for long-running tasks, so this would let Polylogue preserve both quick and long-running delegations. citeturn140962view0turn312060view1

Third, Polylogue can be an **A2A source importer**. Any A2A task stream, JSON-RPC request/response, task status update, artifact update, or Agent Card snapshot can become source evidence in Polylogue. That is probably the most important first slice. It makes A2A work auditable even before Polylogue becomes an active A2A participant.

## What should be adopted versus kept

Adopt A2A for **networked inter-agent task exchange**. Do not invent a bespoke Polylogue protocol for “agent A delegates task to agent B over HTTP.” A2A already owns that shape, and v1.0 is explicitly positioned as the stable production-ready standard for agent-to-agent communication. citeturn524688view0

Keep Polylogue’s own substrate for **local coordination, evidence, memory, safety, and provenance**. A2A does not know about Beads gates, dirty worktrees, local process tables, archive schema readiness, session topology, context-pack freshness, recursive assertion laundering, cost-accounting lanes, or whether an assistant silently proceeded after a failed tool call. Those are Polylogue-native concerns.

Keep Beads for **prospective project work**. An A2A `Task` is a runtime delegation between agents. A Bead is a durable project-planning object with dependencies, readiness, priority, delivery gates, and operator intent. They may reference each other, but they should not collapse into one object.

Keep MCP for **agent-to-Polylogue local tools/resources**. MCP remains a good way for an agent running in Claude Code, Codex, or another harness to ask Polylogue for local archive/context information. A2A is for peer-agent communication. The A2A docs themselves frame MCP and A2A as complementary rather than competing layers. citeturn312060view0

Drop, or at least avoid, any future plan that amounts to a custom networked “agent messaging protocol.” Polylogue can have local scoped messages, advisories, notices, and context-scheduler entries. But if the message is truly between independent agents or agent services, A2A should be the default external format.

## The mapping is almost too good

A2A `AgentCard` maps to a Polylogue-captured capability snapshot: agent name, provider, endpoint, skills, authentication scheme, version, signatures, and hash. Polylogue should store the card as source evidence, not just trust the current live endpoint.

A2A `Task` maps to a Polylogue run/delegation object. In Polylogue terms, it belongs near `session_runs`, `session_observed_events`, and subagent/delegation topology. The external task ID, context ID, agent endpoint, skill ID, tenant, lifecycle state, timestamps, and status history should all be preserved.

A2A `Message` maps to Polylogue messages/blocks. The sender/receiver role, message ID, context ID, task ID, parts, media types, metadata, and raw JSON should be archived.

A2A `Part` maps to Polylogue blocks or attachments. Text parts become text blocks; structured data parts become JSON blocks; file URL/raw parts become attachment/blob acquisition candidates.

A2A `Artifact` maps to Polylogue artifacts, attachments, context packs, reports, patches, proof bundles, or generated documents. Polylogue should content-hash artifact parts and record whether bytes were acquired, unfetchable, redacted, or remote-only.

A2A task status and artifact update events map to `session_observed_events`: `task.created`, `task.working`, `task.input_required`, `task.completed`, `task.failed`, `artifact.updated`, and so on.

A2A `contextId` maps loosely to Polylogue logical session/thread/root-session group, but not perfectly. Polylogue should preserve the A2A context ID as external provenance and then derive its own topology edges.

A2A extensions or Part metadata could carry `polylogue://` refs, source hashes, evidence refs, redaction policy, and grounding class. A2A already supports Part metadata and custom extensions through Agent Cards, so Polylogue-specific evidence refs can degrade gracefully when the other side ignores them. citeturn140962view0

## What I would add to Beads

I would not replace `polylogue-s7ae`. I would amend it.

The amended doctrine should say: **the coordination substrate is protocol-neutral; A2A is the preferred external inter-agent protocol; Polylogue-native messages are local notices/advisories/evidence annotations, not a competing remote protocol.**

Then I would add a new epic under `delivery:K-interop-origin-export`, with a dependency edge into `delivery:D-agent-context-coordination`:

**A2A bridge: Agent Cards + task streams -> canonical evidence -> optional Polylogue A2A server/client.**

Children would be:

`A2A boundary decision`: define exactly what Polylogue owns. Polylogue owns archival truth, context packs, evidence refs, and local coordination state. A2A owns external task/message/artifact exchange. Beads owns prospective task planning.

`A2A source importer`: ingest Agent Cards, SendMessage/SendStreamingMessage calls, Task objects, TaskStatusUpdateEvents, TaskArtifactUpdateEvents, Messages, Parts, and Artifacts into canonical source/index rows.

`A2A schema mapping`: declare how A2A fields map into sessions, runs, observed events, blocks, attachments, artifacts, topology, context snapshots, and user assertions.

`A2A Agent Card publisher`: expose Polylogue as a read-only local A2A server with skills such as archive search, context-pack generation, claim audit, postmortem bundle, and coordination status.

`A2A client capture proxy`: allow configured outbound calls to A2A agents while archiving every request, response, stream event, status transition, artifact, error, and retry.

`A2A security and privacy gate`: local-only by default, explicit allowlist for remote endpoints, no credential persistence, no raw archive excerpt without policy approval, artifact byte limits, redaction, Agent Card hash pinning, and clear provenance for remote/opaque evidence.

`A2A demo`: two agents, one delegated task, one returned artifact, one Polylogue context/audit bundle proving what happened.

This should not be A-trust-floor work. It belongs after the trust floor and basic evidence/read contracts are stable. Otherwise it risks expanding the boundary before the archive can safely expose itself.

## What I would change in existing coordination Beads

`polylogue-s7ae.3`, the “coordination messages and scheduler-mediated advisories” item, should split the word “messages” into two categories.

Local messages are Polylogue notices: scoped advisories, overlap warnings, handoff notices, context-scheduler entries, fatigue-controlled reminders, and Beads/worktree advisories. These stay Polylogue-native.

External messages are A2A Messages. If the recipient is an independent agent service, use A2A rather than inventing a Polylogue message bus.

`polylogue-37t.11`, the context scheduler, becomes even more important. A2A can bring in remote task status, artifacts, and messages, but that does not mean all of it should enter future agent context. The scheduler decides what gets injected, under what trust class, within what token budget, with what evidence refs.

`polylogue-37t.14` and `37t.16`, the recursive-safety and grounding-class work, should explicitly include A2A. A remote agent’s A2A message proves “remote agent said X.” It does not prove X. An A2A artifact may be stronger evidence, depending on type: a patch file, signed report, test log, or URL has different grounding power than prose.

`polylogue-2n6`, the harness remote-control lane, should not be forced into A2A. Claude Code and Codex local harness control is a separate problem. If a harness later exposes A2A, great. Until then, keep harness-specific resume/remote-control adapters.

`polylogue-fs1`, the Hermes bridge, should stay evidence-first. Hermes spans, subagent hooks, and A2A task streams are all agent-runtime evidence sources. Polylogue’s value is normalizing them together, not pretending they are all the same protocol.

`polylogue-wmj`, the OTel GenAI export lane, complements this. A2A is collaboration protocol; OTel is observability export; Polylogue is the canonical archive between them.

## A concrete future workflow

Imagine Claude Code and Codex are both working on one repo.

A Claude session starts. Through MCP, it calls `agent_coordination` and sees: current Bead, dirty paths, another Codex process in a sibling worktree, a running pytest episode, and a handoff ref.

Claude needs another agent to investigate a failing test. It asks Polylogue for a bounded context pack. Polylogue compiles the pack from evidence, not from vague memory.

Claude or Polylogue then opens an A2A task to a configured Codex-agent wrapper. The outgoing A2A Message includes the context pack as structured parts, plus `polylogue://` refs in metadata. The selected Agent Card hash and skill ID are archived.

The Codex agent streams status updates and returns an Artifact: maybe a diagnosis, patch, test log, or structured finding. Polylogue records the task lifecycle, artifacts, messages, and status updates.

A Bead is not automatically closed just because the A2A task completed. Instead, the returned artifact becomes evidence. A hook or human can close the Bead with refs.

Later, Polylogue can answer: who delegated what, to which agent, with which context, what artifact came back, what tests ran, what failed, what changed, what claim was grounded, and what should be injected into the next session.

That is exactly the synthesis: **A2A coordinates the exchange; Polylogue makes the exchange durable, searchable, auditable, and reusable.**

## Should Polylogue extend significantly into this domain?

Yes, but as an **interop/evidence bridge**, not as a full agent orchestration platform.

The major product opportunity is not “Polylogue becomes the orchestrator of all agents.” That space is crowded and will be fought by agent frameworks, cloud platforms, IDEs, and enterprise workflow vendors.

The sharper opportunity is:

**Polylogue makes multi-agent work inspectable.**

A2A will create more multi-agent traffic. That traffic will need a record. It will need provenance, cost semantics, artifact acquisition, claim auditing, task lineage, context-pack generation, and human judgment. That is Polylogue’s home territory.

So I would extend Polylogue in this direction:

Capture A2A as a first-class origin.

Expose Polylogue as a read-only A2A evidence/memory/audit agent.

Optionally proxy A2A calls so local users get automatic recording.

Do not build a custom external communication protocol.

Do not turn Beads into A2A tasks.

Do not turn Polylogue into a general-purpose distributed scheduler before the local trust floor is complete.

## Bottom line

The earlier statement should be revised from:

“Polylogue is not an A2A replacement; it is the evidence layer after the fact.”

to:

**Polylogue is not an A2A replacement. It should become the local evidence, memory, safety, and coordination substrate beneath A2A-style agent collaboration, with an optional A2A server/client/importer at the boundary.**

That direction keeps the project’s distinctive value and gives it a clean role in the broader agent landscape. A2A handles interoperable delegation. MCP handles tool/context access. Beads handles planned work. Polylogue remembers, verifies, contextualizes, and proves what happened.

---

