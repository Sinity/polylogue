## Turn 201 (role=assistant)

Yes. There is a larger interop landscape Polylogue should care about, and it is broader than A2A/MCP. The right move is not to “pick one protocol.” The right move is to define Polylogue as a protocol-neutral evidence core with adapters around it.

My strongest recommendation: **Polylogue should not invent a universal agent protocol. It should become the recorder, indexer, memory layer, and proof surface across agent protocols.** A2A, MCP, Agent Client Protocol, OpenTelemetry/OpenInference, W3C PROV, CloudEvents, AGENTS.md, Agent Skills, and artifact-provenance standards are the most relevant near-term set.

## The core doctrine

Polylogue’s internal model should remain richer than any external protocol. External protocols should be treated as four kinds of boundary:

An **origin**: raw protocol traffic is imported into Polylogue as evidence.

A **serving surface**: Polylogue exposes archive/search/context/audit capabilities through that protocol.

A **proxy/observer**: Polylogue sits between tools or agents and records what crossed the boundary.

An **export format**: Polylogue emits proof bundles, traces, provenance graphs, or artifacts in a standard format.

The mistake would be to let A2A, MCP, OpenTelemetry, or any other protocol become the canonical schema. Those protocols solve narrower problems. Polylogue’s canonical layer should preserve sessions, messages, blocks, actions, files, attachments, tool results, costs, topology, user judgments, context injections, handoffs, identities, and raw source envelopes.

## The “adopt directly” set

### A2A: external agent-to-agent delegation

A2A remains the main candidate for inter-agent delegation. The official A2A spec frames it as an open standard for communication and interoperability between independent, potentially opaque agent systems, with goals around capability discovery, modality negotiation, collaborative task management, and secure information exchange without exposing an agent’s internal memory/tools/state. citeturn778405view0

Polylogue should implement A2A in three stages: first as an **A2A origin importer**, then as a **read-only A2A server** exposing archive/context/audit skills, and only later as an **A2A client/proxy** for outbound delegation. The importer is the highest-leverage slice because it makes A2A traffic auditable immediately.

Do not remodel Beads as A2A Tasks. A Bead is durable project intent with dependencies, acceptance criteria, delivery gates, and operator judgment. An A2A Task is a runtime delegation. They should reference each other, not merge.

### MCP: local tool/context access

MCP is already highly relevant because it standardizes how AI applications connect to external systems, data sources, tools, and workflows. The official MCP docs describe it as an open standard for connecting AI applications to external systems; the common analogy is “USB-C for AI applications.” citeturn778405view1

Polylogue already fits MCP well: agents can call `search`, `read_session`, `get_postmortem_bundle`, `get_pathologies`, and `agent_coordination` without Polylogue needing to own their runtime. The next step is probably not “more MCP tools”; it is **contract hardening**: stable schemas, stable refs, proper refusal/degraded-mode semantics, and low-token context products.

Also watch **MCP Apps**. The MCP Apps extension lets MCP tools return interactive UI components rendered inside the host conversation, which is directly relevant to evidence cockpit panes, trace viewers, cost breakdowns, topology views, and action timelines. citeturn343791search4

### Agent Client Protocol: IDE/editor ↔ coding agent

This one is especially relevant to Polylogue because so much of the project’s value is coding-agent evidence. Agent Client Protocol standardizes communication between code editors/IDEs and coding agents; it uses local JSON-RPC over stdio today, with remote HTTP/WebSocket work in progress, and includes coding-specific concepts such as diffs. citeturn778405view5

This should be a high-priority interop target. It is different from A2A: A2A is agent-to-agent; Agent Client Protocol is client/editor-to-agent. For Polylogue, ACP traffic would be gold: prompts, diffs, file edits, approvals, terminal activity, status, and agent outputs can all become first-class evidence.

Recommended Polylogue role: **ACP capture origin** and maybe later **ACP-compatible evidence sidecar**. It should not replace Claude Code/Codex/Hermes adapters, but it could become the generic lane for future coding agents that support ACP.

### OpenTelemetry + OpenInference: trace import/export

Polylogue should support trace interop. OpenInference defines semantic conventions for AI app observability on top of OpenTelemetry and explicitly covers LLM calls, agent reasoning steps, tool invocations, retrieval operations, token economics, agentic control flow, and privacy sensitivity. citeturn545370view1

The OpenTelemetry GenAI conventions are also increasingly relevant because they define fields around agents, conversations, tools, retrieval, token usage, and model interactions. citeturn327678search5

Polylogue should not become “just another tracing backend.” Its role is stronger: ingest traces when available, export canonical evidence as traces when useful, and preserve richer archive evidence that ordinary traces omit. A tool call in OpenTelemetry is a span/event; in Polylogue it can also be a command, output, error, path touch, claim-support edge, cost event, and future-context ref.

Near-term Bead: **OTel/OpenInference import-export lane**. Import spans as source evidence; export selected sessions/postmortems as trace bundles.

### W3C PROV: formal provenance export

W3C PROV is almost tailor-made for Polylogue’s proof-bundle ambitions. PROV defines provenance as information about entities, activities, and people involved in producing something, useful for assessing quality, reliability, and trustworthiness. Its data model is domain-agnostic and includes entities, activities, agents, derivations, bundles, links, and collections. citeturn545370view2

Polylogue should export selected evidence bundles as PROV. This is the right format for “what produced this artifact?” and “which agent/tool/session/file/test contributed to this claim?” It is not a good internal schema for everything, but it is a strong external proof language.

Mapping is straightforward: files/artifacts/messages are PROV entities; tool calls/runs/agent tasks are activities; humans/agents/models/tools are agents; context packs and reports are bundles; edits and generated artifacts are derivations.

### CloudEvents + AsyncAPI: live event interop

CloudEvents gives a common envelope for event descriptions across services and platforms, and it has graduated in CNCF. citeturn545370view3 AsyncAPI describes message-driven APIs in a machine-readable, protocol-agnostic way across transports such as AMQP, MQTT, WebSockets, Kafka, STOMP, and HTTP. citeturn971984search9

Polylogue should use CloudEvents for daemon/live/capture events and AsyncAPI to document those event streams. For example: `session.ingested`, `message.indexed`, `tool_call.failed`, `artifact.acquired`, `context_pack.generated`, `a2a_task.completed`, `claim.judged`, `bead.linked`.

This avoids inventing a custom event envelope. It also gives external systems a way to subscribe to Polylogue’s evidence lifecycle without scraping logs.

### OpenAPI, JSON Schema, JSON-RPC

OpenAPI defines a standard language-agnostic interface description for HTTP APIs so humans and computers can discover and understand service capabilities without reading source code or inspecting network traffic. citeturn971984search4 JSON Schema defines validation vocabulary for JSON documents. citeturn971984search2 JSON-RPC 2.0 is a lightweight, transport-agnostic RPC protocol, and MCP itself requires JSON-RPC 2.0 for messages. citeturn971984search3turn971984search11

For Polylogue: use OpenAPI for daemon HTTP; JSON Schema for canonical payloads, proof bundles, context packs, and protocol-envelope records; JSON-RPC where protocol compatibility requires it. Do not invent a new RPC style unless absolutely necessary.

This belongs under the existing “contracts own surfaces” direction. The public CLI/MCP/HTTP/Python shapes should be generated or checked against shared schemas.

## The “strongly relevant but not first” set

### AGNTCY: agent discovery, identity, messaging, observability

AGNTCY is broader infrastructure for an “Internet of Agents.” Its docs describe components for agent discovery, secure communication, capability description, decentralized identity, and observability/evaluation. It includes an Agent Directory Service, SLIM messaging, OASF for describing agent attributes/skills/capabilities across A2A/MCP/etc., identity components, and observability/evaluation. citeturn778405view2

This is relevant, but I would not prioritize it above A2A/MCP/ACP/OTel. The most useful piece for Polylogue is probably **OASF**: a neutral way to describe agent capabilities and profiles. The Agent Directory Service is also interesting because it uses OASF and OCI registry ideas for agent metadata distribution and verification. citeturn778405view6

Recommended Polylogue role: import AGNTCY/OASF agent profiles into a canonical `agent_profile` table; preserve discovered Agent Cards/capability descriptors with hashes; maybe later export Polylogue-managed agent capabilities through OASF.

There is also an AGNTCY **Agent Connect Protocol** proposal: a REST/OpenAPI-style interface to invoke/configure remote agents across frameworks. citeturn778405view3 Treat this as watch/prototype unless it clearly wins adoption in the same space as A2A. Do not build against both deeply until there is demand.

One caveat: “ACP” is overloaded. BeeAI/IBM’s Agent Communication Protocol has been folded into A2A under the Linux Foundation, while the Agent Client Protocol is a different editor/coding-agent protocol. citeturn778405view4turn778405view5 Polylogue’s Beads should spell these out fully to avoid acronym bugs.

### NLIP: Natural Language Interaction Protocol

NLIP is a vendor-neutral application-layer protocol for communication between AI agents or between a human and an AI agent. ECMA-430 covers the core; related ECMA bindings cover HTTP/HTTPS, WebSocket, AMQP, and security profiles. citeturn154408search2turn154408search1

This is worth watching because it overlaps conceptually with A2A but has a broader “natural-language interaction” framing. I would not make it a near-term dependency. The practical strategy is: store raw NLIP sessions if encountered, build a generic message/part/artifact mapping, but do not let NLIP compete with A2A in Polylogue’s own roadmap until adoption clarifies.

### AG-UI, A2UI, and MCP Apps: UI interop

AG-UI is an event-based protocol for connecting AI agents to user-facing applications; it explicitly positions itself beside MCP for tools/data and A2A for agent-agent communication. citeturn955304search0turn955304search2

A2UI is a declarative UI protocol for agent-driven interfaces. It lets agents generate rich interactive UI descriptions while the client retains control of rendering, styling, security, and native components. It is designed to travel over transports such as A2A, AG-UI, SSE, and WebSockets. citeturn402505view0turn402505view1

Polylogue should care about this for two reasons. First, if agents produce UI events or UI artifacts, Polylogue should archive them. Second, the future evidence cockpit should not invent a bespoke “agent returns dashboard widgets” format if AG-UI/A2UI/MCP Apps can cover it.

Recommended sequence: capture AG-UI/A2UI/MCP Apps payloads as evidence first; use them to render Polylogue panes later.

### Agent Skills and AGENTS.md: context, instructions, reusable capabilities

Agent Skills are standardized capability bundles: a folder with `SKILL.md`, metadata/instructions, optional scripts, references, and assets. The spec emphasizes progressive disclosure: metadata first, full instructions on activation, resources only as needed. citeturn410624view0turn410624view1 OpenAI also documents Agent Skills as versioned bundles compatible with the open Agent Skills standard. citeturn410624view2

AGENTS.md is an open convention for giving coding agents project guidance, described as a “README for agents” and used by a large number of open-source projects. citeturn181595search0

These are immediately relevant to Polylogue’s context-pack and coordination work. Polylogue should import `AGENTS.md` as context evidence, record which instructions were active in a session, and generate/validate project-specific Agent Skills from durable Polylogue knowledge.

There is a nice product move here: **Polylogue can generate an evidence-backed `AGENTS.md` or Skill pack from judged assertions, repo doctrine, Beads state, and recent failure modes.** That turns the archive into a safer instruction compiler for future agents.

## Artifact, provenance, and supply-chain interop

This area matters because Polylogue is evidence-first. As soon as agents produce files, patches, builds, reports, screenshots, datasets, or model artifacts, you want standard provenance and integrity formats.

SPDX is an ISO/IEC standard for software bills of materials and can represent software components plus other AI/data/security references. citeturn168709search4 CycloneDX is an OWASP/Ecma full-stack Bill of Materials standard with supply-chain risk capabilities and support for software, hardware, ML models, source code, configurations, pedigree, and provenance. citeturn168709search17turn168709search9

SLSA is a supply-chain security framework for artifact provenance: what entity built an artifact, what process was used, and what inputs were involved, with higher levels adding stronger tamper protection. citeturn168709search2turn168709search14 Sigstore provides open-source tools for signing and verifying software artifacts. citeturn168709search3

OCI artifacts matter because registries are no longer just for container images. OCI 1.1 added support for associating metadata artifacts with existing images through `subject`, `artifactType`, and referrers-style discovery. citeturn436966search2turn927607search1

Polylogue should use these standards for “proof products.” A postmortem bundle, generated patch, archived context pack, demo corpus, or reproducibility artifact could carry an SPDX/CycloneDX inventory, SLSA-style provenance, Sigstore signature, and OCI artifact packaging. That would make Polylogue outputs portable and independently verifiable.

C2PA/Content Credentials are relevant for images, screenshots, generated media, and browser-capture artifacts. C2PA describes technical standards for certifying the source and history/provenance of media content, and Content Credentials are positioned as a “nutrition label” for digital content. citeturn436966search0turn436966search20

SCITT is also worth tracking. The IETF SCITT working group defines interoperable building blocks for integrity and accountability in software supply-chain systems, and RFC 9943, “An Architecture for Trustworthy and Transparent Digital Supply Chains,” was published in June 2026. citeturn436966search1turn436966search5

Recommended Polylogue posture: do not implement a supply-chain platform. Instead, export and preserve attestations, signatures, SBOMs, Content Credentials, and SCITT receipts as first-class evidence.

## Identity, authorization, and delegated authority

This becomes critical the moment Polylogue exposes A2A, remote MCP, outbound proxies, payment records, or multi-agent delegation.

OAuth 2.0 is the baseline authorization framework for third-party access to HTTP services. citeturn318747search3 SPIFFE provides specifications for bootstrapping and issuing identities to services across heterogeneous environments. citeturn318747search2 W3C DIDs enable decentralized identifiers, and W3C Verifiable Credentials define a model for tamper-resistant claims made by issuers and checked by verifiers. citeturn318747search0turn318747search1

For Polylogue, this means: do not create a homemade identity scheme. Use OAuth/OIDC for user/API authorization, SPIFFE for service/workload identity when deployed beyond localhost, and DID/VC only where verifiable agent credentials or external attestations actually need that shape.

The FIDO Alliance is now explicitly working on trusted AI-agent interactions, including verifiable user instructions, agent authentication, and trusted delegation for commerce. citeturn643684view1 The OpenID Foundation has also called out the difficulty of authenticating and authorizing autonomous systems, especially with recursive delegation and cross-domain trust propagation. citeturn643684view3

There is also an IETF draft called Agent Identity Protocol, but it is explicitly a work-in-progress Internet-Draft, not a stable standard. citeturn643684view2 Watch it, do not depend on it.

Polylogue needs an internal identity model independent of any one standard: human, local agent process, remote agent, model, tool server, MCP client, A2A peer, browser extension, daemon, and background job. External identity standards should attach to those principals.

## Payments, commerce, and “verifiable intent”

This sounds far from Polylogue, but it is relevant because agents will increasingly spend money, book things, subscribe to services, or call paid APIs.

Google’s Agent Payments Protocol is an open protocol for agent commerce and is positioned as an extension for A2A and Universal Commerce Protocol. It focuses on authorization, authenticity, accountability, cryptographic audit trails, and “mandates” that express user intent for human-present and human-not-present transactions. citeturn252602view0 Google has also donated AP2 to FIDO, and AP2 v0.2 includes human-not-present payments and Verifiable Intent logs. citeturn252602view4

Universal Commerce Protocol is Google’s open standard for commerce surfaces such as AI Mode and Gemini, with merchant control over brand/customer data and accountability trails. citeturn252602view1 x402 is an open HTTP-native payment standard built around HTTP 402, intended for programmatic and agentic payments. citeturn252602view2turn252602view3

Polylogue should not execute payments. It should archive payment intent and results when agents use them: mandate, user instruction, agent identity, payee, quote, approval, transaction receipt, revocation, failure, and artifact purchased. This fits Polylogue’s evidence role perfectly.

A future Bead could be: **AP2/x402 payment-intent origin: archive verifiable user instructions, mandates, receipts, and failures; no payment execution.**

## Data, dataset, and workflow lineage

OpenLineage is an open standard for lineage metadata collection; it tracks datasets, jobs, and runs, and is designed for observing datasets as they move through complex pipelines. citeturn436966search3turn436966search11

This is not central for general chat/coding sessions, but it becomes important if Polylogue archives data agents, notebook agents, ETL workflows, analytics agents, or ML pipeline agents. The mapping is natural: Polylogue sessions/runs can reference OpenLineage runs; files/tables/datasets become datasets; tool calls become jobs or run events.

OpenSharing is also worth watching. The Linux Foundation announced it in June 2026 as an open vendor-neutral protocol for sharing AI/data assets, including agent skills, AI models, unstructured data, and table formats such as Delta Sharing and Apache Iceberg. citeturn493885search0turn493885search4

Polylogue should not become a data-sharing platform, but it should be able to cite, ingest, and export references to shared assets, skills, and models when agents use them.

## Documentation and external-legibility interop

This is less “protocol” and more “agent-readable documentation,” but it matters for Polylogue’s public surface.

`llms.txt` is a proposed convention for putting an LLM-friendly Markdown file at a website root, providing background, guidance, and links to detailed Markdown files. citeturn614425search0 Polylogue should probably publish one once the docs/site are mature. It is low-cost and fits the external-legibility lane.

Model Cards and Data Cards are also relevant as transparency artifacts. Model Cards were proposed as short documents accompanying trained models with intended use, evaluation, and limitations; Data Cards are structured summaries of essential facts about datasets across their lifecycle. citeturn614425search5turn614425search2

Polylogue can borrow this pattern for **Agent Cards, Archive Cards, Demo Corpus Cards, Context Pack Cards, and Evaluation Cards**. These do not need to be formal standards to be useful; they make evidence products legible.

## Plugin/runtime portability

This is optional, but worth keeping in the background. The WebAssembly Component Model is an architecture for interoperable WebAssembly libraries, applications, and environments; WASI defines standards-track APIs for software compiled to WebAssembly to run securely across environments. citeturn927607search0turn927607search2

This only matters if Polylogue eventually wants third-party importers, renderers, analyzers, redactors, or scoring plugins. If that happens, WASM/WASI is a better sandbox story than arbitrary Python plugins. Not near-term, but it belongs in “horizon architecture.”

## What to drop or avoid

Drop any plan that amounts to a custom remote agent messaging protocol. Use A2A, NLIP, or AGNTCY/ACP-style protocols at the boundary.

Avoid a custom event-stream envelope. Use CloudEvents and document streams with AsyncAPI.

Avoid custom API description. Use OpenAPI and JSON Schema.

Avoid custom identity, auth, and payment primitives. Use OAuth/OIDC, SPIFFE, DID/VC where appropriate, and AP2/x402 only as archived evidence unless Polylogue deliberately enters commerce.

Avoid building a full agent marketplace/directory. Watch AGNTCY ADS/OASF and OpenSharing instead.

Avoid making a custom agent-generated UI schema. Capture and possibly emit AG-UI, A2UI, or MCP Apps.

Avoid treating telemetry as the archive. OpenTelemetry/OpenInference are interop surfaces; Polylogue’s archive should remain richer.

## What I would add to Beads

I would add one doctrine epic, probably under `delivery:K-interop-origin-export` with dependencies into `C-read-evidence-contract`, `D-agent-context-coordination`, and `A-trust-floor`.

Name: **Interop boundary doctrine: protocols are adapters, not the archive schema.**

Acceptance criteria: every supported standard is classified as origin, serving surface, proxy/observer, export format, or deliberately unsupported; every protocol adapter stores raw envelopes; every mapping declares lossiness; every remote-serving path has auth, redaction, and degraded-mode behavior.

Then add child Beads:

**A2A origin/importer**: Agent Cards, Tasks, Messages, Parts, Artifacts, status streams, and errors become canonical evidence.

**A2A read-only server**: Polylogue exposes search, context-pack, postmortem, claim-audit, and coordination-status skills.

**Agent Client Protocol capture**: editor/coding-agent traffic becomes sessions/actions/files/diffs/proofs.

**OTel/OpenInference import-export**: canonical sessions/actions/tool calls can round-trip to trace spans/events where possible.

**W3C PROV proof export**: selected reports export as provenance bundles.

**CloudEvents/AsyncAPI daemon stream**: live archive events have a standard envelope and published event spec.

**Agent Skills/AGENTS.md bridge**: import project instructions as evidence; export judged doctrine/context as Skills or AGENTS.md patches.

**UI protocol capture**: AG-UI/A2UI/MCP Apps payloads are stored and rendered as evidence.

**Identity subject model**: stable internal principals for humans, agents, models, tools, services, browser extensions, and remote peers, with OAuth/SPIFFE/DID/VC bindings where needed.

**Artifact provenance pack**: SPDX/CycloneDX/SLSA/Sigstore/OCI/C2PA/SCITT support for proof artifacts.

**Payment-intent evidence**: AP2/x402/UCP records are archived but not executed.

**AGNTCY/OASF watch lane**: import/export agent capability descriptors; do not depend on AGNTCY messaging until adoption demands it.

**OpenLineage bridge**: optional import/export for data-pipeline agent work.

## Priority order

My practical ordering would be:

First: **A2A importer, Agent Client Protocol capture, OpenTelemetry/OpenInference import-export, PROV export, CloudEvents event stream, OpenAPI/JSON Schema contracts.** These directly strengthen the archive and make it useful to agents without expanding too much surface area.

Second: **Agent Skills/AGENTS.md, MCP Apps, AG-UI/A2UI capture, artifact provenance packs.** These turn Polylogue into a better context and evidence product.

Third: **AGNTCY/OASF, NLIP, OpenSharing, OpenLineage, AP2/x402/UCP, DID/VC/SPIFFE hardening.** These are important but should follow the trust floor and read/evidence contracts.

The strategic position is simple: **Polylogue should be the system that makes all these interop protocols accountable.** A2A coordinates agents. MCP connects tools and context. ACP connects coding agents to editors. OTel/OpenInference observes runtime behavior. PROV/SBOM/SLSA/C2PA/SCITT describe provenance and artifacts. AG-UI/A2UI/MCP Apps render interactive surfaces. AP2/x402 describe agent commerce. Polylogue’s job is to preserve the evidence, normalize the work, expose grounded context, and prove what happened.

---

