# The joint Sinex–Polylogue story

## One sentence each

**Polylogue is the local flight recorder and system of record for AI work.**

**Sinex is the local evidence substrate for digital life and agent work.**

**Together they reconstruct not only what an agent said, but what it knew, attempted, changed, verified, omitted, and should be allowed to remember.**

These sentences should remain stable across READMEs, repository descriptions, social cards, talks, demos, and agent orientation prompts.

## The three-system stack

The strongest public explanation includes Beads because it makes the division of authority concrete:

```text
Beads       records intended work: tasks, dependencies, status, and acceptance.
Polylogue   records AI-mediated work: dialogue, tools, lineage, context, memory, and claims.
Sinex       records the wider evidentiary world: source material, machine activity, replay,
            coverage, lifecycle, and cross-domain effects.
```

An LLM or coding agent is an actor and analyst within this stack. It is not the authority over its own history.

## What each project owns

| Concern | Polylogue | Sinex |
|---|---|---|
| Provider exports and runtime formats | Parses and normalizes AI-session semantics | Stores original and normalized material durably |
| Sessions, messages, blocks, tool calls | Domain authority | Durable event/material backend and projections |
| Forks, continuations, subagents, compaction | Domain authority | Persists typed relationships and replay history |
| Physical versus logical accounting | Domain authority | Stores usage observations and serves scalable projections |
| Assertions, lessons, handoffs, context policy | Product/domain lifecycle | Durable proposal, judgment, and retention authority |
| Context compilation | Selects AI-work context and records omissions | Supplies ambient evidence, coverage, and model-effect ledger |
| Terminal, Git, browser, filesystem, desktop, system | Correlates and renders when relevant | Source and evidence authority |
| Source coverage and replayability | Consumes caveats | Owns source contracts, material evidence, continuity, and replay |
| Cross-domain work episodes | Supplies rich AI-work leg | Derives bounded joins across independent domains |
| User experience | Transcript reader, forensics, memory, coordination | Evidence workbench, source health, operations, broad recall |
| Standalone mode | SQLite authority | Not required |
| Backed mode | SQLite edge projection/offline cache | Durable authority |

The rule is:

> **Sinex owns durable persistence and evidentiary lifecycle. Polylogue owns AI-work semantics and product behavior.**

## Why the maximal backend is better than the current bridge

The current bridge can send a low-volume `session_indexed` summary containing a session ID, origin, content hash, message count, model, and optional cost. That is useful for notification and coarse correlation, but it cannot support the strongest combined product.

A metadata-only boundary prevents:

- rebuilding Polylogue from Sinex;
- one retention and deletion lifecycle;
- replaying provider interpretations under new semantics;
- resolving a cross-source finding to transcript bytes;
- sharing one model-effect and embedding ledger;
- durable judgments and context-delivery history;
- exact Agent Work Packets;
- stable cross-device or offline synchronization.

The maximal architecture stores both provider-native and Polylogue-normalized transcript material in Sinex, plus durable domain observations. Polylogue remains the parser and domain kernel. Its SQLite databases become standalone authority or backed-mode projections, caches, UI state, and offline outbox.

This avoids two bad extremes:

```text
Bad extreme 1:
  Polylogue keeps the real evidence; Sinex receives metadata.
  Result: split authority and weak interoperation.

Bad extreme 2:
  Every Polylogue table becomes generic JSON in core.events.
  Result: ontology collapse, slow queries, and event-browser sludge.

Target:
  Sinex material + durable history
  Polylogue domain reducers + projections + product
  explicit stable identity and revision contracts
```

## The shared public proof world

Both repositories should ship one coordinated deterministic scenario, tentatively named **Incident 14:32**.

A small repository contains a flaky test and one Bead. The evidence world includes:

1. A browser/design conversation proposes a clock-related fix.
2. A coding agent edits the wrong fixture.
3. A structural `pytest` tool result exits nonzero.
4. The assistant nevertheless says the issue is resolved.
5. A resumed or forked session copies a large transcript prefix.
6. A compaction summary omits the failed experiment.
7. A second agent reads the repository, runs a different command, and repairs the test.
8. An ambient terminal command occurs outside the agent transcript.
9. Git records the actual change and verification commit.
10. One source is intentionally unavailable for a bounded interval.
11. Parser semantics v1 misclassifies one event; v2 replays the same material correctly.
12. A candidate lesson is proposed, reviewed, and accepted; a stale candidate remains noninjectable.
13. A context image is delivered to the second agent with explicit omissions.
14. An attachment and its original bytes remain resolvable.

This is not one giant demo. It is a reusable evidence corpus from which small construct-valid demos draw.

## The combined flagship demonstration

### “The world around the claim”

The operator opens a claim from the first agent: **“All tests pass; the fix is complete.”**

Polylogue shows:

- the exact assistant sentence;
- the paired test invocation and structural result;
- the nonzero exit status;
- the absence of acknowledgment in the next turn;
- the copied fork prefix and unique tail;
- the compaction boundary;
- physical and logical token/cost views;
- the later successful verification;
- the accepted lesson and the stale rejected candidate.

Sinex expands the same interval with:

- the terminal command that was run outside the transcript;
- repository and branch state;
- file and Git observations;
- browser research;
- the source-outage caveat;
- the old and new parser interpretations;
- material refs to original bytes.

Beads adds:

- the intended task;
- dependency and acceptance state;
- whether the task was actually ready or closed at the relevant time.

The final Agent Work Packet answers:

```text
What was intended?
What did each agent receive?
What did each agent claim?
What did the tools report?
What changed on the machine?
What was verified?
What evidence was unavailable?
What was learned and approved?
What should the next agent do?
```

Every line resolves to a typed object or material anchor. Missing evidence appears as a gap, not a guessed sentence.

## The user journeys

### Resume a hard task

1. Select a Bead or repository.
2. Polylogue finds relevant logical sessions, prior attempts, failures, and reviewed assertions.
3. Sinex supplies branch state, commands, files, research, deployment effects, and coverage caveats.
4. The context compiler selects a bounded packet.
5. A delivery snapshot records exactly what the new agent receives.
6. The new work is captured into the same evidence plane.

### Audit an agent-authored change

1. Start from a commit, pull request, or Bead.
2. Resolve authoring sessions and subagents.
3. Compare prose claims against structural tool outcomes and external machine evidence.
4. Show missing or stale evidence.
5. Produce a citable finding packet.

### Explain a cost spike

1. Preserve provider-reported usage lanes.
2. Normalize fresh input, cache reads/writes, output, and reasoning without double counting.
3. Compose copied lineage separately from physical storage.
4. Join cost to verified outcomes and abandoned work only where outcome evidence exists.
5. State pricing and coverage assumptions.

### Reinterpret history

1. Keep the original provider or source material.
2. Run a new parser semantics version.
3. Emit a new Sinex interpretation.
4. Rebuild Polylogue projections.
5. Preserve stable Polylogue refs through the identity ledger.
6. Show semantic differences and affected findings.

### Forget sensitive material

1. Select stable domain objects or source material.
2. Resolve shared references and derivation dependencies.
3. Tombstone or physically purge under explicit policy.
4. Invalidate projections, FTS, vectors, reports, and context packs.
5. Rebuild local SQLite replicas.
6. produce an excision receipt with remaining references and limitations.

## Public positioning against neighboring categories

| Neighbor | What it mainly does | What the combined system adds |
|---|---|---|
| Chat export viewer | Displays conversations | Structured tool outcomes, lineage, reviewed memory, and evidence refs |
| Agent tracing | Observes one framework’s runs | Historical imports, multiple providers, ambient machine evidence, and long-lived judgment |
| Activity logger | Captures app or machine activity | Source material, replay, interpretation history, authority, and AI-work semantics |
| Vector memory | Retrieves similar chunks | Stable evidence, review, staleness, exact context delivery, and measured retrieval |
| Observability stack | Traces services and incidents | Human work, documents, tasks, conversations, and personal longitudinal state |
| Personal knowledge base | Stores authored notes | Machine-emitted evidence and structural outcomes |
| Data lake | Stores heterogeneous bytes | Typed occurrence semantics, coverage, replay, lifecycle, and product refs |
| Agent orchestrator | Executes or delegates work | Durable record, audited context, and cross-run continuity independent of executor |

## What not to claim yet

The joint story should not currently claim:

- that Sinex is already the canonical Polylogue backend;
- that a full SQLite rebuild from Sinex has been proven;
- that reviewed memory reliably improves task performance;
- that cost can always be attributed to outcomes;
- that all source gaps are detected;
- that all private evidence can be selectively excised;
- that semantic retrieval improves recall on a representative benchmark;
- that multi-agent coordination is production-ready;
- that either project is a general-audience stable release.

These are roadmap hypotheses or active programs, not shipped facts.

## The long-term category

The two projects together occupy a category that is not well named by “memory,” “observability,” or “lifelogging.” A serviceable working name is:

> **Local evidence and memory infrastructure for human–agent work.**

The strongest version of the thesis is:

> More capable agents increase the need for systems of record. As models execute more tools, inherit more summaries, delegate more work, and operate across longer time spans, the hard problem becomes deciding what actually happened, what evidence supports it, what was known at the time, what may persist, and what another agent should trust. Polylogue and Sinex are building that layer.
