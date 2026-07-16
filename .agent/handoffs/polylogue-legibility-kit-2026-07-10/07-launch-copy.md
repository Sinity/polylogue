# Launch copy and talk tracks

## Polylogue one-sentence descriptions

**Primary:** Polylogue is the local flight recorder and system of record for AI work.

**Expanded:** It turns ChatGPT, Claude, Codex, Gemini, and coding-agent histories into one local evidence archive for searching tool activity, auditing claims against outcomes, understanding lineage and cost, and handing reviewed context to the next agent.

**Developer-oriented:** Git records what changed in source. Polylogue records the AI-mediated work around the change: prompts, tool calls, failures, branches, compactions, costs, judgments, and handoffs.

## Sinex one-sentence descriptions

**Primary:** Sinex is the local evidence substrate for digital life and agent work.

**Expanded:** It preserves source material from your machine, interprets it as typed events, records uncertainty and coverage gaps, and can replay those interpretations when parsers or models improve.

**Developer-oriented:** Think Nix-like reproducibility for personal evidence: retained materials, versioned interpretations, rebuildable projections, and explicit authority.

## Joint description

Polylogue explains AI work. Sinex preserves the wider evidentiary world in which that work happened. In Sinex-backed mode, transcripts and durable Polylogue-domain history live in Sinex; Polylogue remains the AI-work parser, ontology, query, memory, and user-experience layer.

## Polylogue README/HN opening

AI agents leave behind far more than chat. They execute tools, fail tests, fork sessions, compact context, consume cached tokens, and make claims about work they may not have completed. Those histories are split across vendors and formats, and a later agent usually receives either nothing or an unverified summary.

Polylogue is a local system of record for that work. It imports provider-native histories, preserves structured tool evidence, composes logical session lineage, separates durable evidence from rebuildable analytics, and lets reviewed assertions—not an agent's own unapproved claims—enter future context.

The private-data-free demo begins with one question: the assistant continued after a command failed; can you see the exact result rather than trust the prose?

## Sinex README/HN opening

Most “digital history” systems record rows and hope the sources stayed healthy. Sinex is built around a stricter question: what can the system honestly claim from the evidence it has?

It preserves source material separately from interpretation, distinguishes occurrence time from interpretation and persistence time, records source gaps as data, and replays retained material when parsers or models improve. Current state is a projection; old interpretations remain auditable.

The thesis demo reconstructs a workday around a failed build and includes the hole where one source was offline.

## Thirty-second Polylogue demo narration

“The assistant says it can continue, but the command returned exit code four. Polylogue renders those as separate facts. This card is the provider-native tool result; this is the assistant's authored text; this ref opens the original source block. Now zoom out: the archive finds every structural failure, not every message containing the word error. This fork inherited two parent messages, so the physical archive keeps both provider artifacts while the logical view counts the prefix once. That is the difference between storing chats and recording AI work.”

## Thirty-second Sinex demo narration

“This is the workday around a failed build. Terminal, Git, browser, files, focus, and agent activity line up on one interval. The gray band is not an empty period; the browser source was unavailable, and Sinex carries that coverage error into the answer. Select any item and it resolves to source material. After a parser fix, the current interpretation changes, but the old reading and the original bytes remain inspectable. Sinex can change its mind without rewriting history.”

## Three-minute joint talk outline

1. Agents need durable evidence, not just larger context windows.
2. Polylogue models the AI-work domain: sessions, tools, lineage, usage, assertions, and delivered context.
3. Sinex models the evidence world: material, time, replay, coverage, lifecycle, and cross-source relations.
4. The combined Agent Work Packet joins Beads intent, Polylogue cognitive/action history, and Sinex machine evidence.
5. An agent-generated lesson remains a candidate until judged. A source outage remains a gap. A copied prefix remains physical evidence but is not charged as distinct logical work.
6. The next agent receives a bounded packet whose evidence and omissions are recorded.

## Suggested repository topics

Polylogue: `ai-agents`, `llm`, `chat-history`, `observability`, `provenance`, `mcp`, `local-first`, `developer-tools`, `agent-memory`, `sqlite`.

Sinex: `event-sourcing`, `provenance`, `local-first`, `personal-data`, `observability`, `replay`, `nats`, `postgresql`, `ai-agents`, `digital-history`.
