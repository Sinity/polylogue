Title: "Deep research: agent-memory and agent-observability landscape refresh — who does what, as of mid-July 2026"

Result ZIP: none — deep-research memo, deliver as a single Markdown response
(the orchestrator archives it). Use the deep-research contract.

## Mission

Polylogue positions as: local-first, multi-provider AI-session archive with
provenance, structured tool outcomes, cost accounting, lineage, search, and
MCP access — the post-run evidence/forensics layer for agent work. Before
external outreach, refresh the competitive/adjacent landscape with primary
sources (repos, docs, changelogs, funding announcements), all claims cited
and dated. The last thorough scan is ~3 weeks old; the field moves weekly.

Map, for each player: what it actually stores/retrieves (schema-level, from
code/docs, not marketing), local-first vs cloud, multi-provider ingestion
breadth (which agents/providers it can read), whether it captures
STRUCTURED tool outcomes (error flags/exit codes) vs prose, and its
demonstrated-retrieval-value evidence (benchmarks, evals).

1. **Funded memory layer**: Mem0, Letta (incl. the RL-trained Memory Models
   direction), Zep, LangMem/LangGraph memory, OpenAI/Anthropic native
   memory features (ChatGPT memory, Claude memory/projects as of July
   2026) — current capabilities and what changed since May 2026.
2. **Session/trace observability**: Langfuse, Braintrust, W&B Weave,
   OpenLLMetry/Traceloop, Arize Phoenix — do any ingest LOCAL coding-agent
   session files (Claude Code/Codex/Gemini session JSONL) as opposed to
   instrumented SDK traces? This distinction is our wedge; verify it still
   holds.
3. **Coding-agent-native history tooling**: anything that archives/searches
   Claude Code, Codex CLI, Cursor, opencode, Hermes Agent sessions
   (community tools included — search GitHub for recent projects); any
   "import your agent history" features shipping in the agents themselves
   (Hermes session search; Claude Code /resume evolution; Codex history).
4. **Claim-verification / agent-audit space**: anyone measuring
   claimed-success-vs-tool-evidence, silent failure rates, or agent
   reliability from real traces? (This is our headline demo's category —
   we need to know if it's still empty.)
5. **Local-first/sovereignty niche**: personal-data-warehouse lineage
   (HPI-style, Rewind/Limitless-style capture) intersecting with agent
   memory — anything new.

## Deliverable

A memo: per-category tables (player, storage model, providers read,
structured-outcome capture yes/no, local-first, evidence of retrieval
value, last-90-day momentum), each row cited; a delta section ("what
changed since late June 2026"); an honest wedge assessment — restate what
remains genuinely uncommon about Polylogue's combination (multi-provider
local corpus + structured outcomes + provenance discipline) and name the
2-3 players closest to closing that gap and how fast they're moving. Flag
unverifiable claims explicitly.
