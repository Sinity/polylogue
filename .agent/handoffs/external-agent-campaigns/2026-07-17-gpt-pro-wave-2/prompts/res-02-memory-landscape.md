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


---

## Research contract

Use Deep Research, not ordinary implementation mode. Prefer current primary and
official sources; record direct URLs, publication/update dates, access date,
and the exact claim each source supports. Distinguish standards, documented
provider policy, measured behavior, informed inference, and proposal. Search
for counterevidence and incompatible constraints rather than writing a survey
that merely confirms the mission premise.

A current Polylogue project-state archive may be attached for product context.
Inspect relevant source and Beads so the research resolves concrete project
decisions, but do not return speculative patches. Map each conclusion to the
named Polylogue decision/Bead, state what should change, what should not change,
and what local experiment would falsify the recommendation.

Present a substantive, self-contained research report with conclusions,
rationale, source-by-source support, counterevidence, limitations, missing
evidence, Polylogue decision mappings, and the likely value of another
iteration. It must remain useful to a reader who has not opened the attached
project-state archive.

Do not perform an adversarial review unless explicitly requested. On an
ordinary **iterate/continue** request, extend the strongest unresolved research
branch and return a revised complete report. On an explicit **adversarial
review** request, try to falsify the prior memo with counterevidence, later or
more authoritative sources, incompatible policies/standards, hidden product
assumptions, and experiments that would overturn its recommendations. Repair
legitimate findings and report what changed, what remains uncertain, and
whether another pass is worthwhile.
