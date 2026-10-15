# Purpose and scope

## In one sentence

Polylogue keeps, on your own machine, every AI conversation and agent session
your tools left behind, exactly as they wrote it, and lets you and your agents
search, read, measure and build on it. Where a record is missing (a provider
with no export, a location that cannot be read), it says so rather than
claiming to be complete.

## Beside the agent loop, not in it

Polylogue records agents' work as it happens (hook lines within seconds,
files within two seconds of being written) and hands it back inside their
loop (briefings at session start, takeovers, MCP reads, notes agents
write), so it is the evidence layer agents run on, not only a later
archive. It is never in the path of a tool call: a hook records and
returns within its budget, a briefing that cannot be built in time is
skipped and says so, and an agent works the same with the daemon down.
This is the spec's answer to the old question whether Polylogue should
sit "underneath" the agent (issue #2384 of the old repository, which
waited for the operator's ratification and has not had it): its capture
and its reads are in the loop, its failures never are.

## Who uses it

- **The operator**, at a terminal or in a browser: finding a past discussion,
  reading what an agent actually did, checking what a week of agent work cost,
  noticing where agents waste time, keeping notes on sessions.
- **Coding agents** (Claude Code, Codex and others), through the CLI, an MCP
  server and a shipped skill: recalling what was decided before, what was
  tried and failed, what the last session in this repo left unfinished, and
  getting a compact, referenced briefing at the start of a session. Agents are
  a primary user, not an afterthought; most queries may come from them.
- **Other local tools** (Lynchpin today, Sinex later), through the CLI's
  JSON output (the Python client), the change feed, the versioned SQL views
  and the export format; Sinex also links `polylogue-model`.
- **Strangers**, eventually: Polylogue is an open-source project meant to be
  installable and useful on someone else's machine with their own data.

## What it does

Grouped by the jobs people bring to it. Each job is specified in detail in the
files named.

1. **Keep everything.** Read every AI session the operator has, from every
   tool, in every format the tools have ever written; keep the exact bytes;
   keep them current in near real time; never lose or invent anything.
   (`sources/`, `storage/`, `convergence.md`)
2. **Find things.** One query language for sessions, messages, tool calls and
   files, with full-text and semantic search, used identically by the CLI,
   MCP, the web reader and the Python client. (`query.md`)
3. **Read things.** Any session, any message, any tool result, rendered for a
   terminal, a browser, an agent's context window or a Markdown file, with
   lineage (forks, resumes, subagents, compaction) shown honestly.
   (`surfaces/`, `rendering.md`)
4. **Measure things.** Tokens, cost, time, tool outcomes, failure patterns,
   agent behavior over time: computed facts, never prose guessed at by code.
   For every agent: what it ran, how each call went, how long it took.
   (`analysis.md`)
5. **Remember and continue.** Notes, tags and claims attached to exact
   messages; briefings for a new agent session; resume candidates for a repo;
   agents writing structured markers into their own output.
   (`user-data.md`, `analysis.md`)
6. **Build on it.** Saved queries and their results as objects that can be
   referred to, investigations that keep a question with its evidence, and
   (later) pages, live agent control and a Sinex backend. (`user-data.md`;
   the order is in `plan/slices.md`) What grows on top of these, the uses
   that justify the substrate, is cataloged in `vision/uses.toml`, each use
   checked against the features it needs.

## Design stance

- **Ambient, not invoked.** "Nearly every Polylogue wish is a request to
  externalize executive function" (operator, bead `jgp`). Value should arrive
  without the operator remembering to ask: briefings at session start, status
  on a bare `polylogue`, resume candidates on entering a repo. Volume stays
  restrained: references and short lines, not dumps.
- **General mechanisms, emergent features.** One query language, one notes
  store, one convergence loop, one daemon, one request shape. Exports,
  "views" and reports fall out of queries plus rendering; a feature that needs
  its own special code path is a sign the general mechanism is missing
  something.
- **A substrate for models.** Polylogue does what code does well (selecting,
  joining, counting, timing, costing, embedding) and gives models clean,
  referenced material for what only they do well (judging, summarizing,
  explaining). It never pretends code understands prose. Polylogue itself
  never calls a model to read or judge anything: that work runs in the
  operator's agent sessions, through their subscription harnesses, which
  read through the CLI or MCP and write attributed notes (annotation passes,
  distillates, answers to questions about meaning). Its only outside model
  call is computing embeddings, when the operator enables them. This is the
  operator's usual rule of subscription-backed local agent paths (noted in
  the 2026-05-06 decision record `llm-semantic-enrichment-deferred`), which
  also says how an exception comes about: a model Polylogue would call
  itself (a local model, or a named API model such as that record's case)
  is added only as a separate, reviewable feature with its provider,
  prompts and provenance recorded and its output compared against the
  deterministic answer. None is planned.
- **Trust before features.** The operator must be able to rely on the archive
  holding everything its sources still offer, being current and correct, and see at a glance when it is not.
  An archive that is fast, faithful and boring beats one with many features
  that is sometimes wrong.

## Explicitly not in scope

- Desktop activity capture (window focus, shell history, browser tabs). That
  is Sinex's job. Polylogue observes only the process trees under agent
  harness processes, for live state (`model.md`, Live state); nothing else on
  the desktop.
- Replacing agent harnesses or hosting an agent loop. Polylogue reads their
  records; where it starts an agent (`continue N`, a ping), it runs the
  harness's own commands: its resume command, or, when a pinged session
  cannot be resumed, the harness's command for a new session of the same
  role, recorded as new (`user-data.md`, pings). It never runs an agent loop
  of its own.
- A general multi-user service. One user, one machine; sharing happens by
  export.
- Whether Polylogue backs a harness's memory tool (Anthropic's Memory tool
  over Polylogue notes) is an open question for the operator
  (`questions.md` #48); Polylogue reads harness memory files as inputs
  either way.
- Generating text: drafting replies, posts or notes in the operator's
  voice is a client of the archive, not a feature of it. Polylogue supplies
  the operator's own writing as examples and context, through queries,
  exports and briefings (`vision/uses.toml`: `my-voice`,
  `blog-from-sessions`).
- Guessing meaning from text with code (keyword "work kind" classifiers,
  heuristic "phases", confidence scores without a basis).
- Repair and maintenance procedures, backup policies, and other operations
  that a correct design does not need. (Backup is the filesystem's job; a
  documented list of what to back up is enough.)

## Relationship to other projects

| Project | Role | Relationship |
|---|---|---|
| Sinex | the operator's personal event store | For the operator, Polylogue runs on Sinex (Sinex epic `sinex-4j2`, Polylogue epic `polylogue-303r`): after finite shadow qualification and a rebuild proof, with Sinex as the source of truth and Polylogue's SQLite a rebuildable read model. The two share the content store's format (`hashed-blobs`) and the model crate; in `primary`, Sinex's store is the only physical store of source material, and Polylogue publishes conversations already merged from their routes (joint decisions 27 and 28). Polylogue also works fully without Sinex, on SQLite alone, and the default build has no Sinex code. Details in `decisions/sinex.md`. |
| Lynchpin | provisional analysis of the operator's personal data, in place until Sinex is ready | Its analysis of AI sessions moves into Polylogue; its cross-source analysis is meant to move to Sinex. Until then it needs a stable read interface (the Python client). |
| Sinnix | the operator's NixOS configuration | Packages and runs Polylogue (service, hooks, extension host). Polylogue must not depend on it to build or test. Its capture services (screen, audio, clipboard and so on) are provisional stand-ins until Sinex takes them over; none of them is Polylogue's concern. |
| Agent harnesses (Claude Code, Codex, Hermes…) | produce sessions | Polylogue reads their files and hooks; it ships a skill and MCP entry for them. |
