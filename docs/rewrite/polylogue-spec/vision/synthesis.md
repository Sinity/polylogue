# Why build Polylogue, and what slices 1–4 give

2026-10-12. The vision catalog (`uses.toml`, 127 uses) is wide; this page
says which uses justify the first four slices, who meets each one, what
real data has shown, and what has not been shown yet. Uses are judged by
their core inputs: a use counts for a slice when that slice delivers the
data it cannot do without, not when every input it might read exists.

## Why

You run several agents at once across many repositories, and their work
disappears into provider files that get rewritten, compacted and deleted.
Three needs recur in your own records and chats:

- **Agents should start from what is already known**, not re-explore: the
  takeover briefing when you paste a session ID or link, your own words back
  after a compaction, and the recent work of a repository at session start.
- **Nothing should be lost**: providers delete and rewrite, so Polylogue
  keeps every revision of every input (principle #5).
- **Measures should be true**: a failure rate that counts unknown outcomes
  as success, or a cost that leaves unpriced usage out, gives the wrong
  answer on your data (below).

Your May diagnosis of earlier tools applies: surfaces you had to remember
to open went unused, while agents used the substrate (`README.md`, the
consumer test). So slices 1–4 are justified by what reaches agents and you
unasked, not by views.

## What each slice gives, and to whom

| Slice | Uses it makes real | Consumer |
|---|---|---|
| 1. Claude Code end to end | `quote-back` (exact text, date and session), every revision kept and readable through exact references | push (status), agent |
| 2. Live daemon, Codex, hooks | `continue-work` and `handoff-across-harness` (takeover briefings, from a pasted ID or link), `context-reboot` (a fresh session seeded by a takeover instead of a lossy compaction), `context-control` (each briefing recorded with what it left out), `missing-sessions` and `is-anything-lost` (provider coverage through `source:`), `ambient-interface`, `conflict-warning` (other agents live in the same repository), `cockpit` and `morning-check` (live state: who waits on you), `self-usage` | push (briefings, status line) |
| 3. Query language and facts | `what-agent-did` (turn by turn, first failing call), `agent-behaviour` (failure rates with denominators), `file-provenance` (which call wrote this file), `lifetime-cost` and `rate-limits` (usage per subscription window, each figure naming its basis), `archive-sql`, `read-long-session` | agent, workflow |
| 4. Web chats and the old archive | `import-web-export` and `all-providers` (ChatGPT and claude.ai merged with agent sessions, nothing counted twice), `mine-prompts` (your prompts as a library, pasted text excluded), `merge-branches`, `share-excerpt` | workflow, agent |

`lineage-cost` needs `parent:` and the `links` unit in slice 6.

After slice 4 you can search and read every session, live, from the CLI,
from MCP and through briefings. Notes, decisions and "what is next" arrive
in slice 7, and the briefings become much more useful then (below).

## What real data has shown

- **The store and loop hold on your corpus.** The prototype converged a
  real 2.4 GB subset with every revision kept. The one slow run (2.7 MB/s)
  came from syncing each new file on its own; with first observations
  batched, a 250 MB build ran at about 15 MB/s (`evidence/real-converge.md`).
- **Briefings work and are cheap.** On real sessions a directory briefing
  takes 4–12 ms warm and a takeover 1–9 ms. The takeover is the most useful
  of the four kinds, and your own prompts come back verbatim after a
  compaction (104 of 174 fit in 2,000 tokens on one long session). The
  directory briefing says what happened but not what is next, because no
  repository had a recorded todo list and notes are slice 7
  (`evidence/briefing-trial.md`).
- **The counting rules change answers.** On the old archive, a failure atlas
  showed ChatGPT at 0% failures only because exports rarely record outcomes,
  and a "lifetime cost" left out Codex, the largest volume, because it had no
  price. Both are errors the rewrite's rules forbid. A prompt library filled
  with pasted instruction files until pasted text was excluded
  (`evidence/vision-prototypes.md`).
- **Nothing useful needed a model.** Every useful result came from data the
  rewrite keeps; questions about meaning go to agent sessions, which read
  through the CLI or MCP.

## What has not been shown

- **Whether agents do better with briefings.** This decides whether
  automatic injection stays (`spec/user-data.md`, delivery on probation).
  The comparison, including reference uptake, has not been run.
- **The full first build on your machine within its targets.** The
  Claude Code build misses its 10-minute target today in the prototype
  (`spec/behavior.md`, speed targets), and one real build peaked at 8.7 GB
  of memory against the 2 GB target (`evidence/briefing-trial.md`, setup);
  the batched build has not been measured for memory yet.
- **The briefing's machine section from Sinex**, with its 100 ms and 25%
  budget, is not built; Sinex names it as its first measurement due.
- **Whether a reboot beats compaction.** No comparison yet of a takeover
  briefing against the harness's own compaction summary for the same
  session.
- **Direction in briefings.** Until notes exist (slice 7), what comes next
  lives only in prompts and replies.

## What would change the plan

- If the briefing comparison shows no benefit, automatic injection stops and
  slice 2's push uses shrink to the takeover briefing, which you ask for by
  pasting an ID.
- If the full first build cannot meet its targets with batched durability,
  the parse-commit granularity decision reopens (`plan/slice-01-starts-here.md`).
- If you answer `questions.md` #47 "converse with agents", the workbench
  (slice 12) moves up and is designed around conversation rather than
  direction.
