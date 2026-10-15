# Workflows

The jobs people actually bring to Polylogue, walked through with real
commands. These are the acceptance stories for the product as a whole: if a
workflow here is awkward, the design is wrong, not the workflow. Outputs are
against the Morrow example archive (`examples/archive/FACTS.md`), clock
2026-10-08 12:00 local. Command syntax is defined in `surfaces/cli.md`;
output formats in `surfaces/cli.md` and `rendering.md`.

## Real needs, by frequency

From `evidence/real-questions.md` (74 needs found in a year of the
operator's sessions and agents' Polylogue calls), the ten that came up most:

| Need | Command |
|---|---|
| "Previous session was `<uuid>`: recover the state and continue" (about 40 times, across harnesses) | nothing extra: pasting the ID or link into the prompt makes the prompt hook inject a takeover briefing of that session; by hand, `polylogue 7f3a2c10-0004 brief` or MCP `brief {"query": "7f3a2c10-0004"}` (a bare native ID from any harness works) |
| Pasting the tail of the last chat by hand (about 25 times) | nothing: with `install-agent` done once, every new session starts with `polylogue brief` for its directory, while the briefing evaluation keeps automatic delivery (`user-data.md`, delivery on probation) |
| What did the operator ask in that session (their prompts only) | `polylogue 'messages where session.id:7f3a2c10-0002 written_by:human'` |
| Recover the agent's todo list and plan | `polylogue 7f3a2c10-0004 brief` (its "plan and todo list" section) |
| The last few things the model said, without tool noise | `polylogue 7f3a2c10-0002 read --view dialogue --last 4` |
| Take over another agent's work in a different harness | `polylogue 01a20000-0006 brief`, from a Claude Code session about a Codex one or the reverse |
| "When did I say this?" (the operator's own past statement) | `polylogue 'messages where written_by:human "chrono" \| sort by at asc'` |
| Is the archive current before I trust the answer? | `up_to_date` in every result; `polylogue status` |
| Open the archived copy of a web chat from its URL | `polylogue https://chatgpt.com/c/68f0a1b2-0007-8000-8000-000000000007 read` |
| Dump my own prompts for analysis | `polylogue 'messages where written_by:human since:30d' list --jsonl` |

## 1. Continue work (the most common need)

The operator, or a fresh agent, returns to `/home/op/morrow`, or has just
had a session compact.

**The agent side needs nothing.** After `polylogue install-agent claude-code`
(or `codex`), the harness's hooks deliver briefings without the agent
deciding to look anything up (`surfaces/mcp-and-hooks.md`, hook command):

- a new session in a directory starts with a briefing for that directory:
  the last sessions here with their age, last prompts and command outcomes,
  the latest todo list and plan, open items, decisions, unresolved failures,
  each with a reference to read more (`user-data.md`, briefings). The
  session asking is never in its own briefing;
- a resumed session gets only what changed while it was away;
- a prompt that names a session ("previous session was 7f3a2c10-0004,
  continue") gets a takeover briefing of exactly that session: its goal,
  todo list and plan, last exchange, files touched with how far the
  repository has moved since, open items and unresolved failures;
- when the harness compacts a session, the agent gets back every prompt the
  operator typed in that session, verbatim, so the summary losing them does
  not matter.

Whether this replaces the hand-pasted handoffs is measured on the archive
itself (`user-data.md`, evaluating whether briefings help).

**Naming a session works however it was copied:**

<!-- test: workflows-1 slice:4 -->
```console
$ polylogue 7f3a2c10-0004 brief                     # bare ID pasted from Claude Code
$ polylogue 01a20000-0006 brief                     # or from Codex
$ polylogue https://claude.ai/chat/0b9d4e6c-0008-4000-8000-000000000008 read
```

If a prefix matches sessions in two origins, the answer lists both instead
of guessing.

**The operator side:**

<!-- test: workflows-2 slice:6 -->
```console
$ polylogue continue
resume candidates for /home/op/morrow
1  Oct 03 14:00  claude-code:7f3a2c10-0004  Date parsing
   open: unanswered_prompt
   last: Clean: clippy passes, all 9 tests pass, and the change is committed as …
   claude --resume 7f3a2c10-0004-4000-8000-000000000004
2  Oct 06 09:30  codex:01a20000-0006        Review parser (try strict mode)
   last: Strict mode is in: Options { strict: true } rejects same-day weekday …
   codex resume 01a20000-0006-7000-8000-000000000006
3  Oct 05 16:00  codex:01a20000-0005        Review parser
   last: Noted on the message above: next friday bug in weekday_offset().
   codex resume 01a20000-0005-7000-8000-000000000005
1 more: polylogue continue --all
$ polylogue continue 1                              # runs: claude --resume 7f3a2c10-0004-…
```

Candidates whose session ended open (an unanswered prompt, an unfinished
tool call, an open question note, outcome `partial`, a final question) come
first, with the reason named (#4 is open: its last message is a prompt with
no reply, so it comes first; the rest follow by recency); the same rules are
the `ending` field, so
`polylogue ending:open since:7d` lists stuck work everywhere.

## 2. "What's going on?" (morning check)

<!-- test: workflows-3 slice:4 -->
```console
$ polylogue
Polylogue · 8 sessions (7 top-level) · last activity Oct 07 18:05
daemon    running · up to date
...
$ polylogue since:yesterday
1  Oct 07 18:00  claude-ai:0b9d4e6c  README wording   4 msgs  —
```

The overview answers "is it working, is it current, what is running, what
happened lately" without any arguments. When agents are working, a "running
now" block lists each with its last tool call; `polylogue running:true follow`
streams their activity line by line. Nothing has to be remembered or invoked
to keep the archive current; the daemon does that.

## 3. "What did we decide about X, and why?"

<!-- test: workflows-4 slice:7 -->
```console
$ polylogue 'notes where kind:decision "chrono"'
1  Oct 02 10:56  decision  "decided: chrono, not time"  operator  on claude-code:7f3a2c10-0002
$ polylogue claude-code:7f3a2c10-0002 read --view dialogue --from 10:50
10:52 model  All 6 tests pass. parse_relative handles "tomorrow at noon" and "in N days" using the time crate.
10:56 you    let's use chrono, not time
11:07 model  Switched to chrono. Tests, clippy and build are clean.
11:11 you    Which crate does parse_absolute need?
11:15 model  chrono covers it too: NaiveDate::parse_from_str for ISO dates. I'll add it next session.
```

The `dialogue` view keeps only what the human typed and what the model said
directly, which is where decisions live; tool calls, their output and
thinking are dropped. The row shows the time of the message the note is
attached to; the note's own `created_at` is shown by `read --view summary`. An agent asks the same through MCP (`search` then `read` with
`view: "dialogue"`). If no decision note exists, a text search over human
prompts (`messages where written_by:human "chrono"`) finds where it was
discussed.

## 4. "What did the agent actually do, and where did it go wrong?"

<!-- test: workflows-5 slice:3 -->
```console
$ polylogue 'tool_calls where outcome:error since:7d'
1  Oct 02 10:37  claude-code:7f3a2c10-0002  Bash  cargo test                    exit 101
2  Oct 03 14:27  claude-code:7f3a2c10-0004  Bash  cargo clippy -- -D warnings   exit 1
$ polylogue @1 read --around --tools full
$ polylogue 'tool_calls where outcome:error | group by next.action | count'
```

Every tool call is a row with its command, outcome, exit code and duration,
so "what failed" is a query, reading the session around it shows the agent's
reasoning and the full output, and `next.` shows what agents do right after a
failure, counted rather than interpreted.

## 5. "What did this week cost?"

<!-- test: workflows-6 slice:3 -->
```console
$ polylogue since:7d analyze --cost --by model
$ polylogue since:09:00 analyze --cost --by origin
```

Costs come from recorded tokens and the price table; unknown costs (web
chats on a subscription) are shown as unknown and counted separately, never
as zero. With `since:` only messages inside the window count, so "since
9:00" means work done since 9:00, not sessions touched since then. With
`polylogue setting subscription.anthropic max-20x`, the overview also shows the
current usage window.

## 6. "How do my agents behave?"

<!-- test: workflows-7 slice:3 -->
```console
$ polylogue 'tool_calls where since:30d | group by tool, outcome | count'
$ polylogue 'tool_calls where outcome:error since:30d | group by command | count | sort by count desc | limit 10'
$ polylogue 'sessions where errors >= 3 since:30d'
$ polylogue 'tool_calls where mcp_server:polylogue | group by tool | count'
```

What agents run, how often it fails, which commands fail repeatedly, how
long tool calls take, which MCP servers they actually use: computed from the
records, groupable by model, repo, harness and time. Polylogue does not
label sessions with guessed "work kinds"; questions about meaning are asked
of an agent session that reads the selected sessions through the CLI or
MCP (Polylogue never calls a model itself, `purpose.md`), and its
conclusions are stored as attributed notes.

## 7. Leave something for the next agent (or for later)

- From the terminal: `polylogue @1 note "parser done except DST"`.
- From an agent, through MCP `note`, or by writing a marker on its own line
  in its answer: `::decision: use chrono for timezone math`. Markers become
  notes when the session is ingested, attributed to the agent's session, and
  are never applied twice. Markers in tool output or fetched pages are
  ignored.
- Next time, the briefing includes it; `polylogue 'notes where since:7d'`
  lists everything noted recently. A wrong pass of agent notes is withdrawn
  in one step (`polylogue 'notes where author:<session>' withdraw`).

## 8. Bring in a web chat export

<!-- test: workflows-8 slice:4 -->
```console
$ polylogue import ~/Downloads/chatgpt-export-2026-10-04.zip
stored ~/Downloads/chatgpt-export-2026-10-04.zip (ChatGPT export, 1 conversation)
```

Importing an overlapping newer export later adds only what is new; a
conversation also captured live by the browser extension is the same session
(`decisions/merging.md`).

## 9. Mine my own prompts

<!-- test: workflows-9 slice:3 -->
```console
$ polylogue 'messages where written_by:human provider_unmarked_bytes:>0 since:30d' list --jsonl > my-prompts.jsonl
$ polylogue 'messages where written_by:human | group by session.origin | count'
```

`written_by:human` selects operator-supplied text rather than harness-injected user-role text. `provider_unmarked_bytes:>0` retains messages with text outside precisely located provider-reported paste spans. This is a prompt-library proxy, not a typing or authorship filter: small unreported pastes and loaded files can remain, while reports with unknown extent leave the metric unknown. Export keeps provider reports, clipboard candidates and their evaluation state separate (`model.md`, Text-origin evidence).

## 10. Read a long session quickly

<!-- test: workflows-10 slice:7 -->
```console
$ polylogue claude-code:7f3a2c10-0002 read --view summary
$ polylogue claude-code:7f3a2c10-0002 read --last 20
$ polylogue claude-code:7f3a2c10-0002 read --from 10:37
```

A summary first (when, where, models, counts, tools, cost, lineage, notes,
first prompt), then any window of the transcript, opened at a position
without loading the whole session.

## 11. Is anything lost?

<!-- test: workflows-11 slice:3 -->
```console
$ polylogue source:gone
$ polylogue source:missing
```

`source:gone` lists sessions whose provider files were deleted (Claude Code
removes old transcripts) and that the archive still holds in full: the blob
store doing its job. `source:missing` lists sessions a harness's own index
knows about but whose transcript was never stored, so a loss is visible
instead of silent. `polylogue status` counts both.

## 12. Investigate something across sessions (later: pages)

First versions: a saved query plus notes.

<!-- test: workflows-12 slice:7 -->
```console
$ polylogue query save next-friday '"next friday"'
$ polylogue query:next-friday
$ polylogue codex:01a20000-0005 note --kind question "why does next friday return today on Fridays?"
```

Later (slice 12): an investigation page collects the question, the evidence
(message references, kept query results), findings written by agents through
markers, and the decision; agents are pinged by address from the page
(`user-data.md`, pages). The page holds the current state of the
investigation, so nobody has to reread the chats to reconstruct it.

## 13. Check what an agent claimed against what happened

An agent says "all tests pass". Was there a passing run, and what failed
before it?

```console
$ polylogue 'id:claude-code:7f3a2c10-0002 | tool_calls where command:"cargo test"'
```

lists every `cargo test` run in the session with its outcome and exit code,
in order, so the claim can be read next to the record. `presentation.md`
builds the README's lead demo on the same idea; a later fix never overwrites the
failure it fixed.

## 14. See what the last agent left, on entering a repository

A person entering a repository runs `polylogue brief` (or puts it in
a devshell or shell hook): the same directory briefing an agent gets at
session start, with references to read more.

## 15. Ask the archive in SQL

For analysis the query language does not cover, `polylogue sql 'SELECT …'`
sends read-only SQL to the daemon, which runs it over the documented views
(`storage/index-db.md`) on a read-only connection. For heavy analysis,
export with `polylogue sql … --jsonl` and load the result elsewhere;
nothing outside the daemon opens the archive's files.

## 16. How is Polylogue itself used?

Every agent's call to Polylogue is a tool call in an archived session, so
use is measured with an ordinary saved query that ships with Polylogue,
not a telemetry table:

```console
$ polylogue 'tool_calls where mcp_server:polylogue | group by tool, outcome | count'
```

A rising error share on `search` (the most used tool, which failed on 28%
of calls in the Python version) or a tool nobody calls is the signal to
fix or drop something. Analytic MCP tools are not added per report: an
`analyze` report is a set of query pipelines (`surfaces/cli.md`, `analyze`),
and the one `search` with grouping runs each of them.

## 17. The story of a project, from the real archive

The main demonstration of Polylogue is the operator's own archive, not a
synthetic one (`evidence/intent-journal.md`): a grounded account of how
coding agents were used and how each project developed. It is a page
(slice 12) per repository, by repository identity (`repo_root`), built from
stored facts and checked notes only:

- top-level sessions per logical week by harness and model, with cost and
  engaged time (`analysis.md`), and delegated work beside them (subagents per
  top-level session, their share of tool calls and cost), never one mixed
  count: in the operator's Claude Code data subagents are most sessions
  (367 of 369 in one week; `evidence/real-archive-story-probe.md`);
- commits and PRs linked to the sessions that made them;
- accepted `decision`, `lesson` and `finding` notes in order, each linked
  to the messages it is about;
- outcome and failure tags from annotation passes the operator has sampled
  (`user-data.md`, checked claims and annotation passes);
- lineage: which sessions continued which, which agents were delegated to.

Any narrative text on the page is an agent's `finding` notes, each
anchored, and shown with its check state; nothing on the page is free text
without a source. The Morrow archive stays the reproducible demo
(`presentation.md`).

## 18. A report of one session

The web reader's session report (slice 10): one page per session with its
turns on a timeline, every tool call with its outcome and duration, the
files it touched, its cost so far, compactions, corrections by the
operator, the sessions it continued or delegated to, and agents' notes
anchored where they apply. It is computed from stored facts, so it works
on every session without asking a model, and an agent's analysis of the
session appears as checked notes on the same page rather than as a
separate generated document.

## 19. Share an annotated excerpt

From the web reader (or the extension on a ChatGPT or claude.ai page whose
conversation is archived), select a range of messages and the notes on
them and save it as one self-contained HTML file. Before saving, the page
lists what will be included and flags blocks marked `has:secret`, each
excluded unless the operator keeps it. Nothing else is redacted: paths,
names and addresses are the operator's own and stay as written, as
everywhere else (`behavior.md`, privacy). Polylogue writes the file and stops there:
publishing it is the operator's act. Slice 10 for the file; slice 12 for excerpts with page
layout.

## 20. Improve agent instructions from history

An agent studies past sessions of a repository for what keeps going wrong:
the operator's corrections, commands that fail the same way, questions
asked again. It writes `lesson` notes, each anchored to the sessions that
show it and carrying a check (for example, that the same correction
appears in at least three sessions). The operator accepts or rejects them
(Judgment). Accepted lessons are turned by the agent into a proposed change
to the repository's `AGENTS.md` or a skill, made through ordinary git;
Polylogue never edits instruction files itself. Afterwards the same saved
query shows whether the correction still recurs in sessions after the
change. Slice 7.

## 21. Study how questions to models went

When the operator asked models to review or analyze something and they
missed problems found later, the archive holds both the questions and the
later corrections. A rubric pass (`user-data.md`) over the review sessions
tags, for each problem on a list the operator provides, whether the model's
answer `noticed` or `missed` it, anchored to the passage; the operator
judges a blind sample; grouping by model, prompt shape and session length
then shows what kind of question gets problems noticed. Slice 7.

Bead and audit IDs behind each section: `evidence/beads/spec-sources-g3.md`.
