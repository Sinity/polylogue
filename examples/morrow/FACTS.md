# The example archive: "Morrow"

A small, fully synthetic archive that every example in the spec runs
against. It describes a week of work (Oct 01–07, plus the parents and live
sets on Oct 08–09) on a fictional Rust library,
`morrow` (parsing human-written dates), using Claude Code, Codex, ChatGPT and
claude.ai. The raw provider files live beside this file
(`claude-code/`, `codex/`, `chatgpt/`, `claude-ai/`); they are
read through the current parser and query rules. The query tests name
remaining differences from the historical expected examples; the public
tour checks the current answers it presents.

Operator's time zone for display: `Europe/Warsaw` (UTC+2 in October 2026).
Times below are local. Working directory for agent sessions:
`/home/op/morrow`. Repository URL (only Codex records it):
`https://github.com/op/morrow`.

## Sessions

| # | Session ID | Origin | Kind (link) | Local time | Title (provider) | Model | Messages (own) | Human prompts |
|---|---|---|---|---|---|---|---|---|
| 1 | `claude-code:7f3a2c10-0001-4000-8000-000000000001` | claude-code | main | 2026-10-01 09:00–09:42 | "Project skeleton" (`ai-title`) | claude-fable-5-1 | 12 | 3 |
| 2 | `claude-code:7f3a2c10-0002-4000-8000-000000000002` | claude-code | main | 2026-10-02 10:00–11:15 | "Date parsing" | claude-fable-5-1 | 18 | 3 |
| 3 | `claude-code:agent-a7f3a2c1000000003` | claude-code | subagent of #2 | 2026-10-02 10:20–10:26 | — | claude-opus-5-5 | 6 | 0 |
| 4 | `claude-code:7f3a2c10-0004-4000-8000-000000000004` | claude-code | main (continuation, resume, of #2) | 2026-10-03 14:00–15:10 | "Date parsing" | claude-fable-5-1 | 10 own + 18 inherited | 3 |
| 5 | `codex:01a20000-0005-7000-8000-000000000005` | codex | main | 2026-10-05 16:00–16:35 | "Review parser" (thread name, from `state_5.sqlite`) | gpt-6.1-sol | 13 | 2 |
| 6 | `codex:01a20000-0006-7000-8000-000000000006` | codex | main (continuation, fork, of #5) | 2026-10-06 09:30–09:50 | "Review parser (try strict mode)" | gpt-6.1-sol | 10 | 1 |
| 7 | `chatgpt:68f0a1b2-0007-8000-8000-000000000007` | chatgpt | chat | 2026-10-04 21:00–21:20 | "Date format edge cases" | gpt-6 | 6 on active path, 2 on an abandoned branch | 3 |
| 8 | `claude-ai:0b9d4e6c-0008-4000-8000-000000000008` | claude-ai | chat | 2026-10-07 18:00–18:05 | "README wording" | not recorded in the export | 4 | 2 |

Top-level sessions (excluding the subagent #3): 7. All sessions: 8.
A subagent's session ID is `claude-code:agent-<agentId>`. Codex item IDs
identify messages across files: copied parent messages keep their IDs, while
distinct messages have distinct IDs. Codex message counts are item
counts, including calls, results and commands recorded separately from their
script calls, rather than model-response groups. Session #5 has ten response
items and three command items; #6 has eight response items and two command
items. Repeated user-message events add no messages.
Usage observations do not combine items into model-response messages.
The source story's prompts, calls and outcomes remain unchanged.
Hook spool lines put `event_id`, `event_type` and `observed_at_ms` first,
as current hook spool files do. Every field from the generated example is kept.
The testkit and query tests read these files through the real parsers.

## What happened (the story the examples rely on)

- **#1** creates the crate: `cargo init`, writes `src/lib.rs` and
  `Cargo.toml`, runs `cargo build` (ok). Tool calls: Bash ×3, Write ×2,
  Read ×1, all `ok`.
- **#2** adds `parse_relative()`. The agent spawns subagent #3 (tool
  `Agent`, subagent type `Explore`) to survey the `chrono` crate. Runs
  `cargo test`: **fails** (exit 101, test `parse_dates::tomorrow_at_noon`),
  edits `src/parse.rs`, runs `cargo test` again: **ok**. The operator writes
  "let's use chrono, not time" (human prompt 2 of 3). Tool calls: Bash ×5 (one
  error), Edit ×4, Read ×3, Agent ×1. Its first model response is written
  as two records sharing one `message.id` (text, then the two `Read` calls),
  as Claude Code does; it is one message, and its usage counts once.
- **#3** (subagent) reads docs with WebFetch ×2 and Read ×2, all `ok`,
  returns a summary.
- **#4** resumes #2 the next day (its file starts with the 18 messages of #2,
  same IDs). Contains a **compaction** at 14:40 (manual, 182,000 → 14,500
  tokens). Adds `parse_absolute()`. One Bash `cargo clippy` with exit 1
  (`error`), then fixed (`ok`) by an edit to `src/parse.rs`; `src/lib.rs`
  re-exports the new function; the session ends by committing
  (`git commit`, output `[main 4e1d2c9] Add parse_absolute() for ISO dates`).
  Tool calls in own messages: Bash ×5 (one error), Edit ×3 (`src/parse.rs`
  ×2, `src/lib.rs` ×1). For its takeover briefing (`spec/user-data.md`):
  files touched `src/parse.rs` (2 edits), `src/lib.rs` (1); branch `main`;
  last commit `4e1d2c9`; no unresolved failure (the clippy error has a later
  success). Its second prompt (14:42) pastes #2's short ID `7f3a2c10-0002`;
  a short ID makes no `mention` link, but the hook spool records the
  briefing it caused ("Hook spool" below).
- **#5** (Codex) reviews the parser, finds that `"next friday"` on a Friday
  returns today. Tool calls: `exec_command` (action shell, each run as the one child of a
  code-mode `exec` wrapper) ×3, all `ok`;
  `apply_patch` ×0. The operator then asks it to note the bug.
- **#6** forks #5 to try a strict mode; runs `cargo test`, which passes.
  Tool calls: `exec_command` (shell, inside `exec` wrappers) ×2 `ok`,
  `apply_patch` ×1 `ok`.
- **#7** (ChatGPT) asks about ambiguous formats like `03/04`. The operator
  edited prompt 2 once; the abandoned branch has 2 messages.
- **#8** (claude.ai) improves the README's first paragraph.

## Times the examples use

Local times of the records, as `build.py` writes them. Terminal output shows
`HH:MM`, cut off rather than rounded.

**The examples' clock** is 2026-10-08 12:00 local (10:00 UTC, as in
`spec/query.md`, Results): after every record, note and the annotation set,
so no example shows something the archive could not know yet. With the
logical day starting at 04:00, `since:yesterday` covers Oct 07 04:00 to Oct
08 04:00 and today starts at Oct 08 04:00. The live set has its own clock
("Live set", below).

| What | Session | Time |
|---|---|---|
| `Agent` call spawning #3 | #2 | 10-02 10:19:30 |
| subagent's final answer; its result reaches #2 | #3, #2 | 10:26:00, 10:26:30 |
| `cargo test` call that fails (result at 10:41:25) | #2 | 10-02 10:37:42 |
| "let's use chrono, not time" (decision note target) | #2 | 10-02 10:56:21 |
| `cargo clippy -- -D warnings` call that fails (result at 14:36:00) | #4 | 10-03 14:27:00 |
| compaction boundary; summary record | #4 | 10-03 14:40:00; 14:40:41 |
| model message reporting the "next friday" bug (note target) | #5 | 10-05 16:25:40 |
| abandoned branch (2 messages) | #7 | 10-04 21:04, 21:05 |

A tool call's duration is the time from its call record to its result
record (the failing `cargo test`: 3m 43s).

## When #4's live file was collected

For the evidence-cutoff examples, tests feed #4's file
to the daemon in three stages, as a live session grows:

| Collected at (local, Oct 03) | The file then ends with | Becomes |
|---|---|---|
| 14:20:00 | the tool results at 14:18:00 (the inherited prefix plus #4's first 3 own messages) | input first seen; `first_seen` 14:20 for those messages |
| 14:45:00 | the compaction summary record at 14:40:41 | a second piece; `first_seen` 14:45 for the messages added |
| 15:12:00 | the last record, "Thanks, done for today." at 15:10:00 | a third piece; the whole file |

So `evidence_until:2026-10-03T14:30` shows #4 with 3 own messages, and
`evidence_until:2026-10-03T15:00` shows it with the compaction and 5 own messages
(the boundary and summary are an event, not messages).

## Tokens and cost

Prices used (from the example price table in `examples/prices.toml`, USD per
million tokens):

| Model | Input | Output | Cache read | Cache write |
|---|---|---|---|---|
| claude-fable-5-1 | 10.00 | 50.00 | 1.00 | 12.50 |
| claude-opus-5-5 | 4.00 | 20.00 | 0.40 | 5.00 |
| gpt-6.1-sol | 3.00 | 15.00 | 0.30 | — |
| gpt-6, web chats | not billed per token (subscription); tokens unknown |

| Session | Input | Output | Cache read | Cache write | Cost (USD) |
|---|---|---|---|---|---|
| #1 | 1,200 | 9,000 | 300,000 | 40,000 | 1.26 |
| #2 | 2,000 | 21,100 | 900,000 | 70,000 | 2.85 |
| #3 | 400 | 3,000 | 60,000 | 15,000 | 0.16 |
| #4 (own messages) | 1,500 | 14,000 | 500,000 | 30,000 | 1.59 |
| #5 (recorded observations) | 30,000 | 6,000 | 120,000 | — | unknown |
| #6 (recorded observations) | 18,000 | 3,000 | 60,000 | — | unknown |
| #7, #8 | unknown | unknown | unknown | unknown | unknown (subscription) |

The four token-count observations in #5 and two in #6 supply the Codex
figures above. Their validated cumulative increments equal their last-usage
counters; at the example prices these observations price to 0.216 and 0.117
USD. These are recorded known subsets, not complete session or request costs:
item-only rollouts cannot establish response membership or all API attempts.
Observation record times also do not establish spend times. The observations
remain evidence independently of message attachment (`spec/sources/codex.md`).

Archive total from sessions with known cost: **5.86 USD** (exact 5.8626;
rounded to cents only for display); 4 sessions have unknown total cost
(#5–#8). The 0.333 USD observation subset is not added as if those sessions
were complete. #4's inherited messages are not counted again.

## Notes in archive.db

| Note | Target | Kind | Author | Written at |
|---|---|---|---|---|
| tag `research` | session #7 | tag | operator | 2026-10-07 20:00 |
| tag `project:morrow` | sessions #1, #2, #4, #5, #6 | tag | operator | 2026-10-07 20:00 |
| "decided: chrono, not time" | the human message in #2 that says "let's use chrono, not time" | decision | operator | 2026-10-02 11:00 |
| "next friday bug found here" | the Codex message in #5 reporting the bug | note | agent `codex:01a2…0005` | 2026-10-05 16:30 |

A note's time is when it was written, never its target message's time
(`spec/model.md`, time). `check_facts.py` carries these times and checks
that each note is written after its target.

## Parents set (the Claude-Code-only home)

`build.py` also writes a second home, `claude-code-only/`, holding the
Claude Code sessions #1–#4 unchanged plus the harder parent cases below
(`spec/model.md`): copies, late parents, continuations and forks. None of
these sessions are in the main archive, so the main totals above are
unchanged. Time: 2026-10-08, local. "L1" etc. are names used here only.

| Name | Session ID | What it is | Own msgs | Copies counted elsewhere | Human | Own cost | Tree cost |
|---|---|---|---|---|---|---|---|
| L1 | `claude-code:7f3a2c10-0101-4000-8000-000000000101` | main, "Release checklist", 08:00–08:40; also present as a byte prefix under a second project folder, `-home-op-morrow-old` | 10 | 0 | 2 | 0.76 | 0.99 |
| L1-side | `claude-code:7f3a2c10-0101-4000-8000-000000000101~side-d1500000-0000-4000-8000-000000000001` | older-layout sidechain inside L1's file (no `agentId`), answering L1's `Task` call `toolu_1102` | 2 | 0 | 0 | 0.03 | 0.03 |
| L2 | `claude-code:7f3a2c10-0101-4000-8000-000000000101/a7f3a2c1000000102` | subagent of L1, spawned by `toolu_1101` (meta `toolUseId`) | 6 | 0 | 0 | 0.10 | 0.15 |
| L3 | `claude-code:7f3a2c10-0101-4000-8000-000000000101/a7f3a2c1000000103` | nested subagent of L2 (meta `parentAgentId`, `spawnDepth: 2`, `toolUseId: toolu_1201`) | 4 | 0 | 0 | 0.05 | 0.05 |
| L4 | `claude-code:7f3a2c10-0101-4000-8000-000000000101/a7f3a2c1000000104` | sibling copy: in L1's folder, repeats L2's first 4 records (L2's `agentId`), then continues; meta has no `toolUseId` | 4 | 4 | 0 | 0.05 | 0.05 |
| L5 | `claude-code:7f3a2c10-0105-4000-8000-000000000105` | explicit fork of L1 after L1's 4th record (`forkedFrom` on the copies), 09:00–09:30, with a microcompaction | 8 | 4 | 2 | 0.60 | 0.66 |
| L6 | `claude-code:7f3a2c10-0105-4000-8000-000000000105/a7f3a2c1000000106` | team `morrow-review`, name `lint`, subagent of L5 (`toolu_1501`) | 4 | 0 | 0 | 0.03 | 0.03 |
| L7 | `claude-code:7f3a2c10-0105-4000-8000-000000000105/a7f3a2c1000000107` | team `morrow-review`, name `tests`, subagent of L5 (`toolu_1502`); receives one peer message from `lint` (`written_by: model`, `from_session` = L6) | 5 | 0 | 0 | 0.03 | 0.03 |
| L8 | `claude-code:7f3a2c10-0108-4000-8000-000000000108` | "Weekday fix", 10:00–10:15; its first 3 records carry the `sessionId` of `7f3a2c10-0199-…`, which is not in the archive | 6 | 0 | 2 | 0.36 | 0.36 |

"Own msgs" counts the messages counted in the session (`counted_as` null); "Copies counted elsewhere" counts its copies whose source is stored (`session_facts.inherited_messages`). L8's three copies carry `inherited_from` but count as its own, since their source is absent (`spec/model.md`).

Links (kind, `how`, evidence):

| Parent | Child | Kind | `how` | Evidence | Spawning call |
|---|---|---|---|---|---|
| #2 | #3 | subagent | fresh | `folder` | `toolu_0203` (from #2's `toolUseResult.agentId`; #3's meta has no `toolUseId`) |
| #2 | #4 | continuation | resume | `copied-sessionId` | — |
| #5 | #6 | continuation | fork | `forked_from_id` (Codex) | — |
| L1 | L2 | subagent | fresh | `folder` | `toolu_1101` |
| L2 | L3 | subagent | fresh | `meta.parentAgentId` | `toolu_1201` |
| L1 | L4 | subagent | fresh | `folder` | none (stored without a block) |
| L2 | L4 | continuation | resume | `copied-agentId` | — |
| L1 | L1-side | subagent | fresh | `isSidechain` | `toolu_1102` |
| L1 | L5 | continuation | fork | `forkedFrom` | — |
| L5 | L6, L7 | subagent | fresh | `folder` | `toolu_1501`, `toolu_1502` |
| `claude-code:7f3a2c10-0199-…` | L8 | continuation | resume (pending: parent not in the archive) | `copied-sessionId` | — |

The main archive also has one `reference` link, #2 → #4,
`how: briefing`, from the delivery row below.

Things the examples rely on:

- **Tree cost.** L1's tree is L1 + L1-side + L2 + L3 + L4 = 0.9852 → $0.99;
  L2's tree is L2 + L3 = $0.15; L5's tree is L5 + L6 + L7 = $0.66. The
  subagents are not counted again in L1's own cost.
- **Copies.** L4's 4 copies of L2 and L5's 4 copies of L1 are inherited
  (their sources are in the archive) and add nothing to counts or cost.
  L8's 3 copies count as its own, because their source is missing; the link
  stays pending.
- **Two folders.** L1 is one session, equal to what the longer file alone
  gives.
- **Known cost of the whole Claude-Code-only home:** **$7.87** (exact
  7.8715): #1–#4 5.8626 plus the parents set 2.0089.


## Live set (hook and process spools)

`build.py` also writes a third home, `live/`, for the live-state tests:
hook spool files with process identities and
job variables, a process spool, and minimal transcripts for three sessions.
None are in the main archive. Clock: 2026-10-09 15:00 local. Boot ID
`7b1e0000-0000-4000-8000-00000000b007`; every session works in
`/home/op/morrow`.

| Name | Session ID | Harness process (PID, start) | Evidence | Live state at 15:00 | Waiting on | Task | Peers |
|---|---|---|---|---|---|---|---|
| V1 | `claude-code:7f3a2c10-0201-4000-8000-000000000201` | 5101, 900000 | started 14:40; `PreToolUse` Bash `cargo test --all` 14:41:00, `PermissionRequest` 14:41:01; no process line (it has no children) | `waiting` since 14:41 | `permission` | — | V2 |
| V2 | `codex:01a20000-0202-7000-8000-000000000202` | 5202, 880000 | started 14:30 with `AGENTCTL_JOB_ID=job-morrow-7`; `PreToolUse` exec 14:58:30 with no `PostToolUse`; child `cargo test parser` (PID 5230) started 14:58:32 | `working`, open tool exec | — | `job-morrow-7` | V1 |
| V3 | `claude-code:7f3a2c10-0203-4000-8000-000000000203` | 5303, 700000 | started 13:00; `PreToolUse` Bash `cargo bench` 13:01; child PID 5330 started 13:01:02; child ended and `root_gone` at 13:40; no `SessionEnd` | `stale` (process gone at 13:40, no end event); at 13:30 it was `working` | — | — | none |

The running-now block at 15:00 lists V1 (under "waiting on you") and V2,
with the line `⚠ 2 agents in /home/op/morrow`. `check_facts.py` re-derives
these states at 15:00 and at 13:30.

## Hook spool

The main archive includes one hook spool file,
`hooks/carriers/claude-code/2026-10-03/4291-619650.ndjson`, named by the
hook process that wrote it (PID 4291, start 619650), so each file has one
writer; its lines carry the harness's identity (PID 4104, start 612000, boot
`7b1e0000-0000-4000-8000-00000000b003`). It holds
#4's `UserPromptSubmit` line for the prompt that pastes `7f3a2c10-0002`,
and the delivery line of the briefing the hook handed over in reply:

| Delivery ID | Recipient | Event | Refs | Outcome | Time |
|---|---|---|---|---|---|
| `01a101c9-1dc0-7000-8000-000000000d01` | #4 | `prompt_reference` (sender `hook`) | #2 | `delivered` | 10-03 14:42:00.180 |

This uses the old synchronous-hook format: the delivery envelope's
`observed_at_ms`, `1791031320180`, records the handoff, 180 ms after the
prompt and the clock used to make the UUID. Delivery time is never inferred
from the UUID. Current records carry a nullable `delivery.handoff_at_ms`;
delayed MCP and ping reports are not part of this fixture.

So `deliveries where outcome:delivered refs:claude-code:7f3a2c10-0002`
returns this one row, and #4 gains the `reference` link from #2 (Links,
above). The line's `content_blake3` is a placeholder, not a real hash.
The Claude-Code-only home has no hook spool. `check_facts.py`
verifies the delivery and the pasted prompt.

## Turns of #2 (joint #22)

Session #2's turns (`spec/query.md`, turns), as Sinex receives them in
backed mode. A turn's `id` is its first prompt's message ID; `end` is the
latest message time in the turn.

| Turn `id` (message) | `start` (local) | `end` (local) | Contains |
|---|---|---|---|
| `…0002:c2000000-0000-4000-8000-000000000001` | 2026-10-02 10:00:00 | 10:52:37 | the subagent call, the failing and the passing `cargo test` |
| `…0002:c2000000-0000-4000-8000-000000000013` | 10:56:21 | 11:07:32 | "let's use chrono, not time" |
| `…0002:c2000000-0000-4000-8000-000000000017` | 11:11:16 | 11:15:00 | the last prompt and answer |

## Annotation set

One agent's rubric pass over the Morrow sessions, in its own home
(`annotation/`) so the main archive's totals do not change; the annotation
tests and the story and judging boards load "Morrow plus the annotation set"
(`spec/user-data.md`, checked claims and annotation passes).

**The rubric** `note:0199f400-0000-7000-8000-000000000400`, revision 1,
written by the operator at 2026-10-08 09:00: instructions "Mark how each
Morrow coding session ended"; target query `sessions where tag:project:morrow`;
allowed: `outcome` with `solved`, `partial`, `abandoned`; `tag` with key
`failure` and values `test`, `build`, `lint`, `env`, `none`; `claim` with
free text and a required check.

**The pass**: Claude Code session `claude-code:7f3a2c10-0301-4000-8000-000000000301`
(2026-10-08 09:10–09:20, claude-opus-5-5) calls MCP `note` five times:

| Call | Target | Note | Result |
|---|---|---|---|
| `toolu_3001` | #2 | `outcome` `solved` | written, `note:0199f401…` |
| `toolu_3002` | #4 | `outcome` `partial` | written, `note:0199f402…` |
| `toolu_3003` | #5 | `outcome` `solved` | written, `note:0199f403…` |
| `toolu_3004` | #6 | `tag` `mood:curious` | **refused**: the rubric does not allow tag key `mood` |
| `toolu_3005` | #2 | `claim` "a recorded cargo test failure is followed by a recorded cargo test success", check `id:claude-code:7f3a2c10-0002 seq(command:"cargo test" outcome:error -> command:"cargo test" outcome:ok)` expecting rows | written, `note:0199f404…`; check state `holds` |

Every written note is authored by the annotator session and carries
`derived_from` → `note:0199f400…@1`.

**Blind judging** by the operator at 2026-10-08 10:00, sample of all three
outcome notes, authors hidden: #2 `solved` accepted; #4 `partial`
**rejected** (#4 ended with commit `4e1d2c9`; its unanswered last prompt
was a sign-off); #5 `solved` accepted. Agreement for the pass: 2 of 3.

## Unrecognized content

The Claude Code file for #2 contains one record of a type Polylogue does not
know, `type: "atis-latch"` (empty). It is counted under unrecognized, not
dropped.
