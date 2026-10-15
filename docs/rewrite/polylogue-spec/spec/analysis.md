# Analysis: facts Polylogue computes

Polylogue computes what code does well: counting, summing, joining, timing, costing, comparing and embedding. It never guesses what prose means.

Everything here is a function of the parsed archive (`model.md`) plus a few
data files, such as the price table. The writer computes it when it writes a
session, and the derive step of the loop (`convergence.md`) recomputes it
after a version or price change; results are stored in `index.db`. Cheap
things are computed at read time instead. Each derived product below has a
definition, its inputs, a version number in code (bumping it recomputes that
product and nothing else), and example values from the Morrow archive
(`examples/archive/FACTS.md`).

Deliberately left out: keyword or regex classifiers of "what kind of work" a
session was, heuristic "phases" or "work events" inferred from text,
confidence scores without a measurable basis, and an "insights" layer of
interpretations. Interpretation is done by models reading Polylogue's
output, and the result is stored as notes (`user-data.md`) carrying their
author.

## 1. Usage and cost

### Token accounting rules

These rules are fixed. Tests check them as laws (`plan/testing.md`).

1. **Copy what the provider reports.** Each message records input, output, cache read, cache write and reasoning tokens, under its model's canonical name: each provider normalizes its raw model string in one table beside its parser (`sources/README.md`, model names), and a model with no price is listed in `status` and its cost stays unknown. Sources: Claude Code `message.usage`, Codex `token_count` events, Gemini CLI `tokens`, Hermes `state.db` counters, AI Studio `tokenCount`. Nothing is estimated silently.
2. **Unknown is not zero, and only model requests have usage.**
   - Usage belongs to **request-bearing messages**, as `glossary.md` defines them. Other messages have no usage by nature; their absence is neither known nor unknown, and they never enter `unknown_usage_messages`, `unpriced_messages` or "unpriced models".
   - A request-bearing message whose provider reports no tokens has unknown usage.
   - Sums say "known total X, plus N messages with unknown usage", never just X.
   - Web chats (ChatGPT, claude.ai) usually have unknown usage.
3. **Cached input is part of input, never added on top.**
   - If a provider reports total input 100 including 30 cached, Polylogue stores uncached 70 and cache read 30.
   - Some providers report cache tokens separately from input (Anthropic does). Their fields are stored as reported, and the meaning of each field is recorded per provider, so totals are never double counted.
   - Each provider's usage mapping states, for every field, either `reported` (missing in a record means unknown) or `not_charged` (the provider bills no such category, such as OpenAI cache writes). A `not_charged` field is stored NULL, adds nothing to cost, does not make a message partial, and is shown as "n/a", not "—". Nothing else may substitute 0 for a missing field.
4. **Reasoning is part of output, never added on top.** `reasoning` is the part of `output` the provider reports as reasoning (Claude Code `thinking_tokens`, Codex `reasoning_output_tokens`). It is never priced on its own and never added to the other fields; a token total is input + cache read + cache write + output. A provider that reported reasoning outside output (none does as of 2026-10) would have its output stored as output plus reasoning, with the field meaning recorded as in rule 3.
5. **Per-request counts, not running totals.** Codex `token_count` events carry both the last request's counts and the running total for the session. An event is a new request only when its running total differs from the previous event's in the same input; an event that repeats the running total adds nothing (repeats come with rate-limit updates, turn ends and resumed streams: 7.7% of events, which would overcount tokens by 5.1%). The new request's counts are stored on that request's message, one message per model request (`sources/codex.md`); the running totals are not stored as usage. They are only a check: the sum of an input's per-request counts should equal its final running total, and a difference is shown in `status` for that input.
6. **One usage event counts once.** A message seen in both an export and a browser capture is one message (merged by ID). A message copied into another session (`inherited_from` set: a resume, a fork, a subagent started with a copy) contributes its usage once, to the session it was copied from. While that source session is missing from the archive, one copy counts, chosen by the counting rule (`decisions/lineage.md`); a copy of a message of an excised session counts nowhere, its usage being in the excision's `usage_json`. Totals a provider reports for a child (Claude Code `toolUseResult.totalTokens` on an Agent result) are evidence only, never added.
7. **Cost is what the transcript records.** `cost` is the API-equivalent price of the model requests recorded in the session's files. Model calls a harness makes without recording them (compaction summaries, titles, suggestions) are not in it. Where the provider reports its own total (Claude Code `cost-state`), it is kept beside Polylogue's computed cost, never merged with it, and session views show both and their difference, named "not in transcript".
8. **Every money figure names its basis**, from one list (operator decision, 2026-05-07, `polylogue-cost-model-contract` and `subscription-cost-primary`): `provider_reported` (a total the harness or provider recorded: Claude Code `cost-state`, Hermes's cost columns), `api_equivalent` (the list price of the recorded tokens, the default `cost`), `subscription_share` (the plan's price for a window spread over the requests in it, in proportion to their API-equivalent cost, only where the plan setting gives a price; "what this session cost me" for a subscription user), `manual` (an amount the operator entered), `estimated` (a figure computed from incomplete data, with what it assumed) and `unknown`. Figures of different bases are never added together; a view that shows several shows each under its own label.

### Price table

Prices are data, not code. They live in `prices.toml`, which ships with
Polylogue with the providers' published prices (updated by hand) and can be
overridden in `$XDG_CONFIG_HOME/polylogue/prices.toml` (decided by the spec
authors, 2026-10-11; the operator may overrule). One row per model and date,
USD per million tokens:

```toml
[[model]]
name = "claude-opus-5-5"
effective = "2026-09-01"          # applies to messages on or after this date
input = 4.00
output = 20.00
cache_read = 0.20
cache_write_5m = 5.00
cache_write_1h = 8.00
web_search_per_1000 = 10.00
fast = { input = 8.00, output = 40.00 }        # when the message's speed is "fast"

[[model]]
name = "claude-haiku-5-5"
effective = "2026-09-01"
input = 0.10
output = 0.50
cache_read = 0.01
cache_write_5m = 0.125
cache_write_1h = 0.20
long = { above = 100000, input = 0.50, output = 2.50, cache_read = 0.05, cache_write_5m = 0.625, cache_write_1h = 1.00 }

[[model]]
name = "gpt-6.1-sol"
effective = "2026-08-15"
input = 2.00
output = 10.00
cache_read = 0.10
long = { above = 272000, input = 4.00, output = 15.00, cache_read = 0.20 }

[tiers]                            # multipliers by recorded service tier
batch = 0.5
flex = 0.5
```

Other shipped rows as of 2026-10-11 (input / 5-minute write / 1-hour write /
cache read / output): Fable 5.1 10 / 12.50 / 20 / 0.25 / 50; Sonnet 5.5
2 / 2.50 / 4 / 0.10 / 10; Haiku 4.5 1 / 1.25 / 2 / 0.10 / 5
(`evidence/research-platforms.md` has the sources).

**Message cost** (computed by the facts product, stored in `message.cost_nusd`):

1. take the row for the message's model with the latest `effective` on or
   before the message's time;
2. choose the `long` tier when the message's prompt tokens (input + cache
   read + cache write) exceed its `above`;
3. use `fast` prices when the message's `speed` is fast;
4. tokens × prices, with cache writes priced by lifetime from
   `cache_write_5m_tokens` and `cache_write_1h_tokens`; any part of
   `cache_write_tokens` with no reported lifetime is priced as 5-minute, the
   message is counted in `assumed_cache_lifetime_messages`, and cost
   figures that include such messages say "N requests priced assuming
   5-minute cache writes" (in this archive no 1-hour writes predate the
   records that split lifetimes, so the assumption changes no total);
5. add web requests × `web_search_per_1000` / 1000;
6. multiply by the `[tiers]` factor when a service tier is recorded.

Anything the rule needs and does not have (no row for the model, a tier with
no factor, a mode with no price) leaves the message's cost unknown; a
`not_charged` token kind needs no price and contributes zero, so it never
makes a cost unknown. Otherwise a missing input leaves the cost unknown, never
guessed, and output lists what is missing so the operator can add it.

- **Recomputed when prices change.** The facts product's version includes
  the hash of `prices.toml`, so editing prices recomputes every message and
  session cost in the derive step (about 50k sessions, seconds), without
  re-parsing. Session and per-model totals (`session_facts.cost_nusd`,
  `session_model_usage.cost_nusd`) are sums of known message costs, with
  `unpriced_messages` counting the rest. A displayed total leads with its
  unpriced share whenever there is one ("$249,653 priced; 603M output
  tokens of Codex usage unpriced, no prices yet"), never a bare sum: on the
  operator's archive the unpriced part was all of Codex
  (`evidence/vision-prototypes.md`).
- **One cost function.** The rule above is one function of a message's
  usage and the price table. Every other cost (session, model, tree, day,
  window, report) is a sum of its results, never a second formula; a drift
  test checks on the Morrow archive that the stored sums equal the sums of
  the function's per-message results.
- **Exact amounts.** Money is integer nano-USD; rounding to cents happens
  only for display.
- **Subscription sessions** (Claude Code on a Max plan, ChatGPT Plus):
  - Cost is the API-equivalent list price of the tokens, labelled as such.
  - List price is never presented as a bill.
- **The Morrow example archive has its own `examples/prices.toml`,**
  deliberately invented and separate from the shipped table, so a real
  price change never moves the example totals ($6.20).

### Subscription windows

Subscription plans limit usage per rolling window (Anthropic: 5-hour windows
and a weekly allowance; OpenAI: similar for Codex). Polylogue shows what it
can know and labels the rest. For a subscription user the **usage outlook
leads cost reporting**: the current window's usage, the burn rate and how
far the allowance will last come first in `status`, the overview and
`analyze --cost`; API-equivalent dollars follow as a comparison, labelled as
such, never as money spent. Subscription credits and USD are different units
and are never added together.

- **Window boundaries.** Windows are per provider account and count own
  request-bearing messages of that provider's models. A boundary the
  harness reports (Codex `rate_limits` `resets_in_seconds` and
  `window_minutes`, Claude Code reset times in limit records) fixes a
  window. Only between reported boundaries is a window computed: it starts
  at the first such message after the previous window ended and lasts the
  plan's window length (5 hours for Anthropic). A computed boundary is
  labelled `inferred` ("computed from this archive's messages", `≈` in the
  terminal). Windows are computed at read time from every stored message and
  never stored, so a late import (an older export, a session collected late)
  re-derives every later computed window on the next read; nothing is kept
  per window that could miss it. Logical-day groupings are read-time too. The weekly allowance runs from the provider's reported weekly
  reset in spans of 7 × 24 hours; it is not a calendar week.
- **Usage per window.** For each window: start, end, tokens by kind,
  API-equivalent cost, sessions active. Hidden sessions count, with their
  number shown: the provider counted their usage (`user-data.md`, hiding). A window holding request-bearing
  messages with unknown usage, or any web-chat session of the same
  provider, shows its usage as "at least X (N messages unknown)".
  `polylogue analyze --windows` lists them; `polylogue status` shows the
  current one.
- **Billing mode.** A session counts against a subscription only with
  evidence: a rate-limit or plan record in its own files, or the plan
  setting in force at its time (a setting keeps the date it was set and
  applies from then on). Otherwise its billing is `unknown`, shown as such;
  "API-equivalent" labels its dollars either way.
- **Limits reached.** When a harness records a usage-limit message or error
  (Claude Code `rate_limit` / usage-limit system records, Codex
  `rate_limits` in `token_count` events), it becomes an event; windows show
  when a limit was hit and what the harness reported as remaining.
- **Plan settings.** `polylogue setting subscription.anthropic max-20x`
  names the plan, using the provider's own plan names (Anthropic `pro`,
  `max-5x`, `max-20x`; OpenAI `plus`, `pro`); a bare `max` is refused as
  ambiguous; the window length per plan is in the shipped price table
  file, next to prices, updated by hand. Allowances the provider does not
  publish are not invented: usage is shown against a limit only when the
  harness itself reported the limit (Codex reports percentages; Claude Code
  reports reset times). "How far the allowance will last" is shown only
  when the harness reported a remaining share; otherwise the line is
  absent.
- Exact (tokens reported per message), estimated (none: Polylogue does not
  estimate tokens for analysis) and unknown (web chats without counts) stay
  separate in every total.

### Morrow examples

| Session | Cost (USD) |
|---|---|
| #1 claude-code `…0001` | 1.26 |
| #2 claude-code `…0002` | 2.85 |
| #3 claude-code `…0002/a7f3a2c1000000003` (subagent) | 0.16 |
| #4 claude-code `…0004` (own messages only) | 1.59 |
| #5 codex `…0005` | 0.22 |
| #6 codex `…0006` | 0.12 |
| #7 chatgpt, #8 claude-ai | unknown |

- **By model:** claude-fable-5-1 5.70, claude-opus-5-5 0.16, gpt-6.1-sol 0.33.
- **Archive:** **6.20 USD known; 2 sessions unknown.**
- **Known tokens:** input 53,100, output 56,100, cache read 1,940,000, cache write 155,000.
- **Inherited messages:** session #4's 18 inherited messages contribute nothing to these totals.

## 2. Tool calls

Every tool call is a row, so an agent's tool use becomes the equivalent of shell history: searchable and countable. Each row holds:
- the session, agent and model;
- when it happened;
- the tool and its input, plus the shell command if it is a shell tool;
- the matched result: outcome, exit code and duration.

### Outcome rules

- **`ok` / `error`** come from the provider's explicit flag (`is_error`, Codex exit status) when present.
- **Derived outcomes:** otherwise the outcome comes from the result itself, such as an exit code in the output text or a provider error object.
- **What outcome means.** `ok`/`error` is the harness's verdict on the call, not whether the command did what was wanted. Where a harness treats a non-zero exit as success (Claude Code `returnCodeInterpretation`), the outcome stays `ok` and the result carries `interpreted_exit`. Failure rates across harnesses are computed on `exit_code != 0` and on `outcome = error` separately, never mixed. A pipeline's outcome is its last stage's.
- **Dispatch is not a result.** Claude Code background tasks report `is_error: false` on dispatch, and an `Agent` call whose result is `async_launched` is only a launch acknowledgement; neither is success. The outcome comes from the later notification naming the task or agent (`sources/claude-code.md`).
- **`pending`** and `unknown` for calls without a result: `model.md`, Block (tool outcome).
- **Wrapper calls:** a call that runs several child commands (Codex `exec`) gets its own outcome and does not inherit a child's.
- **`unknown`** is used only when the input says nothing, and always with a reason (a reason code from `model.md`, for example `no_result`, shown as "no result").

### Shell commands

For shell-like tools (Claude Code `Bash`; Codex `shell`, `shell_command` and `exec_command`, including the children of a code-mode `exec` wrapper; Gemini CLI shell), the command is stored as given, plus a normalized form for grouping:
- **The normalized command** is the one key every statistic, briefing and `command_norm` uses. The line is split as POSIX shell words, outside quotes (so a `jq` filter or a Python one-liner stays whole), into stages at `&&`, `||`, `;`, `|` and `&`. In each stage, leading `NAME=value` assignments and the wrapper programs (`env`, `sudo`, `timeout`, `nice`, `ionice`, `nohup`, `time`, `exec`, each with its own options) are skipped. The normalized form of a stage is the program's base name, plus the subcommand for the multi-command programs (`cargo`, `git`, `nix`, `npm`, `pnpm`, `yarn`, `uv`, `go`, `docker`, `systemctl`, `bd`, `gh`, `kubectl`, `just`; `git -C dir status` → `git status`), plus the flag that makes a reading program write (`sed -i`, `find -delete`, `find -exec`), so a writing run is a different command from a reading one. A loop, conditional or heredoc is `shell script`. Arguments are kept separately.
- The call's normalized command is its **last** stage's, whose exit status the shell reports; `cd` and `pushd` stages count as stages only when nothing follows them. Taking the first word of the whole line instead gave keys such as `-f`, `for id` and `(env` on real sessions, and the first stage made 30% of Bash calls `cd` (`evidence/briefing-trial.md`). No meaning is inferred.

### Failure statistics (computed, not interpreted)

"Later" within a session means later in session order (message `seq`, then
block index). Across sessions (briefing failures, "no later `ok` run in the
same repository") it means a later `at`, and a call without `at` is shown
with that reason.

- **Repeated failing commands:** the same normalized command failing more than once in a session, with the count and the time between attempts.
- **Error then fix:** a failing command later followed in the same session, in the same checkout, by a call whose last-stage normalized form and arguments are equal and whose outcome is `ok`, with the edit-tool calls made in between.
- **Failure rate:** by tool, model, repo and harness, over time, reported as bounds `[error / n, (error + unknown + pending) / n]`, where `n` counts every selected call, with `n`, the error count and the unknown and pending counts. A group with `n = 0` is unmeasured. No single figure is shown when the bounds differ. This is the shipped "tool error rate" (§11).
- **Calls without results:** tool calls whose result never appeared, because the session ended or crashed.

### Morrow examples

Tool calls: 37 in total; 2 errors, 0 pending, 0 unknown.

| Session | Tool calls | Errors |
|---|---|---|
| #1 | 6 (Bash 3, Write 2, Read 1) | 0 |
| #2 | 13 (Bash 5, Edit 4, Read 3, Agent 1) | 1 |
| #3 | 4 (WebFetch 2, Read 2) | 0 |
| #4 | 8 own (Bash 5, Edit 3) | 1 |
| #5 | 3 (`exec_command` 3, each the child of a code-mode `exec` wrapper, which is not counted) | 0 |
| #6 | 3 (`exec_command` 2, `apply_patch` 1) | 0 |

Failing commands:

| Session | Command | Exit | Later success of same command |
|---|---|---|---|
| #2 | `cargo test` (test `parse_dates::tomorrow_at_noon`) | 101 | yes, after 1 edit to `src/parse.rs` |
| #4 | `cargo clippy` | 1 | yes |

Shell commands across the archive: 18 (13 Bash calls in Claude Code, 5 shell calls in Codex).

## 3. Time

Everything here comes from timestamps; nothing is estimated. Every time
figure covers the session's own messages and events only; inherited copies
keep their original session's times. Calendar spans (days, weeks, months)
use calendar arithmetic in the configured time zone, never a fixed number of
seconds; a row without a time is reported as such, never placed in 1970
(`query.md`, rows without a time).

- **Wall-clock span:** last own activity minus first own message or event.
- **Engaged time:** the length of the union of intervals in which the
  session was active. An interval joins consecutive own messages and events
  on the active path, in session order (`seq`), when the later time is
  strictly later and at most 5 minutes after the earlier; it always spans a
  tool call (or subagent call) to its result, however long that took.
  Equal and reversed gaps add nothing and are counted as `unordered_gaps`.
  The 5-minute constant is fixed in code, shown in the output and not
  configurable. A session with fewer than two timed own messages has
  unknown engaged time, never 0.
- **Where the time went** (agent sessions), from tool-call and result timestamps and durations:
  - running tools, split by normalized command (`cargo test` 12 min, `cargo build` 3 min);
  - waiting for the model: from a human prompt or tool result to the next model message;
  - waiting for the human: from a model message that ends the turn to the next human prompt.
- **Harness-reported duration** (`durationMs`, Codex task timing) is kept separately.
- **Turn latency** (the `turns` unit, `query.md`): from a turn's last human
  prompt to the first model message after it.
- **Summed time is not elapsed time.** The sum of several agents' or tool
  calls' durations, the wall-clock time they covered together (the union of
  their intervals), and their overlap are three different numbers, each
  shown under its own name. None of them says how much time a change would
  have saved: that needs evidence of what waited on what, which intervals
  alone do not give.
- **Durations need two different times.** A duration (tool call to result,
  session length) is computed only when both times exist and the later is
  strictly later. Equal or reversed times give `unknown` with the reason
  (`same timestamp`, `result before call`), never 0 or a negative number.
- **Unfinished is not never.** A session or call with no observed end is
  unfinished as of its last observation: its duration is censored there and
  shown apart from completed ones, never averaged in as if it had ended, and
  nothing is labelled "never completes" from a finite capture
  (`polylogue-9l5`).

Example: session #2 spans 10:00–11:15, 75 minutes of wall clock. Its engaged time and activity split come from the message timestamps in the example files and are recorded in slice 1's expected outputs.

## 4. Session facts

There is one row per session in `index.db`, recomputed when the session changes. List views, sorting and most queries read these rows, so they are stored rather than computed on every read.

The table definitions (`session_facts`, `session_model_usage`) are in
`storage/`; this section says what the fields mean.

- **Which messages count.** Message and prompt counts (`messages`,
  `human_prompts`, turns) cover the active path; a message whose path
  status is unknown is counted. Usage, tokens, cost and tool calls cover
  every own request-bearing message on or off the active path, because
  abandoned branches were run and billed. `has:branches` says when the two
  sets differ.
- `messages` counts the messages counted as the session's own (`counted_as` is null; `storage/index-db.md`, session_facts);
  `human_prompts` counts own `written_by = human` messages with
  `prompt_kind = prose`, and `human_commands` those with `prompt_kind =
  command`.
- **Complete, partial, unknown.** A session's cost, and each token sum, is
  complete when every request-bearing message has a known value, partial
  when some do, and unknown when none do. The stored value is the known
  part (NULL when unknown); `unknown_usage_messages` and
  `unpriced_messages` count the request-bearing messages without a value,
  so a sum reports "known X, N messages unknown" whenever they are above
  zero, and a partial value is a lower bound (`query.md`, partial values).
- `display_title` follows the one rule in `model.md` (Session, display
  title).


Morrow, session #4:

| Field | Value |
|---|---|
| `messages` | 10 |
| `inherited_messages` | 18 |
| `human_prompts` | 3 |
| `compactions` | 1 |
| `tool_calls` | 8 |
| `tool_errors` | 1 |

## 5. Lineage marks and tree cost

The writer computes these in the session's own transaction; the derive step
recomputes them only when a parent arrives late or a version changes
(`convergence.md`; rules: `decisions/lineage.md`; evidence per provider:
`sources/`):

- **Inherited messages.** `message.inherited_from` names a copy's source in
  the writer from the record's own evidence and by the derive step by
  matching; `decisions/lineage.md` (when marks are set) lists every
  mechanism. It names the source whether or not that message is in the
  archive.
- **Late parents.** A link whose parent is not yet ingested is pending; when
  the parent arrives, the child's marks and counts are recomputed. Until then
  the child shows "parent not in the archive".
- **Counting.** Every count, sum and search follows `counted_as`: a copy
  counts as its session's own only while it is the copy the counting rule
  chooses for an absent source (`decisions/lineage.md`), and tool calls, usage
  and events inside a message follow it. `inherited:*` includes them (`inherited:true` selects only copies).
- **Own cost and tree cost.** `cost` is a session's own cost (the
  messages counted as its own, `counted_as` null). `tree_cost` is its own cost plus the `tree_cost` of
  every session it spawned (`subagent` links only). A session has at most one
  incoming `subagent` link, so each session is counted exactly once in exactly
  one tree. `tree_unpriced` counts the sessions in the tree whose own cost is
  partial or unknown, plus each subagent call in the tree whose child session
  is not in the archive; a tree cost with `tree_unpriced > 0` is shown as "at
  least X", with the spawning-call references of the missing children.
- **Continuations are not trees.** The cost of a chain of resumes and forks
  is `sessions where lineage:<id> | agg sum:cost`, each session's own cost once.
- **Roots and depth.** Each session's subagent root (follow incoming
  `subagent` links up) and its depth, stored on `session_facts`, used by
  `root:`, `depth` and `lineage:`.

Morrow (main archive):
- #4's first 18 messages have `inherited_from` pointing at #2's messages.
- #3 links to #2 as `subagent`, `how=fresh`. Its meta file has no
  `toolUseId`; the spawning `Agent` call (`toolu_0203`) is found from #2's
  side, by `toolUseResult.agentId` on that call's result.
- #6 links to #5 as `continuation`, `how=fork` (`forked_from_id`; #5 is not a
  spawned thread).
- Tree cost of #2: 2.85 + 0.16 = **3.01** (the subagent's cost appears once,
  in #2's tree; the Oct 02 row of cost by day shows the same 3.01).

The Morrow lineage set (`examples/archive/FACTS.md`, "Lineage set") has the
harder cases with their own and tree costs: a nested subagent, a sibling
copy, an explicit fork, a team, a missing parent.

## 6. Format drift

Providers change file formats without notice. Two cheap mechanisms catch it:

- **During parsing.** Every unrecognized record type and field is counted in `index.db` `unrecognized`, and so is every record type that is deliberately only counted (`counted_only`, such as Claude Code `progress`). `polylogue status` lists the unrecognized entries, newest first by first-seen date, and the count-only types separately:

  ```
  Format changes:
    claude-code  record:atis-latch      1 time   first seen 2026-10-02
  ```

  This comes from Morrow: the one unknown record in #2.
- **Most breaking first.** `status` lists a field whose type changed above a new field,
  and a new field above a new value of a known field: a changed type can
  break a parser, a new field cannot.
- **Field coverage.** For each provider, which fields of the inferred schema (section 7) the parser reads, and which it ignores. Shown by `polylogue schema unread claude-code`. A field nobody reads is either deliberately ignored, listed with a reason in the parser, or a gap.

## 7. Schema inference (offline)

`polylogue schema infer` reads the stored inputs and builds a JSON schema of what has actually been observed, per provider and record type. It records field names, types, how often each field is present, value ranges, enum values for fields with few distinct values, and first and last seen dates per field.

- **Privacy filter.**
  - A string value is kept as an example or enum member only if all hold: it occurs in at least 20 distinct inputs; the field has at most 50 distinct values; the value is at most 64 characters and contains no whitespace run longer than one space; it is not under `/home/` or `/Users/`, not an e-mail address, URL with a query string, or a string of 20+ characters from a base64/hex alphabet.
  - Everything else is recorded only as a type and length range. The thresholds are constants in the inference code, listed in its report header.
  - The output lists the filter's decisions so the operator can check them.
- **Output.** One schema file per provider record type in `$XDG_DATA_HOME/polylogue/schemas/`, plus a report of fields new since the previous run, fields no longer seen, and fields the parser does not read.
- **Uses.**
  - Synthetic provider files for tests and benchmarks that match the real distribution (`plan/testing.md`, layer 3).
  - Parser coverage (section 6).
  - Spotting format changes in bulk.
- **Objects that are maps.** An object's keys are treated as data (a map),
  not field names, when any key carries content (longer than 128 characters,
  a control character, or one of `? < > / \ @`), when it has 256 or more
  keys, when half or more of its keys look generated (a UUID, 24 or more hex
  digits, or a `msg-`, `node-`, `conv-`, `item-`, `att-` ID), or when it has
  24 or more keys and 35% or more look like paths. A map is recorded as one
  key type and one value schema.
- **It does not run during ingest.** Typed parsing already rejects malformed input and counts unknown fields, at no extra cost.

## 8. Embeddings and semantic search (opt-in)

- **Enabling.** `polylogue setting embeddings voyage-4`, with its API key stored by `polylogue auth embeddings` (credentials are state, `principles.md`, kept in `$XDG_STATE_HOME/polylogue/credentials/`, never read from the environment), or a local model served through an OpenAI-compatible endpoint. Voyage is the first provider offered, since the purchased vectors are Voyage's; embeddings stay off until configured. Decided by the spec authors from the existing purchase (2026-10-11); the operator may overrule.
- **What gets embedded:** distinct texts of human prompts, model prose and
  notes (claims, decisions, findings, lessons: short, and written to be
  found again). Tool output and thinking are not embedded. Each distinct
  text is embedded once, keyed by its hash and the full embedding key
  (provider, model, dimensions, input type, normalization), so the same text
  in many places shares one vector without merging the places it occurs
  (`embeddings.db`). Changing the model is a deliberate change of setting;
  vectors of the old model stay.
- **Cost check first.** `polylogue embed estimate` reports how many texts would be embedded and the provider's cost before anything is sent. This is the only outside model call Polylogue makes (other destinations of archive data are listed in `behavior.md`, privacy), and it says so.
- **When it runs.** Embedding runs in the derive step, in batches, for new texts only.
- **What it powers.** `near:` and hybrid search (`query.md`): scope first, then ranking; `near:session:` and `near:message:` from stored vectors without the provider; hybrid fusion by reciprocal rank. A session ranks by its best-matching passage.
- **Checked against plain search.** Before semantic search is the default anywhere, it is compared with full-text search and reference-following on rare details, minority topics, other languages and repeated prefixes in the operator's own archive; it becomes the default only where it finds more.
- **Old vectors.** Vectors bought by the Python version are imported once, by matching text hashes (`plan/migration.md`).
- **Rejected texts.** A text the provider permanently rejects (HTTP 400 and
  similar) is recorded once with the provider's error (`embed_failure`,
  `storage/`), listed in `status` with its reference, and retried only
  when the embedding key or the text changes; it is never retried every round,
  and never silently skipped.
- **Model family reuse.** Vectors are reused across models only where the
  provider documents one shared embedding space (the Voyage 4 family);
  otherwise a model change embeds again. The configured model is used for
  every computation and comparison.

## 9. Git correlation and the changes view (slice 11)

What a session did to code, from evidence in the session and the
repository's own history. Nothing is re-run, and nothing is linked by
timing or similarity alone.

- **Commits made in a session.** A shell tool call whose command runs
  `git commit` (or `git commit` inside a compound command) with outcome
  `ok`, and whose output contains git's summary line
  (`[branch abc1234] message`), links the session to that commit. Codex
  `git` blocks and Claude Code `gitBranch` give the branch.
- **Pull requests.** Claude Code `pr-link` records; `gh pr create` output
  containing a PR URL; Codex records with a PR URL.
- **Commit details.** For each linked commit, the collector reads the
  repository at the session's `cwd`, read-only (`git show --stat` through
  the `gix` crate, under a deadline), to fill message, files and line
  counts, and stores each lookup as a `git-lookup` record: an input with one
  revision per lookup, like the repository lookup record
  (`sources/README.md`). Trailer scans are stored the same way. Derive reads
  only stored records, so a rebuild gives the same links; a repository that
  no longer exists produces no new record, and links keep what earlier
  records hold.
- **Commit trailers.** A commit in a repository the operator has sessions in
  whose message carries a session trailer naming a local session ID is
  linked to it. (Trailers carrying only a web session URL are kept as text;
  they are not mapped by guessing.)
- **Files edited.** File tool calls (`Edit`, `Write`, `apply_patch`, …)
  grouped per path: number of edits, and the cumulative diff when every
  edit's old and new text are recorded. A failed edit is listed too, marked
  "failed, may have changed the file": a failure exit does not prove nothing
  was written (a patch can apply partly). The state of a file at a moment is
  therefore graded: **known** (from a successful edit with recorded text),
  **possible** (a failed edit may have changed it), or **unknown**.
- **The `changes` view** (`rendering.md`) shows, per session: files edited,
  commands run with outcomes (test and build commands first, recognized by
  the command name list in the provider code), commits and PRs.
- **A commit's span.** For a session's commits in call order, a commit's
  span is the session's own messages after the previous commit's call (or
  from the session's start) up to and including its own commit call. Its
  tokens and cost are the sums over that span, unknowns counted as in §1,
  so "tokens per commit" is a read-time sum, never a stored figure. A commit
  made outside any session has no span (`vision/uses.toml`,
  `tokens-per-commit`).
- **Tables:** `commit_link` and `pr_link` (`storage/index-db.md`).
- **Queries:** `sessions where commit:abc1234`, `sessions where pr:42`,
  `files where path:src/parse.rs | group by session`.
- Later (slice 12): reproducing a session's change (check out the base
  commit in a scratch worktree, apply the recorded edits, run only the
  session's own verification commands) and replay (play the session back
  in order, reconstructing files only from known bases and successful
  edits).

Git output is read exactly: quoted file names (`git` quotes names with
spaces, non-ASCII characters or newlines) are unquoted, commit hashes match
at any prefix length git accepts (short hashes in agent output), and a
working directory with spaces is one path; both the checkout path and the
repository-relative path are kept.

## 10. What a record proves

A provider field is evidence only for what it actually records:

- An exit code says how a process ended, not that the task was done. A
  passing command on the wrong revision proves nothing about the right one;
  a failing test can be the intended deliverable (an expected failure that
  demonstrates a bug). Whether a task was accepted is a judgment note.
- What a harness reports about a task (a run's `completed`) is the
  harness's statement, kept apart from observed effects and from judgment
  (`model.md`, the work graph).
- Several records derived from one observation are not independent
  confirmation: a claim repeated in five summaries is still one claim.
- A model's later assessment can prompt a useful notice, but it is never
  counted as a new observation of the event it assesses.
- **What can back a statement**. Some statements need a particular kind of
  record, and briefings, findings and reports follow that when they cite
  evidence: "a PR was merged" is backed by the forge's record, a git commit
  or a tool result showing the merge, never by an agent's sentence saying
  so; "the assistant said X" by the transcript itself; "a command passed" by
  the tool result and its exit code; "a decision was made" by a human
  message or a judgment. This is guidance for whoever writes the citation,
  not a checking mechanism.

**Transported claims retain their source.** When a briefing, quote or generated finding explicitly carries source references, preserve those links through delivery and reingestion. Group linked restatements by their retained evidential roots when reporting independent support; a new file, session, actor or content hash does not make another observation. Keep transported assertions, actor acceptance and genuinely new checks distinguishable. Missing transport links leave independence unknown; this is not a semantic-copy detector and does not discard an unlinked statement as false.

Cross-system acceptance remains unrun: send a Sinex finding through a Polylogue briefing, quote it in an agent reply and reingest that reply five times. Independent support must remain unchanged while the linked statements stay inspectable. Add an actual new test observation and retain it as new evidence. A later qualification or contradiction remains visible even when the numeric finding is unchanged. Signal delivery and reingestion must also respect the declared input scope (`sinex-spec/spec/surfaces.md` §6a).

## 11. Measures and their definitions

Wrong figures usually come from a wrong definition, not wrong arithmetic,
so every figure Polylogue reports names its definition.

- **A figure's definition is a saved query** with a unit, a grain (per
  message, session, turn, task) and a denominator, and a stated rule for
  unknown values (`user-data.md`, saved queries). There is no separate
  registry. The shipped figures (cost, tokens by kind, tool error rate,
  turn latency) are saved queries that ship with Polylogue, from slice 3,
  each with its unknown rule: cost and tokens as complete, partial or
  unknown (§4); the tool error rate as bounds over unknown and pending
  calls (§2); turn latency over turns with both times.
- **Every reported figure** (an `analyze` row, a kept result, a report)
  names the query that produced it and how many values were unknown or left
  out. Those are ordinary result fields (`query.md`, results), not a wrapper
  around every number.
- **Comparing figures.** Figures from different definitions are shown as not
  directly comparable. The same definition over differently selected
  sessions, or over a different capture period, is labelled as such. Unit
  conversions (milliseconds to seconds) are explicit and allowed.

**Outcomes and forecasts:**

- "Did the task finish?" is asked by a stated horizon over the period the
  archive captured. Abandoned (an `outcome` note), superseded, withdrawn,
  not captured and still open at the cutoff are separate answers; "will
  never finish" is not an answer finite history can give.
- With a known population and some outcomes unresolved, rates are reported
  as bounds: 70 solved, 10 failed and 20 unknown out of 100 is a solved rate
  between 0.70 and 0.90, not 0.875 with the unknowns dropped.
- Calibration (a later slice, with the other research features of question
  #11) will use only `forecast` notes written before the event resolved.
- **Repeated corrections** are counted per opportunity: the times an agent
  had new information (a failing result, a correction) and acted again, not
  per message or session. Three edit-and-fail cycles that try three
  different fixes are not the same as three identical retries, and neither
  is labelled "thrashing" by a count alone.
- **Comparing sessions or agents** shows a set of components (cost,
  duration, errors, outcome, corrections), each with its own denominator,
  unit and unknown count. There is no single quality score; a reader who
  wants one defines its weights and unknown rule explicitly, and they are
  recorded with the result. Hiding a component never makes the rest look
  better.
- **API-equivalent cost is not an invoice.** Prices times tokens is what the
  usage would have cost at list price; on a subscription it is not what was
  charged. Subscription usage comes from the provider's own window data
  (§1).

## 12. Compaction

When a harness compacts a session:

- **Snapshot.** When the harness's pre-compaction hook carries the
  assembled context, the hook spool keeps it as an input linked to the
  session; otherwise the transcript up to the compaction boundary stands in
  for it.
- **What it dropped.** For each `compaction` event, the references present
  before it that neither the summary nor any later message mentions again:
  human prompts (by message ID), decision and question notes, file paths in
  tool calls, session and message IDs. These are exact matches on stored
  identifiers and paths, never a judgment of prose. They are computed when
  `read --view context` or the after-compaction briefing needs them, from
  stored rows; nothing extra is stored.
- **Kept apart:** something left out of the summary (an omission), meaning
  kept in other words, something the agent later visibly did not know,
  something mentioned again later, and harm to the task are different
  findings. Only the first is computed; the rest are questions for a reader
  or a model, recorded as notes if answered. A restoration delivered after
  a compaction is not proof it survived the next one.
- **Restoration.** The after-compaction briefing (`user-data.md`) restores
  the dropped items first.

## 13. Reports built from the above

These are queries plus rendering, not separate subsystems. Each can be saved as a saved query (`user-data.md`).

| Report | Built from |
|---|---|
| Daily or weekly activity summary | sessions grouped by day: counts, cost, top repos, failing commands |
| Cost over time by model or origin | per-message cost grouped by day |
| Agent behavior | tool calls grouped by tool, model and outcome, plus failure statistics |
| Where time went | the time breakdown for sessions in a scope |
| Resume candidates for a repo | see `user-data.md` |

Report ideas from the beads fit the same pattern: an aggregation and a rendering, added when the operator wants them, with no new machinery. Examples: pathology rates over time, cost per finished task, cross-provider comparisons, survival curves.

Bead and audit IDs behind each section: `evidence/beads/spec-sources-g3.md`.
