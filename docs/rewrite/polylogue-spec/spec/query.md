# Query language

One expression language finds things in the archive. The CLI, the MCP
server and the web reader all accept the same text and return the same rows.
Exports, "views" and reports are a query plus a rendering
(`surfaces/README.md`); there is no separate filter vocabulary anywhere.

The language is the one the current Polylogue already has, which the
operator asked to keep and extend. This file keeps its surface, removes the
places where it had two spellings for one thing, and maps its fields onto the
new data model (`model.md`).

## Principles

1. **One meaning everywhere.** No surface re-implements filtering, totals or
   paging. The `query` crate parses the text into a plan, compiles it to one
   SQL statement against `index.db` (`storage/index-db.md`); the daemon runs
   it for every surface (`surfaces/README.md`, processes), and each surface
   renders the rows.
2. **Errors point at the problem and give the fix.** An unknown field, a
   bad value or an ambiguous name is an error with a position in the query
   and, whenever one can be formed, a corrected query that runs as given
   (`suggestion`): unbalanced quotes closed, a pasted ID or provider URL
   turned into its reference, a misspelled field replaced. A query never
   falls back to searching its text as plain words, and never widens its
   scope. The reverse holds too: an unquoted query whose first word is a
   command or verb name, or a near miss of a unit, command or verb name
   (one typing mistake for a word of 4–6 letters, two for 7 or more, none
   under 4 letters, counted as Damerau–Levenshtein distance on the typed
   word so a swapped pair of letters is one mistake; `date parsing` and
   `bug fix` are searches), is refused with a suggestion rather than searched for
   (`messges cargo` → `did you mean "messages"?`; `brief yesterday` →
   `"brief" is a command, not a query`), since a silent empty search is a
   wrong answer; quoting the word, or `find`, searches for it. A unit name
   starts a query (`sessions | group by origin | count`), the references
   `last`, `latest` and `self` are never near misses, and a field
   name (`cost > 2.90`) is never taken for a search word (law 135; the joint
   query law "command words"). Agents call search more than anything else, so every string an
   agent might pass (bare words, an ID, a URL, a quoted phrase, unbalanced
   quotes) returns rows or a corrected query, never an unexplained error.
3. **Scopes are strict.** `repo:morrow` returns only Morrow sessions. If
   nothing matches, the result is empty (with an explanation on request), not
   "the closest thing from somewhere else".
4. **Unknown is not zero.** A session with unknown cost does not match
   `cost < 1`, and totals report the known sum plus how many values were
   unknown.
5. **Defaults are visible, overridable and bound once.** Each unit has a few
   default filters (below). They are resolved when the query's source is
   bound, before anything composes with it: a default is dropped only by a
   top-level condition on its field, and nothing composed afterwards can
   change which defaults apply (Defaults, binding). Naming the field only as
   a group key or sort key (`| group by action`) keeps the default (so query
   example 17 counts 37 calls, not 42 with the Codex wrappers).
6. **Discovery has no side effects.** `explain QUERY` counts the rows each
   clause keeps, read-only and within the read time limit, and never writes,
   builds an index or calls a model; completion and error suggestions never
   run searches at all.

## Shape of a query

```
[UNIT where] CONDITIONS [| STAGE]...      e.g.  tool_calls where outcome:error | group by session | count
CONDITIONS [with ATTACHED]                e.g.  origin:codex with messages(id,role)[role:assistant, last:1]
SESSION-QUERY | UNIT where CONDITIONS     e.g.  tag:project:morrow | tool_calls where action:shell
```

- Without `UNIT where`, the unit is `sessions`.
- Conditions separated by spaces are joined with AND. `AND`, `OR`, `NOT`
  and parentheses are available everywhere (not only after `sessions
  where`, as in the old language).
- A session query followed by `| UNIT where …` scopes the second query to
  the matched sessions.

## Grammar

```ebnf
query        = [ unit [ "where" expr ] | expr ] [ attach ] { "|" stage }
             | expr "|" unit "where" expr { "|" stage } ;   (* a bare unit selects all its rows *)
unit         = "sessions" | "messages" | "blocks" | "tool_calls"
             | "files" | "events" | "notes" | "links"
             | "turns" | "runs" | "deliveries" | "processes" ;

expr         = or_expr ;
or_expr      = and_expr { "OR" and_expr } ;
and_expr     = not_expr { [ "AND" ] not_expr } ;      (* juxtaposition = AND *)
not_expr     = ( "NOT" | "-" ) not_expr | atom ;
atom         = "(" expr ")"
             | match | compare | between | text | near | ref
             | "exists" row_unit "(" expr ")"
             | "seq" "(" step { "->" [ "[" gap "]" ] step } ")"
             | "query:" name ;                         (* a saved query, bound as a source (Saved queries) *)
ref          = origin ":" native_id | "session:" id | "message:" id [ ".." id ]
             | "block:" id | "note:" id [ "@" id ] | "result:" id [ ".rows" | ".matches" ]
             | "page:" name | "file:" path "@" commit
             | "@" digits [ "." digits ] | "last" | "latest" | "self" | url ;
                                                       (* user-data.md, references; "last" alone is
                                                          a reference, "last week"/"last month" a time *)

match        = field ":" value ;                        (* equality / substring per field *)
value        = word | quoted | "(" word { "|" word } ")" | "*" | "unknown" ;
compare      = field op ( number | duration | money | time ) ;
op           = ">" | ">=" | "<" | "<=" | "=" | "!=" ;
between      = field "between" ( number | time ) "and" ( number | time ) ;
text         = quoted | word ;                          (* full-text search, see below *)
near         = "near:" ( quoted | ref ) ;               (* semantic similarity *)
step         = atom { [ "AND" ] atom } ;                (* conditions on one tool call; juxtaposition = AND *)
gap          = "next" | "within:" duration ;
row_unit     = "message" | "block" | "tool_call" | "file" | "event" | "note" | "link"
             | "turn" | "run" | "delivery" | "process" ;

attach       = "with" child { "," child } ;
child        = row_unit_plural "(" field { "," field } ")" [ "[" window { "," window } "]" ] ;
window       = condition_literal | "first:" number | "last:" number ;

stage        = "sort by" sort_key [ "asc" | "desc" ]
             | "group by" group_key { "," group_key } "|" ( "count" | agg )
             | agg
             | "limit" number
             | "offset" number
             | "where" expr                             (* filter the current rows; after group by, the groups *)
             | "distinct"                               (* drop rows equal in every field *)
             | "matches" ;                              (* for seq(...): one row per matching step tuple *)
agg          = "agg" aggfn { "," aggfn } ;
aggfn        = "count" | ( "sum" | "avg" | "min" | "max" | "p" digits | "count_distinct" ) ":" field ;
group_key    = field | ( "day" | "week" | "month" ) "(" field ")" ;
sort_key     = field | aggname | "random" [ "(" number ")" ] ;  (* after group by: count, sum_cost, … *)

time         = ISO date or datetime | "today" | "yesterday" | hh ":" mm
             | ( "this" | "last" ) ( "week" | "month" )  (* logical weeks (Monday) and months *)
             | number unit_ago ;                        (* 7d, 2w, 3h; 09:00 = today at 09:00, except in read --from/--until (below) *)
unit_ago     = "h" | "d" | "w" ;                        (* hours back from now: 1d = 24 h, 1w = 168 h *)
duration     = number ( "s" | "m" | "h" | "d" ) ;
money        = number [ "usd" ] ;
field        = ident { "." ident } ;                    (* session.repo, message.role *)
```

A bare word or quoted string that is not a `field:value` is full-text
search. A word whose part before the first `:` is a known origin is a
reference, not a field; it means `id:<that ID>` (message references select
their session for session units). **What a bare word is:** a bare word with
no origin is a pasted native ID only if it has an ID's shape (at least 8
characters of hex digits and dashes containing a digit or a dash, the start
of a UUID, a UUIDv7 or a ULID as `model.md` lists native ID forms) and
resolves to exactly one session or message; then it is that reference.
Otherwise it is text, so `cafe` or `parser` searches; words under 8
characters are never IDs. An ID-shaped word that resolves to nothing or to
several is an error naming the candidates (`not_found`, `ambiguous`). A
reference names exactly one thing and drops the default filters for that
match (Defaults). How the command line tells a query from a
command (and refuses a mistyped command instead of searching for it) is
defined in `surfaces/cli.md`, grammar.

**Times.** `7d`, `2w`, `3h` count back from now in exact hours (`7d` is 168
hours). Dates, `today`, `yesterday`, `this week`, `last month` and `HH:MM`
are in the operator's time zone (`day.zone`: the machine's zone,
`/etc/localtime`, unless set), and a date is a **logical day**: date D runs
from D at `day.logical_start` (default 04:00) to D+1 at the same wall-clock
time, so work at 01:30 belongs to the day before, as a person reads it.
The rule, including DST days of 23 or 25 hours, is joint #15 and is
implemented once, in the module Sinex and Polylogue share
(`polylogue-sinex-joint/prototype/logical-day/`). `group by day` (and week,
month; ISO weeks starting on Monday) uses the same boundary, and briefing
ages count logical days crossed. Plain calendar dates are used only where
the meaning is a calendar date: a provider's statement or billing date.
Subscription usage windows are the provider's own and are not logical days.
In a query condition a bare `HH:MM` is today at that time. Inside one
session, as `read --from`/`--until` and the MCP `read` tool's `from` and
`until` take it, a bare `HH:MM` is that time within the session being read:
its first occurrence at or after the session's start (`read --from 10:50`
on a session of Oct 02 means Oct 02 10:50).
Each query resolves its times once, when it first runs, against one `now`
and one zone, and the result records both (`resolved`, Results). A cursor
carries the resolved instants, so every page uses them; a kept result keeps
them; a saved query or a watch resolves afresh on each run. A date bound is
half-open: `since:D` means at or after the start of D, `until:D` before the
start of the day after D. `F between D1 and D2` with dates means from the
start of D1 to before the start of the day after D2. A local time in a daylight-saving gap means the
first instant after the gap; one in a fold means the earlier instant, and
`notes` says so.

**No limit on what may be asked, one physical one.** Any valid query is
answered (a one-character prefix such as `a*` is correct, just slow): in
one read when it fits the read's time limit, otherwise as a `timeout` error
carrying a narrower query that answers part of it, so the whole meaning is
reachable in steps (Four bounds, execution budget). Parentheses nest at most
64 deep, because the parser is recursive and deeper nesting would exhaust
its stack; deeper nesting is refused with that reason. A read also stops
when its caller goes away or cancels (MCP cancellation, a closed HTTP
connection, Ctrl-C).

**Four bounds, never confused.** A *selection* (`| limit N`, `[last:k]`)
is part of the question and changes the answer. A *result page* (20
sessions, 50 messages by default; the request's `page_size`) is how much of
the answer travels now; `count` and the
cursor cover the rest. A *presentation budget* (a briefing's tokens, a
terminal's width) cuts what is shown, never the population a figure is
computed from: every count and total in a briefing is computed before the
budget is applied (`user-data.md`). An *execution budget* (the read time
limit, Results, `timeout`) decides whether one transaction can answer; when
it cannot, the result is a `timeout` error with a narrower suggestion, never
a silently shortened answer, and the same meaning is reachable in steps.
Every search plan is exact (law 82): walking the newest sessions for a
frequent word returns the same rows as ranking inside the index, so no
result is a scoped or approximate sample unless the query says so.

## Changes from the current language

The changes from the Python version's language are accepted as listed in
`evidence/query-language-changes.md`; named parts and macros come in slice 3
or later (Planned extensions).

## Defaults

| Unit | Default filter | Dropped by a top-level condition on |
|---|---|---|
| `sessions` | `kind` is not `subagent` (subagents are part of their parent's work in listings) | `kind`, `parent`, `root`, `depth`, `lineage`, `link`, `team`, `agent_name` |
| `messages`, `blocks`, `tool_calls` | not inherited (copies of an earlier session's messages are excluded, so nothing is counted twice) | `inherited` |
| `tool_calls` | no code-mode wrapper whose children are counted (`action:code` with children; a wrapper with no children counts itself, `sources/codex.md`, code mode) | `action` |
| `messages`, `blocks`, `tool_calls` | not off the active path: abandoned branches are excluded; messages whose path status is unknown (`model.md`, active path) are kept, and `notes` counts them | `active` |
| every unit | not in a hidden session (`user-data.md`, hiding) | `hidden` |
| `notes` | current revisions only, not withdrawn | `current`, `withdrawn`, `root` |
| `sessions` | not `source:missing` (sessions a harness index lists but whose transcript was never stored have no messages) | `source` |
| every unit | not named by an excision (`storage/archive-db.md`, how excision works): sessions, texts and notes an `excision` row covers are left out from the moment that row commits until their rows are gone, read through the same view as `hidden` | never; excised content cannot be asked for |

A reference (`claude-code:…`, a pasted ID, `@N`, `last`, or an `id:`
condition naming one ID) names exactly one thing and drops the subagent,
hidden, inherited and active-path defaults for that match, so a subagent,
a hidden session, an inherited message or an abandoned branch is reachable
by its reference. The excision default is never dropped.

**Binding.** A query's source is its unit, a saved query (`query:NAME`) or a
kept result (`result:<id>`). Binding the source resolves its defaults into
explicit conditions, once. A default is dropped only by a **top-level
condition** on its field: a condition joined to the rest of the query's own
`where` by AND (parentheses around ANDs are flattened), not one inside `OR`,
`NOT`, `exists`, `seq` or a saved query. `F:*` names the field without
restricting it; it is the explicit way to say "no default for F"
(`kind:*`, `inherited:*`, `action:*`), and there is no second spelling for
it. Inside `exists child(...)` and `with child(...)` the same rule binds the
child's defaults from the child's own top-level conditions. After binding,
every further condition, stage, attachment or outer query only narrows: so
`tool_calls where outcome:error (action:shell OR NOT action:shell)` returns
the same rows as `tool_calls where outcome:error`, and a condition true for
every row never adds rows. The mention-based convenience survives only at
the top level, where it is a statement about the source, never inside a
composition (law 133).

**Result pages.** Without `limit`, a result page holds 20 rows for
`sessions` and 50 for every other unit and for grouped rows (`page_size`,
the same defaults on every surface, MCP included), never a cap on the answer, so `count`, sums and groups
always cover every row, and further rows come by cursor. `limit` takes any
positive integer and is honoured (a selection: it changes the question);
anything else is refused with an error. There is no maximum.

Row units (`messages`, `blocks`, `tool_calls`, `files`, `events`) include
rows from subagent sessions: a subagent's tool calls are real work and are
not duplicates. Session figures that cover every branch (tokens, cost,
`tool_calls`, `errors`; `analysis.md` §4) equal the row units only with
`active:*`; `analyze` uses `active:*` for cost and tool counts.

## Full-text search

Staged replacement data is outside the published search corpus until its
message representation is selected (`convergence.md`, Parse). It cannot affect
matching, relevance scores, counts or facets of that view. This includes
corpus statistics used by ranking: joining hits back to selected blocks is
insufficient if unreachable staged text has already changed FTS document
frequencies or average length. Preparation may use isolated scratch indexes
or another representation, but publication and cleanup must preserve the
declared corpus and remain bounded. This does not redefine the existing
ranking population for ordinary query filters.

- A quoted string or bare word matches the prose of messages: `text` blocks
  written by a human or a model. Tokenization folds case and diacritics
  (`ł` matches `l`) and splits on anything that is not a letter or digit
  (`weekday_offset()` contains the words `weekday` and `offset`); there is
  no stemming; `refactor*` is a prefix search. Several words must all occur
  in the same message; `"a b"` is a phrase.
- In a session query, a text term matches sessions with at least one such
  message. In `messages where`, it matches the messages. Subagents are not
  separate rows by default (`kind:subagent` or `kind:*` shows them): a match
  in a subagent's message makes its root session match, and the hit in
  `hits` names the subagent session it came from.
- Tool output is searched with `output:"…"` through `output_fts`
  (`tool_result` blocks and bodies written through file-writing and
  file-editing calls such as `Write`, `Edit`, `apply_patch`, which are the
  agent's work product; `storage/index-db.md`, full-text search); thinking
  is searched with `thinking:"…"` through `text_fts` restricted to thinking
  blocks; plain text terms use `text_fts` restricted to text blocks. A
  quoted value is a phrase and a bare value a word, as for prose. `command:"…"` and
  `path:"…"` are substring matches through the trigram index on tool calls
  (`tool_call_trigram`, `storage/index-db.md`), so `cargo te` matches `cargo test`.
  `input:"…"` is a substring match on the arguments as given; it has no
  index of its own and narrows tool calls already selected by other
  conditions.
- **Where else the words are.** When a text term matches fewer than three
  prose messages, the result's `notes` give the counts for the same words in
  commands, tool output and thinking, each with its query: `also in 14 tool
  commands: tool_calls where command:"nextest --partition"`. This is how
  "has anyone tried X" finds work that happened in tool calls.
- **Where the index points.** Text is stored once per distinct content
  (`storage/index-db.md`, `text` and `text_fts`), so one indexed text can belong to
  many blocks (the same prompt pasted twice, inherited copies). A hit on a
  text is expanded to every block that uses it, then filtered by the rest of
  the query. Defaults apply there: inherited copies and abandoned branches
  drop out unless the query asks for them, so a resumed session does not
  repeat its parent's hits.
- **Filters before ranking.** All other conditions (origin, time, kind,
  hidden, …) restrict the candidate blocks first; ranking orders only what
  survives, so a filter never hides a good hit behind worse ones.
- **Which side drives the query.** The planner starts from the other
  conditions or from the full-text hits, whichever is more selective, by the
  rule in `storage/index-db.md` (full-text search). Both give the same rows
  in the same order (law 82, `plan/testing/laws.md`); the choice affects
  speed only.
- **Ranking.** SQLite FTS5's `bm25`, with no field weights in the first
  version. A session's rank is its best message's rank. Ties go to the more
  recent message. Relevance is the default sort when a text term is present;
  any `sort by` replaces it.
- **Snippets.** Each hit carries one snippet: about 12 words around the
  first match, matches marked (bold in the terminal, `«…»` in plain-text
  renderings, the `marks` offsets in JSON, Results), `…` where text was cut. `--fields` can ask for the full
  text instead.

Letters with a stroke fold like accented ones: a message containing `Łódź`
is found by `lodz` and by `łódź` (`storage/index-db.md`, folding).

## Semantic search

`near:"how do I handle ambiguous dates"` ranks by embedding similarity;
`near:session:<id>` and `near:message:<id>` rank by similarity to a stored
item. It is a ranking, not a filter: combined with other conditions, the
conditions select the candidates first and `near` orders only those, so a
scoped search never means "the global top hits, then filtered". Vectors are scanned exactly over the candidates, so nothing
is cut off by an approximate index.

- `near:session:<id>` and `near:message:<id>` compare stored vectors and
  need no provider key; only `near:"text"` asks the provider to embed the
  text.
- What is embedded: distinct human prompts, model prose and note texts
  (claims, decisions, findings), so notes can be found by meaning too
  (`analysis.md` §8). A session ranks by its best-matching passage.
- **Hybrid.** A query with both a text term and `near:` fuses the full-text
  and vector rankings by reciprocal rank: each hit scores the sum of 1/(60 +
  rank) over the rankings it appears in. `explain` shows each hit's rank
  in each ranking and its contribution.
- When the vector side of a hybrid query cannot run (no embeddings, provider
  unreachable), the text and tool-call sides still run and are fused alone;
  the outcome is `degraded` and `notes` names the missing side.
- When embeddings are not enabled, `near:"text"` gives outcome `degraded`
  with the note "semantic search is not enabled", and rows are ordered by
  time and marked unranked.
- Candidates without a vector for the configured embedding key are never
  dropped: they follow the ranked rows, unranked, and `unknown.near` counts
  them. `count` is the exact number of candidates.

## Fields

The `Slice` column (or the line under a unit) is the slice from which a
field answers; before it, a query naming the field gets `not_yet_available`
naming that slice. Slice 1's set is the list in `plan/slice-01.md`.

Match fields take `field:value`; numeric and time fields take comparisons.
`value` may be `(a|b|c)` for alternatives, `*` for any value (including
absent), or `unknown` for "the data does not say". `unknown` has only that
meaning: a pending link is `parent:missing`, and a session with no outcome
note is `outcome:none`. A field with a closed set of values refuses any
other value with outcome `error`, naming the set and any unit where the
value does exist (`sessions where outcome:error` → `outcome on sessions is
one of solved, partial, abandoned, none; "error" is an outcome of
tool_calls`), never an empty result. Strings match
case-insensitively; `title`, `path`, `command` and `input` match substrings,
`output` and `thinking` match words (full-text), everything else matches
whole values. `cwd` and `checkout` are scope fields of kind `subtree`: matched by whole
path components and exactly as written (paths are case-sensitive): `cwd:/src/tide`
selects `/src/tide` and directories below it, never `/src/tidepool` or
`/src/Tide`, and `_` or `%` in a path are ordinary characters (the briefing's
directory scope uses the same rule, `surfaces/cli.md`).

### sessions

| Field | Type | Meaning | Slice |
|---|---|---|---|
| `id` | match (prefix) | session ID or a unique prefix | 1 |
| `origin` | match | `claude-code`, `codex`, `chatgpt`, … | 1 |
| `kind` | match | `main` (an agent session started by a person, including one that continues an earlier session: continuing is a `continuation` link, not a kind), `subagent` (it has an incoming `subagent` link: spawned by another session's tool call), `chat` (a web chat). Kinds are listed in `model.md`, Session. | 1 |
| `title` | substring | the displayed title (`session_facts.display_title`, by the one rule in `model.md`, Session) | 3 |
| `title_source` | match | where the displayed title came from: `note`, `provider`, `instruction`, `first_prompt`, `id` (`model.md`, Session) | 3 |
| `repo` | match | the session's repository (`model.md`, Session): matches the name (`morrow`), `owner/name` (`op/morrow`), the URL, or the root commit hash (prefix, at least 7 hex digits). Two spellings of one repository match the same sessions, because the identity is the root commit. Sessions without git evidence never match. | 1 |
| `repo_root` | match (prefix) | the repository's root commit hash, its identity (`model.md`, Session); `repo:` also accepts it | 3 |
| `checkout` | subtree | the worktree path, for sessions in a separate worktree of the repository | 3 |
| `collection` | match | the provider's grouping (ChatGPT project or custom GPT, claude.ai project): its name or ID | 4 |
| `referenced_by` | ref | sessions some given thing points at: `referenced_by:page:<name>`, `referenced_by:note:<root>` (backlinks) | 7 |
| `references` | number | how many notes, links, page slots, kept results and deliveries point at the session; reading any reference lists them (`rendering.md`) | 7 |
| `branch` | match | as recorded | 1 |
| `commit` | match (prefix, at least 7 hex digits) | the commit recorded at session start (Codex `git.commit_hash`) or any commit linked to the session by git correlation (`analysis.md` §9) | 11 |
| `pr` | match | a pull request linked to the session: number (`pr:42`), `owner/name#42`, or URL (`analysis.md` §9) | 11 |
| `outcome` | match | the session's `outcome` note: `solved`, `partial`, `abandoned`, or `none` when no outcome note exists (`user-data.md`) | 7 |
| `hidden` | bool | has a `hidden` note; hidden sessions are excluded unless a top-level condition names `hidden` (Defaults, binding; `user-data.md`) | 7 |
| `cwd` | subtree | working directory, and the directories below it | 1 |
| `source` | match | `present` (a provider file for the session still exists), `gone` (every provider file was deleted, or the session is absent from a later complete export of the same account, which lists the whole account at its export time; the archive kept the bytes), `missing` (a harness index such as Claude Code's `history.jsonl` or Codex's `state_5.sqlite` lists the session, but no transcript was ever stored). Computed from `observation` rows (locations, `deleted` and `back`) and the harness indexes. A location that cannot currently be read is not `gone` (`convergence.md`, collect). | 2 |
| `gone_at` | time | when the rescan first found the session's last provider file missing (the `deleted` observation, `storage/blobs.md`), or the export time of the first complete export of its account that lacks it; unknown for sessions that are not `gone`. A `gone` session stays in ordinary views: keeping what providers delete is the archive's purpose (Sinex's "removed at source" is the same fact, and Sinex hides it by default) | 2 |
| `run` | ref | sessions that carried out the given run, or any attempt of the given task (`model.md`, Run) | 6 |
| `ending` | match | how the session ended, by the exact rules of the resume list (`user-data.md`, resume candidates): `open` (any open reason), `clean` (none), or one reason: `unanswered_prompt`, `unfinished_tool_call`, `open_note`, `outcome_partial`, `question` (computed from records from slice 2, since briefings need it; `open_note` and `outcome_partial` read notes and so apply from slice 7) | 2 |
| `waiting` | bool | the session's live state is `waiting` (`model.md`, Live state): a permission request or notification with nothing after it, or a turn that ended on a question to the operator; computed from records, never from guessing. The overview lists these first (`surfaces/cli.md`) | 2 |
| `running` | bool | the live state is `working`, `waiting`, `idle` or `unknown` (`model.md`, Live state); the last means only file growth is known (it grew in the last 15 minutes). Computed from hook and process records and file times, never from text. It is the one definition of "running now": the overview's list and the `continue` candidates refused without `--anyway` both use it; nothing else is refused or dropped by it | 2 |
| `live` | match | the live state: `working`, `waiting`, `idle`, `ended`, `stale`, `unknown` (slice 2) | 2 |
| `waiting_on` | match | what a waiting session waits for: `permission`, `question`, `notification` | 2 |
| `peers_live` | number | how many other live sessions share its repository (`repo_root`), its checkout, or, when either session has no git evidence, its working directory | 2 |
| `task` | match | the live task: an AgentCTL job ID or a Bead ID, from evidence only (`model.md`, live task): from job variables in hook lines from slice 2, from run evidence from slice 6 | 2 |
| `process_alive` | bool | whether the harness process is alive, from the latest process evidence; null without process evidence | 2 |
| `harness` | match | `claude-code`, `codex-tui`, `codex-exec`, … | 3 |
| `model` | match | a model that wrote at least one message. A session that used two models matches either; `group by model` counts it under each, and `analyze` totals still count it once. | 1 |
| `tag` | match | a tag note; `tag:key:value` for key:value tags | 7 |
| `has` | match | `thinking`, `tool_calls`, `errors`, `attachments`, `compaction`, `branches`, `note` (any current note targets the session or one of its messages), `conflict` (a note on it is in a conflict: a family with two current revisions, or two current command-written notes of one single-valued kind; `user-data.md`), `secret` (a block looks like a credential; `behavior.md`) | 1 (`note`, `conflict`: 7; `secret`: 3) |
| `evidence_until` | time | the evidence cutoff: the evidence Polylogue had up to time T, read with today's parsers, prices and excisions (named for that: Sinex's "state as of T" means its committed interpretation at T, a different question; the joint glossary pairs them). Concretely: the messages and events whose `first_seen <= T` (`convergence.md`) and the notes this archive had stored by T, current at T. The session's counts, tokens, times, display title, order and each message's version are computed from those alone; a message counts as inherited only when the message it was copied from has `first_seen <= T`, otherwise it is its session's own, as it was at T. Prices, parser versions and excisions are today's, and `notes` says so. `evidence_until:` applies to the whole query and can be combined with any unit: `polylogue 'claude-code:7f3a2c10-0004 evidence_until:2026-10-03T14:30' read`. Provider time (`since:`, `until:`, `at`) is a separate question: when it happened, not when Polylogue learned of it. | 3 |
| `temporary` | bool | temporary web chats | 4 |
| `parent` | ref | sessions with an incoming link (any kind) whose parent is the given session: `parent:<id>` lists its subagents, continuations and references. `parent:missing` matches sessions whose link points at a parent that is not in the archive (a pending link). | 6 |
| `root` | ref | sessions in the subagent tree whose top session is the given one (the root itself included) | 6 |
| `depth` | number | depth in the subagent tree: 0 for a session with no incoming `subagent` link, 1 for its subagents, and so on | 6 |
| `link` | match | the kind of an incoming link: `subagent`, `continuation`, `reference`, or a shorthand: `link:fork` means `(kind:continuation how:fork) OR (kind:subagent how:copied)`, `link:resume` means `kind:continuation how:resume`. For other combinations use the `links` unit. | 6 |
| `continues` | ref | sessions with a `continuation` link from the given session (they carry it on) | 6 |
| `continued_by` | ref | sessions that the given session carries on (the reverse of `continues`) | 6 |
| `lineage` | ref | every session connected to the given one through links of any kind, in either direction | 6 |
| `team`, `agent_name` | match | for a named member of an agent team: the team and the member's name (`model.md`, Session) | 6 |
| `path` | substring | a file path the session read, wrote or edited | 3 |
| `tool`, `action` | match | the session made at least one such tool call (see `tool_calls`) | 3 |
| `started`, `last_active` | time | first and last time among the session's own messages and events (inherited copies excluded) | 1 |
| `since:T` | time | `last_active >= T` (active on or after T). On units whose rows have a span (sessions, turns, tool calls with a result time, processes), `since:` and `until:` select by overlap: a row matches `since:T` if it ended at or after T or is still open, and `until:T` if it started at or before T, so a turn, a long tool call or a process that began before a window and was still going during it appears in "what was happening around T" (Sinex's recall follows the same rule). On units without a span (messages, blocks, events, notes), they compare the row's own time (`at`, or `created_at` for notes). | 1 |
| `until:T` | time | `started <= T` (begun on or before T) | 1 |
| `messages` | number | own messages (not inherited, active path) | 1 |
| `human_prompts` | number | own human prompts: messages with `written_by:human` and `prompt_kind:prose` (`model.md`, Message) | 1 |
| `human_commands` | number | own typed commands: `written_by:human` with `prompt_kind:command` (a slash command, `!` shell input, `#` memory input) | 3 |
| `words` | number | words in own prose, counted as the full-text tokenizer splits them | 3 |
| `tool_calls`, `errors` | number | own tool calls; those with outcome `error` | 1 |
| `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens` | number | sums over own request-bearing messages on every branch (`analysis.md` §4): complete, partial (a lower bound) or unknown | 1 |
| `cost` | money | computed from tokens and the price table (`analysis.md`): complete, partial (a lower bound; some request-bearing messages unpriced) or unknown (none priced); slice 1, since slice 1 already computes `message.cost_nusd` from Claude Code usage and its acceptance reads own costs (`plan/slice-01.md`) | 1 |
| `tree_cost` | money | the session's own cost plus that of every session below it in its subagent tree, each counted once (`analysis.md` §5); the subagents are not in `cost` | 6 |
| `duration` | duration | `last_active - started`; for a session still in progress, the duration up to its last record, known and growing, never up to now. "Has kept running for an hour" is a signal's `for` over `live:working`, or `started < now - 1h`, never `duration > 1h`, which lags by the time since the last record (Sinex's open intervals have an unknown duration instead; the joint `for` works on both) | 3 |

### messages

| Field | Type | Meaning | Slice |
|---|---|---|---|
| `id` | match (prefix) | message ID | 3 |
| `role` | match | `user`, `assistant`, `system`, `tool` | 3 |
| `written_by` | match | `human`, `model`, `harness`, `unknown` (a record that matches no listed form; `model.md`, Message) | 3 |
| `prompt_kind` | match | for human messages: `prose` or `command` | 3 |
| `paste_evidence` | match | human messages: `provider_reported` and/or `clipboard_match`; both may hold. No evidence matches neither; unevaluated evidence is unknown, not false (`model.md`, Text-origin evidence) | 3 |
| `operator_supplied_bytes` | number | all human-supplied UTF-8 text bytes, not authored bytes | 3 |
| `provider_unmarked_bytes` | number | bytes outside precise provider-reported paste spans; unknown when extraction or extent is unknown | 3 |
| `clipboard_matched_bytes` | number | union of candidate matching ranges; unknown when matching was not evaluated | 3 |
| `model` | match | | 3 |
| `at` | time | | 3 |
| `inherited` | bool | copy of an earlier session's message | 3 |
| `active` | bool | on the active path | 3 |
| `has` | match | `thinking`, `tool_calls`, `attachments`, `code` (a fenced code block in prose) | 3 |
| `code_lang` | match | the language tag of a fenced code block in the message (`python`, `rust`) | 3 |
| `words` | number | | 3 |
| `input_tokens` … `cost` | number / money | per message | 3 |
| `stop_reason` | match | | 3 |
| `from_session`, `from_name` | ref / match | for a message one agent sent to another: the sending session and the sender's name as written (`model.md`, Message) | 6 |
| `"text"`, `thinking:"…"`, `near:"…"` | text | see full-text and semantic search; `near:` ranks messages as well as sessions | 3 (`near:` 9) |
| `session.<field>` | | any session field of the owning session | 3 |

In any row query, a session field written without `session.` (`cwd:`,
`repo:`, `origin:`) means `session.<field>` when the row unit has no field
of that name; `explain` shows the full form.

### blocks

`kind` (`text`, `thinking`, `tool_call`, `tool_result`, `image`, `file`,
`other`), `media_type`, `file_name`, text search, plus `message.<field>`,
`session.<field>`, and for `tool_call` and `tool_result` blocks `call.<field>`
(the tool call's fields below: `call.tool`, `call.outcome`,
`call.exit_code`), which are stored only in `tool_call`. Slice 3.

### tool_calls

One row per tool call, joined with its result.

| Field | Type | Meaning | Slice |
|---|---|---|---|
| `tool` | match | the tool name as the provider wrote it: `Bash`, `Edit`, `exec_command` (a Codex code-mode child), `apply_patch`, an MCP tool name | 3 |
| `mcp_server` | match | for MCP tools, the server part of the name (`mcp__lynchpin__stats` → `lynchpin`) | 3 |
| `action` | match | a category from a fixed per-provider table of tool names: `shell`, `file_read`, `file_write`, `file_edit`, `search`, `web`, `subagent`, `mcp`, `code` (a Codex code-mode wrapper; excluded from counts, since its children are counted), `other`. Never inferred from text. | 3 |
| `command` | substring | the shell command, for shell-like tools | 3 |
| `command_norm` | match | the normalized command: the program, plus the subcommand for known multi-command programs (`cargo test`, `git commit`); the grouping key for "which commands fail" (`analysis.md`, tool calls) | 3 |
| `input` | substring | text inside the arguments | 3 |
| `output` | words | text inside the result, matched by words through the full-text index of tool output (`storage/index-db.md`) | 3 |
| `path` | substring | a file path in the arguments | 3 |
| `outcome` | match | `ok`, `error`, `pending`, `unknown`: the harness's verdict on the call (an exit code, an error flag); `pending` and `unknown` for calls without a result: `model.md`, Block (tool outcome). It is not whether the task succeeded: an `ok` command on the wrong revision proves nothing about the right one, and an expected failure can be the point (`analysis.md`, what a record proves) | 3 |
| `exit_code` | number | | 3 |
| `at`, `duration` | time / duration | | 3 |
| `child` | ref | for subagent calls (`action:subagent`): the child session, through the `subagent` link whose spawning block is this call. `child.<field>` filters on any session field of the child (`child.messages`, `child.cost`, `child.tree_cost`, `child.outcome`, `child.ending`), plus `child.agent_type`, the agent type the provider recorded on the link: `tool_calls where action:subagent child.tree_cost > 0.10` | 6 |
| `corrected` | bool | a `correction` note targets this tool call (or its result) directly; a correction aimed at the whole session or message does not count here, because it does not say which call it corrects | 7 |
| `next` | ref | the session's next tool call in session order (message `seq`, then block index); `next.<field>` filters or groups on it: `tool_calls where outcome:error \| group by next.action \| count` shows what agents do right after a failure | 3 |
| `message.<field>`, `session.<field>` | | owning message and session | 3 |

### files

One row per (session, file path) the session touched: `path`, `reads`,
`writes`, `edits` (counts), `first_at`, `last_at`, `session.<field>`.
Slice 3.

### links

One row per link between two sessions (`model.md`, Link). Every field:
slice 6.

| Field | Type | Meaning |
|---|---|---|
| `kind` | match | `subagent`, `continuation`, `reference` |
| `how` | match | subagent: `fresh`, `copied`, `summarized`; continuation: `resume`, `fork`, `after_compaction`; reference: `briefing`, `ping`, `mention` |
| `parent`, `child` | ref | the two sessions; `parent.<field>` and `child.<field>` filter on their session fields |
| `pending` | bool | the parent is not in the archive (yet, or ever) |
| `evidence` | match | the input field that established the link (`sources/` lists them) |
| `branch_point`, `spawned_by` | ref | the message where the child branched off; the spawning tool call (subagents). Same names as the export format. |
| `attrs.<key>` | match | a provider fact recorded on the link as given: `attrs.agent_type`, `attrs.role`, `attrs.worktree` |
| `retry_of` | ref | the earlier spawning call this subagent call retries, when the harness recorded one |
| `result_use` | match | how the child's result reached the parent: `returned` (as the spawning call's result or its completion notification), `referenced`, `unknown` (`model.md`, Link) |

`links where kind:subagent how:copied` lists subagents that started with a
copy of their parent's conversation; `links where pending:true | group by
evidence | count` shows which kinds of evidence point at sessions the archive
lacks.

### turns

One row per turn, computed along the active path. A turn starts at a human
prompt (in a session with none, at its first own message) and ends before
the next human prompt on the path. Prompts typed one after another before
any answer form one turn (a burst), not several. A typed command
(`prompt_kind:command`) starts a turn of its own, marked `slash_command`,
and a command that ends a session does not make its turn abandoned.
Messages off the active path belong to the turn of the nearest prompt among
their ancestors; replies under the same prompt are that turn's `attempts`,
and `answer` is the one on the path. In a session whose active path is
unknown, turns follow file order and each row says `path: unknown`. Every
field: slice 3 (`corrected`: 7).

| Field | Type | Meaning |
|---|---|---|
| `id` | ref | the turn's identity: its first prompt's message ID (in a session with no human prompt, its first own message's), so it never moves when later messages arrive |
| `start`, `end` | time | the first prompt's time; the latest time (or `completed_at`) of any message in the turn; unknown when those messages have no time. Sinex uses these spans for questions such as "commands I ran while discussing X" (joint #22) |
| `prompts` | ref | the human prompt messages of the turn, in order |
| `burst` | number | how many prompts it has |
| `slash_command` | bool | the turn starts with a typed slash command |
| `answer` | ref | the model's final message in the turn, on the active path |
| `attempts` | number | replies under the turn's prompt, abandoned branches included |
| `path` | match | `active`, or `unknown` when the session's active path is unknown |
| `latency` | duration | from the last prompt to the first model message; unknown unless both have times |
| `tool_calls`, `errors` | number | in the turn |
| `input_tokens`, `output_tokens`, `cost` | number / money | summed over the turn's model messages |
| `abandoned` | bool | the turn ended the session with no answer |
| `corrected` | bool | the next turn's prompt carries a `correction` note on this turn's answer |
| `session.<field>` | | the session |

`turns where session.repo:morrow | agg p90:latency, avg:cost` gives a
repository's typical turn.

### runs

One row per task or attempt a harness recorded (`model.md`, Run). Every
field: slice 6.

| Field | Type | Meaning |
|---|---|---|
| `id` | match (prefix) | `<origin>:<native run id>` |
| `kind` | match | `task`, `attempt` |
| `task` | ref | an attempt's task; `task.<field>` filters on it |
| `state` | match | as the harness recorded: `running`, `completed`, `failed`, `cancelled`, `unknown` |
| `instruction` | substring | the instruction as given |
| `attempts` | number | for a task: how many attempts it has |
| `started`, `ended`, `duration` | time / duration | |
| `evidence` | match | the input field that established it |
| `session.<field>` | | a session that carried it out (a run with several sessions matches if any does) |

`runs where kind:task attempts > 1 | group by state | count` shows how often
a retried task ended each way.

### deliveries

One row per occasion Polylogue gave context to a session (`model.md`,
Delivery). Every field: slice 2.

| Field | Type | Meaning |
|---|---|---|
| `id` | match (prefix) | the delivery ID, as in the injected reference line |
| `recipient` | ref | the receiving session; `recipient.<field>` filters on it |
| `event` | match | `session_start`, `resume`, `compaction`, `prompt_reference`, `mcp_brief`, `ping`, `advisory` |
| `outcome` | match | `delivered`, `failed`, `unknown`, `not_sent` |
| `content_hash` | match | the content hash; equal content in two rows is two deliveries |
| `refs` | ref | a reference the delivery included |
| `left_out` | ref | a reference left out for budget |
| `at` | time | |
| `bytes` | number | |

`deliveries where outcome:delivered refs:claude-code:7f3a2c10-0002 | group by
recipient | count` lists the sessions that were told about #2: against the
slice-2 Morrow archive, which adds a hook spool with one delivered
`prompt_reference` briefing in #4, from a prompt that pastes `7f3a2c10-0002`
(`examples/archive/FACTS.md`, hook spool; `plan/slices.md`, slice 2), it
returns #4 with count 1.

### processes

One row per harness process or descendant (`model.md`, Process).

| Field | Type | Meaning | Slice |
|---|---|---|---|
| `session` | ref | the session; `session.<field>` filters on it | 2 |
| `role` | match | `harness`, `child` | 2 |
| `comm` | match | the command name | 2 |
| `argv` | text | the argument list, as observed | 2 |
| `alive` | bool | no end seen yet | 2 |
| `started_at`, `ended_at` | time | | 2 |
| `tool_call` | ref | the tool call it belongs to, when linked by timing and ancestry (slice 6); `tool_call.<field>` filters on it | 6 |

`processes where alive:true comm:cargo | group by session.repo | count`
shows which repositories are building or testing now.

### events

`kind` (`compaction`, `model_switch`, `permission`, `cwd_change`,
`hook_run`, `rate_limit`, `away_summary`, `plan`, …; the full list per
provider is in `sources/`), `at`, `model_saw` (bool), `session.<field>`.
Slice 1 stores compaction events (`has:compaction`); the `events` unit:
slice 3.

### notes

`kind` (`tag`, `note`, `title`, `decision`, `claim`, …, see
`user-data.md`), `key`, `value` (substring), `author` (`operator`, a
session ID, `agent` or `script`), `target` (a reference), `source` (the message an inline marker
came from; `source:*` selects notes written from markers), `created_at`, `root` (the revision family,
`user-data.md`), `current` (bool: no later revision replaces it), `withdrawn`
(bool: a current withdrawal withdraws it; withdrawn notes are left out
unless the query says `withdrawn:true`, only withdrawn, or `withdrawn:*`,
both), `has:conflict` (the note is in a conflict: its family has two
current revisions, or its target has two current command-written notes
of one single-valued kind; `user-data.md`), `stale` (bool: the target's text changed since the note
was written, from `note_target.seen_hash`), `check` (`holds`, `fails`,
`unknown` or `unchecked`: a claim's check evaluated at read time, never
stored), `rubric` (a rubric reference, `<root>` or `<root>@<revision>`:
notes written in a pass under it; `user-data.md`, checked claims and
annotation passes). A bare word or quoted string in `notes where` searches
the note text. Slice 7.

## Stages

- `| sort by <field> [asc|desc]`: ascending unless `desc` is written; rows
  whose value is unknown come after every known value in either direction,
  and ties keep the row ID order of the continuation rule (Results). Without
  any `sort by`, each unit has one default order: `sessions` by
  `last_active desc`; `messages`, `blocks`, `tool_calls`, `events` and
  `files` chronologically, oldest first (time, then ID; unknown times
  last); `notes` newest first; `links` by the parent's time; grouped rows
  by `count desc`, then key; with a text term, by relevance. After `group by`, the aggregate
  columns (`count`, `sum_cost`, …) can be sorted on.
- `| group by f1, f2 | count` or `| group by f | agg …`: groups rows. Time
  fields can be grouped by `day(at)`, `week(at)`, `month(at)` in the
  operator's time zone.
- `| agg count, sum:F, avg:F, min:F, max:F, p90:F, count_distinct:F`: one summary row. Unknown
  values are left out of sums and averages and reported as counts:
  `{"sum_cost": 6.20, "unknown_cost": 2, "partial_cost": 0}` (partial
  values, Semantics rules).
- `| sort by random`: a random order (a seed can be given, `random(7)`, for
  a repeatable sample).
- `| where CONDITIONS`: filters the current rows. Before any grouping it is
  the same as adding the conditions to the query's `where` (they narrow the
  bound source; they never rebind its defaults). After `group by`, the
  current rows are the groups, `{keys…, count, sum_cost, …}`, so `| where
  count >= 3` keeps groups and needs no separate `having`.
- `| limit N`, `| offset N`. After `group by` they act on the groups: a
  grouped result is rows like any other, so sorting, filtering, limiting and
  keeping it work as they do on a unit's rows (group closure).

### Attached rows

`QUERY with messages(id, role, text)[role:assistant, last:2]` returns each
matched session with a list of its messages. Inside the brackets, conditions
filter first, then `first:`/`last:` take that many from what remains, then
only the listed fields are returned. Available children: `messages`,
`blocks`, `tool_calls`, `files`, `events`, `notes`, `links`.

## Semantics rules

- **Rows are a bag, not a set.** Equal rows stay distinct: two messages with
  the same text are two rows, and two (failure, fix) pairs with the same
  failure and different fixes are two matches. Removing duplicates needs an
  explicit `| distinct` (all fields) or a `group by`.
- **One `exists`, one row.** `exists tool_call(action:shell AND
  outcome:error)` needs a single tool call that is both. Two separate
  `exists` may be satisfied by different rows.
- **Match rows.** A match found through `exists` or `seq(...)` also yields
  the rows that made it match, in order: `| matches` returns them as rows
  (session, then the matched tool calls or messages), and a kept result
  stores them. A pattern names evidence; it does not assert why the agent
  did it.
- **Sequences.** `seq(A -> B -> C)` matches sessions having tool calls
  matching A, B and C in session order (the message's `seq`, then the block
  index), not by time. `-> [next]` requires B to be the very next tool call
  in that order; `-> [within:5m]` also needs both calls' times, the later
  strictly later and at most 5 minutes after, and a missing or reversed
  time does not satisfy it. `| matches`
  returns each match as rows ("edit, then a failing test, then an edit").
  Matches in one session are found from the earliest tool call forward and
  do not overlap: after a match, the search continues after its last row.
  This selection is observable semantics, not an optimization: a condition
  inside a step narrows the candidates before matches are chosen, while
  `| matches | where …` filters matches already chosen, and the two can
  differ. Other policies (every match, overlapping matches) are explicit
  (Planned extensions), never a planner's choice. The matches never depend
  on the order rows happen to be read in (law 130).
- **Aggregates at the owner's grain.** A value is summed at the row that
  owns it before any join can repeat it: `sessions | agg sum:cost` sums
  each session's cost once, however many messages, tool calls or notes the
  query touched, and two sessions with equal cost are both counted (no
  "sum of distinct values" shortcut). No condition multiplies an owner's
  values implicitly; the only rows at a finer grain are the ones a query
  asks for (`| matches`, below), and the result says so.
- **Child lists stay separate.** `with tool_calls(...), notes(...)` returns
  two lists per session, never every pairing of a tool call with a note.
  A pairing is only made when the query asks for one (a sequence, a
  `next.` relation, a same-file condition).
- **One relation, several uses.** A session and the child rows that satisfy
  a child condition form one relation, and the query's form chooses how it
  is used. `exists child(C)` keeps each owner that has at least one
  matching child, once (a semijoin). `with children(...)[C]` keeps the
  owners exactly as the rest of the query selected them and attaches each
  one's matching children as a list, which may be empty; attaching never
  changes which owners are returned or how many. `[first:k]` and `[last:k]`
  choose per owner, in the child's stated order. `| matches` returns the
  relation itself, one row per owner and matching child (or matched step
  tuple), so its grain is the pair: the result's `unit` is `matches` and
  its `notes` say "one row per match", and an `agg` after it is computed
  over pairs, which is what a pair-grain question wants. Drill-down from a
  count to its rows runs the same relation, so the rows shown are the rows
  counted.
- **Membership, ranking, selection and page.** Four things a result can
  have, kept apart. The **candidates** are the rows the conditions select;
  `count` is always their exact number. A **ranking** (relevance for a text
  term, similarity for `near:`) orders the candidates and never changes
  which they are or how many. A **selection** (`| limit N` after a sort or a
  ranking, `[first:k]`) is part of the question: it changes the answer and
  `read_as` says so ("the 5 most similar"). A **page** is how many rows of
  the answer travel now; it is a request setting, never part of the
  question, and further rows come by cursor. Stages apply in the order
  written, so `| limit 10 | where …` filters a top ten and `| where … |
  limit 10` takes the top ten of what survives; the two are different
  questions, both defined. A query whose answer needs only the candidates
  (`| agg count`, `| group by … | count`) computes no ranking, so `near:"…"
  | agg count` never calls the embedding provider.
- **Unknown is not zero.** A missing value and an explicit `0` are
  different in filters, sums and output (`null` versus `0`).
- **Partial values are lower bounds.** Costs and token sums are complete,
  partial or unknown (`analysis.md` §4); they are never negative, so a
  partial value's known part is a lower bound. `F > x` and `F >= x` match a
  partial value whose known part already satisfies them; `F < x`, `F <= x`
  and `F = x` match only complete values. `agg sum:F` adds known parts and
  reports two counts per field, `{"unknown_cost": n, "partial_cost": m}`;
  a total with `m > 0` renders as `≥ X (m unknown)` (`rendering.md`,
  states).
- **Unknown under `NOT`.** A condition on a value that is unknown (or
  partial, for `<`, `<=`, `=`) is neither true nor false: the row matches
  neither the condition nor its `NOT`, and `!=` is a condition like any
  other (`exit_code != 0` does not match a call whose exit code is unknown;
  "not known to be 0" is not "known to be something else"). A field whose
  stored value is `unknown` (a tool call's `outcome`, `written_by`) counts
  as unknown here, so `NOT outcome:ok` leaves out calls with outcome
  `unknown`. Sinex's language has the same rule (joint #17). `field:unknown` selects such rows,
  and `--why` reports them per condition as "n unknown". `exists child(C)`
  is true when some child satisfies C, false when every child fails it, and
  unknown otherwise, so `NOT exists tool_call(exit_code = 0)` does not
  select a session whose only calls have an unknown exit code. Rewrites
  that look relational hold only with their preconditions: `Q where NOT p`
  equals "Q minus (Q where p)" only when p is never unknown on Q (the
  difference keeps unknown rows, `NOT` leaves them out); two `exists` are not
  one `exists` over their conjunction (One `exists`, one row); a condition
  moved across `| limit` changes the question (Membership, ranking,
  selection and page). The planner may use a rewrite only where its
  precondition holds.
- **Rows without a time.** A row whose time is unknown never matches a time
  condition (`since:`, `until:`, `at > …`) and is never given a substitute
  time; `group by day` (and week, month) puts such rows in a separate
  `unknown` group, counted, never in 1970 and never silently dropped from a
  rollup. `at:unknown` selects them. A time-filtered result says in `notes`
  how many rows it excluded for having no time (rule R8).
- **Partial periods.** Each group of `group by day` (and week, month) says
  whether its period is `complete` or `partial` (in JSON, the group row's
  `period` field): partial when the period is
  still in progress (today, this week) or when a provider location it covers
  was unreadable, or a source not yet collected, for part of it (from the
  observation and collection records; `convergence.md`, collect). A partial
  period's counts are lower bounds. Anything that compares periods (a trend,
  a baseline, "this week against last", a briefing's "more than usual") uses
  complete periods only, unless the query asks for partial ones
  (`partial:include`), and reports how many periods it left out and why;
  mixing a lower bound in silently biases the comparison (the rule Sinex's
  panels follow, `sinex-spec/spec/analysis.md`).
- **Calendar windows** (`group by month(at)`, subscription windows)
  use calendar arithmetic in the configured time zone, never a fixed number
  of seconds, so months have no gaps or overlaps (`analysis.md`).
- **Sorting by counts uses stored counts.** `sort by messages`, `sort by
  words` and `sort by input_tokens` (any count field) read the stored counts
  on `session_facts` (inherited messages excluded, as the defaults say); a
  sort never reads messages to compute its key.
- **Bounds per session.** A per-session window (`with
  tool_calls(...)[last:5]`) applies within each session, never as one global limit shared
  unfairly across sessions. There is no page-wide ceiling: a page of N
  sessions with `limit k` returns up to N × k attached rows, and attached
  text is not cut (rule R48).
- **A random order** uses a seed stored in the cursor, so pages of a random
  sample neither repeat nor skip rows.
- **One view per query.** A query reads one consistent state of the
  archive: one snapshot of `index.db` with `archive.db` attached, plus the
  in-memory vector set at the highest `embedded.id`, so counts, rows, notes
  and vectors all come from the same moment. The result records that state
  in its `view` field: the index generation and counter, the `archive.db`
  write counter and the highest `embedded.id` (`storage/README.md`). The
  daemon acquires the database snapshots and immutable vector-set reference
  together in the short operation serialized with writer commits, index swaps
  and vector publication (`storage/README.md`, one view per read). Query
  execution retains that view without holding serialization. `up_to_date`
  (Results) says whether that state had caught
  up. A query that cannot be answered from one consistent state (for example
  during an index swap) says so; it never mixes states.
- **Continuation.** A cursor encodes the query, its order with the row ID as
  the final tie-break (highest first, in either direction of the main sort;
  rows whose sort time is unknown come after every row with a time), the anchor (the sort key and ID of the last row
  returned) and, for a random order, its seed, so the next page continues by
  key, not by offset. If the archive changed between pages, the next page is
  computed against the current state from that position, and its `notes` say
  "the archive changed since the first page"; new rows sorting before the
  position are not shown, rows sorting after it may appear, and no row whose
  sort key and ID did not change is skipped or shown twice. The cursor also
  carries the first page's `view`, and a later page lists in `notes` the
  rows that match the query, changed after that view and now sort before
  the anchor ("3 sessions changed and moved before your position: …"), so
  sessions that stay active while the default order is paged are not lost
  silently. An enumeration that must be complete uses `keep` or
  `sort by started`. No read
  transaction is held between pages (rule R52). A page may be empty and
  still carry a cursor (filters can skip many candidates); only a result
  without `next_cursor` is finished. The page size (`page_size`), the selection (`| limit`)
  and the work a page may do are separate (Four bounds). Results that must not change between pages
  are kept first (`keep`) and paged from the kept result. A cursor from
  before the index was rebuilt is refused (`refused`, `stale_cursor`), never
  silently restarted. A request that carries a cursor takes the query text,
  filters, sort and unit from the cursor; if it also states any of them and
  they differ, it is refused (`invalid_query`, "this cursor belongs to a
  different query"), and equal values are accepted. Only the page size may change between
  pages, because pages continue by key; a `| limit` is part of the query.

## Results

This section is the one definition of result fields; every surface returns
them for a query (CLI `list`, search and `analyze` with `--json`, MCP
`search`, the HTTP API). A whole session document (`read --format json`,
`export`) is the export format instead (`export-format.md`). Times in JSON
are UTC with `Z` and milliseconds.

<!-- test: query-1 slice:1 -->
```json
{
  "outcome": "ok",                     // ok | empty | degraded | error
  "unit": "sessions",                  // or messages, tool_calls, …, report (analyze)
  "rows": [ ... ],
  "count": 3,                          // exact total of matches (for near:, of candidates)
  "next_cursor": null,
  "notes": [],                         // e.g. "semantic search is not enabled"
  "up_to_date": true,                  // nothing observed or stored was still waiting at the view (below)
  "view": {"generation": "…", "index": 1234, "archive": 87, "embeddings": 12},  // the state the query read
  "resolved": {"now": "2026-10-08T10:00:00.000Z", "zone": "Europe/Warsaw", "logical_start": "04:00"},  // the clock and day boundary its times used (joint #15)
  "query": "…",                        // the query as run (or `query:<name>` and its revision)
  "read_as": "…",                      // the query read back in plain language (errors and explain, below)
  "unknown": {},                       // per aggregated field: how many values were unknown or left out
  "why": null                          // with --why or from explain: [{clause, group, matches, of, hint}] (below)
}
```

`query`, `read_as` and `unknown` are what `analysis.md` §11 means by "a figure names
its definition and its coverage"; `view`, `resolved` and `up_to_date` say
which state and which clock were read. Two additions exist: MCP results add
`next` (`surfaces/README.md`) and, on session rows of a text search,
`more_hits` (the hits beyond the tool's `hits_per_row`), and an `analyze` report adds `group_by` and
`total`, whose aggregate keys follow `agg` (`sum_cost`, `unknown_cost`,
`partial_cost`).

**Rows.** A row has the unit's fields under their query names (sessions:
`id`, `origin`, `kind`, `title` (the displayed title), `title_source`, `started`,
`last_active`, `cwd`, `repo`, `messages`, `human_prompts`, `tool_calls`,
`errors`, `cost`, …; `--fields` chooses); money is a decimal USD number;
text-term results add `hits: [{message, session, via, written_by, at,
snippet, marks}]`: `snippet` is plain text and `marks` the match offsets
in it (`[[start, end], …]`, character offsets); the `«…»` marks appear only
in rendered views.

- `up_to_date` is true only when, at the view, no observed file has growth
  not yet parsed, no stored input waits for parsing, no derived product is
  behind its version (`convergence.md`, what is out of date), and the
  serving index is not held for a rebuild. Otherwise it is `false`, and
  `notes` says which applies. A session whose rebuild, parse or derived
  products are unfinished in the view is *incomplete*: it is returned,
  `notes` says "N sessions in this result are being rebuilt", and an
  aggregate over incomplete sessions is `degraded`, naming them.
- `unknown` also carries `unparsed_inputs`: stored inputs with status
  `failed`, `refused` or `no_parser_yet` whose format belongs to an origin
  the query can select (every origin unless it fixes `origin:`). When that
  is above zero, `notes` says "N stored inputs are not parsed (failed a,
  refused b, parser not yet built c); their sessions are not in this
  result", and a result that would be `empty` is `degraded` instead.
  Inputs that parsed to no sessions (`no_sessions`) are a measured absence
  and are not counted.

In code the outcome is `polylogue_model::types::ResultOutcome`, exported by
`polylogue_query` as `ResultOutcome` (`Ok`, `Empty`, `Degraded { missing }`,
`Error { kind }`). Shared values also own `ErrorKind` and `Failure`: query
maps its errors to those details; surfaces carry them without depending on
query internals. The daemon reply constructs rows only from a successful
read result, keeps failure details separately, and derives each surface's
exit/outcome from that same value. `Empty` is constructed only from
a read that succeeded and matched nothing; a failed read can produce only
`Degraded` or `Error`, so a failure cannot become an empty answer (rule
R2).

- `empty` means the query ran and nothing matched.
- `degraded` means the query ran but part of what it needed is missing (for
  example embeddings disabled, or the index still catching up after a
  rebuild); `notes` says what. A degraded result can have rows.
- A result assembled from several parts (the sides of a hybrid search, the
  figures of `status`, the sections of a briefing, the queries of an
  `analyze` report) takes its outcome by one rule: every part `error` →
  `error`; some parts `error` or `degraded` → `degraded`; otherwise `ok` if
  any part has rows, else `empty`. `notes` names every failed or degraded
  part and what it missed; a part that gave no reason is named with "no
  reason given", never dropped.
- `error` carries `{"kind", "reason", "message", "position", "suggestion",
  "hint", "retryable"}` (`reason` only with `refused`): `kind` is one of `invalid_query`, `not_yet_available` (a
  known unit or field that a later slice adds; the message names the
  slice), `not_found`, `ambiguous` (several rows matched a one-row lookup;
  `hint` lists them), `conflict` (a resent write `id` with different
  content), `timeout`, `not_running`, `io`, or `refused` (a valid request
  the archive declines, with `reason`: `stale_revision` (a revision of a note
  that is no longer current, naming the current one), `operator_only`
  (excision, `--reveal`, an outbound setting asked for from inside a harness
  or by a script), `rubric_disallows` (a pass writing outside its rubric),
  `session_running` (`continue` on a live session without `--anyway`),
  `stale_cursor` (a cursor from before the index was rebuilt),
  `conflict_unresolved` (`@<n>` asked of a note family with two current
  revisions), `target_required` (an agent note with no target),
  or `not_over_http` (excision asked through the web routes)). A value
  outside a field's closed set (law 137) and an unquoted command word or
  near miss (law 135) are `invalid_query`: the request is wrong as written;
  `position`
  and `suggestion` for query errors; `hint` is one sentence on what to do
  next (`use 'explain' to list fields`); `retryable` says whether the same
  call may succeed later (a locked database: yes; a bad query: no).
  A `timeout` (a read that passed its time limit, `storage/blobs.md`, WAL)
  names the stage that was running, the rows scanned so far, and a
  narrower `suggestion` (a time bound, a repository, a unit with an index
  for the condition); it is `retryable: false` for the same query. The
  limit exists because a long read holds back checkpoints and the WAL grows;
  analysis that needs more reads in steps (`export`, `sql --jsonl`), each
  step its own snapshot, so no answer is out of reach, only out of one
  transaction. Sinex keeps the same rule for the same reason (its writer
  shares Postgres).
- Session rows from a query with a text term carry `hits`.
- `why` (with `--why`, and always from `explain QUERY`) lists, for each
  condition, `{clause, group, matches, of, hint}`: how many rows it kept out
  of how many it saw, under its group inside `OR`, `NOT` and parentheses,
  with no limit on their number; a condition whose count could not be
  computed has `matches: null` and the reason in `hint`. The terminal prints
  one line per entry, `origin:codex → 2 of 7`; MCP `explain` returns the
  same list. This is the one shape of a condition-by-condition answer.
- `explain QUERY` prints the parsed query, the defaults applied, the unit
  and the `why` counts (read-only counting within the read time limit; it
  never writes, builds an index or calls a model), followed by the query **read back in plain
  language**, for example `tool_calls where outcome:error | group by session
  | count` → "Count tool calls whose outcome is error, grouped by session,
  over all sessions in this archive, up to now." The sentence is assembled
  from a phrase declared beside each field, operator and stage in the field
  table (no model is involved), so it is exact, and every field must have
  one (a drift check). Every result carries the same sentence as `read_as`,
  so an agent that turned a person's words into a query can show which
  question it actually asked. Sinex renders its queries the same way
  (joint #17).

### Requested settlement

Global `up_to_date` remains a statement about all observed work. A caller that needs one result settled through a particular admitted basis needs a separate owner proof, scoped to the requested output and its dependencies. A source timestamp, maximum parsed position or read-view counter is not that proof.

The owner captures a fixed basis C identifying the retained input revisions/extents and relevant companion, parser, product and configuration versions. Later admission, including an older provider-time record, does not move C. Derive requirements from the supported logical operation and retained provenance, never a caller-supplied dependency list or only its currently matching rows. A not-yet-parsed input remains a possible contributor unless its source contract proves otherwise. Required linked sessions, global title/preamble dependencies and ranking statistics must be included when used; an unused optional attachment is not a prerequisite. Unknown dependency shapes require conservative global settlement or an explicit unsupported result.

Report settled, pending with named owner obligations, unavailable with the retained interpretation failure, or unsupported proof. Bytes at EOF remain pending while required row finalization or products are unfinished (`convergence.md`, Parse and Derive). This reuses existing owner state; it adds no task queue or separate progress ledger. Waiting releases writer gates and snapshots and creates no durable work. A timeout ends that caller's wait; explicitly retained computation has its own operation lifetime.

Keep the target basis C separate from the serving view V. **Settled through C** can serve a newer V only with that fact disclosed and the requested rows in V usable; an old proof cannot certify unfinished newer rows. **Result at C** requires a supported evaluation or retained completed result/generation for C. Current rows relabelled with an old fence do not satisfy it. Both kinds of read enforce current removal and disclosure authority independently of computational settlement.

The first implementation should prove one exact-session read before arbitrary query planning. Acceptance is unrun: leave relevant A unfinished despite a later processed maximum; reach A's byte end with row finalization pending; finish A while unrelated B stays pending; admit A2 after C and preserve C without certifying unfinished A2 rows; reject an unsupported dependency shape; disconnect a waiter and show the owner and another client continue. The retained prototype currently proves only interleaving between committed input turns. It exposes no settlement or exact-basis read capability.

## Saved queries

```
polylogue query save failing 'tool_calls where outcome:error'
polylogue 'query:failing session.tag:project:morrow'
```

A saved query is a note of kind `saved_query` in `archive.db` (`user-data.md`).
`query:NAME` is a typed expression, not text pasted into parentheses. A
saved query has a **shape**: a *relation* (a unit and its conditions, with
no stages) or a *shaped result* (with stages: grouping, sorting, a limit,
an aggregate). A relation can stand where a condition can, in a query over
the same unit: at the top level it is that query's source, bound with its
own defaults and its own resolved times, so other conditions only narrow it
(`tool_calls where query:wrappers outcome:error` keeps the wrappers the
saved query asked for); nested under `OR` or `NOT` it means membership in
its bound relation. A shaped result can only be read (`query:NAME`,
`query:NAME | where …`), never used as a condition; `tool_calls where
query:by-session` is an `invalid_query` error naming its stages, and units
that disagree are an error too. So a saved query selects the same rows used
inline or by name (law 127). Saved queries can be watched (re-evaluated as data arrives, with
new matches reported; `user-data.md`). Keeping the exact rows a query
returned at a moment ("keep this result") is part of `user-data.md`; a kept
result is referable only as `result:<id>` (its UUIDv7). A kept result is
also a source with typed parts: `result:<id>` (or `result:<id>.rows`) reads
its rows as kept, and `result:<id>.matches` the match rows stored with
them, so `result:0199f3c4… | group by origin | count` and
`result:0199f3c4….matches | agg count` are ordinary queries over the kept
values, never a re-run. A kept result may
carry a name, but the name is a label for display, never a reference, so two
machines can keep results with the same name.

## Composition

The rules above make queries compose: the output of one form is a
well-defined input to the next, and composing never reinterprets what was
already bound. The laws that check it (`plan/testing/laws.md`; names shared
with Sinex's where the meaning is the same,
`sinex-spec/prototype/query-ref/composition.py`):

| Law | Holds |
|---|---|
| 126 filter composition | successive filters equal their conjunction over the same bound source |
| 127 definition reuse | a saved query selects the same rows by name and inline |
| 128 attachment membership | `with` never changes which owners are returned |
| 129 group closure | a grouped result takes `where`, `sort by`, `limit` like any rows |
| 130 match order | the order rows are read in never changes a match or its rows |
| 131 ranking separation | a ranking never changes membership or `count` |
| 132 three-valued filters | a condition, its `NOT` and its unknowns partition the source; `exists` is three-valued |
| 133 defaults bound once | a condition true for every row never changes the bound source |
| 134 execution-mode parity | interactive, paged, kept, exported and either search plan give the same answer |

The reference evaluator checks the laws it can on Morrow
(`prototypes/query-ref/composition.py`, run by the query-examples check);
written against it, they found the default rebinding above, and that
`limit` and `where` after `group by` were missing.

## Planned extensions (not in the first slices)

Each is marked as an extension and must keep the rules above.

- **Counted subconditions:** `sessions where count(tool_call where
  outcome:error) >= 3`.
- **Set operations and subqueries:** `query:a minus query:b`,
  `sessions where id in (messages where written_by:human "chrono" | session)`.
- **Macros: saved queries with typed parameters:** `query save failing(repo:
  text) 'tool_calls where outcome:error session.repo:$repo'`, used as
  `query:failing(repo=morrow)`. A macro's definition is versioned (it is a
  note revision), and its parameters are substituted as values, never as
  text, so a parameter cannot change the query's shape. A name resolves to
  one definition revision when the query runs, and the result names that
  revision, so editing the macro later never changes a kept result;
  relative times inside it resolve against the invoking query's clock; a
  macro that refers to itself, directly or through others, is refused once
  with the chain of names (definitions are finite, so detecting cycles is
  enough and no depth limit exists); and a macro is a read, so expanding one never writes a note or
  calls a model (`polylogue-1lm`, via `polylogue-4p1.8`).
- **A limit per group:** `sessions where text:timeout | group by origin |
  first 5` keeps the top five of each group, for any grouping field, by the
  same operator, never a separate per-origin search; a group that matched
  nothing is shown as zero only when it was searched completely, otherwise
  as incomplete (`polylogue-5roxd`).
- **Named parts within one query:** `let failing = tool_calls where
  outcome:error; failing | group by session | count`. Names are local to the
  query.
- **Temporal joins across units**, with Sinex's names and meaning
  (`sinex-spec/spec/query.md`): `during`, `near [d]`, `before [d]`,
  `after [d]` relate rows of two units by their times or intervals, for
  example `processes where comm:cargo during (tool_calls where
  command:"cargo test")` or `sessions near (commit:4e1d2c9) [2h]`. Intervals
  come from each unit's own start and end (a process's `started_at` and
  `ended_at`, a tool call's `at` and duration, a session's first and last
  activity); a row with an unknown time matches neither the join nor its
  `NOT`. `seq(... -> [within d] ...)` stays the form for order inside one
  session. In backed mode the same words join Polylogue rows with Sinex
  events.
- **Clusters by meaning:** `messages where written_by:human | cluster by
  near [k:20]` groups a set into clusters of similar rows by their stored
  vectors, each with its size and a representative row; without embeddings
  the result is `degraded`, never a guess (`vision/uses.toml`: `idea-ledger`,
  `recurring-questions`).
- **Commits as a unit:** `commits where session.repo:morrow | agg
  sum:input_tokens` reads each commit's span (`analysis.md`, git correlation),
  so tokens and cost per commit are queries (`vision/uses.toml`:
  `tokens-per-commit`).
- **Numbers inside tool output:** `tool_calls where output:"MB/s" |
  extract number before "MB/s" as rate | sort by at`, a read-time extraction
  from the output text with the matched text kept beside each value, so a
  measured quantity has a trend; an output with no match gives `unknown`
  (`vision/uses.toml`: `measured-trend`).
- **Derived grouping keys:** `group by` takes, besides fields, one of a
  closed list of key functions: `program(command)` (a shell command's
  program word, after `sudo`, `env` and variable assignments), `day(at)`
  (and week, month; the logical day), and `fingerprint(text)` (the text
  folded to lower case, whitespace collapsed, digits and quoted literals
  replaced by placeholders, then BLAKE3), so `tool_calls where
  outcome:error | group by program(command) | count` and near-duplicate
  prompts (`messages where written_by:human provider_unmarked_bytes:>0 | group by
  fingerprint(text)`) are queries. Each function is defined once and named
  in `read_as`; a row whose key is unknown goes to an `unknown` group
  (`vision/uses.toml`: `agent-behaviour`, `mine-prompts`).
- **Series statistics over grouped rows:** after `group by day` (or week,
  month), `| series median:count p90:count longest_empty_run
  halves:count` computes over the groups as one series: median and
  percentiles, the longest run of consecutive empty periods, and the later
  half of the periods against the earlier, using complete periods only
  unless `partial:include` (Partial periods, above), with the number of
  periods left out reported (`vision/uses.toml`: `trajectory`).
- **The gap between sequence steps:** match rows of `seq(A -> B)` carry, for
  each consecutive pair of steps, `gap_steps` (rows of B's unit between
  them in session order) and `gap_time` (`B.at - A.at`, unknown when either
  time is unknown), so "how long from a failing test to the next passing
  one" is `tool_calls | seq(command:"cargo test" outcome:error ->
  command:"cargo test" outcome:ok) | matches | agg median:gap_time`
  (`vision/uses.toml`: `agent-behaviour`).
- **Other sequence policies:** `seq all(...)` (every match, overlapping
  allowed) beside the default earliest non-overlapping matches, explicit in
  the query and costed by `explain`.
- **Recursion limit.** Nesting (parentheses, subqueries, macros calling
  macros) is checked against a fixed depth before evaluation, so a malformed
  query is an error with its position, never a crash.

## Examples against the Morrow archive

Expected results come from `examples/archive/FACTS.md`. Session numbers refer
to its table (#1–#8). These double as tests (`plan/testing.md`). The last
column is the slice from which an example passes on the main archive (the
slice that adds its fields, or that adds the provider of a session in its
result). Slice 1 runs examples 1, 2, 10, 11, 12, 26, 28 and 35 on the
Claude-Code-only home instead, with the rows in
`examples/archive/expected/slice-1/expected.json`.

| # | Query | Expected result | Slice |
|---|---|---|---|
| 1 | `origin:claude-code` | #4, #2, #1 (newest activity first). 3 rows; the subagent #3 is excluded by default. | 1 |
| 2 | `origin:claude-code kind:subagent` | #3 | 1 |
| 3 | `kind:* \| agg count, sum:cost` | `count 8`, `sum_cost 6.20` (exact 6.1956), `unknown_cost 2`, `partial_cost 0` | 4 |
| 4 | `sessions \| group by origin \| count` | claude-code 3, codex 2, chatgpt 1, claude-ai 1 (7 top-level sessions) | 4 |
| 5 | `repo:morrow` | #6, #5. In real use Claude Code sessions also get a repository, read from git at their working directory (`sources/claude-code.md`); the example archive's one hook spool line (slice 2) carries no `git` object and its working directory does not exist on the test machine, so its Claude Code sessions match `cwd:/home/op/morrow` but not `repo:`. | 2 |
| 6 | `tag:project:morrow` | #6, #5, #4, #2, #1 (5 rows) | 7 |
| 7 | `tag:research` | #7 | 7 |
| 8 | `sessions where cost > 1` | #4 (1.59), #2 (2.85), #1 (1.26) | 3 |
| 9 | `cost:unknown` | #8, #7 | 4 |
| 10 | `sessions where human_prompts >= 3` | #7 (3), #4 (3), #2 (3), #1 (3) | 4 |
| 11 | `sessions where messages >= 10` | #4 (10 own), #2 (18), #1 (12). #4's 18 inherited messages are not counted. | 1 |
| 12 | `sessions where tool_calls > 5` | #4 (8), #2 (13), #1 (6) | 1 |
| 13 | `messages where written_by:human "chrono"` | 2 rows: the operator's message in #8 ("Shorter, and mention chrono.") and in #2 ("let's use chrono, not time") | 4 |
| 14 | `messages where written_by:human inherited:* "chrono"` | 3 rows: #8's message, the original in #2, and its inherited copy at the start of #4 | 4 |
| 15 | `tool_calls where outcome:error` | 2 rows: #2 `cargo test` (exit 101), #4 `cargo clippy` (exit 1) | 3 |
| 16 | `tool_calls where outcome:error \| group by session \| count` | #2: 1, #4: 1 | 3 |
| 17 | `tool_calls \| group by action \| count` | shell 18, file_edit 8, file_read 6, file_write 2, web 2, subagent 1 (37 in total, subagent #3's calls included; the five Codex shell calls are `exec_command` children of code-mode wrappers, which are not counted) | 3 |
| 18 | `tool_calls where tool:Bash \| agg count` | 13 (#1: 3, #2: 5, #4: 5) | 3 |
| 19 | `tool_calls where session.tag:project:morrow \| agg count` | 33 (#1: 6, #2: 13, #4: 8, #5: 3, #6: 3) | 7 |
| 20 | `sessions where exists tool_call(action:shell AND outcome:error)` | #4, #2 | 3 |
| 21 | `sessions where seq(command:"cargo test" AND outcome:error -> tool:Edit -> command:"cargo test" AND outcome:ok)` | #2. (#4's copy of #2's messages is inherited and excluded; #4's own error was `cargo clippy`.) | 3 |
| 22 | `messages where written_by:human \| group by session.origin \| count` | claude-code 9, chatgpt 3, codex 3, claude-ai 2. #7's edited prompt on the abandoned branch is excluded. | 4 |
| 23 | `messages where active:false` | 2 rows, both in #7 (the abandoned branch) | 4 |
| 24 | `parent:claude-code:7f3a2c10-0002-4000-8000-000000000002` | #4 (`continuation`, `how: resume`; and `reference`, `how: briefing`, from the slice-2 hook spool's delivery), #3 (`subagent`, `how: fresh`). Mentioning `parent` drops the subagent default. | 6 |
| 25 | `lineage:codex:01a20000-0005-7000-8000-000000000005` | #6, #5 | 6 |
| 26 | `since:2026-10-05` | #8, #6, #5 | 4 |
| 27 | `started between 2026-10-02 and 2026-10-03` | #4, #2 (dates are whole logical days, inclusive; subagent #3 excluded by default) | 3 |
| 28 | `has:compaction` | #4 | 1 |
| 29 | `model:claude-opus-5-5` | outcome `empty`: only subagent #3 used it, and subagents are excluded by default. `model:claude-opus-5-5 kind:*` returns #3. `model:unknown` returns #8, because the claude.ai export records no model. | 4 |
| 30 | `origin:(chatgpt\|claude-ai)` | #8, #7 | 4 |
| 31 | `notes where kind:decision` | 1 row: "decided: chrono, not time", target = the operator's message in #2 | 7 |
| 32 | `origin:codex with messages(id, role)[role:assistant, last:1]` | #6 and #5, each with one attached message: its last assistant message | 3 |
| 33 | `near:"ambiguous date formats"` | outcome `degraded`, note "semantic search is not enabled", 7 rows ordered by last activity | 4 |
| 34 | `repo:other` | outcome `empty`; with `--why`, `why` is `[{clause: "repo:other", matches: 0, of: 5, hint: "known repos: op/morrow"}]`, printed `repo:other → 0 of 5 (known repos: op/morrow)`: at slice 3 the five Claude Code and Codex top-level sessions are parsed and the chat exports are not yet imported (slice 4), so nothing is unparsed and the outcome is `empty`, not `degraded` | 3 |
| 35 | `reop:morrow` | outcome `error`: `unknown field "reop" at position 0; did you mean "repo"?` | 1 |
| 36 | `links where kind:continuation` | 2 rows: #2 → #4 (`how: resume`), #5 → #6 (`how: fork`, Codex `forked_from_id`; #6 was not spawned) | 6 |
| 37 | `link:fork` | #6 | 6 |
| 38 | `tree_cost > 2.90` | #2 (own cost $2.85, tree $3.01 with subagent #3); `cost > 2.90` returns nothing. The `tree_cost` field arrives in slice 6; slice 1 shows tree cost only in the summary view. | 6 |

Rows are listed in the default order, newest last activity first (#8 10-07,
#6 10-06, #5 10-05, #7 10-04, #4 10-03, #2 and #3 10-02, #1 10-01).

Bead and audit IDs behind each section: `evidence/beads/spec-sources-g3.md`.
